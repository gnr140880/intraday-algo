"""
Live OI Feed — Real-time Open Interest via Kite WebSocket

Subscribes to NIFTY weekly option strikes (ATM ± N strikes) in MODE_FULL,
receives live OI, LTP, volume from tick data, and builds:
  - Live option chain (per-strike CE/PE data)
  - PCR (Put-Call Ratio) near ATM
  - OI change % vs session open / previous close
  - Buildup signals (Long Buildup, Short Buildup, Short Covering, Long Unwinding)
  - Volume + OI spike detection

Usage:
    feed = LiveOIFeed(kite_client)
    feed.start(spot_price=24500, expiry_date=date(2026, 3, 10))
    ...
    chain = feed.get_option_chain()
    oi_data = feed.get_oi_analysis()
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StrikeOIData:
    """Live OI data for a single strike + option type."""
    strike: float
    option_type: str          # CE or PE
    instrument_token: int
    tradingsymbol: str

    # Live tick data (updated every tick)
    ltp: float = 0.0
    volume: int = 0
    oi: int = 0
    oi_day_high: int = 0
    oi_day_low: int = 0
    bid: float = 0.0
    ask: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    day_open: float = 0.0

    # Baseline OI (captured at session start or from prev day)
    oi_at_open: int = 0       # OI at start of today's session
    prev_day_oi: int = 0      # Yesterday's closing OI

    # Computed
    oi_change: int = 0        # oi - oi_at_open
    oi_change_pct: float = 0.0
    price_change_pct: float = 0.0
    buildup: str = ""         # "Long Buildup" / "Short Buildup" / "Short Covering" / "Long Unwinding"

    last_tick_time: str = ""


@dataclass
class OIChainSnapshot:
    """Full option chain snapshot with analytics."""
    timestamp: str = ""
    spot_price: float = 0.0
    atm_strike: float = 0.0
    expiry: str = ""

    # Per-strike data
    strikes: Dict[float, Dict[str, StrikeOIData]] = field(default_factory=dict)
    # strikes = {24500: {"CE": StrikeOIData, "PE": StrikeOIData}, ...}

    # Aggregated analytics (ATM ± oi_analysis_range strikes)
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    pcr: float = 0.0          # PE OI / CE OI
    pcr_change: float = 0.0   # PCR change from session open
    ce_oi_change_pct: float = 0.0
    pe_oi_change_pct: float = 0.0

    # Max pain strike (strike with max combined OI)
    max_pain_strike: float = 0.0

    # Top OI strikes
    highest_ce_oi_strike: float = 0.0
    highest_pe_oi_strike: float = 0.0

    # Volume + OI spike detection
    oi_spikes: List[Dict] = field(default_factory=list)
    volume_spikes: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Live OI Feed
# ---------------------------------------------------------------------------

class LiveOIFeed:
    """
    WebSocket-based live OI data collector for NIFTY options.

    Subscribes to NIFTY weekly option instruments around ATM,
    receives tick data in MODE_FULL, and maintains a live option chain.
    """

    def __init__(self, kite_client, strike_gap: int = 50, num_strikes: int = 10):
        """
        Args:
            kite_client: KiteClient instance (must have valid access_token)
            strike_gap: Gap between strikes (50 for NIFTY, 100 for BANKNIFTY)
            num_strikes: Number of strikes on each side of ATM to subscribe
        """
        self._kite = kite_client
        self._strike_gap = strike_gap
        self._num_strikes = num_strikes

        # Token → StrikeOIData mapping
        self._token_map: Dict[int, StrikeOIData] = {}
        # Strike → {CE: StrikeOIData, PE: StrikeOIData}
        self._chain: Dict[float, Dict[str, StrikeOIData]] = {}

        # State
        self._spot_price: float = 0.0
        self._atm_strike: float = 0.0
        self._expiry: Optional[date] = None
        self._expiry_str: str = ""
        self._running = False
        self._ws_connected = False
        self._session_open_captured = False

        # Baseline PCR for change tracking
        self._pcr_at_open: float = 0.0
        self._total_ce_oi_at_open: int = 0
        self._total_pe_oi_at_open: int = 0

        # NIFTY spot token for live spot tracking
        self._nifty_spot_token: Optional[int] = None

        # Spike detection thresholds
        self._oi_spike_threshold_pct: float = 5.0    # 5% OI jump in single tick
        self._vol_spike_threshold: int = 10000        # volume spike absolute

        # Previous tick OI for spike detection
        self._prev_tick_oi: Dict[int, int] = {}

        # Lock for thread-safe access
        self._lock = threading.Lock()

        logger.info(
            f"LiveOIFeed initialized: strike_gap={strike_gap}, "
            f"num_strikes={num_strikes} per side"
        )

    # ------------------------------------------------------------------
    # Setup — filter instruments and build subscription list
    # ------------------------------------------------------------------
    def _resolve_instruments(
        self, spot: float, expiry_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Fetch NFO instruments, find nearest weekly expiry,
        filter NIFTY options around ATM.
        Returns list of instrument dicts.
        """
        try:
            instruments = self._kite.get_instruments("NFO")
        except Exception as e:
            logger.error(f"LiveOIFeed: cannot fetch instruments: {e}")
            return []

        today = date.today()
        nifty_options = [
            i for i in instruments
            if i.get("name") == "NIFTY"
            and i.get("segment") == "NFO-OPT"
            and i.get("expiry")
            and i["expiry"] >= today
        ]

        if not nifty_options:
            logger.warning("LiveOIFeed: no NIFTY options found")
            return []

        nifty_options.sort(key=lambda x: x["expiry"])

        if expiry_date:
            # Use specified expiry
            weekly_opts = [i for i in nifty_options if i["expiry"] == expiry_date]
            if not weekly_opts:
                # Fallback to nearest
                expiry_date = nifty_options[0]["expiry"]
                weekly_opts = [i for i in nifty_options if i["expiry"] == expiry_date]
        else:
            expiry_date = nifty_options[0]["expiry"]
            weekly_opts = [i for i in nifty_options if i["expiry"] == expiry_date]

        self._expiry = expiry_date
        self._expiry_str = str(expiry_date)

        # ATM strike
        atm = round(spot / self._strike_gap) * self._strike_gap
        self._atm_strike = atm

        # Strike range: ATM ± num_strikes
        strike_range = range(
            int(atm - self._num_strikes * self._strike_gap),
            int(atm + (self._num_strikes + 1) * self._strike_gap),
            self._strike_gap,
        )

        # Both CE and PE for all strikes in range
        relevant = [
            i for i in weekly_opts
            if i["strike"] in strike_range
            and i["instrument_type"] in ("CE", "PE")
        ]

        logger.info(
            f"LiveOIFeed: {len(relevant)} instruments for {expiry_date}, "
            f"ATM={atm}, range={list(strike_range)[0]}-{list(strike_range)[-1]}"
        )
        return relevant

    def _build_token_map(self, instruments: List[Dict]):
        """Build internal token → StrikeOIData mapping."""
        with self._lock:
            self._token_map.clear()
            self._chain.clear()

            for inst in instruments:
                token = inst["instrument_token"]
                strike = inst["strike"]
                opt_type = inst["instrument_type"]

                data = StrikeOIData(
                    strike=strike,
                    option_type=opt_type,
                    instrument_token=token,
                    tradingsymbol=inst["tradingsymbol"],
                )
                self._token_map[token] = data

                if strike not in self._chain:
                    self._chain[strike] = {}
                self._chain[strike][opt_type] = data

    # ------------------------------------------------------------------
    # Fetch previous day OI (baseline for change calculation)
    # ------------------------------------------------------------------
    def _fetch_prev_day_oi(self):
        """
        Fetch previous day's closing OI for all subscribed instruments
        using Kite historical data API with oi=True.
        """
        from_dt = datetime.combine(date.today() - timedelta(days=5), datetime.min.time())
        to_dt = datetime.combine(date.today() - timedelta(days=1), datetime.max.time())

        fetched = 0
        for token, data in self._token_map.items():
            try:
                hist = self._kite.get_historical_data(
                    token, from_dt, to_dt, "day", oi=True
                )
                if hist:
                    last_bar = hist[-1]
                    data.prev_day_oi = last_bar.get("oi", 0)
                    fetched += 1
            except Exception:
                pass  # Some instruments may not have history

        logger.info(f"LiveOIFeed: fetched prev day OI for {fetched}/{len(self._token_map)} instruments")

    # ------------------------------------------------------------------
    # WebSocket tick handler
    # ------------------------------------------------------------------
    def _on_ticks(self, ticks):
        """Process incoming MODE_FULL ticks — extract OI, LTP, volume."""
        with self._lock:
            now = datetime.now().isoformat()

            for tick in ticks:
                token = tick.get("instrument_token")

                # Check if it's the NIFTY spot token
                if token == self._nifty_spot_token:
                    self._spot_price = tick.get("last_price", self._spot_price)
                    continue

                data = self._token_map.get(token)
                if not data:
                    continue

                # Extract tick fields
                ltp = tick.get("last_price", 0)
                volume = tick.get("volume_traded", 0)
                oi = tick.get("oi", 0)

                # OI spike detection (before updating)
                prev_oi = self._prev_tick_oi.get(token, 0)
                if prev_oi > 0 and oi > 0:
                    oi_tick_change_pct = abs(oi - prev_oi) / prev_oi * 100
                    if oi_tick_change_pct > self._oi_spike_threshold_pct:
                        logger.info(
                            f"OI SPIKE: {data.tradingsymbol} "
                            f"OI {prev_oi}→{oi} ({oi_tick_change_pct:+.1f}%)"
                        )
                self._prev_tick_oi[token] = oi

                # Update data
                data.ltp = ltp
                data.volume = volume
                data.oi = oi
                data.oi_day_high = tick.get("oi_day_high", data.oi_day_high)
                data.oi_day_low = tick.get("oi_day_low", data.oi_day_low)
                data.day_high = tick.get("ohlc", {}).get("high", data.day_high)
                data.day_low = tick.get("ohlc", {}).get("low", data.day_low)
                data.day_open = tick.get("ohlc", {}).get("open", data.day_open)
                data.last_tick_time = now

                # Bid/Ask from depth
                depth = tick.get("depth", {})
                if depth.get("buy"):
                    data.bid = depth["buy"][0].get("price", 0)
                if depth.get("sell"):
                    data.ask = depth["sell"][0].get("price", 0)

                # Capture session-open OI (first meaningful tick)
                if not self._session_open_captured and oi > 0 and data.oi_at_open == 0:
                    data.oi_at_open = oi

                # Compute OI change
                baseline_oi = data.oi_at_open if data.oi_at_open > 0 else data.prev_day_oi
                if baseline_oi > 0:
                    data.oi_change = oi - baseline_oi
                    data.oi_change_pct = round(
                        (oi - baseline_oi) / baseline_oi * 100, 2
                    )

                # Compute price change %
                if data.day_open > 0:
                    data.price_change_pct = round(
                        (ltp - data.day_open) / data.day_open * 100, 2
                    )

                # Compute buildup signal
                data.buildup = self._classify_buildup(
                    data.price_change_pct, data.oi_change_pct
                )

            # Mark session open captured after first batch of ticks
            if not self._session_open_captured:
                has_oi = any(d.oi_at_open > 0 for d in self._token_map.values())
                if has_oi:
                    self._session_open_captured = True
                    self._capture_pcr_baseline()
                    logger.info("LiveOIFeed: session open OI baseline captured")

    @staticmethod
    def _classify_buildup(price_change_pct: float, oi_change_pct: float) -> str:
        """
        Classify OI + Price action into buildup types:
          Price UP   + OI UP   → Long Buildup (bullish)
          Price DOWN + OI UP   → Short Buildup (bearish)
          Price UP   + OI DOWN → Short Covering (mildly bullish)
          Price DOWN + OI DOWN → Long Unwinding (mildly bearish)
        """
        if abs(price_change_pct) < 0.1 and abs(oi_change_pct) < 0.5:
            return "No Change"

        price_up = price_change_pct > 0.1
        price_down = price_change_pct < -0.1
        oi_up = oi_change_pct > 0.5
        oi_down = oi_change_pct < -0.5

        if price_up and oi_up:
            return "Long Buildup"
        elif price_down and oi_up:
            return "Short Buildup"
        elif price_up and oi_down:
            return "Short Covering"
        elif price_down and oi_down:
            return "Long Unwinding"
        return "Mixed"

    def _capture_pcr_baseline(self):
        """Capture PCR at session open for change tracking."""
        total_ce = sum(
            d.oi_at_open for d in self._token_map.values()
            if d.option_type == "CE" and d.oi_at_open > 0
        )
        total_pe = sum(
            d.oi_at_open for d in self._token_map.values()
            if d.option_type == "PE" and d.oi_at_open > 0
        )
        self._total_ce_oi_at_open = total_ce
        self._total_pe_oi_at_open = total_pe
        self._pcr_at_open = round(total_pe / max(total_ce, 1), 4)
        logger.info(
            f"LiveOIFeed: baseline PCR={self._pcr_at_open:.3f} "
            f"(CE OI={total_ce:,}, PE OI={total_pe:,})"
        )

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------
    def start(
        self,
        spot_price: float,
        expiry_date: Optional[date] = None,
        nifty_spot_token: Optional[int] = None,
    ):
        """
        Start live OI feed:
        1. Resolve instruments around ATM
        2. Fetch previous day OI baselines
        3. Start WebSocket subscription in MODE_FULL
        """
        if self._running:
            logger.warning("LiveOIFeed already running")
            return

        self._spot_price = spot_price
        self._nifty_spot_token = nifty_spot_token

        # Step 1: Resolve instruments
        instruments = self._resolve_instruments(spot_price, expiry_date)
        if not instruments:
            logger.error("LiveOIFeed: no instruments to subscribe, aborting")
            return

        # Step 2: Build token map
        self._build_token_map(instruments)

        # Step 3: Fetch previous day OI baselines
        try:
            self._fetch_prev_day_oi()
        except Exception as e:
            logger.warning(f"LiveOIFeed: prev day OI fetch failed: {e}")

        # Step 4: Collect all tokens to subscribe
        tokens = list(self._token_map.keys())
        if nifty_spot_token:
            tokens.append(nifty_spot_token)

        # Step 5: Start WebSocket
        try:
            def on_connect(ws, response):
                self._ws_connected = True
                self._running = True
                logger.info(
                    f"LiveOIFeed WebSocket connected: {len(tokens)} tokens "
                    f"({len(self._token_map)} options + spot)"
                )

            self._kite.start_ticker(
                tokens,
                on_tick=self._on_ticks,
                on_connect=on_connect,
            )
            logger.info("LiveOIFeed: WebSocket ticker started")
        except Exception as e:
            logger.error(f"LiveOIFeed: WebSocket start failed: {e}")
            self._running = False

    def stop(self):
        """Stop the WebSocket feed."""
        self._running = False
        self._ws_connected = False
        try:
            self._kite.stop_ticker()
        except Exception:
            pass
        logger.info("LiveOIFeed stopped")

    def update_atm(self, new_spot: float):
        """
        Re-center subscriptions if NIFTY moves significantly.
        Call this periodically (e.g., every 5 minutes) from the engine cycle.
        Only re-subscribes if ATM strike changes.
        """
        new_atm = round(new_spot / self._strike_gap) * self._strike_gap
        if new_atm == self._atm_strike:
            return  # No change needed

        logger.info(
            f"LiveOIFeed: ATM shifted {self._atm_strike} → {new_atm}, "
            f"re-subscribing..."
        )
        self._spot_price = new_spot
        self.stop()
        time.sleep(1)
        self.start(
            spot_price=new_spot,
            expiry_date=self._expiry,
            nifty_spot_token=self._nifty_spot_token,
        )

    @property
    def is_running(self) -> bool:
        return self._running and self._ws_connected

    # ------------------------------------------------------------------
    # Data accessors (thread-safe)
    # ------------------------------------------------------------------
    def get_option_chain(self) -> OIChainSnapshot:
        """
        Build a full option chain snapshot with analytics.
        Call from any thread — access is locked.
        """
        with self._lock:
            snapshot = OIChainSnapshot(
                timestamp=datetime.now().isoformat(),
                spot_price=self._spot_price,
                atm_strike=self._atm_strike,
                expiry=self._expiry_str,
            )

            # Copy chain data
            total_ce_oi = 0
            total_pe_oi = 0
            max_combined_oi = 0
            max_combined_strike = 0.0
            highest_ce_oi = 0
            highest_pe_oi = 0
            highest_ce_strike = 0.0
            highest_pe_strike = 0.0

            for strike in sorted(self._chain.keys()):
                snapshot.strikes[strike] = {}
                combined = 0

                for opt_type in ("CE", "PE"):
                    data = self._chain[strike].get(opt_type)
                    if data:
                        snapshot.strikes[strike][opt_type] = data
                        combined += data.oi

                        if opt_type == "CE":
                            total_ce_oi += data.oi
                            if data.oi > highest_ce_oi:
                                highest_ce_oi = data.oi
                                highest_ce_strike = strike
                        else:
                            total_pe_oi += data.oi
                            if data.oi > highest_pe_oi:
                                highest_pe_oi = data.oi
                                highest_pe_strike = strike

                if combined > max_combined_oi:
                    max_combined_oi = combined
                    max_combined_strike = strike

            snapshot.total_ce_oi = total_ce_oi
            snapshot.total_pe_oi = total_pe_oi
            snapshot.pcr = round(total_pe_oi / max(total_ce_oi, 1), 4)
            snapshot.pcr_change = round(
                snapshot.pcr - self._pcr_at_open, 4
            ) if self._pcr_at_open > 0 else 0.0
            snapshot.max_pain_strike = max_combined_strike
            snapshot.highest_ce_oi_strike = highest_ce_strike
            snapshot.highest_pe_oi_strike = highest_pe_strike

            # Aggregate CE/PE OI change %
            if self._total_ce_oi_at_open > 0:
                snapshot.ce_oi_change_pct = round(
                    (total_ce_oi - self._total_ce_oi_at_open) /
                    self._total_ce_oi_at_open * 100, 2
                )
            if self._total_pe_oi_at_open > 0:
                snapshot.pe_oi_change_pct = round(
                    (total_pe_oi - self._total_pe_oi_at_open) /
                    self._total_pe_oi_at_open * 100, 2
                )

            return snapshot

    def get_oi_analysis(self) -> Dict:
        """
        Get OI analysis dict compatible with options_engine._cached_oi_data format.
        This replaces the old _fetch_oi_analysis() when live feed is active.
        """
        chain = self.get_option_chain()

        # Build strike_oi map: {(strike, type): {oi, prev_oi, oi_change_pct}}
        strike_oi = {}
        for strike, types in chain.strikes.items():
            for opt_type, data in types.items():
                strike_oi[(strike, opt_type)] = {
                    "oi": data.oi,
                    "prev_oi": data.prev_day_oi or data.oi_at_open,
                    "oi_change_pct": data.oi_change_pct,
                    "buildup": data.buildup,
                    "volume": data.volume,
                    "ltp": data.ltp,
                }

        return {
            "strike_oi": strike_oi,
            "ce_oi_change_pct": chain.ce_oi_change_pct,
            "pe_oi_change_pct": chain.pe_oi_change_pct,
            "pcr": chain.pcr,
            "pcr_change": chain.pcr_change,
            "max_pain_strike": chain.max_pain_strike,
            "highest_ce_oi_strike": chain.highest_ce_oi_strike,
            "highest_pe_oi_strike": chain.highest_pe_oi_strike,
            "total_ce_oi": chain.total_ce_oi,
            "total_pe_oi": chain.total_pe_oi,
            "source": "live_websocket",
        }

    def get_strike_ltp(self, strike: float, option_type: str) -> float:
        """Get live LTP for a specific strike. Returns 0 if not available."""
        with self._lock:
            data = self._chain.get(strike, {}).get(option_type)
            return data.ltp if data else 0.0

    def get_candidate_live_data(self, tradingsymbol: str) -> Optional[Dict]:
        """
        Get live tick data for a specific instrument.
        Used to update OptionCandidate with real LTP, OI, volume.
        """
        with self._lock:
            for data in self._token_map.values():
                if data.tradingsymbol == tradingsymbol:
                    return {
                        "ltp": data.ltp,
                        "volume": data.volume,
                        "oi": data.oi,
                        "oi_change_pct": data.oi_change_pct,
                        "bid": data.bid,
                        "ask": data.ask,
                        "buildup": data.buildup,
                    }
        return None

    def get_dashboard_data(self) -> Dict:
        """
        Get a dashboard-friendly summary of live OI data.
        Includes: PCR, OI change, top OI strikes, buildup signals, chain table.
        """
        chain = self.get_option_chain()

        # Build chain table for frontend
        chain_table = []
        for strike in sorted(chain.strikes.keys()):
            row = {"strike": strike}
            ce = chain.strikes[strike].get("CE")
            pe = chain.strikes[strike].get("PE")

            if ce:
                row["ce_ltp"] = ce.ltp
                row["ce_oi"] = ce.oi
                row["ce_oi_change"] = ce.oi_change
                row["ce_oi_change_pct"] = ce.oi_change_pct
                row["ce_volume"] = ce.volume
                row["ce_buildup"] = ce.buildup
                row["ce_bid"] = ce.bid
                row["ce_ask"] = ce.ask
            else:
                row["ce_ltp"] = 0
                row["ce_oi"] = 0
                row["ce_oi_change"] = 0
                row["ce_oi_change_pct"] = 0
                row["ce_volume"] = 0
                row["ce_buildup"] = ""
                row["ce_bid"] = 0
                row["ce_ask"] = 0

            if pe:
                row["pe_ltp"] = pe.ltp
                row["pe_oi"] = pe.oi
                row["pe_oi_change"] = pe.oi_change
                row["pe_oi_change_pct"] = pe.oi_change_pct
                row["pe_volume"] = pe.volume
                row["pe_buildup"] = pe.buildup
                row["pe_bid"] = pe.bid
                row["pe_ask"] = pe.ask
            else:
                row["pe_ltp"] = 0
                row["pe_oi"] = 0
                row["pe_oi_change"] = 0
                row["pe_oi_change_pct"] = 0
                row["pe_volume"] = 0
                row["pe_buildup"] = ""
                row["pe_bid"] = 0
                row["pe_ask"] = 0

            row["is_atm"] = (strike == chain.atm_strike)
            chain_table.append(row)

        # Summary of buildup signals across strikes
        buildup_summary = self._get_buildup_summary()

        return {
            "source": "live_websocket",
            "connected": self._ws_connected,
            "timestamp": chain.timestamp,
            "spot_price": chain.spot_price,
            "atm_strike": chain.atm_strike,
            "expiry": chain.expiry,
            "pcr": chain.pcr,
            "pcr_change": chain.pcr_change,
            "total_ce_oi": chain.total_ce_oi,
            "total_pe_oi": chain.total_pe_oi,
            "ce_oi_change_pct": chain.ce_oi_change_pct,
            "pe_oi_change_pct": chain.pe_oi_change_pct,
            "max_pain_strike": chain.max_pain_strike,
            "highest_ce_oi_strike": chain.highest_ce_oi_strike,
            "highest_pe_oi_strike": chain.highest_pe_oi_strike,
            "chain": chain_table,
            "buildup_summary": buildup_summary,
            "subscribed_tokens": len(self._token_map),
        }

    def _get_buildup_summary(self) -> Dict:
        """Summarize buildup signals across all subscribed strikes."""
        counts = {
            "Long Buildup": 0,
            "Short Buildup": 0,
            "Short Covering": 0,
            "Long Unwinding": 0,
        }
        notable = []  # Strikes with strong signals

        with self._lock:
            for data in self._token_map.values():
                if data.buildup in counts:
                    counts[data.buildup] += 1

                # Flag strong OI moves (> 3% change)
                if abs(data.oi_change_pct) > 3.0:
                    notable.append({
                        "symbol": data.tradingsymbol,
                        "strike": data.strike,
                        "type": data.option_type,
                        "oi_change_pct": data.oi_change_pct,
                        "price_change_pct": data.price_change_pct,
                        "buildup": data.buildup,
                        "oi": data.oi,
                        "volume": data.volume,
                    })

        # Sort notable by OI change magnitude
        notable.sort(key=lambda x: abs(x["oi_change_pct"]), reverse=True)

        # Overall market bias from buildup signals
        bullish = counts["Long Buildup"] + counts["Short Covering"]
        bearish = counts["Short Buildup"] + counts["Long Unwinding"]
        if bullish > bearish * 1.5:
            bias = "BULLISH"
        elif bearish > bullish * 1.5:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return {
            "counts": counts,
            "bias": bias,
            "notable_strikes": notable[:10],  # top 10 strong moves
        }
