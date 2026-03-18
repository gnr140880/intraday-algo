"""
NIFTY Options Engine – Orchestrator

Ties together:
  - ORB strategy (signal generation on NIFTY spot)
  - Scoring engine (multi-factor scoring of option candidates)
  - Risk manager (daily limits, position sizing, square-off)
  - Options chain handling (delta estimation, strike selection)

Runs as a background scheduler inside FastAPI.
"""
import logging
import math
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, date, timedelta

import pandas as pd
import numpy as np

from kite_client import KiteClient
from config import settings
from strategies.nifty_options_orb import NiftyOptionsORBStrategy
from strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from strategies.expiry_premium_sell import ExpiryPremiumSellStrategy
from strategies.pcr_oi_directional import PCROIDirectionalStrategy
from strategies.vwap_breakout import VWAPBreakoutStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.gap_fill import GapFillStrategy
from strategies.straddle_strangle import StraddleStrangleSellStrategy
from strategies.iron_condor import IronCondorStrategy
from strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from strategies.orb_range_scalper import ORBRangeScalperStrategy
from market_regime import detect_regime, filter_signals_by_regime, MarketRegime
from scoring_engine import ScoringEngine, OptionCandidate
from risk_manager import RiskManager, TradeRecord
from telegram_alerts import telegram
from market_data_fallback import (
    fetch_nifty_spot_yf,
    fetch_nifty_history_yf,
    build_option_candidates_from_instruments,
)
from level_calculator import LevelCalculator, IntraDayLevels
from smart_sl_engine import SmartSLEngine, SmartLevels, compute_smart_levels
from exit_signal_generator import (
    ExitSignalGenerator, ExitSignal, PositionContext, ExitPriority, ExitReason,
)
from auto_order_manager import AutoOrderManager, ManagedPosition
from trade_journal import trade_journal
from trailing_sl_manager import TrailingSLManager, TrailingConfig, trailing_sl_manager
from live_oi_feed import LiveOIFeed

logger = logging.getLogger(__name__)

# ============================================================
# Market Session Guard – IST 9:15 AM to 3:30 PM, Mon-Fri
# ============================================================
def is_market_open() -> bool:
    """Check if we are within Indian trading hours (IST 9:15 AM – 3:30 PM, Mon–Fri)."""
    import pytz
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz=tz)

    # Weekend check (0=Monday, 6=Sunday)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False

    # Time check: 09:15 – 15:30 IST
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now.time() <= market_close


def should_skip_new_trades() -> Tuple[bool, str]:
    """
    Check if we should skip new trade entry based on:
    1. Market hours (IST 9:15 AM – 3:30 PM, Mon–Fri)
    2. Trading mode (paper/live only, not 'off')
    Returns: (skip: bool, reason: str)
    """
    if not is_market_open():
        return (True, "Market closed (outside 9:15 AM – 3:30 PM IST or weekend)")

    if settings.trading_mode not in ("paper", "live"):
        return (True, f"Trading mode is '{settings.trading_mode}' (only 'paper' or 'live' allows trades)")

    if not settings.auto_trade_enabled:
        return (True, "Auto-trading disabled in config (auto_trade_enabled=False)")

    return (False, "OK")

# Instrument configuration (multi-instrument support)
INSTRUMENT_CONFIG = {
    "NIFTY": {
        "name": "NIFTY",
        "spot_symbol": "NSE:NIFTY 50",
        "nfo_name": "NIFTY",
        "lot_size_setting": "nifty_lot_size",
        "strike_gap": 50,
        "default_lot_size": 75,
        "token": 256265,
        "expiry_day": 3,            # Thursday (0=Mon)
    },
    "BANKNIFTY": {
        "name": "BANKNIFTY",
        "spot_symbol": "NSE:NIFTY BANK",
        "nfo_name": "BANKNIFTY",
        "lot_size_setting": "banknifty_lot_size",
        "strike_gap": 100,
        "default_lot_size": 30,
        "token": 260105,
        "expiry_day": 2,            # Wednesday
    },
    "FINNIFTY": {
        "name": "FINNIFTY",
        "spot_symbol": "NSE:NIFTY FIN SERVICE",
        "nfo_name": "FINNIFTY",
        "lot_size_setting": "finnifty_lot_size",
        "strike_gap": 50,
        "default_lot_size": 40,
        "token": 257801,
        "expiry_day": 1,            # Tuesday
    },
    "MIDCPNIFTY": {
        "name": "MIDCPNIFTY",
        "spot_symbol": "NSE:NIFTY MID SELECT",
        "nfo_name": "MIDCPNIFTY",
        "lot_size_setting": "midcpnifty_lot_size",
        "strike_gap": 25,
        "default_lot_size": 75,
        "token": 288009,
        "expiry_day": 0,            # Monday
    },
    "SENSEX": {
        "name": "SENSEX",
        "spot_symbol": "BSE:SENSEX",
        "nfo_name": "SENSEX",
        "lot_size_setting": "sensex_lot_size",
        "strike_gap": 100,
        "default_lot_size": 20,
        "token": None,              # BSE token differs
        "expiry_day": 4,            # Friday
    },
}

# Legacy constants (kept for backward compat)
NIFTY_STRIKE_GAP = 50
NIFTY_INSTRUMENT = "NSE:NIFTY 50"
NIFTY_TOKEN = 256265

class NiftyOptionsEngine:
    """
    Main engine – call `run_cycle()` every 5 minutes after 9:30 AM.
    """

    def __init__(self):
        self.strategy = NiftyOptionsORBStrategy()
        self.vwap_strategy = VWAPMeanReversionStrategy()
        self.expiry_strategy = ExpiryPremiumSellStrategy()
        self.hero_zero_strategy = None
        try:
            from strategies.hero_zero_expiry import HeroZeroExpiryStrategy
            self.hero_zero_strategy = HeroZeroExpiryStrategy()
        except Exception:
            self.hero_zero_strategy = None
        # New strategies
        self.pcr_oi_strategy = PCROIDirectionalStrategy()
        self.vwap_breakout_strategy = VWAPBreakoutStrategy()
        self.ema_crossover_strategy = EMACrossoverStrategy()
        self.rsi_divergence_strategy = RSIDivergenceStrategy()
        self.gap_fill_strategy = GapFillStrategy()
        self.straddle_strategy = StraddleStrangleSellStrategy()
        self.iron_condor_strategy = IronCondorStrategy()
        # Sideways / range-bound strategies
        self.bollinger_mr_strategy = BollingerMeanReversionStrategy()
        self.orb_scalper_strategy = ORBRangeScalperStrategy()

        self.scorer = ScoringEngine()
        self.risk_mgr = RiskManager(
            capital=settings.default_capital,
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            max_risk_per_trade_pct=settings.max_risk_per_trade_pct,
            max_concurrent_positions=settings.max_concurrent_positions,
            square_off_time=time(*[int(x) for x in settings.square_off_time.split(":")]),
        )
        self.level_calc = LevelCalculator()
        self.sl_engine = SmartSLEngine()
        self.exit_gen = ExitSignalGenerator(square_off_time=time(15, 15))
        self.order_mgr = AutoOrderManager(self.risk_mgr)
        self.order_mgr.load_state()  # Restore positions after restart
        # Re-register restored positions with risk manager
        for tid, pos in self.order_mgr.positions.items():
            if pos.status != "CLOSED":
                from risk_manager import TradeRecord
                tr = TradeRecord(
                    trade_id=pos.trade_id,
                    symbol=pos.symbol,
                    option_type=pos.option_type,
                    entry_price=pos.entry_price,
                    entry_time=pos.created_at,
                    quantity=pos.quantity,
                    sl=pos.trailing_sl,
                    target=pos.smart_levels.option_t1 if pos.smart_levels else 0,
                    trailing_sl=pos.trailing_sl,
                    score=pos.score,
                )
                self.risk_mgr.register_trade(tr)
        self.journal = trade_journal
        self.tsl_mgr = TrailingSLManager(TrailingConfig(
            breakeven_trigger_pct=settings.tsl_breakeven_trigger_pct,
            early_breakeven_pct=settings.tsl_early_breakeven_pct,
            trail_t1_atr_mult=settings.tsl_trail_t1_atr_mult,
            trail_t2_atr_mult=settings.tsl_trail_t2_atr_mult,
            tight_atr_mult=settings.tsl_tight_atr_mult,
            tight_trigger_pct=settings.tsl_tight_trigger_pct,
            trail_pct_initial=settings.tsl_trail_pct_initial,
            trail_pct_t2=settings.tsl_trail_pct_t2,
            trail_pct_tight=settings.tsl_trail_pct_tight,
            swing_lookback_candles=settings.tsl_swing_lookback,
            sl_buffer_pct=settings.tsl_sl_buffer_pct,
        ))

        # Auto-trading mode: 'off' (signals only), 'paper', 'live'
        self.auto_trade_mode = settings.trading_mode  # 'paper' or 'live'

        # WebSocket tick state
        self._tick_data: Dict[int, Dict] = {}  # token → latest tick data
        self._ws_connected = False

        # Live OI Feed (WebSocket-based)
        self._oi_feed: Optional[LiveOIFeed] = None
        self._oi_feed_started = False

        # State
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.nifty_spot: float = 0.0
        self.last_signal = None
        self.scored_candidates: List[OptionCandidate] = []
        self.all_candidates: List[OptionCandidate] = []
        self.nifty_token: Optional[int] = NIFTY_TOKEN
        self.nifty_spot_symbol: Optional[str] = None
        self._orb_captured: bool = False
        self._cached_levels: Optional[IntraDayLevels] = None
        self._cached_df: Optional[pd.DataFrame] = None
        self._cached_oi_data: Dict = {}  # OI analysis: ce/pe change %, per-strike OI
        self._today: Optional[date] = None
        self._cycle_count = 0
        self._alerted_signal_key: Optional[str] = None   # "BUY@24700" — skip re-alerting same signal
        self._alerted_strikes: set = set()                # strikes already sent via Telegram
        self._traded_strikes: set = set()                 # strikes already attempted for execution

        # Safety filters state
        self._india_vix: float = 0.0           # Current India VIX reading
        self._last_sl_hit: Dict[str, datetime] = {}  # direction → last SL-hit time for cooldown

        # Gap-day classification state
        self._gap_type: str = "NONE"           # GAP_UP, LARGE_GAP_UP, GAP_DOWN, LARGE_GAP_DOWN, NONE
        self._gap_pct: float = 0.0             # gap magnitude in %

        # Multi-timeframe state
        self._st_dir_15m: int = 0              # supertrend direction on 15m
        self._st_dir_1h: int = 0               # supertrend direction on 1h

        # Dashboard state (live)
        self.dashboard_state: Dict = {
            "engine_status": "IDLE",
            "orb": {"high": None, "low": None, "captured": False},
            "signal": None,
            "top_candidates": [],
            "risk": {},
            "trades": [],
            "levels": {},
            "auto_trade_mode": self.auto_trade_mode,
            "active_positions": [],
            "order_log": [],
            "oi_analysis": {"ce_oi_change_pct": 0, "pe_oi_change_pct": 0},
            "last_update": None,
        }
        self._nfo_instruments = []
        self._option_index = {}
        self._strike_ladder = {}
        self._load_nfo_instruments()
        self._build_option_index()
        self._init_nifty_spot_token()

    def _build_option_index(self):
        """Build option index for O(1) lookup and strike ladder."""
        option_index = {}
        strike_ladder = {}
        for row in self._nfo_instruments:
            key = (
                row["name"],
                str(row["expiry"]),
                int(row["strike"]),
                row["instrument_type"]
            )
            option_index[key] = row
            # Build strike ladder
            ladder_key = (row["name"], str(row["expiry"]))
            strike_ladder.setdefault(ladder_key, set()).add(int(row["strike"]))
        # Convert sets to sorted lists
        for k in strike_ladder:
            strike_ladder[k] = sorted(list(strike_ladder[k]))
        self._option_index = option_index
        self._strike_ladder = strike_ladder
        logger.info(f"Option index built: {len(option_index)} keys, {len(strike_ladder)} ladders")

    @staticmethod
    def get_atm_strike(spot, strike_gap=50):
        """Get ATM strike rounded to nearest strike gap."""
        return round(spot / strike_gap) * strike_gap

    def get_option(self, symbol, expiry, strike, option_type):
        """O(1) lookup for option instrument."""
        key = (symbol, str(expiry), int(strike), option_type)
        return self._option_index.get(key)

    def get_strike_ladder(self, symbol, expiry):
        """Get sorted list of strikes for symbol/expiry."""
        return self._strike_ladder.get((symbol, str(expiry)), [])

    def _load_nfo_instruments(self):
        """Load NFO instruments from instruments.csv and cache as list of dicts."""
        import csv
        from pathlib import Path
        csv_path = Path(__file__).resolve().parent / "instruments.csv"
        instruments = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only cache NFO options
                    if row.get("segment") == "NFO-OPT":
                        # Convert fields as needed
                        try:
                            row["strike"] = float(row.get("strike", 0))
                        except Exception:
                            row["strike"] = 0.0
                        row["expiry"] = row.get("expiry")
                        instruments.append(row)
            self._nfo_instruments = instruments
            logger.info(f"Loaded {len(instruments)} NFO option instruments from CSV")
        except Exception as e:
            self._nfo_instruments = []
            logger.error(f"Failed to load NFO instruments from CSV: {e}")

    def _init_nifty_spot_token(self):
        """
        Fetch and set the correct NIFTY spot instrument token and symbol from Zerodha instruments.
        """
        try:
            kite = KiteClient.get_instance()
            instruments = kite.get_instruments("NSE")
            for inst in instruments:
                if inst.get("name") == "NIFTY 50" or inst.get("tradingsymbol") == "NIFTY 50" or inst.get("name") == "NIFTY":
                    self.nifty_token = inst["instrument_token"]
                    self.nifty_spot_symbol = inst["tradingsymbol"]
                    logger.info(f"NIFTY spot instrument resolved: token={self.nifty_token}, symbol={self.nifty_spot_symbol}")
                    return
            logger.warning("NIFTY spot instrument not found in Zerodha instruments.")
        except Exception as e:
            logger.error(f"Failed to resolve NIFTY spot instrument: {e}")

    # ------------------------------------------------------------------
    # Initialise for the day
    # ------------------------------------------------------------------
    def init_day(self):
        today = date.today()
        if self._today == today:
            return
        self._today = today
        self._orb_captured = False
        self.orb_high = None
        self.orb_low = None
        self.last_signal = None
        self.scored_candidates = []
        self.all_candidates = []
        self._cycle_count = 0
        self.risk_mgr.reset_daily()
        self.journal.reset_daily()
        self._last_sl_hit = {}  # Reset cooldown on new day
        self._alerted_signal_key = None
        self._alerted_strikes = set()
        self._traded_strikes = set()
        self._gap_type = "NONE"
        self._gap_pct = 0.0
        self._st_dir_15m = 0
        self._st_dir_1h = 0
        self.dashboard_state["engine_status"] = "WAITING_ORB"
        # Stop previous day's OI feed if running
        if self._oi_feed and self._oi_feed.is_running:
            self._oi_feed.stop()
            self._oi_feed_started = False
        logger.info(f"NiftyOptionsEngine initialised for {today}")

    # ------------------------------------------------------------------
    # Start / manage Live OI Feed
    # ------------------------------------------------------------------
    def _start_oi_feed(self, kite: KiteClient):
        """
        Start the live OI feed via Kite WebSocket.
        Subscribes to NIFTY weekly options around ATM in MODE_FULL.
        Safe to call multiple times — will only start once.
        """
        if not settings.enable_live_oi_feed:
            return
        if self._oi_feed_started and self._oi_feed and self._oi_feed.is_running:
            # Update ATM if spot moved
            if self.nifty_spot > 0:
                self._oi_feed.update_atm(self.nifty_spot)
            return
        if not kite.is_connected:
            return
        if self.nifty_spot <= 0:
            return

        try:
            nifty_token = self._resolve_nifty_token(kite)
            inst_cfg = INSTRUMENT_CONFIG.get("NIFTY", INSTRUMENT_CONFIG["NIFTY"])

            self._oi_feed = LiveOIFeed(
                kite_client=kite,
                strike_gap=inst_cfg["strike_gap"],
                num_strikes=settings.oi_feed_num_strikes,
            )
            self._oi_feed.start(
                spot_price=self.nifty_spot,
                nifty_spot_token=nifty_token,
            )
            self._oi_feed_started = True
            logger.info("Live OI feed started successfully")
        except Exception as e:
            logger.warning(f"Failed to start live OI feed: {e}")
            self._oi_feed_started = False

    def _get_live_oi_data(self) -> Optional[Dict]:
        """
        Get live OI analysis from WebSocket feed.
        Returns None if feed is not running.
        """
        if (
            self._oi_feed
            and self._oi_feed.is_running
        ):
            try:
                return self._oi_feed.get_oi_analysis()
            except Exception as e:
                logger.debug(f"Live OI data fetch failed: {e}")
        return None

    def _update_candidates_from_live_oi(self):
        """
        Update scored candidates with live OI, volume, LTP from WebSocket feed.
        This is more reliable than Kite get_ltp since data comes from MODE_FULL ticks.
        """
        if not self._oi_feed or not self._oi_feed.is_running:
            return
        if not self.scored_candidates:
            return

        updated = 0
        for c in self.scored_candidates:
            live = self._oi_feed.get_candidate_live_data(c.tradingsymbol)
            if live and live["ltp"] > 0:
                c.ltp = live["ltp"]
                c.volume = live["volume"]
                c.oi = live["oi"]
                c.oi_change_pct = live["oi_change_pct"]
                c.bid = live["bid"] if live["bid"] > 0 else round(live["ltp"] * 0.995, 2)
                c.ask = live["ask"] if live["ask"] > 0 else round(live["ltp"] * 1.005, 2)
                c.price_source = "live"
                c.oi_interpretation = live["buildup"]
                # IV spike filter
                if hasattr(c, "iv_percentile") and c.iv_percentile > 85:
                    c.skip_buy = True
                updated += 1

        if updated > 0:
            # Re-score with live data
            for c in self.scored_candidates:
                self.scorer.score_candidate(c, levels=self._cached_levels)
            logger.info(f"Updated {updated} candidates from live OI feed")

    # ------------------------------------------------------------------
    # Refresh option LTPs for all candidates using Kite get_ltp
    # ------------------------------------------------------------------
    def _refresh_candidate_ltps(self, kite: KiteClient):
        """
        Refresh LTP (and recalculate entry/SL/targets) for all scored candidates
        using Kite get_ltp. This ensures dashboard shows current market prices.
        Also updates spot_price so SL/target mapping reflects current NIFTY level.
        """
        if not self.scored_candidates:
            return
        symbols = [f"NFO:{c.tradingsymbol}" for c in self.scored_candidates]
        try:
            ltp_data = kite.get_ltp(symbols[:20])  # batch limit
            updated = 0
            for c in self.scored_candidates:
                key = f"NFO:{c.tradingsymbol}"
                ltp = ltp_data.get(key, {}).get("last_price", 0)
                if ltp > 0 and ltp != c.ltp:
                    c.ltp = ltp
                    c.bid = round(ltp * 0.995, 2)
                    c.ask = round(ltp * 1.005, 2)
                    c.price_source = "live"
                    updated += 1
            # Also refresh NIFTY spot so SL/target mapping is current
            if self.nifty_spot > 0:
                for c in self.scored_candidates:
                    c.spot_price = self.nifty_spot
            if updated > 0:
                # Re-score with updated LTPs to recalculate entry/SL/targets
                for c in self.scored_candidates:
                    self.scorer.score_candidate(c, levels=self._cached_levels)
                logger.info(f"Refreshed LTP for {updated} scored candidates")
        except Exception as e:
            # Mark as stale if refresh failed — user knows prices may be outdated
            for c in self.scored_candidates:
                if c.price_source != "live":
                    c.price_source = "estimated"
            logger.debug(f"LTP refresh for candidates failed: {e}")

    # ------------------------------------------------------------------
    # Fetch India VIX for safety filter
    # ------------------------------------------------------------------
    def _fetch_india_vix(self, kite: KiteClient) -> float:
        """Fetch current India VIX. Returns 0 on failure (trades proceed)."""
        try:
            vix_data = kite.get_ltp(["NSE:INDIA VIX"])
            vix = vix_data.get("NSE:INDIA VIX", {}).get("last_price", 0.0)
            if vix and vix > 0:
                self._india_vix = float(vix)
                return self._india_vix
        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")
        return self._india_vix  # return cached value if fetch fails

    # ------------------------------------------------------------------
    # Re-entry cooldown check
    # ------------------------------------------------------------------
    def _check_reentry_cooldown(self, direction: str) -> bool:
        """Returns True if cooldown is active (should NOT trade)."""
        last_hit = self._last_sl_hit.get(direction)
        if last_hit is None:
            return False
        elapsed = (datetime.now() - last_hit).total_seconds() / 60.0
        cooldown = settings.reentry_cooldown_minutes
        if elapsed < cooldown:
            logger.info(
                f"Re-entry cooldown active for {direction}: "
                f"{elapsed:.0f}/{cooldown} min since last SL hit"
            )
            return True
        return False

    def record_sl_hit(self, direction: str):
        """Record SL hit time for re-entry cooldown."""
        self._last_sl_hit[direction] = datetime.now()
        logger.info(f"SL hit recorded for {direction} – cooldown started")

    # ------------------------------------------------------------------
    # Resolve NIFTY instrument token
    # ------------------------------------------------------------------
    def _resolve_nifty_token(self, kite: KiteClient) -> Optional[int]:
        if self.nifty_token:
            return self.nifty_token
        try:
            instruments = kite.get_instruments("NSE")
            for inst in instruments:
                if inst["tradingsymbol"] == "NIFTY 50" or inst["name"] == "NIFTY 50":
                    self.nifty_token = inst["instrument_token"]
                    return self.nifty_token
        except Exception as e:
            logger.error(f"Failed to resolve NIFTY token: {e}")
        return None

    # ------------------------------------------------------------------
    # Fetch NIFTY spot 5-min data
    # ------------------------------------------------------------------
    def _fetch_spot_data(self, kite: KiteClient) -> Optional[pd.DataFrame]:
        """Fetch NIFTY 5-min data. Tries Kite historical, falls back to yfinance."""
        token = self._resolve_nifty_token(kite)
        if token is not None:
            try:
                to_dt = datetime.now()
                from_dt = to_dt - timedelta(days=5)
                raw = kite.get_historical_data(token, from_dt, to_dt, "5minute")
                if raw:
                    return pd.DataFrame(raw)
            except Exception as e:
                logger.info(f"Kite historical failed ({type(e).__name__}), trying yfinance…")

        # Fallback to yfinance
        df = fetch_nifty_history_yf(days=5, interval="5m")
        if df is not None and not df.empty:
            logger.info(f"Using yfinance spot data: {len(df)} candles")
        return df

    # ------------------------------------------------------------------
    # Delta estimation (Black-Scholes approximation)
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_delta(
        spot: float, strike: float, tte_days: float, iv: float, option_type: str
    ) -> float:
        """
        Quick delta estimate using simplified Black-Scholes.
        tte_days: time to expiry in calendar days
        iv: implied vol as decimal (e.g. 0.15 for 15%)
        """
        try:
            from scipy.stats import norm
        except ImportError:
            class norm:
                @staticmethod
                def cdf(x):
                    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        if tte_days <= 0 or iv <= 0:
            # ITM → ~1, OTM → ~0
            if option_type == "CE":
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0

        T = tte_days / 365
        r = 0.06  # risk-free rate approx
        d1 = (math.log(spot / strike) + (r + iv ** 2 / 2) * T) / (iv * math.sqrt(T))
        if option_type == "CE":
            return round(norm.cdf(d1), 4)
        else:
            return round(norm.cdf(d1) - 1, 4)

    # ------------------------------------------------------------------
    # OI Analysis — fetch previous day OI + compute change %
    # ------------------------------------------------------------------
    def _fetch_oi_analysis(
        self,
        kite: KiteClient,
        weekly_opts: List[Dict],
        atm_strike: float,
        spot: float,
        nearest_expiry,
        strike_gap: int = 50,
    ) -> Dict:
        """
        Fetch OI data for option chain strikes and compute:
          - Per-strike OI change % vs previous close
          - Aggregate CE/PE OI change % for ATM ± 3 strikes
        Returns: {
          "strike_oi": {(strike, type): {"oi": int, "prev_oi": int, "oi_change_pct": float}},
          "ce_oi_change_pct": float,   # aggregate CE OI change near ATM
          "pe_oi_change_pct": float,   # aggregate PE OI change near ATM
        }
        """
        result = {
            "strike_oi": {},
            "ce_oi_change_pct": 0.0,
            "pe_oi_change_pct": 0.0,
        }

        # ATM ± 3 strikes for aggregate CE/PE OI analysis
        oi_strikes = range(
            int(atm_strike - 3 * strike_gap),
            int(atm_strike + 4 * strike_gap),
            strike_gap,
        )
        oi_instruments = [
            i for i in weekly_opts
            if i["strike"] in oi_strikes and i["instrument_type"] in ("CE", "PE")
        ]

        if not oi_instruments:
            return result

        # 1. Fetch current OI via quotes for both CE and PE near ATM
        oi_symbols = [f"NFO:{i['tradingsymbol']}" for i in oi_instruments[:40]]
        try:
            oi_quotes = kite.get_quote(oi_symbols)
        except Exception as e:
            logger.warning(f"OI quote fetch failed: {e}")
            return result

        # 2. Fetch previous day closing OI via historical data (day candle with oi=True)
        prev_oi_map = {}  # instrument_token → prev day OI
        yesterday = date.today() - timedelta(days=1)
        # Go back up to 5 days to find last trading day
        from_dt = datetime.combine(date.today() - timedelta(days=5), time(0, 0))
        to_dt = datetime.combine(date.today() - timedelta(days=1), time(23, 59))

        for inst in oi_instruments:
            token = inst["instrument_token"]
            if token in prev_oi_map:
                continue
            try:
                hist = kite.get_historical_data(
                    token, from_dt, to_dt, "day", oi=True
                )
                if hist:
                    # Last day's OI
                    last_bar = hist[-1]
                    prev_oi_map[token] = last_bar.get("oi", 0)
            except Exception:
                pass  # Some instruments may not have history

        # 3. Compute per-strike OI change and aggregate CE/PE OI changes
        total_ce_oi = 0
        total_ce_prev_oi = 0
        total_pe_oi = 0
        total_pe_prev_oi = 0

        for inst in oi_instruments:
            key = f"NFO:{inst['tradingsymbol']}"
            q = oi_quotes.get(key, {})
            curr_oi = q.get("oi", 0)
            token = inst["instrument_token"]
            prev_oi = prev_oi_map.get(token, 0)

            oi_change_pct = 0.0
            if prev_oi > 0:
                oi_change_pct = round((curr_oi - prev_oi) / prev_oi * 100, 2)

            opt_type = inst["instrument_type"]
            strike = inst["strike"]
            result["strike_oi"][(strike, opt_type)] = {
                "oi": curr_oi,
                "prev_oi": prev_oi,
                "oi_change_pct": oi_change_pct,
            }

            # Aggregate for ATM ± 3 strikes
            if opt_type == "CE":
                total_ce_oi += curr_oi
                total_ce_prev_oi += prev_oi
            else:
                total_pe_oi += curr_oi
                total_pe_prev_oi += prev_oi

        # Aggregate change %
        if total_ce_prev_oi > 0:
            result["ce_oi_change_pct"] = round(
                (total_ce_oi - total_ce_prev_oi) / total_ce_prev_oi * 100, 2
            )
        if total_pe_prev_oi > 0:
            result["pe_oi_change_pct"] = round(
                (total_pe_oi - total_pe_prev_oi) / total_pe_prev_oi * 100, 2
            )

        logger.info(
            f"OI Analysis: CE OI change {result['ce_oi_change_pct']:+.1f}%, "
            f"PE OI change {result['pe_oi_change_pct']:+.1f}% "
            f"(ATM {atm_strike}, {len(result['strike_oi'])} strikes analysed)"
        )
        return result

    # ------------------------------------------------------------------
    # IV Percentile estimation (heuristic without full IV history)
    # ------------------------------------------------------------------
    def _estimate_iv_percentile(self, iv_current: float, tte_days: float) -> float:
        """
        Estimate IV percentile rank using a heuristic approach.
        NIFTY weekly options typical IV range: 8-40%.
        Returns 0-100 percentile value.
        """
        # Typical NIFTY IV ranges by TTE
        if tte_days <= 1:
            # Expiry day: IV is usually higher but collapses fast
            iv_low, iv_high = 10.0, 60.0
        elif tte_days <= 3:
            iv_low, iv_high = 8.0, 45.0
        elif tte_days <= 7:
            iv_low, iv_high = 9.0, 35.0
        else:
            iv_low, iv_high = 10.0, 30.0

        if iv_current <= iv_low:
            return 5.0
        if iv_current >= iv_high:
            return 95.0

        # Linear interpolation within range
        pct = ((iv_current - iv_low) / (iv_high - iv_low)) * 100.0
        return round(min(max(pct, 0.0), 100.0), 1)

    # ------------------------------------------------------------------
    # Gap-day classification
    # ------------------------------------------------------------------
    def _compute_gap_classification(self, df: pd.DataFrame):
        """
        Detect gap-up / gap-down based on today's open vs previous close.
        Updates self._gap_type and self._gap_pct.
        """
        try:
            today = date.today()
            # Get today's open (first 5m candle of today)
            if "date" in df.columns:
                df_today = df[pd.to_datetime(df["date"]).dt.date == today]
            else:
                df_today = df[df.index.date == today] if hasattr(df.index, 'date') else df.tail(78)

            if df_today.empty or len(df_today) < 1:
                return

            today_open = float(df_today.iloc[0]["open"])

            # Get yesterday's close (last candle before today)
            if "date" in df.columns:
                df_prev = df[pd.to_datetime(df["date"]).dt.date < today]
            else:
                df_prev = df[df.index.date < today] if hasattr(df.index, 'date') else pd.DataFrame()

            if df_prev.empty:
                return

            prev_close = float(df_prev.iloc[-1]["close"])
            if prev_close <= 0:
                return

            gap_pct = ((today_open - prev_close) / prev_close) * 100.0
            self._gap_pct = round(gap_pct, 3)

            large_thr = settings.large_gap_threshold_pct
            gap_thr = settings.gap_threshold_pct

            if abs(gap_pct) >= large_thr:
                self._gap_type = "LARGE_GAP_UP" if gap_pct > 0 else "LARGE_GAP_DOWN"
            elif abs(gap_pct) >= gap_thr:
                self._gap_type = "GAP_UP" if gap_pct > 0 else "GAP_DOWN"
            else:
                self._gap_type = "NONE"

            if self._gap_type != "NONE":
                logger.info(f"Gap-day detected: {self._gap_type} ({self._gap_pct:+.2f}%)")

        except Exception as e:
            logger.debug(f"Gap classification failed: {e}")
            self._gap_type = "NONE"
            self._gap_pct = 0.0

    # ------------------------------------------------------------------
    # Multi-timeframe data: fetch 15m + 1h and compute Supertrend
    # ------------------------------------------------------------------
    def _fetch_multi_tf_data(self, kite: KiteClient):
        """
        Fetch 15-minute and 1-hour candles, compute Supertrend on each,
        and store the latest direction in self._st_dir_15m / self._st_dir_1h.
        """
        if not settings.enable_multi_tf:
            return

        token = self._resolve_nifty_token(kite)
        if token is None:
            return

        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=10)

        for interval, attr_name in [("15minute", "_st_dir_15m"), ("hour", "_st_dir_1h")]:
            try:
                raw = kite.get_historical_data(token, from_dt, to_dt, interval)
                if raw:
                    df_tf = pd.DataFrame(raw)
                    if len(df_tf) >= 14:
                        df_tf = self.strategy.compute_supertrend(df_tf)
                        if "st_dir" in df_tf.columns:
                            setattr(self, attr_name, int(df_tf.iloc[-1]["st_dir"]))
                            logger.debug(
                                f"Multi-TF {interval}: st_dir={getattr(self, attr_name)}"
                            )
            except Exception as e:
                logger.debug(f"Multi-TF {interval} fetch failed: {e}")

    # ------------------------------------------------------------------
    # Build option candidates from chain
    # ------------------------------------------------------------------
    def _build_candidates(
        self,
        kite: KiteClient,
        spot: float,
        signal_dir: int,
        df_spot: pd.DataFrame,
        instrument_name: str = "NIFTY",
    ) -> List[OptionCandidate]:
        """
        Fetch NFO options chain near ATM, filter by delta,
        build OptionCandidate objects.
        
        Args:
            instrument_name: "NIFTY" or "BANKNIFTY" (uses INSTRUMENT_CONFIG)
        """
        candidates = []
        inst_cfg = INSTRUMENT_CONFIG.get(instrument_name, INSTRUMENT_CONFIG["NIFTY"])
        strike_gap = inst_cfg["strike_gap"]
        nfo_name = inst_cfg["nfo_name"]
        # Use API for live/paper trading, CSV for offline/backtest
        # Use API for live/paper trading, CSV for offline/backtest
        nfo_instruments = []
        if self.auto_trade_mode in ("live", "paper"):
            try:
                kite_instruments = kite.get_instruments("NFO")
                nfo_instruments = [row for row in kite_instruments if row.get("segment") == "NFO-OPT"]
                logger.info(f"Loaded {len(nfo_instruments)} NFO option instruments from API")
                self._nfo_instruments = nfo_instruments
                self._build_option_index()
            except Exception as e:
                logger.error(f"Failed to load NFO instruments from API: {e}")
                nfo_instruments = self._nfo_instruments if hasattr(self, '_nfo_instruments') else []
        else:
            nfo_instruments = self._nfo_instruments if hasattr(self, '_nfo_instruments') else []
            if not nfo_instruments:
                logger.error("NFO instruments not cached.")
                return []
        # Option index is now built and available
        # Example usage: O(1) ATM lookup
        # These lines should be inside the function body, not at top-level
        # ATM strike and option lookup
        # (If you want to use these, place them after nearest_expiry is defined)

        # Current expiry: nearest weekly
        today = date.today()
        nifty_options = [
            i for i in nfo_instruments
            if i.get("name") == nfo_name
            and i.get("segment") == "NFO-OPT"
            and i.get("expiry")
            and i["expiry"] >= today
        ]
        if not nifty_options:
            logger.warning(f"No {nfo_name} options found in NFO instruments")
            return candidates

        # Sort by expiry, take nearest
        nifty_options.sort(key=lambda x: x["expiry"])
        nearest_expiry = nifty_options[0]["expiry"]
        weekly_opts = [i for i in nifty_options if i["expiry"] == nearest_expiry]

        # ATM strike
        atm_strike = round(spot / strike_gap) * strike_gap
        strike_range = range(
            int(atm_strike - 5 * strike_gap),
            int(atm_strike + 6 * strike_gap),
            strike_gap,
        )

        # Filter CE if bullish, PE if bearish (plus some of the other side for hedging)
        primary_type = "CE" if signal_dir == 1 else "PE"
        relevant = [
            i for i in weekly_opts
            if i["strike"] in strike_range and i["instrument_type"] == primary_type
        ]

        if not relevant:
            return candidates

        # --- OI Analysis: fetch previous day OI + compute CE/PE change % ---
        oi_data = {"strike_oi": {}, "ce_oi_change_pct": 0.0, "pe_oi_change_pct": 0.0}
        try:
            oi_data = self._fetch_oi_analysis(
                kite, weekly_opts, atm_strike, spot, nearest_expiry,
                strike_gap=strike_gap,
            )
            # Cache for strategy confirmation use
            self._cached_oi_data = oi_data
        except Exception as e:
            logger.warning(f"OI analysis failed ({type(e).__name__}): {e}")

        ce_oi_chg = oi_data.get("ce_oi_change_pct", 0.0)
        pe_oi_chg = oi_data.get("pe_oi_change_pct", 0.0)
        strike_oi = oi_data.get("strike_oi", {})

        # Fetch quotes for these instruments
        symbols = [f"NFO:{i['tradingsymbol']}" for i in relevant[:20]]
        try:
            quotes = kite.get_quote(symbols)
        except Exception as e:
            logger.error(f"Failed to fetch option quotes: {e}")
            return candidates

        tte_days = (nearest_expiry - today).days
        if tte_days < 1:
            tte_days = 0.5  # expiry day

        # Supertrend / MACD from spot data
        st_dir = int(df_spot.iloc[-1].get("st_dir", 0)) if "st_dir" in df_spot.columns else signal_dir
        macd_hist = float(df_spot.iloc[-1].get("macd_hist", 0)) if "macd_hist" in df_spot.columns else 0
        macd_hist_prev = float(df_spot.iloc[-2].get("macd_hist", 0)) if "macd_hist" in df_spot.columns else 0
        atr_val = float(self.strategy.calculate_atr(df_spot).iloc[-1])

        for inst in relevant:
            key = f"NFO:{inst['tradingsymbol']}"
            q = quotes.get(key, {})
            if not q:
                continue

            ltp = q.get("last_price", 0)
            volume = q.get("volume", 0)
            oi = q.get("oi", 0)
            depth = q.get("depth", {})
            bid = depth.get("buy", [{}])[0].get("price", 0) if depth.get("buy") else 0
            ask = depth.get("sell", [{}])[0].get("price", 0) if depth.get("sell") else 0

            # Estimate IV from LTP (simplified: use a reasonable guess)
            iv = q.get("implied_volatility", 15) / 100
            if iv <= 0:
                iv = 0.15

            delta = self.estimate_delta(spot, inst["strike"], tte_days, iv, primary_type)

            # Delta filter: use expiry-day-aware delta range
            is_expiry_day = (nearest_expiry == today)
            if is_expiry_day:
                d_min = settings.expiry_day_delta_min
                d_max = settings.expiry_day_delta_max
            else:
                d_min = settings.delta_min
                d_max = settings.delta_max
            if abs(delta) < d_min or abs(delta) > d_max:
                continue

            # Skip buying options on expiry day if disabled
            if is_expiry_day and not settings.allow_expiry_day_buys:
                continue

            # Volume spike: compare to 20-bar option volume (simple: use daily volume > 5000)
            avg_volume = np.mean([i.get("volume", 0) for i in relevant]) if relevant else 0
            vol_spike = volume > avg_volume * 2 if avg_volume > 0 else volume > 5000

            # OI change % for this specific strike
            strike_oi_info = strike_oi.get((inst["strike"], primary_type), {})
            oi_change_pct = strike_oi_info.get("oi_change_pct", 0.0)
            prev_oi = strike_oi_info.get("prev_oi", 0)

            # IV Percentile estimation (compare current IV to typical range)
            iv_pct_val = self._estimate_iv_percentile(iv * 100, tte_days)

            c = OptionCandidate(
                tradingsymbol=inst["tradingsymbol"],
                instrument_token=inst["instrument_token"],
                strike=inst["strike"],
                option_type=primary_type,
                expiry=str(nearest_expiry),
                ltp=ltp,
                spot_price=spot,
                delta=delta,
                iv=round(iv * 100, 2),
                volume=volume,
                oi=oi,
                bid=bid,
                ask=ask,
                oi_change_pct=oi_change_pct,
                ce_oi_change_pct=ce_oi_chg,
                pe_oi_change_pct=pe_oi_chg,
                prev_oi=prev_oi,
                iv_percentile=iv_pct_val,
                tte_days=tte_days,
                is_expiry_day=is_expiry_day,
                gap_type=self._gap_type,
                gap_pct=self._gap_pct,
                supertrend_15m=self._st_dir_15m,
                supertrend_1h=self._st_dir_1h,
                orb_high=self.orb_high or 0,
                orb_low=self.orb_low or 0,
                supertrend_dir=st_dir,
                macd_hist=macd_hist,
                macd_hist_prev=macd_hist_prev,
                vol_spike=vol_spike,
                atr=atr_val,
                price_source="live",
            )
            candidates.append(c)

        return candidates

    # ------------------------------------------------------------------
    # Execute trade on top candidate
    # ------------------------------------------------------------------
    def _execute_trade(
        self, kite: KiteClient, candidate: OptionCandidate, qty_factor: float = 1.0
    ) -> Optional[ManagedPosition]:
        """
        Execute a trade using smart SL levels and auto order management.
        Uses VWAP/CPR/S-R based SL and targets mapped to option premium via delta.
        qty_factor: multiplier for position sizing (e.g. 0.5 when VIX is elevated).
        """
        # Compute smart levels on NIFTY spot
        direction = "BUY" if candidate.option_type == "CE" else "SELL"
        levels = self._cached_levels or IntraDayLevels(
            orb_high=self.orb_high or 0,
            orb_low=self.orb_low or 0,
            atr=candidate.atr,
        )

        smart = compute_smart_levels(
            entry=candidate.spot_price,
            direction=direction,
            levels=levels,
        )

        # Map spot SL/targets to option premium via delta
        SmartSLEngine.map_to_option_premium(
            smart=smart,
            option_entry=candidate.entry_price if candidate.entry_price > 0 else candidate.ltp,
            delta=candidate.delta,
            direction=direction,
        )

        # Calculate quantity from risk
        entry_price = candidate.entry_price if candidate.entry_price > 0 else candidate.ltp
        risk_per_unit = abs(entry_price - smart.option_sl)
        if risk_per_unit <= 0:
            risk_per_unit = entry_price * 0.10  # fallback 10%

        qty = self.risk_mgr.calculate_quantity(
            entry_price, smart.option_sl, settings.nifty_lot_size
        )
        # Apply VIX-based size reduction
        if qty_factor < 1.0 and qty > settings.nifty_lot_size:
            reduced_lots = max(1, int((qty / settings.nifty_lot_size) * qty_factor))
            qty = reduced_lots * settings.nifty_lot_size
            logger.info(f"Qty reduced by factor {qty_factor}: {qty} units")
        if qty <= 0:
            skip_reason = f"Trade skipped: calculated quantity is zero for {candidate.tradingsymbol} | Entry: {entry_price} | SL: {smart.option_sl} | Risk/unit: {risk_per_unit}"
            logger.warning(skip_reason)
            self.dashboard_state.setdefault("skipped_signals", []).append({
                "symbol": candidate.tradingsymbol,
                "reason": skip_reason,
                "entry": entry_price,
                "sl": smart.option_sl,
                "score": candidate.score,
                "time": datetime.now().isoformat(),
            })
            return None

        # Use AutoOrderManager for entry + SL placement
        pos = self.order_mgr.enter_trade(
            kite=kite,
            symbol=candidate.tradingsymbol,
            option_type=candidate.option_type,
            smart_levels=smart,
            quantity=qty,
            delta=candidate.delta,
            score=candidate.score,
        )

        if pos is None:
            skip_reason = f"Trade skipped: risk manager or broker failure for {candidate.tradingsymbol}"
            logger.warning(skip_reason)
            self.dashboard_state.setdefault("skipped_signals", []).append({
                "symbol": candidate.tradingsymbol,
                "reason": skip_reason,
                "entry": entry_price,
                "sl": smart.option_sl,
                "score": candidate.score,
                "time": datetime.now().isoformat(),
            })
        return pos

    # ------------------------------------------------------------------
    # Square off all positions
    # ------------------------------------------------------------------
    def square_off_all(self, kite: KiteClient, reason: str = "SQUARED_OFF"):
        """Square off all open positions via auto order manager."""
        # Collect positions that will be squared off for journal recording
        pre_close_positions = {}
        for tid, trade in self.risk_mgr.open_positions.items():
            pre_close_positions[tid] = trade
        for tid, pos in self.order_mgr.positions.items():
            if pos.status != "CLOSED":
                pre_close_positions[tid] = pos

        # Square off via auto order manager (handles SL cancellation + market sell)
        self.order_mgr.square_off_all(kite, reason)

        # Also handle any positions in the old risk manager that aren't in order_mgr
        positions = self.risk_mgr.get_square_off_list()
        for trade in positions:
            if trade.trade_id in self.order_mgr.positions:
                continue  # already handled by order_mgr
            try:
                ltp_data = kite.get_ltp([f"NFO:{trade.symbol}"])
                ltp = ltp_data.get(f"NFO:{trade.symbol}", {}).get("last_price", trade.entry_price)
                kite.place_order(
                    tradingsymbol=trade.symbol,
                    exchange="NFO",
                    transaction_type="SELL",
                    quantity=trade.quantity,
                    order_type="MARKET",
                    product="MIS",
                    tag="SquareOff",
                )
                closed = self.risk_mgr.close_trade(trade.trade_id, ltp, reason)
                if closed:
                    telegram.alert_trade_exit(
                        symbol=closed.symbol, option_type=closed.option_type,
                        entry_price=closed.entry_price, exit_price=ltp,
                        qty=closed.quantity, pnl=closed.pnl,
                        reason=reason, trade_id=closed.trade_id,
                    )
            except Exception as e:
                logger.error(f"Square-off failed for {trade.symbol}: {e}")

        self.dashboard_state["engine_status"] = "SQUARED_OFF"

        # Record all squared-off trades in journal
        for tid in pre_close_positions:
            trade = self.risk_mgr.trades
            for t in trade:
                if t.trade_id == tid and t.status != "OPEN":
                    self.journal.close_candidate(
                        trade_id=tid,
                        exit_price=t.exit_price,
                        pnl=t.pnl,
                        exit_reason=reason,
                    )
                    break

    # ------------------------------------------------------------------
    # Offline / market-closed data fetch
    # ------------------------------------------------------------------
    def fetch_offline_data(self) -> Dict:
        """
        Fetch NIFTY spot LTP + last session's option chain scored candidates
        even when the market is closed. Updates dashboard_state and returns it.
        """
        # Auto-select exact option contract for signal
        top_option = None
        if self.scored_candidates:
            # Pick top scored candidate
            top_option = self.scored_candidates[0]
            self.dashboard_state["auto_selected_option"] = {
                "tradingsymbol": top_option.tradingsymbol,
                "instrument_token": top_option.instrument_token,
                "strike": top_option.strike,
                "option_type": top_option.option_type,
                "expiry": top_option.expiry,
                "ltp": top_option.ltp,
                "delta": top_option.delta,
                "confidence": getattr(top_option, "score", None),
            }
        else:
            self.dashboard_state["auto_selected_option"] = None
        # Diagnostic error messages
        dashboard_errors = []
        kite = KiteClient.get_instance()

        # 1. Get NIFTY spot — use NIFTY_INSTRUMENT constant for Kite LTP, fallback to yfinance
        spot = 0.0
        try:
            ltp_data = kite.get_ltp([NIFTY_INSTRUMENT])
            logger.info(f"Kite LTP response for NIFTY_INSTRUMENT {NIFTY_INSTRUMENT}: {ltp_data}")
            spot = ltp_data.get(NIFTY_INSTRUMENT, {}).get("last_price", 0)
        except Exception as e:
            logger.info(f"Offline: Kite LTP unavailable ({type(e).__name__}), trying yfinance…")

        if spot <= 0:
            spot = fetch_nifty_spot_yf()
            if spot > 0:
                logger.info(f"Offline: NIFTY spot from yfinance = {spot:.2f}")

        if spot > 0:
            self.nifty_spot = spot
        else:
            dashboard_errors.append("NIFTY spot not available. Check API session or market status.")

        # 2. Fetch historical data — try Kite, fallback to yfinance
        df = None
        try:
            df = self._fetch_spot_data(kite)
        except Exception:
            pass

        if df is None or df.empty:
            df = fetch_nifty_history_yf(days=5, interval="5m")
            if df is not None and not df.empty:
                logger.info(f"Offline: using yfinance history ({len(df)} candles)")
            else:
                dashboard_errors.append("Historical spot data not available.")

        # 3. Compute ORB + indicators from historical data
        st_dir = 1
        macd_hist = 0.0
        macd_hist_prev = 0.0
        atr_val = 0.0

        if df is not None and not df.empty:
            if not self._orb_captured:
                orb_h, orb_l = self.strategy.compute_orb(df)
                if orb_h is not None:
                    self.orb_high = orb_h
                    self.orb_low = orb_l
                    self._orb_captured = True
                    self.dashboard_state["orb"] = {
                        "high": orb_h, "low": orb_l, "captured": True,
                    }

            # Compute indicators
            df = self.strategy.compute_supertrend(df)
            df = self.strategy.compute_macd(df)

            if "st_dir" in df.columns:
                st_dir = int(df.iloc[-1]["st_dir"])
            if "macd_hist" in df.columns:
                macd_hist = float(df.iloc[-1]["macd_hist"])
                if len(df) > 1:
                    macd_hist_prev = float(df.iloc[-2]["macd_hist"])
            try:
                atr_val = float(self.strategy.calculate_atr(df).iloc[-1])
            except Exception:
                atr_val = 0.0

        signal_dir = st_dir

        # 4. Build option candidates — try Kite quotes, fallback to B-S estimates
        if self.nifty_spot > 0 and kite.is_connected:
            try:
                if df is not None and not df.empty:
                    self.all_candidates = self._build_candidates(
                        kite, self.nifty_spot, signal_dir, df
                    )
            except Exception as e:
                logger.info(f"Offline: Kite candidate build failed ({type(e).__name__}): {e}")
                dashboard_errors.append(f"Candidate build failed: {type(e).__name__}: {e}")
                self.all_candidates = []

            # If Kite quotes failed (PermissionException), use B-S fallback
            if not self.all_candidates:
                self.all_candidates = build_option_candidates_from_instruments(
                    kite_client=kite,
                    spot=self.nifty_spot,
                    signal_dir=signal_dir,
                    orb_high=self.orb_high or 0,
                    orb_low=self.orb_low or 0,
                    st_dir=st_dir,
                    macd_hist=macd_hist,
                    macd_hist_prev=macd_hist_prev,
                    atr_val=atr_val,
                )
                if not self.all_candidates:
                    dashboard_errors.append("No option candidates found. Check instrument master and spot data.")

            if self.all_candidates:
                self.scored_candidates = self.scorer.rank_candidates(
                    self.all_candidates, top_pct=10.0,
                    levels=self._cached_levels,
                )

        # Refresh LTPs with live data to ensure entry/SL/targets are accurate
        if self._oi_feed and self._oi_feed.is_running and self.scored_candidates:
            self._update_candidates_from_live_oi()
        elif kite.is_connected and self.scored_candidates:
            self._refresh_candidate_ltps(kite)

        self.dashboard_state["engine_status"] = "MARKET_CLOSED"
        self.dashboard_state["nifty_spot"] = self.nifty_spot
        self.dashboard_state["top_candidates"] = (
            self.scorer.score_summary(self.scored_candidates)
            if self.scored_candidates
            else {"count": 0, "top": []}
        )
        self.dashboard_state["all_candidates_count"] = len(self.all_candidates)
        self.dashboard_state["risk"] = self.risk_mgr.get_status()
        self.dashboard_state["trades"] = self.risk_mgr.get_trades_summary()
        # Ensure _cycle_count is initialized
        if not hasattr(self, "_cycle_count"):
            self._cycle_count = 0
        self.dashboard_state["cycle_count"] = self._cycle_count
        # OI analysis — use live WebSocket data if available
        if self._oi_feed and self._oi_feed.is_running:
            self.dashboard_state["oi_analysis"] = self._oi_feed.get_dashboard_data()
        else:
            # Ensure _cached_oi_data is initialized
            if not hasattr(self, "_cached_oi_data") or self._cached_oi_data is None:
                self._cached_oi_data = {}
            self.dashboard_state["oi_analysis"] = {
                "source": "rest_api",
                "connected": False,
                "ce_oi_change_pct": self._cached_oi_data.get("ce_oi_change_pct", 0),
                "pe_oi_change_pct": self._cached_oi_data.get("pe_oi_change_pct", 0),
                "pcr": self._cached_oi_data.get("pcr", 0),
            }
            if self._cached_oi_data.get("ce_oi_change_pct", 0) == 0 and self._cached_oi_data.get("pe_oi_change_pct", 0) == 0:
                dashboard_errors.append("OI analysis not available. Check option chain and API session.")
        self.dashboard_state["last_update"] = datetime.now().isoformat()
        self.dashboard_state["errors"] = dashboard_errors
        return self.dashboard_state

    # ------------------------------------------------------------------
    # Monitor open positions (SL / target / trailing)
    # ------------------------------------------------------------------
    def monitor_positions(self, kite: KiteClient):
        """
        Monitor open positions using the ExitSignalGenerator + TrailingSLManager.

        Every cycle:
        1. Update watermark (highest premium) and compute trailing SL
        2. If TSL improved → update SL order on broker
        3. Check exit conditions (SL hit, targets, supertrend, MACD, VWAP, time)
        4. Execute exits if triggered

        The trailing SL manager ensures winning trades are protected and
        the trade stays alive as long as the trend continues.
        """
        if not self.order_mgr.positions:
            return

        df = self._cached_df
        vwap = self._cached_levels.vwap if self._cached_levels else 0.0
        spot = self.nifty_spot

        # Compute spot ATR for option ATR estimation
        spot_atr = 0.0
        try:
            if df is not None and len(df) > 14:
                spot_atr = float(self.strategy.calculate_atr(df).iloc[-1])
        except Exception:
            pass

        for trade_id, pos in list(self.order_mgr.positions.items()):
            if pos.status == "CLOSED" or pos.remaining_qty <= 0:
                continue

            try:
                # Get current LTP
                data = kite.get_ltp([f"NFO:{pos.symbol}"])
                ltp = data.get(f"NFO:{pos.symbol}", {}).get("last_price", 0)
                if ltp <= 0:
                    continue

                # --- Step 1: Run Trailing SL Manager ---
                option_atr = TrailingSLManager.estimate_option_atr(
                    spot_atr=spot_atr,
                    delta=pos.delta,
                    entry_price=pos.entry_price,
                )

                t1 = pos.smart_levels.option_t1 if pos.smart_levels else ltp * 1.3
                t2 = pos.smart_levels.option_t2 if pos.smart_levels else ltp * 1.6
                t3 = pos.smart_levels.option_t3 if pos.smart_levels else ltp * 2.0

                new_tsl, new_phase, new_highest, tsl_updated = self.tsl_mgr.compute_trailing_sl(
                    entry_price=pos.entry_price,
                    current_ltp=ltp,
                    current_tsl=pos.trailing_sl,
                    highest_price=pos.highest_price,
                    trailing_phase=pos.trailing_phase,
                    t1_hit=pos.t1_hit,
                    t2_hit=pos.t2_hit,
                    t3_hit=pos.t3_hit,
                    target1=t1,
                    target2=t2,
                    target3=t3,
                    option_atr=option_atr,
                    df=df,
                    spot_atr=spot_atr,
                    delta=pos.delta,
                )

                # Update position state
                pos.highest_price = new_highest
                pos.trailing_phase = new_phase

                if tsl_updated:
                    old_tsl = pos.trailing_sl
                    pos.trailing_sl = new_tsl
                    pos.last_tsl_update = datetime.now().isoformat()

                    # Update SL order on broker
                    if self.auto_trade_mode in ("paper", "live"):
                        self._update_broker_sl(kite, pos, new_tsl)

                    # Telegram alert for significant TSL moves
                    gain_pct = ((ltp - pos.entry_price) / pos.entry_price * 100) if pos.entry_price > 0 else 0
                    if new_phase != "INITIAL":
                        telegram.send_custom(
                            f"🛡️ <b>TSL UPDATE: {pos.symbol}</b>\n"
                            f"SL: ₹{old_tsl:.2f} → <b>₹{new_tsl:.2f}</b>\n"
                            f"Phase: {new_phase}\n"
                            f"LTP: ₹{ltp:.2f} | Gain: {gain_pct:+.1f}%\n"
                            f"Watermark: ₹{new_highest:.2f}\n"
                            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                        )

                # --- Step 2: Build position context with ACTUAL trailing SL ---
                ctx = PositionContext(
                    trade_id=trade_id,
                    symbol=pos.symbol,
                    option_type=pos.option_type,
                    entry_price=pos.entry_price,
                    current_ltp=ltp,
                    quantity=pos.quantity,
                    remaining_qty=pos.remaining_qty,
                    stop_loss=pos.smart_levels.option_sl if pos.smart_levels else ltp * 0.70,
                    trailing_sl=pos.trailing_sl,  # Now uses ACTUAL tracked TSL
                    target1=t1,
                    target2=t2,
                    target3=t3,
                    t1_hit=pos.t1_hit,
                    t2_hit=pos.t2_hit,
                    t3_hit=pos.t3_hit,
                    entry_time=pos.created_at,
                )

                # --- Step 3: Check VWAP cross on spot ---
                vwap_applicable = False
                if vwap > 0 and spot > 0:
                    if pos.option_type == "CE" and spot < vwap:
                        vwap_applicable = True
                    elif pos.option_type == "PE" and spot > vwap:
                        vwap_applicable = True

                # --- Step 4: Evaluate exit conditions ---
                exit_sig = self.exit_gen.evaluate(
                    pos=ctx,
                    df=df,
                    vwap=vwap if vwap_applicable else 0.0,
                )

                if exit_sig:
                    logger.info(
                        f"EXIT SIGNAL DETECTED for {pos.symbol}: {exit_sig.reason.value} "
                        f"({exit_sig.priority.value}) exit_pct={exit_sig.exit_pct} "
                        f"tsl_phase={pos.trailing_phase}"
                    )

                    # If exit signal has a new TSL, apply it (only if higher)
                    if exit_sig.new_trailing_sl and exit_sig.new_trailing_sl > pos.trailing_sl:
                        pos.trailing_sl = exit_sig.new_trailing_sl
                        pos.last_tsl_update = datetime.now().isoformat()
                        if self.auto_trade_mode in ("paper", "live"):
                            self._update_broker_sl(kite, pos, exit_sig.new_trailing_sl)

                    # Send Telegram alert for exit signal (not for TSL-only tightenings)
                    if exit_sig.exit_pct > 0:
                        logger.info(
                            f"Sending Telegram EXIT alert for {pos.symbol} | Reason: {exit_sig.reason.value} | Trade ID: {trade_id}"
                        )
                        telegram.send_custom(
                            f"🚪 <b>EXIT SIGNAL: {exit_sig.reason.value}</b>\n"
                            f"Symbol: {pos.symbol}\n"
                            f"Priority: {exit_sig.priority.value}\n"
                            f"Exit %: {exit_sig.exit_pct * 100:.0f}%\n"
                            f"LTP: ₹{ltp:.2f} | TSL: ₹{pos.trailing_sl:.2f}\n"
                            f"Phase: {pos.trailing_phase}\n"
                            f"{exit_sig.message}\n"
                            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                        )

                    # --- Step 5: Auto-execute the exit if auto-trading is on ---
                    if self.auto_trade_mode in ("paper", "live") and exit_sig.exit_pct > 0:
                        self.order_mgr.handle_exit_signal(kite, trade_id, exit_sig)

                        # Record in journal if position is now closed
                        updated_pos = self.order_mgr.positions.get(trade_id)
                        if updated_pos and updated_pos.status == "CLOSED":
                            pnl = (ltp - pos.entry_price) * pos.quantity
                            self.journal.close_candidate(
                                trade_id=trade_id,
                                exit_price=ltp,
                                pnl=pnl,
                                exit_reason=exit_sig.reason.value,
                            )
                            # Record SL hit for re-entry cooldown
                            if exit_sig.reason.value in ("SL_HIT", "TRAILING_SL_HIT"):
                                sl_direction = "BUY" if pos.option_type == "CE" else "SELL"
                                self.record_sl_hit(sl_direction)

            except Exception as e:
                logger.error(f"Monitor error for {pos.symbol}: {e}")

    def _update_broker_sl(
        self, kite: KiteClient, pos: ManagedPosition, new_tsl: float
    ):
        """Cancel old SL order and place new one at updated trailing SL."""
        try:
            if pos.sl_order_id:
                kite.cancel_order(pos.sl_order_id)
                logger.info(f"Cancelled old SL order {pos.sl_order_id} for {pos.symbol}")

            new_sl_result = kite.place_sl_order(
                tradingsymbol=pos.symbol,
                exchange="NFO",
                transaction_type="SELL",
                quantity=pos.remaining_qty,
                trigger_price=round(new_tsl, 2),
                product="MIS",
                tag=f"AutoTSL_{pos.trailing_phase}",
            )
            if new_sl_result.get("success"):
                pos.sl_order_id = new_sl_result.get("order_id", "")
                logger.info(
                    f"Broker SL updated: {pos.symbol} TSL=₹{new_tsl:.2f} "
                    f"phase={pos.trailing_phase} qty={pos.remaining_qty}"
                )
            else:
                logger.warning(f"Broker SL update failed for {pos.symbol}: {new_sl_result}")
        except Exception as e:
            logger.error(f"Error updating broker SL for {pos.symbol}: {e}")

    # ------------------------------------------------------------------
    # WebSocket Tick Integration
    # ------------------------------------------------------------------
    def start_websocket_feed(self, kite: KiteClient):
        """
        Start WebSocket tick feed for real-time position monitoring.
        Subscribes to NIFTY spot + all active position instruments.
        Only starts if settings.use_websocket_ticks is True.
        """
        if not settings.use_websocket_ticks:
            return

        tokens = set()
        # NIFTY spot token
        nifty_token = self._resolve_nifty_token(kite)
        if nifty_token:
            tokens.add(nifty_token)

        # Active position tokens
        for pos in self.order_mgr.positions.values():
            if pos.status != "CLOSED" and hasattr(pos, "instrument_token"):
                tokens.add(pos.instrument_token)

        if not tokens:
            return

        def on_tick(ticks):
            """Handle incoming ticks — update LTPs for positions."""
            for tick in ticks:
                token = tick.get("instrument_token")
                if token:
                    self._tick_data[token] = {
                        "ltp": tick.get("last_price", 0),
                        "bid": tick.get("depth", {}).get("buy", [{}])[0].get("price", 0),
                        "ask": tick.get("depth", {}).get("sell", [{}])[0].get("price", 0),
                        "volume": tick.get("volume_traded", 0),
                        "oi": tick.get("oi", 0),
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Update NIFTY spot if it's the spot token
                    if token == nifty_token:
                        self.nifty_spot = tick.get("last_price", self.nifty_spot)

        def on_connect(ws, response):
            self._ws_connected = True
            logger.info(f"WebSocket connected, subscribed to {len(tokens)} instruments")

        try:
            kite.start_ticker(list(tokens), on_tick=on_tick, on_connect=on_connect)
            logger.info("WebSocket tick feed started")
        except Exception as e:
            logger.error(f"WebSocket start failed: {e}")

    def get_tick_ltp(self, instrument_token: int) -> Optional[float]:
        """Get latest tick LTP for an instrument. Returns None if no tick data."""
        tick = self._tick_data.get(instrument_token)
        if tick:
            return tick.get("ltp")
        return None

    # ------------------------------------------------------------------
    # Main cycle – called every 5 minutes by scheduler
    # ------------------------------------------------------------------
    def run_cycle(self) -> Dict:
        """
        Main engine cycle. Returns dashboard state dict.
        Enhanced: Multi-strategy, time-based and confluence-based switching.
        """
        self.init_day()
        self._cycle_count += 1
        kite = KiteClient.get_instance()

        if not kite.is_connected:
            self.dashboard_state["engine_status"] = "DISCONNECTED"
            self.dashboard_state["last_update"] = datetime.now().isoformat()
            return self.dashboard_state

        # -- Auto square-off check --
        if self.risk_mgr.should_square_off():
            if self.risk_mgr.open_positions:
                logger.info("3:15 PM – auto square-off triggered")
                self.square_off_all(kite, "SQUARED_OFF")
            return self.fetch_offline_data()

        # -- Daily loss limit check --
        if self.risk_mgr.is_halted:
            self.dashboard_state["engine_status"] = "HALTED_LOSS_LIMIT"
            telegram.alert_loss_limit(
                self.risk_mgr.daily_pnl,
                self.risk_mgr.daily_loss_limit,
                self.risk_mgr.capital,
            )
            self._update_dashboard()
            return self.dashboard_state

        # -- Fetch spot data --
        df = self._fetch_spot_data(kite)
        if df is None or df.empty:
            self.dashboard_state["engine_status"] = "NO_DATA"
            self._update_dashboard()
            return self.dashboard_state

        # -- Capture ORB --
        if not self._orb_captured:
            orb_h, orb_l = self.strategy.compute_orb(df)
            if orb_h is not None:
                self.orb_high = orb_h
                self.orb_low = orb_l
                self._orb_captured = True
                self.dashboard_state["orb"] = {
                    "high": orb_h,
                    "low": orb_l,
                    "captured": True,
                }
                logger.info(f"ORB captured: [{orb_l:.2f} – {orb_h:.2f}]")
                telegram.alert_orb_captured(orb_h, orb_l, df["close"].iloc[-1])
            else:
                self.dashboard_state["engine_status"] = "WAITING_ORB"
                self._update_dashboard()
                return self.dashboard_state

        self.nifty_spot = df["close"].iloc[-1]
        self._start_oi_feed(kite)

        vix = self._fetch_india_vix(kite)
        if vix > settings.vix_max_threshold and vix > 0:
            logger.warning(f"India VIX {vix:.1f} > max threshold {settings.vix_max_threshold}. Skipping new trades.")
            self.dashboard_state["engine_status"] = "VIX_TOO_HIGH"
            self.dashboard_state["india_vix"] = vix
            try:
                self.monitor_positions(kite)
            except Exception as e:
                logger.warning(f"Position monitoring failed ({type(e).__name__}): {e}")
            self._update_dashboard()
            return self.dashboard_state

        # -- Compute indicators on spot data for candidate building --
        df = self.strategy.compute_supertrend(df)
        df = self.strategy.compute_macd(df)
        if self._gap_type == "NONE" and self._cycle_count <= 2:
            self._compute_gap_classification(df)
        if self._cycle_count <= 1 or self._cycle_count % 3 == 0:
            try:
                self._fetch_multi_tf_data(kite)
            except Exception as e:
                logger.debug(f"Multi-TF fetch error: {e}")
        self._cached_levels = self.level_calc.compute(
            df, orb_high=self.orb_high or 0, orb_low=self.orb_low or 0
        )
        self._cached_df = df
        st_dir = int(df.iloc[-1].get("st_dir", 1)) if "st_dir" in df.columns else 1
        macd_hist = float(df.iloc[-1].get("macd_hist", 0)) if "macd_hist" in df.columns else 0.0
        macd_hist_prev = float(df.iloc[-2].get("macd_hist", 0)) if len(df) > 1 and "macd_hist" in df.columns else 0.0
        try:
            atr_val = float(self.strategy.calculate_atr(df).iloc[-1])
        except Exception:
            atr_val = 0.0

        # --- Multi-strategy switching logic ---
        now = datetime.now().time()
        orb_end = (datetime.combine(datetime.today(), time(9, 15)) + timedelta(minutes=settings.orb_minutes)).time()
        supertrend_strategy = None
        try:
            from strategies.supertrend_strategy import SupertrendStrategy
            supertrend_strategy = SupertrendStrategy()
        except Exception:
            pass

        # --- Strategy selection ---
        # ORB strategy runs ALL DAY (9:30–15:00) — it handles its own time guards.
        # Other strategies run alongside after the ORB window for confluence.
        signal = None
        all_signals = []

        # 1. ORB strategy — always active after 9:30 (strategy internally
        #    rejects signals before ORB end and after 15:00)
        try:
            orb_signal = self.strategy.generate_signal(
                df, "NIFTY", oi_data=self._cached_oi_data or None
            )
            if orb_signal:
                all_signals.append(orb_signal)
        except Exception as e:
            logger.warning(f"ORB signal error: {e}")

        # 2. After ORB window, also try auxiliary strategies for confluence
        if now > orb_end:
            # Build option chain for premium-selling strategies
            option_chain = None
            spot = self.nifty_spot
            if hasattr(self, "all_candidates") and self.all_candidates:
                option_chain = [
                    {
                        "tradingsymbol": c.tradingsymbol,
                        "strike": c.strike,
                        "option_type": c.option_type,
                        "ltp": c.ltp,
                        "oi": getattr(c, "oi", 0),
                        "lot_size": getattr(c, "quantity", 50),
                    }
                    for c in self.all_candidates
                ]

            # --- Directional strategies ---
            if settings.enable_vwap_strategy:
                try:
                    s = self.vwap_strategy.generate_signal(
                        df, "NIFTY", levels=self._cached_levels,
                        orb_high=self.orb_high or 0, orb_low=self.orb_low or 0,
                        india_vix=self._india_vix, gap_type=self._gap_type)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"VWAP MR error: {e}")

            if settings.enable_vwap_breakout:
                try:
                    s = self.vwap_breakout_strategy.generate_signal(
                        df, "NIFTY", orb_high=self.orb_high or 0, orb_low=self.orb_low or 0)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"VWAP Breakout error: {e}")

            if settings.enable_ema_crossover:
                try:
                    s = self.ema_crossover_strategy.generate_signal(df, "NIFTY")
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"EMA Crossover error: {e}")

            if settings.enable_rsi_divergence:
                try:
                    s = self.rsi_divergence_strategy.generate_signal(df, "NIFTY")
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"RSI Divergence error: {e}")

            if settings.enable_pcr_oi_strategy:
                try:
                    oi_data = self._get_live_oi_data() or self._cached_oi_data
                    s = self.pcr_oi_strategy.generate_signal(df, "NIFTY", oi_analysis=oi_data)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"PCR/OI error: {e}")

            if settings.enable_gap_fill:
                try:
                    pdc = self._cached_levels.pdc if self._cached_levels else 0
                    s = self.gap_fill_strategy.generate_signal(
                        df, "NIFTY", gap_type=self._gap_type,
                        gap_pct=self._gap_pct, prev_close=pdc)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"Gap Fill error: {e}")

            if supertrend_strategy:
                try:
                    s = supertrend_strategy.generate_signal(df, "NIFTY")
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"Supertrend error: {e}")

            # --- Range-bound / sideways strategies ---
            if settings.enable_bollinger_mr:
                try:
                    s = self.bollinger_mr_strategy.generate_signal(df, "NIFTY")
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"Bollinger MR error: {e}")

            if settings.enable_orb_scalper:
                try:
                    s = self.orb_scalper_strategy.generate_signal(
                        df, "NIFTY",
                        orb_high=self.orb_high or 0,
                        orb_low=self.orb_low or 0)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"ORB Range Scalper error: {e}")

            # --- Premium selling strategies ---
            if settings.enable_expiry_sell_strategy:
                try:
                    s = self.expiry_strategy.generate_signal(
                        df, "NIFTY", india_vix=self._india_vix,
                        orb_high=self.orb_high or 0, orb_low=self.orb_low or 0,
                        gap_type=self._gap_type)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"Expiry Sell error: {e}")

            if settings.enable_straddle_strangle:
                try:
                    s = self.straddle_strategy.generate_signal(
                        df, "NIFTY", india_vix=self._india_vix,
                        orb_high=self.orb_high or 0, orb_low=self.orb_low or 0,
                        option_chain=option_chain, spot=spot)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"Straddle/Strangle error: {e}")

            if settings.enable_iron_condor:
                try:
                    s = self.iron_condor_strategy.generate_signal(
                        df, "NIFTY", india_vix=self._india_vix,
                        orb_high=self.orb_high or 0, orb_low=self.orb_low or 0,
                        option_chain=option_chain, spot=spot)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"Iron Condor error: {e}")

            # --- Hero Zero Expiry ---
            if self.hero_zero_strategy and settings.enable_hero_zero:
                try:
                    s = self.hero_zero_strategy.generate_signal(
                        df, "NIFTY", option_chain=option_chain, spot=spot)
                    if s: all_signals.append(s)
                except Exception as e:
                    logger.debug(f"Hero Zero error: {e}")

        # --- Market Regime Detection + Signal Selection ---
        # Step 1: Detect if market is TRENDING, SIDEWAYS, VOLATILE, or BREAKOUT
        regime = detect_regime(
            df,
            orb_high=self.orb_high or 0,
            orb_low=self.orb_low or 0,
            india_vix=self._india_vix,
        )
        logger.info(
            f"Market Regime: {regime.regime.value} (conf={regime.confidence:.0f}%) | "
            f"ADX={regime.adx} BBW={regime.bbw_pct}% ORB_contained={regime.orb_contained} "
            f"ST_flips={regime.st_flips} | {regime.reasoning}"
        )
        self.dashboard_state["market_regime"] = {
            "regime": regime.regime.value,
            "confidence": regime.confidence,
            "adx": regime.adx,
            "bbw_pct": regime.bbw_pct,
            "orb_contained": regime.orb_contained,
            "st_flips": regime.st_flips,
            "candle_ratio": regime.avg_candle_ratio,
            "reasoning": regime.reasoning,
        }

        # Step 2: Filter signals — block strategies that don't fit the regime
        # e.g., block mean-reversion in a trending market, block breakout in sideways
        filtered_signals = filter_signals_by_regime(all_signals, regime)

        # Step 3: Pick best signal from filtered list
        # Confluence: if 2+ strategies agree on direction → higher conviction
        buy_signals = [s for s in filtered_signals if getattr(s, "signal", None) == SignalType.BUY
                       or getattr(s, "signal", None) == "BUY"]
        sell_signals = [s for s in filtered_signals if getattr(s, "signal", None) == SignalType.SELL
                        or getattr(s, "signal", None) == "SELL"]
        if len(buy_signals) >= 2:
            # Multiple strategies agree on BUY — pick highest confidence
            signal = max(buy_signals, key=lambda s: getattr(s, "confidence", 0))
            signal.confidence = min(95, signal.confidence + 5)  # Confluence bonus
        elif len(sell_signals) >= 2:
            signal = max(sell_signals, key=lambda s: getattr(s, "confidence", 0))
            signal.confidence = min(95, signal.confidence + 5)
        elif filtered_signals:
            signal = max(filtered_signals, key=lambda s: getattr(s, "confidence", 0))

        # Log all signals considered (before + after regime filter)
        self.dashboard_state["all_signals"] = [
            {
                "type": s.signal.value,
                "entry": s.entry_price,
                "sl": s.stop_loss,
                "target": s.target,
                "confidence": s.confidence,
                "conditions": s.conditions_met,
                "reasoning": s.reasoning,
                "strategy": getattr(s, "strategy_name", ""),
                "time": s.timestamp,
            } for s in all_signals
        ]
        self.dashboard_state["filtered_signals_count"] = len(filtered_signals)
        self.dashboard_state["blocked_signals_count"] = len(all_signals) - len(filtered_signals)


        self.last_signal = signal

        # -- Monitor existing positions --
        try:
            self.monitor_positions(kite)
        except Exception as e:
            logger.warning(f"Position monitoring failed ({type(e).__name__}): {e}")

        # Determine direction: from signal if available, else from Supertrend
        if signal is not None:
            signal_dir = 1 if signal.signal == SignalType.BUY else -1
        else:
            signal_dir = st_dir  # use Supertrend direction as fallback

        # -- Always build + score candidates after ORB capture --
        try:
            self.all_candidates = self._build_candidates(kite, self.nifty_spot, signal_dir, df)
        except Exception as e:
            logger.warning(f"Kite candidate build failed ({type(e).__name__}): {e}")
            self.all_candidates = []

        # If Kite quotes failed, use B-S/yfinance fallback
        if not self.all_candidates:
            logger.info("Using fallback candidate builder (B-S estimates)")
            self.all_candidates = build_option_candidates_from_instruments(
                kite_client=kite,
                spot=self.nifty_spot,
                signal_dir=signal_dir,
                orb_high=self.orb_high or 0,
                orb_low=self.orb_low or 0,
                st_dir=st_dir,
                macd_hist=macd_hist,
                macd_hist_prev=macd_hist_prev,
                atr_val=atr_val,
            )

        self.scored_candidates = self.scorer.rank_candidates(
            self.all_candidates, top_pct=20.0, levels=self._cached_levels
        )

        # Update candidates with live OI/LTP from WebSocket (if running)
        self._update_candidates_from_live_oi()

        # -- If we have a signal, trade and alert --
        if signal is not None:
            self.dashboard_state["engine_status"] = "SIGNAL_ACTIVE"
            self.dashboard_state["signal"] = {
                "type": signal.signal.value,
                "entry": signal.entry_price,
                "sl": signal.stop_loss,
                "target": signal.target,
                "confidence": signal.confidence,
                "conditions": signal.conditions_met,
                "reasoning": signal.reasoning,
            }

            # Add levels info to dashboard
            if self._cached_levels:
                lvls = self._cached_levels
                self.dashboard_state["levels"] = {
                    "vwap": lvls.vwap,
                    "vwap_upper_1": lvls.vwap_upper_1,
                    "vwap_lower_1": lvls.vwap_lower_1,
                    "pivot": lvls.pivot,
                    "r1": lvls.r1, "r2": lvls.r2, "r3": lvls.r3,
                    "s1": lvls.s1, "s2": lvls.s2, "s3": lvls.s3,
                    "pdh": lvls.pdh, "pdl": lvls.pdl,
                    "orb_high": lvls.orb_high, "orb_low": lvls.orb_low,
                    "atr": lvls.atr,
                }

            # Only alert once per unique signal direction + entry (rounded to nearest 25 pts)
            # This prevents near-identical signals (e.g., SELL@23717 vs SELL@23718) from
            # spamming Telegram as separate "new" signals.
            sig_key = f"{signal.signal.value}@{round(signal.entry_price / 25) * 25}"
            if sig_key != self._alerted_signal_key:
                logger.info(f"New signal detected: {sig_key} | Entry: {signal.entry_price} | SL: {signal.stop_loss} | Target: {signal.target} | Confidence: {signal.confidence}")
                self._alerted_signal_key = sig_key
                # Include top candidate info if available
                top_candidate = self.scored_candidates[0] if self.scored_candidates else None
                option_strike = top_candidate.strike if top_candidate else None
                option_type = top_candidate.option_type if top_candidate else None
                option_expiry = top_candidate.expiry if top_candidate else None
                option_symbol = top_candidate.tradingsymbol if top_candidate else None
                telegram.alert_signal(
                    signal_type=signal.signal.value,
                    entry=signal.entry_price,
                    sl=signal.stop_loss,
                    target=signal.target,
                    confidence=signal.confidence,
                    conditions=signal.conditions_met,
                    reasoning=signal.reasoning,
                    option_strike=option_strike,
                    option_type=option_type,
                    option_expiry=option_expiry,
                    option_symbol=option_symbol,
                    strategy_name=getattr(signal, "strategy_name", ""),
                )

            # Record signal & candidates in journal
            sig_id = self.journal.record_signal(
                signal_type=signal.signal.value,
                spot_price=self.nifty_spot,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                target=signal.target,
                confidence=signal.confidence,
                conditions=signal.conditions_met,
                reasoning=signal.reasoning,
                source="ORB_BREAKOUT",
            )
            if self.scored_candidates:
                self.journal.record_candidates(sig_id, self.scored_candidates)

            # Execute on top candidate(s) via auto order manager
            # --- CRITICAL: Market-hour and execution control gate ---
            skip_trade, skip_reason = should_skip_new_trades()
            if skip_trade:
                logger.warning(f"New trades blocked: {skip_reason}")
                self.dashboard_state.setdefault("skipped_signals", []).append({
                    "type": signal.signal.value,
                    "entry": signal.entry_price,
                    "reason": skip_reason,
                    "confidence": signal.confidence,
                    "time": signal.timestamp,
                })
                telegram.send_custom(f"🚫 New trades blocked: {skip_reason}")
            elif self.auto_trade_mode in ("paper", "live"):
                skip_reasons = []
                # --- Safety filter: minimum confidence threshold ---
                if signal.confidence < settings.min_confidence_threshold:
                    skip_reasons.append(f"confidence {signal.confidence} < min threshold {settings.min_confidence_threshold}")
                    logger.info(f"Trade skipped: {skip_reasons[-1]}")
                    telegram.send_message(
                        f"⚠️ Signal skipped (low confidence {signal.confidence:.0f}% < {settings.min_confidence_threshold:.0f}%)"
                    )
                # --- Safety filter: re-entry cooldown after SL hit ---
                elif self._check_reentry_cooldown("BUY" if signal.signal == SignalType.BUY else "SELL"):
                    skip_reasons.append(f"re-entry cooldown active for {signal.signal.value}")
                    logger.info(f"Trade skipped: {skip_reasons[-1]}")
                    telegram.send_message(
                        f"⏳ Signal skipped (re-entry cooldown after SL hit, wait {settings.reentry_cooldown_minutes} min)"
                    )
                # --- Add more risk filters here as needed ---
                if skip_reasons:
                    # Track skipped/blocked signals for analytics
                    self.dashboard_state.setdefault("skipped_signals", []).append({
                        "type": signal.signal.value,
                        "entry": signal.entry_price,
                        "reason": ", ".join(skip_reasons),
                        "confidence": signal.confidence,
                        "time": signal.timestamp,
                    })
                else:
                    # --- VIX-based position size reduction ---
                    vix_size_factor = 1.0
                    if self._india_vix > settings.vix_reduce_size_threshold:
                        logger.info(f"Trade size reduced: India VIX {self._india_vix} > threshold {settings.vix_reduce_size_threshold}")
                        vix_size_factor = 0.5
                        logger.info(
                            f"India VIX {self._india_vix:.1f} > {settings.vix_reduce_size_threshold}. "
                            f"Reducing position size by 50%."
                        )

                    for cand in self.scored_candidates:
                        if len(self.order_mgr.positions) >= self.risk_mgr.max_concurrent:
                            logger.info(f"Trade skipped: max concurrent positions reached ({self.risk_mgr.max_concurrent})")
                            break
                        # Skip strikes already attempted this session
                        if cand.tradingsymbol in self._traded_strikes:
                            logger.info(f"Trade skipped: {cand.tradingsymbol} already attempted this session")
                            continue
                        # Only execute if position is not already open for this symbol
                        existing = [p for p in self.order_mgr.positions.values()
                                    if p.symbol == cand.tradingsymbol and p.status != "CLOSED"]
                        if not existing:
                            logger.info(f"Placing trade for {cand.tradingsymbol} | Entry: {cand.entry_price} | SL: {cand.stoploss} | Qty: {cand.quantity if hasattr(cand, 'quantity') else 'N/A'}")
                            self._traded_strikes.add(cand.tradingsymbol)
                            pos = self._execute_trade(kite, cand, qty_factor=vix_size_factor)
                            if pos:
                                self.journal.activate_candidate(
                                    symbol=cand.tradingsymbol,
                                    trade_id=pos.trade_id,
                                    entry_price=pos.entry_price,
                                    quantity=pos.quantity,
                                )

        else:
            self.dashboard_state["engine_status"] = "MONITORING"
            self.dashboard_state["signal"] = None

        # -- Send top candidates via Telegram ONLY when signal is active --
        # Previously this fired every cycle even before ORB capture,
        # flooding Telegram with premature candidate alerts.
        if signal is not None and self.scored_candidates and self._orb_captured:
            # Only alert candidates with strikes not already sent
            new_candidates = [
                c for c in self.scored_candidates[:5]
                if c.tradingsymbol not in self._alerted_strikes
            ]
            if new_candidates:
                top_summary = [
                    {
                        "symbol": c.tradingsymbol,
                        "display_name": c.display_name,
                        "type": c.option_type,
                        "strike": c.strike,
                        "expiry": c.expiry,
                        "delta": c.delta,
                        "iv": c.iv,
                        "score": c.score,
                        "ltp": c.ltp,
                        "entry": c.entry_price,
                        "stoploss": c.stoploss,
                        "target1": c.target1,
                        "target2": c.target2,
                        "target3": c.target3,
                        "risk_reward_pct": c.risk_reward_pct,
                        "oi_change_pct": c.oi_change_pct,
                        "ce_oi_change_pct": c.ce_oi_change_pct,
                        "pe_oi_change_pct": c.pe_oi_change_pct,
                        "oi_interpretation": c.oi_interpretation,
                    }
                    for c in new_candidates
                ]
                telegram.alert_top_candidates(top_summary)
                for c in new_candidates:
                    self._alerted_strikes.add(c.tradingsymbol)

        self._update_dashboard()
        return self.dashboard_state

    # ------------------------------------------------------------------
    # Dashboard state
    # ------------------------------------------------------------------
    def _update_dashboard(self):
        # Refresh candidates from live OI feed first (preferred), then fallback to get_ltp
        try:
            if self._oi_feed and self._oi_feed.is_running and self.scored_candidates:
                self._update_candidates_from_live_oi()
            else:
                kite = KiteClient.get_instance()
                if kite.is_connected and self.scored_candidates:
                    self._refresh_candidate_ltps(kite)
        except Exception:
            pass

        self.dashboard_state["risk"] = self.risk_mgr.get_status()
        self.dashboard_state["trades"] = self.risk_mgr.get_trades_summary()
        self.dashboard_state["top_candidates"] = self.scorer.score_summary(
            self.scored_candidates
        ) if self.scored_candidates else {"count": 0, "top": []}
        self.dashboard_state["nifty_spot"] = self.nifty_spot
        self.dashboard_state["cycle_count"] = self._cycle_count
        self.dashboard_state["auto_trade_mode"] = self.auto_trade_mode
        active_positions = self.order_mgr.get_active_positions()
        trades_today = self.risk_mgr.get_trades_summary()
        logger.info(f"Dashboard update: {len(active_positions)} active positions, {len(trades_today)} trades today.")
        logger.debug(f"Active positions: {active_positions}")
        logger.debug(f"Trades today: {trades_today}")
        self.dashboard_state["active_positions"] = active_positions
        self.dashboard_state["order_log"] = self.order_mgr.get_order_log()
        if self._cached_levels:
            lvls = self._cached_levels
            self.dashboard_state["levels"] = {
                "vwap": lvls.vwap,
                "vwap_upper_1": lvls.vwap_upper_1,
                "vwap_lower_1": lvls.vwap_lower_1,
                "pivot": lvls.pivot,
                "r1": lvls.r1, "r2": lvls.r2, "r3": lvls.r3,
                "s1": lvls.s1, "s2": lvls.s2, "s3": lvls.s3,
                "pdh": lvls.pdh, "pdl": lvls.pdl,
                "orb_high": lvls.orb_high, "orb_low": lvls.orb_low,
                "atr": lvls.atr,
            }
        self.dashboard_state["last_update"] = datetime.now().isoformat()

        # Safety filters info for dashboard
        self.dashboard_state["india_vix"] = self._india_vix
        self.dashboard_state["safety_filters"] = {
            "india_vix": self._india_vix,
            "min_confidence": settings.min_confidence_threshold,
            "vix_max": settings.vix_max_threshold,
            "vix_reduce": settings.vix_reduce_size_threshold,
            "cooldown_min": settings.reentry_cooldown_minutes,
            "daily_orders_used": self.order_mgr._daily_order_count,
            "daily_orders_limit": settings.auto_trade_max_orders_per_day,
            "max_orders": settings.auto_trade_max_orders_per_day,
            "gap_type": self._gap_type,
            "gap_pct": self._gap_pct,
            "st_dir_15m": self._st_dir_15m,
            "st_dir_1h": self._st_dir_1h,
            "vwap_enabled": settings.enable_vwap_strategy,
            "expiry_sell_enabled": settings.enable_expiry_sell_strategy,
            "trend_mode_enabled": settings.enable_trend_mode,
        }

        # OI analysis data for dashboard — use live WebSocket data if available
        if self._oi_feed and self._oi_feed.is_running:
            oi_dashboard = self._oi_feed.get_dashboard_data()
            self.dashboard_state["oi_analysis"] = oi_dashboard
        else:
            self.dashboard_state["oi_analysis"] = {
                "source": "rest_api",
                "connected": False,
                "ce_oi_change_pct": self._cached_oi_data.get("ce_oi_change_pct", 0) if self._cached_oi_data else 0,
                "pe_oi_change_pct": self._cached_oi_data.get("pe_oi_change_pct", 0) if self._cached_oi_data else 0,
                "pcr": self._cached_oi_data.get("pcr", 0) if self._cached_oi_data else 0,
            }

    def get_dashboard(self) -> Dict:
        return self.dashboard_state


# Need the import for signal_dir usage
from strategies.base_strategy import SignalType
