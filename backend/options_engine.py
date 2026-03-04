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
from scoring_engine import ScoringEngine, OptionCandidate
from risk_manager import RiskManager, TradeRecord
from telegram_alerts import telegram
from market_data_fallback import (
    fetch_nifty_spot_yf,
    fetch_nifty_history_yf,
    build_option_candidates_from_instruments,
)

logger = logging.getLogger(__name__)

# Constants
NIFTY_LOT_SIZE = 65
NIFTY_STRIKE_GAP = 50
NIFTY_INSTRUMENT = "NSE:NIFTY 50"


class NiftyOptionsEngine:
    """
    Main engine – call `run_cycle()` every 5 minutes after 9:30 AM.
    """

    def __init__(self):
        self.strategy = NiftyOptionsORBStrategy()
        self.scorer = ScoringEngine()
        self.risk_mgr = RiskManager(
            capital=settings.default_capital,
            daily_loss_limit_pct=2.0,
            max_risk_per_trade_pct=1.0,
            max_concurrent_positions=5,
            square_off_time=time(15, 15),
        )

        # State
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.nifty_spot: float = 0.0
        self.last_signal = None
        self.scored_candidates: List[OptionCandidate] = []
        self.all_candidates: List[OptionCandidate] = []
        self.nifty_token: Optional[int] = None
        self._orb_captured = False
        self._today: Optional[date] = None
        self._cycle_count = 0

        # Dashboard state (live)
        self.dashboard_state: Dict = {
            "engine_status": "IDLE",
            "orb": {"high": None, "low": None, "captured": False},
            "signal": None,
            "top_candidates": [],
            "risk": {},
            "trades": [],
            "last_update": None,
        }

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
        self.dashboard_state["engine_status"] = "WAITING_ORB"
        logger.info(f"NiftyOptionsEngine initialised for {today}")

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
        if tte_days <= 0 or iv <= 0:
            # ITM → ~1, OTM → ~0
            if option_type == "CE":
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0

        from scipy.stats import norm
        T = tte_days / 365
        r = 0.06  # risk-free rate approx
        d1 = (math.log(spot / strike) + (r + iv ** 2 / 2) * T) / (iv * math.sqrt(T))
        if option_type == "CE":
            return round(norm.cdf(d1), 4)
        else:
            return round(norm.cdf(d1) - 1, 4)

    # ------------------------------------------------------------------
    # Build option candidates from chain
    # ------------------------------------------------------------------
    def _build_candidates(
        self,
        kite: KiteClient,
        spot: float,
        signal_dir: int,
        df_spot: pd.DataFrame,
    ) -> List[OptionCandidate]:
        """
        Fetch NFO options chain near ATM, filter by delta 0.3–0.6,
        build OptionCandidate objects.
        """
        candidates = []
        try:
            nfo_instruments = kite.get_instruments("NFO")
        except Exception as e:
            logger.error(f"Failed to fetch NFO instruments: {e}")
            return candidates

        # Current expiry: nearest Thursday (weekly)
        today = date.today()
        nifty_options = [
            i for i in nfo_instruments
            if i.get("name") == "NIFTY"
            and i.get("segment") == "NFO-OPT"
            and i.get("expiry")
            and i["expiry"] >= today
        ]
        if not nifty_options:
            logger.warning("No NIFTY options found in NFO instruments")
            return candidates

        # Sort by expiry, take nearest
        nifty_options.sort(key=lambda x: x["expiry"])
        nearest_expiry = nifty_options[0]["expiry"]
        weekly_opts = [i for i in nifty_options if i["expiry"] == nearest_expiry]

        # ATM strike
        atm_strike = round(spot / NIFTY_STRIKE_GAP) * NIFTY_STRIKE_GAP
        strike_range = range(
            int(atm_strike - 5 * NIFTY_STRIKE_GAP),
            int(atm_strike + 6 * NIFTY_STRIKE_GAP),
            NIFTY_STRIKE_GAP,
        )

        # Filter CE if bullish, PE if bearish (plus some of the other side for hedging)
        primary_type = "CE" if signal_dir == 1 else "PE"
        relevant = [
            i for i in weekly_opts
            if i["strike"] in strike_range and i["instrument_type"] == primary_type
        ]

        if not relevant:
            return candidates

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

            # Delta filter: 0.3 – 0.6
            if abs(delta) < 0.30 or abs(delta) > 0.60:
                continue

            # Volume spike: compare to 20-bar option volume (simple: use daily volume > 5000)
            vol_spike = volume > 5000  # simplified threshold

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
                orb_high=self.orb_high or 0,
                orb_low=self.orb_low or 0,
                supertrend_dir=st_dir,
                macd_hist=macd_hist,
                macd_hist_prev=macd_hist_prev,
                vol_spike=vol_spike,
                atr=atr_val,
            )
            candidates.append(c)

        return candidates

    # ------------------------------------------------------------------
    # Execute trade on top candidate
    # ------------------------------------------------------------------
    def _execute_trade(self, kite: KiteClient, candidate: OptionCandidate) -> Optional[TradeRecord]:
        risk_check = self.risk_mgr.can_take_trade(
            abs(candidate.ltp) * NIFTY_LOT_SIZE * 0.05  # rough risk estimate
        )
        if not risk_check["allowed"]:
            logger.warning(f"Trade blocked: {risk_check['reasons']}")
            return None

        qty = self.risk_mgr.calculate_quantity(
            candidate.ltp,
            candidate.ltp * 0.7,  # 30% SL on option premium
            NIFTY_LOT_SIZE,
        )
        if qty <= 0:
            return None

        # Compute SL and target for option premium
        sl = round(candidate.ltp * 0.70, 2)  # 30% SL on premium
        target = round(candidate.ltp * 1.60, 2)  # 60% target on premium
        trailing_sl = round(candidate.ltp * 0.80, 2)

        tx_type = "BUY"
        result = kite.place_order(
            tradingsymbol=candidate.tradingsymbol,
            exchange="NFO",
            transaction_type=tx_type,
            quantity=qty,
            order_type="MARKET",
            product="MIS",
            tag="NiftyORB",
        )

        if not result.get("success"):
            logger.error(f"Order failed: {result}")
            return None

        trade = TradeRecord(
            trade_id=result.get("order_id", str(uuid.uuid4())),
            symbol=candidate.tradingsymbol,
            option_type=candidate.option_type,
            entry_price=candidate.ltp,
            entry_time=datetime.now().isoformat(),
            quantity=qty,
            sl=sl,
            target=target,
            trailing_sl=trailing_sl,
            score=candidate.score,
        )
        self.risk_mgr.register_trade(trade)
        telegram.alert_trade_entry(
            symbol=candidate.tradingsymbol,
            option_type=candidate.option_type,
            entry_price=candidate.ltp,
            qty=qty,
            sl=sl,
            target=target,
            score=candidate.score,
            trade_id=trade.trade_id,
        )
        return trade

    # ------------------------------------------------------------------
    # Square off all positions
    # ------------------------------------------------------------------
    def square_off_all(self, kite: KiteClient, reason: str = "SQUARED_OFF"):
        positions = self.risk_mgr.get_square_off_list()
        for trade in positions:
            try:
                # Get current LTP
                ltp_data = kite.get_ltp([f"NFO:{trade.symbol}"])
                ltp = ltp_data.get(f"NFO:{trade.symbol}", {}).get("last_price", trade.entry_price)

                # Place exit order
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
        telegram.alert_square_off(len(positions), self.risk_mgr.daily_pnl)
        self.dashboard_state["engine_status"] = "SQUARED_OFF"

    # ------------------------------------------------------------------
    # Offline / market-closed data fetch
    # ------------------------------------------------------------------
    def fetch_offline_data(self) -> Dict:
        """
        Fetch NIFTY spot LTP + last session's option chain scored candidates
        even when the market is closed. Updates dashboard_state and returns it.
        """
        kite = KiteClient.get_instance()

        # 1. Get NIFTY spot — try Kite LTP, fallback to yfinance
        spot = 0.0
        try:
            ltp_data = kite.get_ltp([NIFTY_INSTRUMENT])
            spot = ltp_data.get(NIFTY_INSTRUMENT, {}).get("last_price", 0)
        except Exception as e:
            logger.info(f"Offline: Kite LTP unavailable ({type(e).__name__}), trying yfinance…")

        if spot <= 0:
            spot = fetch_nifty_spot_yf()
            if spot > 0:
                logger.info(f"Offline: NIFTY spot from yfinance = {spot:.2f}")

        if spot > 0:
            self.nifty_spot = spot

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
                logger.info(f"Offline: Kite candidate build failed ({type(e).__name__}), using fallback")
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

            if self.all_candidates:
                self.scored_candidates = self.scorer.rank_candidates(
                    self.all_candidates, top_pct=10.0
                )

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
        self.dashboard_state["cycle_count"] = self._cycle_count
        self.dashboard_state["last_update"] = datetime.now().isoformat()
        return self.dashboard_state

    # ------------------------------------------------------------------
    # Monitor open positions (SL / target / trailing)
    # ------------------------------------------------------------------
    def monitor_positions(self, kite: KiteClient):
        if not self.risk_mgr.open_positions:
            return

        ltp_map = {}
        for tid, trade in list(self.risk_mgr.open_positions.items()):
            try:
                data = kite.get_ltp([f"NFO:{trade.symbol}"])
                ltp = data.get(f"NFO:{trade.symbol}", {}).get("last_price", 0)
                if ltp <= 0:
                    continue
                ltp_map[tid] = ltp

                # Check SL / target
                action = self.risk_mgr.check_sl_target(tid, ltp)
                if action:
                    kite.place_order(
                        tradingsymbol=trade.symbol,
                        exchange="NFO",
                        transaction_type="SELL",
                        quantity=trade.quantity,
                        order_type="MARKET",
                        product="MIS",
                        tag=action,
                    )
                    closed = self.risk_mgr.close_trade(tid, ltp, action)
                    if closed:
                        telegram.alert_trade_exit(
                            symbol=closed.symbol, option_type=closed.option_type,
                            entry_price=closed.entry_price, exit_price=ltp,
                            qty=closed.quantity, pnl=closed.pnl,
                            reason=action, trade_id=closed.trade_id,
                        )
                else:
                    # Update trailing SL
                    atr_val = 0
                    if self.all_candidates:
                        atr_val = self.all_candidates[0].atr
                    if atr_val > 0:
                        self.risk_mgr.update_trailing_sl(tid, ltp, atr_val)
            except Exception as e:
                logger.error(f"Monitor error for {trade.symbol}: {e}")

        self.risk_mgr.update_unrealised(ltp_map)

    # ------------------------------------------------------------------
    # Main cycle – called every 5 minutes by scheduler
    # ------------------------------------------------------------------
    def run_cycle(self) -> Dict:
        """
        Main engine cycle. Returns dashboard state dict.
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
            # Fetch offline data so dashboard still shows spot & candidates
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

        # -- Generate signal from ORB strategy --
        signal = self.strategy.generate_signal(df, "NIFTY")
        self.last_signal = signal

        # -- Monitor existing positions --
        self.monitor_positions(kite)

        # -- If we have a signal and capacity, score and trade --
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

            telegram.alert_signal(
                signal_type=signal.signal.value,
                entry=signal.entry_price,
                sl=signal.stop_loss,
                target=signal.target,
                confidence=signal.confidence,
                conditions=signal.conditions_met,
                reasoning=signal.reasoning,
            )

            # Prepare spot df with indicators for scoring
            df = self.strategy.compute_supertrend(df)
            df = self.strategy.compute_macd(df)

            signal_dir = 1 if signal.signal == SignalType.BUY else -1

            # Build and score candidates
            self.all_candidates = self._build_candidates(kite, self.nifty_spot, signal_dir, df)
            self.scored_candidates = self.scorer.rank_candidates(self.all_candidates, top_pct=10.0)

            # Execute on top candidate(s)
            for cand in self.scored_candidates:
                if len(self.risk_mgr.open_positions) >= self.risk_mgr.max_concurrent:
                    break
                self._execute_trade(kite, cand)

        else:
            self.dashboard_state["engine_status"] = "MONITORING"
            self.dashboard_state["signal"] = None

        self._update_dashboard()
        return self.dashboard_state

    # ------------------------------------------------------------------
    # Dashboard state
    # ------------------------------------------------------------------
    def _update_dashboard(self):
        self.dashboard_state["risk"] = self.risk_mgr.get_status()
        self.dashboard_state["trades"] = self.risk_mgr.get_trades_summary()
        self.dashboard_state["top_candidates"] = self.scorer.score_summary(
            self.scored_candidates
        ) if self.scored_candidates else {"count": 0, "top": []}
        self.dashboard_state["nifty_spot"] = self.nifty_spot
        self.dashboard_state["cycle_count"] = self._cycle_count
        self.dashboard_state["last_update"] = datetime.now().isoformat()

    def get_dashboard(self) -> Dict:
        return self.dashboard_state


# Need the import for signal_dir usage
from strategies.base_strategy import SignalType
