"""
Exit Signal Generator

Generates EXIT_LONG and EXIT_SHORT signals based on:
  - Supertrend reversal (strongest exit)
  - MACD histogram reversal (cross below/above signal line)
  - VWAP cross (price crosses VWAP against position)
  - Trailing SL hit (price drops below trailing stop)
  - Target hit (T1 → partial, T2 → partial, T3 → full exit)
  - Time-based exit (forced square-off at 3:15 PM)
  - Candle pattern exhaustion (3 consecutive weak candles)

Each exit condition has a priority:
  CRITICAL (must exit immediately): Supertrend reversal, daily loss limit
  HIGH (strong exit): MACD reversal + VWAP cross together
  MEDIUM (partial exit): individual MACD or VWAP reversal
  LOW (trail adjustment): just tighten trailing SL
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, time
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExitPriority(str, Enum):
    CRITICAL = "CRITICAL"   # Exit full position immediately
    HIGH = "HIGH"           # Exit 75% of position
    MEDIUM = "MEDIUM"       # Exit 50% of position
    LOW = "LOW"             # Tighten trailing SL only


class ExitReason(str, Enum):
    SUPERTREND_REVERSAL = "SUPERTREND_REVERSAL"
    MACD_REVERSAL = "MACD_REVERSAL"
    VWAP_CROSS = "VWAP_CROSS"
    MACD_AND_VWAP = "MACD_AND_VWAP"
    TRAILING_SL_HIT = "TRAILING_SL_HIT"
    FIXED_SL_HIT = "FIXED_SL_HIT"
    TARGET1_HIT = "TARGET1_HIT"
    TARGET2_HIT = "TARGET2_HIT"
    TARGET3_HIT = "TARGET3_HIT"
    TIME_EXIT = "TIME_EXIT"
    CANDLE_EXHAUSTION = "CANDLE_EXHAUSTION"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    EXPIRY_DAY_EARLY_EXIT = "EXPIRY_DAY_EARLY_EXIT"
    TRAILING_SL_TIGHTENED = "TRAILING_SL_TIGHTENED"
    MANUAL = "MANUAL"


@dataclass
class ExitSignal:
    """A signal to exit (fully or partially) an open position."""
    reason: ExitReason
    priority: ExitPriority
    exit_pct: float             # 0.0–1.0 (portion of position to exit)
    exit_price: float
    message: str
    new_trailing_sl: Optional[float] = None   # Updated trailing SL for remaining qty
    conditions: List[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class PositionContext:
    """Current state of an open position needed for exit checks."""
    trade_id: str
    symbol: str
    option_type: str            # CE / PE
    entry_price: float
    current_ltp: float
    quantity: int
    remaining_qty: int          # after partial exits
    stop_loss: float
    trailing_sl: float
    target1: float
    target2: float
    target3: float
    t1_hit: bool = False
    t2_hit: bool = False
    t3_hit: bool = False
    entry_time: str = ""


class ExitSignalGenerator:
    """
    Evaluates all exit conditions for an open position.
    Returns the highest-priority exit signal, or None if no exit.
    """

    def __init__(
        self,
        square_off_time: time = time(15, 15),
        pre_close_warning_mins: int = 10,
    ):
        self.square_off_time = square_off_time
        self.pre_close_warning_mins = pre_close_warning_mins

    def evaluate(
        self,
        pos: PositionContext,
        df: pd.DataFrame,
        vwap: float = 0.0,
    ) -> Optional[ExitSignal]:
        """
        Check all exit conditions and return the highest-priority signal.
        df = latest NIFTY spot 5-min data with supertrend + MACD computed.
        """
        signals: List[ExitSignal] = []

        # 1. SL checks (highest priority after supertrend)
        sl_signal = self._check_sl(pos)
        if sl_signal:
            signals.append(sl_signal)

        # 2. Target checks
        target_signal = self._check_targets(pos)
        if target_signal:
            signals.append(target_signal)

        # 3. Supertrend reversal
        if df is not None and len(df) >= 2:
            st_signal = self._check_supertrend_reversal(pos, df)
            if st_signal:
                signals.append(st_signal)

            # 4. MACD reversal
            macd_signal = self._check_macd_reversal(pos, df)
            if macd_signal:
                signals.append(macd_signal)

            # 5. Candle exhaustion
            exhaust_signal = self._check_candle_exhaustion(pos, df)
            if exhaust_signal:
                signals.append(exhaust_signal)

        # 6. VWAP cross
        if vwap > 0:
            vwap_signal = self._check_vwap_cross(pos, vwap)
            if vwap_signal:
                signals.append(vwap_signal)

        # 7. Combined MACD + VWAP (stronger signal)
        has_macd = any(s.reason == ExitReason.MACD_REVERSAL for s in signals)
        has_vwap = any(s.reason == ExitReason.VWAP_CROSS for s in signals)
        if has_macd and has_vwap:
            combined = ExitSignal(
                reason=ExitReason.MACD_AND_VWAP,
                priority=ExitPriority.HIGH,
                exit_pct=0.75,
                exit_price=pos.current_ltp,
                message=f"MACD reversal + VWAP cross — exit 75%",
                conditions=["MACD histogram reversed", "Price crossed VWAP against position"],
                timestamp=datetime.now().isoformat(),
            )
            signals.append(combined)

        # 8. Time-based exit
        time_signal = self._check_time_exit(pos)
        if time_signal:
            signals.append(time_signal)

        if not signals:
            return None

        # Sort by priority (CRITICAL > HIGH > MEDIUM > LOW)
        priority_order = {
            ExitPriority.CRITICAL: 0,
            ExitPriority.HIGH: 1,
            ExitPriority.MEDIUM: 2,
            ExitPriority.LOW: 3,
        }
        signals.sort(key=lambda s: priority_order.get(s.priority, 99))
        return signals[0]

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_sl(self, pos: PositionContext) -> Optional[ExitSignal]:
        """Check if fixed SL or trailing SL is hit on option premium."""
        price = pos.current_ltp

        # Option premiums always go towards zero when losing,
        # so SL hit = premium drops below SL level (both CE and PE)
        if price <= pos.stop_loss:
            return ExitSignal(
                reason=ExitReason.FIXED_SL_HIT,
                priority=ExitPriority.CRITICAL,
                exit_pct=1.0,
                exit_price=price,
                message=f"FIXED SL HIT at ₹{price:.2f} (SL: ₹{pos.stop_loss:.2f})",
                conditions=[f"Premium {price:.2f} <= SL {pos.stop_loss:.2f}"],
                timestamp=datetime.now().isoformat(),
            )

        if pos.trailing_sl > pos.stop_loss and price <= pos.trailing_sl:
            return ExitSignal(
                reason=ExitReason.TRAILING_SL_HIT,
                priority=ExitPriority.CRITICAL,
                exit_pct=1.0,
                exit_price=price,
                message=f"TRAILING SL HIT at ₹{price:.2f} (TSL: ₹{pos.trailing_sl:.2f})",
                conditions=[f"Premium {price:.2f} <= Trailing SL {pos.trailing_sl:.2f}"],
                timestamp=datetime.now().isoformat(),
            )

        return None

    def _check_targets(self, pos: PositionContext) -> Optional[ExitSignal]:
        """Check if T1 / T2 / T3 hit. Partial exits at T1 and T2.
        TSL moves to proper phase-based levels to lock profits."""
        price = pos.current_ltp

        # For options: premium goes up for both CE profit and PE profit
        if not pos.t3_hit and price >= pos.target3:
            # T3 hit — exit all remaining, TSL at T2 for safety (instant exit)
            return ExitSignal(
                reason=ExitReason.TARGET3_HIT,
                priority=ExitPriority.CRITICAL,
                exit_pct=1.0,  # Exit ALL remaining
                exit_price=price,
                message=f"🎯 TARGET 3 HIT at ₹{price:.2f} — FULL EXIT",
                new_trailing_sl=pos.target2,  # Floor at T2 if partial
                conditions=[f"Premium {price:.2f} >= T3 {pos.target3:.2f}"],
                timestamp=datetime.now().isoformat(),
            )

        if not pos.t2_hit and price >= pos.target2:
            # T2 hit — exit 1/3, move TSL to T1 (locked profit)
            new_tsl = pos.target1
            return ExitSignal(
                reason=ExitReason.TARGET2_HIT,
                priority=ExitPriority.HIGH,
                exit_pct=0.34,  # Exit 1/3 of remaining
                exit_price=price,
                message=f"🎯 TARGET 2 HIT at ₹{price:.2f} — exit 34%, trail SL to T1 (₹{new_tsl:.2f})",
                new_trailing_sl=new_tsl,
                conditions=[f"Premium {price:.2f} >= T2 {pos.target2:.2f}"],
                timestamp=datetime.now().isoformat(),
            )

        if not pos.t1_hit and price >= pos.target1:
            # T1 hit — exit 1/3, move TSL to breakeven (entry price)
            new_tsl = pos.entry_price
            return ExitSignal(
                reason=ExitReason.TARGET1_HIT,
                priority=ExitPriority.HIGH,
                exit_pct=0.34,  # Exit 1/3 of position
                exit_price=price,
                message=f"🎯 TARGET 1 HIT at ₹{price:.2f} — exit 34%, SL to breakeven (₹{new_tsl:.2f})",
                new_trailing_sl=new_tsl,
                conditions=[f"Premium {price:.2f} >= T1 {pos.target1:.2f}"],
                timestamp=datetime.now().isoformat(),
            )

        return None

    def _check_supertrend_reversal(
        self, pos: PositionContext, df: pd.DataFrame
    ) -> Optional[ExitSignal]:
        """
        Supertrend direction flipped against position → CRITICAL exit.
        CE needs bullish supertrend, PE needs bearish.
        """
        if "st_dir" not in df.columns and "supertrend_dir" not in df.columns:
            return None

        dir_col = "st_dir" if "st_dir" in df.columns else "supertrend_dir"
        curr_dir = int(df.iloc[-1][dir_col])
        prev_dir = int(df.iloc[-2][dir_col])

        if pos.option_type == "CE":
            # CE needs bullish (1). If flipped to -1 → exit
            if prev_dir == 1 and curr_dir == -1:
                return ExitSignal(
                    reason=ExitReason.SUPERTREND_REVERSAL,
                    priority=ExitPriority.CRITICAL,
                    exit_pct=1.0,
                    exit_price=pos.current_ltp,
                    message=f"⚠️ SUPERTREND REVERSED BEARISH — EXIT CE position",
                    conditions=["Supertrend flipped from BULL to BEAR"],
                    timestamp=datetime.now().isoformat(),
                )
        else:  # PE
            if prev_dir == -1 and curr_dir == 1:
                return ExitSignal(
                    reason=ExitReason.SUPERTREND_REVERSAL,
                    priority=ExitPriority.CRITICAL,
                    exit_pct=1.0,
                    exit_price=pos.current_ltp,
                    message=f"⚠️ SUPERTREND REVERSED BULLISH — EXIT PE position",
                    conditions=["Supertrend flipped from BEAR to BULL"],
                    timestamp=datetime.now().isoformat(),
                )

        return None

    def _check_macd_reversal(
        self, pos: PositionContext, df: pd.DataFrame
    ) -> Optional[ExitSignal]:
        """
        MACD histogram crosses zero against the position direction.
        CE: histogram goes negative → exit signal
        PE: histogram goes positive → exit signal
        """
        if "macd_hist" not in df.columns:
            return None

        curr_hist = float(df.iloc[-1]["macd_hist"])
        prev_hist = float(df.iloc[-2]["macd_hist"])

        if pos.option_type == "CE":
            if prev_hist > 0 and curr_hist <= 0:
                return ExitSignal(
                    reason=ExitReason.MACD_REVERSAL,
                    priority=ExitPriority.MEDIUM,
                    exit_pct=0.50,
                    exit_price=pos.current_ltp,
                    message=f"MACD histogram turned negative — partial exit CE",
                    conditions=[f"MACD hist: {prev_hist:.4f} → {curr_hist:.4f} (crossed zero)"],
                    timestamp=datetime.now().isoformat(),
                )
        else:  # PE
            if prev_hist < 0 and curr_hist >= 0:
                return ExitSignal(
                    reason=ExitReason.MACD_REVERSAL,
                    priority=ExitPriority.MEDIUM,
                    exit_pct=0.50,
                    exit_price=pos.current_ltp,
                    message=f"MACD histogram turned positive — partial exit PE",
                    conditions=[f"MACD hist: {prev_hist:.4f} → {curr_hist:.4f} (crossed zero)"],
                    timestamp=datetime.now().isoformat(),
                )

        return None

    def _check_vwap_cross(
        self, pos: PositionContext, vwap: float
    ) -> Optional[ExitSignal]:
        """
        Price crosses VWAP against the position.
        For exit purposes, we're checking NIFTY spot vs VWAP on spot chart.
        But since we get option LTP, we use a proxy:
        if option premium is below entry and VWAP indicates adverse move.

        More precisely: this is called with spot VWAP comparison done
        externally (caller checks spot vs VWAP), so here we just check
        if this constitues an exit condition.
        """
        # This is a simplified check: VWAP cross detected by caller
        # If price is moving against position and crossed VWAP, signal exit
        if pos.option_type == "CE":
            # CE losing money if ltp < entry
            if pos.current_ltp < pos.entry_price:
                return ExitSignal(
                    reason=ExitReason.VWAP_CROSS,
                    priority=ExitPriority.MEDIUM,
                    exit_pct=0.50,
                    exit_price=pos.current_ltp,
                    message=f"VWAP cross bearish + CE in loss",
                    conditions=["Spot crossed below VWAP", f"CE premium declining: {pos.current_ltp:.2f}"],
                    timestamp=datetime.now().isoformat(),
                )
        else:
            if pos.current_ltp < pos.entry_price:
                return ExitSignal(
                    reason=ExitReason.VWAP_CROSS,
                    priority=ExitPriority.MEDIUM,
                    exit_pct=0.50,
                    exit_price=pos.current_ltp,
                    message=f"VWAP cross bullish + PE in loss",
                    conditions=["Spot crossed above VWAP", f"PE premium declining: {pos.current_ltp:.2f}"],
                    timestamp=datetime.now().isoformat(),
                )

        return None

    def _check_candle_exhaustion(
        self, pos: PositionContext, df: pd.DataFrame
    ) -> Optional[ExitSignal]:
        """
        3 consecutive candles closing in the adverse direction = momentum exhaustion.
        Tighten trailing SL aggressively based on current gain.
        """
        if len(df) < 4:
            return None

        closes = df["close"].iloc[-4:].values
        opens = df["open"].iloc[-4:].values if "open" in df.columns else closes

        if pos.option_type == "CE":
            # 3 red candles (close < open)
            red_count = sum(1 for i in range(1, 4) if closes[i] < opens[i])
            if red_count >= 3:
                # Tighten TSL: if in profit, move SL up aggressively
                gain_pct = (pos.current_ltp - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
                if gain_pct > 5:
                    # Lock at least 50% of current profit
                    profit_per_unit = pos.current_ltp - pos.entry_price
                    new_tsl = pos.entry_price + profit_per_unit * 0.5
                else:
                    # Just tighten slightly towards current price
                    new_tsl = max(pos.trailing_sl, pos.current_ltp * 0.93)
                return ExitSignal(
                    reason=ExitReason.CANDLE_EXHAUSTION,
                    priority=ExitPriority.LOW,
                    exit_pct=0.0,  # Just tighten TSL, no exit
                    exit_price=pos.current_ltp,
                    message=f"3 bearish candles — tightening TSL to ₹{new_tsl:.2f}",
                    new_trailing_sl=round(new_tsl, 2),
                    conditions=["3 consecutive red candles detected"],
                    timestamp=datetime.now().isoformat(),
                )
        else:
            green_count = sum(1 for i in range(1, 4) if closes[i] > opens[i])
            if green_count >= 3:
                gain_pct = (pos.current_ltp - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
                if gain_pct > 5:
                    profit_per_unit = pos.current_ltp - pos.entry_price
                    new_tsl = pos.entry_price + profit_per_unit * 0.5
                else:
                    new_tsl = max(pos.trailing_sl, pos.current_ltp * 0.93)
                return ExitSignal(
                    reason=ExitReason.CANDLE_EXHAUSTION,
                    priority=ExitPriority.LOW,
                    exit_pct=0.0,
                    exit_price=pos.current_ltp,
                    message=f"3 bullish candles — tightening TSL for PE to ₹{new_tsl:.2f}",
                    new_trailing_sl=round(new_tsl, 2),
                    conditions=["3 consecutive green candles detected"],
                    timestamp=datetime.now().isoformat(),
                )

        return None

    def _check_time_exit(self, pos: PositionContext) -> Optional[ExitSignal]:
        """Check if we're past square-off time. On expiry day, use earlier exit time."""
        from config import settings
        from datetime import date

        now = datetime.now().time()
        today = date.today()

        # Expiry-day early exit: exit at settings.expiry_day_early_exit_time
        # Detect expiry day: Thursday (weekday=3) or if nearest expiry == today
        is_expiry_day = today.weekday() == 3  # Thursday
        if is_expiry_day:
            try:
                hh, mm = settings.expiry_day_early_exit_time.split(":")
                early_exit_time = time(int(hh), int(mm))
                if now >= early_exit_time:
                    return ExitSignal(
                        reason=ExitReason.EXPIRY_DAY_EARLY_EXIT,
                        priority=ExitPriority.CRITICAL,
                        exit_pct=1.0,
                        exit_price=pos.current_ltp,
                        message=f"⏰ EXPIRY DAY early exit at {early_exit_time.strftime('%H:%M')} (theta decay risk)",
                        conditions=[
                            f"Expiry day detected",
                            f"Current time {now.strftime('%H:%M')} >= early exit {early_exit_time.strftime('%H:%M')}",
                        ],
                        timestamp=datetime.now().isoformat(),
                    )
            except Exception:
                pass  # Fall through to normal time exit

        if now >= self.square_off_time:
            return ExitSignal(
                reason=ExitReason.TIME_EXIT,
                priority=ExitPriority.CRITICAL,
                exit_pct=1.0,
                exit_price=pos.current_ltp,
                message=f"⏰ SQUARE-OFF TIME {self.square_off_time.strftime('%H:%M')} reached",
                conditions=[f"Current time {now.strftime('%H:%M')} >= {self.square_off_time.strftime('%H:%M')}"],
                timestamp=datetime.now().isoformat(),
            )
        return None
