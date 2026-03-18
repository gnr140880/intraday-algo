"""
ORB Range Scalper Strategy

For sideways / range-bound markets where price stays INSIDE the ORB range:
  - After ORB is established (9:15 + 15 min), if price doesn't break out,
    it's a range-bound day
  - BUY (CE) when price bounces off ORB Low support
  - SELL (PUT) when price rejects from ORB High resistance
  - Target: midpoint of ORB range (mean reversion within range)

Why this works:
  - On ~60% of days, the first 15-min range holds for the session
  - Price oscillates between ORB High and ORB Low like support/resistance
  - Traditional ORB breakout strategies MISS these days entirely
  - This strategy captures 3-5 small scalps on range-bound days
  - Works best when ORB range is 0.3%-0.8% of spot (not too wide, not too narrow)

Entry Rules:
  BUY (near ORB Low):
    - Price within 0.1% of ORB Low (touching support)
    - Candle shows bounce (close > open, or wick below ORB Low + close inside)
    - RSI < 40 (not overbought)
    - At least 30 min after ORB capture (not too early)

  SELL / BUY PUT (near ORB High):
    - Price within 0.1% of ORB High (touching resistance)
    - Candle shows rejection (close < open, or wick above ORB High + close inside)
    - RSI > 60 (not oversold)
    - At least 30 min after ORB capture

Risk Management:
  - SL: 0.15% beyond ORB boundary (tight SL since it's a range scalp)
  - Target: ORB midpoint (halfway between ORB High and ORB Low)
  - Max 4 scalps per day
  - 20-min cooldown between trades
  - STOP trading this strategy if ORB breaks (price closes decisively beyond ORB)

Kill Switch:
  - If price closes 0.3% beyond ORB High or Low → ORB broken → STOP range scalping
  - Switch to breakout mode (handled by ORB Breakout strategy)
"""
import logging
from datetime import datetime, time, timedelta, date
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType

logger = logging.getLogger(__name__)


class ORBRangeScalperStrategy(BaseStrategy):
    name = "ORB Range Scalper"
    description = (
        "Scalps bounces within the ORB range on sideways days. "
        "Buys near ORB Low, sells near ORB High, targets midpoint."
    )
    timeframe = "5m"
    min_bars = 15

    def __init__(self, params=None):
        super().__init__(params)
        self._last_signal_time: Optional[datetime] = None
        self._last_signal_dir: Optional[str] = None
        self._daily_count: int = 0
        self._last_reset_date = None
        self._orb_broken: bool = False  # Kill switch if ORB breaks

    def default_params(self) -> Dict[str, Any]:
        return {
            "proximity_pct": 0.15,       # Price within 0.15% of ORB boundary
            "sl_beyond_pct": 0.20,       # SL 0.20% beyond ORB boundary
            "target_mode": "midpoint",   # "midpoint" or "opposite_band"
            "rsi_buy_max": 45,           # RSI must be < 45 for buy (not overbought)
            "rsi_sell_min": 55,          # RSI must be > 55 for sell (not oversold)
            "rsi_period": 14,
            "min_orb_range_pct": 0.25,   # ORB range must be > 0.25% (else no room)
            "max_orb_range_pct": 1.0,    # ORB range must be < 1.0% (else trending)
            "cooldown_minutes": 20,
            "max_trades_per_day": 4,
            "orb_break_threshold_pct": 0.3,  # ORB broken if price 0.3% beyond
            "entry_after": "10:00",      # Wait 30+ min after ORB for confirmation
            "entry_before": "14:30",
            "require_bounce_candle": True,
        }

    def _compute_rsi(self, df: pd.DataFrame, period: int = 14):
        delta = df["close"].astype(float).diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    def _check_cooldown(self, direction: str) -> bool:
        if self._last_signal_time is None:
            return False
        if self._last_signal_dir == direction:
            elapsed = (datetime.now() - self._last_signal_time).total_seconds() / 60
            if elapsed < self.params["cooldown_minutes"]:
                return True
        return False

    def _reset_daily(self):
        today = date.today()
        if self._last_reset_date != today:
            self._daily_count = 0
            self._last_reset_date = today
            self._orb_broken = False

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        orb_high: float = 0,
        orb_low: float = 0,
        **kw,
    ) -> Optional[TradeSignal]:
        """Generate ORB range scalper signal.

        Args:
            orb_high: ORB high from engine (must be provided)
            orb_low: ORB low from engine (must be provided)
        """
        self._reset_daily()

        if len(df) < self.min_bars:
            return None

        # Need ORB values
        if orb_high <= 0 or orb_low <= 0:
            return None

        # Time window
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        if df_copy["date"].dt.tz is not None:
            df_copy["date"] = df_copy["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

        lt = df_copy["date"].iloc[-1]
        hh, mm = self.params["entry_after"].split(":")
        if lt.time() < time(int(hh), int(mm)):
            return None
        hh2, mm2 = self.params["entry_before"].split(":")
        if lt.time() > time(int(hh2), int(mm2)):
            return None

        # Daily trade limit
        if self._daily_count >= self.params["max_trades_per_day"]:
            return None

        # Kill switch: if ORB already broken, stop range scalping
        if self._orb_broken:
            return None

        price = float(df_copy["close"].iloc[-1])
        spot = price
        orb_range = orb_high - orb_low
        orb_mid = (orb_high + orb_low) / 2

        if orb_range <= 0 or spot <= 0:
            return None

        orb_range_pct = (orb_range / spot) * 100

        # ORB range size filter
        if orb_range_pct < self.params["min_orb_range_pct"]:
            return None  # Range too narrow, no room for scalp
        if orb_range_pct > self.params["max_orb_range_pct"]:
            return None  # Range too wide = trending day

        # Check if ORB is broken (kill switch)
        break_threshold = spot * self.params["orb_break_threshold_pct"] / 100
        if price > orb_high + break_threshold or price < orb_low - break_threshold:
            self._orb_broken = True
            logger.info(
                f"ORB Range Scalper: ORB BROKEN at ₹{price:.2f} "
                f"(range {orb_low:.2f}-{orb_high:.2f}). Stopping range scalps."
            )
            return None

        # Compute RSI
        rsi = self._compute_rsi(df_copy, self.params["rsi_period"])
        curr_rsi = float(rsi.iloc[-1])
        atr = self.calculate_atr(df_copy)
        curr_atr = float(atr.iloc[-1])

        # Current candle data
        curr_open = float(df_copy["open"].iloc[-1])
        curr_close = float(df_copy["close"].iloc[-1])
        curr_low = float(df_copy["low"].iloc[-1])
        curr_high = float(df_copy["high"].iloc[-1])

        prox = spot * self.params["proximity_pct"] / 100
        sl_beyond = spot * self.params["sl_beyond_pct"] / 100

        conds = []
        sig = None
        sl = 0
        target = orb_mid  # Default target = midpoint

        # ============================================================
        # BUY (CE): Price near ORB Low — support bounce
        # ============================================================
        if curr_low <= orb_low + prox and price > orb_low:
            # RSI check: must not be overbought
            if curr_rsi < self.params["rsi_buy_max"]:
                # Bounce candle check
                is_bounce = True
                if self.params["require_bounce_candle"]:
                    # Close above open (green candle) OR wick below ORB Low but close inside
                    is_bounce = (curr_close > curr_open) or \
                                (curr_low < orb_low and curr_close > orb_low)

                if is_bounce:
                    conds.append(f"Price ₹{price:.2f} bouncing off ORB Low ₹{orb_low:.2f}")
                    conds.append(f"RSI {curr_rsi:.1f} (not overbought)")
                    conds.append(f"ORB range {orb_range_pct:.2f}% (sideways)")
                    conds.append(f"Target: ORB midpoint ₹{orb_mid:.2f}")

                    sl = orb_low - sl_beyond
                    sig = SignalType.BUY

        # ============================================================
        # SELL (PUT): Price near ORB High — resistance rejection
        # ============================================================
        elif curr_high >= orb_high - prox and price < orb_high:
            if curr_rsi > self.params["rsi_sell_min"]:
                is_rejection = True
                if self.params["require_bounce_candle"]:
                    is_rejection = (curr_close < curr_open) or \
                                   (curr_high > orb_high and curr_close < orb_high)

                if is_rejection:
                    conds.append(f"Price ₹{price:.2f} rejecting ORB High ₹{orb_high:.2f}")
                    conds.append(f"RSI {curr_rsi:.1f} (not oversold)")
                    conds.append(f"ORB range {orb_range_pct:.2f}% (sideways)")
                    conds.append(f"Target: ORB midpoint ₹{orb_mid:.2f}")

                    sl = orb_high + sl_beyond
                    sig = SignalType.SELL

        if sig is None:
            return None

        # Cooldown
        d = "BUY" if sig == SignalType.BUY else "SELL"
        if self._check_cooldown(d):
            return None

        # Risk/Reward
        risk = abs(price - sl)
        reward = abs(target - price)
        rr = reward / risk if risk > 0 else 0

        if rr < 0.8:  # Even 0.8 R:R is acceptable for high-probability scalps
            return None

        confidence = min(75.0, 40.0 + len(conds) * 8.0)

        # Record
        self._last_signal_time = datetime.now()
        self._last_signal_dir = d
        self._daily_count += 1

        conds.append(f"R:R {rr:.1f}x")

        return TradeSignal(
            symbol=symbol,
            signal=sig,
            entry_price=round(price, 2),
            stop_loss=round(sl, 2),
            target=round(target, 2),
            trailing_sl=round(
                price - curr_atr * 0.8 if sig == SignalType.BUY else price + curr_atr * 0.8, 2
            ),
            confidence=confidence,
            strategy_name=self.name,
            reasoning=(
                f"ORB range scalp {d}: price {'bouncing off ORB Low' if d == 'BUY' else 'rejecting ORB High'}, "
                f"targeting midpoint ₹{target:.2f}. "
                f"ORB range={orb_range_pct:.1f}% (sideways). R:R={rr:.1f}x"
            ),
            conditions_met=conds,
            timeframe=self.timeframe,
            timestamp=datetime.now().isoformat(),
        )

    def get_cached_levels(self):
        return None

