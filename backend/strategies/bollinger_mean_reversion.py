"""
Bollinger Band Mean Reversion Strategy

For sideways / range-bound markets (which occur 60-70% of trading days):
  - BUY when price touches lower Bollinger Band and bounces
  - SELL (buy PUT) when price touches upper Bollinger Band and rejects
  - Bollinger Band Width (BBW) must be narrow → confirms range-bound
  - RSI confirms oversold/overbought at the bands

Why this works on sideways days:
  - Price oscillates within a statistical range (mean ± 2σ)
  - Touches of the bands in low-vol regimes tend to revert to the mean
  - Combined with RSI extremes gives high-probability entries
  - Works on NIFTY 5-min chart, 3-5 trades per day typically

Entry Rules:
  BUY:
    - Price closes below lower BB (or wick pierces it)
    - RSI < 35 (oversold confirmation)
    - Previous candle also near lower band (sustained test, not spike)
    - Bollinger Band Width (BBW) < threshold (range-bound)
    - Close back inside band (bounce candle)

  SELL (BUY PUT):
    - Price closes above upper BB (or wick pierces it)
    - RSI > 65 (overbought confirmation)
    - BBW < threshold
    - Close back inside band (rejection candle)

SL: Beyond the opposite 2.5σ band (or ATR-based)
T1: Middle band (20-SMA = mean)
T2: Opposite band

Risk Management:
  - Only trade when BBW < 3% (squeeze = range-bound)
  - Skip if BBW expanding (breakout imminent)
  - Max 3 mean-reversion trades per day
  - 15-min cooldown between same-direction trades
"""
import logging
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType

logger = logging.getLogger(__name__)


class BollingerMeanReversionStrategy(BaseStrategy):
    name = "Bollinger Mean Reversion"
    description = (
        "Buys at lower Bollinger Band, sells at upper band on range-bound days. "
        "Targets mean (20-SMA). Best when Bollinger Band Width is narrow (squeeze)."
    )
    timeframe = "5m"
    min_bars = 30

    def __init__(self, params=None):
        super().__init__(params)
        self._last_signal_time: Optional[datetime] = None
        self._last_signal_dir: Optional[str] = None
        self._daily_signal_count: int = 0
        self._last_reset_date = None

    def default_params(self) -> Dict[str, Any]:
        return {
            "bb_period": 20,           # Bollinger Band SMA period
            "bb_std": 2.0,             # Standard deviations for bands
            "rsi_period": 14,
            "rsi_oversold": 35,        # RSI < 35 for BUY confirmation
            "rsi_overbought": 65,      # RSI > 65 for SELL confirmation
            "bbw_max_pct": 3.0,        # Max band width % (above = trending)
            "bbw_min_pct": 0.3,        # Min band width % (too tight = no move)
            "require_bounce": True,    # Require price to bounce back inside band
            "cooldown_minutes": 15,    # Min minutes between same-direction signals
            "max_signals_per_day": 4,  # Max mean-reversion trades per day
            "sl_atr_mult": 1.5,        # SL = ATR × this multiplier beyond band
            "entry_after": "09:45",    # Don't trade in first 30 min
            "entry_before": "14:45",   # Don't trade in last 45 min
        }

    def _compute_bollinger(self, df: pd.DataFrame):
        """Compute Bollinger Bands, BBW%, and RSI."""
        period = self.params["bb_period"]
        std_mult = self.params["bb_std"]
        rsi_period = self.params["rsi_period"]

        close = df["close"].astype(float)

        # Bollinger Bands
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + std_mult * std
        lower = sma - std_mult * std

        # Band Width % = (Upper - Lower) / Middle × 100
        bbw_pct = ((upper - lower) / sma * 100).fillna(0)

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return sma, upper, lower, bbw_pct, rsi

    def _check_cooldown(self, direction: str) -> bool:
        """Returns True if should BLOCK (cooldown active)."""
        if self._last_signal_time is None:
            return False
        if self._last_signal_dir == direction:
            elapsed = (datetime.now() - self._last_signal_time).total_seconds() / 60
            if elapsed < self.params["cooldown_minutes"]:
                return True
        return False

    def _reset_daily(self):
        from datetime import date
        today = date.today()
        if self._last_reset_date != today:
            self._daily_signal_count = 0
            self._last_reset_date = today

    def generate_signal(self, df: pd.DataFrame, symbol: str, **kw) -> Optional[TradeSignal]:
        """Generate Bollinger Band mean reversion signal."""
        self._reset_daily()

        if len(df) < self.min_bars:
            return None

        # Time window check
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

        # Daily limit
        if self._daily_signal_count >= self.params["max_signals_per_day"]:
            return None

        # Compute indicators
        sma, upper, lower, bbw_pct, rsi = self._compute_bollinger(df_copy)
        atr = self.calculate_atr(df_copy)

        if sma.isna().iloc[-1] or upper.isna().iloc[-1]:
            return None

        price = float(df_copy["close"].iloc[-1])
        prev_price = float(df_copy["close"].iloc[-2])
        curr_low = float(df_copy["low"].iloc[-1])
        curr_high = float(df_copy["high"].iloc[-1])
        curr_upper = float(upper.iloc[-1])
        curr_lower = float(lower.iloc[-1])
        curr_sma = float(sma.iloc[-1])
        curr_bbw = float(bbw_pct.iloc[-1])
        curr_rsi = float(rsi.iloc[-1])
        prev_bbw = float(bbw_pct.iloc[-2])
        curr_atr = float(atr.iloc[-1])

        # --- Range-bound filter: BBW must be within range ---
        if curr_bbw > self.params["bbw_max_pct"]:
            # Bands too wide = trending market, skip mean reversion
            return None
        if curr_bbw < self.params["bbw_min_pct"]:
            # Bands too tight = no volatility, skip
            return None

        # --- BBW should not be expanding rapidly (breakout imminent) ---
        if curr_bbw > prev_bbw * 1.3:
            # BBW expanding > 30% = potential breakout, skip
            return None

        conds = []
        sig = None

        # ============================================================
        # BUY: Price at/below lower Bollinger Band + RSI oversold
        # ============================================================
        if (curr_low <= curr_lower or price <= curr_lower * 1.001) and \
           curr_rsi < self.params["rsi_oversold"]:

            # Bounce check: close must be back above the lower band
            if self.params["require_bounce"] and price < curr_lower:
                # No bounce yet — wick touched but didn't close above
                return None

            conds.append(f"Price ₹{price:.2f} at lower BB ₹{curr_lower:.2f}")
            conds.append(f"RSI {curr_rsi:.1f} < {self.params['rsi_oversold']} (oversold)")
            conds.append(f"BB Width {curr_bbw:.2f}% (range-bound)")

            # Previous candle also near lower band = sustained test
            prev_lower = float(lower.iloc[-2])
            if float(df_copy["low"].iloc[-2]) <= prev_lower * 1.005:
                conds.append("Previous candle also tested lower band")

            sl = curr_lower - curr_atr * self.params["sl_atr_mult"]
            t1 = curr_sma  # Mean (20-SMA)
            t2 = curr_upper  # Opposite band

            sig = SignalType.BUY

        # ============================================================
        # SELL (BUY PUT): Price at/above upper BB + RSI overbought
        # ============================================================
        elif (curr_high >= curr_upper or price >= curr_upper * 0.999) and \
             curr_rsi > self.params["rsi_overbought"]:

            if self.params["require_bounce"] and price > curr_upper:
                return None

            conds.append(f"Price ₹{price:.2f} at upper BB ₹{curr_upper:.2f}")
            conds.append(f"RSI {curr_rsi:.1f} > {self.params['rsi_overbought']} (overbought)")
            conds.append(f"BB Width {curr_bbw:.2f}% (range-bound)")

            prev_upper = float(upper.iloc[-2])
            if float(df_copy["high"].iloc[-2]) >= prev_upper * 0.995:
                conds.append("Previous candle also tested upper band")

            sl = curr_upper + curr_atr * self.params["sl_atr_mult"]
            t1 = curr_sma
            t2 = curr_lower

            sig = SignalType.SELL

        if sig is None:
            return None

        # Cooldown check
        d = "BUY" if sig == SignalType.BUY else "SELL"
        if self._check_cooldown(d):
            return None

        # Compute risk/reward
        risk = abs(price - sl)
        reward = abs(t1 - price)
        rr = reward / risk if risk > 0 else 0

        # Require minimum R:R of 1.0
        if rr < 1.0:
            return None

        confidence = min(80.0, 45.0 + len(conds) * 8.0)

        # Record signal
        self._last_signal_time = datetime.now()
        self._last_signal_dir = d
        self._daily_signal_count += 1

        conds.append(f"Target: 20-SMA ₹{t1:.2f}")
        conds.append(f"R:R {rr:.1f}x")

        return TradeSignal(
            symbol=symbol,
            signal=sig,
            entry_price=round(price, 2),
            stop_loss=round(sl, 2),
            target=round(t1, 2),
            trailing_sl=round(price - curr_atr if sig == SignalType.BUY else price + curr_atr, 2),
            confidence=confidence,
            strategy_name=self.name,
            reasoning=(
                f"BB mean reversion {d}: price at {'lower' if d == 'BUY' else 'upper'} band, "
                f"RSI={curr_rsi:.1f}, targeting 20-SMA ₹{t1:.2f}. "
                f"BBW={curr_bbw:.1f}% (range-bound). R:R={rr:.1f}x"
            ),
            conditions_met=conds,
            timeframe=self.timeframe,
            timestamp=datetime.now().isoformat(),
        )

    def get_cached_levels(self):
        return None

