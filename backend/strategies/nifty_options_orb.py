"""
NIFTY Options ORB (Opening Range Breakout) Strategy

Flow:
  1. Capture NIFTY spot ORB (first 15 min high/low: 9:15–9:30)
  2. Supertrend on NIFTY 5-min as trend filter
  3. MACD on NIFTY 5-min as confirmation
  4. Volume spike detection on option contract
  5. Delta filter 0.3–0.6 applied by scoring engine upstream

Signals are generated AFTER 9:30 AM when ORB window is complete.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, time
from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType


class NiftyOptionsORBStrategy(BaseStrategy):
    name = "NIFTY Options ORB"
    description = (
        "Opening Range Breakout on NIFTY spot with Supertrend trend filter, "
        "MACD confirmation, and volume spike. Trades NIFTY options (CE/PE) "
        "filtered by delta 0.3–0.6."
    )
    timeframe = "5m"
    min_bars = 30

    def default_params(self) -> Dict[str, Any]:
        return {
            # ORB
            "orb_minutes": 15,  # first 15 min from 9:15
            "orb_buffer_pct": 0.05,  # 0.05% buffer above/below ORB
            # Supertrend
            "st_atr_period": 10,
            "st_multiplier": 3.0,
            # MACD
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            # Volume
            "vol_spike_mult": 1.5,  # volume must be 1.5x of 20-bar avg
            "vol_lookback": 20,
            # Risk
            "trailing_atr_mult": 1.5,
            "risk_reward_min": 2.0,
        }

    # ------------------------------------------------------------------
    # ORB capture
    # ------------------------------------------------------------------
    def compute_orb(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Return (orb_high, orb_low) from first 15 min of the session."""
        if "date" not in df.columns:
            return None, None

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        today = df["date"].dt.date.iloc[-1]
        market_open = pd.Timestamp(datetime.combine(today, time(9, 15)))
        orb_end = pd.Timestamp(datetime.combine(today, time(9, 30)))

        orb_bars = df[(df["date"] >= market_open) & (df["date"] < orb_end)]
        if orb_bars.empty:
            return None, None
        return float(orb_bars["high"].max()), float(orb_bars["low"].min())

    # ------------------------------------------------------------------
    # Supertrend (uses parent ATR helper)
    # ------------------------------------------------------------------
    def compute_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params["st_atr_period"]
        mult = self.params["st_multiplier"]
        atr = self.calculate_atr(df, period)

        hl2 = (df["high"] + df["low"]) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        direction = pd.Series(0, index=df.index, dtype=int)
        st_val = pd.Series(np.nan, index=df.index, dtype=float)

        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
                if direction.iloc[i] == 1:
                    lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1])
                else:
                    upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1])

            st_val.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

        df = df.copy()
        df["st_dir"] = direction
        df["st_val"] = st_val
        return df

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.params["macd_fast"]
        slow = self.params["macd_slow"]
        sig = self.params["macd_signal"]

        df = df.copy()
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=sig, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]
        return df

    # ------------------------------------------------------------------
    # Volume spike
    # ------------------------------------------------------------------
    def has_volume_spike(self, df: pd.DataFrame) -> bool:
        if "volume" not in df.columns:
            return True  # pass-through if volume unavailable
        lookback = self.params["vol_lookback"]
        mult = self.params["vol_spike_mult"]
        if len(df) < lookback + 1:
            return False
        avg_vol = df["volume"].iloc[-(lookback + 1):-1].mean()
        curr_vol = df["volume"].iloc[-1]
        return avg_vol > 0 and curr_vol >= avg_vol * mult

    # ------------------------------------------------------------------
    # Main signal generation (on NIFTY spot 5-min data)
    # ------------------------------------------------------------------
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None

        orb_high, orb_low = self.compute_orb(df)
        if orb_high is None:
            return None

        # Only trade after ORB window
        if "date" in df.columns:
            last_ts = pd.to_datetime(df["date"].iloc[-1])
            today = last_ts.date()
            orb_end = pd.Timestamp(datetime.combine(today, time(9, 30)))
            if last_ts < orb_end:
                return None
            # Don't generate new signals after 3:00 PM
            cutoff = pd.Timestamp(datetime.combine(today, time(15, 0)))
            if last_ts > cutoff:
                return None

        df = self.compute_supertrend(df)
        df = self.compute_macd(df)
        atr = self.calculate_atr(df).iloc[-1]

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]
        buffer = orb_high * (self.params["orb_buffer_pct"] / 100)

        conditions = []
        signal_type = None

        # ── BULLISH ORB Breakout ──
        if price > orb_high + buffer and prev["close"] <= orb_high + buffer:
            conditions.append(f"ORB breakout UP: {price:.2f} > {orb_high:.2f}")

            # Supertrend must be bullish
            if curr["st_dir"] == 1:
                conditions.append("Supertrend BULLISH")
            else:
                return None

            # MACD histogram must be positive & rising
            if curr["macd_hist"] > 0:
                conditions.append(f"MACD histogram positive: {curr['macd_hist']:.4f}")
                if curr["macd_hist"] > prev["macd_hist"]:
                    conditions.append("MACD histogram rising")
            else:
                return None

            # Volume spike
            if self.has_volume_spike(df):
                conditions.append("Volume spike detected")

            signal_type = SignalType.BUY

        # ── BEARISH ORB Breakout ──
        elif price < orb_low - buffer and prev["close"] >= orb_low - buffer:
            conditions.append(f"ORB breakdown: {price:.2f} < {orb_low:.2f}")

            if curr["st_dir"] == -1:
                conditions.append("Supertrend BEARISH")
            else:
                return None

            if curr["macd_hist"] < 0:
                conditions.append(f"MACD histogram negative: {curr['macd_hist']:.4f}")
                if curr["macd_hist"] < prev["macd_hist"]:
                    conditions.append("MACD histogram falling")
            else:
                return None

            if self.has_volume_spike(df):
                conditions.append("Volume spike detected")

            signal_type = SignalType.SELL

        else:
            return None

        # Calculate SL / target
        if signal_type == SignalType.BUY:
            sl = round(max(orb_low, price - atr * 1.5), 2)
            risk = price - sl
            target = round(price + risk * self.params["risk_reward_min"], 2)
        else:
            sl = round(min(orb_high, price + atr * 1.5), 2)
            risk = sl - price
            target = round(price - risk * self.params["risk_reward_min"], 2)

        trailing = self.calculate_trailing_sl(
            price, signal_type, atr, self.params["trailing_atr_mult"]
        )

        confidence = min(95.0, 50.0 + len(conditions) * 8.0)

        return TradeSignal(
            symbol=symbol,
            signal=signal_type,
            entry_price=price,
            stop_loss=sl,
            target=target,
            trailing_sl=trailing,
            confidence=confidence,
            strategy_name=self.name,
            reasoning=(
                f"ORB {'breakout' if signal_type == SignalType.BUY else 'breakdown'} "
                f"at {price:.2f}. ORB range [{orb_low:.2f}–{orb_high:.2f}]. "
                f"Supertrend + MACD aligned."
            ),
            conditions_met=conditions,
            timeframe=self.timeframe,
            timestamp=datetime.now().isoformat(),
        )
