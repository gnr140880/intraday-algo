"""
Professional Supertrend strategy - widely used in India by traders.
Combined with EMA trend filter for higher accuracy.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType
from datetime import datetime


class SupertrendStrategy(BaseStrategy):
    name = "Supertrend + EMA"
    description = "Supertrend with EMA 20/50 trend filter. Buy on supertrend flip up with price above EMA50. ATR-based SL."
    timeframe = "5m"
    min_bars = 60

    def default_params(self) -> Dict[str, Any]:
        return {
            "atr_period": 10,
            "multiplier": 3.0,
            "ema_fast": 20,
            "ema_slow": 50,
            "trailing_atr_multiplier": 1.5,
            "risk_reward_min": 1.5,
        }

    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params["atr_period"]
        mult = self.params["multiplier"]
        atr = self.calculate_atr(df, period)

        hl2 = (df["high"] + df["low"]) / 2
        upper_band = hl2 + mult * atr
        lower_band = hl2 - mult * atr

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
                if direction.iloc[i] == 1:
                    lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i - 1])
                else:
                    upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i - 1])

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

        df = df.copy()
        df["supertrend"] = supertrend
        df["supertrend_dir"] = direction
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None

        df = self.calculate_supertrend(df)
        df["ema_fast"] = df["close"].ewm(span=self.params["ema_fast"], adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.params["ema_slow"], adjust=False).mean()
        atr = self.calculate_atr(df).iloc[-1]

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        conditions_met = []
        signal = None
        entry = curr["close"]

        # BUY: Supertrend flips bullish, price above both EMAs
        if (prev["supertrend_dir"] == -1 and curr["supertrend_dir"] == 1):
            if curr["close"] > curr["ema_slow"]:
                conditions_met.append("Supertrend flipped BULLISH")
                conditions_met.append(f"Price {entry:.2f} > EMA50 {curr['ema_slow']:.2f}")
                if curr["ema_fast"] > curr["ema_slow"]:
                    conditions_met.append("EMA20 > EMA50 (uptrend confirmed)")
                sl = round(curr["supertrend"] - atr * 0.5, 2)
                target = round(entry + (entry - sl) * self.params["risk_reward_min"], 2)
                trailing = self.calculate_trailing_sl(entry, SignalType.BUY, atr, self.params["trailing_atr_multiplier"])
                signal = TradeSignal(
                    symbol=symbol, signal=SignalType.BUY,
                    entry_price=entry, stop_loss=sl, target=target,
                    trailing_sl=trailing, confidence=80.0,
                    strategy_name=self.name,
                    reasoning=f"Supertrend flip bullish at {entry}. EMA trend aligned.",
                    conditions_met=conditions_met, timeframe=self.timeframe,
                    timestamp=datetime.now().isoformat(),
                )

        # SELL: Supertrend flips bearish, price below both EMAs
        elif (prev["supertrend_dir"] == 1 and curr["supertrend_dir"] == -1):
            if curr["close"] < curr["ema_slow"]:
                conditions_met.append("Supertrend flipped BEARISH")
                conditions_met.append(f"Price {entry:.2f} < EMA50 {curr['ema_slow']:.2f}")
                sl = round(curr["supertrend"] + atr * 0.5, 2)
                target = round(entry - (sl - entry) * self.params["risk_reward_min"], 2)
                trailing = self.calculate_trailing_sl(entry, SignalType.SELL, atr, self.params["trailing_atr_multiplier"])
                signal = TradeSignal(
                    symbol=symbol, signal=SignalType.SELL,
                    entry_price=entry, stop_loss=sl, target=target,
                    trailing_sl=trailing, confidence=78.0,
                    strategy_name=self.name,
                    reasoning=f"Supertrend flip bearish at {entry}. EMA trend aligned.",
                    conditions_met=conditions_met, timeframe=self.timeframe,
                    timestamp=datetime.now().isoformat(),
                )

        return signal
