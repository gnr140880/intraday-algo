"""
Base strategy class - all strategies inherit from this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import pandas as pd


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


@dataclass
class TradeSignal:
    symbol: str
    signal: SignalType
    entry_price: float
    stop_loss: float
    target: float
    trailing_sl: Optional[float] = None
    quantity: int = 1
    confidence: float = 0.0         # 0-100
    strategy_name: str = ""
    reasoning: str = ""
    risk_reward: float = 0.0
    timeframe: str = "1m"
    timestamp: str = ""
    conditions_met: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.entry_price > 0 and self.stop_loss > 0 and self.target > 0:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.target - self.entry_price)
            if risk > 0:
                self.risk_reward = round(reward / risk, 2)


class BaseStrategy(ABC):
    name: str = "Base"
    description: str = ""
    timeframe: str = "1m"
    min_bars: int = 50

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or self.default_params()

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        pass

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def calculate_trailing_sl(
        self, entry: float, signal: SignalType, atr: float, multiplier: float = 2.0
    ) -> float:
        trail = atr * multiplier
        if signal == SignalType.BUY:
            return round(entry - trail, 2)
        return round(entry + trail, 2)
