"""
Backtester Results

Computes and stores backtesting statistics:
  - Win rate, profit factor, Sharpe ratio
  - Equity curve, drawdown analysis
  - Per-strategy breakdown
  - Trade-by-trade log
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single backtest trade record."""
    trade_id: str = ""
    session_date: str = ""
    strategy: str = ""
    direction: str = ""         # BUY / SELL
    symbol: str = "NIFTY"
    entry_price: float = 0
    exit_price: float = 0
    entry_bar: int = 0
    exit_bar: int = 0
    quantity: int = 1
    pnl: float = 0
    pnl_pct: float = 0
    sl: float = 0
    target: float = 0
    exit_reason: str = ""
    confidence: float = 0
    score: float = 0
    slippage: float = 0
    gap_type: str = "NONE"
    holding_bars: int = 0


@dataclass
class BacktestResults:
    """Complete backtest results with statistics."""
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    initial_capital: float = 1000000
    strategy_name: str = ""
    sessions_tested: int = 0
    start_date: str = ""
    end_date: str = ""

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> List[BacktestTrade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losers(self) -> List[BacktestTrade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return round(len(self.winners) / self.total_trades * 100, 1)

    @property
    def gross_pnl(self) -> float:
        return round(sum(t.pnl for t in self.trades), 2)

    @property
    def net_pnl(self) -> float:
        return round(sum(t.pnl - t.slippage for t in self.trades), 2)

    @property
    def avg_winner(self) -> float:
        if not self.winners:
            return 0
        return round(sum(t.pnl for t in self.winners) / len(self.winners), 2)

    @property
    def avg_loser(self) -> float:
        if not self.losers:
            return 0
        return round(abs(sum(t.pnl for t in self.losers)) / len(self.losers), 2)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.winners)
        gross_loss = abs(sum(t.pnl for t in self.losers))
        if gross_loss == 0:
            return 0 if gross_win == 0 else float("inf")
        return round(gross_win / gross_loss, 2)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0
        curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / peak * 100
        return round(abs(dd.min()), 2)

    @property
    def max_drawdown_amount(self) -> float:
        """Maximum drawdown in absolute ₹ terms."""
        if not self.equity_curve:
            return 0
        curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(curve)
        dd = curve - peak
        return round(abs(dd.min()), 2)

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming daily returns)."""
        if len(self.daily_pnl) < 2:
            return 0
        returns = list(self.daily_pnl.values())
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0
        # Annualize: sqrt(252) for trading days
        return round((mean_ret / std_ret) * np.sqrt(252), 2)

    @property
    def avg_holding_bars(self) -> float:
        if not self.trades:
            return 0
        return round(sum(t.holding_bars for t in self.trades) / len(self.trades), 1)

    @property
    def return_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0
        return round(self.net_pnl / self.initial_capital * 100, 2)

    def summary(self) -> Dict:
        """Human-readable summary of backtest results."""
        return {
            "strategy": self.strategy_name,
            "period": f"{self.start_date} to {self.end_date}",
            "sessions_tested": self.sessions_tested,
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate}%",
            "gross_pnl": f"₹{self.gross_pnl:,.2f}",
            "net_pnl": f"₹{self.net_pnl:,.2f}",
            "return_pct": f"{self.return_pct}%",
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": f"{self.max_drawdown}%",
            "max_drawdown_amount": f"₹{self.max_drawdown_amount:,.2f}",
            "avg_winner": f"₹{self.avg_winner:,.2f}",
            "avg_loser": f"₹{self.avg_loser:,.2f}",
            "avg_holding_bars": self.avg_holding_bars,
            "initial_capital": f"₹{self.initial_capital:,.2f}",
        }

    def per_strategy_breakdown(self) -> Dict[str, Dict]:
        """Break down results by strategy name."""
        strategies = {}
        for t in self.trades:
            s = t.strategy or "UNKNOWN"
            if s not in strategies:
                strategies[s] = {"trades": [], "pnl": 0}
            strategies[s]["trades"].append(t)
            strategies[s]["pnl"] += t.pnl

        result = {}
        for s, data in strategies.items():
            trades = data["trades"]
            wins = [t for t in trades if t.pnl > 0]
            result[s] = {
                "total_trades": len(trades),
                "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
                "gross_pnl": round(data["pnl"], 2),
                "avg_pnl": round(data["pnl"] / len(trades), 2) if trades else 0,
            }
        return result

    def to_dict(self) -> Dict:
        """Serialize for API response."""
        return {
            **self.summary(),
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "date": t.session_date,
                    "direction": t.direction,
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                    "strategy": t.strategy,
                    "gap_type": t.gap_type,
                }
                for t in self.trades
            ],
            "equity_curve": self.equity_curve,
            "daily_pnl": self.daily_pnl,
            "strategy_breakdown": self.per_strategy_breakdown(),
        }
