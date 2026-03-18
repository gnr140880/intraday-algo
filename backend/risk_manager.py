"""
Risk Manager

- Daily P&L tracking with 2% loss limit
- Auto square-off at 3:15 PM IST
- Per-trade risk checks
- Position sizing based on capital
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, time, date

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    option_type: str            # CE / PE
    entry_price: float
    entry_time: str
    quantity: int
    sl: float
    target: float
    trailing_sl: float
    score: float
    status: str = "OPEN"        # OPEN, CLOSED, SL_HIT, TARGET_HIT, SQUARED_OFF
    exit_price: float = 0.0
    exit_time: str = ""
    pnl: float = 0.0
    exit_reason: str = ""


class RiskManager:
    """
    Enforces:
      - Max 2% daily loss on capital
      - Auto square-off at 3:15 PM IST
      - Per-trade position sizing
      - Max concurrent positions
    """

    def __init__(
        self,
        capital: float = 1_000_000.0,
        daily_loss_limit_pct: float = 2.0,
        max_risk_per_trade_pct: float = 1.0,
        max_concurrent_positions: int = 5,
        square_off_time: time = time(15, 15),
    ):
        self.capital = capital
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_concurrent = max_concurrent_positions
        self.square_off_time = square_off_time

        self.daily_pnl: float = 0.0
        self.realised_pnl: float = 0.0
        self.unrealised_pnl: float = 0.0
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}
        self._trading_halted: bool = False
        self._last_reset_date: Optional[date] = None

        # Equity curve tracking
        self._equity_curve: List[Dict] = []  # [{"time": iso, "equity": float, "pnl": float}]
        self._trade_results: List[Dict] = []  # Recent trade results for adaptive sizing

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------
    def reset_daily(self):
        """Call at start of each trading day."""
        today = date.today()
        if self._last_reset_date == today:
            return
        self._last_reset_date = today
        self.daily_pnl = 0.0
        self.realised_pnl = 0.0
        self.unrealised_pnl = 0.0
        self._trading_halted = False
        self.open_positions.clear()
        logger.info(f"Risk manager reset for {today}. Capital: {self.capital}")

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------
    @property
    def daily_loss_limit(self) -> float:
        return self.capital * self.daily_loss_limit_pct / 100

    @property
    def max_risk_per_trade(self) -> float:
        return self.capital * self.max_risk_per_trade_pct / 100

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    def check_daily_limit(self) -> bool:
        """Returns True if we can keep trading; False if halted."""
        if self.daily_pnl <= -self.daily_loss_limit:
            self._trading_halted = True
            logger.warning(
                f"DAILY LOSS LIMIT HIT: {self.daily_pnl:.2f} "
                f"(limit: -{self.daily_loss_limit:.2f}). Trading halted."
            )
            return False
        return True

    def should_square_off(self) -> bool:
        """Returns True if current time >= square-off time."""
        now = datetime.now().time()
        return now >= self.square_off_time

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_quantity(
        self, entry_price: float, stop_loss: float, lot_size: int = 75
    ) -> int:
        """
        Calculate quantity based on max risk per trade and lot size.
        NIFTY lot size = 75 (as of 2025). Caller should pass settings.nifty_lot_size.
        """
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0
        max_qty = int(self.max_risk_per_trade / risk_per_unit)
        # Round down to nearest lot
        lots = max(1, max_qty // lot_size)
        return lots * lot_size

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------
    def can_take_trade(self, risk_amount: float) -> Dict:
        """Full pre-trade risk check."""
        self.reset_daily()
        reasons = []

        if self._trading_halted:
            reasons.append("Daily loss limit reached – trading halted")

        if not self.check_daily_limit():
            reasons.append(f"Daily P&L {self.daily_pnl:.2f} hit limit -{self.daily_loss_limit:.2f}")

        if self.should_square_off():
            reasons.append(f"Past square-off time {self.square_off_time}")

        if len(self.open_positions) >= self.max_concurrent:
            reasons.append(f"Max concurrent positions ({self.max_concurrent}) reached")

        if risk_amount > self.max_risk_per_trade:
            reasons.append(
                f"Risk {risk_amount:.2f} exceeds per-trade limit {self.max_risk_per_trade:.2f}"
            )

        allowed = len(reasons) == 0
        return {"allowed": allowed, "reasons": reasons}

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------
    def register_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        self.open_positions[trade.trade_id] = trade
        logger.info(
            f"Trade registered: {trade.trade_id} {trade.symbol} "
            f"qty={trade.quantity} entry={trade.entry_price}"
        )

    def close_trade(
        self, trade_id: str, exit_price: float, reason: str = "MANUAL"
    ) -> Optional[TradeRecord]:
        trade = self.open_positions.pop(trade_id, None)
        if trade is None:
            return None

        # Both CE and PE are BOUGHT (long options):
        # P&L = (exit_price - entry_price) * quantity
        trade.pnl = (exit_price - trade.entry_price) * trade.quantity

        trade.exit_price = exit_price
        trade.exit_time = datetime.now().isoformat()
        trade.exit_reason = reason
        trade.status = reason if reason in ("SL_HIT", "TARGET_HIT", "SQUARED_OFF") else "CLOSED"

        self.realised_pnl += trade.pnl
        self.daily_pnl = self.realised_pnl + self.unrealised_pnl

        # Track for equity curve and adaptive sizing
        self._equity_curve.append({
            "time": trade.exit_time,
            "equity": self.capital + self.realised_pnl,
            "pnl": trade.pnl,
            "daily_pnl": self.daily_pnl,
        })
        self._trade_results.append({
            "trade_id": trade_id,
            "pnl": trade.pnl,
            "win": trade.pnl > 0,
            "time": trade.exit_time,
        })

        logger.info(
            f"Trade closed: {trade_id} exit={exit_price} pnl={trade.pnl:.2f} "
            f"reason={reason} daily_pnl={self.daily_pnl:.2f}"
        )
        self.check_daily_limit()
        return trade

    def update_unrealised(self, positions: Dict[str, float]):
        """
        positions: {trade_id: current_ltp}
        Recalculates unrealised P&L.
        """
        unrealised = 0.0
        for tid, ltp in positions.items():
            trade = self.open_positions.get(tid)
            if trade:
                # Both CE and PE are BOUGHT (long options)
                unrealised += (ltp - trade.entry_price) * trade.quantity
        self.unrealised_pnl = unrealised
        self.daily_pnl = self.realised_pnl + self.unrealised_pnl

    def get_square_off_list(self) -> List[TradeRecord]:
        """Return all open positions that need square-off."""
        return list(self.open_positions.values())

    # ------------------------------------------------------------------
    # Trailing stop-loss update
    # ------------------------------------------------------------------
    def update_trailing_sl(self, trade_id: str, current_price: float, atr: float):
        """Legacy trailing SL update (use TrailingSLManager for the new system).
        For BOUGHT options (both CE & PE), trail upward only."""
        trade = self.open_positions.get(trade_id)
        if trade is None:
            return

        # Both CE and PE are bought → trail SL upward as premium rises
        new_sl = current_price - atr * 1.5
        if new_sl > trade.trailing_sl:
            trade.trailing_sl = round(new_sl, 2)

    def check_sl_target(self, trade_id: str, current_price: float) -> Optional[str]:
        """Check if SL or target hit. Returns action or None.
        For BOUGHT options: SL hit when price drops, target hit when price rises."""
        trade = self.open_positions.get(trade_id)
        if trade is None:
            return None

        # Both CE and PE are bought → SL when premium drops, target when premium rises
        if current_price <= trade.trailing_sl:
            return "SL_HIT"
        if current_price >= trade.target:
            return "TARGET_HIT"
        return None

    # ------------------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        return {
            "capital": self.capital,
            "daily_pnl": round(self.daily_pnl, 2),
            "realised_pnl": round(self.realised_pnl, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "daily_loss_limit": round(self.daily_loss_limit, 2),
            "loss_pct": round(self.daily_pnl / self.capital * 100, 2) if self.capital > 0 else 0,
            "trading_halted": self._trading_halted,
            "open_positions": len(self.open_positions),
            "total_trades_today": len([t for t in self.trades if t.entry_time[:10] == date.today().isoformat()]),
            "square_off_time": self.square_off_time.strftime("%H:%M"),
            "timestamp": datetime.now().isoformat(),
        }

    def get_trades_summary(self) -> List[Dict]:
        return [
            {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "type": t.option_type,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "qty": t.quantity,
                "pnl": round(t.pnl, 2),
                "score": t.score,
                "status": t.status,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "reason": t.exit_reason,
            }
            for t in self.trades
        ]

    # ------------------------------------------------------------------
    # Adaptive position sizing
    # ------------------------------------------------------------------
    def get_adaptive_size_multiplier(
        self,
        lookback: int = 5,
        min_win_rate: float = 40.0,
        max_win_rate: float = 70.0,
        reduce_factor: float = 0.5,
        increase_factor: float = 1.25,
        drawdown_halt_pct: float = 3.0,
    ) -> float:
        """
        Returns a multiplier (0.0 to 1.25) for position sizing based on
        recent trade performance.

        - Win rate < min_win_rate → reduce by reduce_factor
        - Win rate > max_win_rate → increase by increase_factor
        - Daily drawdown > drawdown_halt_pct → return 0 (halt)
        - Otherwise → 1.0 (normal)
        """
        # Check daily drawdown
        if self.capital > 0:
            dd_pct = abs(min(0, self.daily_pnl)) / self.capital * 100
            if dd_pct >= drawdown_halt_pct:
                logger.warning(f"Adaptive sizing: daily drawdown {dd_pct:.1f}% >= {drawdown_halt_pct}% — halting")
                return 0.0

        # Check recent win rate
        recent = self._trade_results[-lookback:] if self._trade_results else []
        if len(recent) < 3:
            return 1.0  # Not enough data

        wins = sum(1 for t in recent if t["win"])
        win_rate = wins / len(recent) * 100

        if win_rate < min_win_rate:
            logger.info(f"Adaptive sizing: win rate {win_rate:.0f}% < {min_win_rate}% → reducing to {reduce_factor}x")
            return reduce_factor
        elif win_rate > max_win_rate:
            logger.info(f"Adaptive sizing: win rate {win_rate:.0f}% > {max_win_rate}% → increasing to {increase_factor}x")
            return increase_factor
        return 1.0

    def get_equity_curve(self) -> List[Dict]:
        """Return the equity curve data for dashboard."""
        return self._equity_curve

    def get_recent_trade_stats(self, lookback: int = 20) -> Dict:
        """Return recent trade statistics."""
        recent = self._trade_results[-lookback:] if self._trade_results else []
        if not recent:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "avg_pnl": 0}
        wins = sum(1 for t in recent if t["win"])
        losses = len(recent) - wins
        avg_pnl = sum(t["pnl"] for t in recent) / len(recent)
        return {
            "total": len(recent),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(recent) * 100, 1),
            "avg_pnl": round(avg_pnl, 2),
        }
