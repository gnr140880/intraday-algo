"""
Backtester Engine

Replays historical candles through the strategy pipeline:
  1. Per session: compute ORB, indicators, levels
  2. Generate signals using ORB / VWAP strategy
  3. Simulate entry at signal candle close (+ slippage)
  4. Track trailing SL, targets, time exit
  5. Record trade results

Supports:
  - Multiple strategies (ORB, VWAP MR, Expiry Sell)
  - Slippage model integration
  - Gap-day classification per session
  - Configurable risk parameters
"""
import logging
import uuid
from datetime import datetime, date, time, timedelta
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np

from strategies.nifty_options_orb import NiftyOptionsORBStrategy
from strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from strategies.supertrend_strategy import SupertrendStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.vwap_breakout import VWAPBreakoutStrategy
from strategies.gap_fill import GapFillStrategy
from strategies.base_strategy import SignalType
from level_calculator import LevelCalculator
from slippage_model import SlippageModel
from config import settings
from backtester.results import BacktestResults, BacktestTrade

logger = logging.getLogger(__name__)

# Strategy registry for backtesting
BACKTEST_STRATEGY_REGISTRY = {
    "ORB": "orb_strategy",
    "VWAP_MR": "vwap_strategy",
    "VWAP_BREAKOUT": "vwap_breakout_strategy",
    "SUPERTREND": "supertrend_strategy",
    "EMA_CROSSOVER": "ema_crossover_strategy",
    "RSI_DIVERGENCE": "rsi_divergence_strategy",
    "GAP_FILL": "gap_fill_strategy",
}


class BacktestEngine:
    """
    Historical backtesting engine for NIFTY intraday strategies.
    
    Usage:
        engine = BacktestEngine(strategy="ORB", capital=1000000)
        results = engine.run(sessions)  # list of per-day DataFrames
    """

    def __init__(
        self,
        strategy: str = "ORB",
        capital: float = 1000000,
        lot_size: int = 75,
        max_trades_per_day: int = 3,
        slippage_enabled: bool = True,
        slippage_pct: float = 0.05,
    ):
        self.strategy_name = strategy
        self.initial_capital = capital
        self.capital = capital
        self.lot_size = lot_size
        self.max_trades_per_day = max_trades_per_day

        # Strategy instances
        self.orb_strategy = NiftyOptionsORBStrategy()
        self.vwap_strategy = VWAPMeanReversionStrategy()
        self.supertrend_strategy = SupertrendStrategy()
        self.ema_crossover_strategy = EMACrossoverStrategy()
        self.rsi_divergence_strategy = RSIDivergenceStrategy()
        self.vwap_breakout_strategy = VWAPBreakoutStrategy()
        self.gap_fill_strategy = GapFillStrategy()
        self.level_calc = LevelCalculator()

        # Slippage
        self.slippage_model = SlippageModel(
            fixed_pct=slippage_pct,
            enabled=slippage_enabled,
        )

        # Tracking
        self._trades: List[BacktestTrade] = []
        self._equity_curve: List[float] = []
        self._daily_pnl: Dict[str, float] = {}

    def run(
        self,
        sessions: List[pd.DataFrame],
        progress_callback=None,
    ) -> BacktestResults:
        """
        Run backtest across multiple sessions (days).
        
        Args:
            sessions: list of per-day OHLCV DataFrames
            progress_callback: optional fn(session_idx, total) for progress
        
        Returns:
            BacktestResults with all stats
        """
        self._trades = []
        self._equity_curve = [self.initial_capital]
        self._daily_pnl = {}
        self.capital = self.initial_capital

        total_sessions = len(sessions)
        logger.info(f"Starting backtest: {self.strategy_name}, {total_sessions} sessions, ₹{self.capital:,.0f}")

        for idx, session_df in enumerate(sessions):
            if len(session_df) < 10:
                continue

            try:
                day_pnl = self._run_session(session_df)
                self.capital += day_pnl
                self._equity_curve.append(self.capital)

                # Record daily PnL
                session_date = self._get_session_date(session_df)
                if session_date:
                    self._daily_pnl[session_date] = day_pnl

            except Exception as e:
                logger.warning(f"Session {idx} failed: {e}")

            if progress_callback:
                progress_callback(idx + 1, total_sessions)

        # Build results
        results = BacktestResults(
            trades=self._trades,
            equity_curve=self._equity_curve,
            daily_pnl=self._daily_pnl,
            initial_capital=self.initial_capital,
            strategy_name=self.strategy_name,
            sessions_tested=total_sessions,
        )

        if sessions:
            results.start_date = self._get_session_date(sessions[0]) or ""
            results.end_date = self._get_session_date(sessions[-1]) or ""

        logger.info(
            f"Backtest complete: {results.total_trades} trades, "
            f"Win rate: {results.win_rate}%, PnL: ₹{results.net_pnl:,.2f}"
        )

        return results

    def _run_session(self, df: pd.DataFrame) -> float:
        """
        Run one trading session (day). Returns day PnL.
        
        Simulates candle-by-candle replay:
        1. First 3 candles (9:15-9:30): capture ORB
        2. Candle 4+: check for signals, manage positions
        3. Auto exit at 3:15 PM candle
        """
        session_date = self._get_session_date(df) or "unknown"
        day_trades: List[BacktestTrade] = []
        trades_today = 0
        day_pnl = 0.0

        # Compute indicators
        df = self.orb_strategy.compute_supertrend(df)
        df = self.orb_strategy.compute_macd(df)

        # ORB capture
        orb_high, orb_low = self.orb_strategy.compute_orb(df)
        if orb_high is None:
            return 0.0

        # Compute levels
        levels = self.level_calc.compute(df, orb_high=orb_high, orb_low=orb_low)

        # Gap classification
        gap_type = "NONE"
        gap_pct = 0.0
        if "date" in df.columns and len(df) >= 2:
            today_open = float(df.iloc[0]["open"])
            # Use first bar's open as today's open, and try to find prev close
            # In session-split data, we approximate from first bar
            # This is simplified for backtest
            pass

        atr = float(self.orb_strategy.calculate_atr(df).iloc[-1]) if len(df) > 14 else 0

        # Active position tracking
        active_trade: Optional[Dict] = None

        # Replay candles from bar index 3 onwards (after ORB window)
        orb_bars = 3  # 9:15, 9:20, 9:25 → ORB complete at index 3
        for i in range(orb_bars, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            price = float(curr["close"])
            high = float(curr["high"])
            low = float(curr["low"])

            # Check time-based exit (3:15 PM)
            if "date" in df.columns:
                bar_time = pd.to_datetime(curr["date"])
                if hasattr(bar_time, 'time'):
                    if bar_time.time() >= time(15, 15):
                        if active_trade:
                            # Force exit
                            exit_price = price
                            pnl = self._calc_pnl(active_trade, exit_price)
                            trade = self._close_trade(
                                active_trade, exit_price, "TIME_EXIT",
                                i, session_date, pnl
                            )
                            day_trades.append(trade)
                            day_pnl += pnl
                            active_trade = None
                        break

            # Manage active position
            if active_trade:
                direction = active_trade["direction"]
                entry = active_trade["entry_price"]
                sl = active_trade["sl"]
                t1 = active_trade["target"]

                # Check SL hit
                if direction == "BUY" and low <= sl:
                    exit_price = sl
                    pnl = self._calc_pnl(active_trade, exit_price)
                    trade = self._close_trade(
                        active_trade, exit_price, "SL_HIT",
                        i, session_date, pnl
                    )
                    day_trades.append(trade)
                    day_pnl += pnl
                    active_trade = None
                    continue
                elif direction == "SELL" and high >= sl:
                    exit_price = sl
                    pnl = self._calc_pnl(active_trade, exit_price)
                    trade = self._close_trade(
                        active_trade, exit_price, "SL_HIT",
                        i, session_date, pnl
                    )
                    day_trades.append(trade)
                    day_pnl += pnl
                    active_trade = None
                    continue

                # Check target hit
                if direction == "BUY" and high >= t1:
                    exit_price = t1
                    pnl = self._calc_pnl(active_trade, exit_price)
                    trade = self._close_trade(
                        active_trade, exit_price, "TARGET1_HIT",
                        i, session_date, pnl
                    )
                    day_trades.append(trade)
                    day_pnl += pnl
                    active_trade = None
                    continue
                elif direction == "SELL" and low <= t1:
                    exit_price = t1
                    pnl = self._calc_pnl(active_trade, exit_price)
                    trade = self._close_trade(
                        active_trade, exit_price, "TARGET1_HIT",
                        i, session_date, pnl
                    )
                    day_trades.append(trade)
                    day_pnl += pnl
                    active_trade = None
                    continue

                # Update trailing SL (simple: move to breakeven after 50% of target)
                if direction == "BUY":
                    half_target = entry + (t1 - entry) * 0.5
                    if price >= half_target and sl < entry:
                        active_trade["sl"] = entry + (price - entry) * 0.05  # small above BE
                elif direction == "SELL":
                    half_target = entry - (entry - t1) * 0.5
                    if price <= half_target and sl > entry:
                        active_trade["sl"] = entry - (entry - price) * 0.05

                continue  # Position is open, skip signal check

            # No active position → check for new signal
            if trades_today >= self.max_trades_per_day:
                continue

            # Build a mini-DataFrame up to current bar for signal generation
            df_slice = df.iloc[:i + 1].copy()

            signal = None
            strategies_to_run = []
            if self.strategy_name == "ALL":
                strategies_to_run = list(BACKTEST_STRATEGY_REGISTRY.keys())
            elif self.strategy_name in BACKTEST_STRATEGY_REGISTRY:
                strategies_to_run = [self.strategy_name]
            else:
                strategies_to_run = [self.strategy_name]

            for strat_name in strategies_to_run:
                if signal is not None:
                    break
                try:
                    if strat_name == "ORB":
                        signal = self.orb_strategy.generate_signal(df_slice, "NIFTY")
                    elif strat_name == "VWAP_MR":
                        signal = self.vwap_strategy.generate_signal(
                            df_slice, "NIFTY", levels=levels,
                            orb_high=orb_high, orb_low=orb_low,
                            india_vix=0, gap_type=gap_type)
                    elif strat_name == "VWAP_BREAKOUT":
                        signal = self.vwap_breakout_strategy.generate_signal(
                            df_slice, "NIFTY", orb_high=orb_high, orb_low=orb_low)
                    elif strat_name == "SUPERTREND":
                        signal = self.supertrend_strategy.generate_signal(df_slice, "NIFTY")
                    elif strat_name == "EMA_CROSSOVER":
                        signal = self.ema_crossover_strategy.generate_signal(df_slice, "NIFTY")
                    elif strat_name == "RSI_DIVERGENCE":
                        signal = self.rsi_divergence_strategy.generate_signal(df_slice, "NIFTY")
                    elif strat_name == "GAP_FILL":
                        pdc = levels.pdc if levels else 0
                        signal = self.gap_fill_strategy.generate_signal(
                            df_slice, "NIFTY", gap_type=gap_type,
                            gap_pct=gap_pct, prev_close=pdc)
                except Exception as e:
                    logger.debug(f"Backtest {strat_name} signal error: {e}")

            if signal and signal.confidence >= settings.min_confidence_threshold:
                # Apply slippage to entry
                entry_price = self.slippage_model.estimate_fill_price(
                    signal.entry_price, side="BUY" if signal.signal == SignalType.BUY else "SELL"
                )
                slippage_amount = abs(entry_price - signal.entry_price)

                active_trade = {
                    "trade_id": str(uuid.uuid4())[:8],
                    "direction": signal.signal.value,
                    "entry_price": entry_price,
                    "sl": signal.stop_loss,
                    "target": signal.target,
                    "confidence": signal.confidence,
                    "strategy": signal.strategy_name,
                    "entry_bar": i,
                    "quantity": self.lot_size,
                    "slippage": slippage_amount,
                    "gap_type": gap_type,
                }
                trades_today += 1

        # Close any remaining position at session end
        if active_trade:
            exit_price = float(df.iloc[-1]["close"])
            pnl = self._calc_pnl(active_trade, exit_price)
            trade = self._close_trade(
                active_trade, exit_price, "SESSION_END",
                len(df) - 1, session_date, pnl
            )
            day_trades.append(trade)
            day_pnl += pnl

        self._trades.extend(day_trades)
        return day_pnl

    def _calc_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate PnL for a trade."""
        entry = trade["entry_price"]
        qty = trade["quantity"]
        if trade["direction"] == "BUY":
            return (exit_price - entry) * qty
        else:
            return (entry - exit_price) * qty

    def _close_trade(
        self,
        trade: Dict,
        exit_price: float,
        exit_reason: str,
        exit_bar: int,
        session_date: str,
        pnl: float,
    ) -> BacktestTrade:
        """Create a BacktestTrade from an active trade dict."""
        entry = trade["entry_price"]
        pnl_pct = (pnl / (abs(entry) * trade["quantity"]) * 100) if entry > 0 else 0

        return BacktestTrade(
            trade_id=trade["trade_id"],
            session_date=session_date,
            strategy=trade.get("strategy", self.strategy_name),
            direction=trade["direction"],
            symbol="NIFTY",
            entry_price=entry,
            exit_price=exit_price,
            entry_bar=trade["entry_bar"],
            exit_bar=exit_bar,
            quantity=trade["quantity"],
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            sl=trade["sl"],
            target=trade["target"],
            exit_reason=exit_reason,
            confidence=trade.get("confidence", 0),
            slippage=trade.get("slippage", 0),
            gap_type=trade.get("gap_type", "NONE"),
            holding_bars=exit_bar - trade["entry_bar"],
        )

    @staticmethod
    def _get_session_date(df: pd.DataFrame) -> Optional[str]:
        """Extract session date string from DataFrame."""
        if "date" in df.columns and len(df) > 0:
            dt = pd.to_datetime(df.iloc[0]["date"])
            return dt.strftime("%Y-%m-%d")
        return None
