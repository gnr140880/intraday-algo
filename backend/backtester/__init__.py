"""
Backtester Package

Provides historical backtesting for all strategies:
  - DataLoader: fetches/loads historical OHLCV data
  - BacktestEngine: replays candles through strategy + scoring + risk
  - BacktestResults: computes stats, equity curve, drawdown

Usage:
    from backtester import BacktestEngine, DataLoader
    loader = DataLoader()
    df = loader.load_nifty_history(days=90)
    engine = BacktestEngine(strategy="ORB", capital=1000000)
    results = engine.run(df)
    print(results.summary())
"""
from backtester.data_loader import DataLoader
from backtester.engine import BacktestEngine
from backtester.results import BacktestResults

__all__ = ["DataLoader", "BacktestEngine", "BacktestResults"]
