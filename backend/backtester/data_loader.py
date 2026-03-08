"""
Backtester Data Loader

Fetches and prepares historical data for backtesting.
Sources: Kite Historical API, yfinance fallback, CSV files.
"""
import logging
from datetime import datetime, timedelta, date
from typing import Optional, List
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare historical OHLCV data for backtesting."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "backtest_data"
        self.data_dir.mkdir(exist_ok=True)

    def load_nifty_history(
        self,
        days: int = 90,
        interval: str = "5minute",
        source: str = "yfinance",
    ) -> Optional[pd.DataFrame]:
        """
        Load NIFTY spot historical data.
        
        Args:
            days: number of trading days to load
            interval: candle interval (5minute, 15minute, hour, day)
            source: data source (yfinance, kite, csv)
        """
        if source == "yfinance":
            return self._load_from_yfinance(days, interval)
        elif source == "csv":
            return self._load_from_csv("NIFTY", interval)
        elif source == "kite":
            return self._load_from_kite(days, interval)
        return None

    def _load_from_yfinance(
        self, days: int, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from yfinance."""
        try:
            import yfinance as yf

            # yfinance interval mapping
            yf_interval_map = {
                "5minute": "5m",
                "15minute": "15m",
                "hour": "1h",
                "day": "1d",
                "5m": "5m",
                "15m": "15m",
                "1h": "1h",
                "1d": "1d",
            }
            yf_interval = yf_interval_map.get(interval, "5m")

            # yfinance max periods by interval
            if yf_interval in ("5m", "15m"):
                period_days = min(days, 59)  # yfinance 5m limit
            elif yf_interval == "1h":
                period_days = min(days, 729)
            else:
                period_days = days

            ticker = yf.Ticker("^NSEI")
            df = ticker.history(period=f"{period_days}d", interval=yf_interval)

            if df.empty:
                logger.warning("yfinance returned empty data for NIFTY")
                return None

            # Standardize column names
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })

            # Convert timezone-aware index to 'date' column
            df["date"] = df.index
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            df = df.reset_index(drop=True)

            # Keep only necessary columns
            cols = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]

            logger.info(f"Loaded {len(df)} candles from yfinance ({yf_interval}, {period_days}d)")
            return df

        except ImportError:
            logger.error("yfinance not installed. pip install yfinance")
            return None
        except Exception as e:
            logger.error(f"yfinance load failed: {e}")
            return None

    def _load_from_csv(
        self, symbol: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Load from local CSV file."""
        filename = f"{symbol}_{interval}.csv"
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"CSV file not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath, parse_dates=["date"])
            logger.info(f"Loaded {len(df)} candles from {filepath}")
            return df
        except Exception as e:
            logger.error(f"CSV load failed: {e}")
            return None

    def _load_from_kite(
        self, days: int, interval: str
    ) -> Optional[pd.DataFrame]:
        """Load from Kite API (requires active session)."""
        try:
            from kite_client import KiteClient
            kite = KiteClient.get_instance()
            if not kite.is_connected:
                logger.warning("Kite not connected for historical data")
                return None

            # Resolve NIFTY token
            instruments = kite.get_instruments("NSE")
            token = None
            for inst in instruments:
                if inst["tradingsymbol"] == "NIFTY 50":
                    token = inst["instrument_token"]
                    break

            if token is None:
                return None

            to_dt = datetime.now()
            from_dt = to_dt - timedelta(days=days)
            raw = kite.get_historical_data(token, from_dt, to_dt, interval)
            if raw:
                df = pd.DataFrame(raw)
                logger.info(f"Loaded {len(df)} candles from Kite ({interval}, {days}d)")
                return df
            return None
        except Exception as e:
            logger.error(f"Kite historical load failed: {e}")
            return None

    def save_to_csv(
        self, df: pd.DataFrame, symbol: str, interval: str
    ) -> str:
        """Save DataFrame to CSV for future use."""
        filename = f"{symbol}_{interval}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} candles to {filepath}")
        return str(filepath)

    def split_by_session(
        self, df: pd.DataFrame
    ) -> List[pd.DataFrame]:
        """
        Split a multi-day DataFrame into per-session DataFrames.
        Each session = 9:15 AM to 3:30 PM.
        """
        if "date" not in df.columns:
            return [df]

        df["session_date"] = pd.to_datetime(df["date"]).dt.date
        sessions = []
        for d in sorted(df["session_date"].unique()):
            session_df = df[df["session_date"] == d].copy()
            session_df = session_df.drop(columns=["session_date"])
            if len(session_df) >= 5:  # minimum bars
                sessions.append(session_df)

        return sessions
