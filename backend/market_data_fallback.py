"""
Fallback Market Data Provider

Uses free data sources (yfinance for NIFTY spot/history, Kite instruments for
option chain metadata) when Kite API lacks quote/historical permissions.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from functools import lru_cache

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yfinance-based spot data
# ---------------------------------------------------------------------------

def fetch_nifty_spot_yf() -> float:
    """Get the last available NIFTY 50 closing price via yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(period="1d", interval="1m")
        if hist.empty:
            # Try longer period
            hist = ticker.history(period="5d", interval="5m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"yfinance spot fetch failed: {e}")
    return 0.0


def fetch_nifty_history_yf(days: int = 5, interval: str = "5m") -> Optional[pd.DataFrame]:
    """
    Fetch NIFTY 50 intraday OHLCV via yfinance.
    Returns DataFrame with columns: date, open, high, low, close, volume
    matching the format expected by the ORB strategy.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(period=f"{days}d", interval=interval)
        if hist.empty:
            return None
        # Normalise column names to match Kite format
        df = hist.reset_index()
        rename_map = {
            "Datetime": "date",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0
        # Strip timezone info so comparisons with naive datetimes work
        if "date" in df.columns:
            if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
        return df
    except Exception as e:
        logger.warning(f"yfinance history fetch failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Option chain builder (Kite instruments + Black-Scholes estimation)
# ---------------------------------------------------------------------------

def _estimate_delta(spot: float, strike: float, tte_days: float,
                    iv: float, option_type: str) -> float:
    """Simplified Black-Scholes delta."""
    if tte_days <= 0 or iv <= 0:
        if option_type == "CE":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0
    from scipy.stats import norm
    T = tte_days / 365
    r = 0.06
    d1 = (math.log(spot / strike) + (r + iv ** 2 / 2) * T) / (iv * math.sqrt(T))
    if option_type == "CE":
        return round(norm.cdf(d1), 4)
    else:
        return round(norm.cdf(d1) - 1, 4)


def _estimate_option_price(spot: float, strike: float, tte_days: float,
                           iv: float, option_type: str) -> float:
    """Simplified Black-Scholes option premium estimate."""
    if tte_days <= 0:
        # Intrinsic value only
        if option_type == "CE":
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)
    from scipy.stats import norm
    T = tte_days / 365
    r = 0.06
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + iv ** 2 / 2) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    if option_type == "CE":
        price = spot * norm.cdf(d1) - strike * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = strike * math.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return round(max(0, price), 2)


def build_option_candidates_from_instruments(
    kite_client,
    spot: float,
    signal_dir: int,
    orb_high: float,
    orb_low: float,
    st_dir: int,
    macd_hist: float,
    macd_hist_prev: float,
    atr_val: float,
    default_iv: float = 0.15,
    strike_gap: int = 50,
) -> List[Dict]:
    """
    Build OptionCandidate-compatible dicts using Kite instruments metadata
    + Black-Scholes estimated prices/delta.  No quote API needed.
    """
    from scoring_engine import OptionCandidate

    candidates: List[OptionCandidate] = []
    try:
        instruments = kite_client.get_instruments("NFO")
    except Exception as e:
        logger.error(f"Fallback: cannot fetch NFO instruments: {e}")
        return candidates

    today = date.today()
    nifty_options = [
        i for i in instruments
        if i.get("name") == "NIFTY"
        and i.get("segment") == "NFO-OPT"
        and i.get("expiry")
        and i["expiry"] >= today
    ]
    if not nifty_options:
        return candidates

    nifty_options.sort(key=lambda x: x["expiry"])
    nearest_expiry = nifty_options[0]["expiry"]
    weekly_opts = [i for i in nifty_options if i["expiry"] == nearest_expiry]

    atm_strike = round(spot / strike_gap) * strike_gap
    strike_range = range(
        int(atm_strike - 5 * strike_gap),
        int(atm_strike + 6 * strike_gap),
        strike_gap,
    )

    primary_type = "CE" if signal_dir == 1 else "PE"
    relevant = [
        i for i in weekly_opts
        if i["strike"] in strike_range and i["instrument_type"] == primary_type
    ]

    tte_days = (nearest_expiry - today).days
    if tte_days < 1:
        tte_days = 0.5

    for inst in relevant:
        iv = default_iv
        delta = _estimate_delta(spot, inst["strike"], tte_days, iv, primary_type)

        # Delta filter: 0.3 – 0.6
        if abs(delta) < 0.30 or abs(delta) > 0.60:
            continue

        est_price = _estimate_option_price(spot, inst["strike"], tte_days, iv, primary_type)
        if est_price <= 0:
            continue

        # Estimated volume/OI from moneyness (synthetic – better than nothing)
        moneyness = abs(spot - inst["strike"]) / spot
        est_volume = int(max(5000, 50000 * (1 - moneyness * 10)))
        est_oi = est_volume * 3

        c = OptionCandidate(
            tradingsymbol=inst["tradingsymbol"],
            instrument_token=inst["instrument_token"],
            strike=inst["strike"],
            option_type=primary_type,
            expiry=str(nearest_expiry),
            ltp=est_price,
            spot_price=spot,
            delta=delta,
            iv=round(iv * 100, 2),
            volume=est_volume,
            oi=est_oi,
            bid=round(est_price * 0.98, 2),
            ask=round(est_price * 1.02, 2),
            orb_high=orb_high,
            orb_low=orb_low,
            supertrend_dir=st_dir,
            macd_hist=macd_hist,
            macd_hist_prev=macd_hist_prev,
            vol_spike=est_volume > 10000,
            atr=atr_val,
        )
        candidates.append(c)

    logger.info(f"Fallback: built {len(candidates)} candidates (estimated, delta 0.3–0.6)")
    return candidates
