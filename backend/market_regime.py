"""
Market Regime Detector

Classifies the current market into one of these regimes:
  TRENDING_UP    – Strong directional move upward
  TRENDING_DOWN  – Strong directional move downward
  SIDEWAYS       – Range-bound, low directional momentum
  VOLATILE       – High volatility, choppy (avoid trading)
  BREAKOUT       – Price just broke out of a range (transitional)

Each regime determines WHICH family of strategies is prioritised:
  TRENDING    → ORB Breakout, VWAP Breakout, EMA Crossover, Supertrend
  SIDEWAYS    → Bollinger MR, ORB Range Scalper, VWAP MR, Straddle/Strangle
  VOLATILE    → Reduce size or skip trades entirely
  BREAKOUT    → ORB Breakout gets highest priority

Detection uses 5 indicators on the 5-min chart:
  1. ADX (Average Directional Index) → trend strength
  2. Bollinger Band Width % → volatility/squeeze
  3. ORB containment → is price inside or outside ORB range?
  4. Supertrend direction consistency → trending or whipsawing?
  5. Candle range vs ATR → are candles decisive or indecisive?
"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    BREAKOUT = "BREAKOUT"


# Strategy families mapped to regimes
REGIME_STRATEGY_MAP = {
    MarketRegime.TRENDING_UP: {
        "preferred": [
            "ORB Breakout", "VWAP Breakout", "EMA Crossover",
            "Supertrend", "PCR/OI Directional", "Gap Fill",
        ],
        "allowed": [
            "RSI Divergence",  # Can catch pullback entries in trend
        ],
        "blocked": [
            "Bollinger Mean Reversion", "ORB Range Scalper",
            "VWAP Mean Reversion",  # Mean reversion fights the trend
            "Straddle/Strangle Sell", "Iron Condor",  # Naked short risk in trend
        ],
    },
    MarketRegime.TRENDING_DOWN: {
        "preferred": [
            "ORB Breakout", "VWAP Breakout", "EMA Crossover",
            "Supertrend", "PCR/OI Directional", "Gap Fill",
        ],
        "allowed": [
            "RSI Divergence",
        ],
        "blocked": [
            "Bollinger Mean Reversion", "ORB Range Scalper",
            "VWAP Mean Reversion",
            "Straddle/Strangle Sell", "Iron Condor",
        ],
    },
    MarketRegime.SIDEWAYS: {
        "preferred": [
            "Bollinger Mean Reversion", "ORB Range Scalper",
            "VWAP Mean Reversion", "RSI Divergence",
            "Straddle/Strangle Sell", "Iron Condor",
            "Expiry Premium Sell",
        ],
        "allowed": [
            "ORB Breakout",  # Keep watching for breakout
        ],
        "blocked": [
            "VWAP Breakout", "EMA Crossover", "Supertrend",
            # Trend-following strategies whipsaw in sideways
        ],
    },
    MarketRegime.VOLATILE: {
        "preferred": [],  # Reduce trading in volatile regime
        "allowed": [
            "ORB Breakout",  # Only strong breakouts
        ],
        "blocked": [
            "Bollinger Mean Reversion", "ORB Range Scalper",
            "VWAP Mean Reversion", "EMA Crossover",
            "Straddle/Strangle Sell", "Iron Condor",
            # Everything except strong breakout is risky
        ],
    },
    MarketRegime.BREAKOUT: {
        "preferred": [
            "ORB Breakout", "VWAP Breakout", "Supertrend",
        ],
        "allowed": [
            "EMA Crossover", "PCR/OI Directional",
        ],
        "blocked": [
            "Bollinger Mean Reversion", "ORB Range Scalper",
            "VWAP Mean Reversion",
            "Straddle/Strangle Sell", "Iron Condor",
        ],
    },
}


@dataclass
class RegimeAnalysis:
    """Result of market regime detection."""
    regime: MarketRegime
    confidence: float           # 0-100 how confident in regime classification
    adx: float                  # ADX value (>25 = trending)
    bbw_pct: float              # Bollinger Band Width %
    orb_contained: bool         # Is price inside ORB range?
    st_flips: int               # Supertrend direction flips in last 20 bars
    avg_candle_ratio: float     # Avg body/range ratio (>0.6 = decisive)
    reasoning: str              # Human-readable explanation


def detect_regime(
    df: pd.DataFrame,
    orb_high: float = 0,
    orb_low: float = 0,
    india_vix: float = 0,
) -> RegimeAnalysis:
    """
    Detect current market regime from 5-min OHLCV data.

    Args:
        df: NIFTY 5-min OHLCV DataFrame (needs at least 30 bars)
        orb_high: ORB high (0 if not captured)
        orb_low: ORB low (0 if not captured)
        india_vix: India VIX reading (0 if unavailable)

    Returns:
        RegimeAnalysis with regime classification and supporting data
    """
    if len(df) < 25:
        return RegimeAnalysis(
            regime=MarketRegime.SIDEWAYS,
            confidence=30, adx=0, bbw_pct=0,
            orb_contained=True, st_flips=0,
            avg_candle_ratio=0,
            reasoning="Insufficient data, defaulting to SIDEWAYS",
        )

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ── 1. ADX (Average Directional Index) ──────────────────────
    adx_val = _compute_adx(high, low, close, period=14)

    # ── 2. Bollinger Band Width % ───────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bbw_pct = float(((2 * std20 / sma20) * 100).iloc[-1]) if sma20.iloc[-1] > 0 else 0

    # ── 3. ORB containment ──────────────────────────────────────
    orb_contained = True
    if orb_high > 0 and orb_low > 0:
        curr_price = float(close.iloc[-1])
        orb_contained = orb_low <= curr_price <= orb_high

    # ── 4. Supertrend direction flips ───────────────────────────
    st_flips = _count_supertrend_flips(df, lookback=20)

    # ── 5. Candle decisiveness ──────────────────────────────────
    body = (close - df["open"].astype(float)).abs()
    candle_range = (high - low).replace(0, 1e-10)
    body_ratio = body / candle_range
    avg_candle_ratio = float(body_ratio.iloc[-10:].mean())

    # ── Scoring ─────────────────────────────────────────────────
    trend_score = 0      # Positive = trending, Negative = sideways
    volatile_score = 0   # High = volatile/choppy

    # ADX scoring
    if adx_val > 30:
        trend_score += 3       # Strong trend
    elif adx_val > 25:
        trend_score += 2       # Moderate trend
    elif adx_val > 20:
        trend_score += 1       # Weak trend / transitional
    else:
        trend_score -= 2       # No trend = sideways

    # BBW scoring
    if bbw_pct < 1.0:
        trend_score -= 2       # Very tight bands = sideways squeeze
    elif bbw_pct < 2.0:
        trend_score -= 1       # Narrow bands
    elif bbw_pct > 4.0:
        volatile_score += 2    # Wide bands = volatile
    elif bbw_pct > 3.0:
        trend_score += 1       # Expanding bands = possible trend

    # ORB containment
    if orb_contained:
        trend_score -= 2       # Price inside ORB = sideways
    else:
        trend_score += 2       # Price broke ORB = breakout/trend

    # Supertrend flips
    if st_flips >= 4:
        volatile_score += 2    # Many flips = choppy
        trend_score -= 1
    elif st_flips >= 2:
        trend_score -= 1       # Some flips = indecisive
    elif st_flips <= 1:
        trend_score += 1       # Stable direction = trending

    # Candle decisiveness
    if avg_candle_ratio > 0.65:
        trend_score += 1       # Decisive candles = trending
    elif avg_candle_ratio < 0.35:
        trend_score -= 1       # Doji-like candles = sideways
        volatile_score += 1

    # VIX
    if india_vix > 22:
        volatile_score += 2
    elif india_vix > 18:
        volatile_score += 1

    # ── Regime classification ───────────────────────────────────
    reasoning_parts = []

    if volatile_score >= 4:
        regime = MarketRegime.VOLATILE
        confidence = min(85, 50 + volatile_score * 8)
        reasoning_parts.append(f"High volatility (VIX={india_vix:.1f}, BBW={bbw_pct:.1f}%, ST flips={st_flips})")
    elif trend_score >= 4:
        # Strong trend — determine direction
        # Use last 10 bars slope
        slope = float(close.iloc[-1] - close.iloc[-10]) if len(close) >= 10 else 0
        if slope > 0:
            regime = MarketRegime.TRENDING_UP
            reasoning_parts.append(f"Bullish trend (ADX={adx_val:.1f}, slope=+{slope:.1f})")
        else:
            regime = MarketRegime.TRENDING_DOWN
            reasoning_parts.append(f"Bearish trend (ADX={adx_val:.1f}, slope={slope:.1f})")
        confidence = min(85, 45 + trend_score * 7)
    elif trend_score >= 2 and not orb_contained:
        regime = MarketRegime.BREAKOUT
        confidence = min(75, 50 + trend_score * 5)
        reasoning_parts.append(f"Breakout in progress (ORB broken, ADX={adx_val:.1f})")
    else:
        regime = MarketRegime.SIDEWAYS
        confidence = min(85, 50 + abs(trend_score) * 5)
        reasoning_parts.append(f"Range-bound (ADX={adx_val:.1f}, BBW={bbw_pct:.1f}%, ORB contained={orb_contained})")

    reasoning_parts.append(f"Scores: trend={trend_score}, volatile={volatile_score}")

    return RegimeAnalysis(
        regime=regime,
        confidence=confidence,
        adx=round(adx_val, 1),
        bbw_pct=round(bbw_pct, 2),
        orb_contained=orb_contained,
        st_flips=st_flips,
        avg_candle_ratio=round(avg_candle_ratio, 3),
        reasoning=". ".join(reasoning_parts),
    )


def filter_signals_by_regime(
    all_signals: list,
    regime: RegimeAnalysis,
) -> list:
    """
    Filter and re-weight signals based on detected market regime.

    - Preferred strategies get confidence boost (+10)
    - Blocked strategies are removed
    - Allowed strategies pass through unchanged

    Returns filtered + re-weighted signal list.
    """
    mapping = REGIME_STRATEGY_MAP.get(regime.regime, {})
    preferred = set(mapping.get("preferred", []))
    blocked = set(mapping.get("blocked", []))

    filtered = []
    for sig in all_signals:
        strat_name = getattr(sig, "strategy_name", "")

        if strat_name in blocked:
            logger.debug(
                f"Signal BLOCKED by regime {regime.regime.value}: "
                f"{strat_name} {sig.signal.value} @ {sig.entry_price}"
            )
            continue

        # Clone confidence boost for preferred strategies
        if strat_name in preferred:
            # Boost confidence by regime confidence factor
            boost = min(10, regime.confidence / 10)
            sig.confidence = min(95, sig.confidence + boost)
            logger.debug(
                f"Signal BOOSTED by regime {regime.regime.value}: "
                f"{strat_name} +{boost:.0f} → {sig.confidence:.0f}%"
            )

        filtered.append(sig)

    logger.info(
        f"Regime filter: {regime.regime.value} (conf={regime.confidence:.0f}%). "
        f"{len(all_signals)} signals → {len(filtered)} after filter. "
        f"Blocked: {len(all_signals) - len(filtered)}"
    )
    return filtered


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Compute ADX (Average Directional Index)."""
    try:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, 1e-10))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
        adx = dx.ewm(span=period, adjust=False).mean()

        return float(adx.iloc[-1]) if not adx.isna().iloc[-1] else 0
    except Exception:
        return 0


def _count_supertrend_flips(df: pd.DataFrame, lookback: int = 20) -> int:
    """Count how many times Supertrend direction changed in last N bars."""
    if "st_dir" not in df.columns:
        return 0
    try:
        st = df["st_dir"].iloc[-lookback:].astype(int)
        flips = (st.diff().abs() > 0).sum()
        return int(flips)
    except Exception:
        return 0

