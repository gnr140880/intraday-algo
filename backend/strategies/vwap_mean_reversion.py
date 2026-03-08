"""
VWAP Mean Reversion Strategy

For range-bound / low-volatility days:
  - BUY when price touches VWAP lower band (-1σ) and bounces
  - SELL when price touches VWAP upper band (+1σ) and rejects
  - Requires: narrow ORB range, VIX < 18, no gap day

Entry rules:
  - Price < VWAP lower-1σ for BUY (or > VWAP upper-1σ for SELL)
  - Last candle shows reversal (close back toward VWAP)
  - Supertrend is neutral or aligned
  - MACD histogram turning (bullish divergence for BUY / bearish for SELL)
  - Volume above 20-bar average

SL:  Below VWAP lower-2σ (BUY) or above VWAP upper-2σ (SELL)
T1:  VWAP (mean reversion target)
T2:  Opposite band (VWAP upper-1σ for BUY / VWAP lower-1σ for SELL)

Disabled by default — enable via settings.enable_vwap_strategy.
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, time

import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType
from config import settings
from level_calculator import LevelCalculator, IntraDayLevels

logger = logging.getLogger(__name__)


class VWAPMeanReversionStrategy(BaseStrategy):
    name = "VWAP Mean Reversion"
    description = (
        "Buys at VWAP lower band, sells at VWAP upper band on range-bound days. "
        "Targets reversion to VWAP (mean). Suitable for low-VIX, non-gap days."
    )
    timeframe = "5m"
    min_bars = 30

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.level_calc = LevelCalculator()

    def default_params(self) -> Dict[str, Any]:
        return {
            "orb_range_max_pct": 0.5,    # ORB range < 0.5% of spot = range-bound
            "vix_max": 18.0,             # Only active when VIX < 18
            "min_bounce_pct": 0.05,      # Price must bounce at least 0.05% from band
            "vol_lookback": 20,
            "vol_spike_mult": 1.0,       # Above average volume (not spike)
            "macd_turn_threshold": 0.0,  # MACD hist turning positive for BUY
        }

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        levels: Optional[IntraDayLevels] = None,
        orb_high: float = 0,
        orb_low: float = 0,
        india_vix: float = 0,
        gap_type: str = "NONE",
    ) -> Optional[TradeSignal]:
        """
        Generate VWAP mean reversion signal.

        Args:
            df: NIFTY 5-min OHLCV DataFrame
            symbol: instrument name
            levels: pre-computed IntraDayLevels (with VWAP bands)
            orb_high/orb_low: morning ORB range
            india_vix: current India VIX reading
            gap_type: gap-day classification
        """
        if not settings.enable_vwap_strategy:
            return None

        if len(df) < self.min_bars:
            return None

        # --- Pre-conditions: range-bound day ---
        spot = float(df.iloc[-1]["close"])
        if spot <= 0:
            return None

        # Gap day → skip mean reversion (trend day)
        if gap_type != "NONE":
            logger.debug("VWAP MR: skipping — gap day detected")
            return None

        # VIX must be low for mean reversion
        if india_vix > self.params.get("vix_max", 18.0) and india_vix > 0:
            logger.debug(f"VWAP MR: skipping — VIX {india_vix:.1f} > {self.params['vix_max']}")
            return None

        # ORB range must be narrow (range-bound market)
        if orb_high > 0 and orb_low > 0 and spot > 0:
            orb_range_pct = ((orb_high - orb_low) / spot) * 100
            max_orb_pct = settings.vwap_range_bound_max_orb_pct
            if orb_range_pct > max_orb_pct:
                logger.debug(
                    f"VWAP MR: skipping — ORB range {orb_range_pct:.2f}% > {max_orb_pct}% (trend day)"
                )
                return None

        # --- VWAP bands required ---
        if levels is None or levels.vwap <= 0:
            logger.debug("VWAP MR: no VWAP levels available")
            return None

        vwap = levels.vwap
        upper_1 = levels.vwap_upper_1
        lower_1 = levels.vwap_lower_1
        upper_2 = levels.vwap_upper_2
        lower_2 = levels.vwap_lower_2

        if upper_1 <= 0 or lower_1 <= 0:
            return None

        # --- Current bar analysis ---
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(curr["close"])
        prev_close = float(prev["close"])

        # --- Indicator computation ---
        from strategies.nifty_options_orb import NiftyOptionsORBStrategy
        orb_strat = NiftyOptionsORBStrategy()
        df_ind = orb_strat.compute_supertrend(df)
        df_ind = orb_strat.compute_macd(df_ind)

        macd_hist = float(df_ind.iloc[-1].get("macd_hist", 0))
        macd_hist_prev = float(df_ind.iloc[-2].get("macd_hist", 0))
        st_dir = int(df_ind.iloc[-1].get("st_dir", 0))

        atr = float(self.calculate_atr(df).iloc[-1])

        # --- BUY signal: price at/below VWAP lower-1σ and bouncing back ---
        conditions_met = []
        signal = None

        if price <= lower_1 or prev_close <= lower_1:
            # Check bounce: current close above prev low
            if price > float(curr["low"]):
                conditions_met.append(f"Price ₹{price:.2f} at/below VWAP-1σ ₹{lower_1:.2f}")

                # MACD turning bullish
                if macd_hist > macd_hist_prev:
                    conditions_met.append("MACD histogram turning bullish")

                # Volume above average
                if "volume" in df.columns and len(df) > 20:
                    avg_vol = df["volume"].iloc[-21:-1].mean()
                    if float(curr.get("volume", 0)) >= avg_vol:
                        conditions_met.append("Volume above 20-bar average")

                if len(conditions_met) >= 2:
                    sl = lower_2 if lower_2 > 0 else price - 2 * atr
                    t1 = vwap
                    t2 = upper_1

                    risk = abs(price - sl)
                    reward = abs(t1 - price)
                    confidence = min(75.0, 40 + len(conditions_met) * 10)

                    # Supertrend alignment bonus
                    if st_dir == 1:
                        confidence += 5
                        conditions_met.append("Supertrend bullish (aligned)")

                    signal = TradeSignal(
                        symbol=symbol,
                        signal=SignalType.BUY,
                        entry_price=round(price, 2),
                        stop_loss=round(sl, 2),
                        target=round(t1, 2),
                        trailing_sl=round(price - atr, 2),
                        confidence=confidence,
                        strategy_name=self.name,
                        reasoning=(
                            f"VWAP mean reversion BUY: price at lower band "
                            f"(VWAP-1σ=₹{lower_1:.2f}), targeting VWAP=₹{vwap:.2f}. "
                            f"R:R={reward/risk:.1f}x"
                            if risk > 0 else "VWAP MR BUY"
                        ),
                        conditions_met=conditions_met,
                        timeframe=self.timeframe,
                        timestamp=datetime.now().isoformat(),
                    )

        # --- SELL signal: price at/above VWAP upper-1σ and rejecting ---
        elif price >= upper_1 or prev_close >= upper_1:
            if price < float(curr["high"]):
                conditions_met.append(f"Price ₹{price:.2f} at/above VWAP+1σ ₹{upper_1:.2f}")

                # MACD turning bearish
                if macd_hist < macd_hist_prev:
                    conditions_met.append("MACD histogram turning bearish")

                # Volume above average
                if "volume" in df.columns and len(df) > 20:
                    avg_vol = df["volume"].iloc[-21:-1].mean()
                    if float(curr.get("volume", 0)) >= avg_vol:
                        conditions_met.append("Volume above 20-bar average")

                if len(conditions_met) >= 2:
                    sl = upper_2 if upper_2 > 0 else price + 2 * atr
                    t1 = vwap
                    t2 = lower_1

                    risk = abs(sl - price)
                    reward = abs(price - t1)
                    confidence = min(75.0, 40 + len(conditions_met) * 10)

                    # Supertrend alignment bonus
                    if st_dir == -1:
                        confidence += 5
                        conditions_met.append("Supertrend bearish (aligned)")

                    signal = TradeSignal(
                        symbol=symbol,
                        signal=SignalType.SELL,
                        entry_price=round(price, 2),
                        stop_loss=round(sl, 2),
                        target=round(t1, 2),
                        trailing_sl=round(price + atr, 2),
                        confidence=confidence,
                        strategy_name=self.name,
                        reasoning=(
                            f"VWAP mean reversion SELL: price at upper band "
                            f"(VWAP+1σ=₹{upper_1:.2f}), targeting VWAP=₹{vwap:.2f}. "
                            f"R:R={reward/risk:.1f}x"
                            if risk > 0 else "VWAP MR SELL"
                        ),
                        conditions_met=conditions_met,
                        timeframe=self.timeframe,
                        timestamp=datetime.now().isoformat(),
                    )

        if signal:
            logger.info(
                f"VWAP MR signal: {signal.signal.value} @ ₹{signal.entry_price:.2f} "
                f"confidence={signal.confidence:.0f}% conditions={len(signal.conditions_met)}"
            )

        return signal
