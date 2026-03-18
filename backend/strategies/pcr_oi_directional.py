"""
PCR / OI Directional Strategy

Uses Put-Call Ratio and OI buildup data from LiveOIFeed to generate
directional signals.

Logic:
  - BUY when PCR > 1.2 (heavy PE writing = bullish), PE OI rising, CE OI falling
  - SELL when PCR < 0.7 (heavy CE writing = bearish), CE OI rising, PE OI falling
  - Requires Supertrend alignment as trend filter

Works best as a confluence strategy alongside ORB / Supertrend.
"""
import logging
from datetime import datetime, time
from typing import Optional, Dict, Any

import pandas as pd

from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType
from level_calculator import LevelCalculator, IntraDayLevels
from smart_sl_engine import compute_smart_levels

logger = logging.getLogger(__name__)


class PCROIDirectionalStrategy(BaseStrategy):
    name = "PCR/OI Directional"
    description = (
        "Uses Put-Call Ratio and open interest buildup to generate "
        "directional signals. Bullish when PCR > 1.2, bearish when PCR < 0.7."
    )
    timeframe = "5m"
    min_bars = 20

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.level_calc = LevelCalculator()
        self._cached_levels: Optional[IntraDayLevels] = None

    def default_params(self) -> Dict[str, Any]:
        return {
            "pcr_bullish_threshold": 1.2,
            "pcr_bearish_threshold": 0.7,
            "oi_change_min_pct": 3.0,       # minimum OI change % to confirm
            "require_supertrend": True,
            "st_atr_period": 10,
            "st_multiplier": 3.0,
            # Smart SL
            "min_sl_atr_mult": 0.5,
            "max_sl_atr_mult": 2.0,
            "default_sl_atr_mult": 1.0,
            "target_min_rr": 1.5,
        }

    def _compute_supertrend_dir(self, df: pd.DataFrame) -> int:
        """Compute current Supertrend direction: 1=bullish, -1=bearish."""
        period = self.params["st_atr_period"]
        mult = self.params["st_multiplier"]
        atr = self.calculate_atr(df, period)
        hl2 = (df["high"] + df["low"]) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr
        direction = pd.Series(0, index=df.index, dtype=int)
        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
                if direction.iloc[i] == 1:
                    lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1])
                else:
                    upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1])
        return int(direction.iloc[-1])

    def generate_signal(
        self, df: pd.DataFrame, symbol: str, oi_analysis: Dict = None
    ) -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None
        if not oi_analysis:
            return None

        # Time gate: only 9:45 AM – 3:00 PM
        if "date" in df.columns:
            last_ts = pd.to_datetime(df["date"].iloc[-1])
            if hasattr(last_ts, "tz") and last_ts.tz is not None:
                last_ts = last_ts.tz_convert("Asia/Kolkata").tz_localize(None)
            t = last_ts.time()
            if t < time(9, 45) or t > time(15, 0):
                return None

        # Extract OI metrics
        pcr = oi_analysis.get("pcr", 1.0)
        ce_oi_chg = oi_analysis.get("ce_oi_change_pct", 0.0)
        pe_oi_chg = oi_analysis.get("pe_oi_change_pct", 0.0)

        price = float(df["close"].iloc[-1])
        conditions = []
        signal_type = None

        # Supertrend filter
        st_dir = self._compute_supertrend_dir(df) if self.params["require_supertrend"] else 0

        min_oi_chg = self.params["oi_change_min_pct"]

        # ── BULLISH: high PCR + PE writing + CE unwinding ──
        if pcr >= self.params["pcr_bullish_threshold"]:
            conditions.append(f"PCR {pcr:.2f} ≥ {self.params['pcr_bullish_threshold']} (bullish)")
            if pe_oi_chg > min_oi_chg:
                conditions.append(f"PE OI buildup +{pe_oi_chg:.1f}% (put writing)")
            if ce_oi_chg < -min_oi_chg:
                conditions.append(f"CE OI unwinding {ce_oi_chg:.1f}% (short covering)")
            oi_confirms = (pe_oi_chg > min_oi_chg) or (ce_oi_chg < -min_oi_chg)
            st_confirms = (st_dir == 1) if self.params["require_supertrend"] else True
            if oi_confirms and st_confirms:
                if self.params["require_supertrend"]:
                    conditions.append("Supertrend BULLISH")
                signal_type = SignalType.BUY

        # ── BEARISH: low PCR + CE writing + PE unwinding ──
        elif pcr <= self.params["pcr_bearish_threshold"]:
            conditions.append(f"PCR {pcr:.2f} ≤ {self.params['pcr_bearish_threshold']} (bearish)")
            if ce_oi_chg > min_oi_chg:
                conditions.append(f"CE OI buildup +{ce_oi_chg:.1f}% (call writing)")
            if pe_oi_chg < -min_oi_chg:
                conditions.append(f"PE OI unwinding {pe_oi_chg:.1f}%")
            oi_confirms = (ce_oi_chg > min_oi_chg) or (pe_oi_chg < -min_oi_chg)
            st_confirms = (st_dir == -1) if self.params["require_supertrend"] else True
            if oi_confirms and st_confirms:
                if self.params["require_supertrend"]:
                    conditions.append("Supertrend BEARISH")
                signal_type = SignalType.SELL

        if signal_type is None:
            return None

        # Smart SL & targets
        levels = self.level_calc.compute(df)
        self._cached_levels = levels
        direction = "BUY" if signal_type == SignalType.BUY else "SELL"
        smart = compute_smart_levels(
            entry=price, direction=direction, levels=levels,
            min_sl_atr=self.params["min_sl_atr_mult"],
            max_sl_atr=self.params["max_sl_atr_mult"],
            default_sl_atr=self.params["default_sl_atr_mult"],
            target_min_rr=self.params["target_min_rr"],
        )

        conditions.append(f"SL at {smart.stop_loss:.2f} ({smart.sl_type})")
        conditions.append(f"T1={smart.target1:.2f} T2={smart.target2:.2f} T3={smart.target3:.2f}")
        conditions.append(f"R:R {smart.risk_reward}")

        confidence = 65.0 + len(conditions) * 3.0
        confidence = min(95.0, confidence)

        return TradeSignal(
            symbol=symbol,
            signal=signal_type,
            entry_price=price,
            stop_loss=smart.stop_loss,
            target=smart.target1,
            trailing_sl=smart.trailing_sl,
            confidence=confidence,
            strategy_name=self.name,
            reasoning=(
                f"PCR {pcr:.2f} → {direction}. "
                f"CE OI Δ{ce_oi_chg:+.1f}%, PE OI Δ{pe_oi_chg:+.1f}%. "
                f"SL at {smart.sl_type} ({smart.stop_loss:.2f}), "
                f"Targets: T1={smart.target1:.2f} T2={smart.target2:.2f}"
            ),
            conditions_met=conditions,
            timeframe=self.timeframe,
            timestamp=datetime.now().isoformat(),
        )

    def get_cached_levels(self) -> Optional[IntraDayLevels]:
        return self._cached_levels

