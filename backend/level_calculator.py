"""
Level Calculator – VWAP, CPR, Support/Resistance, PDH/PDL

Computes intraday reference levels used for smart SL and target placement.
All levels are calculated from NIFTY spot 5-min candle data.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from datetime import datetime, time, date

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntraDayLevels:
    """All computed reference levels for the trading session."""
    # VWAP
    vwap: float = 0.0
    vwap_upper_1: float = 0.0   # VWAP + 1 std dev
    vwap_lower_1: float = 0.0   # VWAP - 1 std dev
    vwap_upper_2: float = 0.0   # VWAP + 2 std dev
    vwap_lower_2: float = 0.0   # VWAP - 2 std dev

    # CPR (Central Pivot Range) – based on previous day H/L/C
    pivot: float = 0.0          # (PDH + PDL + PDC) / 3
    bc: float = 0.0             # Bottom CPR = (PDH + PDL) / 2
    tc: float = 0.0             # Top CPR = 2 * Pivot - BC
    r1: float = 0.0             # Resistance 1 = 2 * Pivot - PDL
    r2: float = 0.0             # Resistance 2 = Pivot + (PDH - PDL)
    r3: float = 0.0             # Resistance 3 = PDH + 2 * (Pivot - PDL)
    s1: float = 0.0             # Support 1 = 2 * Pivot - PDH
    s2: float = 0.0             # Support 2 = Pivot - (PDH - PDL)
    s3: float = 0.0             # Support 3 = PDL - 2 * (PDH - Pivot)

    # Previous Day High / Low
    pdh: float = 0.0
    pdl: float = 0.0
    pdc: float = 0.0            # Previous Day Close

    # Today's ORB range
    orb_high: float = 0.0
    orb_low: float = 0.0

    # Dynamic S/R from recent swing points
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)

    # ATR for the session
    atr: float = 0.0

    timestamp: str = ""

    def get_buy_sl_levels(self, price: float) -> List[float]:
        """Return all support levels BELOW price, sorted descending (nearest first)."""
        candidates = [
            self.vwap_lower_1, self.vwap_lower_2, self.vwap,
            self.s1, self.s2, self.s3, self.bc, self.pivot,
            self.pdl, self.orb_low,
        ] + self.support_levels
        below = sorted([l for l in candidates if 0 < l < price], reverse=True)
        return below

    def get_sell_sl_levels(self, price: float) -> List[float]:
        """Return all resistance levels ABOVE price, sorted ascending (nearest first)."""
        candidates = [
            self.vwap_upper_1, self.vwap_upper_2, self.vwap,
            self.r1, self.r2, self.r3, self.tc, self.pivot,
            self.pdh, self.orb_high,
        ] + self.resistance_levels
        above = sorted([l for l in candidates if l > price])
        return above

    def get_buy_target_levels(self, price: float) -> List[float]:
        """Return all resistance levels ABOVE price, sorted ascending (nearest first)."""
        candidates = [
            self.vwap_upper_1, self.vwap_upper_2, self.vwap,
            self.r1, self.r2, self.r3, self.tc, self.pivot,
            self.pdh, self.orb_high,
        ] + self.resistance_levels
        above = sorted([l for l in candidates if l > price])
        return above

    def get_sell_target_levels(self, price: float) -> List[float]:
        """Return all support levels BELOW price, sorted descending (nearest first)."""
        candidates = [
            self.vwap_lower_1, self.vwap_lower_2, self.vwap,
            self.s1, self.s2, self.s3, self.bc, self.pivot,
            self.pdl, self.orb_low,
        ] + self.support_levels
        below = sorted([l for l in candidates if 0 < l < price], reverse=True)
        return below


class LevelCalculator:
    """
    Computes VWAP, CPR, S/R levels from 5-min candle data.
    Call `compute(df)` with full multi-day 5-min data (at least 2 sessions).
    """

    @staticmethod
    def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        return df

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------
    @staticmethod
    def compute_vwap(df_today: pd.DataFrame) -> Tuple[float, float, float, float, float]:
        """
        Compute VWAP and 1σ / 2σ bands for today's session data.
        Returns (vwap, upper_1, lower_1, upper_2, lower_2).
        """
        if df_today.empty or "volume" not in df_today.columns:
            # If no volume data, use typical price as proxy
            tp = (df_today["high"] + df_today["low"] + df_today["close"]) / 3
            vwap = tp.mean()
            std = tp.std()
            return vwap, vwap + std, vwap - std, vwap + 2 * std, vwap - 2 * std

        tp = (df_today["high"] + df_today["low"] + df_today["close"]) / 3
        vol = df_today["volume"].replace(0, 1)  # avoid div by zero
        cum_tp_vol = (tp * vol).cumsum()
        cum_vol = vol.cumsum()
        vwap_series = cum_tp_vol / cum_vol
        vwap = float(vwap_series.iloc[-1])

        # Standard deviation of (price - VWAP) weighted by volume
        deviation = tp - vwap_series
        cum_dev_sq_vol = (deviation ** 2 * vol).cumsum()
        std_series = np.sqrt(cum_dev_sq_vol / cum_vol)
        std = float(std_series.iloc[-1])

        return (
            round(vwap, 2),
            round(vwap + std, 2),
            round(vwap - std, 2),
            round(vwap + 2 * std, 2),
            round(vwap - 2 * std, 2),
        )

    # ------------------------------------------------------------------
    # CPR (Central Pivot Range) from previous session H/L/C
    # ------------------------------------------------------------------
    @staticmethod
    def compute_cpr(pdh: float, pdl: float, pdc: float) -> Dict[str, float]:
        """
        Standard pivot + CPR + 3 R/S levels.
        """
        pivot = round((pdh + pdl + pdc) / 3, 2)
        bc = round((pdh + pdl) / 2, 2)
        tc = round(2 * pivot - bc, 2)

        r1 = round(2 * pivot - pdl, 2)
        r2 = round(pivot + (pdh - pdl), 2)
        r3 = round(pdh + 2 * (pivot - pdl), 2)

        s1 = round(2 * pivot - pdh, 2)
        s2 = round(pivot - (pdh - pdl), 2)
        s3 = round(pdl - 2 * (pdh - pivot), 2)

        return {
            "pivot": pivot, "bc": bc, "tc": tc,
            "r1": r1, "r2": r2, "r3": r3,
            "s1": s1, "s2": s2, "s3": s3,
        }

    # ------------------------------------------------------------------
    # Dynamic Support / Resistance from swing points
    # ------------------------------------------------------------------
    @staticmethod
    def find_swing_levels(
        df: pd.DataFrame, lookback: int = 5, max_levels: int = 5
    ) -> Tuple[List[float], List[float]]:
        """
        Find swing highs (resistance) and swing lows (support) from recent data.
        A swing high is a bar whose high is higher than `lookback` bars on either side.
        """
        supports = []
        resistances = []

        if len(df) < lookback * 2 + 1:
            return supports, resistances

        highs = df["high"].values
        lows = df["low"].values

        for i in range(lookback, len(df) - lookback):
            # Swing high
            if highs[i] == max(highs[i - lookback:i + lookback + 1]):
                resistances.append(round(float(highs[i]), 2))
            # Swing low
            if lows[i] == min(lows[i - lookback:i + lookback + 1]):
                supports.append(round(float(lows[i]), 2))

        # Deduplicate (cluster nearby levels within 0.1% range)
        supports = LevelCalculator._cluster_levels(supports, pct=0.001)
        resistances = LevelCalculator._cluster_levels(resistances, pct=0.001)

        return supports[-max_levels:], resistances[-max_levels:]

    @staticmethod
    def _cluster_levels(levels: List[float], pct: float = 0.001) -> List[float]:
        """Cluster nearby levels and return the average of each cluster."""
        if not levels:
            return []
        levels = sorted(set(levels))
        clusters = [[levels[0]]]
        for l in levels[1:]:
            if abs(l - clusters[-1][-1]) / max(clusters[-1][-1], 1) < pct:
                clusters[-1].append(l)
            else:
                clusters.append([l])
        return [round(sum(c) / len(c), 2) for c in clusters]

    # ------------------------------------------------------------------
    # Previous session H / L / C
    # ------------------------------------------------------------------
    @staticmethod
    def get_prev_session_hlc(df: pd.DataFrame) -> Tuple[float, float, float]:
        """Extract previous trading session's high, low, close from multi-day data."""
        if "date" not in df.columns:
            return 0, 0, 0

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

        df["session_date"] = df["date"].dt.date
        dates = sorted(df["session_date"].unique())

        if len(dates) < 2:
            # Single day — use first half as "previous"
            return float(df["high"].max()), float(df["low"].min()), float(df["close"].iloc[-1])

        prev_date = dates[-2]
        prev_df = df[df["session_date"] == prev_date]
        pdh = float(prev_df["high"].max())
        pdl = float(prev_df["low"].min())
        pdc = float(prev_df["close"].iloc[-1])
        return pdh, pdl, pdc

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------
    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return round(float(atr.iloc[-1]), 2)

    # ------------------------------------------------------------------
    # Master compute
    # ------------------------------------------------------------------
    def compute(
        self, df: pd.DataFrame, orb_high: float = 0, orb_low: float = 0
    ) -> IntraDayLevels:
        """
        Compute all intraday levels from 5-min candle data.
        df should contain at least 2 sessions for CPR calculation.
        """
        levels = IntraDayLevels()

        if df is None or df.empty:
            return levels

        if "date" not in df.columns:
            return levels

        df = self._normalize_dates(df)

        # -- Previous session H/L/C for CPR --
        pdh, pdl, pdc = self.get_prev_session_hlc(df)
        levels.pdh = pdh
        levels.pdl = pdl
        levels.pdc = pdc

        # -- CPR --
        if pdh > 0 and pdl > 0 and pdc > 0:
            cpr = self.compute_cpr(pdh, pdl, pdc)
            levels.pivot = cpr["pivot"]
            levels.bc = cpr["bc"]
            levels.tc = cpr["tc"]
            levels.r1 = cpr["r1"]
            levels.r2 = cpr["r2"]
            levels.r3 = cpr["r3"]
            levels.s1 = cpr["s1"]
            levels.s2 = cpr["s2"]
            levels.s3 = cpr["s3"]

        # -- Today's candles for VWAP --
        today = df["date"].dt.date.iloc[-1]
        market_open = pd.Timestamp(datetime.combine(today, time(9, 15)))
        df_today = df[df["date"] >= market_open]

        if not df_today.empty:
            vwap, vu1, vl1, vu2, vl2 = self.compute_vwap(df_today)
            levels.vwap = vwap
            levels.vwap_upper_1 = vu1
            levels.vwap_lower_1 = vl1
            levels.vwap_upper_2 = vu2
            levels.vwap_lower_2 = vl2

        # -- Swing S/R --
        supports, resistances = self.find_swing_levels(df, lookback=5)
        levels.support_levels = supports
        levels.resistance_levels = resistances
        if supports:
            price = float(df["close"].iloc[-1])
            below = [s for s in supports if s < price]
            levels.nearest_support = below[-1] if below else supports[0]
        if resistances:
            price = float(df["close"].iloc[-1])
            above = [r for r in resistances if r > price]
            levels.nearest_resistance = above[0] if above else resistances[-1]

        # -- ORB --
        levels.orb_high = orb_high
        levels.orb_low = orb_low

        # -- ATR --
        levels.atr = self.compute_atr(df)

        levels.timestamp = datetime.now().isoformat()
        return levels
