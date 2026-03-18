"""
Smart SL & Target Engine

Replaces the old "flat %" and "raw ATR" SL/target logic with
level-aware placement using VWAP, CPR, support/resistance, and ATR.

Design principles:
  - SL is placed at the NEAREST structural level (VWAP band, CPR, S/R, ORB edge)
    that is close enough to keep risk tight, but never closer than 0.5×ATR.
  - If no structural level is close enough (within 2×ATR), use 1×ATR as SL.
  - Targets are placed at the next structural levels above/below entry.
  - Trailing SL moves to breakeven at T1, then trails behind each target.
  - For option premiums: SL mapped from spot SL using delta.
"""
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

from level_calculator import IntraDayLevels

logger = logging.getLogger(__name__)


@dataclass
class SmartLevels:
    """Computed SL and target levels for a trade."""
    entry: float = 0.0
    stop_loss: float = 0.0
    sl_type: str = ""            # e.g. "VWAP_LOWER_1", "S1", "ATR_1X"
    target1: float = 0.0
    t1_type: str = ""
    target2: float = 0.0
    t2_type: str = ""
    target3: float = 0.0
    t3_type: str = ""
    trailing_sl: float = 0.0
    risk_points: float = 0.0    # entry - SL (abs)
    reward_points: float = 0.0  # T1 - entry (abs)
    risk_reward: float = 0.0
    # For option premium mapping
    option_sl: float = 0.0
    option_t1: float = 0.0
    option_t2: float = 0.0
    option_t3: float = 0.0


# Label map for readable SL/target descriptions
_LEVEL_LABELS = {
    "vwap": "VWAP",
    "vwap_lower_1": "VWAP-1σ",
    "vwap_upper_1": "VWAP+1σ",
    "vwap_lower_2": "VWAP-2σ",
    "vwap_upper_2": "VWAP+2σ",
    "s1": "CPR S1",
    "s2": "CPR S2",
    "s3": "CPR S3",
    "r1": "CPR R1",
    "r2": "CPR R2",
    "r3": "CPR R3",
    "bc": "CPR BC",
    "tc": "CPR TC",
    "pivot": "Pivot",
    "pdl": "Prev Day Low",
    "pdh": "Prev Day High",
    "orb_low": "ORB Low",
    "orb_high": "ORB High",
    "support": "Swing Support",
    "resistance": "Swing Resistance",
}


def _label_for_level(value: float, levels: IntraDayLevels) -> str:
    """Find the best label for a given numeric level value."""
    tol = 0.5  # tolerance in points
    checks = [
        (levels.vwap, "VWAP"),
        (levels.vwap_lower_1, "VWAP-1σ"),
        (levels.vwap_upper_1, "VWAP+1σ"),
        (levels.vwap_lower_2, "VWAP-2σ"),
        (levels.vwap_upper_2, "VWAP+2σ"),
        (levels.s1, "CPR S1"),
        (levels.s2, "CPR S2"),
        (levels.s3, "CPR S3"),
        (levels.r1, "CPR R1"),
        (levels.r2, "CPR R2"),
        (levels.r3, "CPR R3"),
        (levels.bc, "CPR BC"),
        (levels.tc, "CPR TC"),
        (levels.pivot, "Pivot"),
        (levels.pdl, "PDL"),
        (levels.pdh, "PDH"),
        (levels.orb_low, "ORB Low"),
        (levels.orb_high, "ORB High"),
    ]
    for s in levels.support_levels:
        checks.append((s, "Swing Support"))
    for r in levels.resistance_levels:
        checks.append((r, "Swing Resistance"))

    for lv, label in checks:
        if lv > 0 and abs(lv - value) <= tol:
            return label
    return "ATR-Based"


class SmartSLEngine:
    """
    Computes intelligent SL and targets based on structural levels.

    Parameters:
      min_sl_atr_mult: Minimum SL distance as ATR multiplier (default 0.5)
      max_sl_atr_mult: Maximum SL distance as ATR multiplier (default 2.0)
      default_sl_atr_mult: Default SL if no structural level found (default 1.0)
      target_min_rr: Minimum risk:reward for T1 (default 1.5)
    """

    def __init__(
        self,
        min_sl_atr_mult: float = 0.5,
        max_sl_atr_mult: float = 2.0,
        default_sl_atr_mult: float = 1.0,
        target_min_rr: float = 1.5,
    ):
        self.min_sl_atr = min_sl_atr_mult
        self.max_sl_atr = max_sl_atr_mult
        self.default_sl_atr = default_sl_atr_mult
        self.target_min_rr = target_min_rr

    # ------------------------------------------------------------------
    # BUY side
    # ------------------------------------------------------------------
    def compute_buy_levels(
        self,
        entry: float,
        levels: IntraDayLevels,
    ) -> SmartLevels:
        """
        Compute SL and targets for a BUY trade.
        SL = nearest support below entry within [0.5×ATR, 2×ATR].
        Targets = next resistance levels above entry.
        """
        atr = levels.atr
        if atr <= 0:
            atr = entry * 0.005  # fallback 0.5%

        min_sl_dist = atr * self.min_sl_atr
        max_sl_dist = atr * self.max_sl_atr
        default_sl_dist = atr * self.default_sl_atr

        # -- Find SL from structural supports --
        candidates = levels.get_buy_sl_levels(entry)
        sl = 0.0
        sl_type = "ATR_1X"

        for lvl in candidates:
            dist = entry - lvl
            if min_sl_dist <= dist <= max_sl_dist:
                # Place SL slightly below the structural level (buffer)
                sl = round(lvl - atr * 0.1, 2)
                sl_type = _label_for_level(lvl, levels)
                break

        if sl <= 0:
            # No good structural level — use default ATR-based SL
            sl = round(entry - default_sl_dist, 2)
            sl_type = f"ATR×{self.default_sl_atr}"

        risk = entry - sl


        # -- Find targets at 1:1, 2:1, 3:1 R:R --
        t1 = round(entry + risk * 1, 2)
        t2 = round(entry + risk * 2, 2)
        t3 = round(entry + risk * 3, 2)
        t1_type = "RR×1.0"
        t2_type = "RR×2.0"
        t3_type = "RR×3.0"

        # Trailing SL starts at SL, moves to breakeven when T1 hit
        trailing_sl = sl

        reward = t1 - entry
        rr = round(reward / risk, 2) if risk > 0 else 0

        return SmartLevels(
            entry=entry,
            stop_loss=sl, sl_type=sl_type,
            target1=t1, t1_type=t1_type,
            target2=t2, t2_type=t2_type,
            target3=t3, t3_type=t3_type,
            trailing_sl=trailing_sl,
            risk_points=round(risk, 2),
            reward_points=round(reward, 2),
            risk_reward=rr,
        )

    # ------------------------------------------------------------------
    # SELL side
    # ------------------------------------------------------------------
    def compute_sell_levels(
        self,
        entry: float,
        levels: IntraDayLevels,
    ) -> SmartLevels:
        """
        Compute SL and targets for a SELL trade.
        SL = nearest resistance above entry within [0.5×ATR, 2×ATR].
        Targets = next support levels below entry.
        """
        atr = levels.atr
        if atr <= 0:
            atr = entry * 0.005

        min_sl_dist = atr * self.min_sl_atr
        max_sl_dist = atr * self.max_sl_atr
        default_sl_dist = atr * self.default_sl_atr

        candidates = levels.get_sell_sl_levels(entry)
        sl = 0.0
        sl_type = "ATR_1X"

        for lvl in candidates:
            dist = lvl - entry
            if min_sl_dist <= dist <= max_sl_dist:
                sl = round(lvl + atr * 0.1, 2)
                sl_type = _label_for_level(lvl, levels)
                break

        if sl <= 0:
            sl = round(entry + default_sl_dist, 2)
            sl_type = f"ATR×{self.default_sl_atr}"

        risk = sl - entry


        # -- Find targets at 1:1, 2:1, 3:1 R:R --
        t1 = round(entry - risk * 1, 2)
        t2 = round(entry - risk * 2, 2)
        t3 = round(entry - risk * 3, 2)
        t1_type = "RR×1.0"
        t2_type = "RR×2.0"
        t3_type = "RR×3.0"

        trailing_sl = sl
        reward = entry - t1
        rr = round(reward / risk, 2) if risk > 0 else 0

        return SmartLevels(
            entry=entry,
            stop_loss=sl, sl_type=sl_type,
            target1=t1, t1_type=t1_type,
            target2=t2, t2_type=t2_type,
            target3=t3, t3_type=t3_type,
            trailing_sl=trailing_sl,
            risk_points=round(risk, 2),
            reward_points=round(reward, 2),
            risk_reward=rr,
        )

    # ------------------------------------------------------------------
    # Map spot-level SL/targets to option premium SL/targets using delta
    # ------------------------------------------------------------------
    @staticmethod
    def map_to_option_premium(
        smart: SmartLevels,
        option_entry: float,
        delta: float,
        direction: str = "BUY",  # BUY = CE, SELL = PE
    ) -> SmartLevels:
        """
        Convert spot-level movements to option premium levels using delta.

        For CE options:
          premium_change ≈ spot_change × abs(delta)

        For PE options:
          premium_change ≈ -spot_change × abs(delta)
        """
        d = abs(delta)
        if d <= 0:
            d = 0.4  # fallback

        if direction == "BUY":
            # CE option: premium rises when spot rises
            sl_move = (smart.entry - smart.stop_loss) * d
            t1_move = (smart.target1 - smart.entry) * d
            t2_move = (smart.target2 - smart.entry) * d
            t3_move = (smart.target3 - smart.entry) * d
        else:
            # PE option: premium rises when spot falls
            sl_move = (smart.stop_loss - smart.entry) * d
            t1_move = (smart.entry - smart.target1) * d
            t2_move = (smart.entry - smart.target2) * d
            t3_move = (smart.entry - smart.target3) * d

        smart.option_sl = round(max(option_entry - sl_move, option_entry * 0.5), 2)
        smart.option_t1 = round(option_entry + t1_move, 2)
        smart.option_t2 = round(option_entry + t2_move, 2)
        smart.option_t3 = round(option_entry + t3_move, 2)

        return smart


def compute_smart_levels(
    entry: float,
    direction: str,  # "BUY" or "SELL"
    levels: IntraDayLevels,
    min_sl_atr: float = 0.5,
    max_sl_atr: float = 2.0,
    default_sl_atr: float = 1.0,
    target_min_rr: float = 1.5,
) -> SmartLevels:
    """
    Convenience function — compute smart SL + targets for a trade.
    """
    engine = SmartSLEngine(min_sl_atr, max_sl_atr, default_sl_atr, target_min_rr)
    if direction == "BUY":
        return engine.compute_buy_levels(entry, levels)
    else:
        return engine.compute_sell_levels(entry, levels)
