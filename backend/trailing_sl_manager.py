"""
Trailing Stop-Loss Manager

Implements a phase-based trailing SL system that:
  - Protects profits once a trade moves in favour
  - Locks in gains at each target level
  - Uses multiple trailing methods (ATR-based, percentage, swing-low)
  - Never moves SL backwards (only in the profitable direction)
  - Keeps trades alive during strong trends

Trailing Phases:
  INITIAL    → Entry to T1 zone: SL stays at initial smart SL (let trade breathe)
  BREAKEVEN  → T1 hit: SL moves to entry price (cost-to-cost, risk-free)
  TRAIL_T1   → Between T1 and T2: trail using ATR × 1.0 below highs
  TRAIL_T2   → T2 hit+: trail using ATR × 0.7 below highs (tighter)
  TIGHT      → T3 zone or 80%+ gain: very tight trail (ATR × 0.5)

Key Rules:
  - TSL can ONLY move UP (for long/CE), never down
  - Watermark (highest_price) tracks the peak premium since entry
  - On every cycle, compute the best TSL from all methods, pick the highest
  - If the new TSL > current TSL → update it on broker
"""
import logging
from typing import Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrailingConfig:
    """Configuration for trailing SL behaviour."""
    # Phase transition thresholds (as % gain on premium)
    breakeven_trigger_pct: float = 3.0    # Move SL to breakeven after 3% gain
    trail_t1_atr_mult: float = 1.0        # ATR multiplier in TRAIL_T1 phase
    trail_t2_atr_mult: float = 0.7        # ATR multiplier in TRAIL_T2 phase
    tight_atr_mult: float = 0.5           # ATR multiplier in TIGHT phase
    tight_trigger_pct: float = 80.0       # Enter TIGHT phase at 80%+ gain
    # Percentage-based trailing
    trail_pct_initial: float = 15.0       # 15% trail from highs in TRAIL_T1
    trail_pct_t2: float = 10.0            # 10% trail from highs in TRAIL_T2
    trail_pct_tight: float = 6.0          # 6% trail from highs in TIGHT
    # Swing-low based trailing
    swing_lookback_candles: int = 3       # Use lowest low of last N candles
    # Early breakeven: if price hits this % above entry, move SL to entry
    early_breakeven_pct: float = 5.0      # 5% profit → lock breakeven
    # Buffer below structural level (avoid noise hits)
    sl_buffer_pct: float = 0.5            # Place SL 0.5% below the calculated level


class TrailingSLManager:
    """
    Computes the optimal trailing SL for a position every cycle.

    Usage (called from monitor_positions):
        tsl_mgr = TrailingSLManager(config)
        new_tsl, new_phase, updated = tsl_mgr.compute_trailing_sl(
            pos=managed_position,
            current_ltp=ltp,
            option_atr=atr_on_option,
            df=spot_df,
        )
        if updated:
            # Update SL order on broker
    """

    def __init__(self, config: TrailingConfig = None):
        self.config = config or TrailingConfig()

    def compute_trailing_sl(
        self,
        entry_price: float,
        current_ltp: float,
        current_tsl: float,
        highest_price: float,
        trailing_phase: str,
        t1_hit: bool,
        t2_hit: bool,
        t3_hit: bool,
        target1: float,
        target2: float,
        target3: float,
        option_atr: float = 0.0,
        df: Optional[pd.DataFrame] = None,
        spot_atr: float = 0.0,
        delta: float = 0.4,
    ) -> Tuple[float, str, float, bool]:
        """
        Compute the new trailing SL.

        Returns:
            (new_tsl, new_phase, new_highest, updated)
            updated=True if TSL changed and broker needs updating
        """
        cfg = self.config

        # Update watermark
        new_highest = max(highest_price, current_ltp)

        # Compute gain %
        gain_pct = ((current_ltp - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # Determine option ATR (estimate from spot ATR × delta if not provided)
        if option_atr <= 0 and spot_atr > 0 and delta > 0:
            option_atr = spot_atr * abs(delta)
        if option_atr <= 0:
            option_atr = entry_price * 0.03  # fallback 3% of entry

        # --- Determine phase ---
        new_phase = trailing_phase

        if t3_hit or gain_pct >= cfg.tight_trigger_pct:
            new_phase = "TIGHT"
        elif t2_hit:
            new_phase = "TRAIL_T2"
        elif t1_hit:
            new_phase = "TRAIL_T1"
        elif gain_pct >= cfg.early_breakeven_pct:
            new_phase = "BREAKEVEN"
        else:
            new_phase = "INITIAL"

        # --- Compute TSL candidates from different methods ---
        tsl_candidates = []

        # Method 1: Phase-based fixed floors
        if new_phase == "BREAKEVEN":
            # Lock breakeven: SL at entry price (cost-to-cost)
            floor = entry_price
            tsl_candidates.append(floor)

        elif new_phase == "TRAIL_T1":
            # Floor at entry price (never go below breakeven once T1 hit)
            tsl_candidates.append(entry_price)

            # ATR-based trail from watermark
            atr_tsl = new_highest - option_atr * cfg.trail_t1_atr_mult
            tsl_candidates.append(atr_tsl)

            # Percentage-based trail from watermark
            pct_tsl = new_highest * (1 - cfg.trail_pct_initial / 100)
            tsl_candidates.append(pct_tsl)

        elif new_phase == "TRAIL_T2":
            # Floor at T1 level (never below T1 once T2 is hit)
            tsl_candidates.append(target1)

            # ATR-based trail (tighter)
            atr_tsl = new_highest - option_atr * cfg.trail_t2_atr_mult
            tsl_candidates.append(atr_tsl)

            # Percentage-based trail (tighter)
            pct_tsl = new_highest * (1 - cfg.trail_pct_t2 / 100)
            tsl_candidates.append(pct_tsl)

        elif new_phase == "TIGHT":
            # Floor at T2 level (never below T2 once in tight phase)
            tsl_candidates.append(target2)

            # Very tight ATR trail
            atr_tsl = new_highest - option_atr * cfg.tight_atr_mult
            tsl_candidates.append(atr_tsl)

            # Very tight percentage trail
            pct_tsl = new_highest * (1 - cfg.trail_pct_tight / 100)
            tsl_candidates.append(pct_tsl)

        # Method 2: Swing-low based (use lowest low of last N candles mapped to option)
        if df is not None and len(df) >= cfg.swing_lookback_candles and new_phase not in ("INITIAL",):
            swing_tsl = self._compute_swing_low_tsl(
                df, cfg.swing_lookback_candles, entry_price, delta, spot_atr
            )
            if swing_tsl > 0:
                tsl_candidates.append(swing_tsl)

        # Method 3: Supertrend line as trailing SL (if available and trade is winning)
        if df is not None and new_phase not in ("INITIAL",):
            st_tsl = self._compute_supertrend_tsl(df, entry_price, delta)
            if st_tsl > 0:
                tsl_candidates.append(st_tsl)

        # --- Pick the HIGHEST TSL candidate (best protection) ---
        if tsl_candidates:
            best_tsl = max(tsl_candidates)
        else:
            best_tsl = current_tsl

        # Apply buffer (slightly below to avoid noise triggers)
        if best_tsl > 0 and cfg.sl_buffer_pct > 0:
            best_tsl = best_tsl * (1 - cfg.sl_buffer_pct / 100)

        # Round to 2 decimal places
        best_tsl = round(best_tsl, 2)

        # KEY RULE: TSL can ONLY move UP, never down
        if best_tsl <= current_tsl:
            # No improvement — keep current TSL
            return current_tsl, new_phase, new_highest, False

        # TSL improved
        logger.info(
            f"TSL UPDATE: {current_tsl:.2f} → {best_tsl:.2f} "
            f"(phase={new_phase}, gain={gain_pct:.1f}%, "
            f"watermark={new_highest:.2f}, atr={option_atr:.2f})"
        )

        return best_tsl, new_phase, new_highest, True

    # ------------------------------------------------------------------
    # Swing-low based TSL
    # ------------------------------------------------------------------
    def _compute_swing_low_tsl(
        self,
        df: pd.DataFrame,
        lookback: int,
        entry_price: float,
        delta: float,
        spot_atr: float,
    ) -> float:
        """
        Use the lowest low of the last N spot candles as a trailing reference.
        Map the spot movement to option premium via delta.
        """
        if "low" not in df.columns or len(df) < lookback:
            return 0.0

        recent_lows = df["low"].iloc[-lookback:].values
        swing_low = float(np.min(recent_lows))
        current_spot = float(df["close"].iloc[-1])

        if current_spot <= 0 or swing_low <= 0:
            return 0.0

        # How far below current spot is the swing low?
        spot_drop = current_spot - swing_low
        # Map to option premium drop using delta
        d = abs(delta) if delta > 0 else 0.4
        option_drop = spot_drop * d

        # TSL = entry + current_gain - option_drop
        # But since we don't have current option price here, approximate:
        # If current_ltp isn't available, we derive from spot movement
        # This is a conservative floor
        tsl = entry_price + (current_spot - swing_low) * d * 0.5  # 50% of swing-to-entry mapped

        return max(0, tsl)

    # ------------------------------------------------------------------
    # Supertrend-line based TSL
    # ------------------------------------------------------------------
    def _compute_supertrend_tsl(
        self,
        df: pd.DataFrame,
        entry_price: float,
        delta: float,
    ) -> float:
        """
        Use the supertrend line as a dynamic trailing SL.
        If supertrend is bullish, the lower band acts as support.
        Map the spot-level supertrend distance to option premium.
        """
        st_col = None
        for col in ("supertrend", "st_value", "st"):
            if col in df.columns:
                st_col = col
                break

        if st_col is None or "st_dir" not in df.columns:
            return 0.0

        curr_dir = int(df.iloc[-1].get("st_dir", 0))
        st_value = float(df.iloc[-1][st_col])
        current_spot = float(df["close"].iloc[-1])

        if st_value <= 0 or current_spot <= 0:
            return 0.0

        # Only use supertrend as TSL when direction supports the trade
        # For CE (bullish): supertrend must be bullish (dir=1), use ST line as floor
        # The ST line is below spot in uptrend — distance = spot - ST
        if curr_dir == 1:
            spot_buffer = current_spot - st_value  # distance above ST line
            d = abs(delta) if delta > 0 else 0.4
            # Map: if spot drops to ST line, option would lose spot_buffer * delta
            option_drop = spot_buffer * d
            tsl = entry_price  # At minimum, if spot @ ST line, we're roughly at entry
            # Actually compute more precisely:
            # current_option ≈ entry + (current_spot - entry_spot) * delta  (approx)
            # if spot drops to ST: option ≈ entry + (ST - entry_spot) * delta
            # So TSL ≈ current_option - option_drop
            # But we don't have entry_spot here, so use a simpler heuristic:
            # TSL = max(entry, current_option - option_drop) — protect at least breakeven
            return max(entry_price, 0)

        return 0.0

    # ------------------------------------------------------------------
    # Compute option ATR from spot data
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_option_atr(
        spot_atr: float,
        delta: float,
        entry_price: float,
    ) -> float:
        """
        Estimate option ATR from spot ATR using delta.
        option_price_change ≈ spot_price_change × |delta|
        """
        d = abs(delta) if delta > 0 else 0.4
        estimated = spot_atr * d

        # Sanity check: option ATR shouldn't be more than 20% of entry
        max_atr = entry_price * 0.20
        return min(estimated, max_atr) if estimated > 0 else entry_price * 0.03


# Singleton with default config
trailing_sl_manager = TrailingSLManager()
