"""
Expiry Premium Sell Strategy

For Thursday (weekly expiry) sessions:
  - SELL ATM straddle/strangle to capture theta decay
  - Time decay accelerates on expiry day → premium collapses
  - Enter after 11:00 AM when morning volatility settles
  - Exit by 2:30 PM (before final hour gamma risk)

Entry rules:
  - Expiry day (Thursday) only
  - India VIX < 20 (manageable premium)
  - ORB range narrow (< 0.6% of spot) → no trending
  - Sell ATM CE + ATM PE (short straddle) or OTM ± 1 strike (short strangle)
  - Combined premium must offer at least 0.3% of spot as target

Risk rules:
  - Stop loss at 50% premium gain reversal (e.g., if collected 200, SL at 300)
  - Max loss = 1.5x premium collected
  - Auto square-off at 2:30 PM on expiry

Disabled by default — enable via settings.enable_expiry_sell_strategy.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, time, date

import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExpiryStraddleSignal:
    """Signal for selling ATM straddle/strangle on expiry day."""
    ce_symbol: str
    pe_symbol: str
    ce_strike: float
    pe_strike: float
    ce_premium: float
    pe_premium: float
    combined_premium: float
    spot_price: float
    max_loss: float
    target_profit: float
    confidence: float
    reasoning: str
    conditions_met: List[str]


class ExpiryPremiumSellStrategy(BaseStrategy):
    name = "Expiry Premium Sell"
    description = (
        "Sells ATM straddle on expiry day (Thursday) to capture theta decay. "
        "Enters after 11:00 AM, exits by 2:30 PM. Only on low-VIX, range-bound days."
    )
    timeframe = "5m"
    min_bars = 20

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)

    def default_params(self) -> Dict[str, Any]:
        return {
            "entry_after_time": "11:00",        # Only enter after morning volatility
            "exit_before_time": "14:30",         # Exit before final hour
            "max_vix": 20.0,                     # Skip if VIX > 20
            "max_orb_range_pct": 0.6,            # ORB range must be < 0.6% of spot
            "min_premium_pct": 0.3,              # Combined premium >= 0.3% of spot
            "sl_multiplier": 1.5,                # SL at 1.5x premium collected
            "target_pct_of_premium": 0.6,        # Target 60% of collected premium
            "prefer_strangle": False,            # False = straddle, True = strangle (±1 strike)
            "strangle_offset_strikes": 1,        # OTM by N strikes for strangle
        }

    def is_expiry_day(self) -> bool:
        """Check if today is a weekly expiry day (Thursday)."""
        return date.today().weekday() == 3  # Thursday

    def is_entry_window(self) -> bool:
        """Check if current time is within entry window."""
        now = datetime.now().time()
        try:
            hh, mm = self.params["entry_after_time"].split(":")
            entry_time = time(int(hh), int(mm))
            hh2, mm2 = self.params["exit_before_time"].split(":")
            exit_time = time(int(hh2), int(mm2))
            return entry_time <= now <= exit_time
        except Exception:
            return time(11, 0) <= now <= time(14, 30)

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        india_vix: float = 0,
        orb_high: float = 0,
        orb_low: float = 0,
        gap_type: str = "NONE",
    ) -> Optional[TradeSignal]:
        """
        Generate a SELL signal for expiry premium selling.

        Note: This generates a SELL signal which means the auto_order_manager
        needs to handle short selling of options. Currently the system only
        supports BUY-side. This strategy produces advisory signals.
        """
        if not settings.enable_expiry_sell_strategy:
            return None

        if not self.is_expiry_day():
            return None

        if not self.is_entry_window():
            return None

        if len(df) < self.min_bars:
            return None

        spot = float(df.iloc[-1]["close"])
        if spot <= 0:
            return None

        conditions_met = ["Expiry day (Thursday)"]

        # --- Pre-conditions ---
        # VIX check
        if india_vix > self.params.get("max_vix", 20.0) and india_vix > 0:
            logger.debug(f"Expiry Sell: VIX {india_vix:.1f} too high")
            return None
        conditions_met.append(f"VIX {india_vix:.1f} < {self.params['max_vix']}")

        # Gap day → trending, skip
        if gap_type != "NONE":
            logger.debug("Expiry Sell: gap day, skipping")
            return None
        conditions_met.append("No gap day")

        # ORB range check
        if orb_high > 0 and orb_low > 0:
            orb_range_pct = ((orb_high - orb_low) / spot) * 100
            max_range = self.params.get("max_orb_range_pct", 0.6)
            if orb_range_pct > max_range:
                logger.debug(f"Expiry Sell: ORB range {orb_range_pct:.2f}% > {max_range}%")
                return None
            conditions_met.append(f"ORB range {orb_range_pct:.2f}% < {max_range}%")

        # Compute ATR for SL estimation
        atr = float(self.calculate_atr(df).iloc[-1])
        if atr <= 0:
            atr = spot * 0.005

        # --- Signal generation ---
        # ATM strike
        strike_gap = 50  # NIFTY
        atm_strike = round(spot / strike_gap) * strike_gap

        # Estimate premium from ATR/spot ratio (simplified)
        # On expiry day, ATM options have premium roughly = ATR * 0.4 per side
        estimated_ce_premium = atr * 0.4
        estimated_pe_premium = atr * 0.4
        combined = estimated_ce_premium + estimated_pe_premium

        min_premium = spot * self.params.get("min_premium_pct", 0.3) / 100
        if combined < min_premium:
            logger.debug(f"Expiry Sell: combined premium {combined:.2f} < min {min_premium:.2f}")
            return None

        conditions_met.append(f"Est. combined premium ₹{combined:.0f} > ₹{min_premium:.0f}")

        # SL and target
        sl_loss = combined * self.params.get("sl_multiplier", 1.5)
        target_profit = combined * self.params.get("target_pct_of_premium", 0.6)

        confidence = 55.0
        # Higher confidence if VIX is low (more predictable decay)
        if india_vix < 15:
            confidence += 5
            conditions_met.append("Low VIX bonus")
        # Higher confidence if narrow ORB
        if orb_high > 0 and orb_low > 0:
            orb_pct = ((orb_high - orb_low) / spot) * 100
            if orb_pct < 0.3:
                confidence += 5
                conditions_met.append("Very narrow ORB (strong range-bound)")

        confidence = min(confidence, 75.0)

        signal = TradeSignal(
            symbol=symbol,
            signal=SignalType.SELL,  # SELL signal for premium selling
            entry_price=round(combined, 2),  # Combined premium as entry
            stop_loss=round(combined + sl_loss, 2),  # SL = premium + 1.5x
            target=round(combined * 0.4, 2),  # Target = premium decay to 40%
            trailing_sl=round(combined * 1.2, 2),
            confidence=confidence,
            strategy_name=self.name,
            reasoning=(
                f"Expiry premium sell: ATM {atm_strike} straddle. "
                f"Est. premium ₹{combined:.0f} (CE ₹{estimated_ce_premium:.0f} + PE ₹{estimated_pe_premium:.0f}). "
                f"Target: ₹{target_profit:.0f} profit from theta decay. "
                f"Max loss: ₹{sl_loss:.0f}."
            ),
            conditions_met=conditions_met,
            timeframe=self.timeframe,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(
            f"Expiry Sell signal: ATM {atm_strike} straddle, "
            f"premium=₹{combined:.0f}, confidence={confidence:.0f}%"
        )

        return signal

    def evaluate_straddle_candidates(
        self,
        spot: float,
        option_quotes: Dict,
        strike_gap: int = 50,
    ) -> Optional[ExpiryStraddleSignal]:
        """
        Evaluate actual option quotes to build straddle signal.
        Called from options_engine when live quotes are available.

        Args:
            spot: current NIFTY spot price
            option_quotes: dict of {tradingsymbol: {ltp, iv, oi, bid, ask, ...}}
            strike_gap: strike interval
        """
        atm_strike = round(spot / strike_gap) * strike_gap
        conditions = ["Expiry day confirmed"]

        # Find ATM CE and PE
        ce_candidates = {}
        pe_candidates = {}
        for sym, data in option_quotes.items():
            if f"{int(atm_strike)}CE" in sym:
                ce_candidates[sym] = data
            elif f"{int(atm_strike)}PE" in sym:
                pe_candidates[sym] = data

        if not ce_candidates or not pe_candidates:
            return None

        # Pick best CE and PE
        ce_sym = max(ce_candidates, key=lambda k: ce_candidates[k].get("oi", 0))
        pe_sym = max(pe_candidates, key=lambda k: pe_candidates[k].get("oi", 0))

        ce_data = ce_candidates[ce_sym]
        pe_data = pe_candidates[pe_sym]

        ce_premium = ce_data.get("ltp", 0)
        pe_premium = pe_data.get("ltp", 0)
        combined = ce_premium + pe_premium

        if combined <= 0:
            return None

        min_premium_pct = self.params.get("min_premium_pct", 0.3)
        if combined < spot * min_premium_pct / 100:
            return None

        conditions.append(f"Combined premium ₹{combined:.2f}")
        conditions.append(f"CE premium ₹{ce_premium:.2f} (OI: {ce_data.get('oi', 0):,})")
        conditions.append(f"PE premium ₹{pe_premium:.2f} (OI: {pe_data.get('oi', 0):,})")

        sl_mult = self.params.get("sl_multiplier", 1.5)
        target_pct = self.params.get("target_pct_of_premium", 0.6)

        return ExpiryStraddleSignal(
            ce_symbol=ce_sym,
            pe_symbol=pe_sym,
            ce_strike=atm_strike,
            pe_strike=atm_strike,
            ce_premium=ce_premium,
            pe_premium=pe_premium,
            combined_premium=combined,
            spot_price=spot,
            max_loss=combined * sl_mult,
            target_profit=combined * target_pct,
            confidence=60.0,
            reasoning=f"ATM straddle sell: {atm_strike} CE+PE = ₹{combined:.0f}",
            conditions_met=conditions,
        )
