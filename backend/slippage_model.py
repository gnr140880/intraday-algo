"""
Slippage Model

Estimates realistic fill prices accounting for:
  - Bid-ask spread impact
  - Market impact (large orders)
  - Fixed friction component (exchange fees, taxes)
  - Time-of-day volatility factor

Used by:
  - Backtester: to simulate realistic fills
  - Live trading: to adjust SL/targets for expected slippage
  - Dashboard: to show estimated vs actual slippage

Usage:
    from slippage_model import slippage_model, estimate_slippage
    fill_price = slippage_model.estimate_fill_price(ltp=150, bid=149, ask=151, side="BUY")
    slippage_pct = slippage_model.estimate_slippage_pct(ltp=150, bid=149, ask=151)
"""
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, time

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SlippageEstimate:
    """Detailed slippage breakdown."""
    estimated_fill: float        # expected fill price
    slippage_amount: float       # absolute slippage in ₹
    slippage_pct: float          # slippage as % of LTP
    spread_component: float      # slippage due to bid-ask spread
    impact_component: float      # slippage due to market impact
    fixed_component: float       # fixed friction (fees, etc.)
    volatility_factor: float     # time-of-day adjustment factor


class SlippageModel:
    """
    Multi-component slippage estimator for NIFTY/BANKNIFTY options.
    
    Components:
    1. Spread cost: half the bid-ask spread (crossing the spread)
    2. Market impact: proportional to order size vs average volume
    3. Fixed friction: exchange fees, STT, taxes (~0.05%)
    4. Volatility factor: higher slippage near open/close
    """

    def __init__(
        self,
        fixed_pct: Optional[float] = None,
        impact_coefficient: float = 0.1,
        enabled: Optional[bool] = None,
    ):
        self.fixed_pct = fixed_pct if fixed_pct is not None else settings.slippage_fixed_pct
        self.impact_coefficient = impact_coefficient
        self.enabled = enabled if enabled is not None else settings.slippage_model_enabled

    def _get_volatility_factor(self) -> float:
        """
        Time-of-day volatility adjustment.
        Higher slippage near market open and close.
        """
        now = datetime.now().time()

        # First 15 min: highest volatility
        if now < time(9, 30):
            return 2.0
        # First 30 min: high volatility
        if now < time(9, 45):
            return 1.5
        # Last 30 min: high volatility
        if now >= time(15, 0):
            return 1.5
        # Last 15 min: highest volatility
        if now >= time(15, 15):
            return 2.0
        # Lunch hour: low volatility
        if time(12, 0) <= now <= time(13, 30):
            return 0.7
        # Normal hours
        return 1.0

    def estimate_fill_price(
        self,
        ltp: float,
        bid: float = 0,
        ask: float = 0,
        side: str = "BUY",
        order_qty: int = 0,
        avg_volume: int = 0,
    ) -> float:
        """
        Estimate the realistic fill price for an order.
        
        Args:
            ltp: last traded price
            bid: best bid price (0 if unavailable)
            ask: best ask price (0 if unavailable)
            side: "BUY" or "SELL"
            order_qty: order quantity (for market impact)
            avg_volume: average trading volume (for market impact)
        
        Returns:
            Estimated fill price (after slippage)
        """
        if not self.enabled or ltp <= 0:
            return ltp

        estimate = self.estimate_slippage(ltp, bid, ask, side, order_qty, avg_volume)
        return estimate.estimated_fill

    def estimate_slippage(
        self,
        ltp: float,
        bid: float = 0,
        ask: float = 0,
        side: str = "BUY",
        order_qty: int = 0,
        avg_volume: int = 0,
    ) -> SlippageEstimate:
        """
        Full slippage breakdown.
        
        Returns SlippageEstimate with all components.
        """
        if ltp <= 0:
            return SlippageEstimate(
                estimated_fill=ltp, slippage_amount=0, slippage_pct=0,
                spread_component=0, impact_component=0, fixed_component=0,
                volatility_factor=1.0,
            )

        vol_factor = self._get_volatility_factor()

        # 1. Spread component
        spread_cost = 0.0
        if bid > 0 and ask > 0 and ask > bid:
            spread = ask - bid
            spread_cost = spread / 2.0  # half the spread
        else:
            # Estimate spread from LTP (typical for NIFTY options: 0.5-2 points)
            spread_cost = max(0.5, ltp * 0.003)  # ~0.3% of LTP

        # 2. Market impact (large orders move the market)
        impact_cost = 0.0
        if order_qty > 0 and avg_volume > 0:
            order_pct = order_qty / avg_volume
            # Impact proportional to sqrt of order fraction
            import math
            impact_cost = ltp * self.impact_coefficient * math.sqrt(order_pct)

        # 3. Fixed friction (fees, taxes)
        fixed_cost = ltp * (self.fixed_pct / 100)

        # 4. Apply volatility factor
        total_slippage = (spread_cost + impact_cost + fixed_cost) * vol_factor

        # Direction: BUY → higher fill, SELL → lower fill
        if side.upper() == "BUY":
            fill_price = ltp + total_slippage
        else:
            fill_price = ltp - total_slippage

        # Ensure fill is within reasonable bounds
        if side.upper() == "BUY" and ask > 0:
            fill_price = min(fill_price, ask * 1.01)  # Never more than 1% above ask
        elif side.upper() == "SELL" and bid > 0:
            fill_price = max(fill_price, bid * 0.99)  # Never less than 1% below bid

        slippage_pct = (total_slippage / ltp * 100) if ltp > 0 else 0

        return SlippageEstimate(
            estimated_fill=round(fill_price, 2),
            slippage_amount=round(total_slippage, 2),
            slippage_pct=round(slippage_pct, 4),
            spread_component=round(spread_cost * vol_factor, 2),
            impact_component=round(impact_cost * vol_factor, 2),
            fixed_component=round(fixed_cost * vol_factor, 2),
            volatility_factor=round(vol_factor, 2),
        )

    def estimate_slippage_pct(
        self,
        ltp: float,
        bid: float = 0,
        ask: float = 0,
    ) -> float:
        """Quick slippage percentage estimate (for scoring/filtering)."""
        if not self.enabled or ltp <= 0:
            return 0.0
        est = self.estimate_slippage(ltp, bid, ask)
        return est.slippage_pct

    def adjust_entry_price(
        self,
        entry_price: float,
        bid: float = 0,
        ask: float = 0,
        side: str = "BUY",
    ) -> float:
        """
        Return a slippage-adjusted entry price for SL/target computation.
        Used by risk manager to set realistic profit expectations.
        """
        return self.estimate_fill_price(entry_price, bid, ask, side)

    def adjust_sl_for_slippage(
        self,
        sl_price: float,
        side: str = "BUY",
        buffer_mult: float = 1.0,
    ) -> float:
        """
        Widen SL to account for slippage on exit.
        For BUY positions, SL is a SELL → lower fill expected.
        """
        if not self.enabled or sl_price <= 0:
            return sl_price

        # Estimate slippage at SL level
        est = self.estimate_slippage(sl_price, side="SELL" if side == "BUY" else "BUY")
        adjustment = est.slippage_amount * buffer_mult

        if side.upper() == "BUY":
            # Exit is SELL → fill lower than expected → widen SL down
            return round(sl_price - adjustment, 2)
        else:
            # Exit is BUY → fill higher than expected → widen SL up
            return round(sl_price + adjustment, 2)


# Global instance
slippage_model = SlippageModel()
