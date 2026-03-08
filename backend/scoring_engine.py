from __future__ import annotations
"""
Scoring Engine for NIFTY Options

Multi-factor scoring: ORB strength, Supertrend alignment, MACD momentum,
delta proximity, volume spike, risk/reward. Only top 10% scored candidates
are traded.
"""
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from config import settings


def parse_tradingsymbol(sym: str) -> str:
    """
    Convert Zerodha tradingsymbol to a readable format.
    NIFTY2631024350PE → NIFTY 24350 PE (10 Mar)
    NIFTY26MAR24350PE → NIFTY 24350 PE (Mar 26)
    """
    # Pattern: NAME + YY + M_CODE + DD + STRIKE + CE/PE
    # Month can be 1-digit (1-9) or letter code (O=Oct, N=Nov, D=Dec)
    m = re.match(
        r'^([A-Z]+?)(\d{2})(\d|[A-Z])(\d{2})(\d+)(CE|PE)$', sym
    )
    if m:
        name, yy, month_code, dd, strike, opt_type = m.groups()
        month_map = {
            '1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr',
            '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug',
            '9': 'Sep', 'O': 'Oct', 'N': 'Nov', 'D': 'Dec',
        }
        month_str = month_map.get(month_code, month_code)
        return f"{name} {strike} {opt_type} ({dd} {month_str})"
    # Fallback: try NAME + YY + MMMDD + STRIKE + CE/PE  (e.g. NIFTY26MAR2524350PE)
    m2 = re.match(
        r'^([A-Z]+?)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$', sym
    )
    if m2:
        name, yy, month_str, dd, strike, opt_type = m2.groups()
        return f"{name} {strike} {opt_type} ({dd} {month_str.title()})"
    return sym


@dataclass
class OptionCandidate:
    """A single option contract being evaluated."""
    tradingsymbol: str          # e.g. NIFTY2630620200CE
    instrument_token: int
    strike: float
    option_type: str            # CE or PE
    expiry: str
    ltp: float                  # last traded price of the option
    spot_price: float           # NIFTY spot
    delta: float                # estimated delta
    iv: float                   # implied volatility %
    volume: int
    oi: int
    bid: float
    ask: float
    # OI analysis
    oi_change_pct: float = 0.0      # OI change % vs previous day for THIS strike
    ce_oi_change_pct: float = 0.0   # ATM CE OI change % (chain-wide)
    pe_oi_change_pct: float = 0.0   # ATM PE OI change % (chain-wide)
    prev_oi: int = 0                # previous day OI for this strike
    # IV / Expiry / Gap / Multi-TF context
    iv_percentile: float = 50.0     # current IV percentile vs N-day range (0-100)
    tte_days: float = 5.0           # time to expiry in calendar days
    is_expiry_day: bool = False     # True if option expires today
    gap_type: str = "NONE"          # NONE, GAP_UP, LARGE_GAP_UP, GAP_DOWN, LARGE_GAP_DOWN
    gap_pct: float = 0.0            # gap open % vs prev close
    supertrend_15m: int = 0         # 15-min supertrend direction
    supertrend_1h: int = 0          # 1-hour supertrend direction
    # Signal context
    orb_high: float = 0.0
    orb_low: float = 0.0
    supertrend_dir: int = 0     # 1 = bullish, -1 = bearish
    macd_hist: float = 0.0
    macd_hist_prev: float = 0.0
    vol_spike: bool = False
    atr: float = 0.0
    # OI interpretation
    oi_interpretation: str = ""    # e.g. "Long buildup", "Short covering"
    # Computed
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    rank_pct: float = 0.0       # percentile rank (100 = best)
    # Trade levels (computed after scoring)
    entry_price: float = 0.0    # suggested entry = LTP (or ask for BUY)
    stoploss: float = 0.0       # SL on premium
    target1: float = 0.0        # 1st target (1:1 RR)
    target2: float = 0.0        # 2nd target (1:2 RR)
    target3: float = 0.0        # 3rd target (1:3 RR)
    risk_reward_pct: float = 0.0  # risk as % of entry
    # Price source tracking
    price_source: str = "estimated"  # "live", "estimated", "stale"

    @property
    def display_name(self) -> str:
        return parse_tradingsymbol(self.tradingsymbol)


class ScoringEngine:
    """
    Scores option candidates on 0-100 scale across multiple factors.
    Weights are configurable.
    """

    DEFAULT_WEIGHTS = {
        "orb_strength": 12,
        "supertrend": 10,
        "macd_momentum": 10,
        "delta_quality": 12,
        "volume_spike": 6,
        "risk_reward": 6,
        "spread_quality": 4,
        "oi_buildup": 12,         # OI change supports direction (long/short buildup)
        "oi_pcr_shift": 8,        # CE vs PE OI change ratio favours direction
        "iv_percentile": 8,       # Prefer cheap IV options (low percentile)
        "expiry_day_penalty": 4,  # Penalise expiry-day OTM options
        "multi_tf_alignment": 8,  # 15m + 1h supertrend alignment
    }

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        total = sum(self.weights.values())
        # Normalise to 100
        if total != 100:
            for k in self.weights:
                self.weights[k] = self.weights[k] / total * 100

    # ------------------------------------------------------------------
    # Individual scoring factors (each returns 0.0 – 1.0)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_orb_strength(c: OptionCandidate) -> float:
        """How far price has broken past ORB range."""
        orb_range = c.orb_high - c.orb_low
        if orb_range <= 0:
            return 0.0
        if c.option_type == "CE":
            breakout = c.spot_price - c.orb_high
        else:
            breakout = c.orb_low - c.spot_price
        ratio = max(0, breakout) / orb_range
        return min(1.0, ratio)  # cap at 1.0

    @staticmethod
    def _score_supertrend(c: OptionCandidate) -> float:
        """1.0 if supertrend direction matches trade direction."""
        if c.option_type == "CE" and c.supertrend_dir == 1:
            return 1.0
        if c.option_type == "PE" and c.supertrend_dir == -1:
            return 1.0
        return 0.0

    @staticmethod
    def _score_macd_momentum(c: OptionCandidate) -> float:
        """MACD histogram strength and direction."""
        if c.option_type == "CE":
            if c.macd_hist <= 0:
                return 0.0
            rising = 1.0 if c.macd_hist > c.macd_hist_prev else 0.5
        else:
            if c.macd_hist >= 0:
                return 0.0
            rising = 1.0 if c.macd_hist < c.macd_hist_prev else 0.5
        # Scale by absolute magnitude (capped)
        magnitude = min(abs(c.macd_hist) / max(c.atr * 0.1, 0.01), 1.0)
        return rising * 0.6 + magnitude * 0.4

    @staticmethod
    def _score_delta_quality(c: OptionCandidate) -> float:
        """
        Ideal delta band is 0.40–0.50 (ATM-ish).
        0.3–0.6 acceptable. Outside = 0.
        """
        d = abs(c.delta)
        if d < 0.30 or d > 0.60:
            return 0.0
        # Peak at 0.45
        distance = abs(d - 0.45)
        return max(0.0, 1.0 - distance / 0.15)

    @staticmethod
    def _score_volume_spike(c: OptionCandidate) -> float:
        return 1.0 if c.vol_spike else 0.0

    @staticmethod
    def _score_risk_reward(c: OptionCandidate) -> float:
        """Higher R:R is better. 2.0+ gets full score."""
        orb_range = c.orb_high - c.orb_low
        if orb_range <= 0:
            return 0.5
        risk = c.atr * 1.5
        reward = orb_range * 2.0
        if risk <= 0:
            return 0.0
        rr = reward / risk
        return min(1.0, rr / 2.0)

    @staticmethod
    def _score_spread_quality(c: OptionCandidate) -> float:
        """Tight bid-ask spread is better."""
        if c.ask <= 0:
            return 0.0
        spread_pct = (c.ask - c.bid) / c.ask
        if spread_pct > 0.05:
            return 0.0
        return max(0.0, 1.0 - spread_pct / 0.05)

    @staticmethod
    def _score_oi_buildup(c: OptionCandidate) -> float:
        """
        OI + Price action interpretation for the traded strike:
          CE buy: want +OI (long buildup) → fresh longs entering
          PE buy: want +OI (long buildup on PE) → fresh shorts/puts entering
        Higher OI change % in the direction of the trade = better.
        Negative OI change (unwinding) is a warning.
        """
        oi_chg = c.oi_change_pct  # % change for THIS strike & type

        if oi_chg == 0:
            return 0.3  # neutral — no data or flat

        # For both CE and PE buys: rising OI = fresh positions = good
        # Falling OI = unwinding = bad
        if oi_chg > 0:
            # Long buildup (OI rising + price rising) or Short buildup (OI rising + price falling)
            # Either way, rising OI on the strike we're buying = high conviction
            score = min(1.0, oi_chg / 15.0)  # 15%+ = full score
            return score
        else:
            # Unwinding — OI falling means positions closing
            # Penalise moderately: -10% OI change → 0.0
            return max(0.0, 1.0 + oi_chg / 10.0)

    @staticmethod
    def _score_oi_pcr_shift(c: OptionCandidate) -> float:
        """
        PCR shift: CE vs PE OI change comparison (chain-wide at ATM strikes).
        For CE (bullish) trade:
          - CE OI falling + PE OI rising = bullish (short covering + fresh puts being sold)
          - CE OI rising + PE OI falling = bearish (fresh call writing)
        For PE (bearish) trade:
          - PE OI rising + CE OI falling = bearish confirmation
          - PE OI falling + CE OI rising = bullish (bad for PE)

        Screenshot example: CE OI -5.1%, PE OI +7.4% → strong bearish → good for PE buy.
        """
        ce_chg = c.ce_oi_change_pct
        pe_chg = c.pe_oi_change_pct

        if ce_chg == 0 and pe_chg == 0:
            return 0.3  # neutral

        if c.option_type == "CE":  # Bullish trade
            # Ideal: CE OI falling (short covering), PE OI rising (put writers confident)
            # Or: CE OI rising moderately (fresh longs)
            # Bad: CE OI rising sharply (call writing) + PE OI falling
            bullish_score = 0.0
            # PE OI rising = put writers adding → they expect support → bullish
            if pe_chg > 0:
                bullish_score += min(0.5, pe_chg / 10.0)
            # CE OI falling = short covering → bullish
            if ce_chg < 0:
                bullish_score += min(0.5, abs(ce_chg) / 10.0)
            # CE OI rising moderately can be fresh longs too
            elif 0 < ce_chg < 5:
                bullish_score += 0.15
            # Heavy CE OI buildup = call writing = bearish
            elif ce_chg > 10:
                bullish_score -= 0.3
            return max(0.0, min(1.0, bullish_score))

        else:  # PE — Bearish trade
            # Ideal: PE OI rising (fresh puts/short buildup), CE OI falling
            bearish_score = 0.0
            # PE OI rising = fresh put interest → bearish confirmation
            if pe_chg > 0:
                bearish_score += min(0.5, pe_chg / 10.0)
            # CE OI falling = call unwinding → bearish
            if ce_chg < 0:
                bearish_score += min(0.5, abs(ce_chg) / 10.0)
            # CE OI rising = call writing = bearish (writers expect fall)
            elif ce_chg > 5:
                bearish_score += min(0.3, ce_chg / 20.0)
            # PE OI falling heavily = put unwinding = bad for PE
            if pe_chg < -5:
                bearish_score -= 0.3
            return max(0.0, min(1.0, bearish_score))

    @staticmethod
    def _interpret_oi(c: OptionCandidate) -> str:
        """
        Generate human-readable OI interpretation.
        Price + OI matrix:
          Price UP   + OI UP   = Long Buildup (bullish)
          Price UP   + OI DOWN = Short Covering (mildly bullish)
          Price DOWN + OI UP   = Short Buildup (bearish)
          Price DOWN + OI DOWN = Long Unwinding (mildly bearish)
        """
        oi_chg = c.oi_change_pct
        ce_chg = c.ce_oi_change_pct
        pe_chg = c.pe_oi_change_pct

        parts = []

        # Strike-level interpretation
        if oi_chg > 2:
            parts.append(f"OI +{oi_chg:.1f}% (fresh positions)")
        elif oi_chg < -2:
            parts.append(f"OI {oi_chg:.1f}% (unwinding)")

        # Chain-wide PCR interpretation
        if ce_chg != 0 or pe_chg != 0:
            parts.append(f"CE OI {ce_chg:+.1f}%")
            parts.append(f"PE OI {pe_chg:+.1f}%")

        if not parts:
            return "OI data unavailable"
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # IV Percentile scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _score_iv_percentile(c: OptionCandidate) -> float:
        """
        Prefer options with low IV percentile (cheap premium).
        0-30th percentile = 1.0 (cheap, good for buying)
        30-60th = 0.7
        60-80th = 0.3
        80-100th = 0.0 (expensive, avoid buying)
        If IV is above reject threshold, return 0.
        """
        ivp = c.iv_percentile
        if ivp >= settings.iv_reject_threshold:
            return 0.0  # Too expensive
        if ivp <= 30:
            return 1.0
        elif ivp <= 60:
            return 0.7
        elif ivp <= 80:
            return 0.3
        return 0.1

    # ------------------------------------------------------------------
    # Expiry-day penalty scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _score_expiry_day(c: OptionCandidate) -> float:
        """
        On expiry day:
          - ATM options (delta ~0.45-0.55) = OK (0.7)
          - OTM options (delta < 0.35) = Bad (rapid theta decay) (0.0)
          - Non-expiry day = no penalty (1.0)
          - tte_days > 3 = full score (1.0)
        """
        if not c.is_expiry_day:
            return 1.0  # No penalty
        d = abs(c.delta)
        if d >= 0.40:
            return 0.7  # ATM-ish on expiry = acceptable
        elif d >= 0.35:
            return 0.3  # Slightly OTM on expiry
        return 0.0  # Far OTM on expiry = terrible for buying

    # ------------------------------------------------------------------
    # Multi-timeframe alignment scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _score_multi_tf(c: OptionCandidate) -> float:
        """
        Score based on alignment across 5m, 15m, 1h supertrend directions.
        1.0 = all timeframes aligned with trade direction
        0.5 = 2 of 3 aligned
        0.0 = conflicting directions
        """
        # 5-min direction comes from c.supertrend_dir
        tf_5m = c.supertrend_dir
        tf_15m = c.supertrend_15m
        tf_1h = c.supertrend_1h

        # If no multi-TF data, return neutral
        if tf_15m == 0 and tf_1h == 0:
            return 0.5

        # Determine expected direction based on option type
        expected = 1 if c.option_type == "CE" else -1

        aligned = 0
        total = 0
        for d in [tf_5m, tf_15m, tf_1h]:
            if d != 0:
                total += 1
                if d == expected:
                    aligned += 1

        if total == 0:
            return 0.5
        ratio = aligned / total
        if ratio >= 0.9:
            return 1.0
        elif ratio >= 0.6:
            return 0.5
        return 0.0

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def score_candidate(self, c: OptionCandidate, levels=None) -> OptionCandidate:
        # Generate OI interpretation label
        c.oi_interpretation = self._interpret_oi(c)

        factors = {
            "orb_strength": self._score_orb_strength(c),
            "supertrend": self._score_supertrend(c),
            "macd_momentum": self._score_macd_momentum(c),
            "delta_quality": self._score_delta_quality(c),
            "volume_spike": self._score_volume_spike(c),
            "risk_reward": self._score_risk_reward(c),
            "spread_quality": self._score_spread_quality(c),
            "oi_buildup": self._score_oi_buildup(c),
            "oi_pcr_shift": self._score_oi_pcr_shift(c),
            "iv_percentile": self._score_iv_percentile(c),
            "expiry_day_penalty": self._score_expiry_day(c),
            "multi_tf_alignment": self._score_multi_tf(c),
        }
        total = sum(factors[k] * self.weights.get(k, 0) for k in factors)
        c.score = round(total, 2)
        c.score_breakdown = {k: round(v * self.weights.get(k, 0), 2) for k, v in factors.items()}

        # Compute trade levels using smart SL engine (VWAP / CPR / S-R aware)
        # Import here to avoid circular dependency
        from level_calculator import LevelCalculator, IntraDayLevels
        from smart_sl_engine import SmartSLEngine

        # Use full structural levels if provided, otherwise build minimal from candidate
        if levels is None:
            levels = IntraDayLevels(
                orb_high=c.orb_high,
                orb_low=c.orb_low,
                atr=c.atr,
            )

        sl_engine = SmartSLEngine(
            min_sl_atr_mult=0.5,
            max_sl_atr_mult=2.0,
            default_sl_atr_mult=1.0,
            target_min_rr=1.5,
        )

        # Compute on spot price, then map to option premium via delta
        direction = "BUY" if c.option_type == "CE" else "SELL"
        if direction == "BUY":
            smart = sl_engine.compute_buy_levels(c.spot_price, levels)
        else:
            smart = sl_engine.compute_sell_levels(c.spot_price, levels)

        # Map spot-level SL/targets to option premium using delta
        d = abs(c.delta) if abs(c.delta) > 0 else 0.4

        # Entry price = current LTP (what you'd actually pay now)
        c.entry_price = round(c.ltp, 2)

        if direction == "BUY":
            sl_move = (c.spot_price - smart.stop_loss) * d
            t1_move = (smart.target1 - c.spot_price) * d
            t2_move = (smart.target2 - c.spot_price) * d
            t3_move = (smart.target3 - c.spot_price) * d
        else:
            sl_move = (smart.stop_loss - c.spot_price) * d
            t1_move = (c.spot_price - smart.target1) * d
            t2_move = (c.spot_price - smart.target2) * d
            t3_move = (c.spot_price - smart.target3) * d

        c.stoploss = round(max(c.entry_price - sl_move, c.entry_price * 0.70), 2)
        c.target1 = round(c.entry_price + t1_move, 2)
        c.target2 = round(c.entry_price + t2_move, 2)
        c.target3 = round(c.entry_price + t3_move, 2)

        # Ensure targets are in ascending order on option premium side
        targets_sorted = sorted([c.target1, c.target2, c.target3])
        c.target1, c.target2, c.target3 = targets_sorted

        risk = c.entry_price - c.stoploss
        c.risk_reward_pct = round((risk / c.entry_price) * 100, 1) if c.entry_price > 0 else 0

        return c

    def rank_candidates(
        self, candidates: List[OptionCandidate], top_pct: float = 10.0,
        levels=None,
    ) -> List[OptionCandidate]:
        """
        Score all candidates, assign percentile rank,
        return only top `top_pct`% (default 10%).
        """
        if not candidates:
            return []

        for c in candidates:
            self.score_candidate(c, levels=levels)

        # Sort descending by score
        candidates.sort(key=lambda x: x.score, reverse=True)

        n = len(candidates)
        for i, c in enumerate(candidates):
            c.rank_pct = round((1 - i / max(n, 1)) * 100, 1)

        cutoff = max(1, math.ceil(n * top_pct / 100))
        return candidates[:cutoff]

    def score_summary(self, candidates: List[OptionCandidate]) -> Dict:
        """Dashboard-friendly summary."""
        if not candidates:
            return {"count": 0, "top": [], "avg_score": 0, "timestamp": datetime.now().isoformat()}
        return {
            "count": len(candidates),
            "avg_score": round(sum(c.score for c in candidates) / len(candidates), 2),
            "max_score": round(max(c.score for c in candidates), 2),
            "min_score": round(min(c.score for c in candidates), 2),
            "top": [
                {
                    "symbol": c.tradingsymbol,
                    "display_name": c.display_name,
                    "instrument_token": c.instrument_token,
                    "strike": c.strike,
                    "type": c.option_type,
                    "expiry": c.expiry,
                    "delta": c.delta,
                    "score": c.score,
                    "rank_pct": c.rank_pct,
                    "breakdown": c.score_breakdown,
                    "ltp": c.ltp,
                    "entry": c.entry_price,
                    "stoploss": c.stoploss,
                    "target1": c.target1,
                    "target2": c.target2,
                    "target3": c.target3,
                    "risk_reward_pct": c.risk_reward_pct,
                    "volume": c.volume,
                    "iv": c.iv,
                    "oi": c.oi,
                    "oi_change_pct": c.oi_change_pct,
                    "ce_oi_change_pct": c.ce_oi_change_pct,
                    "pe_oi_change_pct": c.pe_oi_change_pct,
                    "oi_interpretation": c.oi_interpretation,
                    "price_source": c.price_source,
                }
                for c in candidates[:10]
            ],
            "timestamp": datetime.now().isoformat(),
        }
