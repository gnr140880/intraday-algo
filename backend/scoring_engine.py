"""
Scoring Engine for NIFTY Options

Multi-factor scoring: ORB strength, Supertrend alignment, MACD momentum,
delta proximity, volume spike, risk/reward. Only top 10% scored candidates
are traded.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


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
    # Signal context
    orb_high: float
    orb_low: float
    supertrend_dir: int         # 1 = bullish, -1 = bearish
    macd_hist: float
    macd_hist_prev: float
    vol_spike: bool
    atr: float
    # Computed
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    rank_pct: float = 0.0       # percentile rank (100 = best)


class ScoringEngine:
    """
    Scores option candidates on 0-100 scale across multiple factors.
    Weights are configurable.
    """

    DEFAULT_WEIGHTS = {
        "orb_strength": 20,
        "supertrend": 15,
        "macd_momentum": 15,
        "delta_quality": 20,
        "volume_spike": 10,
        "risk_reward": 10,
        "spread_quality": 10,
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

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def score_candidate(self, c: OptionCandidate) -> OptionCandidate:
        factors = {
            "orb_strength": self._score_orb_strength(c),
            "supertrend": self._score_supertrend(c),
            "macd_momentum": self._score_macd_momentum(c),
            "delta_quality": self._score_delta_quality(c),
            "volume_spike": self._score_volume_spike(c),
            "risk_reward": self._score_risk_reward(c),
            "spread_quality": self._score_spread_quality(c),
        }
        total = sum(factors[k] * self.weights.get(k, 0) for k in factors)
        c.score = round(total, 2)
        c.score_breakdown = {k: round(v * self.weights.get(k, 0), 2) for k, v in factors.items()}
        return c

    def rank_candidates(
        self, candidates: List[OptionCandidate], top_pct: float = 10.0
    ) -> List[OptionCandidate]:
        """
        Score all candidates, assign percentile rank,
        return only top `top_pct`% (default 10%).
        """
        if not candidates:
            return []

        for c in candidates:
            self.score_candidate(c)

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
                    "strike": c.strike,
                    "type": c.option_type,
                    "delta": c.delta,
                    "score": c.score,
                    "rank_pct": c.rank_pct,
                    "breakdown": c.score_breakdown,
                    "ltp": c.ltp,
                    "volume": c.volume,
                    "iv": c.iv,
                }
                for c in candidates[:10]
            ],
            "timestamp": datetime.now().isoformat(),
        }
