"""
NIFTY Options ORB (Opening Range Breakout) Strategy

Flow:
  1. Capture NIFTY spot ORB (first 15 min high/low: 9:15–9:30)
  2. Supertrend on NIFTY 5-min as trend filter
  3. MACD on NIFTY 5-min as confirmation
  4. Volume spike detection on option contract
  5. Delta filter 0.3–0.6 applied by scoring engine upstream

Signals are generated AFTER 9:30 AM when ORB window is complete.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, time
from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType
from level_calculator import LevelCalculator, IntraDayLevels
from smart_sl_engine import SmartSLEngine, SmartLevels, compute_smart_levels
from config import settings


class NiftyOptionsORBStrategy(BaseStrategy):
    name = "NIFTY Options ORB"
    description = (
        "Opening Range Breakout on NIFTY spot with Supertrend trend filter, "
        "MACD confirmation, and volume spike. Trades NIFTY options (CE/PE) "
        "filtered by delta 0.3–0.6."
    )
    timeframe = "5m"
    min_bars = 30

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.level_calc = LevelCalculator()
        self.sl_engine = SmartSLEngine(
            min_sl_atr_mult=0.5,
            max_sl_atr_mult=2.0,
            default_sl_atr_mult=1.0,
            target_min_rr=1.5,
        )
        self._cached_levels: Optional[IntraDayLevels] = None

    def default_params(self) -> Dict[str, Any]:
        return {
            # ORB
            "orb_minutes": 15,  # first 15 min from 9:15
            "orb_buffer_pct": 0.05,  # 0.05% buffer above/below ORB
            # Supertrend
            "st_atr_period": 10,
            "st_multiplier": 3.0,
            # MACD
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            # Volume
            "vol_spike_mult": 1.5,  # volume must be 1.5x of 20-bar avg
            "vol_lookback": 20,
            # Risk (used as fallback only; smart SL engine overrides)
            "trailing_atr_mult": 1.5,
            "risk_reward_min": 2.0,
            # Smart SL params
            "min_sl_atr_mult": 0.5,
            "max_sl_atr_mult": 2.0,
            "default_sl_atr_mult": 1.0,
            "target_min_rr": 1.5,
        }

    # ------------------------------------------------------------------
    # ORB capture
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the 'date' column is timezone-naive IST for comparison."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        return df

    def compute_orb(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Return (orb_high, orb_low) from first 15 min of the session."""
        if "date" not in df.columns:
            return None, None

        df = self._normalize_dates(df)
        today = df["date"].dt.date.iloc[-1]
        market_open = pd.Timestamp(datetime.combine(today, time(9, 15)))
        orb_end = pd.Timestamp(datetime.combine(today, time(9, 30)))

        orb_bars = df[(df["date"] >= market_open) & (df["date"] < orb_end)]
        if orb_bars.empty:
            return None, None
        return float(orb_bars["high"].max()), float(orb_bars["low"].min())

    # ------------------------------------------------------------------
    # Supertrend (uses parent ATR helper)
    # ------------------------------------------------------------------
    def compute_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params["st_atr_period"]
        mult = self.params["st_multiplier"]
        atr = self.calculate_atr(df, period)

        hl2 = (df["high"] + df["low"]) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        direction = pd.Series(0, index=df.index, dtype=int)
        st_val = pd.Series(np.nan, index=df.index, dtype=float)

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

            st_val.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

        df = df.copy()
        df["st_dir"] = direction
        df["st_val"] = st_val
        return df

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.params["macd_fast"]
        slow = self.params["macd_slow"]
        sig = self.params["macd_signal"]

        df = df.copy()
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=sig, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]
        return df

    # ------------------------------------------------------------------
    # Volume spike
    # ------------------------------------------------------------------
    def has_volume_spike(self, df: pd.DataFrame) -> bool:
        if "volume" not in df.columns:
            return True  # pass-through if volume unavailable
        lookback = self.params["vol_lookback"]
        mult = self.params["vol_spike_mult"]
        if len(df) < lookback + 1:
            return False
        avg_vol = df["volume"].iloc[-(lookback + 1):-1].mean()
        curr_vol = df["volume"].iloc[-1]
        return avg_vol > 0 and curr_vol >= avg_vol * mult

    # ------------------------------------------------------------------
    # Main signal generation (on NIFTY spot 5-min data)
    # ------------------------------------------------------------------
    def generate_signal(self, df: pd.DataFrame, symbol: str, oi_data: Dict = None) -> Optional[TradeSignal]:
        if len(df) < self.min_bars:
            return None

        # Normalize timezone before any date comparison
        df = self._normalize_dates(df)

        orb_high, orb_low = self.compute_orb(df)
        if orb_high is None:
            return None

        # Only trade after ORB window
        if "date" in df.columns:
            last_ts = pd.to_datetime(df["date"].iloc[-1])
            today = last_ts.date()
            orb_end = pd.Timestamp(datetime.combine(today, time(9, 30)))
            if last_ts < orb_end:
                return None
            # Don't generate new signals after 3:00 PM
            cutoff = pd.Timestamp(datetime.combine(today, time(15, 0)))
            if last_ts > cutoff:
                return None

        df = self.compute_supertrend(df)
        df = self.compute_macd(df)
        atr = self.calculate_atr(df).iloc[-1]

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]

        # ── GAP DAY CLASSIFICATION ──
        gap_type = "NONE"
        gap_pct = 0.0
        if "date" in df.columns:
            df_norm = self._normalize_dates(df.copy())
            df_norm["session_date"] = df_norm["date"].dt.date
            dates = sorted(df_norm["session_date"].unique())
            if len(dates) >= 2:
                prev_date = dates[-2]
                prev_close = float(df_norm[df_norm["session_date"] == prev_date]["close"].iloc[-1])
                today_open = float(df_norm[df_norm["session_date"] == dates[-1]]["open"].iloc[0])
                if prev_close > 0:
                    gap_pct = (today_open - prev_close) / prev_close * 100
                    if gap_pct >= settings.large_gap_threshold_pct:
                        gap_type = "LARGE_GAP_UP"
                    elif gap_pct >= settings.gap_threshold_pct:
                        gap_type = "GAP_UP"
                    elif gap_pct <= -settings.large_gap_threshold_pct:
                        gap_type = "LARGE_GAP_DOWN"
                    elif gap_pct <= -settings.gap_threshold_pct:
                        gap_type = "GAP_DOWN"

        # Adjust ORB buffer for gap days
        base_buffer_pct = self.params["orb_buffer_pct"]
        if gap_type != "NONE":
            base_buffer_pct *= settings.gap_day_buffer_mult
        buffer = orb_high * (base_buffer_pct / 100)

        conditions = []
        signal_type = None
        signal_mode = None  # "BREAKOUT", "SUSTAINED", "TREND"

        # ── MODE 1: FRESH ORB Breakout (exact candle) ──
        if price > orb_high + buffer and prev["close"] <= orb_high + buffer:
            conditions.append(f"ORB breakout UP: {price:.2f} > {orb_high:.2f}")
            if curr["st_dir"] == 1:
                conditions.append("Supertrend BULLISH")
            else:
                return None
            if curr["macd_hist"] > 0:
                conditions.append(f"MACD histogram positive: {curr['macd_hist']:.4f}")
                if curr["macd_hist"] > prev["macd_hist"]:
                    conditions.append("MACD histogram rising")
            else:
                return None
            if self.has_volume_spike(df):
                conditions.append("Volume spike detected")
            signal_type = SignalType.BUY
            signal_mode = "BREAKOUT"

        elif price < orb_low - buffer and prev["close"] >= orb_low - buffer:
            conditions.append(f"ORB breakdown: {price:.2f} < {orb_low:.2f}")
            if curr["st_dir"] == -1:
                conditions.append("Supertrend BEARISH")
            else:
                return None
            if curr["macd_hist"] < 0:
                conditions.append(f"MACD histogram negative: {curr['macd_hist']:.4f}")
                if curr["macd_hist"] < prev["macd_hist"]:
                    conditions.append("MACD histogram falling")
            else:
                return None
            if self.has_volume_spike(df):
                conditions.append("Volume spike detected")
            signal_type = SignalType.SELL
            signal_mode = "BREAKOUT"

        # ── MODE 2: SUSTAINED breakout (price already beyond ORB) ──
        # Supertrend OR MACD confirmation is sufficient (price already proved direction)
        elif price > orb_high + buffer:
            conditions.append(f"Sustained above ORB high: {price:.2f} > {orb_high:.2f}")
            st_bull = curr["st_dir"] == 1
            macd_bull = curr["macd_hist"] > 0
            if st_bull:
                conditions.append("Supertrend BULLISH")
            if macd_bull:
                conditions.append(f"MACD positive: {curr['macd_hist']:.4f}")
            if st_bull or macd_bull:
                signal_type = SignalType.BUY
                signal_mode = "SUSTAINED"

        elif price < orb_low - buffer:
            conditions.append(f"Sustained below ORB low: {price:.2f} < {orb_low:.2f}")
            st_bear = curr["st_dir"] == -1
            macd_bear = curr["macd_hist"] < 0
            if st_bear:
                conditions.append("Supertrend BEARISH")
            if macd_bear:
                conditions.append(f"MACD negative: {curr['macd_hist']:.4f}")
            if st_bear or macd_bear:
                signal_type = SignalType.SELL
                signal_mode = "SUSTAINED"

        # ── MODE 3: TREND signal (inside ORB – config-controlled, strict) ──
        else:
            if not settings.enable_trend_mode:
                # TREND mode disabled — skip signals inside ORB range
                return None

            min_votes = settings.trend_mode_min_votes  # default 3 = unanimous
            bull_votes = 0
            bear_votes = 0
            vote_reasons: list[str] = []

            # Vote 1: Supertrend direction
            if curr["st_dir"] == 1:
                bull_votes += 1
                vote_reasons.append("Supertrend BULLISH")
            elif curr["st_dir"] == -1:
                bear_votes += 1
                vote_reasons.append("Supertrend BEARISH")

            # Vote 2: MACD histogram sign
            if curr["macd_hist"] > 0:
                bull_votes += 1
                vote_reasons.append(f"MACD positive ({curr['macd_hist']:.4f})")
            elif curr["macd_hist"] < 0:
                bear_votes += 1
                vote_reasons.append(f"MACD negative ({curr['macd_hist']:.4f})")

            # Vote 3: MACD momentum (rising = bullish, falling = bearish)
            if curr["macd_hist"] > prev["macd_hist"]:
                bull_votes += 1
                vote_reasons.append("MACD momentum rising")
            elif curr["macd_hist"] < prev["macd_hist"]:
                bear_votes += 1
                vote_reasons.append("MACD momentum falling")

            if bull_votes >= min_votes:
                conditions.extend(vote_reasons)
                conditions.append(
                    f"Price in ORB range: {price:.2f} ({orb_low:.2f}–{orb_high:.2f})"
                )
                signal_type = SignalType.BUY
                signal_mode = "TREND"
            elif bear_votes >= min_votes:
                conditions.extend(vote_reasons)
                conditions.append(
                    f"Price in ORB range: {price:.2f} ({orb_low:.2f}–{orb_high:.2f})"
                )
                signal_type = SignalType.SELL
                signal_mode = "TREND"

        if signal_type is None:
            return None

        # ── SMART SL & TARGET using VWAP / CPR / S-R levels ──
        levels = self.level_calc.compute(df, orb_high=orb_high, orb_low=orb_low)
        self._cached_levels = levels

        direction = "BUY" if signal_type == SignalType.BUY else "SELL"
        smart = compute_smart_levels(
            entry=price,
            direction=direction,
            levels=levels,
            min_sl_atr=self.params.get("min_sl_atr_mult", 0.5),
            max_sl_atr=self.params.get("max_sl_atr_mult", 2.0),
            default_sl_atr=self.params.get("default_sl_atr_mult", 1.0),
            target_min_rr=self.params.get("target_min_rr", 1.5),
        )

        sl = smart.stop_loss
        target = smart.target1
        trailing = smart.trailing_sl

        # Confidence scale: BREAKOUT > SUSTAINED > TREND
        base_conf = {"BREAKOUT": 60.0, "SUSTAINED": 45.0, "TREND": 30.0}.get(signal_mode, 30.0)

        # Expiry-day confidence penalty for TREND mode
        is_expiry_day = oi_data.get("is_expiry_day", False) if oi_data else False
        if is_expiry_day and signal_mode == "TREND":
            base_conf = 15.0  # Very low confidence for TREND on expiry day

        # Gap-day info
        if gap_type != "NONE":
            conditions.append(f"Gap day: {gap_type} ({gap_pct:+.2f}%)")
            # Penalize counter-gap signals
            if gap_type in ("GAP_UP", "LARGE_GAP_UP") and signal_type == SignalType.SELL:
                base_conf -= 10.0
                conditions.append("⚠ Selling against gap-up")
            elif gap_type in ("GAP_DOWN", "LARGE_GAP_DOWN") and signal_type == SignalType.BUY:
                base_conf -= 10.0
                conditions.append("⚠ Buying against gap-down")

        confidence = min(95.0, base_conf + len(conditions) * 6.0)

        # ── OI CONFIRMATION ──
        # Add OI data as confirmation conditions and adjust confidence
        oi_boost = 0.0
        if oi_data and (oi_data.get("ce_oi_change_pct", 0) != 0 or oi_data.get("pe_oi_change_pct", 0) != 0):
            ce_chg = oi_data.get("ce_oi_change_pct", 0.0)
            pe_chg = oi_data.get("pe_oi_change_pct", 0.0)
            conditions.append(f"CE OI: {ce_chg:+.1f}%")
            conditions.append(f"PE OI: {pe_chg:+.1f}%")

            if signal_type == SignalType.BUY:  # Bullish
                # Bullish OI: CE OI falling (short covering) or PE OI rising (put writing)
                if ce_chg < -2:
                    oi_boost += 5.0
                    conditions.append("OI: CE short covering (bullish)")
                if pe_chg > 2:
                    oi_boost += 5.0
                    conditions.append("OI: PE writers adding (bullish)")
                # Bearish OI warning: heavy CE writing + PE unwinding
                if ce_chg > 10 and pe_chg < -5:
                    oi_boost -= 8.0
                    conditions.append("⚠ OI: Heavy CE writing + PE unwinding (bearish warning)")
            else:  # Bearish (SELL → PE buy)
                # Bearish OI: PE OI rising (fresh puts) or CE OI falling
                if pe_chg > 2:
                    oi_boost += 5.0
                    conditions.append("OI: Fresh PE buildup (bearish)")
                if ce_chg < -2:
                    oi_boost += 5.0
                    conditions.append("OI: CE unwinding (bearish)")
                # CE OI rising = call writing = also bearish
                if ce_chg > 5:
                    oi_boost += 3.0
                    conditions.append("OI: CE writing (bearish)")
                # Bullish OI warning: PE unwinding + CE covering
                if pe_chg < -5 and ce_chg < -5:
                    oi_boost -= 8.0
                    conditions.append("⚠ OI: PE + CE unwinding (bullish warning)")

            confidence = min(95.0, max(10.0, confidence + oi_boost))

        mode_label = f"[{signal_mode}] " if signal_mode else ""

        # Add SL / target reasoning to conditions
        conditions.append(f"SL at {sl:.2f} ({smart.sl_type})")
        conditions.append(f"T1 at {smart.target1:.2f} ({smart.t1_type})")
        conditions.append(f"T2 at {smart.target2:.2f} ({smart.t2_type})")
        conditions.append(f"T3 at {smart.target3:.2f} ({smart.t3_type})")
        conditions.append(f"Risk: {smart.risk_points:.2f} pts | R:R {smart.risk_reward}")
        if levels.vwap > 0:
            conditions.append(f"VWAP: {levels.vwap:.2f}")
        if levels.pivot > 0:
            conditions.append(f"CPR Pivot: {levels.pivot:.2f}")

        return TradeSignal(
            symbol=symbol,
            signal=signal_type,
            entry_price=price,
            stop_loss=sl,
            target=target,
            trailing_sl=trailing,
            confidence=confidence,
            strategy_name=self.name,
            reasoning=(
                f"{mode_label}ORB {'breakout' if signal_type == SignalType.BUY else 'breakdown'} "
                f"at {price:.2f}. ORB range [{orb_low:.2f}–{orb_high:.2f}]. "
                f"SL at {smart.sl_type} ({sl:.2f}), "
                f"Targets: T1={smart.target1:.2f} T2={smart.target2:.2f} T3={smart.target3:.2f}. "
                f"Supertrend + MACD + OI aligned."
            ),
            conditions_met=conditions,
            timeframe=self.timeframe,
            timestamp=datetime.now().isoformat(),
        )

    def get_cached_levels(self) -> Optional[IntraDayLevels]:
        """Return the last computed levels (for use by the engine)."""
        return self._cached_levels

    def get_smart_levels(self, df: pd.DataFrame, price: float, direction: str) -> SmartLevels:
        """Compute smart SL/targets on demand for a given price and direction."""
        orb_h, orb_l = self.compute_orb(df)
        levels = self.level_calc.compute(df, orb_high=orb_h or 0, orb_low=orb_l or 0)
        return compute_smart_levels(
            entry=price,
            direction=direction,
            levels=levels,
            min_sl_atr=self.params.get("min_sl_atr_mult", 0.5),
            max_sl_atr=self.params.get("max_sl_atr_mult", 2.0),
            default_sl_atr=self.params.get("default_sl_atr_mult", 1.0),
            target_min_rr=self.params.get("target_min_rr", 1.5),
        )
