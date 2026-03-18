"""VWAP Breakout Strategy - BUY above VWAP+1s, SELL below VWAP-1s with Supertrend+MACD+Volume."""
import logging
from datetime import datetime, time
from typing import Optional, Dict, Any
import pandas as pd, numpy as np
from strategies.base_strategy import BaseStrategy, TradeSignal, SignalType
from level_calculator import LevelCalculator, IntraDayLevels
from smart_sl_engine import compute_smart_levels
logger = logging.getLogger(__name__)
class VWAPBreakoutStrategy(BaseStrategy):
    name = "VWAP Breakout"
    description = "Trades VWAP band breakouts on trending days."
    timeframe = "5m"
    min_bars = 30
    def __init__(self, params=None):
        super().__init__(params)
        self.level_calc = LevelCalculator()
        self._cached_levels = None
    def default_params(self):
        return {"vol_spike_mult":1.3,"vol_lookback":20,"macd_fast":12,"macd_slow":26,"macd_signal":9,
                "st_atr_period":10,"st_multiplier":3.0,"min_sl_atr_mult":0.5,"max_sl_atr_mult":2.0,
                "default_sl_atr_mult":1.0,"target_min_rr":1.5}
    @staticmethod
    def _norm(df):
        df=df.copy(); df["date"]=pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None: df["date"]=df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        return df
    def _st_dir(self, df):
        p,m=self.params["st_atr_period"],self.params["st_multiplier"]
        atr=self.calculate_atr(df,p); hl2=(df["high"]+df["low"])/2
        upper,lower=hl2+m*atr,hl2-m*atr
        d=pd.Series(0,index=df.index,dtype=int)
        for i in range(1,len(df)):
            if df["close"].iloc[i]>upper.iloc[i-1]: d.iloc[i]=1
            elif df["close"].iloc[i]<lower.iloc[i-1]: d.iloc[i]=-1
            else:
                d.iloc[i]=d.iloc[i-1]
                if d.iloc[i]==1: lower.iloc[i]=max(lower.iloc[i],lower.iloc[i-1])
                else: upper.iloc[i]=min(upper.iloc[i],upper.iloc[i-1])
        return int(d.iloc[-1])
    def _macd_h(self, df):
        f,s,g=self.params["macd_fast"],self.params["macd_slow"],self.params["macd_signal"]
        ml=df["close"].ewm(span=f,adjust=False).mean()-df["close"].ewm(span=s,adjust=False).mean()
        return float((ml-ml.ewm(span=g,adjust=False).mean()).iloc[-1])
    def _vol_spike(self, df):
        if "volume" not in df.columns: return True
        lb,m=self.params["vol_lookback"],self.params["vol_spike_mult"]
        if len(df)<lb+1: return False
        avg=df["volume"].iloc[-(lb+1):-1].mean()
        return avg>0 and df["volume"].iloc[-1]>=avg*m
    def generate_signal(self, df, symbol, orb_high=0, orb_low=0, **kw):
        if len(df)<self.min_bars: return None
        df=self._norm(df); lt=df["date"].iloc[-1]
        if lt.time()<time(9,45) or lt.time()>time(15,0): return None
        mkt=pd.Timestamp(datetime.combine(lt.date(),time(9,15)))
        dt=df[df["date"]>=mkt].copy()
        if len(dt)<5: return None
        tp=(dt["high"]+dt["low"]+dt["close"])/3; vol=dt["volume"].replace(0,1)
        vwap=(tp*vol).cumsum()/vol.cumsum()
        dev=tp-vwap; std=np.sqrt((dev**2*vol).cumsum()/vol.cumsum())
        v,s=float(vwap.iloc[-1]),float(std.iloc[-1])
        if s<=0: return None
        up1,lo1=v+s,v-s
        price,prev=float(df["close"].iloc[-1]),float(df["close"].iloc[-2])
        conds,sig=[],None
        if price>up1 and prev<=up1:
            conds.append(f"Price {price:.2f} > VWAP+1s ({up1:.2f})")
            if self._st_dir(df)!=1: return None
            conds.append("Supertrend BULLISH")
            mh=self._macd_h(df)
            if mh<=0: return None
            conds.append(f"MACD positive: {mh:.4f}")
            if self._vol_spike(df): conds.append("Volume spike")
            sig=SignalType.BUY
        elif price<lo1 and prev>=lo1:
            conds.append(f"Price {price:.2f} < VWAP-1s ({lo1:.2f})")
            if self._st_dir(df)!=-1: return None
            conds.append("Supertrend BEARISH")
            mh=self._macd_h(df)
            if mh>=0: return None
            conds.append(f"MACD negative: {mh:.4f}")
            if self._vol_spike(df): conds.append("Volume spike")
            sig=SignalType.SELL
        if sig is None: return None
        levels=self.level_calc.compute(df,orb_high=orb_high,orb_low=orb_low)
        self._cached_levels=levels; d="BUY" if sig==SignalType.BUY else "SELL"
        sm=compute_smart_levels(entry=price,direction=d,levels=levels,
            min_sl_atr=self.params["min_sl_atr_mult"],max_sl_atr=self.params["max_sl_atr_mult"],
            default_sl_atr=self.params["default_sl_atr_mult"],target_min_rr=self.params["target_min_rr"])
        conds+=[f"VWAP: {v:.2f}",f"SL at {sm.stop_loss:.2f} ({sm.sl_type})",
                f"T1={sm.target1:.2f}",f"R:R {sm.risk_reward}"]
        conf=min(95.0,70.0+len(conds)*4.0)
        return TradeSignal(symbol=symbol,signal=sig,entry_price=price,stop_loss=sm.stop_loss,
            target=sm.target1,trailing_sl=sm.trailing_sl,confidence=conf,strategy_name=self.name,
            reasoning=f"VWAP breakout {d} at {price:.2f}. VWAP={v:.2f}. SL {sm.sl_type}",
            conditions_met=conds,timeframe=self.timeframe,timestamp=datetime.now().isoformat())
    def get_cached_levels(self): return self._cached_levels
