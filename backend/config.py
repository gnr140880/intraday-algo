from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""
    kite_request_token: str = ""
    kite_user_id: str = ""         # Zerodha client ID (e.g. "AB1234")
    kite_password: str = ""        # Zerodha login password
    kite_totp_secret: str = ""     # Base32 TOTP secret from Zerodha 2FA setup
    news_api_key: str = ""
    app_secret_key: str = "supersecretkey"
    database_url: str = "sqlite+aiosqlite:///./algo_trading.db"
    trading_mode: str = "paper"  # paper or live
    max_risk_per_trade: float = 2.0  # percentage
    max_portfolio_risk: float = 10.0  # percentage
    default_capital: float = 1000000.0
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # --- NIFTY Options Engine ---
    nifty_lot_size: int = 75                  # NIFTY lot size (75 as of 2025)
    orb_minutes: int = 15
    daily_loss_limit_pct: float = 2.0
    max_risk_per_trade_pct: float = 1.0
    max_concurrent_positions: int = 5
    square_off_time: str = "15:15"          # HH:MM IST
    delta_min: float = 0.30
    delta_max: float = 0.60
    top_score_pct: float = 10.0             # trade only top 10%
    engine_cycle_seconds: int = 60          # 1-minute cycle
    vol_spike_multiplier: float = 1.5

    # --- Smart SL Engine ---
    min_sl_atr_mult: float = 0.5            # min SL distance = 0.5×ATR
    max_sl_atr_mult: float = 2.0            # max SL distance = 2×ATR
    default_sl_atr_mult: float = 1.0        # default SL if no structural level
    target_min_rr: float = 1.5              # minimum risk:reward for T1

    # --- Auto Trading ---
    auto_trade_enabled: bool = True          # enable auto order placement
    auto_trade_max_orders_per_day: int = 10  # safety limit

    # --- Safety Filters ---
    min_confidence_threshold: float = 50.0   # skip signals below this confidence
    vix_max_threshold: float = 25.0          # skip trades when India VIX > this
    vix_reduce_size_threshold: float = 18.0  # reduce qty by 50% when VIX > this
    reentry_cooldown_minutes: int = 30       # wait N minutes after SL hit before re-entry same direction

    # --- TREND Mode ---
    enable_trend_mode: bool = False          # TREND mode disabled by default (whipsaw risk)
    trend_mode_min_confidence: float = 55.0  # separate higher threshold for TREND signals
    trend_mode_min_votes: int = 3            # require 3/3 votes instead of 2/3

    # --- IV & Expiry ---
    iv_reject_threshold: float = 85.0        # skip options above 85th percentile IV
    expiry_day_early_exit_time: str = "14:30" # earlier square-off on expiry day
    allow_expiry_day_buys: bool = True        # allow buying options on expiry day
    expiry_day_delta_min: float = 0.40        # tighter delta range on expiry day
    expiry_day_delta_max: float = 0.55

    # --- Gap Day ---
    gap_threshold_pct: float = 0.5           # ≥0.5% open vs prev close = gap day
    large_gap_threshold_pct: float = 1.0     # ≥1.0% = large gap
    gap_day_buffer_mult: float = 2.0         # widen ORB buffer on gap days

    # --- Multi-Instrument ---
    instruments_to_trade: str = "NIFTY"      # comma-separated: "NIFTY,BANKNIFTY"
    banknifty_lot_size: int = 30
    banknifty_strike_gap: int = 100
    finnifty_lot_size: int = 40
    finnifty_strike_gap: int = 50
    midcpnifty_lot_size: int = 75
    midcpnifty_strike_gap: int = 25
    sensex_lot_size: int = 20
    sensex_strike_gap: int = 100

    # --- Strategy Toggles ---
    enable_vwap_strategy: bool = True        # VWAP mean-reversion (range-bound days)
    enable_expiry_sell_strategy: bool = True  # Expiry premium selling
    vwap_range_bound_max_orb_pct: float = 0.5  # ORB range < 0.5% of spot = range-bound
    enable_pcr_oi_strategy: bool = True      # PCR/OI directional
    enable_vwap_breakout: bool = True        # VWAP band breakout (trending days)
    enable_ema_crossover: bool = True        # EMA 9/21 crossover
    enable_rsi_divergence: bool = True       # RSI divergence reversal
    enable_gap_fill: bool = True             # Gap fill strategy
    enable_straddle_strangle: bool = False   # Straddle/Strangle sell (needs margin)
    enable_iron_condor: bool = False         # Iron Condor (needs margin + multi-leg)
    enable_hero_zero: bool = True            # Hero Zero expiry lottery
    enable_bollinger_mr: bool = True         # Bollinger Band mean reversion (sideways)
    enable_orb_scalper: bool = True          # ORB range scalper (sideways days)

    # --- Adaptive Risk ---
    enable_adaptive_sizing: bool = False     # Auto-adjust position size
    adaptive_lookback_trades: int = 5        # Look at last N trades for win rate
    adaptive_min_win_rate: float = 40.0      # Reduce size below this win rate
    adaptive_max_win_rate: float = 70.0      # Increase size above this win rate
    adaptive_reduce_factor: float = 0.5      # Multiply qty by this when losing
    adaptive_increase_factor: float = 1.25   # Multiply qty by this when winning
    adaptive_drawdown_halt_pct: float = 3.0  # Halt trading at this daily loss %

    # --- Stock F&O ---
    stock_fno_enabled: bool = False
    stock_fno_watchlist: str = "RELIANCE,TCS,HDFCBANK,ICICIBANK,INFY"

    # --- Multi-Timeframe ---
    enable_multi_tf: bool = True             # use 15m + 1h confirmation
    multi_tf_weight: int = 8                 # scoring weight for TF alignment

    # --- Slippage ---
    slippage_model_enabled: bool = True
    slippage_fixed_pct: float = 0.05         # 0.05% fixed slippage component

    # --- WebSocket Ticks ---
    use_websocket_ticks: bool = False
    tick_batch_interval_ms: int = 1000

    # --- Live OI Feed (WebSocket) ---
    enable_live_oi_feed: bool = True          # enable real-time OI via WebSocket
    oi_feed_num_strikes: int = 10             # subscribe ATM ± N strikes (each side)
    oi_feed_resubscribe_interval: int = 300   # re-center ATM every N seconds

    # --- Trailing Stop-Loss ---
    tsl_breakeven_trigger_pct: float = 3.0   # move SL to breakeven after X% gain
    tsl_early_breakeven_pct: float = 5.0     # early breakeven threshold
    tsl_trail_t1_atr_mult: float = 1.0       # ATR multiplier in TRAIL_T1 phase
    tsl_trail_t2_atr_mult: float = 0.7       # ATR multiplier in TRAIL_T2 phase
    tsl_tight_atr_mult: float = 0.5          # ATR multiplier in TIGHT phase
    tsl_tight_trigger_pct: float = 80.0      # enter TIGHT phase at X%+ gain
    tsl_trail_pct_initial: float = 15.0      # % trail from highs in TRAIL_T1
    tsl_trail_pct_t2: float = 10.0           # % trail from highs in TRAIL_T2
    tsl_trail_pct_tight: float = 6.0         # % trail from highs in TIGHT
    tsl_swing_lookback: int = 3              # candles for swing-low trailing
    tsl_sl_buffer_pct: float = 0.5           # place SL this % below calc level

    # --- Telegram Alerts ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- API Security (protect trading endpoints) ---
    # Set this in `.env` as API_KEY="<long-random-string>"
    api_key: str = ""

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = str(_ENV_FILE)
        extra = "ignore"


settings = Settings()

# --- Stock Universe: Nifty 50 + Mid Cap Select + F&O actives ---
NIFTY50_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
    "INFOSYS", "SBIN", "HINDUNILVR", "ITC", "LT",
    "HCLTECH", "KOTAKBANK", "MARUTI", "SUNPHARMA", "AXISBANK",
    "TITAN", "BAJFINANCE", "WIPRO", "NTPC", "POWERGRID",
    "ULTRACEMCO", "ASIANPAINT", "BAJAJFINSV", "NESTLEIND", "TECHM",
    "ONGC", "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "BPCL", "DRREDDY",
    "DIVISLAB", "EICHERMOT", "GRASIM", "HEROMOTOCO", "HINDALCO",
    "CIPLA", "HDFCLIFE", "INDUSINDBK", "SBILIFE", "APOLLOHOSP",
    "TRENT", "BEL", "SHRIRAMFIN", "BAJAJ-AUTO", "BRITANNIA"
]

MIDCAP_WATCHLIST = [
    "PIDILITIND", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE",
    "POLYCAB", "CUMMINSIND", "VOLTAS", "GODREJPROP", "LUPIN",
    "TORNTPHARM", "AUROPHARMA", "IPCALAB", "MANKIND", "ALKEM",
    "FEDERALBNK", "IDFCFIRSTB", "RBLBANK", "BANDHANBNK", "KARURVYSYA"
]

FON_ACTIVES = [
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "SENSEX", "BANKEX"
]

# Global indices symbols for yfinance
GLOBAL_INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai": "000001.SS",
    "SGX Nifty": "^NSEI",
    "Dollar Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "VIX (US)": "^VIX",
}

# Zerodha exchange tokens for indices
INDIA_INDICES = {
    "NIFTY 50": "NSE:NIFTY 50",
    "BANK NIFTY": "NSE:NIFTY BANK",
    "NIFTY IT": "NSE:NIFTY IT",
    "INDIA VIX": "NSE:INDIA VIX",
    "NIFTY MIDCAP": "NSE:NIFTY MIDCAP 100",
    "SENSEX": "BSE:SENSEX",
}
