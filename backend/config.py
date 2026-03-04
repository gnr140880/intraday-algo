from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""
    kite_request_token: str = ""
    news_api_key: str = ""
    app_secret_key: str = "supersecretkey"
    database_url: str = "sqlite+aiosqlite:///./algo_trading.db"
    trading_mode: str = "paper"  # paper or live
    max_risk_per_trade: float = 2.0  # percentage
    max_portfolio_risk: float = 10.0  # percentage
    default_capital: float = 1000000.0
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # --- NIFTY Options Engine ---
    nifty_lot_size: int = 25
    orb_minutes: int = 15
    daily_loss_limit_pct: float = 2.0
    max_risk_per_trade_pct: float = 1.0
    max_concurrent_positions: int = 5
    square_off_time: str = "15:15"          # HH:MM IST
    delta_min: float = 0.30
    delta_max: float = 0.60
    top_score_pct: float = 10.0             # trade only top 10%
    engine_cycle_seconds: int = 300         # 5-minute cycle
    vol_spike_multiplier: float = 1.5

    # --- Telegram Alerts ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
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
