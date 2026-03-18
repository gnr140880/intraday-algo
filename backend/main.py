from datetime import datetime, timedelta, date, time
"""
AlgoTest - FastAPI backend for algorithmic trading on Indian markets.
"""
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
import threading

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import json


from config import settings, NIFTY50_STOCKS, MIDCAP_WATCHLIST, FON_ACTIVES, GLOBAL_INDICES, INDIA_INDICES
from kite_client import KiteClient
import io
import csv as csv_mod
import pandas as pd
from pathlib import Path

# --- Well-known index instrument tokens ---
INDEX_TOKENS = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
    "FINNIFTY": 257801,
    "MIDCPNIFTY": 288009,
}

# --- Kite historical interval limits (max calendar days allowed) ---
INTERVAL_MAX_DAYS = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "hour": 400,
    "day": 2000,
}
VALID_INTERVALS = list(INTERVAL_MAX_DAYS.keys())
# from auto_login import refresh_access_token, is_token_valid  # auto-login disabled
from strategies.supertrend_strategy import SupertrendStrategy
from strategies.nifty_options_orb import NiftyOptionsORBStrategy
from strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from strategies.pcr_oi_directional import PCROIDirectionalStrategy
from strategies.vwap_breakout import VWAPBreakoutStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.gap_fill import GapFillStrategy
from strategies.straddle_strangle import StraddleStrangleSellStrategy
from strategies.iron_condor import IronCondorStrategy
from options_engine import NiftyOptionsEngine
from scoring_engine import ScoringEngine
from risk_manager import RiskManager
from news_aggregator import news_aggregator
from telegram_alerts import telegram
from trade_journal import trade_journal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Strategy registry ---
STRATEGIES = {
    "supertrend_ema": SupertrendStrategy,
    "nifty_options_orb": NiftyOptionsORBStrategy,
    "vwap_mean_reversion": VWAPMeanReversionStrategy,
    "pcr_oi_directional": PCROIDirectionalStrategy,
    "vwap_breakout": VWAPBreakoutStrategy,
    "ema_crossover": EMACrossoverStrategy,
    "rsi_divergence": RSIDivergenceStrategy,
    "gap_fill": GapFillStrategy,
    "straddle_strangle": StraddleStrangleSellStrategy,
    "iron_condor": IronCondorStrategy,
}

# --- NIFTY Options Engine singleton ---
nifty_engine = NiftyOptionsEngine()
scheduler = AsyncIOScheduler()

_ENGINE_CYCLE_LOCK = threading.Lock()


def _require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """
    Protect trading-capable endpoints.
    Configure via `.env` as API_KEY="<long-random-string>".
    """
    if not settings.api_key:
        raise HTTPException(
            status_code=500,
            detail="API_KEY is not configured on server. Set API_KEY in backend/.env to enable protected endpoints.",
        )

    presented = x_api_key
    if not presented and authorization:
        # Accept: Authorization: Bearer <token>
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            presented = parts[1]

    if presented != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


async def _run_cycle_locked() -> dict:
    """
    Run one engine cycle with mutual exclusion.
    Executes in a worker thread to avoid blocking the asyncio loop.
    """
    def _do():
        with _ENGINE_CYCLE_LOCK:
            return nifty_engine.run_cycle()

    return await asyncio.to_thread(_do)


async def engine_cycle_job():
    """Scheduler job: runs one engine cycle."""
    try:
        state = await _run_cycle_locked()
        await ws_dashboard_manager.broadcast(state)
    except Exception as e:
        logger.exception("Engine cycle error")


def eod_report_job():
    """Scheduler job: sends comprehensive EOD trade journal report at 3:30 PM IST."""
    try:
        logger.info("EOD report job triggered")
        risk_status = nifty_engine.risk_mgr.get_status()
        messages = trade_journal.generate_eod_telegram(risk_status)
        telegram.alert_eod_journal(messages)
        logger.info(f"EOD report sent via Telegram ({len(messages)} messages)")
    except Exception as e:
        logger.error(f"EOD report job error: {e}")


# Auto-login disabled — token_refresh_job removed
# def token_refresh_job():
#     """Scheduler job: refresh Kite access token daily at 6:05 AM IST."""
#     ...


# --- Pydantic request/response models ---

class OrderRequest(BaseModel):
    tradingsymbol: str
    exchange: str = "NSE"
    transaction_type: str  # BUY or SELL
    quantity: int
    order_type: str = "MARKET"
    product: str = "MIS"
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    variety: str = "regular"  # regular, co, amo, iceberg, auction
    disclosed_quantity: Optional[int] = None


class ModifyOrderRequest(BaseModel):
    order_type: Optional[str] = None
    quantity: Optional[int] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None


class SignalRequest(BaseModel):
    symbol: str
    exchange: str = "NSE"
    strategy: str = "supertrend_ema"
    interval: str = "5minute"
    days: int = 5


# --- App lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AlgoTest backend starting up")
    logger.info(f"Trading mode: {settings.trading_mode}")
    # Start engine scheduler (every 5 min, 9:16–15:20 IST)
    scheduler.add_job(
        engine_cycle_job,
        "interval",
        seconds=settings.engine_cycle_seconds,
        id="nifty_engine_cycle",
        replace_existing=True,
    )
    # EOD report at 3:30 PM IST every weekday
    scheduler.add_job(
        eod_report_job,
        "cron",
        hour=15,
        minute=30,
        day_of_week="mon-fri",
        id="eod_report",
        replace_existing=True,
    )
    # Auto-login disabled — no scheduled token refresh
    scheduler.start()
    logger.info(f"Options engine scheduler started (every {settings.engine_cycle_seconds}s)")
    logger.info("EOD report scheduled at 3:30 PM IST (Mon-Fri)")

    # --- Manual login only ---
    kite = KiteClient.get_instance()
    if kite.access_token:
        logger.info("Access token loaded from .env (use /api/auth/login-url if expired)")
    else:
        logger.warning("No access token in .env — use /api/auth/login-url to login manually")

    yield
    scheduler.shutdown(wait=False)
    logger.info("AlgoTest backend shutting down")
    kite = KiteClient.get_instance()
    kite.stop_ticker()


app = FastAPI(
    title="AlgoTest",
    description="Algorithmic trading backend for Indian markets",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health
# ============================================================

@app.get("/api/health")
async def health():
    kite = KiteClient.get_instance()
    return {
        "status": "ok",
        "trading_mode": settings.trading_mode,
        "kite_connected": kite.is_connected,
    }


# ============================================================
# Auth – Kite Connect login flow
# ============================================================

@app.get("/api/auth/login-url")
async def get_login_url():
    kite = KiteClient.get_instance()
    return {"login_url": kite.get_login_url()}


@app.get("/api/auth/callback")
@app.get("/kite/callback")
async def auth_callback(request_token: str):
    kite = KiteClient.get_instance()
    result = kite.generate_session(request_token)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])
    return result


@app.get("/api/auth/status")
async def auth_status():
    kite = KiteClient.get_instance()
    return {"connected": kite.is_connected}


# Auto-login disabled — refresh-token endpoint removed
# @app.post("/api/auth/refresh-token")
# async def refresh_token():
#     ...


# ============================================================
# Portfolio – Profile, Holdings, Positions, Funds
# ============================================================

def _require_connection():
    kite = KiteClient.get_instance()
    if not kite.is_connected:
        raise HTTPException(status_code=401, detail="Kite not connected. Login first.")
    return kite


@app.get("/api/profile")
async def get_profile():
    kite = _require_connection()
    return kite.get_profile()


@app.get("/api/holdings")
async def get_holdings():
    kite = _require_connection()
    return kite.get_holdings()


@app.get("/api/positions")
async def get_positions():
    kite = _require_connection()
    return kite.get_positions()


@app.get("/api/funds")
async def get_funds():
    kite = _require_connection()
    return kite.get_funds()


# ============================================================
# Orders
# ============================================================

@app.get("/api/orders")
async def get_orders():
    kite = _require_connection()
    return kite.get_orders()


@app.get("/api/trades")
async def get_trades():
    kite = _require_connection()
    return kite.get_trades()


@app.post("/api/orders")
async def place_order(order: OrderRequest, _ok: bool = Depends(_require_api_key)):
    kite = _require_connection()
    result = kite.place_order(
        tradingsymbol=order.tradingsymbol,
        exchange=order.exchange,
        transaction_type=order.transaction_type,
        quantity=order.quantity,
        order_type=order.order_type,
        product=order.product,
        price=order.price,
        trigger_price=order.trigger_price,
        variety=order.variety,
        disclosed_quantity=order.disclosed_quantity,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.put("/api/orders/{order_id}")
async def modify_order(order_id: str, body: ModifyOrderRequest, _ok: bool = Depends(_require_api_key)):
    kite = _require_connection()
    kwargs = body.model_dump(exclude_none=True)
    if not kwargs:
        raise HTTPException(status_code=400, detail="Nothing to modify")
    result = kite.modify_order(order_id, **kwargs)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str, variety: str = "regular", _ok: bool = Depends(_require_api_key)):
    kite = _require_connection()
    result = kite.cancel_order(order_id, variety=variety)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ============================================================
# Market Data
# ============================================================

@app.get("/api/market/quote")
async def get_quote(instruments: str = Query(..., description="Comma-separated, e.g. NSE:RELIANCE,NSE:TCS")):
    kite = _require_connection()
    inst_list = [i.strip() for i in instruments.split(",")]
    return kite.get_quote(inst_list)


@app.get("/api/market/ltp")
async def get_ltp(instruments: str = Query(..., description="Comma-separated, e.g. NSE:RELIANCE")):
    kite = _require_connection()
    inst_list = [i.strip() for i in instruments.split(",")]
    return kite.get_ltp(inst_list)


@app.get("/api/market/ohlc")
async def get_ohlc(instruments: str = Query(..., description="Comma-separated, e.g. NSE:RELIANCE")):
    kite = _require_connection()
    inst_list = [i.strip() for i in instruments.split(",")]
    return kite.get_ohlc(inst_list)


@app.get("/api/market/history/{instrument_token}")
async def get_historical(
    instrument_token: int,
    interval: str = "minute",
    days: int = 5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    oi: bool = False,
):
    """Fetch historical OHLCV data from Kite for any instrument token."""
    kite = _require_connection()
    if interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval. Must be one of: {VALID_INTERVALS}")
    max_days = INTERVAL_MAX_DAYS.get(interval, 60)
    if from_date and to_date:
        fd = datetime.strptime(from_date, "%Y-%m-%d")
        td = datetime.strptime(to_date, "%Y-%m-%d")
        # Clamp to API limit
        if (td - fd).days > max_days:
            fd = td - timedelta(days=max_days)
    else:
        td = datetime.now()
        clamped_days = min(days, max_days)
        fd = td - timedelta(days=clamped_days)
    data = kite.get_historical_data(instrument_token, fd, td, interval, oi=oi)
    # Serialize datetime objects for JSON
    for row in data:
        if "date" in row and hasattr(row["date"], "isoformat"):
            row["date"] = row["date"].isoformat()
    return {
        "count": len(data),
        "instrument_token": instrument_token,
        "interval": interval,
        "from_date": fd.strftime("%Y-%m-%d"),
        "to_date": td.strftime("%Y-%m-%d"),
        "data": data,
    }


@app.get("/api/market/history/index/{symbol}")
async def get_index_historical(
    symbol: str,
    interval: str = "5minute",
    days: int = 5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """Fetch historical OHLCV for NIFTY/BANKNIFTY by name (no token needed)."""
    kite = _require_connection()
    sym = symbol.upper()
    token = INDEX_TOKENS.get(sym)
    if token is None:
        raise HTTPException(404, f"Unknown index: {symbol}. Available: {list(INDEX_TOKENS.keys())}")
    if interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval. Must be one of: {VALID_INTERVALS}")
    max_days = INTERVAL_MAX_DAYS.get(interval, 60)
    if from_date and to_date:
        fd = datetime.strptime(from_date, "%Y-%m-%d")
        td = datetime.strptime(to_date, "%Y-%m-%d")
        if (td - fd).days > max_days:
            fd = td - timedelta(days=max_days)
    else:
        td = datetime.now()
        clamped_days = min(days, max_days)
        fd = td - timedelta(days=clamped_days)
    data = kite.get_historical_data(token, fd, td, interval)
    for row in data:
        if "date" in row and hasattr(row["date"], "isoformat"):
            row["date"] = row["date"].isoformat()
    return {
        "count": len(data),
        "symbol": sym,
        "instrument_token": token,
        "interval": interval,
        "from_date": fd.strftime("%Y-%m-%d"),
        "to_date": td.strftime("%Y-%m-%d"),
        "data": data,
    }


@app.get("/api/market/history/{instrument_token}/csv")
async def get_historical_csv(
    instrument_token: int,
    interval: str = "5minute",
    days: int = 5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    oi: bool = False,
):
    """Download historical OHLCV data as CSV file."""
    kite = _require_connection()
    if interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval. Must be one of: {VALID_INTERVALS}")
    max_days = INTERVAL_MAX_DAYS.get(interval, 60)
    if from_date and to_date:
        fd = datetime.strptime(from_date, "%Y-%m-%d")
        td = datetime.strptime(to_date, "%Y-%m-%d")
        if (td - fd).days > max_days:
            fd = td - timedelta(days=max_days)
    else:
        td = datetime.now()
        clamped_days = min(days, max_days)
        fd = td - timedelta(days=clamped_days)
    data = kite.get_historical_data(instrument_token, fd, td, interval, oi=oi)
    if not data:
        raise HTTPException(404, "No historical data available for the given parameters")

    # Build CSV in memory
    output = io.StringIO()
    fields = ["date", "open", "high", "low", "close", "volume"]
    if oi:
        fields.append("oi")
    writer = csv_mod.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        csv_row = dict(row)
        if "date" in csv_row and hasattr(csv_row["date"], "isoformat"):
            csv_row["date"] = csv_row["date"].isoformat()
        writer.writerow(csv_row)

    output.seek(0)
    filename = f"history_{instrument_token}_{interval}_{fd.strftime('%Y%m%d')}_{td.strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/market/instruments/search")
async def search_instruments(
    query: str = Query(..., min_length=2, description="Search by trading symbol"),
    exchange: str = "NFO",
    limit: int = 20,
):
    """Search instruments by symbol name. Returns matching instruments with tokens."""
    kite = _require_connection()
    try:
        instruments = kite.get_instruments(exchange)
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch instruments: {e}")

    q = query.upper()
    matches = [
        {
            "instrument_token": i["instrument_token"],
            "tradingsymbol": i["tradingsymbol"],
            "name": i.get("name", ""),
            "exchange": i.get("exchange", exchange),
            "expiry": str(i.get("expiry", "")),
            "strike": i.get("strike", 0),
            "instrument_type": i.get("instrument_type", ""),
            "lot_size": i.get("lot_size", 0),
        }
        for i in instruments
        if q in i.get("tradingsymbol", "").upper()
    ][:limit]

    return {"count": len(matches), "instruments": matches}


@app.get("/api/market/instruments")
async def get_instruments(exchange: Optional[str] = None):
    """List all instruments, optionally filtered by exchange."""
    kite = _require_connection()
    return kite.get_instruments(exchange)


# ============================================================
# Watchlists / Stock Universe
# ============================================================

@app.get("/api/watchlist")
async def get_watchlists():
    return {
        "nifty50": NIFTY50_STOCKS,
        "midcap": MIDCAP_WATCHLIST,
        "fno": FON_ACTIVES,
        "global_indices": GLOBAL_INDICES,
        "india_indices": INDIA_INDICES,
    }


# ============================================================
# News – Global & India Stock Market
# ============================================================

@app.get("/api/news")
async def get_all_news(refresh: bool = False):
    """All news with categories and sentiment. Cached 10 min."""
    return news_aggregator.fetch_all(force_refresh=refresh)


@app.get("/api/news/india")
async def get_india_news(limit: int = 30):
    """India market, NIFTY, sector news."""
    return {"count": limit, "articles": news_aggregator.get_india_news(limit)}


@app.get("/api/news/global")
async def get_global_news(limit: int = 30):
    """Global markets, commodities, forex news."""
    return {"count": limit, "articles": news_aggregator.get_global_news(limit)}


@app.get("/api/news/sentiment")
async def get_news_sentiment():
    """Market mood / sentiment summary."""
    return news_aggregator.get_sentiment_summary()


# ============================================================
# Strategies
# ============================================================

@app.get("/api/strategies")
async def list_strategies():
    result = []
    for key, cls in STRATEGIES.items():
        s = cls()
        result.append({
            "id": key,
            "name": s.name,
            "description": s.description,
            "timeframe": s.timeframe,
            "params": s.params,
        })
    return result


@app.post("/api/strategies/signal")
async def generate_signal(req: SignalRequest):
    kite = _require_connection()

    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

    # Resolve instrument token
    instruments = kite.get_instruments("NSE" if req.exchange == "NSE" else req.exchange)
    token = None
    for inst in instruments:
        if inst["tradingsymbol"] == req.symbol:
            token = inst["instrument_token"]
            break
    if token is None:
        raise HTTPException(status_code=404, detail=f"Instrument not found: {req.symbol}")

    # Fetch historical data
    to_date = datetime.now()
    from_date = to_date - timedelta(days=req.days)
    import pandas as pd
    raw = kite.get_historical_data(token, from_date, to_date, req.interval)
    if not raw:
        raise HTTPException(status_code=404, detail="No historical data available")

    df = pd.DataFrame(raw)
    strategy = STRATEGIES[req.strategy]()
    signal = strategy.generate_signal(df, req.symbol)

    if signal is None:
        return {"signal": None, "message": "No signal generated for current data"}

    return {
        "signal": {
            "symbol": signal.symbol,
            "type": signal.signal.value,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "target": signal.target,
            "trailing_sl": signal.trailing_sl,
            "confidence": signal.confidence,
            "risk_reward": signal.risk_reward,
            "strategy": signal.strategy_name,
            "reasoning": signal.reasoning,
            "conditions_met": signal.conditions_met,
            "timeframe": signal.timeframe,
            "timestamp": signal.timestamp,
        }
    }


# ============================================================
# WebSocket – Real-time tick streaming
# ============================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for conn in self.active_connections:
            try:
                await conn.send_json(data)
            except Exception:
                pass


ws_manager = ConnectionManager()


@app.websocket("/ws/ticks")
async def websocket_ticks(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Client can send subscription updates as JSON
            # e.g. {"action": "subscribe", "tokens": [123, 456]}
            import json
            try:
                msg = json.loads(data)
                await websocket.send_json({"ack": True, "message": msg})
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============================================================
# Config / Settings (read-only)
# ============================================================

@app.get("/api/config")
async def get_config():
    return {
        "trading_mode": settings.trading_mode,
        "max_risk_per_trade": settings.max_risk_per_trade,
        "max_portfolio_risk": settings.max_portfolio_risk,
        "default_capital": settings.default_capital,
        "daily_loss_limit_pct": settings.daily_loss_limit_pct,
        "delta_range": [settings.delta_min, settings.delta_max],
        "top_score_pct": settings.top_score_pct,
        "square_off_time": settings.square_off_time,
    }


# ============================================================
# NIFTY Options Engine – Dashboard API
# ============================================================

@app.get("/api/engine/dashboard")
async def engine_dashboard():
    """Live dashboard state: scores, ORB, signals, trades, P&L."""
    return nifty_engine.get_dashboard()


@app.get("/api/engine/status")
async def engine_status():
    return {
        "status": nifty_engine.dashboard_state.get("engine_status"),
        "orb": nifty_engine.dashboard_state.get("orb"),
        "nifty_spot": nifty_engine.nifty_spot,
        "halted": nifty_engine.risk_mgr.is_halted,
        "open_positions": len(nifty_engine.risk_mgr.open_positions),
        "auto_trade_mode": nifty_engine.auto_trade_mode,
        "active_auto_positions": len([p for p in nifty_engine.order_mgr.positions.values() if p.status != "CLOSED"]),
        "levels_computed": nifty_engine._cached_levels is not None,
    }


@app.get("/api/engine/scores")
async def engine_scores():
    """Top scored option candidates."""
    return nifty_engine.scorer.score_summary(nifty_engine.scored_candidates)


@app.get("/api/engine/risk")
async def engine_risk():
    """Risk manager state: P&L, limits, positions."""
    return nifty_engine.risk_mgr.get_status()


@app.get("/api/engine/trades")
async def engine_trades():
    """All trades taken today."""
    return nifty_engine.risk_mgr.get_trades_summary()


@app.get("/api/dashboard/equity-curve")
async def dashboard_equity_curve():
    """Equity curve data for charting."""
    return nifty_engine.risk_mgr.get_equity_curve()


@app.get("/api/dashboard/trade-stats")
async def dashboard_trade_stats(lookback: int = 20):
    """Recent trade statistics (win rate, avg PnL, etc.)."""
    return nifty_engine.risk_mgr.get_recent_trade_stats(lookback)


@app.get("/api/dashboard/strategies")
async def dashboard_strategies():
    """List all available strategies and their enabled status."""
    return {
        "strategies": [
            {"name": "ORB Breakout", "key": "nifty_options_orb", "enabled": True},
            {"name": "Supertrend+EMA", "key": "supertrend_ema", "enabled": True},
            {"name": "VWAP Mean Reversion", "key": "vwap_mean_reversion", "enabled": settings.enable_vwap_strategy},
            {"name": "VWAP Breakout", "key": "vwap_breakout", "enabled": settings.enable_vwap_breakout},
            {"name": "EMA Crossover", "key": "ema_crossover", "enabled": settings.enable_ema_crossover},
            {"name": "RSI Divergence", "key": "rsi_divergence", "enabled": settings.enable_rsi_divergence},
            {"name": "PCR/OI Directional", "key": "pcr_oi_directional", "enabled": settings.enable_pcr_oi_strategy},
            {"name": "Gap Fill", "key": "gap_fill", "enabled": settings.enable_gap_fill},
            {"name": "Expiry Premium Sell", "key": "expiry_premium_sell", "enabled": settings.enable_expiry_sell_strategy},
            {"name": "Straddle/Strangle", "key": "straddle_strangle", "enabled": settings.enable_straddle_strangle},
            {"name": "Iron Condor", "key": "iron_condor", "enabled": settings.enable_iron_condor},
            {"name": "Hero Zero", "key": "hero_zero", "enabled": settings.enable_hero_zero},
        ],
        "instruments": settings.instruments_to_trade.split(","),
    }


@app.get("/api/dashboard/daily-pnl")
async def dashboard_daily_pnl():
    """Daily P&L summary."""
    return {
        "daily_pnl": nifty_engine.risk_mgr.daily_pnl,
        "realised_pnl": nifty_engine.risk_mgr.realised_pnl,
        "unrealised_pnl": nifty_engine.risk_mgr.unrealised_pnl,
        "capital": nifty_engine.risk_mgr.capital,
        "trades_today": len(nifty_engine.risk_mgr.open_positions) + len([
            t for t in nifty_engine.risk_mgr.trades
            if t.exit_time and t.exit_time[:10] == date.today().isoformat()
        ]),
    }


@app.post("/api/engine/run-cycle")
async def engine_run_cycle(_ok: bool = Depends(_require_api_key)):
    """Manually trigger one engine cycle (for testing)."""
    try:
        state = await _run_cycle_locked()
        return state
    except Exception as e:
        logger.exception("run_cycle failed")
        return {"error": str(e), "engine_status": "ERROR"}


@app.post("/api/engine/square-off")
async def engine_square_off(_ok: bool = Depends(_require_api_key)):
    """Manually trigger square-off of all positions."""
    kite = _require_connection()
    nifty_engine.square_off_all(kite, "MANUAL_SQUAREOFF")
    return {"success": True, "message": "All positions squared off"}


@app.get("/api/engine/offline-data")
async def engine_offline_data():
    """
    Fetch NIFTY spot + top scored candidates even when market is closed.
    Uses last available LTP and previous session's option chain data.
    """
    state = nifty_engine.fetch_offline_data()
    return state


# ============================================================
# Auto-Trading & Smart SL/Target API
# ============================================================

@app.get("/api/engine/levels")
async def engine_levels():
    """Current VWAP, CPR, S/R levels used for SL/target calculation."""
    levels = nifty_engine._cached_levels
    if levels is None:
        return {"computed": False, "message": "Levels not yet computed. Run a cycle first."}
    return {
        "computed": True,
        "vwap": levels.vwap,
        "vwap_upper_1": levels.vwap_upper_1,
        "vwap_lower_1": levels.vwap_lower_1,
        "vwap_upper_2": levels.vwap_upper_2,
        "vwap_lower_2": levels.vwap_lower_2,
        "pivot": levels.pivot,
        "bc": levels.bc,
        "tc": levels.tc,
        "r1": levels.r1, "r2": levels.r2, "r3": levels.r3,
        "s1": levels.s1, "s2": levels.s2, "s3": levels.s3,
        "pdh": levels.pdh, "pdl": levels.pdl, "pdc": levels.pdc,
        "orb_high": levels.orb_high, "orb_low": levels.orb_low,
        "nearest_support": levels.nearest_support,
        "nearest_resistance": levels.nearest_resistance,
        "support_levels": levels.support_levels,
        "resistance_levels": levels.resistance_levels,
        "atr": levels.atr,
        "timestamp": levels.timestamp,
    }


@app.get("/api/engine/active-positions")
async def engine_active_positions():
    """Active auto-managed positions with SL/target details."""
    return {
        "positions": nifty_engine.order_mgr.get_active_positions(),
        "count": len([p for p in nifty_engine.order_mgr.positions.values() if p.status != "CLOSED"]),
    }


@app.get("/api/engine/order-log")
async def engine_order_log():
    """Full audit trail of all auto-placed orders."""
    return {
        "orders": nifty_engine.order_mgr.get_order_log(),
        "count": len(nifty_engine.order_mgr.order_log),
    }


@app.get("/api/engine/auto-trade-mode")
async def get_auto_trade_mode():
    """Get current auto-trading mode."""
    return {
        "mode": nifty_engine.auto_trade_mode,
        "enabled": settings.auto_trade_enabled,
    }


class AutoTradeModeRequest(BaseModel):
    mode: str  # 'off', 'paper', 'live'


@app.post("/api/engine/auto-trade-mode")
async def set_auto_trade_mode(req: AutoTradeModeRequest, _ok: bool = Depends(_require_api_key)):
    """Switch between off/paper/live auto-trading."""
    if req.mode not in ("off", "paper", "live"):
        raise HTTPException(status_code=400, detail="Mode must be 'off', 'paper', or 'live'")
    nifty_engine.auto_trade_mode = req.mode
    return {"success": True, "mode": req.mode}


@app.post("/api/engine/force-exit/{trade_id}")
async def force_exit_position(trade_id: str, reason: str = "MANUAL", _ok: bool = Depends(_require_api_key)):
    """Force exit a specific position."""
    kite = _require_connection()
    success = nifty_engine.order_mgr.full_exit(kite, trade_id, reason=reason)
    if not success:
        raise HTTPException(status_code=404, detail=f"Position {trade_id} not found or already closed")
    return {"success": True, "trade_id": trade_id, "reason": reason}


# ============================================================
# Telegram Alerts
# ============================================================

@app.get("/api/telegram/status")
async def telegram_status():
    return {
        "enabled": telegram.enabled,
        "configured": bool(settings.telegram_bot_token and settings.telegram_chat_id),
    }


@app.post("/api/telegram/test")
async def telegram_test(_ok: bool = Depends(_require_api_key)):
    """Send a test message to verify Telegram bot setup."""
    result = telegram.test()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Send failed"))
    return result


class TelegramMessageRequest(BaseModel):
    message: str


@app.post("/api/telegram/send")
async def telegram_send(body: TelegramMessageRequest, _ok: bool = Depends(_require_api_key)):
    """Send a custom Telegram message."""
    if not telegram.enabled:
        raise HTTPException(status_code=400, detail="Telegram not configured")
    telegram.send_custom(body.message)
    return {"success": True}


@app.post("/api/telegram/summary")
async def telegram_daily_summary(_ok: bool = Depends(_require_api_key)):
    """Send today's P&L summary to Telegram."""
    telegram.alert_daily_summary(
        nifty_engine.risk_mgr.get_status(),
        nifty_engine.risk_mgr.get_trades_summary(),
    )
    return {"success": True}


# ============================================================
# EOD Trade Journal / Report
# ============================================================

@app.get("/api/engine/eod-report")
async def engine_eod_report():
    """Get comprehensive EOD trade journal report data."""
    risk_status = nifty_engine.risk_mgr.get_status()
    return trade_journal.generate_eod_report(risk_status)


@app.get("/api/engine/journal")
async def engine_journal():
    """Get raw trade journal data (signals + candidates)."""
    return trade_journal.get_journal_data()


@app.post("/api/engine/send-eod-report")
async def send_eod_report(_ok: bool = Depends(_require_api_key)):
    """Manually trigger EOD report to Telegram."""
    risk_status = nifty_engine.risk_mgr.get_status()
    messages = trade_journal.generate_eod_telegram(risk_status)
    telegram.alert_eod_journal(messages)
    return {"success": True, "messages_sent": len(messages)}


# ============================================================
# Phase 2-4: New Feature APIs
# ============================================================

@app.get("/api/engine/safety-filters")
async def engine_safety_filters():
    """Current safety filter status (VIX, confidence, cooldown, gap, IV)."""
    return {
        "india_vix": nifty_engine._india_vix,
        "vix_max_threshold": settings.vix_max_threshold,
        "vix_reduce_size_threshold": settings.vix_reduce_size_threshold,
        "min_confidence_threshold": settings.min_confidence_threshold,
        "cooldown_minutes": settings.reentry_cooldown_minutes,
        "daily_orders_used": nifty_engine.order_mgr._daily_order_count,
        "daily_orders_limit": settings.auto_trade_max_orders_per_day,
        "gap_type": nifty_engine._gap_type,
        "gap_pct": nifty_engine._gap_pct,
        "iv_reject_threshold": settings.iv_reject_threshold,
        "expiry_day_early_exit": settings.expiry_day_early_exit_time,
        "trend_mode_enabled": settings.enable_trend_mode,
        "multi_tf_enabled": settings.enable_multi_tf,
        "st_dir_15m": nifty_engine._st_dir_15m,
        "st_dir_1h": nifty_engine._st_dir_1h,
        "ws_connected": nifty_engine._ws_connected,
    }


@app.get("/api/engine/strategies")
async def engine_strategies():
    """Available strategies and their enabled/disabled status."""
    return {
        "orb_breakout": {"enabled": True, "name": "NIFTY Options ORB"},
        "trend_mode": {
            "enabled": settings.enable_trend_mode,
            "name": "TREND Mode",
            "min_votes": settings.trend_mode_min_votes,
        },
        "vwap_mean_reversion": {
            "enabled": settings.enable_vwap_strategy,
            "name": "VWAP Mean Reversion",
        },
        "expiry_premium_sell": {
            "enabled": settings.enable_expiry_sell_strategy,
            "name": "Expiry Premium Sell",
        },
        "instruments": settings.instruments_to_trade.split(","),
    }


@app.get("/api/engine/multi-tf")
async def engine_multi_tf():
    """Multi-timeframe indicator status."""
    return {
        "enabled": settings.enable_multi_tf,
        "supertrend_5m": int(nifty_engine._cached_df.iloc[-1].get("st_dir", 0))
            if nifty_engine._cached_df is not None and "st_dir" in nifty_engine._cached_df.columns
            else 0,
        "supertrend_15m": nifty_engine._st_dir_15m,
        "supertrend_1h": nifty_engine._st_dir_1h,
        "aligned": (
            nifty_engine._st_dir_15m != 0
            and nifty_engine._st_dir_1h != 0
            and nifty_engine._st_dir_15m == nifty_engine._st_dir_1h
        ),
    }



# ------------------------------------------------------------------
# Live OI Feed endpoints
# ------------------------------------------------------------------

@app.get("/api/engine/oi-chain")
async def engine_oi_chain():
    """
    Get live option chain with OI data from WebSocket feed.
    Returns full chain table, PCR, buildup signals, max pain, etc.
    """
    feed = nifty_engine._oi_feed
    if feed and feed.is_running:
        return feed.get_dashboard_data()
    # Fallback to cached OI data from REST API
    return {
        "source": "rest_api",
        "connected": False,
        "message": "Live OI feed not running. Enable with ENABLE_LIVE_OI_FEED=true",
        "ce_oi_change_pct": nifty_engine._cached_oi_data.get("ce_oi_change_pct", 0) if nifty_engine._cached_oi_data else 0,
        "pe_oi_change_pct": nifty_engine._cached_oi_data.get("pe_oi_change_pct", 0) if nifty_engine._cached_oi_data else 0,
    }


@app.get("/api/engine/oi-analysis")
async def engine_oi_analysis():
    """
    Get OI analysis summary (PCR, CE/PE OI change, buildup bias).
    Uses live WebSocket data when available, falls back to REST-fetched data.
    """
    feed = nifty_engine._oi_feed
    if feed and feed.is_running:
        analysis = feed.get_oi_analysis()
        analysis["source"] = "live_websocket"
        return analysis
    return {
        "source": "rest_api",
        "connected": False,
        **({k: v for k, v in nifty_engine._cached_oi_data.items()} if nifty_engine._cached_oi_data else {}),
    }


@app.post("/api/engine/oi-feed/start")
async def start_oi_feed(_ok: bool = Depends(_require_api_key)):
    """Manually start the live OI feed."""
    kite = KiteClient.get_instance()
    if not kite.is_connected:
        raise HTTPException(400, "Kite not connected")
    if nifty_engine.nifty_spot <= 0:
        raise HTTPException(400, "NIFTY spot price not available — run a cycle first")
    try:
        nifty_engine._start_oi_feed(kite)
        return {"success": True, "message": "Live OI feed started"}
    except Exception as e:
        raise HTTPException(500, f"Failed to start OI feed: {e}")


@app.post("/api/engine/oi-feed/stop")
async def stop_oi_feed(_ok: bool = Depends(_require_api_key)):
    """Stop the live OI feed."""
    feed = nifty_engine._oi_feed
    if feed:
        feed.stop()
        nifty_engine._oi_feed_started = False
        return {"success": True, "message": "Live OI feed stopped"}
    return {"success": False, "message": "OI feed not running"}


# --- Backtest request model ---
class BacktestRequest(BaseModel):
    strategy: str = "ORB"
    data_file: str = ""
    days: int = 30
    capital: float = 1000000

# Map frontend-friendly names → engine strategy keys
_STRATEGY_ALIAS = {
    "momentum": "SUPERTREND",
    "orb": "ORB",
    "orb_breakout": "ORB",
    "vwap_mr": "VWAP_MR",
    "vwap_mean_reversion": "VWAP_MR",
    "vwap_breakout": "VWAP_BREAKOUT",
    "vwap_bo": "VWAP_BREAKOUT",
    "supertrend": "SUPERTREND",
    "supertrend_ema": "SUPERTREND",
    "ema_crossover": "EMA_CROSSOVER",
    "ema": "EMA_CROSSOVER",
    "rsi_divergence": "RSI_DIVERGENCE",
    "rsi": "RSI_DIVERGENCE",
    "gap_fill": "GAP_FILL",
    "all": "ALL",
}


def _resolve_strategy(name: str) -> str:
    """Resolve a frontend strategy name to engine key."""
    key = name.strip().lower().replace(" ", "_")
    return _STRATEGY_ALIAS.get(key, name.upper())


def _load_csv_and_resample(csv_path: Path) -> pd.DataFrame:
    """Load a 1-min (or any) CSV and resample to 5-min OHLCV."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # Detect interval: if avg gap ≤ 1.5 min → 1-min data, needs resampling
    if len(df) >= 2:
        avg_gap = (df["date"].diff().dropna().dt.total_seconds().median()) / 60
        if avg_gap <= 1.5:
            # Resample 1-min → 5-min
            df = df.set_index("date")
            df = df.resample("5min", label="left", closed="left").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna(subset=["open"]).reset_index()

    return df


@app.post("/api/backtest/download")
async def backtest_download(
    ticker: str = "NIFTY 50",
    from_date: str = "",
    to_date: str = "",
):
    """Download 1-min historical data from Kite and save to data/ for backtesting."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    kite = _require_connection()

    # Resolve symbol → instrument token
    sym = ticker.upper().replace(" ", "")
    token = INDEX_TOKENS.get(sym) or INDEX_TOKENS.get(ticker.upper().split()[0])
    if token is None:
        # Try NIFTY variants
        for k, v in INDEX_TOKENS.items():
            if k in sym:
                token = v
                break
    if token is None:
        token = 256265  # default NIFTY

    if from_date and to_date:
        fd = datetime.strptime(from_date, "%Y-%m-%d")
        td = datetime.strptime(to_date, "%Y-%m-%d")
    else:
        td = datetime.now()
        fd = td - timedelta(days=5)

    # Clamp to Kite 1-min limit (60 days)
    max_days = INTERVAL_MAX_DAYS.get("minute", 60)
    if (td - fd).days > max_days:
        fd = td - timedelta(days=max_days)

    data = kite.get_historical_data(token, fd, td, "minute")
    if not data:
        raise HTTPException(400, detail="No data returned from Kite for the given range")

    # Build CSV
    safe_ticker = ticker.lower().replace(" ", "_")
    filename = f"{safe_ticker}_1m.csv"
    filepath = data_dir / filename

    output = io.StringIO()
    fields = ["date", "open", "high", "low", "close", "volume"]
    writer = csv_mod.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        csv_row = dict(row)
        if "date" in csv_row and hasattr(csv_row["date"], "isoformat"):
            csv_row["date"] = csv_row["date"].isoformat()
        writer.writerow(csv_row)
    filepath.write_text(output.getvalue(), encoding="utf-8")

    return {
        "success": True,
        "rows": len(data),
        "file": f"data/{filename}",
        "message": f"Downloaded {len(data)} rows to data/{filename}",
    }


@app.post("/api/backtest/run")
async def run_backtest(req: BacktestRequest):
    """Run a historical backtest on downloaded CSV data (or live Kite/yfinance)."""
    from backtester import BacktestEngine, DataLoader

    strategy_key = _resolve_strategy(req.strategy)
    data_dir = Path(__file__).parent

    # 1. Try loading from the specified data_file
    df = None
    if req.data_file:
        csv_path = data_dir / req.data_file
        if not csv_path.exists():
            # Also check just filename in data/
            csv_path = data_dir / "data" / Path(req.data_file).name
        if csv_path.exists():
            try:
                df = _load_csv_and_resample(csv_path)
                logger.info(f"Backtest: loaded {len(df)} bars from {csv_path}")
            except Exception as e:
                logger.error(f"Backtest CSV load failed: {e}")
                raise HTTPException(400, detail=f"Failed to load data file: {e}")

    # 2. Fallback: Try any CSV in data/ directory
    if df is None or df.empty:
        data_folder = data_dir / "data"
        if data_folder.exists():
            csvs = sorted(data_folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            for csv_path in csvs:
                try:
                    df = _load_csv_and_resample(csv_path)
                    if df is not None and not df.empty:
                        logger.info(f"Backtest: loaded {len(df)} bars from {csv_path}")
                        break
                except Exception:
                    continue

    # 3. Fallback: Try Kite/yfinance
    if df is None or df.empty:
        loader = DataLoader()
        df = loader.load_nifty_history(days=req.days, interval="5minute", source="kite")

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data available. Download data first using the Download button.")

    # Split into daily sessions
    loader = DataLoader()
    sessions = loader.split_by_session(df)
    if not sessions:
        raise HTTPException(status_code=400, detail="No valid trading sessions found in data")

    try:
        engine = BacktestEngine(strategy=strategy_key, capital=req.capital)
        results = engine.run(sessions)
        return results.to_dict()
    except Exception as e:
        logger.error(f"Backtest engine error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.get("/api/database/trades")
async def db_trades(date_str: Optional[str] = None):
    """Get trades from database for a date (default: today)."""
    from database import db
    from datetime import date as date_type
    db.init()
    if date_str:
        trade_date = date_type.fromisoformat(date_str)
    else:
        trade_date = date_type.today()
    return db.get_trades_for_date(trade_date)


@app.get("/api/database/daily-stats")
async def db_daily_stats(from_date: str = "", to_date: str = ""):
    """Get daily stats for equity curve display."""
    from database import db
    from datetime import date as date_type
    db.init()
    fd = date_type.fromisoformat(from_date) if from_date else date_type.today() - timedelta(days=30)
    td = date_type.fromisoformat(to_date) if to_date else date_type.today()
    return db.get_daily_stats_range(fd, td)


# ============================================================
# WebSocket – Dashboard live updates
# ============================================================

class DashboardWSManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, data: dict):
        for ws in self.connections[:]:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(ws)


ws_dashboard_manager = DashboardWSManager()


@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    """WebSocket endpoint for live dashboard updates."""
    await ws_dashboard_manager.connect(websocket)
    # Send initial state
    try:
      await websocket.send_json(nifty_engine.get_dashboard())
    except Exception:
      pass
    try:
      while True:
        try:
          msg = await websocket.receive_text()
        except RuntimeError as e:
          # WebSocket not connected or closed
          ws_dashboard_manager.disconnect(websocket)
          break
        try:
          parsed = json.loads(msg)
          if parsed.get("action") == "run_cycle":
            # Require API key for manual cycle triggers via WS
            api_key = parsed.get("api_key")
            if not settings.api_key or api_key != settings.api_key:
              await websocket.send_json({"error": "Unauthorized"})
              continue
            state = await _run_cycle_locked()
            await websocket.send_json(state)
          elif parsed.get("action") == "get_state":
            await websocket.send_json(nifty_engine.get_dashboard())
        except json.JSONDecodeError:
          await websocket.send_json({"error": "Invalid JSON"})
        except Exception:
          pass
    except Exception:
      ws_dashboard_manager.disconnect(websocket)


# ============================================================
# Dashboard HTML page (loaded from external file)
# ============================================================

DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Self-contained live dashboard."""
    if not DASHBOARD_HTML_PATH.exists():
        return HTMLResponse("<h2>dashboard.html not found</h2>", status_code=404)
    return HTMLResponse(DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))


# Serve API Test HTML page at /api_test.html and /dashboard/api_test.html
API_TEST_HTML_PATH = Path(__file__).parent / "api_test.html"

@app.get("/api_test.html", response_class=HTMLResponse)
@app.get("/dashboard/api_test.html", response_class=HTMLResponse)
async def api_test_page():
  """Serve the API Test Dashboard HTML page."""
  if not API_TEST_HTML_PATH.exists():
    return HTMLResponse("<h2>api_test.html not found</h2>", status_code=404)
  return HTMLResponse(API_TEST_HTML_PATH.read_text(encoding="utf-8"))


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
