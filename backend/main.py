from datetime import datetime, timedelta, date, time
"""
AlgoTest - FastAPI backend for algorithmic trading on Indian markets.
"""
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import json


from config import settings, NIFTY50_STOCKS, MIDCAP_WATCHLIST, FON_ACTIVES, GLOBAL_INDICES, INDIA_INDICES
from kite_client import KiteClient
# from auto_login import refresh_access_token, is_token_valid  # auto-login disabled
from strategies.supertrend_strategy import SupertrendStrategy
from strategies.nifty_options_orb import NiftyOptionsORBStrategy
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
}

# --- NIFTY Options Engine singleton ---
nifty_engine = NiftyOptionsEngine()
scheduler = AsyncIOScheduler()


def engine_cycle_job():
    """Scheduler job: runs engine cycle synchronously."""
    try:
        state = nifty_engine.run_cycle()
        # Broadcast to WebSocket clients via the running event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(ws_dashboard_manager.broadcast(state))
            else:
                loop.run_until_complete(ws_dashboard_manager.broadcast(state))
        except RuntimeError:
            # If no event loop available, skip broadcast (next HTTP poll will get it)
            logger.debug("No event loop for WS broadcast, skipping")
    except Exception as e:
        logger.error(f"Engine cycle error: {e}")


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
async def place_order(order: OrderRequest):
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
async def modify_order(order_id: str, body: ModifyOrderRequest):
    kite = _require_connection()
    kwargs = body.model_dump(exclude_none=True)
    if not kwargs:
        raise HTTPException(status_code=400, detail="Nothing to modify")
    result = kite.modify_order(order_id, **kwargs)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str, variety: str = "regular"):
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
):
    kite = _require_connection()
    if from_date and to_date:
        fd = datetime.strptime(from_date, "%Y-%m-%d")
        td = datetime.strptime(to_date, "%Y-%m-%d")
    else:
        td = datetime.now()
        fd = td - timedelta(days=days)
    data = kite.get_historical_data(instrument_token, fd, td, interval)
    return {"count": len(data), "data": data}


@app.get("/api/market/instruments")
async def get_instruments(exchange: Optional[str] = None):
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


@app.post("/api/engine/run-cycle")
async def engine_run_cycle():
    """Manually trigger one engine cycle (for testing)."""
    try:
        state = nifty_engine.run_cycle()
        return state
    except Exception as e:
        logger.exception("run_cycle failed")
        return {"error": str(e), "engine_status": "ERROR"}


@app.post("/api/engine/square-off")
async def engine_square_off():
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
async def set_auto_trade_mode(req: AutoTradeModeRequest):
    """Switch between off/paper/live auto-trading."""
    if req.mode not in ("off", "paper", "live"):
        raise HTTPException(status_code=400, detail="Mode must be 'off', 'paper', or 'live'")
    nifty_engine.auto_trade_mode = req.mode
    return {"success": True, "mode": req.mode}


@app.post("/api/engine/force-exit/{trade_id}")
async def force_exit_position(trade_id: str, reason: str = "MANUAL"):
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
async def telegram_test():
    """Send a test message to verify Telegram bot setup."""
    result = telegram.test()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Send failed"))
    return result


class TelegramMessageRequest(BaseModel):
    message: str


@app.post("/api/telegram/send")
async def telegram_send(body: TelegramMessageRequest):
    """Send a custom Telegram message."""
    if not telegram.enabled:
        raise HTTPException(status_code=400, detail="Telegram not configured")
    telegram.send_custom(body.message)
    return {"success": True}


@app.post("/api/telegram/summary")
async def telegram_daily_summary():
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
async def send_eod_report():
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


@app.post("/api/backtest/run")


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
async def start_oi_feed():
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
async def stop_oi_feed():
    """Stop the live OI feed."""
    feed = nifty_engine._oi_feed
    if feed:
        feed.stop()
        nifty_engine._oi_feed_started = False
        return {"success": True, "message": "Live OI feed stopped"}
    return {"success": False, "message": "OI feed not running"}
async def run_backtest(
    strategy: str = "ORB",
    days: int = 30,
    capital: float = 1000000,
):
    """Run a historical backtest."""
    from backtester import BacktestEngine, DataLoader

    loader = DataLoader()
    df = loader.load_nifty_history(days=days, interval="5minute", source="yfinance")
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Failed to load historical data")

    sessions = loader.split_by_session(df)
    if not sessions:
        raise HTTPException(status_code=400, detail="No valid sessions in data")

    engine = BacktestEngine(strategy=strategy, capital=capital)
    results = engine.run(sessions)
    return results.to_dict()


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
            state = nifty_engine.run_cycle()
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
# Dashboard HTML page (self-contained)
# ============================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NIFTY Options Engine – Live Dashboard</title>
<style>
  :root { --bg: #0f1117; --card: #1a1d29; --accent: #3b82f6; --green: #22c55e;
           --red: #ef4444; --yellow: #eab308; --text: #e2e8f0; --muted: #94a3b8; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
         color: var(--text); padding: 16px; }
  h1 { font-size: 1.4rem; margin-bottom: 12px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          gap: 12px; margin-bottom: 16px; }
  .card { background: var(--card); border-radius: 10px; padding: 16px; }
  .card h2 { font-size: 1rem; color: var(--muted); margin-bottom: 8px; }
  .stat { font-size: 2rem; font-weight: 700; }
  .stat.green { color: var(--green); }
  .stat.red { color: var(--red); }
  .stat.yellow { color: var(--yellow); }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
           font-size: 0.75rem; font-weight: 600; }
  .badge-green { background: #22c55e33; color: var(--green); }
  .badge-red { background: #ef444433; color: var(--red); }
  .badge-yellow { background: #eab30833; color: var(--yellow); }
  .badge-blue { background: #3b82f633; color: var(--accent); }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; color: var(--muted); padding: 6px; border-bottom: 1px solid #2a2d3a; }
  td { padding: 6px; border-bottom: 1px solid #1f2233; }
  .bar-bg { background: #2a2d3a; border-radius: 4px; height: 8px; width: 100%; }
  .bar-fill { height: 8px; border-radius: 4px; background: var(--accent); }
  .pnl-bar { height: 4px; border-radius: 2px; margin-top: 4px; }
  .meta { font-size: 0.75rem; color: var(--muted); margin-top: 4px; }
  #status-dot { width: 10px; height: 10px; border-radius: 50%;
                display: inline-block; margin-right: 6px; }
  .connected { background: var(--green); }
  .disconnected { background: var(--red); }
  .flex { display: flex; align-items: center; gap: 8px; }
  .conditions li { font-size: 0.8rem; color: var(--muted); margin: 2px 0; }
  button { background: var(--accent); color: #fff; border: none; padding: 8px 16px;
           border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
  button:hover { opacity: 0.85; }
  button.danger { background: var(--red); }
  a.kite-chart-link { color: var(--accent); text-decoration: none; font-weight: 700;
    cursor: pointer; transition: color 0.15s; }
  a.kite-chart-link:hover { color: #60a5fa; text-decoration: underline; }
  .news-section { margin-top: 16px; }
  .news-tabs { display: flex; gap: 6px; margin-bottom: 10px; }
  .news-tab { background: var(--card); color: var(--muted); border: 1px solid #2a2d3a;
              padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; }
  .news-tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .news-list { max-height: 440px; overflow-y: auto; }
  .news-item { padding: 10px 0; border-bottom: 1px solid #1f2233; }
  .news-item:last-child { border-bottom: none; }
  .news-title { font-size: 0.9rem; font-weight: 600; margin-bottom: 3px; }
  .news-title a { color: var(--text); text-decoration: none; }
  .news-title a:hover { color: var(--accent); }
  .news-summary { font-size: 0.78rem; color: var(--muted); line-height: 1.4;
                  max-height: 2.8em; overflow: hidden; }
  .news-meta { font-size: 0.7rem; color: #64748b; margin-top: 3px; display: flex; gap: 10px; }
  .sentiment-bar { display: flex; gap: 12px; align-items: center; margin: 4px 0 8px; }
  .mood-chip { padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 700; }
  .mood-BULLISH { background: #22c55e22; color: var(--green); }
  .mood-BEARISH { background: #ef444422; color: var(--red); }
  .mood-NEUTRAL { background: #eab30822; color: var(--yellow); }
</style>
</head>
<body>

<div class="flex" style="justify-content: space-between; margin-bottom: 12px;">
  <h1>🎯 NIFTY Options Engine</h1>
  <div class="flex">
    <span id="status-dot" class="disconnected"></span>
    <span id="ws-status">Disconnected</span>
    <button onclick="runCycle()">Run Cycle</button>
    <button class="danger" onclick="squareOff()">Square Off All</button>
  </div>
</div>

<div id="offline-bar" style="display:none; align-items:center; gap:12px; padding:8px 16px; margin-bottom:12px; background:#1a1a2e; border:1px solid #ffd700; border-radius:8px;">
  <span class="offline-label" style="color:#ffd700; font-weight:600;">🌙 Market Closed</span>
  <button id="offline-btn" onclick="fetchOfflineData()" style="margin-left:auto; background:#ffd700; color:#0a0a1a; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-weight:600;">🔄 Fetch Offline Data</button>
</div>

<!-- Row 1: Key metrics -->
<div class="grid">
  <div class="card">
    <h2>Engine Status</h2>
    <div class="stat" id="engine-status">IDLE</div>
    <div class="meta" id="last-update">—</div>
  </div>
  <div class="card">
    <h2>NIFTY Spot</h2>
    <div class="stat" id="nifty-spot">—</div>
    <div class="meta" id="orb-range">ORB: —</div>
  </div>
  <div class="card">
    <h2>Daily P&L</h2>
    <div class="stat" id="daily-pnl">₹0</div>
    <div class="meta" id="pnl-detail">Realised: — | Unrealised: —</div>
    <div class="bar-bg"><div class="pnl-bar" id="pnl-bar" style="width:0;background:var(--green)"></div></div>
  </div>
  <div class="card">
    <h2>Open Positions</h2>
    <div class="stat" id="open-pos">0</div>
    <div class="meta" id="trades-count">Trades today: 0</div>
  </div>
  <div class="card">
    <h2>OI Analysis</h2>
    <div style="display:flex;gap:16px;align-items:baseline;">
      <div><span style="font-size:0.8rem;color:var(--muted)">CE OI:</span> <span class="stat" style="font-size:1.5rem" id="ce-oi-chg">—</span></div>
      <div><span style="font-size:0.8rem;color:var(--muted)">PE OI:</span> <span class="stat" style="font-size:1.5rem" id="pe-oi-chg">—</span></div>
    </div>
    <div class="meta" id="oi-interpretation">OI change % vs previous day (ATM ± 3 strikes)</div>
  </div>
  <div class="card">
    <h2>Safety Filters</h2>
    <div style="font-size:0.85rem;">
      <div style="margin-bottom:4px;">VIX: <b id="sf-vix">—</b> <span id="sf-vix-status" class="badge badge-green">OK</span></div>
      <div style="margin-bottom:4px;">Gap: <b id="sf-gap">NONE</b></div>
      <div style="margin-bottom:4px;">Multi-TF: 15m=<b id="sf-st15m">—</b> 1h=<b id="sf-st1h">—</b></div>
      <div style="margin-bottom:4px;">Strategies: <span id="sf-strategies">ORB</span></div>
    </div>
    <div class="meta">Orders: <span id="sf-orders">0</span>/<span id="sf-orders-max">10</span></div>
  </div>
</div>

<!-- Row 2: Signal + Top Scores -->
<div class="grid">
  <div class="card">
    <h2>Active Signal</h2>
    <div id="signal-box">
      <span class="badge badge-blue">NO SIGNAL</span>
    </div>
    <ul class="conditions" id="conditions-list"></ul>
  </div>
  <div class="card" style="grid-column: span 2;">
    <h2>Top Scored Candidates (Top 10%)</h2>
    <table>
      <thead>
        <tr><th>Option 📈</th><th>Type</th><th>Entry</th><th>SL</th><th>TGT 1</th><th>TGT 2</th><th>TGT 3</th><th>R:R%</th><th>Score</th><th>OI Chg%</th><th>OI Signal</th><th>Expiry</th></tr>
      </thead>
      <tbody id="scores-body"></tbody>
    </table>
    <div class="meta" id="score-meta">—</div>
  </div>
</div>

<!-- Row 3: Trade log -->
<div class="card">
  <h2>Trades Today</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Symbol</th><th>Type</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>Score</th><th>Status</th></tr>
    </thead>
    <tbody id="trades-body"></tbody>
  </table>
</div>

<!-- Row 3.2: Active Auto-Managed Positions with Trailing SL -->
<div class="card" style="margin-top:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <h2>🎯 Active Positions – Trailing SL</h2>
    <button onclick="loadActivePositions()" style="font-size:0.75rem;padding:4px 10px;">⟳ Refresh</button>
  </div>
  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Type</th><th>Entry</th><th>Qty</th>
        <th>Init SL</th><th>Trailing SL</th><th>Phase</th><th>Watermark</th>
        <th>T1</th><th>T2</th><th>T3</th>
        <th>Targets Hit</th><th>Last TSL Update</th>
      </tr>
    </thead>
    <tbody id="auto-pos-body"></tbody>
  </table>
  <div class="meta" id="auto-pos-meta" style="margin-top:6px;">Click "Refresh" or positions update automatically.</div>
</div>

<!-- Row 3.5: EOD Trade Journal -->
<div class="card" id="eod-section" style="margin-top:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <h2>📋 EOD Trade Journal</h2>
    <div style="display:flex;gap:8px;">
      <button onclick="fetchEODReport()" style="font-size:0.75rem;padding:4px 10px;">📊 Load Report</button>
      <button onclick="sendEODTelegram()" style="font-size:0.75rem;padding:4px 10px;background:var(--green);">📲 Send to Telegram</button>
    </div>
  </div>
  <div id="eod-summary" style="display:none;margin-top:10px;">
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-bottom:12px;">
      <div style="padding:8px;background:#1f2233;border-radius:6px;text-align:center;">
        <div class="meta">Signals</div><div class="stat" style="font-size:1.4rem;" id="eod-signals">0</div>
      </div>
      <div style="padding:8px;background:#1f2233;border-radius:6px;text-align:center;">
        <div class="meta">Candidates</div><div class="stat" style="font-size:1.4rem;" id="eod-candidates">0</div>
      </div>
      <div style="padding:8px;background:#1f2233;border-radius:6px;text-align:center;">
        <div class="meta">Activated</div><div class="stat" style="font-size:1.4rem;" id="eod-activated">0</div>
      </div>
      <div style="padding:8px;background:#1f2233;border-radius:6px;text-align:center;">
        <div class="meta">Win Rate</div><div class="stat" style="font-size:1.4rem;" id="eod-winrate">—</div>
      </div>
      <div style="padding:8px;background:#1f2233;border-radius:6px;text-align:center;">
        <div class="meta">Net P&L</div><div class="stat" style="font-size:1.4rem;" id="eod-pnl">₹0</div>
      </div>
      <div style="padding:8px;background:#1f2233;border-radius:6px;text-align:center;">
        <div class="meta">Avg Duration</div><div class="stat" style="font-size:1.4rem;" id="eod-duration">—</div>
      </div>
    </div>

    <h3 style="font-size:0.9rem;color:var(--muted);margin:12px 0 6px;">Trade Details</h3>
    <table>
      <thead>
        <tr>
          <th>Call</th><th>Symbol</th><th>Type</th><th>Signal Entry</th>
          <th>Activated?</th><th>Actual Entry</th><th>Exit</th>
          <th>P&L</th><th>Duration</th><th>Reason</th><th>Score</th>
        </tr>
      </thead>
      <tbody id="eod-trades-body"></tbody>
    </table>
  </div>
  <div id="eod-empty" class="meta" style="margin-top:8px;">Click "Load Report" to view today's trade journal.</div>
</div>

<!-- Row 4: News Section -->
<div class="news-section">
  <div class="grid">
    <div class="card">
      <h2>📰 Market Mood</h2>
      <div class="sentiment-bar">
        <div class="mood-chip" id="mood-chip">LOADING</div>
        <span class="meta" id="mood-stats">—</span>
      </div>
      <div style="display:flex;gap:16px;margin-top:6px;">
        <div><span style="color:var(--green)">▲</span> Bullish: <b id="bull-count">0</b></div>
        <div><span style="color:var(--red)">▼</span> Bearish: <b id="bear-count">0</b></div>
        <div><span style="color:var(--yellow)">—</span> Neutral: <b id="neut-count">0</b></div>
      </div>
      <div class="meta" id="news-updated">Last updated: —</div>
      <button style="margin-top:8px;font-size:0.75rem;padding:4px 10px;" onclick="refreshNews()">⟳ Refresh News</button>
    </div>
    <div class="card" style="grid-column: span 2;">
      <h2>📰 Today's Market News</h2>
      <div class="news-tabs">
        <div class="news-tab active" data-cat="top" onclick="switchTab(this,'top')">Top Stories</div>
        <div class="news-tab" data-cat="india" onclick="switchTab(this,'india')">India</div>
        <div class="news-tab" data-cat="global" onclick="switchTab(this,'global')">Global</div>
        <div class="news-tab" data-cat="nifty" onclick="switchTab(this,'nifty')">NIFTY</div>
        <div class="news-tab" data-cat="commodity" onclick="switchTab(this,'commodity')">Commodities</div>
        <div class="news-tab" data-cat="forex" onclick="switchTab(this,'forex')">Forex</div>
      </div>
      <div class="news-list" id="news-list"><div class="meta">Loading news...</div></div>
    </div>
  </div>
</div>

<script>
const WS_URL = `ws://${location.host}/ws/dashboard`;
let ws;

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    document.getElementById('status-dot').className = 'connected';
    document.getElementById('ws-status').textContent = 'Live';
  };
  ws.onclose = () => {
    document.getElementById('status-dot').className = 'disconnected';
    document.getElementById('ws-status').textContent = 'Disconnected';
    setTimeout(connect, 3000);
  };
  ws.onmessage = (e) => {
    try { updateDashboard(JSON.parse(e.data)); } catch {}
  };
}

function updateDashboard(d) {
  // Engine status
  const es = d.engine_status || 'IDLE';
  document.getElementById('engine-status').textContent = es;
  document.getElementById('engine-status').className = 'stat ' +
    (es.includes('HALTED') || es.includes('CLOSED') ? 'red' :
     es.includes('SIGNAL') ? 'green' : 'yellow');
  document.getElementById('last-update').textContent =
    'Cycle #' + (d.cycle_count||0) + ' | ' + (d.last_update||'').replace('T',' ').slice(0,19);

  // Show/hide market-closed banner + offline btn
  const offlineBar = document.getElementById('offline-bar');
  if (es === 'MARKET_CLOSED' || es === 'IDLE' || es === 'DISCONNECTED') {
    offlineBar.style.display = 'flex';
    offlineBar.querySelector('.offline-label').textContent =
      es === 'DISCONNECTED' ? '⚠ Broker disconnected' : '🌙 Market Closed – showing last available data';
  } else {
    offlineBar.style.display = 'none';
  }

  // Spot + ORB
  document.getElementById('nifty-spot').textContent = d.nifty_spot ? d.nifty_spot.toFixed(2) : '—';
  const orb = d.orb || {};
  document.getElementById('orb-range').textContent = orb.captured
    ? `ORB: [${orb.low?.toFixed(2)} – ${orb.high?.toFixed(2)}]` : 'ORB: waiting…';

  // P&L
  const risk = d.risk || {};
  const pnl = risk.daily_pnl || 0;
  document.getElementById('daily-pnl').textContent = '₹' + pnl.toFixed(2);
  document.getElementById('daily-pnl').className = 'stat ' + (pnl >= 0 ? 'green' : 'red');
  document.getElementById('pnl-detail').textContent =
    `Realised: ₹${(risk.realised_pnl||0).toFixed(2)} | Unrealised: ₹${(risk.unrealised_pnl||0).toFixed(2)}`;
  const limit = risk.daily_loss_limit || 1;
  const pnlPct = Math.min(100, Math.abs(pnl) / limit * 100);
  const bar = document.getElementById('pnl-bar');
  bar.style.width = pnlPct + '%';
  bar.style.background = pnl >= 0 ? 'var(--green)' : 'var(--red)';

  // Open positions
  document.getElementById('open-pos').textContent = risk.open_positions || 0;
  document.getElementById('trades-count').textContent =
    `Trades today: ${risk.total_trades_today||0} | Halted: ${risk.trading_halted?'YES':'NO'}`;

  // OI Analysis
  const oi = d.oi_analysis || {};
  const ceOiEl = document.getElementById('ce-oi-chg');
  const peOiEl = document.getElementById('pe-oi-chg');
  const ceChg = oi.ce_oi_change_pct || 0;
  const peChg = oi.pe_oi_change_pct || 0;
  ceOiEl.textContent = ceChg !== 0 ? (ceChg > 0 ? '+' : '') + ceChg.toFixed(1) + '%' : '—';
  ceOiEl.style.color = ceChg > 0 ? 'var(--red)' : ceChg < 0 ? 'var(--green)' : 'var(--muted)';
  peOiEl.textContent = peChg !== 0 ? (peChg > 0 ? '+' : '') + peChg.toFixed(1) + '%' : '—';
  peOiEl.style.color = peChg > 0 ? 'var(--red)' : peChg < 0 ? 'var(--green)' : 'var(--muted)';
  // Interpretation
  const oiInterpEl = document.getElementById('oi-interpretation');
  if (ceChg !== 0 || peChg !== 0) {
    let mood = '';
    if (ceChg < -2 && peChg > 2) mood = '🟢 Bullish OI (CE unwinding + PE writing)';
    else if (ceChg > 2 && peChg < -2) mood = '🔴 Bearish OI (CE writing + PE unwinding)';
    else if (ceChg > 5) mood = '🔴 Heavy CE writing (bearish)';
    else if (peChg > 5) mood = '🔴 Heavy PE buildup (bearish)';
    else if (ceChg < -5) mood = '🟢 CE short covering (bullish)';
    else if (peChg < -5) mood = '🟢 PE unwinding (bullish)';
    else mood = '🟡 Mixed OI signals';
    oiInterpEl.textContent = mood;
  }

  // Safety Filters
  const sf = d.safety_filters || {};
  const vixVal = d.india_vix || sf.india_vix || 0;
  document.getElementById('sf-vix').textContent = vixVal ? vixVal.toFixed(1) : '—';
  const vixSt = document.getElementById('sf-vix-status');
  if (vixVal > 20) { vixSt.textContent='HIGH'; vixSt.className='badge badge-red'; }
  else if (vixVal > 15) { vixSt.textContent='WARN'; vixSt.className='badge badge-yellow'; }
  else { vixSt.textContent='OK'; vixSt.className='badge badge-green'; }
  const gapType = sf.gap_type || d.gap_type || 'NONE';
  const gapPct = sf.gap_pct || d.gap_pct || 0;
  document.getElementById('sf-gap').textContent = gapType + (gapPct ? ' (' + gapPct.toFixed(2) + '%)' : '');
  document.getElementById('sf-st15m').textContent = sf.st_dir_15m === 1 ? '▲' : sf.st_dir_15m === -1 ? '▼' : '—';
  document.getElementById('sf-st1h').textContent = sf.st_dir_1h === 1 ? '▲' : sf.st_dir_1h === -1 ? '▼' : '—';
  const strats = [];
  strats.push('ORB');
  if (sf.vwap_enabled) strats.push('VWAP');
  if (sf.expiry_sell_enabled) strats.push('ExpSell');
  if (sf.trend_mode_enabled) strats.push('TREND');
  document.getElementById('sf-strategies').textContent = strats.join(' | ') || 'ORB';
  document.getElementById('sf-orders').textContent = risk.total_trades_today || 0;
  document.getElementById('sf-orders-max').textContent = sf.max_orders || 10;

  // Signal
  const sig = d.signal;
  const box = document.getElementById('signal-box');
  const cList = document.getElementById('conditions-list');
  cList.innerHTML = '';
  if (sig) {
    const cls = sig.type === 'BUY' ? 'badge-green' : 'badge-red';
    box.innerHTML = `<span class="badge ${cls}">${sig.type}</span>
      <span style="margin-left:8px">Entry: ${sig.entry?.toFixed(2)} | SL: ${sig.sl?.toFixed(2)} | Tgt: ${sig.target?.toFixed(2)}</span>
      <div class="meta">Confidence: ${sig.confidence?.toFixed(1)}%</div>`;
    (sig.conditions || []).forEach(c => {
      const li = document.createElement('li');
      li.textContent = '✓ ' + c;
      cList.appendChild(li);
    });
  } else {
    box.innerHTML = '<span class="badge badge-blue">NO SIGNAL</span>';
  }

  // Scores table
  const tc = d.top_candidates || {};
  const tops = tc.top || [];
  const tbody = document.getElementById('scores-body');
  tbody.innerHTML = '';
  tops.forEach(s => {
    const tr = document.createElement('tr');
    const name = s.display_name || s.symbol;
    const cls = s.type === 'CE' ? 'badge-green' : 'badge-red';
    const oiChg = s.oi_change_pct || 0;
    const oiCls = oiChg > 0 ? 'color:var(--green)' : oiChg < 0 ? 'color:var(--red)' : '';
    const oiInterp = s.oi_interpretation || '—';
    // Build Kite chart URL: https://kite.zerodha.com/markets/ext/chart/web/tvc/NFO-OPT/{tradingsymbol}/{instrument_token}
    const kiteChartUrl = s.instrument_token
      ? `https://kite.zerodha.com/markets/ext/chart/web/tvc/NFO-OPT/${encodeURIComponent(s.symbol)}/${s.instrument_token}`
      : null;
    const nameHtml = kiteChartUrl
      ? `<a class="kite-chart-link" href="${kiteChartUrl}" target="_blank" title="Open premium chart on Kite">${name}</a>`
      : `<b>${name}</b>`;
    tr.innerHTML = `<td>${nameHtml}</td>
      <td><span class="badge ${cls}">${s.type}</span></td>
      <td><b>₹${(s.entry||s.ltp)?.toFixed(0)}</b></td>
      <td>₹${(s.stoploss||0)?.toFixed(0)}</td>
      <td>₹${(s.target1||0)?.toFixed(0)}</td>
      <td>₹${(s.target2||0)?.toFixed(0)}</td>
      <td>₹${(s.target3||0)?.toFixed(0)}+</td>
      <td><b>${(s.risk_reward_pct||0)?.toFixed(1)}%</b></td>
      <td><b>${s.score?.toFixed(1)}</b></td>
      <td style="${oiCls}">${oiChg > 0 ? '+' : ''}${oiChg.toFixed(1)}%</td>
      <td class="meta">${oiInterp}</td>
      <td>${s.expiry||''}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('score-meta').textContent =
    `${tc.count||0} candidates (top 10% of ${d.all_candidates_count||'—'}) | Avg score: ${tc.avg_score||0} | Max: ${tc.max_score||0}`;

  // Trades table
  const trades = d.trades || [];
  const ttbody = document.getElementById('trades-body');
  ttbody.innerHTML = '';
  trades.forEach(t => {
    const tr = document.createElement('tr');
    const pnlCls = t.pnl >= 0 ? 'green' : 'red';
    tr.innerHTML = `<td>${t.trade_id?.slice(-8)}</td><td>${t.symbol}</td><td>${t.type}</td>
      <td>₹${t.entry?.toFixed(2)}</td><td>${t.exit?'₹'+t.exit.toFixed(2):'—'}</td>
      <td>${t.qty}</td><td style="color:var(--${pnlCls})">₹${t.pnl?.toFixed(2)}</td>
      <td>${t.score?.toFixed(1)}</td><td>${t.status}</td>`;
    ttbody.appendChild(tr);
  });
}

function runCycle() { ws && ws.send(JSON.stringify({action: 'run_cycle'})); }
function squareOff() {
  if (confirm('Square off ALL open positions?'))
    fetch('/api/engine/square-off', {method:'POST'}).then(r=>r.json()).then(d=>alert(d.message));
}

// ── Active Positions with Trailing SL ──
function loadActivePositions() {
  fetch('/api/engine/active-positions').then(r=>r.json()).then(data=>{
    const positions = data.positions || [];
    const tbody = document.getElementById('auto-pos-body');
    tbody.innerHTML = '';
    positions.forEach(p => {
      const tr = document.createElement('tr');
      const typeCls = p.option_type === 'CE' ? 'badge-green' : 'badge-red';
      const phaseCls = p.trailing_phase === 'TIGHT' ? 'badge-green'
        : p.trailing_phase === 'TRAIL_T2' ? 'badge-green'
        : p.trailing_phase === 'TRAIL_T1' ? 'badge-blue'
        : p.trailing_phase === 'BREAKEVEN' ? 'badge-yellow'
        : 'badge-red';
      const tslImproved = p.trailing_sl > p.sl;
      const tslStyle = tslImproved ? 'color:var(--green);font-weight:700' : '';
      const targets = [];
      if (p.t1_hit) targets.push('T1✓');
      if (p.t2_hit) targets.push('T2✓');
      if (p.t3_hit) targets.push('T3✓');
      const targetsStr = targets.length ? targets.join(' ') : '—';
      const lastUpdate = p.last_tsl_update || '—';
      tr.innerHTML = `
        <td><b>${p.symbol?.split(':').pop() || p.symbol}</b></td>
        <td><span class="badge ${typeCls}">${p.option_type}</span></td>
        <td>₹${(p.entry||0).toFixed(1)}</td>
        <td>${p.remaining_qty}/${p.qty}</td>
        <td>₹${(p.sl||0).toFixed(1)}</td>
        <td style="${tslStyle}">₹${(p.trailing_sl||0).toFixed(1)}</td>
        <td><span class="badge ${phaseCls}">${p.trailing_phase||'INITIAL'}</span></td>
        <td>₹${(p.highest_price||0).toFixed(1)}</td>
        <td>₹${(p.t1||0).toFixed(0)}</td>
        <td>₹${(p.t2||0).toFixed(0)}</td>
        <td>₹${(p.t3||0).toFixed(0)}</td>
        <td>${targetsStr}</td>
        <td class="meta">${lastUpdate}</td>`;
      tbody.appendChild(tr);
    });
    document.getElementById('auto-pos-meta').textContent =
      `${positions.length} active position(s) | Auto-refreshes every cycle`;
  }).catch(()=>{
    document.getElementById('auto-pos-meta').textContent = 'Failed to load positions';
  });
}
// Auto-refresh active positions every 30 seconds
setInterval(loadActivePositions, 30000);
// Load once on page load
setTimeout(loadActivePositions, 2000);
function fetchOfflineData() {
  document.getElementById('offline-btn').textContent = 'Loading…';
  fetch('/api/engine/offline-data').then(r=>r.json()).then(d=>{
    updateDashboard(d);
    document.getElementById('offline-btn').textContent = '🔄 Fetch Offline Data';
  }).catch(()=>{ document.getElementById('offline-btn').textContent = '🔄 Fetch Offline Data'; });
}

// ── EOD Trade Journal functions ──
function fetchEODReport() {
  document.getElementById('eod-empty').textContent = 'Loading…';
  fetch('/api/engine/eod-report').then(r=>r.json()).then(data=>{
    const s = data.summary || {};
    document.getElementById('eod-signals').textContent = s.total_signals || 0;
    document.getElementById('eod-candidates').textContent = s.total_candidates || 0;
    document.getElementById('eod-activated').textContent = s.activated_count || 0;
    document.getElementById('eod-winrate').textContent = (s.win_rate||0).toFixed(0) + '%';
    const pnl = s.total_pnl || 0;
    const pnlEl = document.getElementById('eod-pnl');
    pnlEl.textContent = '₹' + pnl.toFixed(0);
    pnlEl.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('eod-duration').textContent = s.avg_duration || '—';

    // Populate trade details table
    const tbody = document.getElementById('eod-trades-body');
    tbody.innerHTML = '';
    (data.trades || []).forEach((t, i) => {
      const tr = document.createElement('tr');
      const activated = t.activated;
      const pnlVal = t.pnl || 0;
      const pnlCls = pnlVal >= 0 ? 'green' : 'red';
      const activatedBadge = activated
        ? '<span class="badge badge-green">YES</span>'
        : '<span class="badge badge-yellow">NO</span>';
      tr.innerHTML = `<td>${i+1}</td>
        <td><b>${t.display_name||t.symbol}</b></td>
        <td><span class="badge ${t.option_type==='CE'?'badge-green':'badge-red'}">${t.option_type}</span></td>
        <td>₹${(t.suggested_entry||0).toFixed(0)}</td>
        <td>${activatedBadge}</td>
        <td>${activated?'₹'+(t.actual_entry||0).toFixed(2):'—'}</td>
        <td>${activated&&t.actual_exit?'₹'+t.actual_exit.toFixed(2):'—'}</td>
        <td style="color:var(--${pnlCls});font-weight:700">${activated?'₹'+pnlVal.toFixed(2):'—'}</td>
        <td>${t.duration||'—'}</td>
        <td>${t.exit_reason||t.status||'—'}</td>
        <td>${(t.score||0).toFixed(0)}</td>`;
      tbody.appendChild(tr);
    });

    document.getElementById('eod-summary').style.display = 'block';
    document.getElementById('eod-empty').style.display = 'none';
  }).catch(e=>{
    document.getElementById('eod-empty').textContent = 'Failed to load report.';
  });
}

function sendEODTelegram() {
  fetch('/api/engine/send-eod-report', {method:'POST'}).then(r=>r.json()).then(d=>{
    alert('EOD report sent to Telegram! (' + (d.messages_sent||0) + ' messages)');
  }).catch(()=>alert('Failed to send EOD report.'));
}

// ── News functions ──
let newsData = null;
let currentTab = 'top';

function fetchNews(refresh=false) {
  const url = refresh ? '/api/news?refresh=true' : '/api/news';
  fetch(url).then(r=>r.json()).then(data => {
    newsData = data;
    updateMood(data.sentiment_summary || {});
    renderNews(currentTab);
    document.getElementById('news-updated').textContent =
      'Last updated: ' + (data.last_updated||"").replace('T',' ').slice(0,19)
      + ' | ' + (data.total_articles||0) + ' articles';
  }).catch(()=>{});
}

function updateMood(s) {
  const chip = document.getElementById('mood-chip');
  chip.textContent = s.overall_mood || 'NEUTRAL';
  chip.className = 'mood-chip mood-' + (s.overall_mood||'NEUTRAL');
  document.getElementById('mood-stats').textContent = 'Avg sentiment: ' + (s.avg_sentiment||0).toFixed(4);
  document.getElementById('bull-count').textContent = s.bullish_count || 0;
  document.getElementById('bear-count').textContent = s.bearish_count || 0;
  document.getElementById('neut-count').textContent = s.neutral_count || 0;
}

function renderNews(tab) {
  if (!newsData) return;
  const list = document.getElementById('news-list');
  let articles = [];
  const cats = newsData.categories || {};
  if (tab === 'top') articles = newsData.top_stories || [];
  else if (tab === 'india') articles = [...(cats.india_market||[]), ...(cats.nifty||[]), ...(cats.sector||[])];
  else if (tab === 'global') articles = [...(cats.global_market||[]), ...(cats.commodity||[]), ...(cats.forex||[])];
  else articles = cats[tab] || [];
  // Sort by relevance
  articles.sort((a,b) => (b.relevance_score||0) - (a.relevance_score||0));
  if (!articles.length) {
    list.innerHTML = '<div class="meta">No articles in this category.</div>';
    return;
  }
  list.innerHTML = articles.slice(0, 30).map(a => {
    const sentCls = a.sentiment_label==='bullish' ? 'badge-green' : a.sentiment_label==='bearish' ? 'badge-red' : 'badge-yellow';
    const timeStr = a.published ? new Date(a.published).toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'}) : '';
    return `<div class="news-item">
      <div class="news-title"><a href="${a.url}" target="_blank">${a.title}</a></div>
      <div class="news-summary">${a.summary||''}</div>
      <div class="news-meta">
        <span>${a.source}</span><span>${timeStr}</span>
        <span class="badge ${sentCls}">${a.sentiment_label} ${a.sentiment>0?'+':''}${a.sentiment?.toFixed(3)||''}</span>
        <span>Rel: ${a.relevance_score?.toFixed(0)||0}</span>
      </div>
    </div>`;
  }).join('');
}

function switchTab(el, tab) {
  currentTab = tab;
  document.querySelectorAll('.news-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  renderNews(tab);
}

function refreshNews() { fetchNews(true); }

connect();
// Fetch news on load + every 10 min
fetchNews();
setInterval(fetchNews, 600000);
// Also poll engine every 30s as fallback
setInterval(() => {
  fetch('/api/engine/dashboard').then(r=>r.json()).then(updateDashboard).catch(()=>{});
}, 30000);
// Auto-fetch offline data on page load if market is closed
fetch('/api/engine/dashboard').then(r=>r.json()).then(d => {
  updateDashboard(d);
  const st = d.engine_status || 'IDLE';
  if (st === 'MARKET_CLOSED' || st === 'IDLE' || st === 'DISCONNECTED') {
    fetchOfflineData();
  }
}).catch(()=>{});
</script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Self-contained live dashboard."""
    return DASHBOARD_HTML


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
