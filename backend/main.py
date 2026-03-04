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
from strategies.supertrend_strategy import SupertrendStrategy
from strategies.nifty_options_orb import NiftyOptionsORBStrategy
from options_engine import NiftyOptionsEngine
from scoring_engine import ScoringEngine
from risk_manager import RiskManager
from news_aggregator import news_aggregator
from telegram_alerts import telegram

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
        # Broadcast to WebSocket clients
        asyncio.create_task(ws_dashboard_manager.broadcast(state))
    except Exception as e:
        logger.error(f"Engine cycle error: {e}")


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
    stoploss: Optional[float] = None
    squareoff: Optional[float] = None
    trailing_stoploss: Optional[float] = None


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
    scheduler.start()
    logger.info(f"Options engine scheduler started (every {settings.engine_cycle_seconds}s)")
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
        stoploss=order.stoploss,
        squareoff=order.squareoff,
        trailing_stoploss=order.trailing_stoploss,
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
    state = nifty_engine.run_cycle()
    return state


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
            msg = await websocket.receive_text()
            try:
                parsed = json.loads(msg)
                if parsed.get("action") == "run_cycle":
                    state = nifty_engine.run_cycle()
                    await websocket.send_json(state)
                elif parsed.get("action") == "get_state":
                    await websocket.send_json(nifty_engine.get_dashboard())
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
    except WebSocketDisconnect:
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
        <tr><th>Symbol</th><th>Strike</th><th>Type</th><th>Delta</th><th>Score</th><th>LTP</th><th>Volume</th></tr>
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
    tr.innerHTML = `<td>${s.symbol}</td><td>${s.strike}</td><td>${s.type}</td>
      <td>${s.delta?.toFixed(3)}</td><td><b>${s.score?.toFixed(1)}</b></td>
      <td>₹${s.ltp?.toFixed(2)}</td><td>${s.volume}</td>`;
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
function fetchOfflineData() {
  document.getElementById('offline-btn').textContent = 'Loading…';
  fetch('/api/engine/offline-data').then(r=>r.json()).then(d=>{
    updateDashboard(d);
    document.getElementById('offline-btn').textContent = '🔄 Fetch Offline Data';
  }).catch(()=>{ document.getElementById('offline-btn').textContent = '🔄 Fetch Offline Data'; });
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
