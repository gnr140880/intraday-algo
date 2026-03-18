"""
Auto Order Manager – Places and manages orders on Zerodha automatically.

Responsibilities:
  - Place entry orders (MARKET) when signal triggers
  - Place SL-M (stop-loss market) orders simultaneously
  - Handle partial exits at T1/T2 (modify SL order, place partial sell)
  - Full exit at T3 or on exit signals
  - Track all orders and their states
  - Retry failed orders
  - Log every order action for audit

Order flow:
  1. ENTRY: Market BUY order for CE/PE option
  2. SL: SL-M SELL order placed immediately after entry fill
  3. T1 HIT: Sell 1/3 qty, cancel old SL, place new SL at breakeven
  4. T2 HIT: Sell 1/3 qty, modify SL to T1 level
  5. T3 HIT or EXIT SIGNAL: Sell all remaining, cancel SL order
  6. SQUARE-OFF: Cancel all pending, sell all at market
"""
import logging
import uuid
import json
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path

from kite_client import KiteClient
from smart_sl_engine import SmartLevels
from risk_manager import RiskManager, TradeRecord
from telegram_alerts import telegram
from config import settings

logger = logging.getLogger(__name__)

NIFTY_LOT_SIZE = settings.nifty_lot_size
_PERSISTENCE_FILE = Path(__file__).resolve().parent / "positions_state.json"


@dataclass
class ManagedOrder:
    """Tracks an individual order through its lifecycle."""
    order_id: str
    trade_id: str
    symbol: str
    exchange: str = "NFO"
    order_type: str = "MARKET"      # MARKET, SL-M, SL, LIMIT
    transaction_type: str = "BUY"   # BUY, SELL
    quantity: int = 0
    price: float = 0.0
    trigger_price: float = 0.0
    status: str = "PENDING"         # PENDING, PLACED, FILLED, CANCELLED, FAILED
    purpose: str = "ENTRY"          # ENTRY, SL, PARTIAL_EXIT, FULL_EXIT, SQUARE_OFF
    placed_at: str = ""
    filled_at: str = ""
    filled_price: float = 0.0
    error: str = ""
    retries: int = 0


@dataclass
class ManagedPosition:
    """A fully managed position with entry, SL, and target tracking."""
    trade_id: str
    symbol: str
    option_type: str                # CE / PE
    entry_price: float
    entry_order_id: str
    sl_order_id: str = ""           # Current active SL order
    quantity: int = 0
    remaining_qty: int = 0
    smart_levels: Optional[SmartLevels] = None
    t1_hit: bool = False
    t2_hit: bool = False
    t3_hit: bool = False
    status: str = "ACTIVE"          # ACTIVE, PARTIAL, CLOSED
    orders: List[ManagedOrder] = field(default_factory=list)
    delta: float = 0.4
    score: float = 0.0
    created_at: str = ""
    # --- Trailing SL state ---
    trailing_sl: float = 0.0        # Current trailing SL price (only moves in favour)
    highest_price: float = 0.0      # Watermark: highest premium since entry
    trailing_phase: str = "INITIAL" # INITIAL → BREAKEVEN → TRAIL_T1 → TRAIL_T2 → TIGHT
    last_tsl_update: str = ""       # Timestamp of last TSL update


class AutoOrderManager:
    """
    Manages the complete order lifecycle on Zerodha.

    Usage:
      mgr = AutoOrderManager(risk_mgr)
      # On signal → entry
      pos = mgr.enter_trade(kite, symbol, "CE", smart_levels, qty, delta, score)
      # On target/exit_signal → partial or full exit
      mgr.handle_exit_signal(kite, pos, exit_signal)
      # At EOD → square off everything
      mgr.square_off_all(kite)
    """

    def __init__(self, risk_mgr: RiskManager):
        self.risk_mgr = risk_mgr
        self.positions: Dict[str, ManagedPosition] = {}  # trade_id → position
        self.order_log: List[ManagedOrder] = []
        self._daily_order_count: int = 0  # orders placed today
        self._daily_order_date: str = ""  # date string for daily reset

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def enter_trade(
        self,
        kite: KiteClient,
        symbol: str,
        option_type: str,
        smart_levels: SmartLevels,
        quantity: int,
        delta: float = 0.4,
        score: float = 0.0,
    ) -> Optional[ManagedPosition]:
        """
        Place entry order + SL order for a new trade.
        Returns ManagedPosition on success, None on failure.
        """
        # --- Daily order limit check ---
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._daily_order_date != today_str:
            self._daily_order_count = 0
            self._daily_order_date = today_str
        if self._daily_order_count >= settings.auto_trade_max_orders_per_day:
            logger.warning(
                f"Trade skipped: daily order limit reached ({self._daily_order_count}/{settings.auto_trade_max_orders_per_day}) for {symbol}"
            )
            return None

        trade_id = f"AT_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # --- Risk check ---
        risk_amount = abs(smart_levels.entry - smart_levels.option_sl) * quantity
        risk_check = self.risk_mgr.can_take_trade(risk_amount)
        if not risk_check["allowed"]:
            logger.warning(f"Trade skipped: risk manager blocked trade for {symbol} | Reasons: {risk_check['reasons']}")
            return None

        # --- Place ENTRY order (MARKET) ---
        entry_result = kite.place_order(
            tradingsymbol=symbol,
            exchange="NFO",
            transaction_type="BUY",
            quantity=quantity,
            order_type="MARKET",
            product="MIS",
            tag="AutoEntry",
        )

        if not entry_result.get("success"):
            logger.error(f"Trade skipped: entry order failed for {symbol} | Response: {entry_result}")
            return None

        entry_order_id = entry_result.get("order_id", "")
        entry_order = ManagedOrder(
            order_id=entry_order_id,
            trade_id=trade_id,
            symbol=symbol,
            transaction_type="BUY",
            quantity=quantity,
            purpose="ENTRY",
            status="PLACED",
            placed_at=datetime.now().isoformat(),
        )
        self.order_log.append(entry_order)

        # Use the smart SL level for the option premium
        sl_trigger = smart_levels.option_sl
        if sl_trigger <= 0:
            sl_trigger = round(smart_levels.entry * 0.85, 2)  # fallback 15% SL

        # --- Place SL-M order ---
        sl_result = kite.place_sl_order(
            tradingsymbol=symbol,
            exchange="NFO",
            transaction_type="SELL",
            quantity=quantity,
            trigger_price=sl_trigger,
            product="MIS",
            tag="AutoSL",
        )

        sl_order_id = ""
        if sl_result.get("success"):
            sl_order_id = sl_result.get("order_id", "")
            sl_order = ManagedOrder(
                order_id=sl_order_id,
                trade_id=trade_id,
                symbol=symbol,
                order_type="SL-M",
                transaction_type="SELL",
                quantity=quantity,
                trigger_price=sl_trigger,
                purpose="SL",
                status="PLACED",
                placed_at=datetime.now().isoformat(),
            )
            self.order_log.append(sl_order)
        else:
            logger.warning(f"Trade skipped: SL order failed for {symbol} | Response: {sl_result}. Will manage via polling.")

        # --- Create managed position ---
        entry_px = smart_levels.entry if smart_levels.entry > 0 else 0
        pos = ManagedPosition(
            trade_id=trade_id,
            symbol=symbol,
            option_type=option_type,
            entry_price=entry_px,
            entry_order_id=entry_order_id,
            sl_order_id=sl_order_id,
            quantity=quantity,
            remaining_qty=quantity,
            smart_levels=smart_levels,
            delta=delta,
            score=score,
            created_at=datetime.now().isoformat(),
            trailing_sl=sl_trigger,              # Start TSL at initial SL
            highest_price=entry_px,              # Watermark starts at entry
            trailing_phase="INITIAL",
        )
        self.positions[trade_id] = pos
        self._daily_order_count += 1  # Increment daily order counter

        # Register with risk manager
        trade_record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            option_type=option_type,
            entry_price=smart_levels.entry,
            entry_time=datetime.now().isoformat(),
            quantity=quantity,
            sl=sl_trigger,
            target=smart_levels.option_t1,
            trailing_sl=sl_trigger,
            score=score,
        )
        self.risk_mgr.register_trade(trade_record)

        # Telegram alert
        telegram.alert_trade_entry(
            symbol=symbol,
            option_type=option_type,
            entry_price=smart_levels.entry,
            qty=quantity,
            sl=sl_trigger,
            target=smart_levels.option_t1,
            score=score,
            trade_id=trade_id,
        )

        logger.info(
            f"AUTO TRADE ENTERED: {trade_id} {symbol} {option_type} "
            f"qty={quantity} entry~{smart_levels.entry:.2f} "
            f"SL={sl_trigger:.2f} T1={smart_levels.option_t1:.2f} "
            f"T2={smart_levels.option_t2:.2f} T3={smart_levels.option_t3:.2f}"
        )

        self.save_state()  # Persist after entry
        return pos

    # ------------------------------------------------------------------
    # Partial exit (at T1, T2 targets)
    # ------------------------------------------------------------------
    def partial_exit(
        self,
        kite: KiteClient,
        trade_id: str,
        exit_qty: int,
        new_sl_trigger: float,
        reason: str = "TARGET_HIT",
    ) -> bool:
        """
        Exit a portion of the position and update the SL order.
        """
        pos = self.positions.get(trade_id)
        if not pos or pos.status == "CLOSED":
            return False

        # Round to lot size
        exit_qty = max(NIFTY_LOT_SIZE, (exit_qty // NIFTY_LOT_SIZE) * NIFTY_LOT_SIZE)
        exit_qty = min(exit_qty, pos.remaining_qty)

        if exit_qty <= 0:
            return False

        # --- Place partial SELL order ---
        sell_result = kite.place_order(
            tradingsymbol=pos.symbol,
            exchange="NFO",
            transaction_type="SELL",
            quantity=exit_qty,
            order_type="MARKET",
            product="MIS",
            tag=f"Auto_{reason}",
        )

        if not sell_result.get("success"):
            logger.error(f"Partial exit failed for {pos.symbol}: {sell_result}")
            return False

        sell_order = ManagedOrder(
            order_id=sell_result.get("order_id", ""),
            trade_id=trade_id,
            symbol=pos.symbol,
            transaction_type="SELL",
            quantity=exit_qty,
            purpose="PARTIAL_EXIT",
            status="PLACED",
            placed_at=datetime.now().isoformat(),
        )
        self.order_log.append(sell_order)
        pos.orders.append(sell_order)

        pos.remaining_qty -= exit_qty
        new_remaining = pos.remaining_qty

        # --- Cancel old SL order, place new one with updated trigger ---
        if pos.sl_order_id:
            cancel_result = kite.cancel_order(pos.sl_order_id)
            if cancel_result.get("success"):
                logger.info(f"Cancelled old SL order {pos.sl_order_id}")
            else:
                logger.warning(f"Failed to cancel old SL: {cancel_result}")

        if new_remaining > 0 and new_sl_trigger > 0:
            new_sl_result = kite.place_sl_order(
                tradingsymbol=pos.symbol,
                exchange="NFO",
                transaction_type="SELL",
                quantity=new_remaining,
                trigger_price=round(new_sl_trigger, 2),
                product="MIS",
                tag="AutoSL_Updated",
            )
            if new_sl_result.get("success"):
                pos.sl_order_id = new_sl_result.get("order_id", "")
                logger.info(f"New SL order placed: trigger={new_sl_trigger:.2f} qty={new_remaining}")
            else:
                logger.warning(f"New SL order failed: {new_sl_result}")
                pos.sl_order_id = ""

        if new_remaining <= 0:
            pos.status = "CLOSED"
        else:
            pos.status = "PARTIAL"

        logger.info(
            f"PARTIAL EXIT: {trade_id} {pos.symbol} sold {exit_qty}, "
            f"remaining {new_remaining}, new SL trigger {new_sl_trigger:.2f}, "
            f"reason={reason}"
        )

        return True

    # ------------------------------------------------------------------
    # Full exit
    # ------------------------------------------------------------------
    def full_exit(
        self,
        kite: KiteClient,
        trade_id: str,
        reason: str = "EXIT_SIGNAL",
    ) -> bool:
        """Exit entire remaining position."""
        pos = self.positions.get(trade_id)
        if not pos or pos.status == "CLOSED" or pos.remaining_qty <= 0:
            return False

        # Cancel SL order first
        if pos.sl_order_id:
            kite.cancel_order(pos.sl_order_id)

        # Place full sell
        sell_result = kite.place_order(
            tradingsymbol=pos.symbol,
            exchange="NFO",
            transaction_type="SELL",
            quantity=pos.remaining_qty,
            order_type="MARKET",
            product="MIS",
            tag=f"Auto_{reason}",
        )

        if not sell_result.get("success"):
            logger.error(f"Full exit failed for {pos.symbol}: {sell_result}")
            return False

        sell_order = ManagedOrder(
            order_id=sell_result.get("order_id", ""),
            trade_id=trade_id,
            symbol=pos.symbol,
            transaction_type="SELL",
            quantity=pos.remaining_qty,
            purpose="FULL_EXIT",
            status="PLACED",
            placed_at=datetime.now().isoformat(),
        )
        self.order_log.append(sell_order)
        pos.orders.append(sell_order)

        # Get current LTP for P&L
        try:
            ltp_data = kite.get_ltp([f"NFO:{pos.symbol}"])
            exit_price = ltp_data.get(f"NFO:{pos.symbol}", {}).get("last_price", pos.entry_price)
        except Exception:
            exit_price = pos.entry_price

        # Close in risk manager
        self.risk_mgr.close_trade(trade_id, exit_price, reason)

        # Telegram alert
        pnl = (exit_price - pos.entry_price) * pos.remaining_qty
        logger.info(
            f"Sending Telegram EXIT alert for {pos.symbol} | Reason: {reason} | Trade ID: {trade_id} | Exit Price: {exit_price} | Qty: {pos.remaining_qty}"
        )
        telegram.alert_trade_exit(
            symbol=pos.symbol,
            option_type=pos.option_type,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.remaining_qty,
            pnl=pnl,
            reason=reason,
            trade_id=trade_id,
        )

        pos.remaining_qty = 0
        pos.status = "CLOSED"

        logger.info(f"FULL EXIT: {trade_id} {pos.symbol} reason={reason} exit~{exit_price:.2f}")
        return True

    # ------------------------------------------------------------------
    # Handle exit signal from ExitSignalGenerator
    # ------------------------------------------------------------------
    def handle_exit_signal(
        self,
        kite: KiteClient,
        trade_id: str,
        exit_signal,  # ExitSignal from exit_signal_generator
    ):
        """
        Process an exit signal — partial or full exit based on priority.
        Updates trailing SL on the position when targets are hit.
        """
        pos = self.positions.get(trade_id)
        if not pos or pos.status == "CLOSED":
            return

        exit_pct = exit_signal.exit_pct
        new_tsl = exit_signal.new_trailing_sl

        # Update trailing SL on position (only moves up, never down)
        if new_tsl and new_tsl > 0 and new_tsl > pos.trailing_sl:
            pos.trailing_sl = new_tsl
            pos.last_tsl_update = datetime.now().isoformat()
            # Telegram alert for TSL tightening
            telegram.send_custom(
                f"🔔 <b>Trailing SL tightened</b>\n"
                f"Symbol: <b>{pos.symbol}</b> ({pos.option_type})\n"
                f"New TSL: <b>₹{new_tsl:.2f}</b>\n"
                f"Phase: {pos.trailing_phase}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )

        if exit_pct >= 1.0:
            # Full exit
            self.full_exit(kite, trade_id, reason=exit_signal.reason.value)

        elif exit_pct > 0:
            # Partial exit
            exit_qty = max(
                NIFTY_LOT_SIZE,
                int(pos.remaining_qty * exit_pct) // NIFTY_LOT_SIZE * NIFTY_LOT_SIZE,
            )
            # Use the higher of: exit signal TSL or position's current TSL
            sl_trigger = round(pos.trailing_sl, 2) if pos.trailing_sl > 0 else (
                pos.smart_levels.option_sl if pos.smart_levels else 0
            )
            self.partial_exit(kite, trade_id, exit_qty, sl_trigger, exit_signal.reason.value)

            # Update target-hit flags and trailing phase
            from exit_signal_generator import ExitReason
            if exit_signal.reason == ExitReason.TARGET1_HIT:
                pos.t1_hit = True
                pos.trailing_phase = "TRAIL_T1"
            elif exit_signal.reason == ExitReason.TARGET2_HIT:
                pos.t2_hit = True
                pos.trailing_phase = "TRAIL_T2"
            elif exit_signal.reason == ExitReason.TARGET3_HIT:
                pos.t3_hit = True
                pos.trailing_phase = "TIGHT"

        elif new_tsl and new_tsl > 0 and new_tsl > (pos.smart_levels.option_sl if pos.smart_levels else 0):
            # Just tighten trailing SL (no qty exit) — update on broker
            if pos.sl_order_id:
                kite.cancel_order(pos.sl_order_id)
            new_sl_result = kite.place_sl_order(
                tradingsymbol=pos.symbol,
                exchange="NFO",
                transaction_type="SELL",
                quantity=pos.remaining_qty,
                trigger_price=round(new_tsl, 2),
                product="MIS",
                tag=f"AutoTSL_{pos.trailing_phase}",
            )
            if new_sl_result.get("success"):
                pos.sl_order_id = new_sl_result.get("order_id", "")
                logger.info(f"Trailing SL tightened to {new_tsl:.2f} for {trade_id} (phase={pos.trailing_phase})")

        self.save_state()  # Persist after any exit/TSL change

    # ------------------------------------------------------------------
    # Square-off all
    # ------------------------------------------------------------------
    def square_off_all(self, kite: KiteClient, reason: str = "SQUARED_OFF"):
        """Exit all open positions at market price."""
        for trade_id, pos in list(self.positions.items()):
            if pos.status != "CLOSED" and pos.remaining_qty > 0:
                self.full_exit(kite, trade_id, reason=reason)

        count = len([p for p in self.positions.values() if p.status == "CLOSED"])
        telegram.alert_square_off(count, self.risk_mgr.daily_pnl)
        logger.info(f"SQUARE OFF ALL: {count} positions closed, reason={reason}")
        self.save_state()  # Persist after square-off

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_active_positions(self) -> List[Dict]:
        """Return active positions for dashboard."""
        result = []
        for pos in self.positions.values():
            if pos.status == "CLOSED":
                continue
            lvls = pos.smart_levels
            result.append({
                "trade_id": pos.trade_id,
                "symbol": pos.symbol,
                "option_type": pos.option_type,
                "entry": pos.entry_price,
                "qty": pos.quantity,
                "remaining_qty": pos.remaining_qty,
                "sl": lvls.option_sl if lvls else 0,
                "sl_type": lvls.sl_type if lvls else "",
                "t1": lvls.option_t1 if lvls else 0,
                "t1_type": lvls.t1_type if lvls else "",
                "t2": lvls.option_t2 if lvls else 0,
                "t3": lvls.option_t3 if lvls else 0,
                "t1_hit": pos.t1_hit,
                "t2_hit": pos.t2_hit,
                "t3_hit": pos.t3_hit,
                "score": pos.score,
                "delta": pos.delta,
                "status": pos.status,
                "risk_reward": lvls.risk_reward if lvls else 0,
                "sl_order_id": pos.sl_order_id,
                "trailing_sl": pos.trailing_sl,
                "highest_price": pos.highest_price,
                "trailing_phase": pos.trailing_phase,
                "last_tsl_update": pos.last_tsl_update,
            })
        return result

    def get_order_log(self) -> List[Dict]:
        """Return all orders for audit trail."""
        return [
            {
                "order_id": o.order_id,
                "trade_id": o.trade_id,
                "symbol": o.symbol,
                "type": o.transaction_type,
                "qty": o.quantity,
                "order_type": o.order_type,
                "trigger": o.trigger_price,
                "purpose": o.purpose,
                "status": o.status,
                "placed_at": o.placed_at,
            }
            for o in self.order_log
        ]

    # ------------------------------------------------------------------
    # Position persistence (survive server restarts)
    # ------------------------------------------------------------------
    def save_state(self):
        """Save open positions to disk as JSON for crash recovery."""
        try:
            state = {
                "saved_at": datetime.now().isoformat(),
                "daily_order_count": self._daily_order_count,
                "daily_order_date": self._daily_order_date,
                "positions": {},
            }
            for tid, pos in self.positions.items():
                if pos.status == "CLOSED":
                    continue
                state["positions"][tid] = {
                    "trade_id": pos.trade_id,
                    "symbol": pos.symbol,
                    "option_type": pos.option_type,
                    "entry_price": pos.entry_price,
                    "entry_order_id": pos.entry_order_id,
                    "sl_order_id": pos.sl_order_id,
                    "quantity": pos.quantity,
                    "remaining_qty": pos.remaining_qty,
                    "t1_hit": pos.t1_hit,
                    "t2_hit": pos.t2_hit,
                    "t3_hit": pos.t3_hit,
                    "status": pos.status,
                    "delta": pos.delta,
                    "score": pos.score,
                    "created_at": pos.created_at,
                    "trailing_sl": pos.trailing_sl,
                    "highest_price": pos.highest_price,
                    "trailing_phase": pos.trailing_phase,
                    "last_tsl_update": pos.last_tsl_update,
                    # SmartLevels as dict (if present)
                    "smart_levels": asdict(pos.smart_levels) if pos.smart_levels else None,
                }
            _PERSISTENCE_FILE.write_text(json.dumps(state, indent=2))
            logger.debug(f"State saved: {len(state['positions'])} open positions")
        except Exception as e:
            logger.error(f"Failed to save position state: {e}")

    def load_state(self):
        """Restore open positions from disk after a restart."""
        if not _PERSISTENCE_FILE.exists():
            return
        try:
            raw = json.loads(_PERSISTENCE_FILE.read_text())
            # Only restore if saved today (don't carry over old day positions)
            saved_date = raw.get("saved_at", "")[:10]
            today_str = datetime.now().strftime("%Y-%m-%d")
            if saved_date != today_str:
                logger.info(f"Stale position state from {saved_date}, ignoring.")
                _PERSISTENCE_FILE.unlink(missing_ok=True)
                return

            self._daily_order_count = raw.get("daily_order_count", 0)
            self._daily_order_date = raw.get("daily_order_date", "")

            for tid, pdata in raw.get("positions", {}).items():
                sl_data = pdata.get("smart_levels")
                smart = SmartLevels(**sl_data) if sl_data else None
                pos = ManagedPosition(
                    trade_id=pdata["trade_id"],
                    symbol=pdata["symbol"],
                    option_type=pdata["option_type"],
                    entry_price=pdata["entry_price"],
                    entry_order_id=pdata.get("entry_order_id", ""),
                    sl_order_id=pdata.get("sl_order_id", ""),
                    quantity=pdata["quantity"],
                    remaining_qty=pdata["remaining_qty"],
                    smart_levels=smart,
                    t1_hit=pdata.get("t1_hit", False),
                    t2_hit=pdata.get("t2_hit", False),
                    t3_hit=pdata.get("t3_hit", False),
                    status=pdata.get("status", "ACTIVE"),
                    delta=pdata.get("delta", 0.4),
                    score=pdata.get("score", 0.0),
                    created_at=pdata.get("created_at", ""),
                    trailing_sl=pdata.get("trailing_sl", 0.0),
                    highest_price=pdata.get("highest_price", 0.0),
                    trailing_phase=pdata.get("trailing_phase", "INITIAL"),
                    last_tsl_update=pdata.get("last_tsl_update", ""),
                )
                self.positions[tid] = pos
            logger.info(f"Restored {len(self.positions)} positions from disk ({saved_date})")
        except Exception as e:
            logger.error(f"Failed to load position state: {e}")
