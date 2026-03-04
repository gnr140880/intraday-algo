"""
Telegram Alert Service

Sends formatted alerts to a Telegram chat/group via Bot API.

Alert types:
  - ORB captured
  - Signal generated (BUY/SELL)
  - Trade executed (entry)
  - SL hit / Target hit / Trailing SL hit
  - Auto square-off (3:15 PM)
  - Daily loss limit hit
  - Daily P&L summary
  - Engine status changes
  - News sentiment shift

Setup:
  1. Create bot via @BotFather → get TELEGRAM_BOT_TOKEN
  2. Add bot to your group/channel or DM it
  3. Get chat_id via https://api.telegram.org/bot<TOKEN>/getUpdates
  4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
"""
import logging
import httpx
from typing import Optional, Dict, List
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    """
    Fire-and-forget Telegram alerts with rate limiting.
    Uses synchronous httpx for simplicity (called from engine thread).
    """

    def __init__(self):
        self.token: str = settings.telegram_bot_token
        self.chat_id: str = settings.telegram_chat_id
        self.enabled: bool = bool(self.token and self.chat_id)
        self._last_sent: Dict[str, float] = {}  # rate limit tracker
        self._min_interval = 5  # min seconds between same alert type

        if self.enabled:
            logger.info("Telegram alerts enabled")
        else:
            logger.info("Telegram alerts disabled (set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in .env)")

    def reload_config(self):
        """Reload token/chat_id from settings (if updated at runtime)."""
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)

    def _rate_ok(self, key: str) -> bool:
        now = datetime.now().timestamp()
        last = self._last_sent.get(key, 0)
        if now - last < self._min_interval:
            return False
        self._last_sent[key] = now
        return True

    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram. Returns True on success."""
        if not self.enabled:
            return False
        try:
            url = TELEGRAM_API.format(token=self.token)
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            resp = httpx.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            else:
                logger.warning(f"Telegram API error {resp.status_code}: {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Alert: ORB Captured
    # ------------------------------------------------------------------
    def alert_orb_captured(self, orb_high: float, orb_low: float, spot: float):
        if not self._rate_ok("orb"):
            return
        orb_range = orb_high - orb_low
        text = (
            "📊 <b>ORB CAPTURED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"High: <b>{orb_high:.2f}</b>\n"
            f"Low:  <b>{orb_low:.2f}</b>\n"
            f"Range: {orb_range:.2f} pts\n"
            f"Spot:  {spot:.2f}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: Signal Generated
    # ------------------------------------------------------------------
    def alert_signal(self, signal_type: str, entry: float, sl: float,
                     target: float, confidence: float, conditions: List[str],
                     reasoning: str):
        if not self._rate_ok(f"signal_{signal_type}"):
            return
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        cond_text = "\n".join(f"  ✓ {c}" for c in conditions[:5])
        rr = abs(target - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        text = (
            f"{emoji} <b>SIGNAL: {signal_type}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Entry:  <b>₹{entry:.2f}</b>\n"
            f"SL:     ₹{sl:.2f}\n"
            f"Target: ₹{target:.2f}\n"
            f"R:R:    {rr:.1f}\n"
            f"Confidence: {confidence:.0f}%\n\n"
            f"<b>Conditions:</b>\n{cond_text}\n\n"
            f"💡 {reasoning}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: Trade Executed
    # ------------------------------------------------------------------
    def alert_trade_entry(self, symbol: str, option_type: str, entry_price: float,
                          qty: int, sl: float, target: float, score: float,
                          trade_id: str):
        if not self._rate_ok(f"entry_{trade_id}"):
            return
        emoji = "📈" if option_type == "CE" else "📉"
        text = (
            f"{emoji} <b>TRADE ENTRY</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Type:   {option_type}\n"
            f"Entry:  <b>₹{entry_price:.2f}</b>\n"
            f"Qty:    {qty}\n"
            f"SL:     ₹{sl:.2f}\n"
            f"Target: ₹{target:.2f}\n"
            f"Score:  {score:.1f}/100\n"
            f"ID:     <code>{trade_id[-8:]}</code>\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: Trade Exit (SL / Target / Square-off)
    # ------------------------------------------------------------------
    def alert_trade_exit(self, symbol: str, option_type: str, entry_price: float,
                         exit_price: float, qty: int, pnl: float,
                         reason: str, trade_id: str):
        if not self._rate_ok(f"exit_{trade_id}"):
            return
        if reason == "SL_HIT":
            emoji, label = "🛑", "STOP LOSS HIT"
        elif reason == "TARGET_HIT":
            emoji, label = "🎯", "TARGET HIT"
        elif reason in ("SQUARED_OFF", "MANUAL_SQUAREOFF"):
            emoji, label = "⏹️", "SQUARED OFF"
        else:
            emoji, label = "📤", f"EXIT ({reason})"

        pnl_emoji = "✅" if pnl >= 0 else "❌"
        text = (
            f"{emoji} <b>{label}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Symbol: <b>{symbol}</b> ({option_type})\n"
            f"Entry:  ₹{entry_price:.2f}\n"
            f"Exit:   ₹{exit_price:.2f}\n"
            f"Qty:    {qty}\n"
            f"{pnl_emoji} P&L:  <b>₹{pnl:+.2f}</b>\n"
            f"ID:     <code>{trade_id[-8:]}</code>\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: Daily Loss Limit Hit
    # ------------------------------------------------------------------
    def alert_loss_limit(self, daily_pnl: float, limit: float, capital: float):
        if not self._rate_ok("loss_limit"):
            return
        pct = daily_pnl / capital * 100 if capital > 0 else 0
        text = (
            "🚨🚨 <b>DAILY LOSS LIMIT HIT</b> 🚨🚨\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Daily P&L: <b>₹{daily_pnl:+.2f}</b> ({pct:+.2f}%)\n"
            f"Limit:     -₹{limit:.2f}\n"
            f"Capital:   ₹{capital:,.0f}\n\n"
            "⛔ <b>Trading halted for the day.</b>\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: Auto Square-off
    # ------------------------------------------------------------------
    def alert_square_off(self, positions_closed: int, total_pnl: float):
        if not self._rate_ok("square_off"):
            return
        text = (
            "⏹️ <b>AUTO SQUARE-OFF (3:15 PM)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Positions closed: {positions_closed}\n"
            f"Day P&L: <b>₹{total_pnl:+.2f}</b>\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: Daily P&L Summary (EOD)
    # ------------------------------------------------------------------
    def alert_daily_summary(self, risk_status: Dict, trades: List[Dict]):
        if not self._rate_ok("daily_summary"):
            return
        total_trades = len(trades)
        winners = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losers = sum(1 for t in trades if t.get("pnl", 0) < 0)
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        trade_list = ""
        for t in trades[:10]:
            pnl_e = "✅" if t.get("pnl", 0) >= 0 else "❌"
            trade_list += (
                f"  {pnl_e} {t.get('symbol', '?')} "
                f"₹{t.get('pnl', 0):+.2f} ({t.get('reason', '')})\n"
            )

        text = (
            "📋 <b>DAILY SUMMARY</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Date: {datetime.now().strftime('%d %b %Y')}\n\n"
            f"💰 Net P&L: <b>₹{total_pnl:+.2f}</b>\n"
            f"📊 Realised: ₹{risk_status.get('realised_pnl', 0):+.2f}\n"
            f"📈 Trades: {total_trades} (W:{winners} L:{losers})\n"
            f"📊 Win Rate: {win_rate:.0f}%\n\n"
            f"<b>Trades:</b>\n{trade_list}\n"
            f"Capital: ₹{risk_status.get('capital', 0):,.0f}\n"
            f"⏰ {datetime.now().strftime('%H:%M')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: Engine status change
    # ------------------------------------------------------------------
    def alert_engine_status(self, status: str, detail: str = ""):
        if not self._rate_ok(f"status_{status}"):
            return
        emoji_map = {
            "DISCONNECTED": "🔌",
            "WAITING_ORB": "⏳",
            "MONITORING": "👁️",
            "SIGNAL_ACTIVE": "⚡",
            "HALTED_LOSS_LIMIT": "🚨",
            "MARKET_CLOSED": "🏁",
            "SQUARED_OFF": "⏹️",
        }
        emoji = emoji_map.get(status, "ℹ️")
        text = f"{emoji} Engine: <b>{status}</b>"
        if detail:
            text += f"\n{detail}"
        text += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: Scored candidates summary
    # ------------------------------------------------------------------
    def alert_top_candidates(self, candidates: List[Dict]):
        if not self._rate_ok("candidates"):
            return
        if not candidates:
            return
        lines = []
        for i, c in enumerate(candidates[:5], 1):
            lines.append(
                f"  {i}. {c.get('symbol', '?')} Δ{c.get('delta', 0):.2f} "
                f"Score:<b>{c.get('score', 0):.1f}</b> ₹{c.get('ltp', 0):.2f}"
            )
        text = (
            "🏆 <b>TOP CANDIDATES</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(lines)
            + f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Alert: News sentiment shift
    # ------------------------------------------------------------------
    def alert_sentiment(self, mood: str, avg_sentiment: float,
                        bullish: int, bearish: int, neutral: int):
        if not self._rate_ok("sentiment"):
            return
        emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(mood, "⚪")
        text = (
            f"{emoji} <b>Market Mood: {mood}</b>\n"
            f"Avg sentiment: {avg_sentiment:+.4f}\n"
            f"🟢 Bullish: {bullish} | 🔴 Bearish: {bearish} | ⚪ Neutral: {neutral}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Generic alert
    # ------------------------------------------------------------------
    def send_custom(self, message: str):
        """Send a custom text message."""
        self._send(message)

    # ------------------------------------------------------------------
    # Test connectivity
    # ------------------------------------------------------------------
    def test(self) -> Dict:
        """Send a test message. Returns success/error."""
        if not self.enabled:
            return {"success": False, "error": "Telegram not configured. Set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in .env"}
        ok = self._send(
            "✅ <b>AlgoTest Telegram Alert Test</b>\n"
            f"Bot connected successfully!\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return {"success": ok}


# Singleton
telegram = TelegramAlerter()
