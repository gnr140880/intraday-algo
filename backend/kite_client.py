"""
Zerodha KiteConnect client wrapper with token management.
Handles authentication, order placement, and market data.
"""
import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from kiteconnect import KiteConnect, KiteTicker
from config import settings

logger = logging.getLogger(__name__)


class KiteClient:
    _instance: Optional["KiteClient"] = None

    def __init__(self):
        self.kite = KiteConnect(api_key=settings.kite_api_key)
        self.ticker: Optional[KiteTicker] = None
        self.access_token: Optional[str] = settings.kite_access_token or None
        self._connected = False
        self._subscribed_tokens: set = set()
        self._tick_callbacks = []

        if self.access_token:
            self.kite.set_access_token(self.access_token)
            self._connected = True
            logger.info("KiteConnect initialized with stored access token")

    @classmethod
    def get_instance(cls) -> "KiteClient":
        if cls._instance is None:
            cls._instance = KiteClient()
        return cls._instance

    def get_login_url(self) -> str:
        return self.kite.login_url()

    def generate_session(self, request_token: str) -> Dict:
        try:
            data = self.kite.generate_session(
                request_token, api_secret=settings.kite_api_secret
            )
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self._connected = True

            # Persist token to .env
            self._update_env_token(self.access_token)
            logger.info("Session generated successfully")
            return {"success": True, "access_token": self.access_token}
        except Exception as e:
            logger.error(f"Session generation failed: {e}")
            return {"success": False, "error": str(e)}

    def _update_env_token(self, token: str):
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        lines = []
        token_found = False
        if os.path.exists(env_path):
            with open(env_path) as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                if line.startswith("KITE_ACCESS_TOKEN="):
                    new_lines.append(f"KITE_ACCESS_TOKEN={token}\n")
                    token_found = True
                else:
                    new_lines.append(line)
            if not token_found:
                new_lines.append(f"KITE_ACCESS_TOKEN={token}\n")
            with open(env_path, "w") as f:
                f.writelines(new_lines)

    @property
    def is_connected(self) -> bool:
        return self._connected and bool(self.access_token)

    def get_profile(self) -> Dict:
        return self.kite.profile()

    def get_holdings(self) -> List[Dict]:
        return self.kite.holdings()

    def get_positions(self) -> Dict:
        return self.kite.positions()

    def get_orders(self) -> List[Dict]:
        return self.kite.orders()

    def get_trades(self) -> List[Dict]:
        return self.kite.trades()

    def get_funds(self) -> Dict:
        return self.kite.margins()

    def get_quote(self, instruments: List[str]) -> Dict:
        return self.kite.quote(instruments)

    def get_ohlc(self, instruments: List[str]) -> Dict:
        return self.kite.ohlc(instruments)

    def get_ltp(self, instruments: List[str]) -> Dict:
        return self.kite.ltp(instruments)

    def get_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "minute",
        continuous: bool = False,
        oi: bool = False,
    ) -> List[Dict]:
        return self.kite.historical_data(
            instrument_token, from_date, to_date, interval, continuous, oi
        )

    def get_instruments(self, exchange: str = None) -> List[Dict]:
        if exchange:
            return self.kite.instruments(exchange)
        return self.kite.instruments()

    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        product: str = "MIS",
        price: float = None,
        trigger_price: float = None,
        stoploss: float = None,
        squareoff: float = None,
        trailing_stoploss: float = None,
        validity: str = "DAY",
        tag: str = "AlgoTrade",
    ) -> Dict:
        try:
            if settings.trading_mode == "paper":
                return {
                    "success": True,
                    "order_id": f"PAPER_{datetime.now().strftime('%H%M%S%f')}",
                    "paper_trade": True,
                    "details": {
                        "symbol": tradingsymbol,
                        "type": transaction_type,
                        "qty": quantity,
                        "order_type": order_type,
                        "price": price,
                    },
                }

            params = {
                "tradingsymbol": tradingsymbol,
                "exchange": exchange,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "product": product,
                "validity": validity,
                "tag": tag,
            }
            if price:
                params["price"] = price
            if trigger_price:
                params["trigger_price"] = trigger_price

            # Bracket order support
            if stoploss and squareoff:
                params["variety"] = KiteConnect.VARIETY_BO
                params["stoploss"] = stoploss
                params["squareoff"] = squareoff
                if trailing_stoploss:
                    params["trailing_stoploss"] = trailing_stoploss
                order_id = self.kite.place_order(**params)
            else:
                params["variety"] = KiteConnect.VARIETY_REGULAR
                order_id = self.kite.place_order(**params)

            logger.info(f"Order placed: {order_id} for {tradingsymbol}")
            return {"success": True, "order_id": order_id}

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return {"success": False, "error": str(e)}

    def modify_order(self, order_id: str, **kwargs) -> Dict:
        try:
            self.kite.modify_order(variety=KiteConnect.VARIETY_REGULAR, order_id=order_id, **kwargs)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cancel_order(self, order_id: str, variety: str = "regular") -> Dict:
        try:
            self.kite.cancel_order(variety=variety, order_id=order_id)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_ticker(self, instrument_tokens: List[int], on_tick=None, on_connect=None):
        if not self.access_token:
            logger.error("No access token available for ticker")
            return

        self.ticker = KiteTicker(
            api_key=settings.kite_api_key,
            access_token=self.access_token,
        )

        def _on_connect(ws, response):
            ws.subscribe(instrument_tokens)
            ws.set_mode(ws.MODE_FULL, instrument_tokens)
            self._subscribed_tokens.update(instrument_tokens)
            logger.info(f"Ticker connected, subscribed to {len(instrument_tokens)} tokens")
            if on_connect:
                on_connect(ws, response)

        def _on_ticks(ws, ticks):
            if on_tick:
                on_tick(ticks)

        def _on_error(ws, code, reason):
            logger.error(f"Ticker error: {code} - {reason}")

        def _on_close(ws, code, reason):
            logger.warning(f"Ticker closed: {code} - {reason}")

        self.ticker.on_ticks = _on_ticks
        self.ticker.on_connect = _on_connect
        self.ticker.on_error = _on_error
        self.ticker.on_close = _on_close
        self.ticker.connect(threaded=True)

    def stop_ticker(self):
        if self.ticker:
            self.ticker.stop()

    def search_instruments(self, exchange: str, query: str) -> List[Dict]:
        try:
            return self.kite.instruments(exchange)
        except Exception as e:
            logger.error(f"Instrument search failed: {e}")
            return []


kite_client = KiteClient.get_instance()
