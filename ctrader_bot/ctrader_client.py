"""cTrader Open API client — JSON over WebSocket, no SDK.

Presents the exact same duck-typed interface as `oanda_client.OANDAClient`
(same method names, same dataclasses, same OANDA-shaped dicts) so the
strategy layer and main loop port over unchanged.

Transport choice: the official `ctrader-open-api` SDK is dormant (last
release mid-2024) and hard-pins Twisted==24.3.0 / protobuf==3.20.1, so this
client speaks the documented JSON protocol on port 5036 instead — the JSON
messages mirror the protobuf messages by name (help.ctrader.com/open-api).
All payload types, field names and scaling factors below were taken from
spotware/openapi-proto-messages (fetched 2026-07-05), not from memory.

Architecture: a background thread runs an asyncio loop holding one
persistent WebSocket. The public methods are synchronous facades that post
coroutines into that loop — main.py's polling loop stays unchanged. The
loop maintains heartbeats (every 9s; server requires <=10s), auto-reconnect
with re-auth/re-subscribe, an OAuth token refresh cycle persisted to disk,
a live spot-price cache, and an execution-event router for order waiters.
"""

import asyncio
import json
import logging
import ssl
import threading
import uuid
from concurrent.futures import Future as ThreadFuture
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from config import INSTRUMENTS, TradingConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Protocol constants (spotware/openapi-proto-messages, 2026-07-05)
# ─────────────────────────────────────────────────────────────

HEARTBEAT_EVENT = 51
ERROR_RES = 50

PT = {
    "APPLICATION_AUTH_REQ": 2100,
    "APPLICATION_AUTH_RES": 2101,
    "ACCOUNT_AUTH_REQ": 2102,
    "ACCOUNT_AUTH_RES": 2103,
    "NEW_ORDER_REQ": 2106,
    "CANCEL_ORDER_REQ": 2108,
    "CLOSE_POSITION_REQ": 2111,
    "SYMBOLS_LIST_REQ": 2114,
    "SYMBOLS_LIST_RES": 2115,
    "SYMBOL_BY_ID_REQ": 2116,
    "SYMBOL_BY_ID_RES": 2117,
    "TRADER_REQ": 2121,
    "TRADER_RES": 2122,
    "RECONCILE_REQ": 2124,
    "RECONCILE_RES": 2125,
    "EXECUTION_EVENT": 2126,
    "SUBSCRIBE_SPOTS_REQ": 2127,
    "SPOT_EVENT": 2131,
    "ORDER_ERROR_EVENT": 2132,
    "GET_TRENDBARS_REQ": 2137,
    "GET_TRENDBARS_RES": 2138,
    "OA_ERROR_RES": 2142,
    "ACCOUNTS_TOKEN_INVALIDATED_EVENT": 2147,
    "REFRESH_TOKEN_REQ": 2173,
    "REFRESH_TOKEN_RES": 2174,
    "DEAL_LIST_BY_POSITION_ID_REQ": 2179,
    "DEAL_LIST_BY_POSITION_ID_RES": 2180,
    "ORDER_DETAILS_REQ": 2181,
    "ORDER_DETAILS_RES": 2182,
}

PRICE_SCALE = 100_000  # spot/trendbar prices are in 1/100000 of a unit
VOLUME_SCALE = 100     # order volumes are centi-units (1000 == 10.00 units)

TRENDBAR_PERIOD = {"M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M10": 6,
                   "M15": 7, "M30": 8, "H1": 9, "H4": 10, "H12": 11, "D1": 12}
PERIOD_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}

ORDER_TYPE_LIMIT = 2
TRADE_SIDE = {"BUY": 1, "SELL": 2}

# ProtoOAExecutionType
EXEC_ORDER_ACCEPTED = 2
EXEC_ORDER_FILLED = 3
EXEC_ORDER_CANCELLED = 5
EXEC_ORDER_EXPIRED = 6
EXEC_ORDER_REJECTED = 7
EXEC_NAMES = {  # JSON proxy may send enum names instead of numbers
    "ORDER_ACCEPTED": 2, "ORDER_FILLED": 3, "ORDER_REPLACED": 4,
    "ORDER_CANCELLED": 5, "ORDER_EXPIRED": 6, "ORDER_REJECTED": 7,
    "ORDER_CANCEL_REJECTED": 8, "SWAP": 9, "DEPOSIT_WITHDRAW": 10,
    "ORDER_PARTIAL_FILL": 11, "BONUS_DEPOSIT_WITHDRAW": 12,
}
# ProtoOAOrderStatus -> OANDA-style state strings used by main.py
ORDER_STATUS_TO_STATE = {
    1: "PENDING",    # ORDER_STATUS_ACCEPTED (working)
    2: "FILLED",
    3: "CANCELLED",  # REJECTED
    4: "CANCELLED",  # EXPIRED
    5: "CANCELLED",
}
ORDER_STATUS_NAMES = {
    "ORDER_STATUS_ACCEPTED": 1, "ORDER_STATUS_FILLED": 2,
    "ORDER_STATUS_REJECTED": 3, "ORDER_STATUS_EXPIRED": 4,
    "ORDER_STATUS_CANCELLED": 5,
}
TRADE_SIDE_NAMES = {"BUY": 1, "SELL": 2}


def _enum(value, names: dict) -> Optional[int]:
    """Normalize a JSON enum that may arrive as a number or a name string."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return names.get(str(value))


# ─────────────────────────────────────────────────────────────
#  Data structures (identical shapes to oanda_client)
# ─────────────────────────────────────────────────────────────


@dataclass
class Candle:
    """A single price candle."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool


@dataclass
class Price:
    """Current bid/ask price."""
    instrument: str
    bid: float
    ask: float
    time: datetime

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class Position:
    """Open position summary."""
    instrument: str
    units: int          # signed: positive = long, negative = short
    avg_price: float
    unrealized_pnl: float = 0.0


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    fill_price: Optional[float] = None
    units: Optional[int] = None
    error: Optional[str] = None


@dataclass
class SymbolInfo:
    """Broker symbol metadata needed for rounding and sizing."""
    symbol_id: int
    name: str
    digits: int
    min_volume: int    # centi-units
    max_volume: int
    step_volume: int


class CTraderError(Exception):
    pass


# ─────────────────────────────────────────────────────────────
#  Connection (background asyncio thread)
# ─────────────────────────────────────────────────────────────


class CTraderConnection:
    """Owns the WebSocket, auth, heartbeats, reconnects, and event routing.

    Runs an asyncio loop in a daemon thread. All public access goes through
    `call(coro)` which bridges sync callers onto the loop.
    """

    HEARTBEAT_SECS = 9
    REQUEST_TIMEOUT = 15
    RECONNECT_BACKOFF = (1, 2, 5, 10, 30)

    def __init__(self):
        env = TradingConfig.CTRADER_ENVIRONMENT
        self.url = f"wss://{'live' if env == 'live' else 'demo'}.ctraderapi.com:5036"
        self.account_id = int(TradingConfig.CTRADER_ACCOUNT_ID)
        self._tokens = self._load_tokens()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="ctrader-io", daemon=True)
        self._ws = None
        self._pending: dict[str, asyncio.Future] = {}
        self._order_waiters: dict[str, asyncio.Future] = {}  # clientOrderId/orderId key
        self.ready = threading.Event()
        self._stopping = False

        # caches maintained by the reader
        self.spot_prices: dict[int, dict] = {}       # symbolId -> {bid, ask, ts}
        self.symbols: dict[str, SymbolInfo] = {}     # internal key -> SymbolInfo
        self._symbol_by_id: dict[int, str] = {}      # symbolId -> internal key

    # ── token persistence ────────────────────────────────────

    def _token_path(self) -> Path:
        return Path(TradingConfig.CTRADER_TOKEN_FILE)

    def _load_tokens(self) -> dict:
        """Persisted tokens win over .env — refresh rotates them."""
        p = self._token_path()
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Token file unreadable ({e}); falling back to .env tokens")
        return {
            "accessToken": TradingConfig.CTRADER_ACCESS_TOKEN,
            "refreshToken": TradingConfig.CTRADER_REFRESH_TOKEN,
        }

    def _save_tokens(self) -> None:
        p = self._token_path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._tokens))
        tmp.replace(p)  # atomic — a crash mid-write can't lose the refresh token

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._supervisor(), self._loop)

    def call(self, coro, timeout: float = None):
        """Run a coroutine on the IO loop from sync code and wait for it."""
        fut: ThreadFuture = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout or self.REQUEST_TIMEOUT + 5)

    async def _supervisor(self) -> None:
        """Connect-authorize-read loop with backoff reconnect."""
        import websockets
        attempt = 0
        while not self._stopping:
            try:
                # certifi CA bundle: macOS python.org builds (and some minimal
                # Linux images) can't see system roots from Python.
                import certifi
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                async with websockets.connect(
                        self.url, ssl=ssl_ctx, ping_interval=None,
                        max_size=16 * 1024 * 1024) as ws:
                    self._ws = ws
                    logger.info(f"cTrader WebSocket connected: {self.url}")
                    # The read loop must be running BEFORE the first request:
                    # it is what resolves the response futures _request awaits.
                    reader = asyncio.ensure_future(self._read_loop())
                    heartbeat = None
                    try:
                        await self._authorize()
                        await self._resolve_symbols()
                        await self._subscribe_spots()
                        attempt = 0
                        self.ready.set()
                        heartbeat = asyncio.ensure_future(self._heartbeat_loop())
                        await reader
                    finally:
                        reader.cancel()
                        if heartbeat is not None:
                            heartbeat.cancel()
            except Exception as e:
                logger.error(f"cTrader connection error: {e!r}")
            self.ready.clear()
            self._ws = None
            self._fail_all_pending("connection lost")
            if self._stopping:
                return
            delay = self.RECONNECT_BACKOFF[min(attempt, len(self.RECONNECT_BACKOFF) - 1)]
            attempt += 1
            logger.info(f"Reconnecting in {delay}s ...")
            await asyncio.sleep(delay)

    def _fail_all_pending(self, reason: str) -> None:
        for fut in list(self._pending.values()) + list(self._order_waiters.values()):
            if not fut.done():
                fut.set_exception(CTraderError(reason))
        self._pending.clear()
        self._order_waiters.clear()

    # ── wire helpers ─────────────────────────────────────────

    async def _send(self, payload_type: int, payload: dict, msg_id: str = None) -> None:
        frame = {"clientMsgId": msg_id or str(uuid.uuid4()),
                 "payloadType": payload_type, "payload": payload}
        await self._ws.send(json.dumps(frame))

    async def _request(self, payload_type: int, payload: dict,
                       expect: int = None) -> dict:
        """Send a request and await the correlated response by clientMsgId."""
        msg_id = str(uuid.uuid4())
        fut = self._loop.create_future()
        self._pending[msg_id] = fut
        try:
            await self._send(payload_type, payload, msg_id)
            msg = await asyncio.wait_for(fut, timeout=self.REQUEST_TIMEOUT)
        finally:
            self._pending.pop(msg_id, None)
        pt = msg.get("payloadType")
        if pt in (ERROR_RES, PT["OA_ERROR_RES"]):
            p = msg.get("payload", {})
            raise CTraderError(f"{p.get('errorCode')}: {p.get('description')}")
        if expect is not None and pt != expect:
            raise CTraderError(f"Expected payloadType {expect}, got {pt}: {msg}")
        return msg.get("payload", {})

    async def _read_loop(self) -> None:
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON frame ignored: {raw[:120]!r}")
                continue
            msg_id = msg.get("clientMsgId")
            pt = msg.get("payloadType")

            if msg_id and msg_id in self._pending:
                fut = self._pending.pop(msg_id)
                if not fut.done():
                    fut.set_result(msg)
                # Execution events also flow to waiters below even when they
                # answered a tracked request — fall through only for events.
                if pt != PT["EXECUTION_EVENT"]:
                    continue

            if pt == PT["SPOT_EVENT"]:
                self._on_spot(msg.get("payload", {}))
            elif pt == PT["EXECUTION_EVENT"]:
                self._on_execution(msg.get("payload", {}))
            elif pt == PT["ORDER_ERROR_EVENT"]:
                self._on_order_error(msg.get("payload", {}))
            elif pt == PT["ACCOUNTS_TOKEN_INVALIDATED_EVENT"]:
                logger.warning("Access token invalidated — refreshing")
                asyncio.ensure_future(self._refresh_token_and_reauth())
            elif pt == HEARTBEAT_EVENT:
                pass
            # everything else: responses without waiters are ignored

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.HEARTBEAT_SECS)
            try:
                await self._send(HEARTBEAT_EVENT, {})
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                return

    # ── auth ─────────────────────────────────────────────────

    async def _authorize(self) -> None:
        await self._request(PT["APPLICATION_AUTH_REQ"], {
            "clientId": TradingConfig.CTRADER_CLIENT_ID,
            "clientSecret": TradingConfig.CTRADER_CLIENT_SECRET,
        }, expect=PT["APPLICATION_AUTH_RES"])
        try:
            await self._account_auth()
        except CTraderError as e:
            # expired access token — refresh once and retry
            logger.warning(f"Account auth failed ({e}); attempting token refresh")
            await self._refresh_token()
            await self._account_auth()
        logger.info(f"cTrader account {self.account_id} authorized")

    async def _account_auth(self) -> None:
        await self._request(PT["ACCOUNT_AUTH_REQ"], {
            "ctidTraderAccountId": self.account_id,
            "accessToken": self._tokens["accessToken"],
        }, expect=PT["ACCOUNT_AUTH_RES"])

    async def _refresh_token(self) -> None:
        payload = await self._request(PT["REFRESH_TOKEN_REQ"], {
            "refreshToken": self._tokens["refreshToken"],
        }, expect=PT["REFRESH_TOKEN_RES"])
        self._tokens = {
            "accessToken": payload["accessToken"],
            "refreshToken": payload["refreshToken"],
        }
        self._save_tokens()
        logger.info("cTrader tokens refreshed and persisted")

    async def _refresh_token_and_reauth(self) -> None:
        try:
            await self._refresh_token()
            await self._account_auth()
        except Exception as e:
            logger.error(f"Token refresh/re-auth failed: {e}")

    # ── symbols & spots ──────────────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        return name.upper().replace("/", "").replace(".", "").replace("_", "").replace(" ", "")

    async def _resolve_symbols(self) -> None:
        payload = await self._request(
            PT["SYMBOLS_LIST_REQ"], {"ctidTraderAccountId": self.account_id},
            expect=PT["SYMBOLS_LIST_RES"])
        listed = payload.get("symbol", [])
        by_norm: dict[str, list] = {}
        for s in listed:
            by_norm.setdefault(self._normalize(s.get("symbolName", "")), []).append(s)

        wanted_ids = []
        for internal, broker_name in TradingConfig.SYMBOL_MAP.items():
            if internal not in INSTRUMENTS:
                continue
            matches = by_norm.get(self._normalize(broker_name), [])
            if not matches:
                raise CTraderError(
                    f"Symbol {broker_name!r} (for {internal}) not found on this account. "
                    f"Check SYMBOL_MAP against the broker's symbol list."
                )
            sym = matches[0]
            self._symbol_by_id[int(sym["symbolId"])] = internal
            wanted_ids.append(int(sym["symbolId"]))
            self.symbols[internal] = SymbolInfo(
                symbol_id=int(sym["symbolId"]), name=sym.get("symbolName", broker_name),
                digits=5, min_volume=VOLUME_SCALE, max_volume=2**53, step_volume=VOLUME_SCALE)

        # fill in digits / volume constraints from full symbol details
        detail = await self._request(PT["SYMBOL_BY_ID_REQ"], {
            "ctidTraderAccountId": self.account_id, "symbolId": wanted_ids,
        }, expect=PT["SYMBOL_BY_ID_RES"])
        for full in detail.get("symbol", []):
            internal = self._symbol_by_id.get(int(full["symbolId"]))
            if not internal:
                continue
            info = self.symbols[internal]
            info.digits = int(full.get("digits", info.digits))
            info.min_volume = int(full.get("minVolume", info.min_volume))
            info.max_volume = int(full.get("maxVolume", info.max_volume))
            info.step_volume = int(full.get("stepVolume", info.step_volume))
        for k, v in self.symbols.items():
            logger.info(f"Symbol resolved: {k} -> {v.name} (id={v.symbol_id}, "
                        f"digits={v.digits}, minVol={v.min_volume}, step={v.step_volume})")

    async def _subscribe_spots(self) -> None:
        ids = [s.symbol_id for s in self.symbols.values()]
        if not ids:
            return
        await self._request(PT["SUBSCRIBE_SPOTS_REQ"], {
            "ctidTraderAccountId": self.account_id, "symbolId": ids,
        })

    def _on_spot(self, p: dict) -> None:
        sid = int(p.get("symbolId", 0))
        cache = self.spot_prices.setdefault(sid, {})
        if p.get("bid") is not None:
            cache["bid"] = int(p["bid"]) / PRICE_SCALE
        if p.get("ask") is not None:
            cache["ask"] = int(p["ask"]) / PRICE_SCALE
        cache["ts"] = datetime.now(timezone.utc)

    # ── execution events ─────────────────────────────────────

    def register_order_waiter(self, key: str) -> asyncio.Future:
        fut = self._loop.create_future()
        self._order_waiters[key] = fut
        return fut

    def _on_execution(self, p: dict) -> None:
        etype = _enum(p.get("executionType"), EXEC_NAMES)
        order = p.get("order", {}) or {}
        client_oid = order.get("clientOrderId")
        order_id = str(order.get("orderId", "")) or None
        position_id = order.get("positionId") or (p.get("position") or {}).get("positionId")
        position_id = str(position_id) if position_id else None
        logger.info(
            f"EXECUTION EVENT | type={p.get('executionType')} order={order_id} "
            f"clientOrderId={client_oid} position={position_id}"
        )
        # Route to a waiter by, in order: our clientOrderId (new orders),
        # the orderId (cancels), or the positionId (position closes — the
        # closing order has a fresh orderId, so only positionId correlates).
        for key in (client_oid, order_id, position_id):
            if key and key in self._order_waiters:
                fut = self._order_waiters.pop(key)
                if not fut.done():
                    fut.set_result(p)
                break

    def _on_order_error(self, p: dict) -> None:
        logger.error(f"ORDER ERROR EVENT | {p.get('errorCode')}: {p.get('description')}")
        oid = str(p.get("orderId", "")) or None
        for key in (oid,):
            if key and key in self._order_waiters:
                fut = self._order_waiters.pop(key)
                if not fut.done():
                    fut.set_exception(CTraderError(
                        f"{p.get('errorCode')}: {p.get('description')}"))


# ─────────────────────────────────────────────────────────────
#  Public client (sync facade, OANDAClient-compatible)
# ─────────────────────────────────────────────────────────────


class CTraderClient:
    """Drop-in replacement for OANDAClient backed by the cTrader Open API."""

    READY_TIMEOUT = 30

    def __init__(self, connection: CTraderConnection = None):
        # injectable connection for hermetic tests
        self.conn = connection or CTraderConnection()
        if connection is None:
            self.conn.start()

    # ── plumbing ─────────────────────────────────────────────

    def _ensure_ready(self) -> bool:
        if not self.conn.ready.wait(timeout=self.READY_TIMEOUT):
            logger.error("cTrader connection not ready")
            return False
        return True

    def _sym(self, instrument: str) -> SymbolInfo:
        info = self.conn.symbols.get(instrument)
        if not info:
            raise CTraderError(f"Unknown instrument {instrument}")
        return info

    # ── interface: connection / account ──────────────────────

    def test_connection(self) -> bool:
        """Verify auth + account access; log balance like the OANDA client."""
        if not self._ensure_ready():
            return False
        try:
            trader = self._get_trader()
            money_digits = int(trader.get("moneyDigits", 2))
            balance = int(trader["balance"]) / 10 ** money_digits
            logger.info(
                f"Connected to cTrader | Account: {self.conn.account_id} | "
                f"Balance: {balance:.4f} | Leverage: 1:{int(trader.get('leverageInCents', 0)) // 100}"
            )
            return True
        except Exception as e:
            logger.error(f"cTrader connection test failed: {e}")
            return False

    def _get_trader(self) -> dict:
        payload = self.conn.call(self.conn._request(
            PT["TRADER_REQ"], {"ctidTraderAccountId": self.conn.account_id},
            expect=PT["TRADER_RES"]))
        return payload.get("trader", {})

    def get_account_balance(self) -> float:
        try:
            trader = self._get_trader()
            return int(trader["balance"]) / 10 ** int(trader.get("moneyDigits", 2))
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    # ── interface: prices & candles ──────────────────────────

    def get_price(self, instrument: str) -> Optional[Price]:
        try:
            info = self._sym(instrument)
        except CTraderError:
            return None
        cache = self.conn.spot_prices.get(info.symbol_id, {})
        if "bid" not in cache or "ask" not in cache:
            return None
        return Price(instrument=instrument, bid=cache["bid"], ask=cache["ask"],
                     time=cache.get("ts", datetime.now(timezone.utc)))

    def get_candles(
        self,
        instrument: str,
        granularity: str = "M5",
        count: int = 10,
        include_incomplete: bool = True,
    ) -> list[Candle]:
        """Historical trendbars, decoded to Candle (times = bar open, UTC)."""
        try:
            info = self._sym(instrument)
            period_sec = PERIOD_SECONDS[granularity]
            now = datetime.now(timezone.utc)
            payload = self.conn.call(self.conn._request(PT["GET_TRENDBARS_REQ"], {
                "ctidTraderAccountId": self.conn.account_id,
                "period": TRENDBAR_PERIOD[granularity],
                "symbolId": info.symbol_id,
                "count": count,
                "toTimestamp": int(now.timestamp() * 1000),
                "fromTimestamp": int((now - timedelta(seconds=period_sec * (count + 10))).timestamp() * 1000),
            }, expect=PT["GET_TRENDBARS_RES"]))
        except Exception as e:
            logger.error(f"Failed to get candles for {instrument}: {e}")
            return []

        candles = []
        for tb in payload.get("trendbar", []):
            c = self._decode_trendbar(tb, period_sec)
            if c and (include_incomplete or c.complete):
                candles.append(c)
        candles.sort(key=lambda c: c.time)
        return candles[-count:]

    @staticmethod
    def _decode_trendbar(tb: dict, period_sec: int) -> Optional[Candle]:
        try:
            low = int(tb["low"])
            open_ = low + int(tb.get("deltaOpen", 0))
            high = low + int(tb.get("deltaHigh", 0))
            close = low + int(tb.get("deltaClose", 0))
            bar_open = datetime.fromtimestamp(
                int(tb["utcTimestampInMinutes"]) * 60, tz=timezone.utc)
            complete = datetime.now(timezone.utc) >= bar_open + timedelta(seconds=period_sec)
            return Candle(
                time=bar_open,
                open=open_ / PRICE_SCALE,
                high=high / PRICE_SCALE,
                low=low / PRICE_SCALE,
                close=close / PRICE_SCALE,
                volume=int(tb.get("volume", 0)),
                complete=complete,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Undecodable trendbar {tb}: {e}")
            return None

    def get_latest_complete_candle(
        self, instrument: str, granularity: str = "M5"
    ) -> Optional[Candle]:
        candles = self.get_candles(instrument, granularity, count=3,
                                   include_incomplete=False)
        return candles[-1] if candles else None

    # ── interface: positions & trades ────────────────────────

    def _reconcile(self) -> dict:
        return self.conn.call(self.conn._request(
            PT["RECONCILE_REQ"], {"ctidTraderAccountId": self.conn.account_id},
            expect=PT["RECONCILE_RES"]))

    def _positions_for(self, instrument: str = None) -> list[dict]:
        payload = self._reconcile()
        out = []
        for pos in payload.get("position", []):
            td = pos.get("tradeData", {})
            internal = self.conn._symbol_by_id.get(int(td.get("symbolId", 0)))
            if instrument and internal != instrument:
                continue
            pos["_instrument"] = internal
            out.append(pos)
        return out

    def get_position(self, instrument: str) -> Optional[Position]:
        try:
            positions = self._positions_for(instrument)
        except Exception as e:
            logger.error(f"Failed to get position for {instrument}: {e}")
            return None
        if not positions:
            return None
        pos = positions[0]
        td = pos.get("tradeData", {})
        units = int(td.get("volume", 0)) // VOLUME_SCALE
        if _enum(td.get("tradeSide"), TRADE_SIDE_NAMES) == TRADE_SIDE["SELL"]:
            units = -units
        return Position(instrument=instrument, units=units,
                        avg_price=float(pos.get("price", 0.0)))

    def get_open_trades(self, instrument: str = None) -> list[dict]:
        """OANDA-shaped trade dicts (id/price/openTime/stopLossOrder/...)."""
        try:
            positions = self._positions_for(instrument)
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}")
            return []
        trades = []
        for pos in positions:
            td = pos.get("tradeData", {})
            open_ts = int(td.get("openTimestamp", 0))
            trade = {
                "id": str(pos.get("positionId")),
                "instrument": pos.get("_instrument"),
                "price": str(pos.get("price", 0.0)),
                "openTime": datetime.fromtimestamp(
                    open_ts / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                if open_ts else None,
            }
            if pos.get("stopLoss") is not None:
                trade["stopLossOrder"] = {"price": str(pos["stopLoss"])}
            if pos.get("takeProfit") is not None:
                trade["takeProfitOrder"] = {"price": str(pos["takeProfit"])}
            trades.append(trade)
        return trades

    def get_order_details(self, order_id: str) -> Optional[dict]:
        """OANDA-shaped order state: PENDING / FILLED / CANCELLED."""
        try:
            payload = self.conn.call(self.conn._request(PT["ORDER_DETAILS_REQ"], {
                "ctidTraderAccountId": self.conn.account_id,
                "orderId": int(order_id),
            }, expect=PT["ORDER_DETAILS_RES"]))
        except Exception as e:
            logger.error(f"Failed to get order details {order_id}: {e}")
            return None
        order = payload.get("order", {})
        status = _enum(order.get("orderStatus"), ORDER_STATUS_NAMES)
        state = ORDER_STATUS_TO_STATE.get(status, "PENDING")
        details = {"id": str(order.get("orderId", order_id)), "state": state}
        if order.get("positionId"):
            details["tradeOpenedID"] = str(order["positionId"])
        if order.get("executionPrice") is not None:
            details["price"] = str(order["executionPrice"])
        return details

    def get_trade_details(self, trade_id: str) -> Optional[dict]:
        """Closed-trade info; provides averageClosePrice like OANDA."""
        try:
            now = datetime.now(timezone.utc)
            payload = self.conn.call(self.conn._request(
                PT["DEAL_LIST_BY_POSITION_ID_REQ"], {
                    "ctidTraderAccountId": self.conn.account_id,
                    "positionId": int(trade_id),
                    "fromTimestamp": int((now - timedelta(days=7)).timestamp() * 1000),
                    "toTimestamp": int(now.timestamp() * 1000),
                }, expect=PT["DEAL_LIST_BY_POSITION_ID_RES"]))
        except Exception as e:
            logger.error(f"Failed to get trade details {trade_id}: {e}")
            return None
        closing = [d for d in payload.get("deal", [])
                   if d.get("closePositionDetail") and d.get("executionPrice") is not None]
        details = {"id": str(trade_id)}
        if closing:
            total_vol = sum(int(d.get("filledVolume", 0)) for d in closing) or 1
            vwap = sum(float(d["executionPrice"]) * int(d.get("filledVolume", 0))
                       for d in closing) / total_vol
            details["averageClosePrice"] = str(round(vwap, 5))
        return details

    # ── interface: orders ────────────────────────────────────

    def place_limit_order(
        self,
        instrument: str,
        units: int,
        price: float,
        stop_loss: float,
        take_profit: float,
    ) -> OrderResult:
        """LIMIT order with attached absolute SL/TP (position-level on fill)."""
        try:
            info = self._sym(instrument)
        except CTraderError as e:
            return OrderResult(success=False, error=str(e))

        volume = abs(units) * VOLUME_SCALE
        volume -= volume % info.step_volume  # round down to step
        if volume < info.min_volume:
            return OrderResult(success=False, error=(
                f"Volume {abs(units)} units < broker minimum "
                f"{info.min_volume / VOLUME_SCALE} for {instrument}"))
        volume = min(volume, info.max_volume)

        digits = info.digits
        client_oid = f"msorb-{uuid.uuid4().hex[:16]}"
        payload = {
            "ctidTraderAccountId": self.conn.account_id,
            "symbolId": info.symbol_id,
            "orderType": ORDER_TYPE_LIMIT,
            "tradeSide": TRADE_SIDE["BUY"] if units > 0 else TRADE_SIDE["SELL"],
            "volume": volume,
            "limitPrice": round(price, digits),
            "stopLoss": round(stop_loss, digits),
            "takeProfit": round(take_profit, digits),
            "clientOrderId": client_oid,
            "label": "MS-ORB ConfigA",
        }

        async def _place():
            waiter = self.conn.register_order_waiter(client_oid)
            await self.conn._send(PT["NEW_ORDER_REQ"], payload)
            return await asyncio.wait_for(waiter, timeout=self.conn.REQUEST_TIMEOUT)

        try:
            event = self.conn.call(_place())
        except Exception as e:
            logger.error(f"place_limit_order failed for {instrument}: {e}")
            return OrderResult(success=False, error=str(e))

        etype = _enum(event.get("executionType"), EXEC_NAMES)
        order = event.get("order", {}) or {}
        order_id = str(order.get("orderId", "")) or None
        if etype == EXEC_ORDER_REJECTED:
            return OrderResult(success=False, order_id=order_id,
                               error=event.get("errorCode", "ORDER_REJECTED"))
        if etype == EXEC_ORDER_FILLED:
            deal = event.get("deal", {}) or {}
            position_id = (deal.get("positionId")
                           or order.get("positionId")
                           or (event.get("position") or {}).get("positionId"))
            fill = deal.get("executionPrice")
            return OrderResult(
                success=True, order_id=order_id,
                trade_id=str(position_id) if position_id else None,
                fill_price=float(fill) if fill is not None else None,
                units=units)
        # ORDER_ACCEPTED — pending limit order
        return OrderResult(success=True, order_id=order_id, units=units)

    def cancel_order(self, order_id: str) -> bool:
        async def _cancel():
            waiter = self.conn.register_order_waiter(str(order_id))
            await self.conn._send(PT["CANCEL_ORDER_REQ"], {
                "ctidTraderAccountId": self.conn.account_id,
                "orderId": int(order_id),
            })
            return await asyncio.wait_for(waiter, timeout=self.conn.REQUEST_TIMEOUT)

        try:
            event = self.conn.call(_cancel())
        except Exception as e:
            logger.error(f"cancel_order {order_id} failed: {e}")
            return False
        return _enum(event.get("executionType"), EXEC_NAMES) in (
            EXEC_ORDER_CANCELLED, EXEC_ORDER_EXPIRED)

    def close_position(self, instrument: str) -> OrderResult:
        """Close the full open position at market."""
        try:
            positions = self._positions_for(instrument)
        except Exception as e:
            return OrderResult(success=False, error=str(e))
        if not positions:
            return OrderResult(success=False, error=f"No open position for {instrument}")
        pos = positions[0]
        position_id = pos["positionId"]
        volume = int(pos.get("tradeData", {}).get("volume", 0))

        async def _close():
            waiter = self.conn.register_order_waiter(str(position_id))
            await self.conn._send(PT["CLOSE_POSITION_REQ"], {
                "ctidTraderAccountId": self.conn.account_id,
                "positionId": position_id,
                "volume": volume,
            })
            return await asyncio.wait_for(waiter, timeout=self.conn.REQUEST_TIMEOUT)

        # closing execution events reference the closing ORDER, whose
        # positionId routes back to our waiter via clientOrderId=None path —
        # register under positionId and match in _on_execution's fallback.
        try:
            event = self.conn.call(_close())
            deal = event.get("deal", {}) or {}
            fill = deal.get("executionPrice")
            return OrderResult(success=True, trade_id=str(position_id),
                               fill_price=float(fill) if fill is not None else None)
        except Exception as e:
            logger.error(f"close_position {instrument} failed: {e}")
            return OrderResult(success=False, error=str(e))
