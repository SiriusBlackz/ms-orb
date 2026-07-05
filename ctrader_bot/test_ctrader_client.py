"""Hermetic tests for the CTraderClient adapter — no network, no websockets.

A FakeConnection stands in for CTraderConnection: `_request` returns canned
payloads keyed by payloadType, `_send` synthesizes execution events for
order waiters. Covers the adapter logic the strategy depends on:

  1. Trendbar decoding (delta fields, 1/100000 scaling, open-time UTC,
     completeness) and include_incomplete filtering.
  2. Volume conversion (units -> centi-units), step rounding, min-volume
     rejection.
  3. Limit-order placement: accepted (pending) and instant-fill paths,
     rejection mapping.
  4. Order-details state mapping (numeric and string enum forms).
  5. Open-trades OANDA dict shaping (id/price/openTime/stopLossOrder).
  6. Position sign for SELL, trade-details VWAP close price.
"""

import asyncio
import os
import sys
import threading
import unittest
from datetime import datetime, timedelta, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

os.environ.update(
    CTRADER_CLIENT_ID="test-id", CTRADER_CLIENT_SECRET="test-secret",
    CTRADER_ACCESS_TOKEN="test-token", CTRADER_ACCOUNT_ID="12345",
    ENABLED_SESSIONS="TOKYO",
)

from ctrader_client import (  # noqa: E402
    PT, PRICE_SCALE, CTraderClient, SymbolInfo,
)

NOW = datetime.now(timezone.utc)


def bar(minutes_ago: int, low: float, d_open: float, d_high: float,
        d_close: float, volume: int = 100) -> dict:
    t = NOW - timedelta(minutes=minutes_ago)
    open_minute = int(t.timestamp() // 60)
    return {
        "low": int(low * PRICE_SCALE),
        "deltaOpen": int(d_open * PRICE_SCALE),
        "deltaHigh": int(d_high * PRICE_SCALE),
        "deltaClose": int(d_close * PRICE_SCALE),
        "utcTimestampInMinutes": open_minute,
        "volume": volume,
    }


class FakeConnection:
    """Stands in for CTraderConnection with canned responses."""

    REQUEST_TIMEOUT = 2

    def __init__(self):
        self.account_id = 12345
        self.ready = threading.Event()
        self.ready.set()
        self.symbols = {
            "XAU_USD": SymbolInfo(symbol_id=41, name="XAUUSD", digits=2,
                                  min_volume=100, max_volume=10_000_000, step_volume=100),
            "NAS100_USD": SymbolInfo(symbol_id=205, name="US100", digits=1,
                                     min_volume=10_000, max_volume=10_000_000, step_volume=10_000),
        }
        self._symbol_by_id = {41: "XAU_USD", 205: "NAS100_USD"}
        self.spot_prices = {41: {"bid": 4800.10, "ask": 4800.50, "ts": NOW}}
        self.responses: dict[int, dict] = {}   # payloadType -> canned payload
        self.next_execution_event: dict = {}   # synthesized on _send of orders
        self.sent: list[tuple[int, dict]] = []
        self._waiters: dict[str, asyncio.Future] = {}

    def call(self, coro, timeout=None):
        return asyncio.run(coro)

    async def _request(self, payload_type, payload, expect=None):
        self.sent.append((payload_type, payload))
        if payload_type not in self.responses:
            raise AssertionError(f"No canned response for payloadType {payload_type}")
        return self.responses[payload_type]

    def register_order_waiter(self, key):
        fut = asyncio.get_event_loop().create_future()
        self._waiters[key] = fut
        return fut

    async def _send(self, payload_type, payload, msg_id=None):
        self.sent.append((payload_type, payload))
        event = dict(self.next_execution_event)
        # resolve whichever waiter exists (tests register exactly one)
        for key, fut in list(self._waiters.items()):
            if not fut.done():
                fut.set_result(event)
            del self._waiters[key]


def make_client():
    conn = FakeConnection()
    return CTraderClient(connection=conn), conn


class TestTrendbarDecoding(unittest.TestCase):

    def test_decode_and_scaling(self):
        client, conn = make_client()
        conn.responses[PT["GET_TRENDBARS_REQ"]] = {
            "trendbar": [bar(15, 4800.00, 0.50, 2.00, 1.25)],
        }
        candles = client.get_candles("XAU_USD", "M5", count=5)
        self.assertEqual(len(candles), 1)
        c = candles[0]
        self.assertAlmostEqual(c.low, 4800.00)
        self.assertAlmostEqual(c.open, 4800.50)
        self.assertAlmostEqual(c.high, 4802.00)
        self.assertAlmostEqual(c.close, 4801.25)
        self.assertEqual(c.volume, 100)
        self.assertTrue(c.complete)   # 15-minute-old M5 bar is closed
        self.assertEqual(c.time.tzinfo, timezone.utc)

    def test_incomplete_bar_filtered(self):
        client, conn = make_client()
        conn.responses[PT["GET_TRENDBARS_REQ"]] = {
            "trendbar": [bar(10, 100.0, 0, 1, 0.5), bar(2, 101.0, 0, 1, 0.5)],
        }
        complete_only = client.get_candles("XAU_USD", "M5", count=5,
                                           include_incomplete=False)
        self.assertEqual(len(complete_only), 1)
        with_partial = client.get_candles("XAU_USD", "M5", count=5,
                                          include_incomplete=True)
        self.assertEqual(len(with_partial), 2)
        self.assertFalse(with_partial[-1].complete)

    def test_latest_complete_candle(self):
        client, conn = make_client()
        conn.responses[PT["GET_TRENDBARS_REQ"]] = {
            "trendbar": [bar(12, 100.0, 0, 1, 0.5), bar(7, 101.0, 0, 1, 0.5),
                         bar(2, 102.0, 0, 1, 0.5)],
        }
        c = client.get_latest_complete_candle("XAU_USD", "M5")
        self.assertAlmostEqual(c.low, 101.0)


class TestOrderPlacement(unittest.TestCase):

    def test_pending_limit_accepted(self):
        client, conn = make_client()
        conn.next_execution_event = {
            "executionType": "ORDER_ACCEPTED",
            "order": {"orderId": 777, "clientOrderId": "x"},
        }
        r = client.place_limit_order("XAU_USD", units=32, price=4804.223,
                                     stop_loss=4801.4159, take_profit=4818.42)
        self.assertTrue(r.success)
        self.assertEqual(r.order_id, "777")
        self.assertIsNone(r.trade_id)
        pt, payload = conn.sent[-1]
        self.assertEqual(pt, PT["NEW_ORDER_REQ"])
        self.assertEqual(payload["volume"], 3200)          # 32 units -> centi
        self.assertEqual(payload["limitPrice"], 4804.22)   # rounded to digits=2
        self.assertEqual(payload["stopLoss"], 4801.42)
        self.assertEqual(payload["tradeSide"], 1)          # BUY

    def test_instant_fill(self):
        client, conn = make_client()
        conn.next_execution_event = {
            "executionType": "ORDER_FILLED",
            "order": {"orderId": 778, "positionId": 9001},
            "deal": {"positionId": 9001, "executionPrice": 4804.20,
                     "filledVolume": 3200},
        }
        r = client.place_limit_order("XAU_USD", units=32, price=4804.22,
                                     stop_loss=4801.42, take_profit=4818.42)
        self.assertTrue(r.success)
        self.assertEqual(r.trade_id, "9001")
        self.assertAlmostEqual(r.fill_price, 4804.20)

    def test_rejection_maps_to_failure(self):
        client, conn = make_client()
        conn.next_execution_event = {
            "executionType": "ORDER_REJECTED",
            "order": {"orderId": 779},
            "errorCode": "NOT_ENOUGH_MONEY",
        }
        r = client.place_limit_order("XAU_USD", units=32, price=4804.22,
                                     stop_loss=4801.42, take_profit=4818.42)
        self.assertFalse(r.success)
        self.assertIn("NOT_ENOUGH_MONEY", r.error)

    def test_min_volume_rejected_locally(self):
        """NAS100 min volume is 100 units (10000 centi) in the fake broker;
        a 5-unit order must be refused BEFORE hitting the wire."""
        client, conn = make_client()
        r = client.place_limit_order("NAS100_USD", units=5, price=25000.0,
                                     stop_loss=24900.0, take_profit=25300.0)
        self.assertFalse(r.success)
        self.assertIn("minimum", r.error)
        self.assertEqual([s for s in conn.sent if s[0] == PT["NEW_ORDER_REQ"]], [])

    def test_step_rounding_and_sell_side(self):
        client, conn = make_client()
        conn.next_execution_event = {
            "executionType": "ORDER_ACCEPTED", "order": {"orderId": 780},
        }
        r = client.place_limit_order("NAS100_USD", units=-157, price=25000.0,
                                     stop_loss=25100.0, take_profit=24700.0)
        self.assertTrue(r.success)
        _, payload = conn.sent[-1]
        self.assertEqual(payload["volume"], 10_000)  # 157 units -> floor to step
        self.assertEqual(payload["tradeSide"], 2)    # SELL


class TestStateMapping(unittest.TestCase):

    def test_order_details_states(self):
        client, conn = make_client()
        for status, expected in ((1, "PENDING"), (2, "FILLED"), (5, "CANCELLED"),
                                 ("ORDER_STATUS_FILLED", "FILLED")):
            conn.responses[PT["ORDER_DETAILS_REQ"]] = {
                "order": {"orderId": 777, "orderStatus": status, "positionId": 9001},
            }
            details = client.get_order_details("777")
            self.assertEqual(details["state"], expected, f"status={status}")
            self.assertEqual(details["tradeOpenedID"], "9001")

    def test_open_trades_shape(self):
        client, conn = make_client()
        open_ts = int(NOW.timestamp() * 1000)
        conn.responses[PT["RECONCILE_REQ"]] = {
            "position": [{
                "positionId": 9001, "price": 4804.20,
                "stopLoss": 4801.42, "takeProfit": 4818.42,
                "tradeData": {"symbolId": 41, "volume": 3200,
                              "tradeSide": "BUY", "openTimestamp": open_ts},
            }],
        }
        trades = client.get_open_trades("XAU_USD")
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["id"], "9001")
        self.assertEqual(float(t["price"]), 4804.20)
        self.assertEqual(float(t["stopLossOrder"]["price"]), 4801.42)
        self.assertTrue(t["openTime"].endswith("Z"))

    def test_position_sign_for_sell(self):
        client, conn = make_client()
        conn.responses[PT["RECONCILE_REQ"]] = {
            "position": [{
                "positionId": 9002, "price": 25000.0,
                "tradeData": {"symbolId": 205, "volume": 20000, "tradeSide": 2},
            }],
        }
        pos = client.get_position("NAS100_USD")
        self.assertEqual(pos.units, -200)

    def test_trade_details_vwap_close(self):
        client, conn = make_client()
        conn.responses[PT["DEAL_LIST_BY_POSITION_ID_REQ"]] = {
            "deal": [
                {"executionPrice": 4810.0, "filledVolume": 1600,
                 "closePositionDetail": {"closedVolume": 1600}},
                {"executionPrice": 4812.0, "filledVolume": 1600,
                 "closePositionDetail": {"closedVolume": 1600}},
                {"executionPrice": 4804.2, "filledVolume": 3200},  # opening deal — excluded
            ],
        }
        d = client.get_trade_details("9001")
        self.assertAlmostEqual(float(d["averageClosePrice"]), 4811.0)

    def test_spot_price_cache(self):
        client, conn = make_client()
        p = client.get_price("XAU_USD")
        self.assertAlmostEqual(p.mid, 4800.30)
        self.assertIsNone(client.get_price("NAS100_USD"))  # no spot yet


if __name__ == "__main__":
    unittest.main(verbosity=2)
