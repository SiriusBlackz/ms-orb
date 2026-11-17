"""Hermetic tests for IBKR order placement, exit detection and reconciliation.

Runs with plain `python test_order_flow.py`. No gateway and no network: the
real IBKRClient / TradingBot are used, with the `ib_insync.IB` object swapped
for a scripted fake.

Covers the failures found in the 2026-11-17 audit of the paper bot:
  1. An entry that is still PendingSubmit after 2s but fills shortly after is
     a SUCCESS (previously reported as failed while the order stayed live).
  2. An entry that never fills is CANCELLED - nothing is left working.
  3. Exits are read from the bracket's own fills (target / stop), with a
     fallback to the day's executions.
  4. "Flat" is not believed while disconnected; an unknown exit is recorded
     only after repeated confirmed-flat checks.
  5. A failed entry releases the session (no phantom IN_TRADE).
  6. An unexplained position blocks new entries; stale orders are cancelled
     at startup; a position with its bracket working is adopted.
  7. The time exit waits for the closing fill and records its real price.
"""

import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone

_TMP = tempfile.mkdtemp(prefix="ibkr_test_")
os.environ["DB_PATH"] = os.path.join(_TMP, "test_trades.db")
os.environ["LOG_DIR"] = os.path.join(_TMP, "logs")
os.environ["ENTRY_FILL_TIMEOUT"] = "5"
os.environ["EXIT_FILL_TIMEOUT"] = "5"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ibkr_client as ic  # noqa: E402
from ibkr_client import IBKRClient  # noqa: E402
from config import SessionName  # noqa: E402
from session_manager import SessionState, TradeRecord  # noqa: E402
from strategy import BreakoutSignal, Direction, OpeningRange  # noqa: E402
import main as bot_main  # noqa: E402


class FakeTrade:
    def __init__(self, contract, order):
        self.contract = contract
        self.order = order
        self.orderStatus = types.SimpleNamespace(
            status="PendingSubmit", avgFillPrice=0.0, filled=0)

    def fill(self, price):
        self.orderStatus.status = "Filled"
        self.orderStatus.avgFillPrice = price
        self.orderStatus.filled = self.order.totalQuantity


class FakeIB:
    """Scripted stand-in for ib_insync.IB."""

    def __init__(self):
        self.trades = []
        self.clock = 0.0
        self.connected = True
        self.position_units = 0
        self.avg_cost = 0.0
        self.day_fills = []
        self.cancelled_ids = []
        self.on_sleep = None      # callable(fake) run on every sleep
        self._next_id = 100

    # -- order API ---------------------------------------------------------
    def bracketOrder(self, action, quantity, limitPrice, takeProfitPrice, stopLossPrice):
        rev = "SELL" if action == "BUY" else "BUY"
        mk = lambda **kw: types.SimpleNamespace(orderId=self._id(), transmit=True, **kw)
        return [
            mk(action=action, totalQuantity=quantity, orderType="LMT", lmtPrice=limitPrice, auxPrice=0.0),
            mk(action=rev, totalQuantity=quantity, orderType="LMT", lmtPrice=takeProfitPrice, auxPrice=0.0),
            mk(action=rev, totalQuantity=quantity, orderType="STP", lmtPrice=0.0, auxPrice=stopLossPrice),
        ]

    def _id(self):
        self._next_id += 1
        return self._next_id

    def placeOrder(self, contract, order):
        if not getattr(order, "orderId", 0):
            order.orderId = self._id()
        t = FakeTrade(contract, order)
        self.trades.append(t)
        return t

    def cancelOrder(self, order):
        self.cancelled_ids.append(order.orderId)
        for t in self.trades:
            if t.order is order and t.orderStatus.status != "Filled":
                t.orderStatus.status = "Cancelled"

    def openTrades(self):
        return [t for t in self.trades
                if t.orderStatus.status not in ("Filled", "Cancelled", "ApiCancelled", "Inactive")]

    # -- misc API ----------------------------------------------------------
    def sleep(self, seconds):
        self.clock += seconds
        if self.on_sleep:
            self.on_sleep(self)

    def isConnected(self):
        return self.connected

    def positions(self):
        if self.position_units == 0:
            return []
        return [types.SimpleNamespace(contract=types.SimpleNamespace(symbol="TSLA"),
                                      position=self.position_units, avgCost=self.avg_cost)]

    def fills(self):
        return self.day_fills

    def qualifyContracts(self, contract):
        return [contract]

    def accountSummary(self):
        return [types.SimpleNamespace(tag="NetLiquidation", value="1000000", account="TEST")]

    def reqMarketDataType(self, t):
        pass

    def disconnect(self):
        self.connected = False


def make_client():
    client = IBKRClient()
    client.ib = FakeIB()
    return client


def make_fill(side, shares, price, when):
    return types.SimpleNamespace(
        contract=types.SimpleNamespace(symbol="TSLA"),
        execution=types.SimpleNamespace(side=side, shares=shares, price=price),
        time=when)


class TestEntry(unittest.TestCase):
    def test_slow_fill_is_success(self):
        """PendingSubmit at 2s, Filled at 3s -> success with the real fill price."""
        client = make_client()
        fake = client.ib

        def later(f):
            if f.clock >= 3 and f.trades[0].orderStatus.status == "PendingSubmit":
                f.trades[0].fill(333.52)
        fake.on_sleep = later

        res = client.place_market_order("TSLA", 100, stop_loss=330.86, take_profit=337.23)
        self.assertTrue(res.success)
        self.assertEqual(res.fill_price, 333.52)
        self.assertEqual(fake.cancelled_ids, [])
        self.assertIn("TSLA", client._brackets)

    def test_never_filled_is_cancelled(self):
        """Stuck in PendingSubmit -> failure AND the whole bracket is cancelled."""
        client = make_client()
        fake = client.ib
        res = client.place_market_order("TSLA", -100, stop_loss=380.06, take_profit=370.50)
        self.assertFalse(res.success)
        self.assertIn("bracket cancelled", res.error)
        self.assertEqual(len(fake.cancelled_ids), 3)
        self.assertEqual(fake.openTrades(), [])
        self.assertNotIn("TSLA", client._brackets)

    def test_fill_crossing_the_cancel_counts_as_filled(self):
        """If the parent fills while being cancelled, the trade is live and tracked."""
        client = make_client()
        fake = client.ib
        orig_cancel = fake.cancelOrder

        def racing_cancel(order):
            if order is fake.trades[0].order:
                fake.trades[0].fill(378.00)   # exchange filled before the cancel landed
                return
            # children of a filled parent stay working at IB; do not cancel them
        fake.cancelOrder = racing_cancel
        res = client.place_market_order("TSLA", -100, stop_loss=380.06, take_profit=370.50)
        fake.cancelOrder = orig_cancel
        self.assertTrue(res.success)
        self.assertEqual(res.fill_price, 378.00)
        self.assertIn("TSLA", client._brackets)


class TestExitDetection(unittest.TestCase):
    def _open(self, units=100):
        client = make_client()
        client.ib.on_sleep = lambda f: f.trades[0].fill(100.0)
        res = client.place_market_order("TSLA", units, stop_loss=99.0, take_profit=103.0)
        assert res.success
        client.ib.on_sleep = None
        return client

    def test_target_fill(self):
        client = self._open()
        client.ib.trades[1].fill(103.01)
        fill = client.get_exit_fill("TSLA")
        self.assertEqual((fill.reason, fill.price), ("target", 103.01))

    def test_stop_fill(self):
        client = self._open()
        client.ib.trades[2].fill(98.94)
        fill = client.get_exit_fill("TSLA")
        self.assertEqual((fill.reason, fill.price), ("stop", 98.94))

    def test_fallback_to_executions(self):
        """No bracket objects (restart): use closing-side executions since entry."""
        client = make_client()
        t0 = datetime.now(timezone.utc)
        client.ib.day_fills = [
            make_fill("BOT", 100, 100.0, t0),                       # the entry: ignored
            make_fill("SLD", 50, 99.0, t0 + timedelta(minutes=5)),  # earlier unrelated day part
            make_fill("SLD", 60, 99.0, t0 + timedelta(minutes=30)),
            make_fill("SLD", 40, 98.5, t0 + timedelta(minutes=30)),
        ]
        fill = client.get_exit_fill("TSLA", since=t0 + timedelta(minutes=10), entry_action="BUY")
        self.assertIsNone(fill.reason)
        self.assertAlmostEqual(fill.price, (60 * 99.0 + 40 * 98.5) / 100)

    def test_no_fill_found(self):
        client = make_client()
        self.assertIsNone(client.get_exit_fill("TSLA", entry_action="BUY"))


def make_bot():
    bot = bot_main.TradingBot(dry_run=False)
    bot.client.ib = FakeIB()
    return bot


def make_signal(direction=Direction.LONG):
    rng = OpeningRange(session=SessionName.NY, instrument="TSLA", high=100.0, low=98.0,
                       start_time=datetime.now(), end_time=datetime.now())
    if direction == Direction.LONG:
        entry, sl, tp = 100.5, 99.5, 103.5
    else:
        entry, sl, tp = 97.5, 98.5, 94.5
    candle = ic.Candle(time=datetime.now(), open=entry, high=entry, low=entry,
                       close=entry, volume=1, complete=True)
    return BreakoutSignal(session=SessionName.NY, instrument="TSLA", direction=direction,
                          entry_price=entry, stop_loss=sl, take_profit=tp, opening_range=rng,
                          signal_time=datetime.now(), breakout_candle=candle,
                          risk_per_unit=abs(entry - sl), rr_target=3.0)


def arm_session(bot, signal):
    """Put the session manager where it is when a signal has just fired."""
    ctx = bot.session_manager.context
    ctx.state = SessionState.IN_TRADE
    ctx.current_trade = TradeRecord(
        direction=signal.direction, entry_price=signal.entry_price,
        stop_loss=signal.stop_loss, take_profit=signal.take_profit,
        entry_time=datetime.now())
    return ctx


class TestBotFlow(unittest.TestCase):
    def test_failed_entry_releases_session(self):
        bot = make_bot()
        sig = make_signal()
        ctx = arm_session(bot, sig)
        bot._on_signal(sig)                       # order never fills
        self.assertIsNone(bot._open_position)
        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        self.assertEqual(ctx.trade_count, 0)
        self.assertEqual(bot.client.ib.openTrades(), [])

    def _enter(self, bot, sig, fill_price):
        fake = bot.client.ib
        ctx = arm_session(bot, sig)
        fake.on_sleep = lambda f: f.trades[0].fill(fill_price)
        bot._on_signal(sig)
        fake.on_sleep = None
        fake.position_units = bot._open_position["units"] * (1 if sig.direction == Direction.LONG else -1)
        return ctx

    def test_stop_exit_logged_with_real_price(self):
        bot = make_bot()
        sig = make_signal()
        ctx = self._enter(bot, sig, 100.60)
        self.assertIsNotNone(bot._open_position)
        bot._check_position()                      # still open
        self.assertIsNotNone(bot._open_position)

        fake = bot.client.ib
        fake.trades[2].fill(99.45)                 # stop fills
        fake.position_units = 0
        bot._check_position()
        self.assertIsNone(bot._open_position)
        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        import sqlite3
        con = sqlite3.connect(os.environ["DB_PATH"])
        exit_price, reason, pnl = con.execute(
            "SELECT exit_price, exit_reason, pnl FROM trades ORDER BY id DESC LIMIT 1").fetchone()
        con.close()
        self.assertEqual((exit_price, reason), (99.45, "stop"))
        self.assertLess(pnl, 0)

    def test_disconnect_is_not_an_exit(self):
        bot = make_bot()
        sig = make_signal()
        self._enter(bot, sig, 100.60)
        fake = bot.client.ib
        fake.connected = False                     # nightly gateway restart
        fake.position_units = 0                    # positions() would look empty
        for _ in range(10):
            bot._check_position()
        self.assertIsNotNone(bot._open_position)

    def test_unknown_exit_needs_repeated_confirmation(self):
        bot = make_bot()
        sig = make_signal()
        ctx = self._enter(bot, sig, 100.60)
        fake = bot.client.ib
        fake.position_units = 0                    # flat, but no closing fill anywhere
        bot._check_position(); bot._check_position()
        self.assertIsNotNone(bot._open_position)
        bot._check_position()
        self.assertIsNone(bot._open_position)
        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)

    def test_time_exit_uses_closing_fill(self):
        bot = make_bot()
        sig = make_signal(Direction.SHORT)
        ctx = self._enter(bot, sig, 97.40)
        fake = bot.client.ib

        def fill_close(f):
            for t in f.trades[3:]:
                if t.orderStatus.status == "PendingSubmit":
                    t.fill(96.10)
                    f.position_units = 0
        fake.on_sleep = fill_close
        bot._get_current_time_ny = lambda: bot_main.NY_TZ.localize(datetime(2026, 11, 18, 15, 56))
        bot._check_time_exit()
        self.assertIsNone(bot._open_position)
        # bracket children were cancelled before the close went in
        self.assertIn(fake.trades[1].order.orderId, fake.cancelled_ids)
        self.assertIn(fake.trades[2].order.orderId, fake.cancelled_ids)
        import sqlite3
        con = sqlite3.connect(os.environ["DB_PATH"])
        exit_price, reason, pnl = con.execute(
            "SELECT exit_price, exit_reason, pnl FROM trades ORDER BY id DESC LIMIT 1").fetchone()
        con.close()
        self.assertEqual((exit_price, reason), (96.10, "time"))
        self.assertGreater(pnl, 0)

    def test_time_exit_not_filled_keeps_position_tracked(self):
        bot = make_bot()
        sig = make_signal()
        self._enter(bot, sig, 100.60)
        bot._get_current_time_ny = lambda: bot_main.NY_TZ.localize(datetime(2026, 11, 18, 15, 56))
        bot._check_time_exit()                     # closing order never fills
        self.assertIsNotNone(bot._open_position)   # will retry next loop


class TestDelayedData(unittest.TestCase):
    def _run(self, minutes_old):
        bot = make_bot()
        now = bot_main.NY_TZ.localize(datetime(2026, 11, 18, 10, 30, 30))
        bot._get_current_time_ny = lambda: now
        bot.session_manager.is_session_active = lambda: True
        seen = []
        bot.session_manager.process_candle = lambda c: seen.append(c)
        bar_start = now - timedelta(minutes=5 + minutes_old, seconds=30)
        candle = ic.Candle(time=bar_start, open=1, high=1, low=1, close=1, volume=1, complete=True)
        bot.client.get_latest_complete_candle = lambda *a, **k: candle
        bot._process_candles()
        return seen

    def test_fresh_bar_is_processed(self):
        self.assertEqual(len(self._run(0)), 1)

    def test_fifteen_minute_old_bar_is_ignored(self):
        self.assertEqual(len(self._run(15)), 0)


class TestStartupReconcile(unittest.TestCase):
    def test_unexplained_position_blocks_trading(self):
        bot = make_bot()
        fake = bot.client.ib
        fake.position_units = 1658
        fake.avg_cost = 380.07
        bot._reconcile_on_start()
        self.assertIsNone(bot._open_position)
        self.assertIn("1658", bot._blocked_reason)

        sig = make_signal()
        ctx = arm_session(bot, sig)
        bot._on_signal(sig)
        self.assertEqual(fake.trades, [])          # no order was sent
        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)

        fake.position_units = 0                    # user flattened the account
        bot._refresh_block()
        self.assertIsNone(bot._blocked_reason)

    def test_stale_orders_cancelled_when_flat(self):
        bot = make_bot()
        fake = bot.client.ib
        contract = types.SimpleNamespace(symbol="TSLA")
        fake.placeOrder(contract, types.SimpleNamespace(
            orderId=7, action="SELL", totalQuantity=100, orderType="STP", auxPrice=99.0, lmtPrice=0.0))
        bot._reconcile_on_start()
        self.assertEqual(fake.cancelled_ids, [7])
        self.assertIsNone(bot._blocked_reason)

    def test_position_with_bracket_is_adopted(self):
        bot = make_bot()
        fake = bot.client.ib
        fake.position_units = -200
        fake.avg_cost = 97.40
        contract = types.SimpleNamespace(symbol="TSLA")
        fake.placeOrder(contract, types.SimpleNamespace(
            orderId=8, action="BUY", totalQuantity=200, orderType="STP", auxPrice=98.5, lmtPrice=0.0))
        fake.placeOrder(contract, types.SimpleNamespace(
            orderId=9, action="BUY", totalQuantity=200, orderType="LMT", auxPrice=0.0, lmtPrice=94.5))
        bot._reconcile_on_start()
        self.assertIsNone(bot._blocked_reason)
        self.assertEqual(bot._open_position["direction"], Direction.SHORT)
        self.assertEqual(bot._open_position["stop_loss"], 98.5)
        self.assertEqual(bot._open_position["take_profit"], 94.5)
        self.assertEqual(bot.session_manager.context.state, SessionState.IN_TRADE)
        self.assertEqual(fake.cancelled_ids, [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
