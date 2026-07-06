"""Hermetic tests for the IBKR-bot SessionManager (Config-A rules on TSLA).

No broker SDK, no network — `ibkr_client` is stubbed in sys.modules before
the bot modules are imported; a FakeClient serves the indicator fetches.

Covers (must match `oanda_bot/backtest.py --config-a`, NY session):
  1. A pre-range 09:25 candle is NOT folded into the 09:30 opening range.
  2. The six 09:30..09:55 candles ARE the opening range.
  3. An HTF-misaligned first breakout closes the session with no trade.
  4. An HTF-aligned breakout gets a 1.5xATR stop and 3:1 target.
  5. An indicator-fetch failure rejects the signal (no trade).
  6. ANY exit (stop included) closes the session — no re-entry.
"""

import os
import sys
import types
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytz

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

os.environ.update(
    RR_TARGET="3.0",
    MAX_TRADES_PER_SESSION="1",
    STOP_MODE="atr",
    ATR_STOP_MULT="1.5",
    ATR_PERIOD="14",
    HTF_ALIGN="true",
    HTF_EMA_PERIOD="100",
    FIRST_BREAKOUT_ONLY="true",
)


@dataclass
class Candle:
    """Minimal stand-in for ibkr_client.Candle (duck-typed by the bot)."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool


_stub = types.ModuleType("ibkr_client")
_stub.Candle = Candle
_stub.IBKRClient = type("IBKRClient", (), {})
sys.modules["ibkr_client"] = _stub

from config import SessionName  # noqa: E402
from strategy import Direction, OpeningRange, BreakoutDetector  # noqa: E402
from session_manager import (  # noqa: E402
    SessionManager,
    SessionState,
    TradeRecord,
)

NY = pytz.timezone("America/New_York")


def ny(hour, minute):
    """NY-local tz-aware datetime on a fixed weekday (2026-06-10, a Wednesday)."""
    return NY.localize(datetime(2026, 6, 10, hour, minute))


def make_candle(hour, minute, high, low, close=None):
    mid = (high + low) / 2
    return Candle(
        time=ny(hour, minute), open=mid, high=high, low=low,
        close=close if close is not None else mid, volume=1000, complete=True,
    )


class FakeClient:
    """Serves a configurable candle history for indicator computation."""

    def __init__(self, history=None, fail=False):
        self.history = history or []
        self.fail = fail
        self.calls = 0

    def get_candles(self, instrument, bar_size="5 mins", duration="3600 S",
                    include_incomplete=True):
        self.calls += 1
        if self.fail:
            raise ConnectionError("simulated fetch failure")
        return self.history


def flat_history(price, n=200, end_hour=10, end_minute=5, spread=1.0):
    """n identical candles ending at the given NY time (EMA100 -> ~price,
    Wilder ATR14 -> ~spread). Last slot is overwritten with the breakout."""
    end = ny(end_hour, end_minute)
    out = []
    for i in range(n):
        t = end - timedelta(minutes=5 * (n - 1 - i))
        out.append(Candle(
            time=t, open=price, high=price + spread / 2,
            low=price - spread / 2, close=price, volume=1000, complete=True,
        ))
    return out


class FakeClock:
    def __init__(self):
        self.now = ny(9, 30)


def build_manager(client=None):
    mgr = SessionManager(instrument="TSLA", client=client or FakeClient())
    clock = FakeClock()
    mgr.get_current_time_ny = lambda: clock.now
    return mgr, clock


RANGE_HIGHS = [430.0, 432.0, 435.0, 433.0, 434.0, 432.5]
RANGE_LOWS = [428.0, 427.0, 426.0, 425.0, 424.0, 427.5]
EXPECTED_HIGH = max(RANGE_HIGHS)  # 435.0
EXPECTED_LOW = min(RANGE_LOWS)    # 424.0


def feed_ny_range(mgr, clock, include_pre_range=False):
    """Drive the NY session through its 09:30-10:00 range period."""
    if include_pre_range:
        clock.now = ny(9, 30)
        mgr.process_candle(make_candle(9, 25, high=9999.0, low=0.0001))

    minutes = [30, 35, 40, 45, 50, 55]
    for i, m in enumerate(minutes):
        nxt = m + 5
        clock.now = ny(10, 0) if nxt >= 60 else ny(9, nxt)
        mgr.process_candle(make_candle(9, m, high=RANGE_HIGHS[i], low=RANGE_LOWS[i]))


def breakout_scenario(history_price, breakout_close=436.0):
    breakout = make_candle(10, 5, high=breakout_close + 0.5,
                           low=breakout_close - 2.0, close=breakout_close)
    history = flat_history(history_price, end_hour=10, end_minute=5)
    history[-1] = breakout
    client = FakeClient(history=history)
    mgr, clock = build_manager(client)
    feed_ny_range(mgr, clock)
    clock.now = ny(10, 10)
    signal = mgr.process_candle(breakout)
    return mgr, signal, client


def armed_in_trade(mgr, direction=Direction.LONG):
    ctx = mgr.context
    opening_range = OpeningRange(
        session=SessionName.NY, instrument="TSLA",
        high=EXPECTED_HIGH, low=EXPECTED_LOW,
        start_time=ny(9, 30), end_time=ny(10, 0), candles_used=6,
    )
    detector = BreakoutDetector(opening_range, rr_target=3.0)
    if direction == Direction.LONG:
        detector._breakout_up_triggered = True
    else:
        detector._breakout_down_triggered = True
    ctx.opening_range = opening_range
    ctx.breakout_detector = detector
    ctx.current_trade = TradeRecord(
        direction=direction, entry_price=EXPECTED_HIGH, stop_loss=EXPECTED_LOW,
        take_profit=EXPECTED_HIGH + 30.0, entry_time=ny(10, 5),
    )
    ctx.state = SessionState.IN_TRADE
    return ctx, detector


class TestRangeRecording(unittest.TestCase):

    def test_pre_range_candle_excluded(self):
        """A 09:25 candle must not enter the 09:30 opening range."""
        mgr, clock = build_manager()
        clock.now = ny(9, 30)
        mgr.process_candle(make_candle(9, 25, high=9999.0, low=0.0001))
        ctx = mgr.context
        self.assertEqual(ctx.state, SessionState.RECORDING_RANGE)
        self.assertEqual(
            len(ctx.range_recorder._candles), 0,
            "pre-range 09:25 candle leaked into the opening range",
        )

    def test_six_range_candles_included(self):
        mgr, clock = build_manager()
        feed_ny_range(mgr, clock, include_pre_range=True)
        ctx = mgr.context
        self.assertEqual(ctx.state, SessionState.WATCHING_FOR_BREAKOUT)
        self.assertEqual(ctx.opening_range.candles_used, 6)
        self.assertEqual(ctx.opening_range.high, EXPECTED_HIGH)
        self.assertEqual(ctx.opening_range.low, EXPECTED_LOW)


class TestConfigARules(unittest.TestCase):

    def test_aligned_breakout_gets_atr_stop(self):
        """History at 430 -> EMA100 ~430 < close 436 -> LONG aligned;
        stop = 1.5 x Wilder ATR14 at the breakout, TP at 3R."""
        mgr, signal, client = breakout_scenario(history_price=430.0)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.LONG)
        self.assertGreaterEqual(client.calls, 1)
        stop_dist = signal.entry_price - signal.stop_loss
        reward = signal.take_profit - signal.entry_price
        self.assertAlmostEqual(reward / stop_dist, 3.0, places=6)
        from indicators import compute_atr_series
        atr = compute_atr_series(client.history, period=14)
        self.assertAlmostEqual(
            stop_dist, 1.5 * atr[signal.breakout_candle.time], places=6)
        self.assertEqual(mgr.context.state, SessionState.IN_TRADE)

    def test_misaligned_breakout_closes_session(self):
        """History at 500 -> EMA100 ~500 > close 436 -> LONG NOT aligned."""
        mgr, signal, client = breakout_scenario(history_price=500.0)
        self.assertIsNone(signal)
        self.assertEqual(mgr.context.state, SessionState.SESSION_CLOSED)
        self.assertEqual(mgr.context.trade_count, 0)

    def test_indicator_fetch_failure_rejects(self):
        breakout = make_candle(10, 5, high=436.5, low=434.0, close=436.0)
        client = FakeClient(fail=True)
        mgr, clock = build_manager(client)
        feed_ny_range(mgr, clock)
        clock.now = ny(10, 10)
        signal = mgr.process_candle(breakout)
        self.assertIsNone(signal)
        self.assertEqual(mgr.context.state, SessionState.SESSION_CLOSED)


class TestNoReentry(unittest.TestCase):

    def test_target_exit_closes_session(self):
        mgr, clock = build_manager()
        clock.now = ny(11, 0)
        ctx, detector = armed_in_trade(mgr)
        mgr.handle_trade_exit("target", EXPECTED_HIGH + 30.0)
        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        self.assertTrue(
            detector._breakout_up_triggered,
            "direction reset after a target exit (would allow a bad re-entry)",
        )

    def test_stop_exit_also_closes_session(self):
        """Config A: stop exit -> SESSION_CLOSED (no re-entry)."""
        mgr, clock = build_manager()
        clock.now = ny(11, 0)
        ctx, detector = armed_in_trade(mgr)
        mgr.handle_trade_exit("stop", EXPECTED_LOW)
        self.assertEqual(
            ctx.state, SessionState.SESSION_CLOSED,
            "stop exit re-armed the session — first-breakout-only is broken",
        )
        self.assertEqual(ctx.trade_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
