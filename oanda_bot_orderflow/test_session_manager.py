"""Hermetic tests for the CVD+ramp orderflow SessionManager.

Same canonical rules as the base bot (range selection by candle timestamp;
re-entry only after a stop-out), plus a guard that the order-flow `range_candles`
list — the CVD filter's input — contains only the in-window range candles.

Runs with plain `python test_session_manager.py` — no broker SDK, no network.
"""

import os
import sys
import types
import unittest
from dataclasses import dataclass
from datetime import datetime

import pytz

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)


@dataclass
class Candle:
    """Minimal stand-in for oanda_client.Candle (duck-typed by the bot)."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool


_stub = types.ModuleType("oanda_client")
_stub.Candle = Candle
_stub.OANDAClient = type("OANDAClient", (), {})
_stub.Position = type("Position", (), {})
sys.modules["oanda_client"] = _stub

from config import SessionName  # noqa: E402
from strategy import Direction, OpeningRange, BreakoutDetector  # noqa: E402
from session_manager import (  # noqa: E402
    SessionManager,
    SessionState,
    TradeRecord,
)

NY = pytz.timezone("America/New_York")


def ny(hour, minute):
    return NY.localize(datetime(2026, 6, 10, hour, minute))


def make_candle(hour, minute, high, low):
    mid = (high + low) / 2
    return Candle(
        time=ny(hour, minute),
        open=mid,
        high=high,
        low=low,
        close=mid,
        volume=100,
        complete=True,
    )


class FakeClock:
    def __init__(self):
        self.now = ny(18, 0)


def build_manager():
    mgr = SessionManager(instrument="XAU_USD", client=object())
    clock = FakeClock()
    mgr.get_current_time_ny = lambda: clock.now
    return mgr, clock


RANGE_HIGHS = [101.0, 102.0, 105.0, 103.0, 104.0, 102.5]
RANGE_LOWS = [99.0, 98.0, 97.0, 96.0, 95.0, 98.5]
EXPECTED_HIGH = max(RANGE_HIGHS)
EXPECTED_LOW = min(RANGE_LOWS)


def feed_tokyo_range(mgr, clock, include_pre_range=False):
    if include_pre_range:
        clock.now = ny(18, 0)
        mgr.process_candle(make_candle(17, 55, high=9999.0, low=0.0001))

    minutes = [0, 5, 10, 15, 20, 25]
    for i, m in enumerate(minutes):
        nxt = m + 5
        clock.now = ny(18, 30) if nxt >= 30 else ny(18, nxt)
        mgr.process_candle(make_candle(18, m, high=RANGE_HIGHS[i], low=RANGE_LOWS[i]))


def armed_in_trade_context(mgr, direction=Direction.LONG):
    ctx = mgr.contexts[SessionName.TOKYO]
    opening_range = OpeningRange(
        session=SessionName.TOKYO,
        instrument="XAU_USD",
        high=EXPECTED_HIGH,
        low=EXPECTED_LOW,
        start_time=ny(18, 0),
        end_time=ny(18, 30),
        candles_used=6,
    )
    detector = BreakoutDetector(opening_range, rr_target=5.0)
    if direction == Direction.LONG:
        detector._breakout_up_triggered = True
    else:
        detector._breakout_down_triggered = True
    ctx.opening_range = opening_range
    ctx.breakout_detector = detector
    ctx.current_trade = TradeRecord(
        direction=direction,
        entry_price=EXPECTED_HIGH,
        stop_loss=EXPECTED_LOW,
        take_profit=EXPECTED_HIGH + 50.0,
        entry_time=ny(18, 35),
    )
    ctx.state = SessionState.IN_TRADE
    return ctx, detector


class TestRangeRecording(unittest.TestCase):

    def test_pre_range_candle_excluded(self):
        mgr, clock = build_manager()
        clock.now = ny(18, 0)
        mgr.process_candle(make_candle(17, 55, high=9999.0, low=0.0001))

        ctx = mgr.contexts[SessionName.TOKYO]
        self.assertEqual(ctx.state, SessionState.RECORDING_RANGE)
        self.assertEqual(len(ctx.range_recorder._candles), 0)
        self.assertEqual(
            len(ctx.range_candles), 0,
            "pre-range 17:55 candle leaked into the CVD filter's range_candles",
        )

    def test_six_range_candles_included(self):
        mgr, clock = build_manager()
        feed_tokyo_range(mgr, clock, include_pre_range=True)

        ctx = mgr.contexts[SessionName.TOKYO]
        self.assertEqual(ctx.state, SessionState.WATCHING_FOR_BREAKOUT)
        self.assertEqual(ctx.opening_range.candles_used, 6)
        self.assertEqual(ctx.opening_range.high, EXPECTED_HIGH)
        self.assertEqual(ctx.opening_range.low, EXPECTED_LOW)

    def test_range_candles_match_window(self):
        """The order-flow range_candles list holds exactly the six in-window candles."""
        mgr, clock = build_manager()
        feed_tokyo_range(mgr, clock, include_pre_range=True)

        ctx = mgr.contexts[SessionName.TOKYO]
        self.assertEqual(len(ctx.range_candles), 6)
        highs = [c.high for c in ctx.range_candles]
        self.assertNotIn(9999.0, highs, "pre-range candle present in range_candles")
        self.assertEqual(highs, RANGE_HIGHS)


class TestReentryRules(unittest.TestCase):

    def test_target_exit_closes_session(self):
        mgr, clock = build_manager()
        clock.now = ny(18, 40)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "target", EXPECTED_HIGH + 50.0)

        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        self.assertEqual(ctx.trade_count, 1)
        self.assertTrue(detector._breakout_up_triggered)

    def test_time_exit_closes_session(self):
        mgr, clock = build_manager()
        clock.now = ny(2, 50)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "time", EXPECTED_LOW)

        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        self.assertTrue(detector._breakout_up_triggered)

    def test_stop_exit_reenters_under_cap(self):
        mgr, clock = build_manager()
        clock.now = ny(18, 50)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "stop", EXPECTED_LOW)

        self.assertEqual(ctx.state, SessionState.WATCHING_FOR_BREAKOUT)
        self.assertEqual(ctx.trade_count, 1)
        self.assertFalse(detector._breakout_up_triggered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
