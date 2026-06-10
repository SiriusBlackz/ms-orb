"""Hermetic tests for SessionManager range-recording and re-entry rules.

Runs with plain `python test_session_manager.py` (or unittest) — no broker SDK
and no network. The real `oanda_client` (which imports oandapyV20) is stubbed in
sys.modules before the bot modules are imported, so `Candle` is a lightweight
dataclass and SessionManager needs no live client.

Covers the canonical MS-ORB rules (must match backtest.py):
  1. A pre-range candle (17:55) is NOT folded into the Tokyo 18:00 opening range.
  2. The six in-window Tokyo candles 18:00..18:25 ARE the opening range.
  3. A target exit closes the session and arms no re-entry.
  4. A stop exit (under the trade cap) returns to WATCHING_FOR_BREAKOUT.
"""

import os
import sys
import types
import unittest
from dataclasses import dataclass
from datetime import datetime

import pytz

# --- Stub the broker client module BEFORE importing the bot modules ----------
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

# Now safe to import the modules under test (they resolve Candle from the stub).
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


def make_candle(hour, minute, high, low):
    """A complete M5 candle starting at the given NY time."""
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
    """Holds a mutable 'now' so get_current_time_ny is deterministic."""

    def __init__(self):
        self.now = ny(18, 0)


def build_manager():
    """A SessionManager with a stub client and a controllable clock."""
    mgr = SessionManager(instrument="XAU_USD", client=object())
    clock = FakeClock()
    mgr.get_current_time_ny = lambda: clock.now
    return mgr, clock


# The six legitimate Tokyo range candles (18:00..18:25) with distinct extremes.
RANGE_HIGHS = [101.0, 102.0, 105.0, 103.0, 104.0, 102.5]
RANGE_LOWS = [99.0, 98.0, 97.0, 96.0, 95.0, 98.5]
EXPECTED_HIGH = max(RANGE_HIGHS)  # 105.0 (the 18:10 candle)
EXPECTED_LOW = min(RANGE_LOWS)    # 95.0  (the 18:20 candle)


def feed_tokyo_range(mgr, clock, include_pre_range=False):
    """Drive the Tokyo session through its range period.

    If include_pre_range, first feeds a 17:55 candle with extreme high/low so
    that any leak would be visible in the finalized range.
    """
    if include_pre_range:
        clock.now = ny(18, 0)
        mgr.process_candle(make_candle(17, 55, high=9999.0, low=0.0001))

    minutes = [0, 5, 10, 15, 20, 25]
    for i, m in enumerate(minutes):
        # When candle (18:00+m) completes, wall-clock is 5 minutes later.
        nxt = m + 5
        clock.now = ny(18, 30) if nxt >= 30 else ny(18, nxt)
        mgr.process_candle(make_candle(18, m, high=RANGE_HIGHS[i], low=RANGE_LOWS[i]))


def armed_in_trade_context(mgr, direction=Direction.LONG):
    """Put the Tokyo context into IN_TRADE with a triggered breakout detector."""
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
        """A 17:55 candle must not enter the Tokyo 18:00 range."""
        mgr, clock = build_manager()
        clock.now = ny(18, 0)
        mgr.process_candle(make_candle(17, 55, high=9999.0, low=0.0001))

        ctx = mgr.contexts[SessionName.TOKYO]
        self.assertEqual(ctx.state, SessionState.RECORDING_RANGE)
        self.assertEqual(
            len(ctx.range_recorder._candles), 0,
            "pre-range 17:55 candle leaked into the opening range",
        )

    def test_six_range_candles_included(self):
        """Exactly the six 18:00..18:25 candles form the opening range."""
        mgr, clock = build_manager()
        feed_tokyo_range(mgr, clock, include_pre_range=True)

        ctx = mgr.contexts[SessionName.TOKYO]
        self.assertEqual(ctx.state, SessionState.WATCHING_FOR_BREAKOUT)
        self.assertIsNotNone(ctx.opening_range)
        self.assertEqual(ctx.opening_range.candles_used, 6)
        self.assertEqual(ctx.opening_range.high, EXPECTED_HIGH)
        self.assertEqual(ctx.opening_range.low, EXPECTED_LOW)


class TestReentryRules(unittest.TestCase):

    def test_target_exit_closes_session(self):
        """A target exit -> SESSION_CLOSED, no re-entry, direction not reset."""
        mgr, clock = build_manager()
        clock.now = ny(18, 40)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "target", EXPECTED_HIGH + 50.0)

        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        self.assertEqual(ctx.trade_count, 1)
        self.assertTrue(
            detector._breakout_up_triggered,
            "direction was reset after a target exit (would allow a bad re-entry)",
        )

    def test_time_exit_closes_session(self):
        """A time exit -> SESSION_CLOSED, no re-entry."""
        mgr, clock = build_manager()
        clock.now = ny(2, 50)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "time", EXPECTED_LOW)

        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        self.assertTrue(detector._breakout_up_triggered)

    def test_stop_exit_reenters_under_cap(self):
        """A stop exit (under cap) -> WATCHING_FOR_BREAKOUT, direction reset."""
        mgr, clock = build_manager()
        clock.now = ny(18, 50)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "stop", EXPECTED_LOW)

        self.assertEqual(ctx.state, SessionState.WATCHING_FOR_BREAKOUT)
        self.assertEqual(ctx.trade_count, 1)
        self.assertFalse(
            detector._breakout_up_triggered,
            "direction not reset after a stop-out (re-entry would be impossible)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
