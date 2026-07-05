"""Hermetic tests for the Config-A SessionManager.

Runs with plain `python test_session_manager.py` (or unittest) — no broker SDK
and no network. The real `oanda_client` (which imports oandapyV20) is stubbed in
sys.modules before the bot modules are imported, so `Candle` is a lightweight
dataclass and SessionManager needs no live client (a FakeClient serves the
indicator candle fetches).

Covers the Config-A rules (must match `oanda_bot/backtest.py --config-a`):
  1. A pre-range candle (17:55) is NOT folded into the Tokyo 18:00 opening range.
  2. The six in-window Tokyo candles 18:00..18:25 ARE the opening range.
  3. Only ENABLED_SESSIONS (Tokyo) get contexts; London/NY are ignored.
  4. An HTF-misaligned first breakout closes the session with no trade.
  5. An HTF-aligned breakout gets a 1.5xATR stop and 3:1 target.
  6. An indicator-fetch failure rejects the signal (no trade, session closed).
  7. ANY exit (stop included) closes the session — no re-entry.
"""

import os
import sys
import types
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytz

# --- Stub the broker client module BEFORE importing the bot modules ----------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

# Pin the strategy env vars so a stray repo-root .env can't change what this
# test suite validates.
os.environ.update(
    RR_TARGET="3.0",
    MAX_TRADES_PER_SESSION="1",
    STOP_MODE="atr",
    ATR_STOP_MULT="1.5",
    ATR_PERIOD="14",
    HTF_ALIGN="true",
    HTF_EMA_PERIOD="100",
    FIRST_BREAKOUT_ONLY="true",
    ENABLED_SESSIONS="TOKYO",
    INDICATOR_CANDLE_COUNT="1000",
)


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
from config import SessionName, TradingConfig  # noqa: E402
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
    """A complete M5 candle starting at the given NY time."""
    mid = (high + low) / 2
    return Candle(
        time=ny(hour, minute),
        open=mid,
        high=high,
        low=low,
        close=close if close is not None else mid,
        volume=100,
        complete=True,
    )


class FakeClient:
    """Serves a configurable candle history for indicator computation."""

    def __init__(self, history=None, fail=False):
        self.history = history or []
        self.fail = fail
        self.calls = 0

    def get_candles(self, instrument, granularity="M5", count=10, include_incomplete=True):
        self.calls += 1
        if self.fail:
            raise ConnectionError("simulated fetch failure")
        return self.history


def flat_history(price, n=200, end_hour=18, end_minute=35, spread=1.0):
    """n identical candles ending at the given NY time; EMA100 -> ~price,
    Wilder ATR14 -> ~spread (high-low). The last candle is the breakout slot —
    callers overwrite it so indicator lookups hit the exact timestamp."""
    end = ny(end_hour, end_minute)
    out = []
    for i in range(n):
        t = end - timedelta(minutes=5 * (n - 1 - i))
        out.append(Candle(
            time=t, open=price, high=price + spread / 2,
            low=price - spread / 2, close=price, volume=100, complete=True,
        ))
    return out


class FakeClock:
    """Holds a mutable 'now' so get_current_time_ny is deterministic."""

    def __init__(self):
        self.now = ny(18, 0)


def build_manager(client=None):
    """A SessionManager with a fake client and a controllable clock."""
    mgr = SessionManager(instrument="XAU_USD", client=client or FakeClient())
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


def breakout_scenario(history_price, breakout_close=106.0):
    """Build a manager whose Tokyo range is finalized and whose fake client
    serves a flat history at history_price with the breakout candle appended
    (so exact-timestamp indicator lookups succeed)."""
    breakout = make_candle(18, 35, high=breakout_close + 0.5,
                           low=breakout_close - 2.0, close=breakout_close)
    history = flat_history(history_price, end_hour=18, end_minute=35)
    history[-1] = breakout
    client = FakeClient(history=history)
    mgr, clock = build_manager(client)
    feed_tokyo_range(mgr, clock)
    clock.now = ny(18, 40)
    signal = mgr.process_candle(breakout)
    return mgr, signal, client


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
    detector = BreakoutDetector(opening_range, rr_target=3.0)
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


class TestSessionRestriction(unittest.TestCase):

    def test_only_tokyo_context_exists(self):
        """ENABLED_SESSIONS=TOKYO -> no London/NY contexts, no trades there."""
        mgr, _ = build_manager()
        self.assertEqual(set(mgr.contexts), {SessionName.TOKYO})

    def test_london_time_has_no_active_session(self):
        mgr, clock = build_manager()
        clock.now = ny(4, 0)  # inside London window
        self.assertIsNone(mgr.get_active_session())
        self.assertIsNone(mgr.infer_session_from_time(ny(4, 0)))


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


class TestConfigARules(unittest.TestCase):

    def test_aligned_breakout_gets_atr_stop(self):
        """History at 100 -> EMA100 ~100 < close 106 -> LONG aligned.
        ATR14 ~1.0 -> stop_dist 1.5, SL = 106 - 1.5, TP = 106 + 3 * 1.5."""
        mgr, signal, client = breakout_scenario(history_price=100.0)
        self.assertIsNotNone(signal, "aligned breakout should produce a signal")
        self.assertEqual(signal.direction, Direction.LONG)
        self.assertGreaterEqual(client.calls, 1, "indicator fetch did not happen")
        self.assertEqual(signal.rr_target, 3.0)
        stop_dist = signal.entry_price - signal.stop_loss
        reward = signal.take_profit - signal.entry_price
        self.assertAlmostEqual(reward / stop_dist, 3.0, places=6)
        # Expected stop = 1.5 x Wilder ATR at the breakout candle, computed
        # the exact way the bot does (indicators.compute_atr_series).
        from indicators import compute_atr_series
        atr = compute_atr_series(client.history, period=14)
        expected_atr = atr[signal.breakout_candle.time]
        self.assertAlmostEqual(stop_dist, 1.5 * expected_atr, places=6)
        self.assertNotAlmostEqual(
            signal.stop_loss, signal.breakout_candle.low, places=3,
            msg="stop looks candle-based, not ATR-based",
        )
        ctx = mgr.contexts[SessionName.TOKYO]
        self.assertEqual(ctx.state, SessionState.PENDING_ORDER)

    def test_misaligned_breakout_closes_session(self):
        """History at 120 -> EMA100 ~120 > close 106 -> LONG NOT aligned.
        First breakout rejected -> SESSION_CLOSED, no trade today."""
        mgr, signal, client = breakout_scenario(history_price=120.0)
        self.assertIsNone(signal)
        ctx = mgr.contexts[SessionName.TOKYO]
        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        self.assertEqual(ctx.trade_count, 0)

    def test_indicator_fetch_failure_rejects(self):
        """A failed candle fetch must never yield a tradeable signal."""
        breakout = make_candle(18, 35, high=106.5, low=104.0, close=106.0)
        client = FakeClient(fail=True)
        mgr, clock = build_manager(client)
        feed_tokyo_range(mgr, clock)
        clock.now = ny(18, 40)
        signal = mgr.process_candle(breakout)
        self.assertIsNone(signal)
        self.assertEqual(
            mgr.contexts[SessionName.TOKYO].state, SessionState.SESSION_CLOSED
        )


class TestNoReentry(unittest.TestCase):

    def test_target_exit_closes_session(self):
        """A target exit -> SESSION_CLOSED, no re-entry."""
        mgr, clock = build_manager()
        clock.now = ny(18, 40)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "target", EXPECTED_HIGH + 50.0)

        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)
        self.assertEqual(ctx.trade_count, 1)

    def test_time_exit_closes_session(self):
        """A time exit -> SESSION_CLOSED, no re-entry."""
        mgr, clock = build_manager()
        clock.now = ny(2, 50)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "time", EXPECTED_LOW)

        self.assertEqual(ctx.state, SessionState.SESSION_CLOSED)

    def test_stop_exit_also_closes_session(self):
        """Config A: a stop exit closes the session too — first breakout only,
        NO re-entry (this inverts the base bot's behavior deliberately)."""
        mgr, clock = build_manager()
        clock.now = ny(18, 50)
        ctx, detector = armed_in_trade_context(mgr, Direction.LONG)

        mgr.handle_trade_exit(SessionName.TOKYO, "stop", EXPECTED_LOW)

        self.assertEqual(
            ctx.state, SessionState.SESSION_CLOSED,
            "stop exit re-armed the session — first-breakout-only is broken",
        )
        self.assertEqual(ctx.trade_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
