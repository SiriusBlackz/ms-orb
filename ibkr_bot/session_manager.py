"""Session state machine for the MS-ORB IBKR bot (Config-A rules).

NY session only for TSLA. Config-A semantics (match backtest.py --config-a):
- Only the session's FIRST breakout signal is eligible; if it fails the HTF
  filter the session closes with no trade, and there is no re-entry after
  any exit.
- SL = ATR_STOP_MULT x ATR14 from entry; TP = entry +/- RR_TARGET x risk.
- Breakouts must be HTF-aligned (close vs EMA100 at the breakout candle).

Also brings this bot to parity with the June 2026 OANDA fixes it had missed:
range membership is decided by the candle's own timestamp (not wall clock),
and re-entry (when FIRST_BREAKOUT_ONLY is off) is allowed only after a
stop-out.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional, Callable
import pytz

from config import (
    SessionConfig,
    SessionName,
    SESSIONS,
    TradingConfig,
    TIMEZONE,
)
from indicators import compute_atr_series, compute_ema_series, value_at_or_last
from strategy import (
    Direction,
    OpeningRange,
    BreakoutSignal,
    RangeRecorder,
    BreakoutDetector,
)
from ibkr_client import Candle, IBKRClient

logger = logging.getLogger(__name__)

# New York timezone
NY_TZ = pytz.timezone(TIMEZONE)


class SessionState(Enum):
    """State machine states for a trading session."""
    WAITING_FOR_SESSION = "WAITING_FOR_SESSION"
    RECORDING_RANGE = "RECORDING_RANGE"
    WATCHING_FOR_BREAKOUT = "WATCHING_FOR_BREAKOUT"
    IN_TRADE = "IN_TRADE"
    SESSION_CLOSED = "SESSION_CLOSED"


@dataclass
class TradeRecord:
    """Record of a trade taken in the session."""
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop', 'target', 'time'
    trade_id: Optional[str] = None


@dataclass
class SessionContext:
    """Context data for a single instrument in the NY session."""
    session: SessionName
    instrument: str
    state: SessionState = SessionState.WAITING_FOR_SESSION
    opening_range: Optional[OpeningRange] = None
    breakout_detector: Optional[BreakoutDetector] = None
    range_recorder: Optional[RangeRecorder] = None
    trades: list[TradeRecord] = field(default_factory=list)
    current_trade: Optional[TradeRecord] = None
    current_trade_id: Optional[str] = None
    last_candle_time: Optional[datetime] = None

    @property
    def trade_count(self) -> int:
        """Number of trades taken this session."""
        return len(self.trades)

    @property
    def can_trade(self) -> bool:
        """Check if more trades are allowed this session."""
        return self.trade_count < TradingConfig.MAX_TRADES_PER_SESSION

    def reset(self) -> None:
        """Reset context for a new session."""
        self.state = SessionState.WAITING_FOR_SESSION
        self.opening_range = None
        self.breakout_detector = None
        self.range_recorder = None
        self.trades = []
        self.current_trade = None
        self.current_trade_id = None
        self.last_candle_time = None
        logger.info(f"Session context reset: {self.session.value} {self.instrument}")


class SessionManager:
    """Manages the state machine for TSLA in NY session."""

    def __init__(
        self,
        instrument: str,
        client: IBKRClient,
        on_signal: Optional[Callable[[BreakoutSignal], None]] = None,
        on_trade_closed: Optional[Callable[[TradeRecord], None]] = None,
    ):
        self.instrument = instrument
        self.client = client
        self.on_signal = on_signal
        self.on_trade_closed = on_trade_closed

        # Single context for NY session
        self.context = SessionContext(session=SessionName.NY, instrument=instrument)
        self._last_check_time: Optional[datetime] = None

    def get_current_time_ny(self) -> datetime:
        """Get current time in New York timezone."""
        return datetime.now(NY_TZ)

    def is_session_active(self) -> bool:
        """Check if NY session is currently active."""
        current_time = self.get_current_time_ny().time()
        config = SESSIONS[SessionName.NY]
        return config.range_start <= current_time <= config.session_close

    def _is_in_range_period(self, current_time: time) -> bool:
        """Check if current time is in the range recording period."""
        config = SESSIONS[SessionName.NY]
        return config.range_start <= current_time < config.range_end

    def _candle_in_range_window(self, candle: Candle) -> bool:
        """Range membership by the candle's OWN NY-local start time.

        Wall-clock gating would fold in a pre-range candle that completes
        exactly at range_start (e.g. a 09:25-09:30 bar) — same fix as the
        OANDA bots' June 2026 range-selection correction.
        """
        config = SESSIONS[SessionName.NY]
        t = candle.time.astimezone(NY_TZ).time()
        return config.range_start <= t < config.range_end

    def _is_in_entry_window(self, current_time: time) -> bool:
        """Check if current time is in the entry window."""
        config = SESSIONS[SessionName.NY]
        return config.range_end <= current_time <= config.entry_end

    def _is_time_to_close(self, current_time: time) -> bool:
        """Check if it's time for auto-close."""
        config = SESSIONS[SessionName.NY]
        return current_time >= config.session_close

    def process_candle(self, candle: Candle) -> Optional[BreakoutSignal]:
        """Process a new candle and update state machine.

        Args:
            candle: Complete 5-minute candle.

        Returns:
            BreakoutSignal if one is generated, None otherwise.
        """
        if not candle.complete:
            return None

        now = self.get_current_time_ny()
        current_time = now.time()
        ctx = self.context
        config = SESSIONS[SessionName.NY]

        if not self.is_session_active():
            if ctx.state != SessionState.SESSION_CLOSED:
                ctx.state = SessionState.SESSION_CLOSED
            return None

        # Skip if same candle already processed
        if ctx.last_candle_time and candle.time <= ctx.last_candle_time:
            return None

        ctx.last_candle_time = candle.time

        # State machine logic
        return self._update_state(ctx, config, candle, current_time)

    def _update_state(
        self,
        ctx: SessionContext,
        config: SessionConfig,
        candle: Candle,
        current_time: time,
    ) -> Optional[BreakoutSignal]:
        """Update state machine for the session context."""
        signal = None

        if ctx.state == SessionState.WAITING_FOR_SESSION:
            if self._is_in_range_period(current_time):
                ctx.state = SessionState.RECORDING_RANGE
                ctx.range_recorder = RangeRecorder(ctx.session, self.instrument)
                logger.info(f"[NY] {self.instrument} -> RECORDING_RANGE")
                if self._candle_in_range_window(candle):
                    ctx.range_recorder.add_candle(candle)

        elif ctx.state == SessionState.RECORDING_RANGE:
            if self._candle_in_range_window(candle):
                ctx.range_recorder.add_candle(candle)
            if self._is_in_range_period(current_time):
                pass  # still recording
            else:
                # Range period ended - finalize and move to watching
                ctx.opening_range = ctx.range_recorder.finalize()
                if ctx.opening_range:
                    ctx.breakout_detector = BreakoutDetector(
                        ctx.opening_range,
                        rr_target=TradingConfig.RR_TARGET,
                    )
                    ctx.state = SessionState.WATCHING_FOR_BREAKOUT
                    logger.info(
                        f"[NY] {self.instrument} -> WATCHING_FOR_BREAKOUT "
                        f"(Range: H={ctx.opening_range.high:.2f} L={ctx.opening_range.low:.2f})"
                    )
                else:
                    logger.warning(f"[NY] {self.instrument}: Failed to finalize range")
                    ctx.state = SessionState.SESSION_CLOSED

        elif ctx.state == SessionState.WATCHING_FOR_BREAKOUT:
            if self._is_time_to_close(current_time):
                ctx.state = SessionState.SESSION_CLOSED
                logger.info(f"[NY] {self.instrument} -> SESSION_CLOSED (time)")
                return None

            # Check for breakout if we can still trade and in entry window
            if ctx.can_trade and self._is_in_entry_window(current_time):
                signal = ctx.breakout_detector.check_breakout(candle)
                if signal:
                    signal = self._apply_config_a_rules(ctx, signal, candle)
                if signal:
                    ctx.current_trade = TradeRecord(
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        entry_time=self.get_current_time_ny(),
                    )
                    ctx.state = SessionState.IN_TRADE
                    logger.info(
                        f"[NY] {self.instrument} -> IN_TRADE "
                        f"({signal.direction.value} @ {signal.entry_price:.2f})"
                    )

                    if self.on_signal:
                        self.on_signal(signal)

        elif ctx.state == SessionState.IN_TRADE:
            if self._is_time_to_close(current_time):
                self._close_trade(ctx, "time", candle.close)
                ctx.state = SessionState.SESSION_CLOSED
                logger.info(f"[NY] {self.instrument} -> SESSION_CLOSED (time exit)")
                return None

        elif ctx.state == SessionState.SESSION_CLOSED:
            # Check if new session started (next day)
            if self._is_in_range_period(current_time):
                ctx.reset()
                ctx.state = SessionState.RECORDING_RANGE
                ctx.range_recorder = RangeRecorder(ctx.session, self.instrument)
                logger.info(f"[NY] {self.instrument} -> RECORDING_RANGE (new session)")
                if self._candle_in_range_window(candle):
                    ctx.range_recorder.add_candle(candle)

        return signal

    def _close_trade(
        self,
        ctx: SessionContext,
        reason: str,
        exit_price: float,
    ) -> None:
        """Close current trade and record it."""
        if ctx.current_trade is None:
            return

        ctx.current_trade.exit_time = self.get_current_time_ny()
        ctx.current_trade.exit_price = exit_price
        ctx.current_trade.exit_reason = reason

        ctx.trades.append(ctx.current_trade)

        exit_s = f"{exit_price:.2f}" if exit_price is not None else "unknown"
        logger.info(
            f"[NY] {self.instrument} Trade closed: "
            f"{ctx.current_trade.direction.value} exit={exit_s} reason={reason}"
        )

        if self.on_trade_closed:
            self.on_trade_closed(ctx.current_trade)

        # Reset breakout direction to allow re-entry ONLY after a stop-out
        # (matches backtest.py: target/time exits end the session).
        if reason == "stop" and ctx.breakout_detector:
            ctx.breakout_detector.reset_direction(ctx.current_trade.direction)

        ctx.current_trade = None
        ctx.current_trade_id = None

    def handle_order_failed(self, reason: str) -> None:
        """The entry order for the current signal did not result in a position.

        The signal moved the session to IN_TRADE before the order was sent;
        undo that. No trade is recorded (nothing was filled) and the session
        is closed for the day: Config A takes only the first breakout, and an
        order failure means the broker link is not trustworthy right now.
        """
        ctx = self.context
        if ctx.state == SessionState.IN_TRADE:
            ctx.current_trade = None
            ctx.current_trade_id = None
            ctx.state = SessionState.SESSION_CLOSED
            logger.warning(
                f"[NY] {self.instrument} -> SESSION_CLOSED (entry order failed: {reason})"
            )

    def adopt_trade(
        self,
        direction: Direction,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        trade_id: Optional[str] = None,
    ) -> None:
        """Take over a position that was already open when the bot started."""
        ctx = self.context
        ctx.current_trade = TradeRecord(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=self.get_current_time_ny(),
            trade_id=trade_id,
        )
        ctx.current_trade_id = trade_id
        ctx.state = SessionState.IN_TRADE
        logger.info(
            f"[NY] {self.instrument} -> IN_TRADE (adopted {direction.value} @ {entry_price:.2f})"
        )

    def handle_trade_exit(
        self,
        reason: str,
        exit_price: Optional[float],
    ) -> None:
        """Handle external notification of trade exit (from position monitor)."""
        ctx = self.context

        if ctx.state == SessionState.IN_TRADE and ctx.current_trade:
            self._close_trade(ctx, reason, exit_price)

            # Re-entry only after a stop-out, under the trade cap, and never
            # when first-breakout-only is active (Config A: one shot/session).
            if (
                reason == "stop"
                and ctx.can_trade
                and ctx.breakout_detector
                and not TradingConfig.FIRST_BREAKOUT_ONLY
            ):
                ctx.state = SessionState.WATCHING_FOR_BREAKOUT
                logger.info(
                    f"[NY] {self.instrument} -> WATCHING_FOR_BREAKOUT "
                    f"(re-entry after stop, trades: {ctx.trade_count}/{TradingConfig.MAX_TRADES_PER_SESSION})"
                )
            else:
                ctx.state = SessionState.SESSION_CLOSED
                logger.info(f"[NY] {self.instrument} -> SESSION_CLOSED ({reason} exit)")

    def _apply_config_a_rules(
        self,
        ctx: SessionContext,
        signal: BreakoutSignal,
        candle: Candle,
    ) -> Optional[BreakoutSignal]:
        """Apply the Config-A gates to a raw breakout signal.

        1. HTF filter: candle close on the trend side of EMA100.
        2. ATR stop: SL = entry -/+ ATR_STOP_MULT x ATR14, TP at RR_TARGET
           (falls back to the opening-range size during ATR warm-up, matching
           the backtest). Indicator-fetch failure rejects the signal — never
           trade with an unverified stop.

        A rejected first breakout closes the session (Config A trades only
        the session's first signal).
        """
        indicators = self._fetch_indicators(signal, candle)
        if indicators is None:
            logger.error(
                f"[NY] {self.instrument} SIGNAL REJECTED | "
                f"indicator fetch failed — cannot compute ATR stop / HTF filter"
            )
            self._reject_first_breakout(ctx, signal)
            return None
        atr_value, ema_value = indicators

        if TradingConfig.HTF_ALIGN:
            aligned = ema_value is not None and (
                (signal.direction == Direction.LONG and candle.close > ema_value)
                or (signal.direction == Direction.SHORT and candle.close < ema_value)
            )
            if not aligned:
                logger.info(
                    f"[NY] {self.instrument} SIGNAL FILTERED | "
                    f"{signal.direction.value} not HTF-aligned "
                    f"(close={candle.close:.2f} EMA{TradingConfig.HTF_EMA_PERIOD}="
                    f"{ema_value if ema_value is None else round(ema_value, 2)})"
                )
                self._reject_first_breakout(ctx, signal)
                return None

        if TradingConfig.STOP_MODE == "atr":
            range_size = ctx.opening_range.range_size if ctx.opening_range else None
            base = atr_value if atr_value else range_size
            if not base or base <= 0:
                logger.error(
                    f"[NY] {self.instrument} SIGNAL REJECTED | "
                    f"no ATR and no range size to place a stop"
                )
                self._reject_first_breakout(ctx, signal)
                return None
            stop_dist = TradingConfig.ATR_STOP_MULT * base
            entry = signal.entry_price
            if signal.direction == Direction.LONG:
                stop_loss = entry - stop_dist
                take_profit = entry + TradingConfig.RR_TARGET * stop_dist
            else:
                stop_loss = entry + stop_dist
                take_profit = entry - TradingConfig.RR_TARGET * stop_dist
            signal = BreakoutSignal(
                session=signal.session,
                instrument=signal.instrument,
                direction=signal.direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                opening_range=signal.opening_range,
                signal_time=signal.signal_time,
                breakout_candle=signal.breakout_candle,
                risk_per_unit=stop_dist,
                rr_target=TradingConfig.RR_TARGET,
            )
            logger.info(
                f"[NY] {self.instrument} ATR stop applied | "
                f"ATR={atr_value if atr_value is None else round(atr_value, 4)} "
                f"stop_dist={stop_dist:.4f} SL={stop_loss:.2f} TP={take_profit:.2f}"
            )

        return signal

    def _fetch_indicators(
        self,
        signal: BreakoutSignal,
        candle: Candle,
    ) -> Optional[tuple[Optional[float], Optional[float]]]:
        """Fetch recent RTH 5-min bars and compute (ATR14, EMA100).

        useRTH=True in ibkr_client keeps the bar stream identical in kind to
        the RTH dataset the strategy was validated on. Values are taken at
        the breakout candle's timestamp when present, else the latest bar
        (one bar of staleness at most; logged).
        """
        try:
            candles = self.client.get_candles(
                self.instrument,
                bar_size=TradingConfig.CANDLE_GRANULARITY,
                duration=TradingConfig.INDICATOR_CANDLE_DURATION,
                include_incomplete=False,
            )
        except Exception as e:
            logger.error(f"{self.instrument} indicator candle fetch failed: {e}")
            return None
        if not candles:
            return None

        atr_series = compute_atr_series(candles, period=TradingConfig.ATR_PERIOD)
        ema_series = compute_ema_series(candles, period=TradingConfig.HTF_EMA_PERIOD)
        atr_value, atr_exact = value_at_or_last(atr_series, signal.breakout_candle.time)
        ema_value, ema_exact = value_at_or_last(ema_series, signal.breakout_candle.time)
        if not atr_exact or not ema_exact:
            logger.warning(
                f"{self.instrument} indicator lookup fell back to latest value "
                f"(breakout candle {signal.breakout_candle.time} not in history; "
                f"ATR exact={atr_exact} EMA exact={ema_exact})"
            )
        return atr_value, ema_value

    def _reject_first_breakout(self, ctx: SessionContext, signal: BreakoutSignal) -> None:
        """A rejected signal under first-breakout-only closes the session."""
        if TradingConfig.FIRST_BREAKOUT_ONLY:
            ctx.state = SessionState.SESSION_CLOSED
            logger.info(
                f"[NY] {self.instrument} -> SESSION_CLOSED "
                f"(first breakout rejected, no trade today)"
            )
        else:
            ctx.breakout_detector.reset_direction(signal.direction)

    def set_trade_id(self, trade_id: str) -> None:
        """Set the IBKR order ID for the current trade."""
        ctx = self.context
        if ctx.current_trade:
            ctx.current_trade.trade_id = trade_id
            ctx.current_trade_id = trade_id

    def get_status(self) -> dict:
        """Get current status."""
        ctx = self.context
        return {
            "NY": {
                "state": ctx.state.value,
                "trades": ctx.trade_count,
                "can_trade": ctx.can_trade,
                "opening_range": str(ctx.opening_range) if ctx.opening_range else None,
                "current_trade": {
                    "direction": ctx.current_trade.direction.value,
                    "entry": ctx.current_trade.entry_price,
                    "sl": ctx.current_trade.stop_loss,
                    "tp": ctx.current_trade.take_profit,
                } if ctx.current_trade else None,
            }
        }
