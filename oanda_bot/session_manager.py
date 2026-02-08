"""Session state machine for MS-ORB trading bot."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
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
from strategy import (
    Direction,
    OpeningRange,
    BreakoutSignal,
    RangeRecorder,
    BreakoutDetector,
)
from oanda_client import Candle, OANDAClient, Position

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
    """Context data for a single instrument in a single session."""
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
    """Manages the state machine for a single instrument across all sessions."""

    def __init__(
        self,
        instrument: str,
        client: OANDAClient,
        on_signal: Optional[Callable[[BreakoutSignal], None]] = None,
        on_trade_closed: Optional[Callable[[TradeRecord], None]] = None,
    ):
        """Initialize session manager.

        Args:
            instrument: Instrument name (e.g., 'XAU_USD').
            client: OANDA client for data/trading.
            on_signal: Callback when a breakout signal is generated.
            on_trade_closed: Callback when a trade is closed.
        """
        self.instrument = instrument
        self.client = client
        self.on_signal = on_signal
        self.on_trade_closed = on_trade_closed

        # Create context for each session
        self.contexts: dict[SessionName, SessionContext] = {
            name: SessionContext(session=name, instrument=instrument)
            for name in SessionName
        }

        self._last_check_time: Optional[datetime] = None

    def get_current_time_ny(self) -> datetime:
        """Get current time in New York timezone."""
        return datetime.now(NY_TZ)

    def get_active_session(self) -> Optional[SessionName]:
        """Determine which session is currently active based on NY time.

        Returns:
            Active session name or None if between sessions.
        """
        now = self.get_current_time_ny()
        current_time = now.time()

        for session_name, config in SESSIONS.items():
            if self._is_time_in_session(current_time, config):
                return session_name

        return None

    def _is_time_in_session(self, current_time: time, config: SessionConfig) -> bool:
        """Check if current time falls within a session's active period.

        Handles sessions that span midnight (e.g., Tokyo).

        Args:
            current_time: Current time.
            config: Session configuration.

        Returns:
            True if within session period.
        """
        # Session spans midnight if range_start > session_close
        if config.range_start > config.session_close:
            # Tokyo-style: 18:00 -> 02:54 (next day)
            return current_time >= config.range_start or current_time <= config.session_close
        else:
            # Normal session: same day
            return config.range_start <= current_time <= config.session_close

    def _is_in_range_period(self, current_time: time, config: SessionConfig) -> bool:
        """Check if current time is in the range recording period.

        Args:
            current_time: Current time.
            config: Session configuration.

        Returns:
            True if in range period.
        """
        return config.range_start <= current_time < config.range_end

    def _is_in_entry_window(self, current_time: time, config: SessionConfig) -> bool:
        """Check if current time is in the entry window.

        Args:
            current_time: Current time.
            config: Session configuration.

        Returns:
            True if in entry window.
        """
        return config.range_end <= current_time <= config.entry_end

    def _is_time_to_close(self, current_time: time, config: SessionConfig) -> bool:
        """Check if it's time for auto-close.

        Args:
            current_time: Current time.
            config: Session configuration.

        Returns:
            True if should close positions.
        """
        # Handle midnight crossing
        if config.range_start > config.session_close:
            # For Tokyo: after entry_end and before/at session_close (next day)
            if current_time > config.entry_end and current_time <= time(23, 59, 59):
                # Still same day, past entry window
                return current_time >= config.session_close
            elif current_time <= config.session_close:
                # Next day
                return current_time >= config.session_close
            return False
        else:
            return current_time >= config.session_close

    def process_candle(self, candle: Candle) -> Optional[BreakoutSignal]:
        """Process a new candle and update state machines.

        Args:
            candle: Complete 5-minute candle.

        Returns:
            BreakoutSignal if one is generated, None otherwise.
        """
        if not candle.complete:
            return None

        now = self.get_current_time_ny()
        current_time = now.time()
        active_session = self.get_active_session()

        if active_session is None:
            # No active session - reset all contexts
            for ctx in self.contexts.values():
                if ctx.state != SessionState.SESSION_CLOSED:
                    ctx.state = SessionState.SESSION_CLOSED
            return None

        ctx = self.contexts[active_session]
        config = SESSIONS[active_session]

        # Skip if same candle already processed
        if ctx.last_candle_time and candle.time <= ctx.last_candle_time:
            return None

        ctx.last_candle_time = candle.time

        # State machine logic
        signal = self._update_state(ctx, config, candle, current_time)

        return signal

    def _update_state(
        self,
        ctx: SessionContext,
        config: SessionConfig,
        candle: Candle,
        current_time: time,
    ) -> Optional[BreakoutSignal]:
        """Update state machine for a session context.

        Args:
            ctx: Session context.
            config: Session configuration.
            candle: Current candle.
            current_time: Current time.

        Returns:
            BreakoutSignal if generated.
        """
        signal = None

        if ctx.state == SessionState.WAITING_FOR_SESSION:
            # Check if range period started
            if self._is_in_range_period(current_time, config):
                ctx.state = SessionState.RECORDING_RANGE
                ctx.range_recorder = RangeRecorder(ctx.session, self.instrument)
                logger.info(
                    f"[{ctx.session.value}] {self.instrument} -> RECORDING_RANGE"
                )
                ctx.range_recorder.add_candle(candle)

        elif ctx.state == SessionState.RECORDING_RANGE:
            if self._is_in_range_period(current_time, config):
                # Still in range period - add candle
                ctx.range_recorder.add_candle(candle)
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
                        f"[{ctx.session.value}] {self.instrument} -> WATCHING_FOR_BREAKOUT "
                        f"(Range: H={ctx.opening_range.high:.5f} L={ctx.opening_range.low:.5f})"
                    )
                else:
                    logger.warning(
                        f"[{ctx.session.value}] {self.instrument}: Failed to finalize range"
                    )
                    ctx.state = SessionState.SESSION_CLOSED

        elif ctx.state == SessionState.WATCHING_FOR_BREAKOUT:
            # Check for time close
            if self._is_time_to_close(current_time, config):
                ctx.state = SessionState.SESSION_CLOSED
                logger.info(
                    f"[{ctx.session.value}] {self.instrument} -> SESSION_CLOSED (time)"
                )
                return None

            # Check if still in entry window
            if not self._is_in_entry_window(current_time, config):
                # Past entry window but before session close
                # Keep watching for position management
                pass

            # Check for breakout if we can still trade
            if ctx.can_trade and self._is_in_entry_window(current_time, config):
                signal = ctx.breakout_detector.check_breakout(candle)
                if signal:
                    # Create trade record
                    ctx.current_trade = TradeRecord(
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        entry_time=self.get_current_time_ny(),
                    )
                    ctx.state = SessionState.IN_TRADE
                    logger.info(
                        f"[{ctx.session.value}] {self.instrument} -> IN_TRADE "
                        f"({signal.direction.value} @ {signal.entry_price:.5f})"
                    )

                    if self.on_signal:
                        self.on_signal(signal)

        elif ctx.state == SessionState.IN_TRADE:
            # Check for time close
            if self._is_time_to_close(current_time, config):
                self._close_trade(ctx, "time", candle.close)
                ctx.state = SessionState.SESSION_CLOSED
                logger.info(
                    f"[{ctx.session.value}] {self.instrument} -> SESSION_CLOSED (time exit)"
                )
                return None

            # Position monitoring handled by main loop checking OANDA positions

        elif ctx.state == SessionState.SESSION_CLOSED:
            # Check if new session started
            if self._is_in_range_period(current_time, config):
                ctx.reset()
                ctx.state = SessionState.RECORDING_RANGE
                ctx.range_recorder = RangeRecorder(ctx.session, self.instrument)
                logger.info(
                    f"[{ctx.session.value}] {self.instrument} -> RECORDING_RANGE (new session)"
                )
                ctx.range_recorder.add_candle(candle)

        return signal

    def _close_trade(
        self,
        ctx: SessionContext,
        reason: str,
        exit_price: float,
    ) -> None:
        """Close current trade and record it.

        Args:
            ctx: Session context.
            reason: Exit reason ('stop', 'target', 'time').
            exit_price: Exit price.
        """
        if ctx.current_trade is None:
            return

        ctx.current_trade.exit_time = self.get_current_time_ny()
        ctx.current_trade.exit_price = exit_price
        ctx.current_trade.exit_reason = reason

        ctx.trades.append(ctx.current_trade)

        logger.info(
            f"[{ctx.session.value}] {self.instrument} Trade closed: "
            f"{ctx.current_trade.direction.value} exit={exit_price:.5f} reason={reason}"
        )

        if self.on_trade_closed:
            self.on_trade_closed(ctx.current_trade)

        # Reset breakout direction to allow re-entry
        if ctx.breakout_detector:
            ctx.breakout_detector.reset_direction(ctx.current_trade.direction)

        ctx.current_trade = None
        ctx.current_trade_id = None

    def handle_trade_exit(
        self,
        session: SessionName,
        reason: str,
        exit_price: float,
    ) -> None:
        """Handle external notification of trade exit (from position monitor).

        Args:
            session: Session name.
            reason: Exit reason.
            exit_price: Exit price.
        """
        ctx = self.contexts[session]

        if ctx.state == SessionState.IN_TRADE and ctx.current_trade:
            self._close_trade(ctx, reason, exit_price)

            # Move to watching if we can still trade
            if ctx.can_trade and ctx.breakout_detector:
                ctx.state = SessionState.WATCHING_FOR_BREAKOUT
                logger.info(
                    f"[{session.value}] {self.instrument} -> WATCHING_FOR_BREAKOUT "
                    f"(re-entry allowed, trades: {ctx.trade_count}/{TradingConfig.MAX_TRADES_PER_SESSION})"
                )
            else:
                ctx.state = SessionState.SESSION_CLOSED
                logger.info(
                    f"[{session.value}] {self.instrument} -> SESSION_CLOSED "
                    f"(max trades reached)"
                )

    def set_trade_id(self, session: SessionName, trade_id: str) -> None:
        """Set the OANDA trade ID for the current trade.

        Args:
            session: Session name.
            trade_id: OANDA trade ID.
        """
        ctx = self.contexts[session]
        if ctx.current_trade:
            ctx.current_trade.trade_id = trade_id
            ctx.current_trade_id = trade_id

    def get_context(self, session: SessionName) -> SessionContext:
        """Get context for a session.

        Args:
            session: Session name.

        Returns:
            Session context.
        """
        return self.contexts[session]

    def get_all_active_contexts(self) -> list[SessionContext]:
        """Get all contexts that are actively trading or watching.

        Returns:
            List of active contexts.
        """
        return [
            ctx for ctx in self.contexts.values()
            if ctx.state in (
                SessionState.RECORDING_RANGE,
                SessionState.WATCHING_FOR_BREAKOUT,
                SessionState.IN_TRADE,
            )
        ]

    def get_status(self) -> dict:
        """Get current status of all sessions.

        Returns:
            Status dictionary.
        """
        return {
            session.value: {
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
            for session, ctx in self.contexts.items()
        }


if __name__ == "__main__":
    # Test session timing
    logging.basicConfig(level=logging.DEBUG)

    # Test time detection
    test_times = [
        time(18, 0),   # Tokyo range start
        time(18, 15),  # Tokyo range middle
        time(18, 30),  # Tokyo entry start
        time(21, 0),   # Tokyo entry middle
        time(2, 54),   # Tokyo close
        time(3, 0),    # London range start
        time(3, 30),   # London entry start
        time(8, 25),   # London close
        time(9, 30),   # NY range start
        time(10, 0),   # NY entry start
        time(15, 55),  # NY close
    ]

    print("Session Time Detection Test:")
    print("-" * 50)

    for test_time in test_times:
        for session_name, config in SESSIONS.items():
            # Create a mock manager to use its methods
            class MockManager:
                def _is_time_in_session(self, current_time, config):
                    if config.range_start > config.session_close:
                        return current_time >= config.range_start or current_time <= config.session_close
                    else:
                        return config.range_start <= current_time <= config.session_close

                def _is_in_range_period(self, current_time, config):
                    return config.range_start <= current_time < config.range_end

                def _is_in_entry_window(self, current_time, config):
                    return config.range_end <= current_time <= config.entry_end

            mgr = MockManager()
            if mgr._is_time_in_session(test_time, config):
                in_range = mgr._is_in_range_period(test_time, config)
                in_entry = mgr._is_in_entry_window(test_time, config)
                phase = "RANGE" if in_range else "ENTRY" if in_entry else "MONITOR"
                print(f"{test_time} -> {session_name.value}: {phase}")
                break
        else:
            print(f"{test_time} -> No active session")
