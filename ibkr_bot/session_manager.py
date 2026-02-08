"""Session state machine for MS-ORB IBKR trading bot.

NY session only for TSLA.
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
                ctx.range_recorder.add_candle(candle)

        elif ctx.state == SessionState.RECORDING_RANGE:
            if self._is_in_range_period(current_time):
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

        logger.info(
            f"[NY] {self.instrument} Trade closed: "
            f"{ctx.current_trade.direction.value} exit={exit_price:.2f} reason={reason}"
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
        reason: str,
        exit_price: float,
    ) -> None:
        """Handle external notification of trade exit (from position monitor)."""
        ctx = self.context

        if ctx.state == SessionState.IN_TRADE and ctx.current_trade:
            self._close_trade(ctx, reason, exit_price)

            # Move to watching if we can still trade
            if ctx.can_trade and ctx.breakout_detector:
                ctx.state = SessionState.WATCHING_FOR_BREAKOUT
                logger.info(
                    f"[NY] {self.instrument} -> WATCHING_FOR_BREAKOUT "
                    f"(re-entry allowed, trades: {ctx.trade_count}/{TradingConfig.MAX_TRADES_PER_SESSION})"
                )
            else:
                ctx.state = SessionState.SESSION_CLOSED
                logger.info(f"[NY] {self.instrument} -> SESSION_CLOSED (max trades reached)")

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
