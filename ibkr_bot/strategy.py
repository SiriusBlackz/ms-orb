"""MS-ORB strategy logic for range detection and breakout signals."""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from config import SessionName, TradingConfig
from ibkr_client import Candle

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class OpeningRange:
    """Opening range data for a session."""
    session: SessionName
    instrument: str
    high: float
    low: float
    start_time: datetime
    end_time: datetime
    candles_used: int = 0

    @property
    def range_size(self) -> float:
        """Size of the opening range."""
        return self.high - self.low

    def __str__(self) -> str:
        return (
            f"OpeningRange({self.session.value} {self.instrument}: "
            f"H={self.high:.2f} L={self.low:.2f} Size={self.range_size:.2f})"
        )


@dataclass
class BreakoutSignal:
    """Signal generated when price breaks the opening range."""
    session: SessionName
    instrument: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    opening_range: OpeningRange
    signal_time: datetime
    breakout_candle: Candle
    risk_per_unit: float
    rr_target: float

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate actual risk-reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0

    def __str__(self) -> str:
        return (
            f"BreakoutSignal({self.direction.value} {self.instrument} @ {self.entry_price:.2f}, "
            f"SL={self.stop_loss:.2f}, TP={self.take_profit:.2f}, RR={self.risk_reward_ratio:.2f})"
        )


class RangeRecorder:
    """Records and tracks the opening range for a session."""

    def __init__(self, session: SessionName, instrument: str):
        self.session = session
        self.instrument = instrument
        self._high: Optional[float] = None
        self._low: Optional[float] = None
        self._start_time: Optional[datetime] = None
        self._candles: list[Candle] = []
        self._complete = False

    def reset(self) -> None:
        """Reset the range recorder for a new session."""
        self._high = None
        self._low = None
        self._start_time = None
        self._candles = []
        self._complete = False
        logger.debug(f"Range recorder reset for {self.session.value} {self.instrument}")

    def add_candle(self, candle: Candle) -> None:
        """Add a candle to the range calculation."""
        if self._complete:
            return

        if not candle.complete:
            logger.debug(f"Skipping incomplete candle: {candle.time}")
            return

        if self._start_time is None:
            self._start_time = candle.time

        if self._high is None or candle.high > self._high:
            self._high = candle.high

        if self._low is None or candle.low < self._low:
            self._low = candle.low

        self._candles.append(candle)
        logger.debug(
            f"Range update {self.session.value} {self.instrument}: "
            f"H={self._high:.2f} L={self._low:.2f} (candles: {len(self._candles)})"
        )

    def finalize(self) -> Optional[OpeningRange]:
        """Finalize the opening range after the range period ends."""
        if self._high is None or self._low is None:
            logger.warning(f"Cannot finalize range: no candles recorded for {self.session.value}")
            return None

        if len(self._candles) < 1:
            logger.warning(f"Cannot finalize range: insufficient candles for {self.session.value}")
            return None

        self._complete = True
        end_time = self._candles[-1].time if self._candles else self._start_time

        opening_range = OpeningRange(
            session=self.session,
            instrument=self.instrument,
            high=self._high,
            low=self._low,
            start_time=self._start_time,
            end_time=end_time,
            candles_used=len(self._candles),
        )

        logger.info(f"Opening range finalized: {opening_range}")
        return opening_range

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def current_high(self) -> Optional[float]:
        return self._high

    @property
    def current_low(self) -> Optional[float]:
        return self._low


class BreakoutDetector:
    """Detects breakout signals from completed candles."""

    def __init__(
        self,
        opening_range: OpeningRange,
        rr_target: float = None,
    ):
        self.opening_range = opening_range
        self.rr_target = rr_target or TradingConfig.RR_TARGET

        self._breakout_up_triggered = False
        self._breakout_down_triggered = False
        self._last_processed_candle: Optional[datetime] = None

    def reset(self) -> None:
        """Reset breakout detector state."""
        self._breakout_up_triggered = False
        self._breakout_down_triggered = False
        self._last_processed_candle = None

    def check_breakout(self, candle: Candle) -> Optional[BreakoutSignal]:
        """Check if a candle represents a breakout.

        A breakout occurs when:
        - LONG: 5-min candle CLOSES above the range high
        - SHORT: 5-min candle CLOSES below the range low
        """
        if not candle.complete:
            return None

        if self._last_processed_candle and candle.time <= self._last_processed_candle:
            return None

        self._last_processed_candle = candle.time

        # Check for upside breakout (LONG)
        if candle.close > self.opening_range.high:
            if not self._breakout_up_triggered:
                signal = self._create_signal(candle, Direction.LONG)
                if signal:
                    self._breakout_up_triggered = True
                    logger.info(f"Upside breakout detected: {signal}")
                    return signal

        # Check for downside breakout (SHORT)
        if candle.close < self.opening_range.low:
            if not self._breakout_down_triggered:
                signal = self._create_signal(candle, Direction.SHORT)
                if signal:
                    self._breakout_down_triggered = True
                    logger.info(f"Downside breakout detected: {signal}")
                    return signal

        return None

    def _create_signal(
        self,
        candle: Candle,
        direction: Direction,
    ) -> Optional[BreakoutSignal]:
        """Create a breakout signal."""
        entry_price = candle.close

        if direction == Direction.LONG:
            stop_loss = candle.low
            risk = entry_price - stop_loss

            if risk <= 0:
                logger.warning(f"Invalid long signal: entry={entry_price}, stop={stop_loss}")
                return None

            take_profit = entry_price + (risk * self.rr_target)

        else:  # SHORT
            stop_loss = candle.high
            risk = stop_loss - entry_price

            if risk <= 0:
                logger.warning(f"Invalid short signal: entry={entry_price}, stop={stop_loss}")
                return None

            take_profit = entry_price - (risk * self.rr_target)

        return BreakoutSignal(
            session=self.opening_range.session,
            instrument=self.opening_range.instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opening_range=self.opening_range,
            signal_time=candle.time,
            breakout_candle=candle,
            risk_per_unit=abs(entry_price - stop_loss),
            rr_target=self.rr_target,
        )

    def can_reenter(self, direction: Direction) -> bool:
        """Check if re-entry is allowed in a direction.

        Re-entry is allowed if:
        - Previously stopped out and price re-breaks in same direction
        - Price breaks in the opposite direction (first time)
        """
        if direction == Direction.LONG:
            return not self._breakout_up_triggered or self._breakout_down_triggered
        else:
            return not self._breakout_down_triggered or self._breakout_up_triggered

    def reset_direction(self, direction: Direction) -> None:
        """Reset a direction's trigger after a stop out."""
        if direction == Direction.LONG:
            self._breakout_up_triggered = False
            logger.debug("Long breakout trigger reset (allow re-entry)")
        else:
            self._breakout_down_triggered = False
            logger.debug("Short breakout trigger reset (allow re-entry)")


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
) -> int:
    """Calculate position size based on risk.

    Args:
        account_balance: Current account balance.
        risk_percent: Risk as decimal (0.01 = 1%).
        entry_price: Entry price.
        stop_price: Stop loss price.

    Returns:
        Number of shares to trade.
    """
    risk_amount = account_balance * risk_percent
    risk_per_unit = abs(entry_price - stop_price)

    if risk_per_unit <= 0:
        return 0

    units = int(risk_amount / risk_per_unit)
    return max(units, 0)
