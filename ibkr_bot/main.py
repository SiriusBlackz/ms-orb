#!/usr/bin/env python3
"""MS-ORB IBKR Trading Bot - Main entry point.

Live trading bot for TSLA via Interactive Brokers (ib_insync).
Implements the Multi-Session Opening Range Breakout strategy.
NY session only.
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytz

from config import (
    INSTRUMENT,
    LOG_DIR,
    LOG_LEVEL,
    SESSIONS,
    SessionName,
    TradingConfig,
    TIMEZONE,
)
from ibkr_client import IBKRClient, Candle
from session_manager import SessionManager, SessionState
from strategy import BreakoutSignal, Direction, calculate_position_size
from trade_logger import TradeLogger, calculate_pnl, calculate_rr_achieved

# Set up logging
LOG_PATH = Path(LOG_DIR)
LOG_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH / "ibkr_bot.log"),
    ],
)

logger = logging.getLogger(__name__)

# Timezone
NY_TZ = pytz.timezone(TIMEZONE)


class TradingBot:
    """Main IBKR trading bot orchestrator."""

    def __init__(self, dry_run: bool = False):
        """Initialize trading bot.

        Args:
            dry_run: If True, don't place real orders.
        """
        self.instrument = INSTRUMENT
        self.dry_run = dry_run
        self._running = False
        self._shutdown_requested = False

        # Validate config
        errors = TradingConfig.validate()
        if errors:
            for error in errors:
                logger.error(f"Config error: {error}")
            raise ValueError("Configuration validation failed")

        # Initialize components
        self.client = IBKRClient()
        self.trade_logger = TradeLogger()

        # Session manager for TSLA
        self.session_manager = SessionManager(
            instrument=self.instrument,
            client=self.client,
            on_signal=self._on_signal,
            on_trade_closed=self._on_trade_closed,
        )

        # Track open position
        self._open_position: Optional[dict] = None
        self._last_candle_time: Optional[datetime] = None

        logger.info(f"IBKR Trading bot initialized for {self.instrument}")
        logger.info(f"Dry run mode: {self.dry_run}")

    def _get_current_time_ny(self) -> datetime:
        """Get current time in New York timezone."""
        return datetime.now(NY_TZ)

    def _on_signal(self, signal: BreakoutSignal) -> None:
        """Handle a breakout signal."""
        logger.info(
            f"SIGNAL | NY {self.instrument} | "
            f"{signal.direction.value} @ {signal.entry_price:.2f} "
            f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f}"
        )

        if self.dry_run:
            logger.info("DRY RUN - Order not placed")
            return

        # Get account balance for position sizing
        balance = self.client.get_account_balance()
        if balance <= 0:
            logger.error("Failed to get account balance, skipping trade")
            return

        # Calculate position size
        units = calculate_position_size(
            account_balance=balance,
            risk_percent=TradingConfig.RISK_PER_TRADE,
            entry_price=signal.entry_price,
            stop_price=signal.stop_loss,
        )

        if units <= 0:
            logger.warning("Position size is 0, skipping trade")
            return

        # Adjust units for direction
        if signal.direction == Direction.SHORT:
            units = -units

        # Place bracket order
        result = self.client.place_market_order(
            instrument=self.instrument,
            units=units,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        if result.success:
            logger.info(
                f"ORDER FILLED | {self.instrument} | {result.units} shares @ {result.fill_price} "
                f"| Order ID: {result.order_id}"
            )

            # Log to database
            db_id = self.trade_logger.log_entry(
                instrument=self.instrument,
                session=signal.session,
                direction=signal.direction,
                entry_price=result.fill_price or signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                units=abs(result.units or units),
                trade_id=result.trade_id,
                opening_range_high=signal.opening_range.high,
                opening_range_low=signal.opening_range.low,
            )

            # Track position
            self._open_position = {
                "trade_id": result.trade_id,
                "db_id": db_id,
                "direction": signal.direction,
                "entry_price": result.fill_price or signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "units": abs(result.units or units),
            }

            # Update session manager
            if result.trade_id:
                self.session_manager.set_trade_id(result.trade_id)

        else:
            logger.error(f"ORDER FAILED | {self.instrument} | {result.error}")

    def _on_trade_closed(self, record) -> None:
        """Handle trade closed notification from session manager."""
        logger.info(
            f"TRADE CLOSED (session) | {self.instrument} | "
            f"Exit: {record.exit_price} Reason: {record.exit_reason}"
        )

    def _check_position(self) -> None:
        """Check and update open position from IBKR."""
        if self._open_position is None:
            return

        pos = self.client.get_position(self.instrument)

        if pos is None or pos.units == 0:
            # Position was closed (by SL or TP)
            logger.info(f"Position closed detected: {self.instrument}")

            pos_info = self._open_position
            exit_price = None
            exit_reason = "unknown"

            # Determine exit reason from price
            price = self.client.get_price(self.instrument)
            if price:
                current_price = price.mid

                if pos_info["direction"] == Direction.LONG:
                    if current_price <= pos_info["stop_loss"]:
                        exit_reason = "stop"
                        exit_price = pos_info["stop_loss"]
                    elif current_price >= pos_info["take_profit"]:
                        exit_reason = "target"
                        exit_price = pos_info["take_profit"]
                    else:
                        exit_reason = "unknown"
                        exit_price = current_price
                else:
                    if current_price >= pos_info["stop_loss"]:
                        exit_reason = "stop"
                        exit_price = pos_info["stop_loss"]
                    elif current_price <= pos_info["take_profit"]:
                        exit_reason = "target"
                        exit_price = pos_info["take_profit"]
                    else:
                        exit_reason = "unknown"
                        exit_price = current_price

            if exit_price:
                pnl = calculate_pnl(
                    pos_info["direction"].value,
                    pos_info["entry_price"],
                    exit_price,
                    pos_info["units"],
                )
                rr = calculate_rr_achieved(
                    pos_info["direction"].value,
                    pos_info["entry_price"],
                    exit_price,
                    pos_info["stop_loss"],
                )

                self.trade_logger.log_exit(
                    row_id=pos_info["db_id"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    rr_achieved=rr,
                )

                logger.info(
                    f"EXIT LOGGED | {self.instrument} | "
                    f"Price: {exit_price:.2f} Reason: {exit_reason} "
                    f"PnL: ${pnl:.2f} RR: {rr:.2f}"
                )
            else:
                logger.warning(
                    f"Could not determine exit price for {self.instrument}, "
                    f"using entry price as fallback"
                )
                exit_price = pos_info["entry_price"]
                exit_reason = "unknown"

            # Always notify session manager so it can transition
            # back to WATCHING_FOR_BREAKOUT for re-entry
            self.session_manager.handle_trade_exit(exit_reason, exit_price)

            self._open_position = None

    def _check_time_exit(self) -> None:
        """Check if position needs to be closed due to session time exit."""
        if self._open_position is None:
            return

        now = self._get_current_time_ny()
        current_time = now.time()
        config = SESSIONS[SessionName.NY]

        if current_time >= config.session_close:
            logger.info(
                f"TIME EXIT | {self.instrument} | "
                f"Session NY closing at {current_time}"
            )

            if not self.dry_run:
                result = self.client.close_position(self.instrument)
                if result.success:
                    exit_price = result.fill_price or 0
                    pos_info = self._open_position

                    pnl = calculate_pnl(
                        pos_info["direction"].value,
                        pos_info["entry_price"],
                        exit_price,
                        pos_info["units"],
                    )
                    rr = calculate_rr_achieved(
                        pos_info["direction"].value,
                        pos_info["entry_price"],
                        exit_price,
                        pos_info["stop_loss"],
                    )

                    self.trade_logger.log_exit(
                        row_id=pos_info["db_id"],
                        exit_price=exit_price,
                        exit_reason="time",
                        pnl=pnl,
                        rr_achieved=rr,
                    )

                    logger.info(
                        f"TIME EXIT COMPLETE | {self.instrument} | "
                        f"Exit: {exit_price:.2f} PnL: ${pnl:.2f}"
                    )

                    self.session_manager.handle_trade_exit("time", exit_price)
                    self._open_position = None
                else:
                    logger.error(f"Failed to close position: {result.error}")

    def _process_candles(self) -> None:
        """Fetch and process latest candles for TSLA."""
        try:
            candle = self.client.get_latest_complete_candle(
                self.instrument,
                TradingConfig.CANDLE_GRANULARITY,
            )

            if not candle:
                return

            # Skip if already processed
            if self._last_candle_time and candle.time <= self._last_candle_time:
                return

            self._last_candle_time = candle.time

            logger.debug(
                f"Candle | {self.instrument} | "
                f"O={candle.open:.2f} H={candle.high:.2f} "
                f"L={candle.low:.2f} C={candle.close:.2f}"
            )

            # Process through session manager
            self.session_manager.process_candle(candle)

        except Exception as e:
            logger.error(f"Error processing candles for {self.instrument}: {e}")

    def _log_status(self) -> None:
        """Log current bot status."""
        now = self._get_current_time_ny()
        logger.info(f"=== Status at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ===")

        status = self.session_manager.get_status()
        for session_name, session_status in status.items():
            if session_status["state"] != "WAITING_FOR_SESSION":
                logger.info(
                    f"{self.instrument} | {session_name} | "
                    f"State: {session_status['state']} | "
                    f"Trades: {session_status['trades']}"
                )

        if self._open_position:
            pos = self._open_position
            logger.info(
                f"{self.instrument} | POSITION | "
                f"{pos['direction'].value} {pos['units']} shares @ {pos['entry_price']:.2f}"
            )

    def run(self) -> None:
        """Run the main trading loop."""
        logger.info("=" * 60)
        logger.info("MS-ORB IBKR Trading Bot Starting")
        logger.info(f"Instrument: {self.instrument}")
        logger.info(f"Connection: {TradingConfig.IBKR_HOST}:{TradingConfig.IBKR_PORT}")
        logger.info(f"Risk per trade: {TradingConfig.RISK_PER_TRADE:.1%}")
        logger.info(f"RR Target: {TradingConfig.RR_TARGET}")
        logger.info(f"Max trades per session: {TradingConfig.MAX_TRADES_PER_SESSION}")
        logger.info("=" * 60)

        # Connect to IBKR
        if not self.client.connect():
            logger.error("Failed to connect to IBKR")
            return

        if not self.client.test_connection():
            logger.error("IBKR connection test failed")
            return

        # Check for existing position
        pos = self.client.get_position(self.instrument)
        if pos:
            logger.warning(
                f"Existing position found: {self.instrument} | "
                f"{pos.units} shares @ {pos.avg_price}"
            )

        self._running = True
        last_status_log = datetime.now()
        status_interval = timedelta(minutes=5)

        try:
            while self._running and not self._shutdown_requested:
                loop_start = time.time()

                try:
                    # Check connection
                    if not self.client.is_connected():
                        logger.warning("IBKR disconnected, reconnecting...")
                        if not self.client.connect():
                            logger.error("Reconnection failed, waiting...")
                            time.sleep(30)
                            continue

                    # Process candles
                    self._process_candles()

                    # Check position
                    self._check_position()

                    # Check time exit
                    self._check_time_exit()

                    # Log status periodically
                    if datetime.now() - last_status_log > status_interval:
                        self._log_status()
                        last_status_log = datetime.now()

                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)

                # Sleep using ib_insync event loop
                elapsed = time.time() - loop_start
                sleep_time = max(0, TradingConfig.PRICE_POLL_INTERVAL - elapsed)
                if sleep_time > 0:
                    self.client.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shut down the bot."""
        logger.info("Shutting down IBKR trading bot...")
        self._running = False

        # Log final status
        self._log_status()

        # Get trade stats
        stats = self.trade_logger.get_trade_stats()
        logger.info(f"Trade Statistics: {stats}")

        # Disconnect
        self.client.disconnect()

        logger.info("IBKR trading bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MS-ORB IBKR Trading Bot (TSLA)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without placing real orders",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="IBKR port override (7497=TWS paper, 7496=TWS live)",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.port:
        TradingConfig.IBKR_PORT = args.port

    # Set up signal handlers
    bot = TradingBot(dry_run=args.dry_run)

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        bot._shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the bot
    bot.run()


if __name__ == "__main__":
    main()
