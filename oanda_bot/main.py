#!/usr/bin/env python3
"""MS-ORB Trading Bot - Main entry point.

Live trading bot for XAUUSD and NAS100 via OANDA v20 REST API.
Implements the Multi-Session Opening Range Breakout strategy.
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
    INSTRUMENTS,
    LOG_DIR,
    LOG_LEVEL,
    SESSIONS,
    SessionName,
    TradingConfig,
    TIMEZONE,
)
from oanda_client import OANDAClient, Candle
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
        logging.FileHandler(LOG_PATH / "bot.log"),
    ],
)

logger = logging.getLogger(__name__)

# Timezone
NY_TZ = pytz.timezone(TIMEZONE)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(
        self,
        instruments: list[str] = None,
        dry_run: bool = False,
    ):
        """Initialize trading bot.

        Args:
            instruments: List of instruments to trade. Defaults to config INSTRUMENTS.
            dry_run: If True, don't place real orders.
        """
        self.instruments = instruments or INSTRUMENTS
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
        self.client = OANDAClient()
        self.trade_logger = TradeLogger()

        # Session managers per instrument
        self.session_managers: dict[str, SessionManager] = {}
        for instrument in self.instruments:
            self.session_managers[instrument] = SessionManager(
                instrument=instrument,
                client=self.client,
                on_signal=lambda sig, inst=instrument: self._on_signal(sig, inst),
                on_trade_closed=lambda rec, inst=instrument: self._on_trade_closed(rec, inst),
            )

        # Track open positions
        self._open_positions: dict[str, dict] = {}  # instrument -> position info
        self._db_trade_ids: dict[str, int] = {}  # oanda_trade_id -> db_row_id
        self._last_candle_times: dict[str, datetime] = {}  # instrument -> last candle time

        logger.info(f"Trading bot initialized for instruments: {self.instruments}")
        logger.info(f"Dry run mode: {self.dry_run}")

    def _get_current_time_ny(self) -> datetime:
        """Get current time in New York timezone."""
        return datetime.now(NY_TZ)

    def _on_signal(self, signal: BreakoutSignal, instrument: str) -> None:
        """Handle a breakout signal.

        Args:
            signal: The breakout signal.
            instrument: Instrument name.
        """
        logger.info(
            f"SIGNAL | {signal.session.value} {instrument} | "
            f"{signal.direction.value} @ {signal.entry_price:.5f} "
            f"SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f}"
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
            logger.warning(f"Position size is 0, skipping trade")
            return

        # Adjust units for direction
        if signal.direction == Direction.SHORT:
            units = -units

        # Place order
        result = self.client.place_limit_order(
            instrument=instrument,
            units=units,
            price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        if result.success:
            if result.fill_price:
                logger.info(
                    f"LIMIT ORDER FILLED | {instrument} | {result.units} units @ {result.fill_price} "
                    f"| Trade ID: {result.trade_id}"
                )
            else:
                logger.info(
                    f"LIMIT ORDER PLACED | {instrument} | {units} units @ {signal.entry_price:.5f} "
                    f"| Order ID: {result.order_id}"
                )

            # Log to database
            db_id = self.trade_logger.log_entry(
                instrument=instrument,
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
            self._open_positions[instrument] = {
                "trade_id": result.trade_id,
                "db_id": db_id,
                "session": signal.session,
                "direction": signal.direction,
                "entry_price": result.fill_price or signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "units": abs(result.units or units),
            }

            if result.trade_id:
                self._db_trade_ids[result.trade_id] = db_id

            # Update session manager
            sm = self.session_managers[instrument]
            sm.set_trade_id(signal.session, result.trade_id)

        else:
            logger.error(f"ORDER FAILED | {instrument} | {result.error}")

    def _on_trade_closed(self, record, instrument: str) -> None:
        """Handle trade closed notification from session manager.

        Args:
            record: Trade record.
            instrument: Instrument name.
        """
        logger.info(
            f"TRADE CLOSED (session) | {instrument} | "
            f"Exit: {record.exit_price} Reason: {record.exit_reason}"
        )

    def _check_positions(self) -> None:
        """Check and update open positions from OANDA."""
        for instrument in self.instruments:
            if instrument not in self._open_positions:
                continue

            pos_info = self._open_positions[instrument]
            trade_id = pos_info["trade_id"]

            if not trade_id:
                continue

            # Check if trade is still open
            open_trades = self.client.get_open_trades(instrument)
            trade_ids = [t["id"] for t in open_trades]

            if trade_id not in trade_ids:
                # Trade was closed
                logger.info(f"Position closed detected: {instrument} trade_id={trade_id}")

                # Get final price from position history if available
                exit_price = None
                exit_reason = "unknown"

                # Try to determine exit reason from price movement
                pos = pos_info
                price = self.client.get_price(instrument)
                if price:
                    current_price = price.mid

                    if pos["direction"] == Direction.LONG:
                        if current_price <= pos["stop_loss"]:
                            exit_reason = "stop"
                            exit_price = pos["stop_loss"]
                        elif current_price >= pos["take_profit"]:
                            exit_reason = "target"
                            exit_price = pos["take_profit"]
                        else:
                            exit_reason = "unknown"
                            exit_price = current_price
                    else:
                        if current_price >= pos["stop_loss"]:
                            exit_reason = "stop"
                            exit_price = pos["stop_loss"]
                        elif current_price <= pos["take_profit"]:
                            exit_reason = "target"
                            exit_price = pos["take_profit"]
                        else:
                            exit_reason = "unknown"
                            exit_price = current_price

                if exit_price:
                    # Calculate PnL
                    pnl = calculate_pnl(
                        pos["direction"].value,
                        pos["entry_price"],
                        exit_price,
                        pos["units"],
                    )
                    rr = calculate_rr_achieved(
                        pos["direction"].value,
                        pos["entry_price"],
                        exit_price,
                        pos["stop_loss"],
                    )

                    # Log exit
                    self.trade_logger.log_exit(
                        row_id=pos["db_id"],
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl=pnl,
                        rr_achieved=rr,
                    )

                    logger.info(
                        f"EXIT LOGGED | {instrument} | "
                        f"Price: {exit_price:.5f} Reason: {exit_reason} "
                        f"PnL: {pnl:.2f} RR: {rr:.2f}"
                    )
                else:
                    logger.warning(
                        f"Could not determine exit price for {instrument}, "
                        f"using entry price as fallback"
                    )
                    exit_price = pos["entry_price"]
                    exit_reason = "unknown"

                # Always notify session manager so it can transition
                # back to WATCHING_FOR_BREAKOUT for re-entry
                sm = self.session_managers[instrument]
                sm.handle_trade_exit(pos["session"], exit_reason, exit_price)

                # Clean up
                del self._open_positions[instrument]
                if trade_id in self._db_trade_ids:
                    del self._db_trade_ids[trade_id]

    def _check_time_exits(self) -> None:
        """Check if any positions need to be closed due to session time exit."""
        now = self._get_current_time_ny()
        current_time = now.time()

        for instrument in list(self._open_positions.keys()):
            pos_info = self._open_positions[instrument]
            session = pos_info["session"]
            config = SESSIONS[session]

            # Check if it's time to close
            should_close = False

            # Handle midnight crossing for Tokyo
            if config.range_start > config.session_close:
                # Tokyo style
                if current_time > config.entry_end:
                    # After entry end same day
                    should_close = current_time >= config.session_close
                elif current_time <= config.session_close:
                    # Next day before session close
                    should_close = current_time >= config.session_close
            else:
                should_close = current_time >= config.session_close

            if should_close:
                logger.info(
                    f"TIME EXIT | {instrument} | "
                    f"Session {session.value} closing at {current_time}"
                )

                if not self.dry_run:
                    result = self.client.close_position(instrument)
                    if result.success:
                        exit_price = result.fill_price or 0

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
                            f"TIME EXIT COMPLETE | {instrument} | "
                            f"Exit: {exit_price:.5f} PnL: {pnl:.2f}"
                        )

                        # Notify session manager
                        sm = self.session_managers[instrument]
                        sm.handle_trade_exit(session, "time", exit_price)

                        del self._open_positions[instrument]
                    else:
                        logger.error(f"Failed to close position: {result.error}")

    def _process_candles(self) -> None:
        """Fetch and process latest candles for all instruments."""
        for instrument in self.instruments:
            try:
                candle = self.client.get_latest_complete_candle(
                    instrument,
                    TradingConfig.CANDLE_GRANULARITY,
                )

                if not candle:
                    continue

                # Skip if already processed
                last_time = self._last_candle_times.get(instrument)
                if last_time and candle.time <= last_time:
                    continue

                self._last_candle_times[instrument] = candle.time

                logger.debug(
                    f"Candle | {instrument} | "
                    f"O={candle.open:.5f} H={candle.high:.5f} "
                    f"L={candle.low:.5f} C={candle.close:.5f}"
                )

                # Process through session manager
                sm = self.session_managers[instrument]
                signal = sm.process_candle(candle)

                # Signal handling is done via callback

            except Exception as e:
                logger.error(f"Error processing candles for {instrument}: {e}")

    def _log_status(self) -> None:
        """Log current bot status."""
        now = self._get_current_time_ny()
        logger.info(f"=== Status at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ===")

        for instrument in self.instruments:
            sm = self.session_managers[instrument]
            status = sm.get_status()

            for session_name, session_status in status.items():
                if session_status["state"] != "WAITING_FOR_SESSION":
                    logger.info(
                        f"{instrument} | {session_name} | "
                        f"State: {session_status['state']} | "
                        f"Trades: {session_status['trades']}"
                    )

        # Log positions
        for instrument, pos in self._open_positions.items():
            logger.info(
                f"{instrument} | POSITION | "
                f"{pos['direction'].value} {pos['units']} units @ {pos['entry_price']:.5f}"
            )

    def run(self) -> None:
        """Run the main trading loop."""
        logger.info("=" * 60)
        logger.info("MS-ORB Trading Bot Starting")
        logger.info(f"Instruments: {self.instruments}")
        logger.info(f"Environment: {TradingConfig.OANDA_ENVIRONMENT}")
        logger.info(f"Risk per trade: {TradingConfig.RISK_PER_TRADE:.1%}")
        logger.info(f"RR Target: {TradingConfig.RR_TARGET}")
        logger.info(f"Max trades per session: {TradingConfig.MAX_TRADES_PER_SESSION}")
        logger.info("=" * 60)

        # Test connection
        if not self.client.test_connection():
            logger.error("Failed to connect to OANDA")
            return

        # Check for any existing open positions
        for instrument in self.instruments:
            pos = self.client.get_position(instrument)
            if pos:
                logger.warning(
                    f"Existing position found: {instrument} | "
                    f"{pos.units} units @ {pos.avg_price}"
                )

        self._running = True
        last_status_log = datetime.now()
        status_interval = timedelta(minutes=5)

        try:
            while self._running and not self._shutdown_requested:
                loop_start = time.time()

                try:
                    # Process candles
                    self._process_candles()

                    # Check positions
                    self._check_positions()

                    # Check time exits
                    self._check_time_exits()

                    # Log status periodically
                    if datetime.now() - last_status_log > status_interval:
                        self._log_status()
                        last_status_log = datetime.now()

                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)

                # Sleep until next poll
                elapsed = time.time() - loop_start
                sleep_time = max(0, TradingConfig.PRICE_POLL_INTERVAL - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shut down the bot."""
        logger.info("Shutting down trading bot...")
        self._running = False

        # Log final status
        self._log_status()

        # Get trade stats
        stats = self.trade_logger.get_trade_stats()
        logger.info(f"Trade Statistics: {stats}")

        logger.info("Trading bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MS-ORB Trading Bot")
    parser.add_argument(
        "--instruments",
        type=str,
        help="Comma-separated list of instruments (e.g., XAU_USD,NAS100_USD)",
    )
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
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    instruments = None
    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]

    # Set up signal handlers
    bot = TradingBot(instruments=instruments, dry_run=args.dry_run)

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        bot._shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the bot
    bot.run()


if __name__ == "__main__":
    main()
