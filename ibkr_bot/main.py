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

        # Consecutive loops on which the broker reported us flat while we
        # still track a position but no closing fill could be found.
        self._flat_checks = 0
        # Set when the account holds a position the bot cannot explain; no
        # new entries are sent until the account is flat again.
        self._blocked_reason: Optional[str] = None

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

        # Never stack a new trade on top of a position we did not open.
        self._refresh_block()
        if self._blocked_reason:
            logger.critical(
                f"ORDER SKIPPED | {self.instrument} | {self._blocked_reason}"
            )
            self.session_manager.handle_order_failed(self._blocked_reason)
            return

        # Get account balance for position sizing
        balance = self.client.get_account_balance()
        if balance <= 0:
            logger.error("Failed to get account balance, skipping trade")
            self.session_manager.handle_order_failed("no account balance")
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
            self.session_manager.handle_order_failed("position size 0")
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
            if result.fill_price:
                risk = abs(signal.entry_price - signal.stop_loss)
                slip = result.fill_price - signal.entry_price
                if signal.direction == Direction.SHORT:
                    slip = -slip
                logger.info(
                    f"ENTRY SLIPPAGE | {self.instrument} | signal {signal.entry_price:.2f} "
                    f"fill {result.fill_price:.2f} | {slip:+.2f} = "
                    f"{(slip / risk if risk else 0):+.2f}R (positive = worse)"
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
                "entry_action": "BUY" if signal.direction == Direction.LONG else "SELL",
                "since": datetime.now().astimezone(),
            }
            self._flat_checks = 0

            # Update session manager
            if result.trade_id:
                self.session_manager.set_trade_id(result.trade_id)

        else:
            logger.error(f"ORDER FAILED | {self.instrument} | {result.error}")
            # The client has already cancelled the bracket; release the session.
            self.session_manager.handle_order_failed(result.error or "order failed")

    def _on_trade_closed(self, record) -> None:
        """Handle trade closed notification from session manager."""
        logger.info(
            f"TRADE CLOSED (session) | {self.instrument} | "
            f"Exit: {record.exit_price} Reason: {record.exit_reason}"
        )

    def _refresh_block(self) -> None:
        """Re-evaluate whether an unexplained position should block trading."""
        if self._open_position is not None:
            return
        units = self.client.get_position_units(self.instrument)
        if units is None:
            return  # unknown: keep the current view
        if units != 0:
            if not self._blocked_reason:
                logger.critical(
                    f"UNEXPLAINED POSITION | {self.instrument} | {units} shares held but "
                    f"not opened by this run - trading is blocked until the account is flat"
                )
            self._blocked_reason = f"unexplained position of {units} shares"
        elif self._blocked_reason:
            logger.warning(f"{self.instrument} account is flat again - trading unblocked")
            self._blocked_reason = None

    def _reconcile_on_start(self) -> None:
        """Bring the bot's view in line with the account at startup.

        - Flat with working orders: cancel them (stale bracket children could
          otherwise open a position on their own).
        - Position with a matching stop + target working: adopt it so the
          time exit and exit logging still apply.
        - Any other position: block new entries until the account is flat.
        """
        units = self.client.get_position_units(self.instrument)
        if units is None:
            logger.warning("Could not read positions at startup; will re-check")
            return

        orders = self.client.get_open_orders(self.instrument)
        if units == 0:
            if orders:
                logger.warning(
                    f"Flat but {len(orders)} working {self.instrument} order(s) found - cancelling"
                )
                self.client.cancel_all_orders(self.instrument)
            return

        closing = "SELL" if units > 0 else "BUY"
        stops = [o for o in orders if o["type"] in ("STP", "STP LMT")
                 and o["action"] == closing and int(o["quantity"]) == abs(units)]
        targets = [o for o in orders if o["type"] == "LMT"
                   and o["action"] == closing and int(o["quantity"]) == abs(units)]
        pos = self.client.get_position(self.instrument)

        if len(stops) == 1 and len(targets) == 1 and pos:
            direction = Direction.LONG if units > 0 else Direction.SHORT
            stop_loss = float(stops[0]["stop_price"])
            take_profit = float(targets[0]["limit_price"])
            self._open_position = {
                "trade_id": None,
                "db_id": None,
                "direction": direction,
                "entry_price": pos.avg_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "units": abs(units),
                "entry_action": "BUY" if units > 0 else "SELL",
                "since": datetime.now().astimezone(),
            }
            self.session_manager.adopt_trade(direction, pos.avg_price, stop_loss, take_profit)
            logger.warning(
                f"ADOPTED POSITION | {self.instrument} | {direction.value} {abs(units)} shares "
                f"@ {pos.avg_price:.2f} | SL={stop_loss} TP={take_profit}"
            )
        else:
            self._refresh_block()

    def _record_exit(self, exit_price: Optional[float], exit_reason: str) -> None:
        """Write the exit to the database and tell the session manager."""
        pos_info = self._open_position
        pnl = rr = None
        try:
            if exit_price is not None:
                pnl = calculate_pnl(
                    pos_info["direction"].value, pos_info["entry_price"],
                    exit_price, pos_info["units"],
                )
                rr = calculate_rr_achieved(
                    pos_info["direction"].value, pos_info["entry_price"],
                    exit_price, pos_info["stop_loss"],
                )
            if pos_info["db_id"] is not None:
                self.trade_logger.log_exit(
                    row_id=pos_info["db_id"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    rr_achieved=rr,
                )
            price_s = f"{exit_price:.2f}" if exit_price is not None else "unknown"
            pnl_s = f"${pnl:.2f}" if pnl is not None else "n/a"
            rr_s = f"{rr:.2f}" if rr is not None else "n/a"
            logger.info(
                f"EXIT LOGGED | {self.instrument} | Price: {price_s} "
                f"Reason: {exit_reason} PnL: {pnl_s} RR: {rr_s}"
            )
        except Exception as e:
            logger.error(f"Error logging exit for {self.instrument}: {e}", exc_info=True)
        finally:
            # Always release the position and the session, even if logging failed.
            self.session_manager.handle_trade_exit(exit_reason, exit_price)
            self.client.clear_bracket(self.instrument)
            self._open_position = None
            self._flat_checks = 0

    def _check_position(self) -> None:
        """Detect that the open position was closed by its stop or target.

        The exit price and reason come from the bracket's own fills. "Flat"
        is only believed when the broker link is up; if we are flat but no
        closing fill can be found, wait a few loops before giving up and
        recording the exit as unknown.
        """
        if self._open_position is None:
            return

        units = self.client.get_position_units(self.instrument)
        if units is None or units != 0:
            self._flat_checks = 0
            return  # still open, or state unknown (disconnected)

        pos_info = self._open_position
        fill = self.client.get_exit_fill(
            self.instrument,
            since=pos_info.get("since"),
            entry_action=pos_info.get("entry_action"),
        )

        if fill is None:
            self._flat_checks += 1
            if self._flat_checks < 3:
                return
            logger.warning(
                f"Position closed detected: {self.instrument}, but no closing fill found"
            )
            self._record_exit(None, "unknown")
            return

        logger.info(f"Position closed detected: {self.instrument}")
        reason = fill.reason
        if reason is None:
            # Infer from which bracket level the fill is nearer to.
            sl_dist = abs(fill.price - pos_info["stop_loss"])
            tp_dist = abs(fill.price - pos_info["take_profit"])
            reason = "stop" if sl_dist < tp_dist else "target"
        self._record_exit(fill.price, reason)

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
                    logger.info(
                        f"TIME EXIT COMPLETE | {self.instrument} | Exit: {result.fill_price}"
                    )
                    self._record_exit(result.fill_price, "time")
                else:
                    # Retried on the next loop while the position is still tracked.
                    logger.error(f"Failed to close position: {result.error}")

    def _candle_lag_seconds(self, candle: Candle) -> Optional[float]:
        """Seconds between the bar's close and now (same clock the session uses)."""
        try:
            bar_end = candle.time.astimezone(NY_TZ) + timedelta(minutes=5)
            return (self._get_current_time_ny() - bar_end).total_seconds()
        except Exception:
            return None

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

            # How late is this bar? (seconds since its close)
            lag = self._candle_lag_seconds(candle)
            if lag is not None and self.session_manager.is_session_active():
                logger.info(f"CANDLE LAG | {self.instrument} | bar closed {lag:.0f}s ago")
                max_lag = TradingConfig.DATA_MAX_LAG_SECONDS
                if max_lag and lag > max_lag:
                    logger.critical(
                        f"DELAYED DATA | {self.instrument} | latest bar is {lag / 60:.1f} min old "
                        f"(limit {max_lag:.0f}s) - ignoring it. The account needs a real-time "
                        f"market-data subscription for API use."
                    )
                    return

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

        if self._blocked_reason:
            logger.critical(
                f"{self.instrument} | TRADING BLOCKED | {self._blocked_reason}"
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

        # Reconcile with whatever the account already holds / has working
        self._reconcile_on_start()

        self._running = True
        last_status_log = datetime.now()
        status_interval = timedelta(minutes=5)

        try:
            while self._running and not self._shutdown_requested:
                loop_start = time.time()

                try:
                    # Check connection
                    if not self.client.is_connected():
                        logger.warning("IBKR disconnected, attempting reconnect...")
                        if not self.client.reconnect():
                            logger.critical("All reconnect attempts exhausted, shutting down")
                            break
                        # Re-check position state after reconnect (may have filled/closed while disconnected)
                        self._check_position()

                    # Process candles
                    self._process_candles()

                    # Check position
                    self._check_position()

                    # Check time exit
                    self._check_time_exit()

                    # Log status periodically
                    if datetime.now() - last_status_log > status_interval:
                        self._refresh_block()
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
        help="IBKR port override (4002=GW paper, 4001=GW live, 7497=TWS paper, 7496=TWS live)",
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
