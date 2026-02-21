"""Interactive Brokers client wrapper using ib_insync."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ib_insync import IB, Stock, MarketOrder, StopOrder, LimitOrder, Trade, Contract
from ib_insync import util

from config import TradingConfig, INSTRUMENT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLC candle data."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool


@dataclass
class Price:
    """Current price data."""
    instrument: str
    time: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class Position:
    """Open position data."""
    instrument: str
    units: int  # Positive for long, negative for short
    avg_price: float
    unrealized_pnl: float
    market_value: float


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: Optional[int] = None
    trade_id: Optional[str] = None
    fill_price: Optional[float] = None
    units: Optional[int] = None
    error: Optional[str] = None


class IBKRClient:
    """Client for Interactive Brokers via ib_insync."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        client_id: int = None,
    ):
        """Initialize IBKR client.

        Args:
            host: TWS/Gateway host. Defaults to TradingConfig.IBKR_HOST.
            port: TWS/Gateway port. Defaults to TradingConfig.IBKR_PORT.
            client_id: Client ID. Defaults to TradingConfig.IBKR_CLIENT_ID.
        """
        self.host = host or TradingConfig.IBKR_HOST
        self.port = port or TradingConfig.IBKR_PORT
        self.client_id = client_id or TradingConfig.IBKR_CLIENT_ID
        self.ib = IB()
        self._connected = False

        # Cache the TSLA contract
        self._contracts: dict[str, Contract] = {}

        # Register event handlers for disconnect/error detection
        self.ib.disconnectedEvent += self._on_disconnect
        self.ib.errorEvent += self._on_error

    def connect(self) -> bool:
        """Connect to TWS/Gateway.

        Returns:
            True if connection successful.
        """
        try:
            self.ib.connect(
                self.host,
                self.port,
                clientId=self.client_id,
                timeout=20,
            )
            self._connected = True
            logger.info(
                f"Connected to IBKR | Host: {self.host}:{self.port} | "
                f"Client ID: {self.client_id}"
            )
            return True
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            return False

    def _on_disconnect(self):
        """Handle disconnect event from ib_insync."""
        self._connected = False
        logger.warning("IBKR disconnected event received")

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB error events for connectivity issues."""
        if errorCode == 1100:
            logger.warning(f"IB connectivity lost: {errorString}")
        elif errorCode == 1102:
            logger.info(f"IB connectivity restored: {errorString}")
        elif errorCode == 504:
            logger.error(f"Not connected to TWS/Gateway: {errorString}")

    def reconnect(self) -> bool:
        """Reconnect to IBKR with exponential backoff.

        Returns:
            True if reconnection successful, False after all attempts exhausted.
        """
        max_attempts = 50
        base_delay = 5
        max_delay = 300

        for attempt in range(1, max_attempts + 1):
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                f"Reconnect attempt {attempt}/{max_attempts}, waiting {delay}s..."
            )

            try:
                self.ib.disconnect()
            except Exception:
                pass

            self.ib.sleep(delay)

            try:
                self.ib.connect(
                    self.host, self.port,
                    clientId=self.client_id, timeout=20,
                )
                self._connected = True
                self._contracts = {}  # Force re-qualification after reconnect
                logger.info(f"Reconnected to IBKR on attempt {attempt}")
                return True
            except Exception as e:
                logger.error(f"Reconnect attempt {attempt} failed: {e}")

        logger.critical(f"Failed to reconnect after {max_attempts} attempts")
        return False

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self.ib.isConnected()

    def _get_contract(self, instrument: str) -> Contract:
        """Get or create a qualified contract for an instrument.

        Args:
            instrument: Instrument symbol (e.g., 'TSLA').

        Returns:
            Qualified IB Contract.
        """
        if instrument in self._contracts:
            return self._contracts[instrument]

        config = INSTRUMENT_CONFIG.get(instrument, {})
        contract = Stock(
            symbol=instrument,
            exchange=config.get("exchange", "SMART"),
            currency=config.get("currency", "USD"),
        )

        # Qualify the contract to get conId
        qualified = self.ib.qualifyContracts(contract)
        if qualified:
            self._contracts[instrument] = qualified[0]
            logger.info(f"Contract qualified: {instrument} -> conId={qualified[0].conId}")
            return qualified[0]

        # Fallback: use unqualified
        self._contracts[instrument] = contract
        return contract

    def test_connection(self) -> bool:
        """Test API connection by fetching account info.

        Returns:
            True if connection successful.
        """
        try:
            if not self.is_connected():
                return self.connect()

            summary = self.ib.accountSummary()
            nav_items = [s for s in summary if s.tag == "NetLiquidation"]
            if nav_items:
                logger.info(
                    f"Connected to IBKR | Account: {nav_items[0].account} | "
                    f"NAV: ${float(nav_items[0].value):,.2f}"
                )
            return True
        except Exception as e:
            logger.error(f"IBKR connection test failed: {e}")
            return False

    def get_account_balance(self) -> float:
        """Get current account net liquidation value.

        Returns:
            Account NAV in USD.
        """
        try:
            summary = self.ib.accountSummary()
            nav_items = [s for s in summary if s.tag == "NetLiquidation"]
            if nav_items:
                return float(nav_items[0].value)
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return 0.0

    def get_price(self, instrument: str) -> Optional[Price]:
        """Get current price for an instrument.

        Args:
            instrument: Instrument symbol (e.g., 'TSLA').

        Returns:
            Price object or None if failed.
        """
        try:
            contract = self._get_contract(instrument)
            self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)  # Allow data to arrive

            ticker = self.ib.ticker(contract)
            if ticker and ticker.bid and ticker.ask:
                return Price(
                    instrument=instrument,
                    time=datetime.now(),
                    bid=ticker.bid,
                    ask=ticker.ask,
                )

            # Fallback: use last price
            if ticker and ticker.last:
                return Price(
                    instrument=instrument,
                    time=datetime.now(),
                    bid=ticker.last,
                    ask=ticker.last,
                )

            return None
        except Exception as e:
            logger.error(f"Failed to get price for {instrument}: {e}")
            return None

    def get_candles(
        self,
        instrument: str,
        bar_size: str = "5 mins",
        duration: str = "3600 S",
        include_incomplete: bool = True,
    ) -> list[Candle]:
        """Get historical candles for an instrument.

        Args:
            instrument: Instrument symbol.
            bar_size: Bar size (e.g., '5 mins', '1 min', '1 hour').
            duration: Duration string (e.g., '3600 S', '1 D').
            include_incomplete: Whether to include the current bar.

        Returns:
            List of Candle objects.
        """
        try:
            contract = self._get_contract(instrument)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours only
                formatDate=1,
            )

            if not bars:
                return []

            candles = []
            for i, bar in enumerate(bars):
                is_last = (i == len(bars) - 1)
                is_complete = not is_last  # Last bar may be incomplete

                if not include_incomplete and not is_complete:
                    continue

                candles.append(Candle(
                    time=bar.date if isinstance(bar.date, datetime) else datetime.combine(bar.date, datetime.min.time()),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=int(bar.volume),
                    complete=is_complete,
                ))

            return candles
        except Exception as e:
            logger.error(f"Failed to get candles for {instrument}: {e}")
            return []

    def get_latest_complete_candle(
        self,
        instrument: str,
        bar_size: str = "5 mins",
    ) -> Optional[Candle]:
        """Get the most recent complete candle.

        Args:
            instrument: Instrument symbol.
            bar_size: Bar size string.

        Returns:
            Most recent complete Candle or None.
        """
        candles = self.get_candles(
            instrument, bar_size, duration="1800 S", include_incomplete=True
        )

        # Find the last complete candle
        for candle in reversed(candles):
            if candle.complete:
                return candle

        return None

    def get_position(self, instrument: str) -> Optional[Position]:
        """Get current open position for an instrument.

        Args:
            instrument: Instrument symbol.

        Returns:
            Position object or None if no position.
        """
        try:
            positions = self.ib.positions()
            for pos in positions:
                if pos.contract.symbol == instrument:
                    if pos.position != 0:
                        return Position(
                            instrument=instrument,
                            units=int(pos.position),
                            avg_price=pos.avgCost,
                            unrealized_pnl=0.0,  # Updated via portfolio
                            market_value=0.0,
                        )
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {instrument}: {e}")
            return None

    def get_all_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of Position objects.
        """
        try:
            result = []
            for pos in self.ib.positions():
                if pos.position != 0:
                    result.append(Position(
                        instrument=pos.contract.symbol,
                        units=int(pos.position),
                        avg_price=pos.avgCost,
                        unrealized_pnl=0.0,
                        market_value=0.0,
                    ))
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def calculate_units(
        self,
        instrument: str,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_price: float,
    ) -> int:
        """Calculate position size based on risk.

        Args:
            instrument: Instrument symbol.
            account_balance: Current account balance.
            risk_percent: Risk as decimal (0.01 = 1%).
            entry_price: Entry price.
            stop_price: Stop loss price.

        Returns:
            Number of shares to trade.
        """
        risk_amount = account_balance * risk_percent
        risk_per_share = abs(entry_price - stop_price)

        if risk_per_share <= 0:
            return 0

        shares = int(risk_amount / risk_per_share)

        config = INSTRUMENT_CONFIG.get(instrument, {})
        min_units = config.get("min_trade_units", 1)

        return max(shares, min_units) if shares > 0 else 0

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float,
        take_profit: float,
    ) -> OrderResult:
        """Place a market order with bracket (stop loss + take profit).

        Uses IB bracket order: parent market + SL stop + TP limit.

        Args:
            instrument: Instrument symbol.
            units: Number of shares (positive=buy, negative=sell).
            stop_loss: Stop loss price.
            take_profit: Take profit price.

        Returns:
            OrderResult with order details or error.
        """
        try:
            contract = self._get_contract(instrument)
            action = "BUY" if units > 0 else "SELL"
            quantity = abs(units)

            # Create bracket order
            bracket = self.ib.bracketOrder(
                action=action,
                quantity=quantity,
                limitPrice=0,  # Will be overridden - we use market
                takeProfitPrice=round(take_profit, 2),
                stopLossPrice=round(stop_loss, 2),
            )

            # Replace parent with market order
            parent = bracket[0]
            parent.orderType = "MKT"
            parent.lmtPrice = 0
            parent.transmit = False

            # Take profit (limit)
            tp_order = bracket[1]
            tp_order.transmit = False

            # Stop loss
            sl_order = bracket[2]
            sl_order.transmit = True  # Last order transmits all

            # Place all three orders
            parent_trade = self.ib.placeOrder(contract, parent)
            self.ib.placeOrder(contract, tp_order)
            self.ib.placeOrder(contract, sl_order)

            # Wait for fill (up to 10 seconds)
            self.ib.sleep(2)

            # Check fill status
            if parent_trade.orderStatus.status in ('Filled', 'PreSubmitted', 'Submitted'):
                fill_price = parent_trade.orderStatus.avgFillPrice or 0
                order_id = parent_trade.order.orderId

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    trade_id=str(order_id),
                    fill_price=fill_price if fill_price > 0 else None,
                    units=units,
                )
            else:
                status = parent_trade.orderStatus.status
                return OrderResult(
                    success=False,
                    error=f"Order status: {status}",
                )

        except Exception as e:
            logger.error(f"Order failed for {instrument}: {e}")
            return OrderResult(success=False, error=str(e))

    def close_position(self, instrument: str) -> OrderResult:
        """Close all positions for an instrument.

        Args:
            instrument: Instrument symbol.

        Returns:
            OrderResult with close details or error.
        """
        try:
            position = self.get_position(instrument)
            if not position:
                return OrderResult(success=True, error="No position to close")

            contract = self._get_contract(instrument)

            # Cancel any open orders for this contract first
            open_orders = self.ib.openOrders()
            for order in open_orders:
                trades = self.ib.openTrades()
                for trade in trades:
                    if trade.contract.symbol == instrument:
                        self.ib.cancelOrder(trade.order)

            self.ib.sleep(1)

            # Place closing market order
            action = "SELL" if position.units > 0 else "BUY"
            quantity = abs(position.units)

            order = MarketOrder(action=action, totalQuantity=quantity)
            trade = self.ib.placeOrder(contract, order)

            self.ib.sleep(2)

            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                return OrderResult(
                    success=True,
                    order_id=trade.order.orderId,
                    fill_price=fill_price,
                    units=-position.units,
                )

            return OrderResult(
                success=True,
                order_id=trade.order.orderId,
            )

        except Exception as e:
            logger.error(f"Failed to close position for {instrument}: {e}")
            return OrderResult(success=False, error=str(e))

    def get_open_orders(self, instrument: str = None) -> list[dict]:
        """Get open orders.

        Args:
            instrument: Optional instrument filter.

        Returns:
            List of order dictionaries.
        """
        try:
            trades = self.ib.openTrades()
            result = []
            for trade in trades:
                if instrument and trade.contract.symbol != instrument:
                    continue
                result.append({
                    "id": str(trade.order.orderId),
                    "symbol": trade.contract.symbol,
                    "action": trade.order.action,
                    "quantity": trade.order.totalQuantity,
                    "type": trade.order.orderType,
                    "status": trade.orderStatus.status,
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def cancel_all_orders(self, instrument: str = None) -> None:
        """Cancel all open orders for an instrument.

        Args:
            instrument: Optional instrument filter. None = cancel all.
        """
        try:
            trades = self.ib.openTrades()
            for trade in trades:
                if instrument and trade.contract.symbol != instrument:
                    continue
                self.ib.cancelOrder(trade.order)
                logger.info(f"Cancelled order {trade.order.orderId} for {trade.contract.symbol}")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

    def sleep(self, seconds: float) -> None:
        """Sleep while keeping IB event loop alive.

        Args:
            seconds: Seconds to sleep.
        """
        self.ib.sleep(seconds)


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    client = IBKRClient()

    if client.connect():
        print(f"\nAccount Balance: ${client.get_account_balance():,.2f}")

        # Test price fetching
        price = client.get_price("TSLA")
        if price:
            print(f"\nTSLA:")
            print(f"  Bid: {price.bid}")
            print(f"  Ask: {price.ask}")
            print(f"  Mid: {price.mid}")

        # Get recent candles
        candles = client.get_candles("TSLA", "5 mins", duration="3600 S")
        if candles:
            print(f"\n  Recent candles:")
            for c in candles[-3:]:
                print(f"    {c.time}: O={c.open} H={c.high} L={c.low} C={c.close} Complete={c.complete}")

        # Check positions
        positions_list = client.get_all_positions()
        if positions_list:
            print("\nOpen Positions:")
            for pos in positions_list:
                print(f"  {pos.instrument}: {pos.units} shares @ {pos.avg_price}")
        else:
            print("\nNo open positions")

        client.disconnect()
    else:
        print("Failed to connect to IBKR")
