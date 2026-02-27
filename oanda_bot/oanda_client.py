"""OANDA v20 REST API client wrapper."""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades

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
    margin_used: float


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    fill_price: Optional[float] = None
    units: Optional[int] = None
    error: Optional[str] = None


class OANDAClient:
    """Client for OANDA v20 REST API."""

    def __init__(
        self,
        api_key: str = None,
        account_id: str = None,
        environment: str = None,
    ):
        """Initialize OANDA client.

        Args:
            api_key: OANDA API key. Defaults to TradingConfig.OANDA_API_KEY.
            account_id: OANDA account ID. Defaults to TradingConfig.OANDA_ACCOUNT_ID.
            environment: 'practice' or 'live'. Defaults to TradingConfig.OANDA_ENVIRONMENT.
        """
        self.api_key = api_key or TradingConfig.OANDA_API_KEY
        self.account_id = account_id or TradingConfig.OANDA_ACCOUNT_ID
        self.environment = environment or TradingConfig.OANDA_ENVIRONMENT

        # Set up API client
        if self.environment == "live":
            self.api = oandapyV20.API(access_token=self.api_key, environment="live")
        else:
            self.api = oandapyV20.API(access_token=self.api_key, environment="practice")

        logger.info(f"OANDA client initialized for {self.environment} environment")

    def test_connection(self) -> bool:
        """Test API connection by fetching account info.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            r = accounts.AccountSummary(self.account_id)
            self.api.request(r)
            account = r.response.get("account", {})
            logger.info(
                f"Connected to OANDA | Account: {self.account_id} | "
                f"Balance: {account.get('balance', 'N/A')} | "
                f"Currency: {account.get('currency', 'N/A')}"
            )
            return True
        except Exception as e:
            logger.error(f"OANDA connection test failed: {e}")
            return False

    def get_account_balance(self) -> float:
        """Get current account balance.

        Returns:
            Account balance in account currency.
        """
        try:
            r = accounts.AccountSummary(self.account_id)
            self.api.request(r)
            balance = float(r.response["account"]["balance"])
            return balance
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return 0.0

    def get_account_nav(self) -> float:
        """Get account Net Asset Value (NAV).

        Returns:
            NAV including unrealized P&L.
        """
        try:
            r = accounts.AccountSummary(self.account_id)
            self.api.request(r)
            nav = float(r.response["account"]["NAV"])
            return nav
        except Exception as e:
            logger.error(f"Failed to get account NAV: {e}")
            return 0.0

    def get_price(self, instrument: str) -> Optional[Price]:
        """Get current price for an instrument.

        Args:
            instrument: Instrument name (e.g., 'XAU_USD').

        Returns:
            Price object or None if failed.
        """
        try:
            params = {"instruments": instrument}
            r = pricing.PricingInfo(self.account_id, params=params)
            self.api.request(r)

            prices = r.response.get("prices", [])
            if not prices:
                return None

            p = prices[0]
            return Price(
                instrument=p["instrument"],
                time=datetime.fromisoformat(p["time"].replace("Z", "+00:00")),
                bid=float(p["bids"][0]["price"]) if p.get("bids") else 0.0,
                ask=float(p["asks"][0]["price"]) if p.get("asks") else 0.0,
            )
        except Exception as e:
            logger.error(f"Failed to get price for {instrument}: {e}")
            return None

    def get_candles(
        self,
        instrument: str,
        granularity: str = "M5",
        count: int = 10,
        include_incomplete: bool = True,
    ) -> list[Candle]:
        """Get historical candles for an instrument.

        Args:
            instrument: Instrument name (e.g., 'XAU_USD').
            granularity: Candle granularity (M1, M5, M15, H1, etc.).
            count: Number of candles to fetch.
            include_incomplete: Whether to include the current incomplete candle.

        Returns:
            List of Candle objects.
        """
        try:
            params = {
                "granularity": granularity,
                "count": count,
            }
            r = instruments.InstrumentsCandles(instrument, params=params)
            self.api.request(r)

            candles = []
            for c in r.response.get("candles", []):
                if not include_incomplete and not c["complete"]:
                    continue

                mid = c["mid"]
                candles.append(Candle(
                    time=datetime.fromisoformat(c["time"].replace("Z", "+00:00")),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=int(c["volume"]),
                    complete=c["complete"],
                ))

            return candles
        except Exception as e:
            logger.error(f"Failed to get candles for {instrument}: {e}")
            return []

    def get_latest_complete_candle(
        self,
        instrument: str,
        granularity: str = "M5",
    ) -> Optional[Candle]:
        """Get the most recent complete candle.

        Args:
            instrument: Instrument name.
            granularity: Candle granularity.

        Returns:
            Most recent complete Candle or None.
        """
        candles = self.get_candles(
            instrument, granularity, count=2, include_incomplete=True
        )

        # Find the last complete candle
        for candle in reversed(candles):
            if candle.complete:
                return candle

        return None

    def get_position(self, instrument: str) -> Optional[Position]:
        """Get current open position for an instrument.

        Args:
            instrument: Instrument name.

        Returns:
            Position object or None if no position.
        """
        try:
            r = positions.PositionDetails(self.account_id, instrument)
            self.api.request(r)

            pos = r.response.get("position", {})
            long_units = int(float(pos.get("long", {}).get("units", 0)))
            short_units = int(float(pos.get("short", {}).get("units", 0)))

            if long_units == 0 and short_units == 0:
                return None

            if long_units > 0:
                return Position(
                    instrument=instrument,
                    units=long_units,
                    avg_price=float(pos["long"]["averagePrice"]),
                    unrealized_pnl=float(pos["long"]["unrealizedPL"]),
                    margin_used=float(pos.get("marginUsed", 0)),
                )
            else:
                return Position(
                    instrument=instrument,
                    units=short_units,  # Will be negative
                    avg_price=float(pos["short"]["averagePrice"]),
                    unrealized_pnl=float(pos["short"]["unrealizedPL"]),
                    margin_used=float(pos.get("marginUsed", 0)),
                )
        except oandapyV20.exceptions.V20Error as e:
            # 404 means no position exists
            if "404" in str(e):
                return None
            logger.error(f"Failed to get position for {instrument}: {e}")
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
            r = positions.OpenPositions(self.account_id)
            self.api.request(r)

            result = []
            for pos in r.response.get("positions", []):
                instrument = pos["instrument"]
                long_units = int(float(pos.get("long", {}).get("units", 0)))
                short_units = int(float(pos.get("short", {}).get("units", 0)))

                if long_units > 0:
                    result.append(Position(
                        instrument=instrument,
                        units=long_units,
                        avg_price=float(pos["long"]["averagePrice"]),
                        unrealized_pnl=float(pos["long"]["unrealizedPL"]),
                        margin_used=float(pos.get("marginUsed", 0)),
                    ))
                elif short_units != 0:
                    result.append(Position(
                        instrument=instrument,
                        units=short_units,
                        avg_price=float(pos["short"]["averagePrice"]),
                        unrealized_pnl=float(pos["short"]["unrealizedPL"]),
                        margin_used=float(pos.get("marginUsed", 0)),
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
            instrument: Instrument name.
            account_balance: Current account balance.
            risk_percent: Risk as decimal (0.01 = 1%).
            entry_price: Entry price.
            stop_price: Stop loss price.

        Returns:
            Number of units to trade.
        """
        risk_amount = account_balance * risk_percent
        risk_per_unit = abs(entry_price - stop_price)

        if risk_per_unit <= 0:
            return 0

        units = int(risk_amount / risk_per_unit)

        # Apply minimum trade units
        config = INSTRUMENT_CONFIG.get(instrument, {})
        min_units = config.get("min_trade_units", 1)

        return max(units, min_units) if units > 0 else 0

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float,
        take_profit: float,
    ) -> OrderResult:
        """Place a market order with stop loss and take profit.

        Args:
            instrument: Instrument name.
            units: Number of units (positive for long, negative for short).
            stop_loss: Stop loss price.
            take_profit: Take profit price.

        Returns:
            OrderResult with order details or error.
        """
        try:
            # Format prices based on instrument precision
            config = INSTRUMENT_CONFIG.get(instrument, {})
            pip_location = config.get("pip_location", -4)
            precision = abs(pip_location) + 1

            data = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(int(units)),
                    "timeInForce": "FOK",  # Fill or Kill
                    "stopLossOnFill": {
                        "price": f"{stop_loss:.{precision}f}",
                    },
                    "takeProfitOnFill": {
                        "price": f"{take_profit:.{precision}f}",
                    },
                }
            }

            r = orders.OrderCreate(self.account_id, data=data)
            self.api.request(r)

            response = r.response

            # Check if order was filled
            if "orderFillTransaction" in response:
                fill = response["orderFillTransaction"]
                return OrderResult(
                    success=True,
                    order_id=fill.get("orderID"),
                    trade_id=fill.get("tradeOpened", {}).get("tradeID"),
                    fill_price=float(fill.get("price", 0)),
                    units=int(float(fill.get("units", 0))),
                )
            elif "orderCancelTransaction" in response:
                cancel = response["orderCancelTransaction"]
                return OrderResult(
                    success=False,
                    error=cancel.get("reason", "Order cancelled"),
                )
            else:
                return OrderResult(
                    success=False,
                    error="Unknown order response",
                )

        except oandapyV20.exceptions.V20Error as e:
            logger.error(f"Order failed for {instrument}: {e}")
            return OrderResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Order failed for {instrument}: {e}")
            return OrderResult(success=False, error=str(e))

    def place_limit_order(
        self,
        instrument: str,
        units: int,
        price: float,
        stop_loss: float,
        take_profit: float,
        gtd_time: Optional[str] = None,
    ) -> OrderResult:
        """Place a limit order with stop loss and take profit.

        Args:
            instrument: Instrument name.
            units: Number of units (positive for long, negative for short).
            price: Limit price to enter at.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            gtd_time: Optional GTD expiry time in RFC3339 format.
                       If None, uses GTC (Good Till Cancelled).

        Returns:
            OrderResult with order details or error.
        """
        try:
            config = INSTRUMENT_CONFIG.get(instrument, {})
            pip_location = config.get("pip_location", -4)
            precision = abs(pip_location) + 1

            order_data = {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(int(units)),
                "price": f"{price:.{precision}f}",
                "timeInForce": "GTD" if gtd_time else "GTC",
                "stopLossOnFill": {
                    "price": f"{stop_loss:.{precision}f}",
                },
                "takeProfitOnFill": {
                    "price": f"{take_profit:.{precision}f}",
                },
                "triggerCondition": "DEFAULT",
            }

            if gtd_time:
                order_data["gtdTime"] = gtd_time

            data = {"order": order_data}

            r = orders.OrderCreate(self.account_id, data=data)
            self.api.request(r)

            response = r.response

            # Limit order may fill immediately if price is already at limit
            if "orderFillTransaction" in response:
                fill = response["orderFillTransaction"]
                return OrderResult(
                    success=True,
                    order_id=fill.get("orderID"),
                    trade_id=fill.get("tradeOpened", {}).get("tradeID"),
                    fill_price=float(fill.get("price", 0)),
                    units=int(float(fill.get("units", 0))),
                )
            elif "orderCreateTransaction" in response:
                create = response["orderCreateTransaction"]
                return OrderResult(
                    success=True,
                    order_id=create.get("id"),
                )
            elif "orderCancelTransaction" in response:
                cancel = response["orderCancelTransaction"]
                return OrderResult(
                    success=False,
                    error=cancel.get("reason", "Order cancelled"),
                )
            else:
                return OrderResult(
                    success=False,
                    error="Unknown order response",
                )

        except oandapyV20.exceptions.V20Error as e:
            logger.error(f"Limit order failed for {instrument}: {e}")
            return OrderResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Limit order failed for {instrument}: {e}")
            return OrderResult(success=False, error=str(e))

    def get_order_details(self, order_id: str) -> Optional[dict]:
        """Get details of a specific order.

        Args:
            order_id: OANDA order ID.

        Returns:
            Order dict with 'state', 'type', 'tradeOpenedID' etc., or None if failed.
        """
        try:
            r = orders.OrderDetails(self.account_id, orderID=order_id)
            self.api.request(r)
            return r.response.get("order", {})
        except oandapyV20.exceptions.V20Error as e:
            if "404" in str(e):
                return None
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: OANDA order ID.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        try:
            r = orders.OrderCancel(self.account_id, orderID=order_id)
            self.api.request(r)
            return True
        except oandapyV20.exceptions.V20Error as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def close_position(self, instrument: str) -> OrderResult:
        """Close all positions for an instrument.

        Args:
            instrument: Instrument name.

        Returns:
            OrderResult with close details or error.
        """
        try:
            # Get current position
            position = self.get_position(instrument)
            if not position:
                return OrderResult(success=True, error="No position to close")

            # Close the position
            data = {"longUnits": "ALL"} if position.units > 0 else {"shortUnits": "ALL"}

            r = positions.PositionClose(self.account_id, instrument, data=data)
            self.api.request(r)

            response = r.response

            if "longOrderFillTransaction" in response:
                fill = response["longOrderFillTransaction"]
            elif "shortOrderFillTransaction" in response:
                fill = response["shortOrderFillTransaction"]
            else:
                return OrderResult(success=True)

            return OrderResult(
                success=True,
                order_id=fill.get("orderID"),
                fill_price=float(fill.get("price", 0)),
                units=int(float(fill.get("units", 0))),
            )

        except Exception as e:
            logger.error(f"Failed to close position for {instrument}: {e}")
            return OrderResult(success=False, error=str(e))

    def get_open_trades(self, instrument: str = None) -> list[dict]:
        """Get open trades.

        Args:
            instrument: Optional instrument filter.

        Returns:
            List of trade dictionaries.
        """
        try:
            params = {}
            if instrument:
                params["instrument"] = instrument

            r = trades.OpenTrades(self.account_id, params=params if params else None)
            self.api.request(r)

            return r.response.get("trades", [])
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}")
            return []

    def modify_trade(
        self,
        trade_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Modify an open trade's SL/TP.

        Args:
            trade_id: Trade ID to modify.
            stop_loss: New stop loss price (None to keep existing).
            take_profit: New take profit price (None to keep existing).

        Returns:
            True if successful, False otherwise.
        """
        try:
            data = {}

            if stop_loss is not None:
                data["stopLoss"] = {"price": str(stop_loss)}

            if take_profit is not None:
                data["takeProfit"] = {"price": str(take_profit)}

            if not data:
                return True

            r = trades.TradeCRCDO(self.account_id, trade_id, data=data)
            self.api.request(r)

            return True
        except Exception as e:
            logger.error(f"Failed to modify trade {trade_id}: {e}")
            return False

    def close_trade(self, trade_id: str, units: str = "ALL") -> OrderResult:
        """Close a specific trade.

        Args:
            trade_id: Trade ID to close.
            units: Units to close ("ALL" or specific number).

        Returns:
            OrderResult with close details.
        """
        try:
            data = {"units": units}
            r = trades.TradeClose(self.account_id, trade_id, data=data)
            self.api.request(r)

            response = r.response
            if "orderFillTransaction" in response:
                fill = response["orderFillTransaction"]
                return OrderResult(
                    success=True,
                    order_id=fill.get("orderID"),
                    trade_id=trade_id,
                    fill_price=float(fill.get("price", 0)),
                    units=int(float(fill.get("units", 0))),
                )

            return OrderResult(success=True)
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return OrderResult(success=False, error=str(e))


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    client = OANDAClient()

    if client.test_connection():
        print(f"\nAccount Balance: ${client.get_account_balance():,.2f}")
        print(f"Account NAV: ${client.get_account_nav():,.2f}")

        # Test price fetching
        for instrument in ["XAU_USD", "NAS100_USD"]:
            price = client.get_price(instrument)
            if price:
                print(f"\n{instrument}:")
                print(f"  Bid: {price.bid}")
                print(f"  Ask: {price.ask}")
                print(f"  Mid: {price.mid}")

            # Get recent candles
            candles = client.get_candles(instrument, "M5", count=3)
            if candles:
                print(f"  Recent candles:")
                for c in candles:
                    print(f"    {c.time}: O={c.open} H={c.high} L={c.low} C={c.close} Complete={c.complete}")

        # Check positions
        positions_list = client.get_all_positions()
        if positions_list:
            print("\nOpen Positions:")
            for pos in positions_list:
                print(f"  {pos.instrument}: {pos.units} units @ {pos.avg_price}, PnL: {pos.unrealized_pnl}")
        else:
            print("\nNo open positions")
    else:
        print("Failed to connect to OANDA")
