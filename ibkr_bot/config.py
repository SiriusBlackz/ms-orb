"""Configuration for MS-ORB IBKR trading bot.

TSLA only, NY session only. All times in New York timezone.
"""

import os
from dataclasses import dataclass
from datetime import time
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class SessionName(Enum):
    """Trading session names."""
    NY = "NY"


@dataclass(frozen=True)
class SessionConfig:
    """Configuration for a trading session.

    All times are in NY timezone (America/New_York / UTC-5 or UTC-4 DST).
    """
    name: SessionName
    range_start: time      # Start of 30-min opening range
    range_end: time        # End of opening range (entry window starts)
    entry_end: time        # End of entry window
    session_close: time    # Auto-close 5 mins before session end

    @property
    def range_duration_minutes(self) -> int:
        """Duration of opening range in minutes."""
        return 30


# Session time configurations (all times NY / America/New_York)
SESSIONS = {
    SessionName.NY: SessionConfig(
        name=SessionName.NY,
        range_start=time(9, 30),    # 09:30 NY
        range_end=time(10, 0),      # 10:00 NY
        entry_end=time(13, 0),      # 13:00 NY
        session_close=time(15, 55), # 15:55 NY
    ),
}


# Instrument
INSTRUMENT = "TSLA"

# Instrument-specific configuration
INSTRUMENT_CONFIG = {
    "TSLA": {
        "display_name": "Tesla",
        "sec_type": "STK",
        "exchange": "SMART",
        "currency": "USD",
        "primary_exchange": "NASDAQ",
        "min_trade_units": 1,
    },
}


# Trading parameters
class TradingConfig:
    """Trading configuration loaded from environment."""

    # Risk management
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1% of account
    RR_TARGET: float = float(os.getenv("RR_TARGET", "5.0"))  # Risk-reward ratio
    MAX_TRADES_PER_SESSION: int = int(os.getenv("MAX_TRADES_PER_SESSION", "2"))

    # Polling intervals
    PRICE_POLL_INTERVAL: int = int(os.getenv("PRICE_POLL_INTERVAL", "10"))  # seconds
    CANDLE_GRANULARITY: str = "5 mins"  # ib_insync bar size

    # IBKR connection configuration
    IBKR_HOST: str = os.getenv("IBKR_HOST", "127.0.0.1")
    IBKR_PORT: int = int(os.getenv("IBKR_PORT", "7497"))  # 7497=TWS paper, 7496=TWS live, 4002=GW paper, 4001=GW live
    IBKR_CLIENT_ID: int = int(os.getenv("IBKR_CLIENT_ID", "1"))

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if cls.RISK_PER_TRADE <= 0 or cls.RISK_PER_TRADE > 0.1:
            errors.append(f"RISK_PER_TRADE must be between 0 and 0.1, got: {cls.RISK_PER_TRADE}")
        if cls.RR_TARGET <= 0:
            errors.append(f"RR_TARGET must be positive, got: {cls.RR_TARGET}")
        if cls.IBKR_PORT not in (7496, 7497, 4001, 4002):
            errors.append(f"IBKR_PORT should be 7496/7497 (TWS) or 4001/4002 (Gateway), got: {cls.IBKR_PORT}")

        return errors


# Logging configuration
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "ibkr_trades.db")


# Timezone
TIMEZONE = "America/New_York"
