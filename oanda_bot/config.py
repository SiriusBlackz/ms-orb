"""Configuration for MS-ORB trading bot.

All times are in New York timezone (America/New_York).
"""

import os
from dataclasses import dataclass
from datetime import time
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class SessionName(Enum):
    """Trading session names."""
    TOKYO = "TOKYO"
    LONDON = "LONDON"
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
    SessionName.TOKYO: SessionConfig(
        name=SessionName.TOKYO,
        range_start=time(18, 0),    # 18:00 NY
        range_end=time(18, 30),     # 18:30 NY
        entry_end=time(21, 30),     # 21:30 NY
        session_close=time(2, 54),  # 02:54 NY (next day)
    ),
    SessionName.LONDON: SessionConfig(
        name=SessionName.LONDON,
        range_start=time(3, 0),     # 03:00 NY
        range_end=time(3, 30),      # 03:30 NY
        entry_end=time(6, 30),      # 06:30 NY
        session_close=time(8, 25),  # 08:25 NY
    ),
    SessionName.NY: SessionConfig(
        name=SessionName.NY,
        range_start=time(9, 30),    # 09:30 NY
        range_end=time(10, 0),      # 10:00 NY
        entry_end=time(13, 0),      # 13:00 NY
        session_close=time(15, 55), # 15:55 NY
    ),
}


# Instruments to trade
INSTRUMENTS = [
    "XAU_USD",      # Gold
    "NAS100_USD",   # Nasdaq 100
]

# Instrument-specific configurations
INSTRUMENT_CONFIG = {
    "XAU_USD": {
        "display_name": "Gold",
        "pip_location": -2,  # 0.01 = 1 pip for gold
        "min_trade_units": 1,
    },
    "NAS100_USD": {
        "display_name": "Nasdaq 100",
        "pip_location": 0,   # 1.0 = 1 pip for indices
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
    CANDLE_GRANULARITY: str = "M5"  # 5-minute candles

    # OANDA API configuration
    OANDA_API_KEY: str = os.getenv("OANDA_API_KEY", "")
    OANDA_ACCOUNT_ID: str = os.getenv("OANDA_ACCOUNT_ID", "")
    OANDA_ENVIRONMENT: str = os.getenv("OANDA_ENVIRONMENT", "practice")  # 'practice' or 'live'

    @classmethod
    def get_oanda_url(cls) -> str:
        """Get OANDA API base URL based on environment."""
        if cls.OANDA_ENVIRONMENT == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not cls.OANDA_API_KEY:
            errors.append("OANDA_API_KEY is not set")
        if not cls.OANDA_ACCOUNT_ID:
            errors.append("OANDA_ACCOUNT_ID is not set")
        if cls.OANDA_ENVIRONMENT not in ("practice", "live"):
            errors.append(f"OANDA_ENVIRONMENT must be 'practice' or 'live', got: {cls.OANDA_ENVIRONMENT}")
        if cls.RISK_PER_TRADE <= 0 or cls.RISK_PER_TRADE > 0.1:
            errors.append(f"RISK_PER_TRADE must be between 0 and 0.1, got: {cls.RISK_PER_TRADE}")
        if cls.RR_TARGET <= 0:
            errors.append(f"RR_TARGET must be positive, got: {cls.RR_TARGET}")

        return errors


# Logging configuration
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "trades.db")


# Timezone
TIMEZONE = "America/New_York"
