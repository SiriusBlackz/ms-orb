"""Configuration for the MS-ORB IBKR bot — Config-A strategy on TSLA.

TSLA only, NY session only. Config A (validated in oanda_bot/backtest.py
--config-a and on 4 years of TSLA 5-min data): first breakout only, HTF
EMA100-aligned, 1.5xATR14 stop, 3:1 reward:risk.

All times in New York timezone.
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
    RR_TARGET: float = float(os.getenv("RR_TARGET", "3.0"))  # Risk-reward ratio
    MAX_TRADES_PER_SESSION: int = int(os.getenv("MAX_TRADES_PER_SESSION", "1"))

    # Config-A strategy rules (must match backtest.py --config-a)
    STOP_MODE: str = os.getenv("STOP_MODE", "atr")  # 'atr' or 'candle'
    ATR_STOP_MULT: float = float(os.getenv("ATR_STOP_MULT", "1.5"))
    ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
    HTF_ALIGN: bool = os.getenv("HTF_ALIGN", "true").lower() in ("1", "true", "yes")
    HTF_EMA_PERIOD: int = int(os.getenv("HTF_EMA_PERIOD", "100"))
    FIRST_BREAKOUT_ONLY: bool = os.getenv("FIRST_BREAKOUT_ONLY", "true").lower() in ("1", "true", "yes")
    # IBKR duration string for the indicator fetch. RTH-only 5-min bars
    # (useRTH=True in ibkr_client) match the backtest's RTH dataset; 20
    # calendar days ~= 14 trading days ~= 1,090 bars — EMA100/ATR14 fully
    # converged, one reqHistoricalData call.
    INDICATOR_CANDLE_DURATION: str = os.getenv("INDICATOR_CANDLE_DURATION", "20 D")

    # Polling intervals
    PRICE_POLL_INTERVAL: int = int(os.getenv("PRICE_POLL_INTERVAL", "10"))  # seconds

    # Order handling: how long to wait for a market order to fill before the
    # entry bracket is cancelled / the close is reported as not done.
    ENTRY_FILL_TIMEOUT: float = float(os.getenv("ENTRY_FILL_TIMEOUT", "30"))
    EXIT_FILL_TIMEOUT: float = float(os.getenv("EXIT_FILL_TIMEOUT", "30"))

    # Data freshness: a completed 5-min bar should be seen within seconds of
    # its close. Without a real-time market-data subscription IB serves bars
    # ~15 minutes late; signals from those are worthless, so candles older
    # than this (seconds past their close, during the session) are ignored.
    # 0 disables the check.
    DATA_MAX_LAG_SECONDS: float = float(os.getenv("DATA_MAX_LAG_SECONDS", "180"))
    CANDLE_GRANULARITY: str = "5 mins"  # ib_insync bar size

    # IBKR connection configuration
    IBKR_HOST: str = os.getenv("IBKR_HOST", "127.0.0.1")
    IBKR_PORT: int = int(os.getenv("IBKR_PORT", "4002"))  # 4002=GW paper, 4001=GW live, 7497=TWS paper, 7496=TWS live
    IBKR_CLIENT_ID: int = int(os.getenv("IBKR_CLIENT_ID", "1"))

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if cls.RISK_PER_TRADE <= 0 or cls.RISK_PER_TRADE > 0.1:
            errors.append(f"RISK_PER_TRADE must be between 0 and 0.1, got: {cls.RISK_PER_TRADE}")
        if cls.RR_TARGET <= 0:
            errors.append(f"RR_TARGET must be positive, got: {cls.RR_TARGET}")
        if cls.STOP_MODE not in ("atr", "candle"):
            errors.append(f"STOP_MODE must be 'atr' or 'candle', got: {cls.STOP_MODE}")
        if cls.ATR_STOP_MULT <= 0:
            errors.append(f"ATR_STOP_MULT must be positive, got: {cls.ATR_STOP_MULT}")
        if cls.IBKR_PORT not in (7496, 7497, 4001, 4002):
            errors.append(f"IBKR_PORT should be 7496/7497 (TWS) or 4001/4002 (Gateway), got: {cls.IBKR_PORT}")

        return errors


# Logging configuration
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "ibkr_trades.db")


# Timezone
TIMEZONE = "America/New_York"
