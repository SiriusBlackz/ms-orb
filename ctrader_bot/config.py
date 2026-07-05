"""Configuration for the MS-ORB cTrader bot (Config-A strategy, prop-firm port).

Same strategy as oanda_bot_configa (`oanda_bot/backtest.py --config-a`):
XAU_USD + NAS100_USD, TOKYO session only, first breakout only (no re-entry),
HTF-aligned (close vs EMA100), 1.5xATR stop, 3:1 reward:risk — executed via
the cTrader Open API (FTMO / FundingPips compatible) instead of OANDA v20.

Internal instrument keys stay OANDA-style (XAU_USD, NAS100_USD) so the
strategy layer is byte-identical; SYMBOL_MAP translates them to the broker's
cTrader symbol names at the client boundary.

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

# Sessions this bot is allowed to trade. Config A is Tokyo-only — London and
# NY are net-negative under these rules (validated in backtest.py --config-a).
ENABLED_SESSIONS = [
    SessionName(s.strip().upper())
    for s in os.getenv("ENABLED_SESSIONS", "TOKYO").split(",")
    if s.strip()
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
    RR_TARGET: float = float(os.getenv("RR_TARGET", "3.0"))  # Risk-reward ratio
    MAX_TRADES_PER_SESSION: int = int(os.getenv("MAX_TRADES_PER_SESSION", "1"))

    # Config-A strategy rules (must match backtest.py --config-a)
    STOP_MODE: str = os.getenv("STOP_MODE", "atr")  # 'atr' or 'candle'
    ATR_STOP_MULT: float = float(os.getenv("ATR_STOP_MULT", "1.5"))
    ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
    HTF_ALIGN: bool = os.getenv("HTF_ALIGN", "true").lower() in ("1", "true", "yes")
    HTF_EMA_PERIOD: int = int(os.getenv("HTF_EMA_PERIOD", "100"))
    FIRST_BREAKOUT_ONLY: bool = os.getenv("FIRST_BREAKOUT_ONLY", "true").lower() in ("1", "true", "yes")
    # M5 candles fetched to compute ATR/EMA at a breakout. 1000 bars fully
    # converges both series (EMA100 residual weight < 1e-8) in one API request.
    INDICATOR_CANDLE_COUNT: int = int(os.getenv("INDICATOR_CANDLE_COUNT", "1000"))

    # Polling intervals
    PRICE_POLL_INTERVAL: int = int(os.getenv("PRICE_POLL_INTERVAL", "10"))  # seconds
    CANDLE_GRANULARITY: str = "M5"  # 5-minute candles

    # cTrader Open API configuration
    CTRADER_CLIENT_ID: str = os.getenv("CTRADER_CLIENT_ID", "")
    CTRADER_CLIENT_SECRET: str = os.getenv("CTRADER_CLIENT_SECRET", "")
    CTRADER_ACCESS_TOKEN: str = os.getenv("CTRADER_ACCESS_TOKEN", "")
    CTRADER_REFRESH_TOKEN: str = os.getenv("CTRADER_REFRESH_TOKEN", "")
    # ctidTraderAccountId of the trading account (int on the wire)
    CTRADER_ACCOUNT_ID: str = os.getenv("CTRADER_ACCOUNT_ID", "")
    CTRADER_ENVIRONMENT: str = os.getenv("CTRADER_ENVIRONMENT", "demo")  # 'demo' or 'live'
    # Where refreshed tokens are persisted so restarts survive token rotation
    CTRADER_TOKEN_FILE: str = os.getenv("CTRADER_TOKEN_FILE", "ctrader_tokens.json")

    # Internal instrument key -> broker cTrader symbol name.
    # FTMO / FundingPips typically list gold as XAUUSD; the NAS100 CFD name
    # varies by broker (US100 / NAS100 / USTEC) — set per broker in .env.
    SYMBOL_MAP: dict = {
        pair.split("=")[0].strip(): pair.split("=")[1].strip()
        for pair in os.getenv(
            "SYMBOL_MAP", "XAU_USD=XAUUSD,NAS100_USD=US100"
        ).split(",")
        if "=" in pair
    }

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not cls.CTRADER_CLIENT_ID:
            errors.append("CTRADER_CLIENT_ID is not set")
        if not cls.CTRADER_CLIENT_SECRET:
            errors.append("CTRADER_CLIENT_SECRET is not set")
        if not cls.CTRADER_ACCESS_TOKEN:
            errors.append("CTRADER_ACCESS_TOKEN is not set")
        if not cls.CTRADER_ACCOUNT_ID:
            errors.append("CTRADER_ACCOUNT_ID is not set")
        if cls.CTRADER_ENVIRONMENT not in ("demo", "live"):
            errors.append(f"CTRADER_ENVIRONMENT must be 'demo' or 'live', got: {cls.CTRADER_ENVIRONMENT}")
        missing = [i for i in INSTRUMENTS if i not in cls.SYMBOL_MAP]
        if missing:
            errors.append(f"SYMBOL_MAP missing entries for: {missing}")
        if cls.RISK_PER_TRADE <= 0 or cls.RISK_PER_TRADE > 0.1:
            errors.append(f"RISK_PER_TRADE must be between 0 and 0.1, got: {cls.RISK_PER_TRADE}")
        if cls.RR_TARGET <= 0:
            errors.append(f"RR_TARGET must be positive, got: {cls.RR_TARGET}")
        if cls.STOP_MODE not in ("atr", "candle"):
            errors.append(f"STOP_MODE must be 'atr' or 'candle', got: {cls.STOP_MODE}")
        if cls.ATR_STOP_MULT <= 0:
            errors.append(f"ATR_STOP_MULT must be positive, got: {cls.ATR_STOP_MULT}")
        if not ENABLED_SESSIONS:
            errors.append("ENABLED_SESSIONS is empty")

        return errors


# Logging configuration
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "trades.db")


# Timezone
TIMEZONE = "America/New_York"
