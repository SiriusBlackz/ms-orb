"""Load and merge trade data from OANDA and IBKR SQLite databases."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Default DB paths relative to this file's directory
_ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_OANDA_DB = _ANALYSIS_DIR / ".." / "oanda_bot" / "trades.db"
DEFAULT_IBKR_DB = _ANALYSIS_DIR / ".." / "ibkr_bot" / "ibkr_trades.db"

COLUMNS = [
    "id", "instrument", "session", "direction", "entry_price", "stop_loss",
    "take_profit", "units", "entry_time", "exit_time", "exit_price",
    "exit_reason", "pnl", "rr_achieved", "trade_id",
    "opening_range_high", "opening_range_low", "created_at",
]


def _read_db(db_path: Path, source: str) -> pd.DataFrame:
    """Read trades from a single SQLite database.

    Returns an empty DataFrame with correct columns if the DB doesn't exist
    or has no trades table.
    """
    if not db_path.exists():
        df = pd.DataFrame(columns=COLUMNS + ["source"])
        return df

    try:
        conn = sqlite3.connect(str(db_path))
        # Check if trades table exists
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'",
            conn,
        )
        if tables.empty:
            conn.close()
            return pd.DataFrame(columns=COLUMNS + ["source"])

        df = pd.read_sql_query(
            "SELECT * FROM trades WHERE exit_time IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception:
        return pd.DataFrame(columns=COLUMNS + ["source"])

    df["source"] = source
    return df


def load_trades(
    oanda_db: str = None,
    ibkr_db: str = None,
    period: str = "all",
    instrument: str = "all",
    session: str = "all",
) -> pd.DataFrame:
    """Load and merge trades from both databases with optional filtering.

    Args:
        oanda_db: Path to OANDA trades DB. Uses default if None.
        ibkr_db: Path to IBKR trades DB. Uses default if None.
        period: Time filter - "all", "7d", "30d", "90d", "1y", etc.
        instrument: Filter by instrument name, or "all".
        session: Filter by session name, or "all".

    Returns:
        Combined DataFrame of closed trades, sorted by entry_time.
    """
    oanda_path = Path(oanda_db) if oanda_db else DEFAULT_OANDA_DB
    ibkr_path = Path(ibkr_db) if ibkr_db else DEFAULT_IBKR_DB

    oanda_df = _read_db(oanda_path, "oanda")
    ibkr_df = _read_db(ibkr_path, "ibkr")

    df = pd.concat([oanda_df, ibkr_df], ignore_index=True)

    if df.empty:
        return df

    # Parse datetime columns
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")

    # Ensure numeric columns
    for col in ["entry_price", "stop_loss", "take_profit", "units",
                 "exit_price", "pnl", "rr_achieved",
                 "opening_range_high", "opening_range_low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by entry time
    df = df.sort_values("entry_time").reset_index(drop=True)

    # Filter by period
    if period != "all":
        df = _filter_period(df, period)

    # Filter by instrument
    if instrument.lower() != "all":
        df = df[df["instrument"].str.upper() == instrument.upper()]

    # Filter by session
    if session.lower() != "all":
        df = df[df["session"].str.upper() == session.upper()]

    return df.reset_index(drop=True)


def _filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filter DataFrame by time period string like '7d', '30d', '1y'."""
    if df.empty:
        return df

    now = datetime.utcnow()
    period = period.strip().lower()

    if period.endswith("d"):
        days = int(period[:-1])
        cutoff = now - timedelta(days=days)
    elif period.endswith("w"):
        weeks = int(period[:-1])
        cutoff = now - timedelta(weeks=weeks)
    elif period.endswith("m"):
        months = int(period[:-1])
        cutoff = now - timedelta(days=months * 30)
    elif period.endswith("y"):
        years = int(period[:-1])
        cutoff = now - timedelta(days=years * 365)
    else:
        return df

    return df[df["entry_time"] >= cutoff]
