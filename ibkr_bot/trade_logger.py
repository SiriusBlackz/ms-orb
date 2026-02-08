"""SQLite trade logging for MS-ORB IBKR trading bot."""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DB_PATH, SessionName
from strategy import Direction

logger = logging.getLogger(__name__)


@dataclass
class TradeLog:
    """Trade log entry for database."""
    id: Optional[int]
    instrument: str
    session: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    units: int
    entry_time: datetime
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    exit_reason: Optional[str]
    pnl: Optional[float]
    rr_achieved: Optional[float]
    trade_id: Optional[str]
    opening_range_high: Optional[float]
    opening_range_low: Optional[float]


class TradeLogger:
    """SQLite-based trade logger."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Create database and tables if they don't exist."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT NOT NULL,
                session TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                units INTEGER NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl REAL,
                rr_achieved REAL,
                trade_id TEXT,
                opening_range_high REAL,
                opening_range_low REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_instrument
            ON trades(instrument)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_session
            ON trades(session)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time
            ON trades(entry_time)
        """)

        conn.commit()
        conn.close()

        logger.info(f"Trade database initialized: {self.db_path}")

    def log_entry(
        self,
        instrument: str,
        session: SessionName,
        direction: Direction,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        units: int,
        trade_id: str = None,
        opening_range_high: float = None,
        opening_range_low: float = None,
    ) -> int:
        """Log a trade entry. Returns database row ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                instrument, session, direction, entry_price, stop_loss,
                take_profit, units, entry_time, trade_id,
                opening_range_high, opening_range_low
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            instrument,
            session.value,
            direction.value,
            entry_price,
            stop_loss,
            take_profit,
            units,
            datetime.utcnow().isoformat(),
            trade_id,
            opening_range_high,
            opening_range_low,
        ))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(
            f"Trade entry logged: {direction.value} {instrument} {units} shares @ {entry_price}"
        )

        return row_id

    def log_exit(
        self,
        row_id: int,
        exit_price: float,
        exit_reason: str,
        pnl: float = None,
        rr_achieved: float = None,
    ) -> None:
        """Log a trade exit."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE trades
            SET exit_time = ?, exit_price = ?, exit_reason = ?,
                pnl = ?, rr_achieved = ?
            WHERE id = ?
        """, (
            datetime.utcnow().isoformat(),
            exit_price,
            exit_reason,
            pnl,
            rr_achieved,
            row_id,
        ))

        conn.commit()
        conn.close()

        logger.info(
            f"Trade exit logged: id={row_id} exit={exit_price} reason={exit_reason} "
            f"pnl={pnl:.2f if pnl else 'N/A'} rr={rr_achieved:.2f if rr_achieved else 'N/A'}"
        )

    def log_exit_by_trade_id(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float = None,
        rr_achieved: float = None,
    ) -> None:
        """Log a trade exit using IBKR order ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE trades
            SET exit_time = ?, exit_price = ?, exit_reason = ?,
                pnl = ?, rr_achieved = ?
            WHERE trade_id = ? AND exit_time IS NULL
        """, (
            datetime.utcnow().isoformat(),
            exit_price,
            exit_reason,
            pnl,
            rr_achieved,
            trade_id,
        ))

        conn.commit()
        conn.close()

        logger.info(
            f"Trade exit logged by trade_id: {trade_id} exit={exit_price} reason={exit_reason}"
        )

    def get_open_trades(self) -> list[TradeLog]:
        """Get all open trades (no exit logged)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, instrument, session, direction, entry_price, stop_loss,
                   take_profit, units, entry_time, exit_time, exit_price,
                   exit_reason, pnl, rr_achieved, trade_id,
                   opening_range_high, opening_range_low
            FROM trades
            WHERE exit_time IS NULL
            ORDER BY entry_time DESC
        """)

        trades = []
        for row in cursor.fetchall():
            trades.append(TradeLog(
                id=row[0], instrument=row[1], session=row[2],
                direction=row[3], entry_price=row[4], stop_loss=row[5],
                take_profit=row[6], units=row[7],
                entry_time=datetime.fromisoformat(row[8]) if row[8] else None,
                exit_time=datetime.fromisoformat(row[9]) if row[9] else None,
                exit_price=row[10], exit_reason=row[11],
                pnl=row[12], rr_achieved=row[13], trade_id=row[14],
                opening_range_high=row[15], opening_range_low=row[16],
            ))

        conn.close()
        return trades

    def get_recent_trades(self, limit: int = 50) -> list[TradeLog]:
        """Get recent trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, instrument, session, direction, entry_price, stop_loss,
                   take_profit, units, entry_time, exit_time, exit_price,
                   exit_reason, pnl, rr_achieved, trade_id,
                   opening_range_high, opening_range_low
            FROM trades
            ORDER BY entry_time DESC
            LIMIT ?
        """, (limit,))

        trades = []
        for row in cursor.fetchall():
            trades.append(TradeLog(
                id=row[0], instrument=row[1], session=row[2],
                direction=row[3], entry_price=row[4], stop_loss=row[5],
                take_profit=row[6], units=row[7],
                entry_time=datetime.fromisoformat(row[8]) if row[8] else None,
                exit_time=datetime.fromisoformat(row[9]) if row[9] else None,
                exit_price=row[10], exit_reason=row[11],
                pnl=row[12], rr_achieved=row[13], trade_id=row[14],
                opening_range_high=row[15], opening_range_low=row[16],
            ))

        conn.close()
        return trades

    def get_trade_stats(self) -> dict:
        """Get overall trade statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL")
        total_trades = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
        winning_trades = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL")
        total_pnl = cursor.fetchone()[0] or 0

        cursor.execute("SELECT AVG(rr_achieved) FROM trades WHERE rr_achieved IS NOT NULL")
        avg_rr = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "average_rr": avg_rr,
        }


def calculate_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    units: int,
) -> float:
    """Calculate profit/loss for a trade."""
    if direction == "LONG":
        return (exit_price - entry_price) * abs(units)
    else:
        return (entry_price - exit_price) * abs(units)


def calculate_rr_achieved(
    direction: str,
    entry_price: float,
    exit_price: float,
    stop_loss: float,
) -> float:
    """Calculate risk-reward achieved."""
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return 0

    if direction == "LONG":
        reward = exit_price - entry_price
    else:
        reward = entry_price - exit_price

    return reward / risk
