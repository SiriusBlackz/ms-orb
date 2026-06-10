"""SQLite trade logging for MS-ORB trading bot."""

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
        """Initialize trade logger.

        Args:
            db_path: Path to SQLite database. Defaults to config DB_PATH.
        """
        self.db_path = db_path or DB_PATH
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Create database and tables if they don't exist."""
        # Ensure directory exists
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create trades table
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

        # Create index for common queries
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
        """Log a trade entry.

        Args:
            instrument: Instrument name.
            session: Session name.
            direction: Trade direction.
            entry_price: Entry price.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            units: Number of units.
            trade_id: OANDA trade ID.
            opening_range_high: Opening range high.
            opening_range_low: Opening range low.

        Returns:
            Database row ID.
        """
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
            f"Trade entry logged: {direction.value} {instrument} {units} units @ {entry_price}"
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
        """Log a trade exit.

        Args:
            row_id: Database row ID from log_entry.
            exit_price: Exit price.
            exit_reason: Exit reason ('stop', 'target', 'time').
            pnl: Profit/loss in account currency.
            rr_achieved: Risk-reward achieved.
        """
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
        """Log a trade exit using OANDA trade ID.

        Args:
            trade_id: OANDA trade ID.
            exit_price: Exit price.
            exit_reason: Exit reason.
            pnl: Profit/loss.
            rr_achieved: Risk-reward achieved.
        """
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
        """Get all open trades (no exit logged).

        Returns:
            List of open trade logs.
        """
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
                id=row[0],
                instrument=row[1],
                session=row[2],
                direction=row[3],
                entry_price=row[4],
                stop_loss=row[5],
                take_profit=row[6],
                units=row[7],
                entry_time=datetime.fromisoformat(row[8]) if row[8] else None,
                exit_time=datetime.fromisoformat(row[9]) if row[9] else None,
                exit_price=row[10],
                exit_reason=row[11],
                pnl=row[12],
                rr_achieved=row[13],
                trade_id=row[14],
                opening_range_high=row[15],
                opening_range_low=row[16],
            ))

        conn.close()
        return trades

    def get_recent_trades(self, limit: int = 50) -> list[TradeLog]:
        """Get recent trades.

        Args:
            limit: Maximum number of trades to return.

        Returns:
            List of trade logs.
        """
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
                id=row[0],
                instrument=row[1],
                session=row[2],
                direction=row[3],
                entry_price=row[4],
                stop_loss=row[5],
                take_profit=row[6],
                units=row[7],
                entry_time=datetime.fromisoformat(row[8]) if row[8] else None,
                exit_time=datetime.fromisoformat(row[9]) if row[9] else None,
                exit_price=row[10],
                exit_reason=row[11],
                pnl=row[12],
                rr_achieved=row[13],
                trade_id=row[14],
                opening_range_high=row[15],
                opening_range_low=row[16],
            ))

        conn.close()
        return trades

    def get_trade_stats(self) -> dict:
        """Get overall trade statistics.

        Returns:
            Dictionary with trade statistics.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL")
        total_trades = cursor.fetchone()[0]

        # Winning trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
        winning_trades = cursor.fetchone()[0]

        # Total PnL
        cursor.execute("SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL")
        total_pnl = cursor.fetchone()[0] or 0

        # Average RR
        cursor.execute("SELECT AVG(rr_achieved) FROM trades WHERE rr_achieved IS NOT NULL")
        avg_rr = cursor.fetchone()[0] or 0

        # By session
        cursor.execute("""
            SELECT session, COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl)
            FROM trades
            WHERE exit_time IS NOT NULL
            GROUP BY session
        """)
        by_session = {
            row[0]: {
                "total": row[1],
                "wins": row[2],
                "pnl": row[3] or 0,
            }
            for row in cursor.fetchall()
        }

        # By instrument
        cursor.execute("""
            SELECT instrument, COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl)
            FROM trades
            WHERE exit_time IS NOT NULL
            GROUP BY instrument
        """)
        by_instrument = {
            row[0]: {
                "total": row[1],
                "wins": row[2],
                "pnl": row[3] or 0,
            }
            for row in cursor.fetchall()
        }

        conn.close()

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "average_rr": avg_rr,
            "by_session": by_session,
            "by_instrument": by_instrument,
        }

    def get_trade_by_id(self, row_id: int) -> Optional[TradeLog]:
        """Get a specific trade by database ID.

        Args:
            row_id: Database row ID.

        Returns:
            TradeLog or None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, instrument, session, direction, entry_price, stop_loss,
                   take_profit, units, entry_time, exit_time, exit_price,
                   exit_reason, pnl, rr_achieved, trade_id,
                   opening_range_high, opening_range_low
            FROM trades
            WHERE id = ?
        """, (row_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return TradeLog(
            id=row[0],
            instrument=row[1],
            session=row[2],
            direction=row[3],
            entry_price=row[4],
            stop_loss=row[5],
            take_profit=row[6],
            units=row[7],
            entry_time=datetime.fromisoformat(row[8]) if row[8] else None,
            exit_time=datetime.fromisoformat(row[9]) if row[9] else None,
            exit_price=row[10],
            exit_reason=row[11],
            pnl=row[12],
            rr_achieved=row[13],
            trade_id=row[14],
            opening_range_high=row[15],
            opening_range_low=row[16],
        )


def calculate_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    units: int,
) -> float:
    """Calculate profit/loss for a trade.

    Args:
        direction: 'LONG' or 'SHORT'.
        entry_price: Entry price.
        exit_price: Exit price.
        units: Number of units (positive).

    Returns:
        Profit/loss in price units.
    """
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
    """Calculate risk-reward achieved.

    Args:
        direction: 'LONG' or 'SHORT'.
        entry_price: Entry price.
        exit_price: Exit price.
        stop_loss: Stop loss price.

    Returns:
        Risk-reward ratio achieved (negative if stopped out).
    """
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return 0

    if direction == "LONG":
        reward = exit_price - entry_price
    else:
        reward = entry_price - exit_price

    return reward / risk


if __name__ == "__main__":
    # Test the trade logger
    logging.basicConfig(level=logging.DEBUG)

    import tempfile
    import os

    # Use a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_trades.db")
        logger_instance = TradeLogger(db_path=test_db)

        # Log a trade entry
        row_id = logger_instance.log_entry(
            instrument="XAU_USD",
            session=SessionName.NY,
            direction=Direction.LONG,
            entry_price=2050.00,
            stop_loss=2045.00,
            take_profit=2075.00,
            units=10,
            trade_id="12345",
            opening_range_high=2048.00,
            opening_range_low=2040.00,
        )
        print(f"Logged entry with ID: {row_id}")

        # Check open trades
        open_trades = logger_instance.get_open_trades()
        print(f"Open trades: {len(open_trades)}")

        # Log exit
        pnl = calculate_pnl("LONG", 2050.00, 2075.00, 10)
        rr = calculate_rr_achieved("LONG", 2050.00, 2075.00, 2045.00)

        logger_instance.log_exit(
            row_id=row_id,
            exit_price=2075.00,
            exit_reason="target",
            pnl=pnl,
            rr_achieved=rr,
        )
        print(f"PnL: {pnl}, RR: {rr}")

        # Get stats
        stats = logger_instance.get_trade_stats()
        print(f"\nTrade Stats: {stats}")

        # Get recent trades
        recent = logger_instance.get_recent_trades(limit=10)
        print(f"\nRecent trades: {len(recent)}")
        for trade in recent:
            print(f"  {trade.direction} {trade.instrument} @ {trade.entry_price} -> {trade.exit_price}")
