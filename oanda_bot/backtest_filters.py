#!/usr/bin/env python3
"""MS-ORB Filter Optimization Backtest.

Tests range size, time-to-break, trend alignment (EMA), and breakout candle
size filters against historical M5 data from pickle cache.

Usage:
    python backtest_filters.py [--instruments XAU_USD,NAS100_USD]
                               [--sessions TOKYO,LONDON,NY]
                               [--rr-target 5.0] [--verbose]
"""

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytz

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SESSIONS,
    TIMEZONE,
    SessionConfig,
    SessionName,
    TradingConfig,
)
from oanda_client import Candle
from strategy import (
    BreakoutDetector,
    Direction,
    OpeningRange,
    RangeRecorder,
)

logger = logging.getLogger(__name__)

NY_TZ = pytz.timezone(TIMEZONE)
CACHE_DIR = Path(__file__).parent / "cache"


# ─────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────


@dataclass
class TradeResult:
    """Single simulated trade with filter metadata."""
    instrument: str
    session: SessionName
    trade_date: date
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float
    exit_reason: str          # "stop", "target", "time"
    rr_achieved: float
    entry_time: datetime
    exit_time: datetime
    # Filter metadata
    range_size_pct: float     # (H-L)/L * 100
    time_to_break_mins: float # minutes from range_end to breakout
    ema_aligned: bool         # price vs 6-period EMA at breakout
    breakout_candle_pct: float  # breakout candle range as % of opening range


# ─────────────────────────────────────────────────────────────
#  Time helpers (duplicated from backtest.py to stay standalone)
# ─────────────────────────────────────────────────────────────


def candle_ny_time(candle: Candle) -> datetime:
    utc_time = candle.time
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    return utc_time.astimezone(NY_TZ)


def assign_session_date(ny_time: datetime, session_cfg: SessionConfig) -> date:
    t = ny_time.time()
    if session_cfg.name == SessionName.TOKYO:
        if t >= session_cfg.range_start:
            return ny_time.date()
        elif t <= session_cfg.session_close:
            return (ny_time - timedelta(days=1)).date()
        else:
            return ny_time.date()
    else:
        return ny_time.date()


def is_in_session_window(ny_time: datetime, session_cfg: SessionConfig) -> bool:
    t = ny_time.time()
    if session_cfg.name == SessionName.TOKYO:
        return t >= session_cfg.range_start or t <= session_cfg.session_close
    else:
        return session_cfg.range_start <= t <= session_cfg.session_close


def is_in_range_period(ny_time: datetime, session_cfg: SessionConfig) -> bool:
    t = ny_time.time()
    return session_cfg.range_start <= t < session_cfg.range_end


def is_in_entry_window(ny_time: datetime, session_cfg: SessionConfig) -> bool:
    t = ny_time.time()
    return session_cfg.range_end <= t < session_cfg.entry_end


def is_past_session_close(ny_time: datetime, session_cfg: SessionConfig) -> bool:
    t = ny_time.time()
    if session_cfg.name == SessionName.TOKYO:
        if t <= session_cfg.session_close:
            return False
        elif t < session_cfg.range_start:
            return True
        else:
            return False
    else:
        return t >= session_cfg.session_close


def group_candles_by_session_day(
    candles: list[Candle],
    sessions: list[SessionConfig],
) -> dict[tuple[SessionName, date], list[Candle]]:
    groups: dict[tuple[SessionName, date], list[Candle]] = {}
    for candle in candles:
        if not candle.complete:
            continue
        ny_time = candle_ny_time(candle)
        for session_cfg in sessions:
            if is_in_session_window(ny_time, session_cfg):
                session_date = assign_session_date(ny_time, session_cfg)
                key = (session_cfg.name, session_date)
                if key not in groups:
                    groups[key] = []
                groups[key].append(candle)
    for key in groups:
        groups[key].sort(key=lambda c: c.time)
    return groups


# ─────────────────────────────────────────────────────────────
#  Trade simulation
# ─────────────────────────────────────────────────────────────


def _calc_rr(direction: Direction, entry: float, exit_price: float, risk: float) -> float:
    if risk <= 0:
        return 0.0
    if direction == Direction.LONG:
        return (exit_price - entry) / risk
    else:
        return (entry - exit_price) / risk


def simulate_trade(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: Direction,
    risk: float,
    candles_after_entry: list[Candle],
    session_cfg: SessionConfig,
) -> Optional[tuple[float, str, datetime]]:
    """Walk candles after entry and return (exit_price, reason, exit_time)."""
    for candle in candles_after_entry:
        ny_time = candle_ny_time(candle)

        if is_past_session_close(ny_time, session_cfg):
            return candle.open, "time", candle.time

        if direction == Direction.LONG:
            sl_hit = candle.low <= stop_loss
            tp_hit = candle.high >= take_profit
        else:
            sl_hit = candle.high >= stop_loss
            tp_hit = candle.low <= take_profit

        if sl_hit and tp_hit:
            dist_sl = abs(candle.open - stop_loss)
            dist_tp = abs(candle.open - take_profit)
            if dist_sl <= dist_tp:
                return stop_loss, "stop", candle.time
            else:
                return take_profit, "target", candle.time
        elif sl_hit:
            return stop_loss, "stop", candle.time
        elif tp_hit:
            return take_profit, "target", candle.time

    if candles_after_entry:
        last = candles_after_entry[-1]
        return last.close, "time", last.time
    return None


# ─────────────────────────────────────────────────────────────
#  EMA calculation
# ─────────────────────────────────────────────────────────────


def compute_ema_series(candles: list[Candle], period: int = 6) -> dict[datetime, float]:
    """Compute EMA on close prices. Returns {candle.time: ema_value}."""
    if len(candles) < period:
        return {}

    ema_map: dict[datetime, float] = {}
    multiplier = 2.0 / (period + 1)

    # Seed with SMA of first `period` candles
    sma = sum(c.close for c in candles[:period]) / period
    ema_map[candles[period - 1].time] = sma

    ema = sma
    for candle in candles[period:]:
        ema = (candle.close - ema) * multiplier + ema
        ema_map[candle.time] = ema

    return ema_map


# ─────────────────────────────────────────────────────────────
#  Core simulation — collect all trades with metadata
# ─────────────────────────────────────────────────────────────


def collect_all_trades(
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]],
    instrument_ema: dict[str, dict[datetime, float]],
    session_configs: dict[SessionName, SessionConfig],
    rr_target: float,
    max_trades_per_session: int = 2,
    verbose: bool = False,
) -> list[TradeResult]:
    """Run simulation across all instruments/sessions, returning every trade with metadata."""
    all_trades: list[TradeResult] = []

    for inst, groups in instrument_groups.items():
        ema_map = instrument_ema.get(inst, {})

        sorted_items = sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0].value))

        for (session_name, session_date), candles in sorted_items:
            if session_name not in session_configs:
                continue

            session_cfg = session_configs[session_name]

            range_candles = []
            post_range_candles = []
            for candle in candles:
                ny_time = candle_ny_time(candle)
                if is_in_range_period(ny_time, session_cfg):
                    range_candles.append(candle)
                else:
                    post_range_candles.append(candle)

            if not range_candles:
                continue

            # Build opening range
            recorder = RangeRecorder(session_name, inst)
            for candle in range_candles:
                recorder.add_candle(candle)
            opening_range = recorder.finalize()
            if opening_range is None:
                continue

            range_size = opening_range.high - opening_range.low
            range_size_pct = (range_size / opening_range.low) * 100 if opening_range.low > 0 else 0.0

            # Compute range_end as a datetime for time-to-break measurement
            # Use the last range candle's time + 5 min as proxy
            range_end_utc = range_candles[-1].time + timedelta(minutes=5)

            # Detect breakouts and simulate
            detector = BreakoutDetector(opening_range, rr_target=rr_target)
            trades_taken = 0
            i = 0

            while i < len(post_range_candles) and trades_taken < max_trades_per_session:
                candle = post_range_candles[i]
                ny_time = candle_ny_time(candle)

                if is_in_entry_window(ny_time, session_cfg):
                    signal = detector.check_breakout(candle)
                else:
                    signal = None

                if signal:
                    # --- Compute filter metadata ---

                    # Time to break (minutes from range end)
                    time_to_break = (candle.time - range_end_utc).total_seconds() / 60.0
                    time_to_break = max(0.0, time_to_break)

                    # EMA alignment at breakout candle
                    ema_val = ema_map.get(candle.time)
                    if ema_val is not None:
                        if signal.direction == Direction.LONG:
                            ema_aligned = candle.close > ema_val
                        else:
                            ema_aligned = candle.close < ema_val
                    else:
                        ema_aligned = True  # no data = no filter

                    # Breakout candle size as % of opening range
                    breakout_candle_range = candle.high - candle.low
                    if range_size > 0:
                        breakout_candle_pct = (breakout_candle_range / range_size) * 100
                    else:
                        breakout_candle_pct = 0.0

                    # --- Simulate trade ---
                    remaining = post_range_candles[i + 1:]
                    result = simulate_trade(
                        signal.entry_price,
                        signal.stop_loss,
                        signal.take_profit,
                        signal.direction,
                        signal.risk_per_unit,
                        remaining,
                        session_cfg,
                    )

                    if result:
                        exit_price, exit_reason, exit_time = result
                        rr = _calc_rr(signal.direction, signal.entry_price, exit_price, signal.risk_per_unit)

                        trade = TradeResult(
                            instrument=inst,
                            session=session_name,
                            trade_date=session_date,
                            direction=signal.direction,
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            rr_achieved=rr,
                            entry_time=candle.time,
                            exit_time=exit_time,
                            range_size_pct=range_size_pct,
                            time_to_break_mins=time_to_break,
                            ema_aligned=ema_aligned,
                            breakout_candle_pct=breakout_candle_pct,
                        )
                        all_trades.append(trade)
                        trades_taken += 1

                        if verbose:
                            _print_trade(trade)

                        if exit_reason == "stop":
                            detector.reset_direction(signal.direction)
                            while i < len(post_range_candles) and post_range_candles[i].time <= exit_time:
                                i += 1
                            continue
                        else:
                            break

                i += 1

    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


def _print_trade(t: TradeResult) -> None:
    print(
        f"  {t.trade_date} {t.session.value:6s} {t.instrument:11s} "
        f"{t.direction.value:5s} entry={t.entry_price:>10.2f} "
        f"exit={t.exit_price:>10.2f} reason={t.exit_reason:6s} "
        f"RR={t.rr_achieved:+.2f}  range%={t.range_size_pct:.3f} "
        f"ttb={t.time_to_break_mins:.0f}m ema={'Y' if t.ema_aligned else 'N'} "
        f"bkout%={t.breakout_candle_pct:.0f}"
    )


# ─────────────────────────────────────────────────────────────
#  Filter analysis and reporting
# ─────────────────────────────────────────────────────────────


def _band_stats(trades: list[TradeResult]) -> tuple[int, float, float]:
    """Return (count, win_rate, avg_rr) for a list of trades."""
    n = len(trades)
    if n == 0:
        return 0, 0.0, 0.0
    wins = sum(1 for t in trades if t.rr_achieved > 0)
    avg_rr = sum(t.rr_achieved for t in trades) / n
    return n, wins / n, avg_rr


def _total_r(trades: list[TradeResult]) -> float:
    return sum(t.rr_achieved for t in trades)


def report_range_size_filter(trades: list[TradeResult]) -> None:
    bands = [
        ("0.1-0.3%", 0.1, 0.3),
        ("0.3-0.5%", 0.3, 0.5),
        ("0.5-0.8%", 0.5, 0.8),
        ("0.8%+",    0.8, float("inf")),
    ]

    print("\n" + "=" * 80)
    print("  FILTER: Range Size  (opening range H-L as % of price)")
    print("=" * 80)
    print(f"  {'Band':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'Stops':>6s} {'Targets':>8s}")
    print("  " + "-" * 68)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.range_size_pct < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        stops = sum(1 for t in subset if t.exit_reason == "stop")
        targets = sum(1 for t in subset if t.exit_reason == "target")
        print(f"  {label:<12s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {stops:>6d} {targets:>8d}")

    # Also show <0.1% if any
    tiny = [t for t in trades if t.range_size_pct < 0.1]
    if tiny:
        n, wr, avg_rr = _band_stats(tiny)
        total = _total_r(tiny)
        print(f"  {'<0.1%':<12s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")

    print()


def report_time_to_break_filter(trades: list[TradeResult]) -> None:
    bands = [
        ("0-30 mins",   0,  30),
        ("30-60 mins",  30, 60),
        ("60-90 mins",  60, 90),
        ("90-120 mins", 90, 120),
        ("120+ mins",   120, float("inf")),
    ]

    print("=" * 80)
    print("  FILTER: Time to Break  (minutes from range close to first breakout)")
    print("=" * 80)
    print(f"  {'Band':<14s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s}")
    print("  " + "-" * 52)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.time_to_break_mins < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        print(f"  {label:<14s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")

    print()


def report_trend_alignment_filter(trades: list[TradeResult]) -> None:
    aligned = [t for t in trades if t.ema_aligned]
    not_aligned = [t for t in trades if not t.ema_aligned]

    print("=" * 80)
    print("  FILTER: Trend Alignment  (6-period EMA on 5min as proxy for 30min EMA)")
    print("  Long only if price > EMA, Short only if price < EMA")
    print("=" * 80)
    print(f"  {'Group':<20s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 64)

    for label, subset in [("All trades", trades), ("EMA-aligned", aligned), ("Counter-EMA", not_aligned)]:
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        gross_p = sum(t.rr_achieved for t in subset if t.rr_achieved > 0)
        gross_l = abs(sum(t.rr_achieved for t in subset if t.rr_achieved <= 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        print(f"  {label:<20s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    # Breakdown by direction
    print()
    print(f"  {'Direction breakdown':<20s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s}")
    print("  " + "-" * 56)

    for dir_label, direction in [("LONG", Direction.LONG), ("SHORT", Direction.SHORT)]:
        dir_trades = [t for t in trades if t.direction == direction]
        dir_aligned = [t for t in dir_trades if t.ema_aligned]
        dir_counter = [t for t in dir_trades if not t.ema_aligned]

        n, wr, avg_rr = _band_stats(dir_trades)
        total = _total_r(dir_trades)
        print(f"  {dir_label + ' all':<20s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")

        n, wr, avg_rr = _band_stats(dir_aligned)
        total = _total_r(dir_aligned)
        print(f"  {dir_label + ' EMA-aligned':<20s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")

        n, wr, avg_rr = _band_stats(dir_counter)
        total = _total_r(dir_counter)
        print(f"  {dir_label + ' counter-EMA':<20s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")

    print()


def report_breakout_candle_size_filter(trades: list[TradeResult]) -> None:
    small = [t for t in trades if t.breakout_candle_pct < 50]
    large = [t for t in trades if t.breakout_candle_pct >= 50]

    bands = [
        ("<25%",    0,  25),
        ("25-50%",  25, 50),
        ("50-100%", 50, 100),
        ("100-150%", 100, 150),
        ("150%+",   150, float("inf")),
    ]

    print("=" * 80)
    print("  FILTER: Breakout Candle Size  (candle H-L as % of opening range)")
    print("  Hypothesis: smaller candles = tighter SL = better RR")
    print("=" * 80)
    print(f"  {'Band':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'Avg SL%':>8s}")
    print("  " + "-" * 58)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.breakout_candle_pct < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        # Average SL distance as % of entry
        if subset:
            avg_sl_pct = sum(abs(t.entry_price - t.stop_loss) / t.entry_price * 100 for t in subset) / len(subset)
        else:
            avg_sl_pct = 0.0
        print(f"  {label:<12s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {avg_sl_pct:>7.3f}%")

    print()
    print(f"  Summary: <50% range → {len(small)} trades | >=50% range → {len(large)} trades")
    n_s, wr_s, avg_s = _band_stats(small)
    n_l, wr_l, avg_l = _band_stats(large)
    print(f"    <50%:  win={wr_s:.1%} avg_rr={avg_s:+.2f} total_r={_total_r(small):+.2f}")
    print(f"    >=50%: win={wr_l:.1%} avg_rr={avg_l:+.2f} total_r={_total_r(large):+.2f}")
    print()


def report_instrument_session_breakdown(trades: list[TradeResult]) -> None:
    print("=" * 80)
    print("  BREAKDOWN: Instrument x Session")
    print("=" * 80)
    print(f"  {'Instrument':<12s} {'Session':<8s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s}")
    print("  " + "-" * 58)

    instruments = sorted(set(t.instrument for t in trades))
    sessions = [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]

    for inst in instruments:
        for sess in sessions:
            subset = [t for t in trades if t.instrument == inst and t.session == sess]
            if not subset:
                continue
            n, wr, avg_rr = _band_stats(subset)
            total = _total_r(subset)
            print(f"  {inst:<12s} {sess.value:<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")

    print("  " + "-" * 58)
    n, wr, avg_rr = _band_stats(trades)
    total = _total_r(trades)
    print(f"  {'TOTAL':<12s} {'ALL':<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")
    print()


def report_combined_filter(trades: list[TradeResult]) -> None:
    """Test the best combination: EMA-aligned + range size sweet spot."""
    print("=" * 80)
    print("  COMBINED FILTERS: Best Range Band + EMA Aligned")
    print("=" * 80)

    range_bands = [
        ("0.1-0.3%", 0.1, 0.3),
        ("0.3-0.5%", 0.3, 0.5),
        ("0.5-0.8%", 0.5, 0.8),
    ]

    print(f"  {'Range + EMA':<24s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s}")
    print("  " + "-" * 60)

    for label, lo, hi in range_bands:
        range_subset = [t for t in trades if lo <= t.range_size_pct < hi]
        aligned = [t for t in range_subset if t.ema_aligned]

        n_all, wr_all, avg_all = _band_stats(range_subset)
        total_all = _total_r(range_subset)
        print(f"  {label + ' all':<24s} {n_all:>7d} {wr_all:>6.1%} {avg_all:>+8.2f} {total_all:>+9.2f}")

        n_a, wr_a, avg_a = _band_stats(aligned)
        total_a = _total_r(aligned)
        print(f"  {label + ' + EMA':<24s} {n_a:>7d} {wr_a:>6.1%} {avg_a:>+8.2f} {total_a:>+9.2f}")

    print()


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────


def load_candles_from_cache(instrument: str) -> Optional[list[Candle]]:
    """Load candle data from the most recent pickle file for an instrument."""
    pattern = f"{instrument}_*_M5.pkl"
    matches = sorted(CACHE_DIR.glob(pattern))
    if not matches:
        return None
    cache_file = matches[-1]  # most recent
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-ORB Filter Optimization Backtest")
    parser.add_argument(
        "--instruments", type=str, default="XAU_USD,NAS100_USD",
        help="Comma-separated instruments (default: XAU_USD,NAS100_USD)",
    )
    parser.add_argument(
        "--sessions", type=str, default="TOKYO,LONDON,NY",
        help="Comma-separated sessions (default: TOKYO,LONDON,NY)",
    )
    parser.add_argument(
        "--rr-target", type=float, default=5.0,
        help="Risk-reward target (default: 5.0)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print individual trade details",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.WARNING  # suppress strategy debug spam
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    instruments = [s.strip() for s in args.instruments.split(",")]
    session_names = []
    for s in args.sessions.split(","):
        s = s.strip().upper()
        try:
            session_names.append(SessionName(s))
        except ValueError:
            print(f"Unknown session: {s}. Valid: TOKYO, LONDON, NY")
            sys.exit(1)

    session_configs = {name: SESSIONS[name] for name in session_names}

    print()
    print("=" * 80)
    print("  MS-ORB FILTER OPTIMIZATION BACKTEST")
    print("=" * 80)
    print(f"  Instruments: {', '.join(instruments)}")
    print(f"  Sessions:    {', '.join(s.value for s in session_names)}")
    print(f"  RR Target:   {args.rr_target}")
    print()

    # Load data
    instrument_candles: dict[str, list[Candle]] = {}
    for inst in instruments:
        candles = load_candles_from_cache(inst)
        if candles is None:
            print(f"  WARNING: No cached data for {inst} in {CACHE_DIR}")
            continue
        instrument_candles[inst] = candles
        first_ny = candle_ny_time(candles[0])
        last_ny = candle_ny_time(candles[-1])
        print(f"  {inst}: {len(candles)} candles ({first_ny.date()} to {last_ny.date()})")

    if not instrument_candles:
        print("ERROR: No data available. Run backtest.py first to cache data.")
        sys.exit(1)

    # Pre-compute EMA series (6-period on 5-min ≈ 30-period on 30-min)
    instrument_ema: dict[str, dict[datetime, float]] = {}
    for inst, candles in instrument_candles.items():
        instrument_ema[inst] = compute_ema_series(candles, period=6)

    # Group candles by session day
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]] = {}
    for inst, candles in instrument_candles.items():
        groups = group_candles_by_session_day(candles, list(session_configs.values()))
        instrument_groups[inst] = groups
        print(f"  {inst}: {len(groups)} session-days")

    print()
    print("  Running simulation...")

    # Collect all trades with filter metadata
    trades = collect_all_trades(
        instrument_groups=instrument_groups,
        instrument_ema=instrument_ema,
        session_configs=session_configs,
        rr_target=args.rr_target,
        max_trades_per_session=TradingConfig.MAX_TRADES_PER_SESSION,
        verbose=args.verbose,
    )

    print(f"  Total trades simulated: {len(trades)}")
    print()

    if not trades:
        print("No trades found. Check your data and session configuration.")
        return

    # Print all filter reports
    report_instrument_session_breakdown(trades)
    report_range_size_filter(trades)
    report_time_to_break_filter(trades)
    report_trend_alignment_filter(trades)
    report_breakout_candle_size_filter(trades)
    report_combined_filter(trades)


if __name__ == "__main__":
    main()
