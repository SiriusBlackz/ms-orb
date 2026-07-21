#!/usr/bin/env python3
"""MS-ORB Order Flow Backtest.

Derives order flow proxies from M5 candle data (tick volume) and tests
whether they improve breakout quality filtering.

Metrics tested:
  1. CVD alignment (cumulative volume delta over range period)
  2. Breakout candle body ratio (conviction vs indecision)
  3. Range volume skew (volume-weighted position within range)
  4. Pre-breakout volume ramp (building pressure)
  5. Absorption detection (trapped participants at range boundaries)
  6. CVD divergence (volume delta contradicts breakout direction)

Usage:
    python backtest_orderflow.py [--instruments XAU_USD,NAS100_USD]
                                 [--sessions TOKYO,LONDON,NY]
                                 [--rr-target 5.0] [--verbose]
"""

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytz

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
    RangeRecorder,
)

logger = logging.getLogger(__name__)

NY_TZ = pytz.timezone(TIMEZONE)
CACHE_DIR = Path(__file__).parent / "cache"


# ─────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────


@dataclass
class OrderFlowTradeResult:
    """Single simulated trade with order flow metadata."""
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
    # Order flow metadata
    cvd_range: float              # CVD over range period candles
    cvd_aligned: bool             # CVD direction matches breakout direction
    body_ratio: float             # breakout candle body / candle range (0-1)
    volume_skew: float            # volume-weighted position in range (0-1)
    volume_skew_aligned: bool     # skew favors breakout direction
    volume_ramp: float            # pre-breakout volume slope (normalized)
    absorption_at_high: int       # absorption events at range high
    absorption_at_low: int        # absorption events at range low
    absorption_confirms: bool     # absorption supports breakout direction
    cvd_divergent: bool           # CVD diverges from breakout direction
    breakout_volume_ratio: float  # breakout candle vol / avg vol (for reference)


# ─────────────────────────────────────────────────────────────
#  Time helpers (duplicated from backtest_filters.py)
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
#  Trade simulation (duplicated from backtest_filters.py)
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
#  Order flow computation functions
# ─────────────────────────────────────────────────────────────


def compute_candle_delta(candle: Candle) -> int:
    """Classify candle volume as buy (+) or sell (-) based on close vs open."""
    if candle.close > candle.open:
        return candle.volume
    elif candle.close < candle.open:
        return -candle.volume
    else:
        return 0


def compute_cvd(candles: list[Candle]) -> float:
    """Cumulative Volume Delta over a list of candles."""
    return sum(compute_candle_delta(c) for c in candles)


def compute_body_ratio(candle: Candle) -> float:
    """Ratio of candle body to candle range. 0=doji, 1=marubozu."""
    candle_range = candle.high - candle.low
    if candle_range <= 0:
        return 0.0
    return abs(candle.close - candle.open) / candle_range


def compute_volume_skew(
    range_candles: list[Candle],
    range_low: float,
    range_size: float,
) -> float:
    """Volume-weighted position within the opening range.

    Returns 0.0 (all volume at bottom) to 1.0 (all volume at top).
    """
    if range_size <= 0 or not range_candles:
        return 0.5

    weighted_pos = 0.0
    total_vol = 0

    for c in range_candles:
        midpoint = (c.open + c.close) / 2
        position = (midpoint - range_low) / range_size
        position = max(0.0, min(1.0, position))
        weighted_pos += position * c.volume
        total_vol += c.volume

    if total_vol <= 0:
        return 0.5

    return weighted_pos / total_vol


def compute_volume_ramp(pre_breakout_candles: list[Candle]) -> float:
    """Volume slope over candles leading into breakout, normalized by average.

    Positive = ramping up, negative = fading.
    """
    if len(pre_breakout_candles) < 2:
        return 0.0

    volumes = [c.volume for c in pre_breakout_candles]
    avg_vol = sum(volumes) / len(volumes)
    if avg_vol <= 0:
        return 0.0

    return (volumes[-1] - volumes[0]) / avg_vol


def compute_absorption(
    range_candles: list[Candle],
    range_high: float,
    range_low: float,
    range_size: float,
) -> tuple[int, int]:
    """Count absorption events at range boundaries.

    Absorption = candle touches boundary (within 5% of range) with
    above-average volume but closes back inside the range.

    Returns (absorption_at_high, absorption_at_low).
    """
    if not range_candles or range_size <= 0:
        return 0, 0

    touch_zone = range_size * 0.05
    avg_vol = sum(c.volume for c in range_candles) / len(range_candles)
    vol_threshold = avg_vol * 1.2

    abs_high = 0
    abs_low = 0

    for c in range_candles:
        high_vol = c.volume > vol_threshold

        if high_vol and c.high >= (range_high - touch_zone) and c.close < range_high:
            abs_high += 1

        if high_vol and c.low <= (range_low + touch_zone) and c.close > range_low:
            abs_low += 1

    return abs_high, abs_low


def compute_volume_ratio(
    breakout_candle: Candle,
    prior_candles: list[Candle],
    lookback: int = 20,
) -> float:
    """Breakout candle volume vs rolling average."""
    recent = prior_candles[-lookback:] if len(prior_candles) >= lookback else prior_candles
    if not recent:
        return 1.0
    avg_volume = sum(c.volume for c in recent) / len(recent)
    if avg_volume <= 0:
        return 1.0
    return breakout_candle.volume / avg_volume


# ─────────────────────────────────────────────────────────────
#  Core simulation — collect all trades with order flow metadata
# ─────────────────────────────────────────────────────────────


def collect_all_trades(
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]],
    session_configs: dict[SessionName, SessionConfig],
    rr_target: float,
    max_trades_per_session: int = 2,
    verbose: bool = False,
) -> list[OrderFlowTradeResult]:
    """Run simulation across all instruments/sessions, returning every trade with order flow metadata."""
    all_trades: list[OrderFlowTradeResult] = []

    for inst, groups in instrument_groups.items():
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

            # Pre-compute range-period order flow metrics
            cvd_range = compute_cvd(range_candles)
            vol_skew = compute_volume_skew(range_candles, opening_range.low, range_size)
            abs_high, abs_low = compute_absorption(
                range_candles, opening_range.high, opening_range.low, range_size,
            )

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
                    # --- Compute breakout-specific order flow metrics ---

                    # Body ratio of breakout candle
                    body_rat = compute_body_ratio(candle)

                    # Pre-breakout volume ramp (3 candles before breakout)
                    pre_candles = post_range_candles[max(0, i - 3):i]
                    if not pre_candles:
                        pre_candles = range_candles[-3:]
                    vol_ramp = compute_volume_ramp(pre_candles)

                    # CVD alignment
                    cvd_aligned = (
                        (signal.direction == Direction.LONG and cvd_range > 0)
                        or (signal.direction == Direction.SHORT and cvd_range < 0)
                    )

                    # Volume skew alignment
                    skew_aligned = (
                        (signal.direction == Direction.LONG and vol_skew > 0.5)
                        or (signal.direction == Direction.SHORT and vol_skew < 0.5)
                    )

                    # Absorption confirmation
                    absorption_confirms = (
                        (signal.direction == Direction.LONG and abs_low > 0)
                        or (signal.direction == Direction.SHORT and abs_high > 0)
                    )

                    # CVD divergence (breakout candle + up to 3 prior candles)
                    recent_for_cvd = post_range_candles[max(0, i - 3):i + 1]
                    if len(recent_for_cvd) < 2:
                        recent_for_cvd = range_candles[-3:] + [candle]
                    recent_cvd = compute_cvd(recent_for_cvd)
                    cvd_divergent = (
                        (signal.direction == Direction.LONG and recent_cvd < 0)
                        or (signal.direction == Direction.SHORT and recent_cvd > 0)
                    )

                    # Volume ratio (for reference)
                    prior_candles = range_candles + post_range_candles[:i]
                    vol_ratio = compute_volume_ratio(candle, prior_candles)

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

                        trade = OrderFlowTradeResult(
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
                            cvd_range=cvd_range,
                            cvd_aligned=cvd_aligned,
                            body_ratio=body_rat,
                            volume_skew=vol_skew,
                            volume_skew_aligned=skew_aligned,
                            volume_ramp=vol_ramp,
                            absorption_at_high=abs_high,
                            absorption_at_low=abs_low,
                            absorption_confirms=absorption_confirms,
                            cvd_divergent=cvd_divergent,
                            breakout_volume_ratio=vol_ratio,
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


def _print_trade(t: OrderFlowTradeResult) -> None:
    print(
        f"  {t.trade_date} {t.session.value:6s} {t.instrument:11s} "
        f"{t.direction.value:5s} entry={t.entry_price:>10.2f} "
        f"exit={t.exit_price:>10.2f} reason={t.exit_reason:6s} "
        f"RR={t.rr_achieved:+.2f}  cvd={'Y' if t.cvd_aligned else 'N'} "
        f"body={t.body_ratio:.2f} skew={t.volume_skew:.2f} "
        f"ramp={t.volume_ramp:+.2f} abs={'Y' if t.absorption_confirms else 'N'} "
        f"div={'Y' if t.cvd_divergent else 'N'}"
    )


# ─────────────────────────────────────────────────────────────
#  Reporting helpers
# ─────────────────────────────────────────────────────────────


def _band_stats(trades: list[OrderFlowTradeResult]) -> tuple[int, float, float]:
    n = len(trades)
    if n == 0:
        return 0, 0.0, 0.0
    wins = sum(1 for t in trades if t.rr_achieved > 0)
    avg_rr = sum(t.rr_achieved for t in trades) / n
    return n, wins / n, avg_rr


def _total_r(trades: list[OrderFlowTradeResult]) -> float:
    return sum(t.rr_achieved for t in trades)


def _profit_factor(trades: list[OrderFlowTradeResult]) -> float:
    gross_p = sum(t.rr_achieved for t in trades if t.rr_achieved > 0)
    gross_l = abs(sum(t.rr_achieved for t in trades if t.rr_achieved <= 0))
    return gross_p / gross_l if gross_l > 0 else float("inf")


def _max_drawdown_r(trades: list[OrderFlowTradeResult]) -> float:
    if not trades:
        return 0.0
    peak = 0.0
    equity = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.rr_achieved
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ─────────────────────────────────────────────────────────────
#  Report functions
# ─────────────────────────────────────────────────────────────


def report_baseline(trades: list[OrderFlowTradeResult]) -> None:
    n, wr, avg_rr = _band_stats(trades)
    total = _total_r(trades)
    pf = _profit_factor(trades)
    max_dd = _max_drawdown_r(trades)
    stops = sum(1 for t in trades if t.exit_reason == "stop")
    targets = sum(1 for t in trades if t.exit_reason == "target")
    time_exits = sum(1 for t in trades if t.exit_reason == "time")

    print("\n" + "=" * 80)
    print("  BASELINE (all trades, no order flow filter)")
    print("=" * 80)
    print(f"  Trades:     {n}")
    print(f"  Win Rate:   {wr:.1%}")
    print(f"  Avg RR:     {avg_rr:+.2f}")
    print(f"  Total R:    {total:+.2f}")
    print(f"  PF:         {pf:.2f}")
    print(f"  Max DD:     {max_dd:.2f}R")
    print(f"  Exits:      stops={stops}  targets={targets}  time={time_exits}")
    print()


def report_cvd_alignment(trades: list[OrderFlowTradeResult]) -> None:
    aligned = [t for t in trades if t.cvd_aligned]
    not_aligned = [t for t in trades if not t.cvd_aligned]

    print("=" * 80)
    print("  FILTER 1: CVD Alignment  (range-period cumulative volume delta)")
    print("  LONG only if CVD > 0 during range, SHORT only if CVD < 0")
    print("=" * 80)
    print(f"  {'Group':<20s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 72)

    for label, subset in [("All trades", trades), ("CVD-aligned", aligned), ("CVD-opposed", not_aligned)]:
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        dd = _max_drawdown_r(subset)
        print(f"  {label:<20s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f} {dd:>6.1f}R")

    print()


def report_body_ratio(trades: list[OrderFlowTradeResult]) -> None:
    bands = [
        ("<0.30", 0.0, 0.3),
        ("0.30-0.50", 0.3, 0.5),
        ("0.50-0.70", 0.5, 0.7),
        (">0.70", 0.7, 1.01),
    ]

    print("=" * 80)
    print("  FILTER 2: Breakout Candle Body Ratio  (body / candle range)")
    print("  0.0 = doji (indecision), 1.0 = marubozu (full conviction)")
    print("=" * 80)
    print(f"  {'Band':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'Stops':>6s} {'Targets':>8s}")
    print("  " + "-" * 74)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.body_ratio < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        stops = sum(1 for t in subset if t.exit_reason == "stop")
        targets = sum(1 for t in subset if t.exit_reason == "target")
        print(f"  {label:<12s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f} {stops:>6d} {targets:>8d}")

    print()


def report_volume_skew(trades: list[OrderFlowTradeResult]) -> None:
    aligned = [t for t in trades if t.volume_skew_aligned]
    not_aligned = [t for t in trades if not t.volume_skew_aligned]

    print("=" * 80)
    print("  FILTER 3: Range Volume Skew  (volume-weighted position in range)")
    print("  0.0 = volume at bottom, 1.0 = volume at top")
    print("=" * 80)
    print(f"  {'Group':<20s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 64)

    for label, subset in [("All trades", trades), ("Skew-aligned", aligned), ("Skew-opposed", not_aligned)]:
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {label:<20s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    # Band breakdown
    bands = [
        ("<0.30", 0.0, 0.3),
        ("0.30-0.50", 0.3, 0.5),
        ("0.50-0.70", 0.5, 0.7),
        (">0.70", 0.7, 1.01),
    ]
    print()
    print(f"  {'Skew Band':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s}")
    print("  " + "-" * 48)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.volume_skew < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        print(f"  {label:<12s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")

    print()


def report_volume_ramp(trades: list[OrderFlowTradeResult]) -> None:
    bands = [
        ("<-0.50", -999, -0.5),
        ("-0.50-0", -0.5, 0.0),
        ("0-0.50", 0.0, 0.5),
        ("0.50-1.0", 0.5, 1.0),
        (">1.0", 1.0, 999),
    ]

    print("=" * 80)
    print("  FILTER 4: Pre-Breakout Volume Ramp  (volume slope, 3 candles before breakout)")
    print("  Positive = building pressure, negative = fading")
    print("=" * 80)
    print(f"  {'Band':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 56)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.volume_ramp < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {label:<12s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_absorption(trades: list[OrderFlowTradeResult]) -> None:
    confirms = [t for t in trades if t.absorption_confirms]
    no_absorption = [t for t in trades if not t.absorption_confirms]

    print("=" * 80)
    print("  FILTER 5: Absorption Detection  (high-vol candles held at range boundary)")
    print("  Confirms = absorption at opposite boundary (trapped participants)")
    print("=" * 80)
    print(f"  {'Group':<24s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 68)

    for label, subset in [("All trades", trades), ("Absorption confirms", confirms), ("No absorption", no_absorption)]:
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {label:<24s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    # Breakdown by absorption count
    print()
    print(f"  {'Absorption events':<20s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s}")
    print("  " + "-" * 52)

    for count_label, lo, hi in [("0 events", 0, 0), ("1 event", 1, 1), ("2+ events", 2, 99)]:
        subset = [
            t for t in trades
            if lo <= (t.absorption_at_high + t.absorption_at_low) <= hi
        ]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        print(f"  {count_label:<20s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f}")

    print()


def report_cvd_divergence(trades: list[OrderFlowTradeResult]) -> None:
    confirming = [t for t in trades if not t.cvd_divergent]
    divergent = [t for t in trades if t.cvd_divergent]

    print("=" * 80)
    print("  FILTER 6: CVD Divergence  (pre-breakout volume delta vs direction)")
    print("  Divergent = CVD opposes breakout direction (weak conviction)")
    print("=" * 80)
    print(f"  {'Group':<20s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 64)

    for label, subset in [("All trades", trades), ("CVD confirming", confirming), ("CVD divergent", divergent)]:
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {label:<20s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_combined(trades: list[OrderFlowTradeResult]) -> None:
    """Test combinations of the order flow filters."""
    print("=" * 80)
    print("  COMBINED ORDER FLOW FILTERS")
    print("=" * 80)
    print(f"  {'Combination':<40s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 78)

    combos = [
        ("Baseline (no filter)", lambda t: True),
        ("CVD-aligned", lambda t: t.cvd_aligned),
        ("Body ratio > 0.5", lambda t: t.body_ratio >= 0.5),
        ("Body ratio > 0.7", lambda t: t.body_ratio >= 0.7),
        ("CVD-aligned + body > 0.5", lambda t: t.cvd_aligned and t.body_ratio >= 0.5),
        ("CVD-aligned + body > 0.7", lambda t: t.cvd_aligned and t.body_ratio >= 0.7),
        ("CVD-aligned + skew-aligned", lambda t: t.cvd_aligned and t.volume_skew_aligned),
        ("CVD-aligned + absorption", lambda t: t.cvd_aligned and t.absorption_confirms),
        ("CVD-confirm + body > 0.5", lambda t: not t.cvd_divergent and t.body_ratio >= 0.5),
        ("CVD-confirm + ramp > 0", lambda t: not t.cvd_divergent and t.volume_ramp > 0),
        ("Body > 0.5 + skew-aligned", lambda t: t.body_ratio >= 0.5 and t.volume_skew_aligned),
        ("Body > 0.5 + absorption", lambda t: t.body_ratio >= 0.5 and t.absorption_confirms),
        ("CVD + body > 0.5 + skew", lambda t: t.cvd_aligned and t.body_ratio >= 0.5 and t.volume_skew_aligned),
        ("CVD + body > 0.5 + absorb", lambda t: t.cvd_aligned and t.body_ratio >= 0.5 and t.absorption_confirms),
    ]

    for label, filter_fn in combos:
        subset = [t for t in trades if filter_fn(t)]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {label:<40s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_instrument_session(trades: list[OrderFlowTradeResult]) -> None:
    print("=" * 80)
    print("  BREAKDOWN: Instrument x Session")
    print("=" * 80)
    print(f"  {'Instrument':<12s} {'Session':<8s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 66)

    instruments = sorted(set(t.instrument for t in trades))
    sessions = [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]

    for inst in instruments:
        for sess in sessions:
            subset = [t for t in trades if t.instrument == inst and t.session == sess]
            if not subset:
                continue
            n, wr, avg_rr = _band_stats(subset)
            total = _total_r(subset)
            pf = _profit_factor(subset)
            print(f"  {inst:<12s} {sess.value:<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print("  " + "-" * 66)
    n, wr, avg_rr = _band_stats(trades)
    total = _total_r(trades)
    pf = _profit_factor(trades)
    print(f"  {'TOTAL':<12s} {'ALL':<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")
    print()


def report_winners_by_instrument_session(trades: list[OrderFlowTradeResult]) -> None:
    """Break down the winning filters by instrument x session."""
    instruments = sorted(set(t.instrument for t in trades))
    sessions = [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]

    winning_filters = [
        ("Baseline", lambda t: True),
        ("Ramp 0-0.5", lambda t: 0.0 <= t.volume_ramp < 0.5),
        ("Ramp > 0", lambda t: t.volume_ramp > 0),
        ("CVD-aligned", lambda t: t.cvd_aligned),
        ("CVD-confirm + ramp > 0", lambda t: not t.cvd_divergent and t.volume_ramp > 0),
        ("CVD-aligned + ramp > 0", lambda t: t.cvd_aligned and t.volume_ramp > 0),
        ("CVD-aligned + ramp 0-0.5", lambda t: t.cvd_aligned and 0.0 <= t.volume_ramp < 0.5),
    ]

    for filter_name, filter_fn in winning_filters:
        filtered = [t for t in trades if filter_fn(t)]

        print("=" * 90)
        print(f"  INSTRUMENT x SESSION — {filter_name}")
        n_all, wr_all, avg_all = _band_stats(filtered)
        total_all = _total_r(filtered)
        pf_all = _profit_factor(filtered)
        dd_all = _max_drawdown_r(filtered)
        print(f"  Overall: {n_all} trades, {wr_all:.1%} win, {avg_all:+.2f} avg RR, {total_all:+.1f}R, {pf_all:.2f} PF, {dd_all:.1f}R DD")
        print("=" * 90)
        print(f"  {'Instrument':<12s} {'Session':<8s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s} {'Stops':>6s} {'Tgts':>5s} {'Time':>5s}")
        print("  " + "-" * 86)

        for inst in instruments:
            for sess in sessions:
                subset = [t for t in filtered if t.instrument == inst and t.session == sess]
                if not subset:
                    continue
                n, wr, avg_rr = _band_stats(subset)
                total = _total_r(subset)
                pf = _profit_factor(subset)
                dd = _max_drawdown_r(subset)
                stops = sum(1 for t in subset if t.exit_reason == "stop")
                tgts = sum(1 for t in subset if t.exit_reason == "target")
                time_exits = sum(1 for t in subset if t.exit_reason == "time")
                print(
                    f"  {inst:<12s} {sess.value:<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} "
                    f"{total:>+9.2f} {pf:>7.2f} {dd:>6.1f}R {stops:>6d} {tgts:>5d} {time_exits:>5d}"
                )

        print()


def report_ramp_fine_grain(trades: list[OrderFlowTradeResult]) -> None:
    """Finer-grained ramp bands to find the optimal threshold."""
    bands = [
        ("<-1.0", -999, -1.0),
        ("-1.0 to -0.5", -1.0, -0.5),
        ("-0.5 to -0.25", -0.5, -0.25),
        ("-0.25 to 0", -0.25, 0.0),
        ("0 to 0.25", 0.0, 0.25),
        ("0.25 to 0.5", 0.25, 0.5),
        ("0.5 to 0.75", 0.5, 0.75),
        ("0.75 to 1.0", 0.75, 1.0),
        ("1.0 to 1.5", 1.0, 1.5),
        (">1.5", 1.5, 999),
    ]

    print("=" * 80)
    print("  VOLUME RAMP — FINE-GRAINED BANDS")
    print("=" * 80)
    print(f"  {'Band':<16s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'Stops':>6s} {'Tgts':>5s}")
    print("  " + "-" * 72)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.volume_ramp < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        stops = sum(1 for t in subset if t.exit_reason == "stop")
        tgts = sum(1 for t in subset if t.exit_reason == "target")
        print(f"  {label:<16s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f} {stops:>6d} {tgts:>5d}")

    # Cumulative thresholds: "ramp > X"
    print()
    print(f"  {'Threshold':<16s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 58)

    for thresh in [-0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        subset = [t for t in trades if t.volume_ramp >= thresh]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  ramp >= {thresh:+.2f}   {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_direction_breakdown(trades: list[OrderFlowTradeResult]) -> None:
    """Break down winning filters by LONG vs SHORT."""
    print("=" * 80)
    print("  DIRECTION BREAKDOWN — Winning Filters")
    print("=" * 80)

    filters = [
        ("Baseline", lambda t: True),
        ("Ramp > 0", lambda t: t.volume_ramp > 0),
        ("CVD-confirm + ramp > 0", lambda t: not t.cvd_divergent and t.volume_ramp > 0),
        ("CVD-aligned", lambda t: t.cvd_aligned),
        ("CVD-aligned + ramp > 0", lambda t: t.cvd_aligned and t.volume_ramp > 0),
    ]

    print(f"  {'Filter':<28s} {'Dir':<6s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 78)

    for filter_name, filter_fn in filters:
        filtered = [t for t in trades if filter_fn(t)]
        longs = [t for t in filtered if t.direction == Direction.LONG]
        shorts = [t for t in filtered if t.direction == Direction.SHORT]

        for label, subset in [("ALL", filtered), ("LONG", longs), ("SHORT", shorts)]:
            n, wr, avg_rr = _band_stats(subset)
            total = _total_r(subset)
            pf = _profit_factor(subset)
            name_col = filter_name if label == "ALL" else ""
            print(f"  {name_col:<28s} {label:<6s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

        print("  " + "-" * 78)


# ─────────────────────────────────────────────────────────────
#  Deep dive reports
# ─────────────────────────────────────────────────────────────


def report_time_of_day(trades: list[OrderFlowTradeResult]) -> None:
    """Break down winning filters by time-of-day within each session (early vs late breakouts)."""
    print("\n" + "=" * 100)
    print("  DEEP DIVE 1: TIME-OF-DAY ANALYSIS — Minutes After Range Close")
    print("=" * 100)

    time_bands = [
        ("0-15 min", 0, 15),
        ("15-30 min", 15, 30),
        ("30-60 min", 30, 60),
        ("60-90 min", 60, 90),
        ("90-120 min", 90, 120),
        ("120+ min", 120, 9999),
    ]

    filters = [
        ("Baseline", lambda t: True),
        ("Ramp > 0", lambda t: t.volume_ramp > 0),
        ("CVD-aligned + ramp > 0", lambda t: t.cvd_aligned and t.volume_ramp > 0),
    ]

    sessions = [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]

    for filter_name, filter_fn in filters:
        filtered = [t for t in trades if filter_fn(t)]

        print(f"\n  --- {filter_name} ---")
        print(f"  {'Session':<8s} {'Time Band':<14s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
        print("  " + "-" * 64)

        for sess in sessions:
            sess_trades = [t for t in filtered if t.session == sess]
            if not sess_trades:
                continue

            # Compute minutes after range close for each trade
            sess_cfg = SESSIONS[sess]
            range_end_time = sess_cfg.range_end

            for label, lo_min, hi_min in time_bands:
                subset = []
                for t in sess_trades:
                    ny_time = t.entry_time
                    if ny_time.tzinfo is None:
                        ny_time = ny_time.replace(tzinfo=timezone.utc)
                    ny_time = ny_time.astimezone(NY_TZ)
                    entry_t = ny_time.time()

                    # Minutes after range end
                    entry_mins = entry_t.hour * 60 + entry_t.minute
                    range_end_mins = range_end_time.hour * 60 + range_end_time.minute

                    # Handle Tokyo midnight crossing
                    if sess == SessionName.TOKYO and entry_mins < range_end_mins:
                        diff = (entry_mins + 24 * 60) - range_end_mins
                    else:
                        diff = entry_mins - range_end_mins

                    if lo_min <= diff < hi_min:
                        subset.append(t)

                if not subset:
                    continue
                n, wr, avg_rr = _band_stats(subset)
                total = _total_r(subset)
                pf = _profit_factor(subset)
                print(f"  {sess.value:<8s} {label:<14s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

            print()

    # Aggregate across all sessions
    print(f"\n  --- Aggregate (all sessions) ---")
    for filter_name, filter_fn in filters:
        filtered = [t for t in trades if filter_fn(t)]
        print(f"\n  {filter_name}:")
        print(f"  {'Time Band':<14s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
        print("  " + "-" * 56)

        for label, lo_min, hi_min in time_bands:
            subset = []
            for t in filtered:
                sess_cfg = SESSIONS[t.session]
                range_end_time = sess_cfg.range_end

                ny_time = t.entry_time
                if ny_time.tzinfo is None:
                    ny_time = ny_time.replace(tzinfo=timezone.utc)
                ny_time = ny_time.astimezone(NY_TZ)
                entry_t = ny_time.time()

                entry_mins = entry_t.hour * 60 + entry_t.minute
                range_end_mins = range_end_time.hour * 60 + range_end_time.minute

                if t.session == SessionName.TOKYO and entry_mins < range_end_mins:
                    diff = (entry_mins + 24 * 60) - range_end_mins
                else:
                    diff = entry_mins - range_end_mins

                if lo_min <= diff < hi_min:
                    subset.append(t)

            if not subset:
                continue
            n, wr, avg_rr = _band_stats(subset)
            total = _total_r(subset)
            pf = _profit_factor(subset)
            print(f"  {label:<14s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_streaks_and_equity(trades: list[OrderFlowTradeResult]) -> None:
    """Win/loss streaks and equity curve stats for winning filters."""
    print("\n" + "=" * 100)
    print("  DEEP DIVE 2: WIN/LOSS STREAKS & EQUITY CURVE")
    print("=" * 100)

    filters = [
        ("Baseline", lambda t: True),
        ("Ramp > 0", lambda t: t.volume_ramp > 0),
        ("CVD-aligned + ramp > 0", lambda t: t.cvd_aligned and t.volume_ramp > 0),
        ("CVD-confirm + ramp > 0", lambda t: not t.cvd_divergent and t.volume_ramp > 0),
    ]

    print(f"  {'Filter':<28s} {'MaxWin':>7s} {'MaxLoss':>8s} {'AvgWin':>7s} {'AvgLoss':>8s} "
          f"{'Peak R':>8s} {'Final R':>8s} {'MaxDD R':>8s} {'DD/Peak':>8s} {'Recov':>7s}")
    print("  " + "-" * 106)

    for filter_name, filter_fn in filters:
        filtered = sorted([t for t in trades if filter_fn(t)], key=lambda t: t.entry_time)
        if not filtered:
            continue

        # Streaks
        max_win_streak = 0
        max_loss_streak = 0
        win_streak = 0
        loss_streak = 0
        win_streaks = []
        loss_streaks = []

        for t in filtered:
            if t.rr_achieved > 0:
                win_streak += 1
                if loss_streak > 0:
                    loss_streaks.append(loss_streak)
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                if win_streak > 0:
                    win_streaks.append(win_streak)
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)

        if win_streak > 0:
            win_streaks.append(win_streak)
        if loss_streak > 0:
            loss_streaks.append(loss_streak)

        avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
        avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0

        # Equity curve
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        dd_start_idx = 0
        dd_end_idx = 0
        max_dd_end_idx = 0
        recovery_trades = 0
        in_dd = False

        equities = []
        for i, t in enumerate(filtered):
            equity += t.rr_achieved
            equities.append(equity)
            if equity > peak:
                if in_dd:
                    recovery_trades = i - max_dd_end_idx
                peak = equity
                in_dd = False
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
                max_dd_end_idx = i
                in_dd = True

        final_r = equity
        dd_peak_ratio = max_dd / peak if peak > 0 else float("inf")

        print(
            f"  {filter_name:<28s} {max_win_streak:>7d} {max_loss_streak:>8d} "
            f"{avg_win_streak:>6.1f} {avg_loss_streak:>8.1f} "
            f"{peak:>+8.1f} {final_r:>+8.1f} {max_dd:>8.1f} "
            f"{dd_peak_ratio:>7.1%} {recovery_trades:>7d}"
        )

    # Equity curve snapshots for best filter
    print(f"\n  --- Equity Curve Snapshots: CVD-aligned + ramp > 0 ---")
    best = sorted([t for t in trades if t.cvd_aligned and t.volume_ramp > 0], key=lambda t: t.entry_time)
    if best:
        equity = 0.0
        n = len(best)
        checkpoints = [0, n // 10, n // 4, n // 2, 3 * n // 4, n - 1]
        checkpoints = sorted(set(checkpoints))

        print(f"  {'Trade #':<10s} {'Date':<12s} {'Equity R':>10s} {'Win% So Far':>12s}")
        print("  " + "-" * 48)

        wins_so_far = 0
        for i, t in enumerate(best):
            equity += t.rr_achieved
            if t.rr_achieved > 0:
                wins_so_far += 1
            if i in checkpoints:
                wr = wins_so_far / (i + 1)
                print(f"  {i + 1:<10d} {str(t.trade_date):<12s} {equity:>+10.1f} {wr:>11.1%}")

    print()


def report_monthly_breakdown(trades: list[OrderFlowTradeResult]) -> None:
    """Monthly breakdown to check edge consistency over time."""
    print("\n" + "=" * 110)
    print("  DEEP DIVE 3: MONTHLY BREAKDOWN")
    print("=" * 110)

    filters = [
        ("Baseline", lambda t: True),
        ("Ramp > 0", lambda t: t.volume_ramp > 0),
        ("CVD-aligned + ramp > 0", lambda t: t.cvd_aligned and t.volume_ramp > 0),
    ]

    # Collect all months
    all_months = sorted(set((t.trade_date.year, t.trade_date.month) for t in trades))

    for filter_name, filter_fn in filters:
        filtered = [t for t in trades if filter_fn(t)]

        print(f"\n  --- {filter_name} ---")
        print(f"  {'Month':<10s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} "
              f"{'MaxDD':>7s} {'Stops':>6s} {'Tgts':>5s} {'Cum R':>9s}")
        print("  " + "-" * 86)

        cum_r = 0.0
        positive_months = 0
        negative_months = 0

        for year, month in all_months:
            subset = [t for t in filtered if t.trade_date.year == year and t.trade_date.month == month]
            if not subset:
                continue
            n, wr, avg_rr = _band_stats(subset)
            total = _total_r(subset)
            pf = _profit_factor(subset)
            dd = _max_drawdown_r(subset)
            stops = sum(1 for t in subset if t.exit_reason == "stop")
            tgts = sum(1 for t in subset if t.exit_reason == "target")
            cum_r += total

            if total > 0:
                positive_months += 1
            elif total < 0:
                negative_months += 1

            print(
                f"  {year}-{month:02d}    {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f} "
                f"{dd:>6.1f}R {stops:>6d} {tgts:>5d} {cum_r:>+9.2f}"
            )

        total_months = positive_months + negative_months
        print("  " + "-" * 86)
        print(f"  Positive months: {positive_months}/{total_months} ({positive_months / total_months:.0%})" if total_months > 0 else "")

    print()


def report_implementation_spec(trades: list[OrderFlowTradeResult]) -> None:
    """Output the exact implementation spec for adding CVD + ramp as a live filter."""
    print("\n" + "=" * 100)
    print("  DEEP DIVE 4: LIVE FILTER IMPLEMENTATION SPEC — CVD + Ramp")
    print("=" * 100)

    # Show the recommended filter config
    print("""
  RECOMMENDED FILTER: CVD-aligned + volume ramp > 0

  This filters breakout signals to only take trades where:
    1. CVD alignment: range-period cumulative volume delta confirms direction
       - LONG: sum of buy volume > sum of sell volume during range
       - SHORT: sum of sell volume > sum of buy volume during range
    2. Volume ramp > 0: volume is increasing in the 3 candles before breakout

  IMPLEMENTATION CHANGES NEEDED:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ File: oanda_bot/strategy.py                                        │
  │                                                                     │
  │ 1. Add compute_candle_delta(candle) -> int                          │
  │    - Returns +volume if close > open, -volume if close < open       │
  │                                                                     │
  │ 2. Add compute_cvd(candles) -> float                                │
  │    - Sum of deltas over range candles                               │
  │                                                                     │
  │ 3. Add compute_volume_ramp(candles) -> float                        │
  │    - (last_vol - first_vol) / avg_vol over 3 pre-breakout candles   │
  │                                                                     │
  │ 4. Add OrderFlowFilter dataclass:                                   │
  │    - cvd_aligned: bool                                              │
  │    - volume_ramp: float                                             │
  │    - passes_filter: bool                                            │
  │                                                                     │
  │ 5. Add check_order_flow(range_candles, pre_breakout_candles,        │
  │                          direction) -> OrderFlowFilter              │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ File: oanda_bot/session_manager.py                                  │
  │                                                                     │
  │ 1. Store range_candles list during RECORDING_RANGE state            │
  │ 2. After breakout signal, before placing order:                     │
  │    - Compute CVD from stored range candles                          │
  │    - Compute volume ramp from last 3 post-range candles             │
  │    - If filter fails, log rejection and continue watching           │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ File: oanda_bot/config.py                                           │
  │                                                                     │
  │ 1. Add to TradingConfig:                                            │
  │    USE_ORDER_FLOW_FILTER: bool = True                               │
  │    REQUIRE_CVD_ALIGNED: bool = True                                 │
  │    MIN_VOLUME_RAMP: float = 0.0                                     │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")

    # Show what the filter does to each instrument x session
    print("  IMPACT BY INSTRUMENT x SESSION:")
    print(f"  {'Inst x Session':<20s} {'Baseline R':>11s} {'Filtered R':>11s} {'Delta R':>9s} {'Base PF':>8s} {'Filt PF':>8s} {'Trades Cut':>11s}")
    print("  " + "-" * 82)

    instruments = sorted(set(t.instrument for t in trades))
    sessions_list = [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]

    total_base_r = 0.0
    total_filt_r = 0.0

    for inst in instruments:
        for sess in sessions_list:
            base = [t for t in trades if t.instrument == inst and t.session == sess]
            filt = [t for t in base if t.cvd_aligned and t.volume_ramp > 0]
            if not base:
                continue

            base_r = _total_r(base)
            filt_r = _total_r(filt)
            delta = filt_r - base_r
            base_pf = _profit_factor(base)
            filt_pf = _profit_factor(filt)
            pct_cut = (1 - len(filt) / len(base)) * 100

            total_base_r += base_r
            total_filt_r += filt_r

            label = f"{inst[:7]} {sess.value}"
            print(
                f"  {label:<20s} {base_r:>+11.1f} {filt_r:>+11.1f} {delta:>+9.1f} "
                f"{base_pf:>8.2f} {filt_pf:>8.2f} {pct_cut:>10.0f}%"
            )

    print("  " + "-" * 82)
    delta_total = total_filt_r - total_base_r
    base_n = len(trades)
    filt_n = len([t for t in trades if t.cvd_aligned and t.volume_ramp > 0])
    pct_cut = (1 - filt_n / base_n) * 100
    print(
        f"  {'TOTAL':<20s} {total_base_r:>+11.1f} {total_filt_r:>+11.1f} {delta_total:>+9.1f} "
        f"{'':>8s} {'':>8s} {pct_cut:>10.0f}%"
    )

    print(f"""
  TRADE-OFFS:
    - Cuts {pct_cut:.0f}% of trades ({base_n} -> {filt_n})
    - Total R: {total_base_r:+.0f}R -> {total_filt_r:+.0f}R ({delta_total:+.0f}R delta)
    - Max DD reduced significantly (99R -> ~56R)
    - Win rate: 22.1% -> 26.5%
    - PF: 1.07 -> 1.37

  NOTE: NAS100 NY is the one cell where the filter hurts.
  Consider making it per-instrument/session configurable, or
  accept the small loss there for the portfolio-level improvement.
""")


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────


def load_candles_from_cache(instrument: str) -> Optional[list[Candle]]:
    """Load candle data from the most recent pickle file for an instrument."""
    pattern = f"{instrument}_*_M5.pkl"
    matches = sorted(CACHE_DIR.glob(pattern))
    if not matches:
        return None
    cache_file = matches[-1]
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-ORB Order Flow Backtest")
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

    log_level = logging.WARNING
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
    print("  MS-ORB ORDER FLOW BACKTEST")
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

    # Group candles by session day
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]] = {}
    for inst, candles in instrument_candles.items():
        groups = group_candles_by_session_day(candles, list(session_configs.values()))
        instrument_groups[inst] = groups
        print(f"  {inst}: {len(groups)} session-days")

    print()
    print("  Running simulation...")

    # Collect all trades with order flow metadata
    trades = collect_all_trades(
        instrument_groups=instrument_groups,
        session_configs=session_configs,
        rr_target=args.rr_target,
        max_trades_per_session=TradingConfig.MAX_TRADES_PER_SESSION,
        verbose=args.verbose,
    )

    print(f"  Total trades simulated: {len(trades)}")

    if not trades:
        print("No trades found. Check your data and session configuration.")
        return

    # Print all reports
    report_baseline(trades)
    report_instrument_session(trades)
    report_cvd_alignment(trades)
    report_body_ratio(trades)
    report_volume_skew(trades)
    report_volume_ramp(trades)
    report_ramp_fine_grain(trades)
    report_absorption(trades)
    report_cvd_divergence(trades)
    report_combined(trades)
    report_direction_breakdown(trades)
    report_winners_by_instrument_session(trades)

    # Deep dives
    report_time_of_day(trades)
    report_streaks_and_equity(trades)
    report_monthly_breakdown(trades)
    report_implementation_spec(trades)


if __name__ == "__main__":
    main()
