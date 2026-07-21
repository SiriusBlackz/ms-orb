#!/usr/bin/env python3
"""Strategy 1: Fade the Fakeout — Range Reversion Backtest.

Instead of trading breakouts, this strategy detects FAILED breakouts
(price closes beyond range then reverses back inside within N candles)
and enters in the opposite direction, targeting the other side of the range.

Hypothesis: 77% of MS-ORB breakouts hit stop loss. The opening range
boundaries act as liquidity traps — breakout traders get stopped out and
price reverts. This strategy exploits that failure.

Usage:
    python backtest_fakeout.py [--instruments XAU_USD,NAS100_USD]
                               [--sessions TOKYO,LONDON,NY]
                               [--rr-target 2.0]
                               [--fakeout-window 3]
                               [--verbose]
"""

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
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
class FakeoutTradeResult:
    """Single simulated fakeout-fade trade."""
    instrument: str
    session: SessionName
    trade_date: date
    direction: Direction          # Direction of the FADE (opposite of breakout)
    breakout_direction: str       # "UP" or "DOWN" — which side faked out
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float
    exit_reason: str              # "stop", "target", "time"
    rr_achieved: float
    entry_time: datetime
    exit_time: datetime
    # Fakeout metadata
    range_high: float
    range_low: float
    range_size: float
    breakout_candle_close: float  # Where the breakout candle closed
    fakeout_candle_close: float   # Where price closed back inside range
    candles_to_fakeout: int       # How many candles until reversion
    breakout_penetration: float   # How far beyond range the breakout went (%)
    cvd_range: float              # CVD during range period
    volume_ramp: float            # Volume ramp at breakout


# ─────────────────────────────────────────────────────────────
#  Time helpers (duplicated per established pattern)
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
#  Order flow helpers (reused from backtest_orderflow.py)
# ─────────────────────────────────────────────────────────────


def compute_candle_delta(candle: Candle) -> int:
    if candle.close > candle.open:
        return candle.volume
    elif candle.close < candle.open:
        return -candle.volume
    return 0


def compute_cvd(candles: list[Candle]) -> float:
    return sum(compute_candle_delta(c) for c in candles)


def compute_volume_ramp(candles: list[Candle]) -> float:
    if len(candles) < 2:
        return 0.0
    volumes = [c.volume for c in candles]
    avg_vol = sum(volumes) / len(volumes)
    if avg_vol <= 0:
        return 0.0
    return (volumes[-1] - volumes[0]) / avg_vol


# ─────────────────────────────────────────────────────────────
#  Fakeout detection and simulation
# ─────────────────────────────────────────────────────────────


def collect_fakeout_trades(
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]],
    session_configs: dict[SessionName, SessionConfig],
    rr_target: float,
    fakeout_window: int = 3,
    max_trades_per_session: int = 2,
    verbose: bool = False,
) -> list[FakeoutTradeResult]:
    """Detect fakeouts and simulate fade trades.

    Fakeout detection logic:
    1. Walk post-range candles looking for a breakout (close > range_high or < range_low)
    2. After breakout detected, watch next `fakeout_window` candles
    3. If any of those candles CLOSE back inside the range → fakeout confirmed
    4. Enter FADE trade:
       - If breakout was UP (failed long), enter SHORT at fakeout candle close
       - If breakout was DOWN (failed short), enter LONG at fakeout candle close
    5. SL: beyond the fakeout extreme (highest high for failed long, lowest low for failed short)
    6. TP: opposite range boundary, or entry ± risk * rr_target (whichever is closer)
    """
    all_trades: list[FakeoutTradeResult] = []

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

            range_high = opening_range.high
            range_low = opening_range.low
            range_size = range_high - range_low
            if range_size <= 0:
                continue

            # Pre-compute range CVD
            cvd_range = compute_cvd(range_candles)

            # Walk post-range candles looking for fakeouts
            trades_taken = 0
            i = 0

            while i < len(post_range_candles) and trades_taken < max_trades_per_session:
                candle = post_range_candles[i]
                ny_time = candle_ny_time(candle)

                # Only look for fakeouts during entry window
                if not is_in_entry_window(ny_time, session_cfg):
                    i += 1
                    continue

                breakout_up = candle.close > range_high
                breakout_down = candle.close < range_low

                if not breakout_up and not breakout_down:
                    i += 1
                    continue

                # Breakout detected — watch next N candles for reversion
                breakout_idx = i
                breakout_candle = candle
                breakout_dir = "UP" if breakout_up else "DOWN"

                # Track the extreme during the fakeout window
                if breakout_up:
                    extreme = candle.high
                else:
                    extreme = candle.low

                fakeout_found = False
                fakeout_candle = None
                candles_to_fakeout = 0

                for j in range(1, fakeout_window + 1):
                    check_idx = breakout_idx + j
                    if check_idx >= len(post_range_candles):
                        break

                    check_candle = post_range_candles[check_idx]
                    check_ny = candle_ny_time(check_candle)

                    # Don't look past session close
                    if is_past_session_close(check_ny, session_cfg):
                        break

                    # Update extreme
                    if breakout_up:
                        extreme = max(extreme, check_candle.high)
                    else:
                        extreme = min(extreme, check_candle.low)

                    # Check if this candle closed back inside the range
                    if breakout_up and check_candle.close <= range_high:
                        fakeout_found = True
                        fakeout_candle = check_candle
                        candles_to_fakeout = j
                        break
                    elif breakout_down and check_candle.close >= range_low:
                        fakeout_found = True
                        fakeout_candle = check_candle
                        candles_to_fakeout = j
                        break

                if not fakeout_found:
                    # Not a fakeout — skip past the window
                    i = breakout_idx + fakeout_window + 1
                    continue

                # Fakeout confirmed — set up the fade trade
                entry_price = fakeout_candle.close

                if breakout_dir == "UP":
                    # Failed long breakout → enter SHORT
                    fade_direction = Direction.SHORT
                    stop_loss = extreme  # Above the fakeout high
                    risk = stop_loss - entry_price

                    if risk <= 0:
                        i = breakout_idx + candles_to_fakeout + 1
                        continue

                    # TP: opposite boundary (range_low) or RR-based, whichever is closer
                    rr_tp = entry_price - (risk * rr_target)
                    range_tp = range_low
                    take_profit = max(rr_tp, range_tp)  # For shorts, higher = closer

                else:
                    # Failed short breakout → enter LONG
                    fade_direction = Direction.LONG
                    stop_loss = extreme  # Below the fakeout low
                    risk = entry_price - stop_loss

                    if risk <= 0:
                        i = breakout_idx + candles_to_fakeout + 1
                        continue

                    # TP: opposite boundary (range_high) or RR-based
                    rr_tp = entry_price + (risk * rr_target)
                    range_tp = range_high
                    take_profit = min(rr_tp, range_tp)  # For longs, lower = closer

                # Compute volume ramp at breakout
                pre_breakout = post_range_candles[max(0, breakout_idx - 3):breakout_idx]
                if not pre_breakout:
                    pre_breakout = range_candles[-3:]
                vol_ramp = compute_volume_ramp(pre_breakout)

                # Breakout penetration (how far beyond range)
                if breakout_dir == "UP":
                    penetration = (breakout_candle.close - range_high) / range_size * 100
                else:
                    penetration = (range_low - breakout_candle.close) / range_size * 100

                # Simulate the fade trade
                entry_idx = breakout_idx + candles_to_fakeout
                remaining = post_range_candles[entry_idx + 1:]
                result = simulate_trade(
                    entry_price, stop_loss, take_profit,
                    fade_direction, risk, remaining, session_cfg,
                )

                if result:
                    exit_price, exit_reason, exit_time = result
                    rr = _calc_rr(fade_direction, entry_price, exit_price, risk)

                    trade = FakeoutTradeResult(
                        instrument=inst,
                        session=session_name,
                        trade_date=session_date,
                        direction=fade_direction,
                        breakout_direction=breakout_dir,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        rr_achieved=rr,
                        entry_time=fakeout_candle.time,
                        exit_time=exit_time,
                        range_high=range_high,
                        range_low=range_low,
                        range_size=range_size,
                        breakout_candle_close=breakout_candle.close,
                        fakeout_candle_close=fakeout_candle.close,
                        candles_to_fakeout=candles_to_fakeout,
                        breakout_penetration=penetration,
                        cvd_range=cvd_range,
                        volume_ramp=vol_ramp,
                    )
                    all_trades.append(trade)
                    trades_taken += 1

                    if verbose:
                        _print_trade(trade)

                    # Advance past exit
                    if exit_reason == "stop":
                        while i < len(post_range_candles) and post_range_candles[i].time <= exit_time:
                            i += 1
                        continue
                    else:
                        break  # Target or time exit → done for session

                # Advance past fakeout
                i = breakout_idx + candles_to_fakeout + 1

    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


def _print_trade(t: FakeoutTradeResult) -> None:
    print(
        f"  {t.trade_date} {t.session.value:6s} {t.instrument:11s} "
        f"fake={t.breakout_direction:4s} fade={t.direction.value:5s} "
        f"entry={t.entry_price:>10.2f} exit={t.exit_price:>10.2f} "
        f"reason={t.exit_reason:6s} RR={t.rr_achieved:+.2f} "
        f"pen={t.breakout_penetration:.0f}% ttf={t.candles_to_fakeout}"
    )


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────


def _band_stats(trades: list[FakeoutTradeResult]) -> tuple[int, float, float]:
    n = len(trades)
    if n == 0:
        return 0, 0.0, 0.0
    wins = sum(1 for t in trades if t.rr_achieved > 0)
    avg_rr = sum(t.rr_achieved for t in trades) / n
    return n, wins / n, avg_rr


def _total_r(trades: list[FakeoutTradeResult]) -> float:
    return sum(t.rr_achieved for t in trades)


def _profit_factor(trades: list[FakeoutTradeResult]) -> float:
    gross_p = sum(t.rr_achieved for t in trades if t.rr_achieved > 0)
    gross_l = abs(sum(t.rr_achieved for t in trades if t.rr_achieved <= 0))
    return gross_p / gross_l if gross_l > 0 else float("inf")


def _max_drawdown_r(trades: list[FakeoutTradeResult]) -> float:
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


def report_baseline(trades: list[FakeoutTradeResult]) -> None:
    n, wr, avg_rr = _band_stats(trades)
    total = _total_r(trades)
    pf = _profit_factor(trades)
    max_dd = _max_drawdown_r(trades)
    stops = sum(1 for t in trades if t.exit_reason == "stop")
    targets = sum(1 for t in trades if t.exit_reason == "target")
    time_exits = sum(1 for t in trades if t.exit_reason == "time")

    print("\n" + "=" * 80)
    print("  FADE THE FAKEOUT — BASELINE RESULTS")
    print("=" * 80)
    print(f"  Trades:     {n}")
    print(f"  Win Rate:   {wr:.1%}")
    print(f"  Avg RR:     {avg_rr:+.2f}")
    print(f"  Total R:    {total:+.2f}")
    print(f"  PF:         {pf:.2f}")
    print(f"  Max DD:     {max_dd:.2f}R")
    print(f"  Exits:      stops={stops}  targets={targets}  time={time_exits}")

    # Direction breakdown
    longs = [t for t in trades if t.direction == Direction.LONG]
    shorts = [t for t in trades if t.direction == Direction.SHORT]
    print(f"\n  {'Direction':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 54)
    for label, subset in [("LONG (fade↓)", longs), ("SHORT(fade↑)", shorts)]:
        sn, swr, savg = _band_stats(subset)
        st = _total_r(subset)
        spf = _profit_factor(subset)
        print(f"  {label:<12s} {sn:>7d} {swr:>6.1%} {savg:>+8.2f} {st:>+9.2f} {spf:>7.2f}")
    print()


def report_instrument_session(trades: list[FakeoutTradeResult]) -> None:
    print("=" * 80)
    print("  BREAKDOWN: Instrument x Session")
    print("=" * 80)
    print(f"  {'Instrument':<12s} {'Session':<8s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 72)

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
            dd = _max_drawdown_r(subset)
            print(f"  {inst:<12s} {sess.value:<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f} {dd:>6.1f}R")

    print("  " + "-" * 72)
    n, wr, avg_rr = _band_stats(trades)
    total = _total_r(trades)
    pf = _profit_factor(trades)
    print(f"  {'TOTAL':<12s} {'ALL':<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")
    print()


def report_fakeout_window(trades: list[FakeoutTradeResult]) -> None:
    """Break down by how quickly the fakeout resolved."""
    print("=" * 80)
    print("  FAKEOUT SPEED: Candles to Reversion")
    print("=" * 80)
    print(f"  {'Candles':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 56)

    for n_candles in range(1, 6):
        subset = [t for t in trades if t.candles_to_fakeout == n_candles]
        if not subset:
            continue
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {n_candles} candle{'s' if n_candles > 1 else ' ':<6s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_penetration(trades: list[FakeoutTradeResult]) -> None:
    """Break down by how far the breakout penetrated beyond the range."""
    bands = [
        ("0-10%", 0, 10),
        ("10-25%", 10, 25),
        ("25-50%", 25, 50),
        ("50-100%", 50, 100),
        ("100%+", 100, 9999),
    ]

    print("=" * 80)
    print("  BREAKOUT PENETRATION: How far beyond range before reverting")
    print("=" * 80)
    print(f"  {'Band':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 56)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.breakout_penetration < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {label:<12s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_cvd_filter(trades: list[FakeoutTradeResult]) -> None:
    """Test if CVD helps filter fakeout trades."""
    print("=" * 80)
    print("  CVD FILTER: Does range CVD predict fakeout quality?")
    print("  For fade trades, CVD OPPOSING the breakout = market was already fading")
    print("=" * 80)

    # CVD opposing breakout = confirms the fade
    # If breakout was UP but CVD was negative → market was selling during range → fade-friendly
    fade_confirmed = [
        t for t in trades
        if (t.breakout_direction == "UP" and t.cvd_range < 0)
        or (t.breakout_direction == "DOWN" and t.cvd_range > 0)
    ]
    fade_opposed = [t for t in trades if t not in fade_confirmed]

    print(f"  {'Group':<24s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 64)

    for label, subset in [("All trades", trades), ("CVD confirms fade", fade_confirmed), ("CVD opposes fade", fade_opposed)]:
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {label:<24s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_volume_ramp_filter(trades: list[FakeoutTradeResult]) -> None:
    """Test if low volume ramp at breakout predicts fakeout quality."""
    print("=" * 80)
    print("  VOLUME RAMP AT BREAKOUT: Low ramp = weak breakout = better fakeout?")
    print("=" * 80)

    bands = [
        ("<-0.5", -999, -0.5),
        ("-0.5 to 0", -0.5, 0.0),
        ("0 to 0.5", 0.0, 0.5),
        (">0.5", 0.5, 999),
    ]

    print(f"  {'Ramp Band':<14s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 58)

    for label, lo, hi in bands:
        subset = [t for t in trades if lo <= t.volume_ramp < hi]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        print(f"  {label:<14s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    print()


def report_combined_filters(trades: list[FakeoutTradeResult]) -> None:
    """Test filter combinations."""
    print("=" * 80)
    print("  COMBINED FILTERS")
    print("=" * 80)

    combos = [
        ("Baseline", lambda t: True),
        ("1-candle fakeout only", lambda t: t.candles_to_fakeout == 1),
        ("Penetration < 25%", lambda t: t.breakout_penetration < 25),
        ("Penetration < 50%", lambda t: t.breakout_penetration < 50),
        ("CVD confirms fade", lambda t:
            (t.breakout_direction == "UP" and t.cvd_range < 0) or
            (t.breakout_direction == "DOWN" and t.cvd_range > 0)),
        ("Low ramp (< 0)", lambda t: t.volume_ramp < 0),
        ("1-candle + CVD fade", lambda t:
            t.candles_to_fakeout == 1 and (
                (t.breakout_direction == "UP" and t.cvd_range < 0) or
                (t.breakout_direction == "DOWN" and t.cvd_range > 0))),
        ("1-candle + pen < 50%", lambda t:
            t.candles_to_fakeout == 1 and t.breakout_penetration < 50),
        ("CVD fade + low ramp", lambda t:
            t.volume_ramp < 0 and (
                (t.breakout_direction == "UP" and t.cvd_range < 0) or
                (t.breakout_direction == "DOWN" and t.cvd_range > 0))),
        ("1-candle + CVD + pen<50", lambda t:
            t.candles_to_fakeout == 1 and t.breakout_penetration < 50 and (
                (t.breakout_direction == "UP" and t.cvd_range < 0) or
                (t.breakout_direction == "DOWN" and t.cvd_range > 0))),
    ]

    print(f"  {'Combination':<28s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 80)

    for label, filter_fn in combos:
        subset = [t for t in trades if filter_fn(t)]
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        dd = _max_drawdown_r(subset)
        print(f"  {label:<28s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f} {dd:>6.1f}R")

    print()


def report_monthly(trades: list[FakeoutTradeResult]) -> None:
    """Monthly consistency check."""
    print("=" * 80)
    print("  MONTHLY BREAKDOWN")
    print("=" * 80)
    print(f"  {'Month':<10s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'Cum R':>9s}")
    print("  " + "-" * 64)

    all_months = sorted(set((t.trade_date.year, t.trade_date.month) for t in trades))
    cum_r = 0.0
    pos_months = 0
    neg_months = 0

    for year, month in all_months:
        subset = [t for t in trades if t.trade_date.year == year and t.trade_date.month == month]
        if not subset:
            continue
        n, wr, avg_rr = _band_stats(subset)
        total = _total_r(subset)
        pf = _profit_factor(subset)
        cum_r += total
        if total > 0:
            pos_months += 1
        elif total < 0:
            neg_months += 1
        print(f"  {year}-{month:02d}    {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f} {cum_r:>+9.2f}")

    total_months = pos_months + neg_months
    print("  " + "-" * 64)
    if total_months > 0:
        print(f"  Positive months: {pos_months}/{total_months} ({pos_months / total_months:.0%})")
    print()


# ─────────────────────────────────────────────────────────────
#  RR target sweep
# ─────────────────────────────────────────────────────────────


def report_rr_sweep(
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]],
    session_configs: dict[SessionName, SessionConfig],
    fakeout_window: int,
    max_trades: int,
) -> None:
    """Test multiple RR targets to find the sweet spot."""
    print("=" * 80)
    print("  RR TARGET SWEEP")
    print("=" * 80)
    print(f"  {'RR Target':<10s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 64)

    for rr in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        trades = collect_fakeout_trades(
            instrument_groups, session_configs, rr,
            fakeout_window=fakeout_window,
            max_trades_per_session=max_trades,
        )
        n, wr, avg_rr = _band_stats(trades)
        total = _total_r(trades)
        pf = _profit_factor(trades)
        dd = _max_drawdown_r(trades)
        print(f"  {rr:<10.1f} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {total:>+9.2f} {pf:>7.2f} {dd:>6.1f}R")

    print()


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────


def load_candles_from_cache(instrument: str) -> Optional[list[Candle]]:
    pattern = f"{instrument}_*_M5.pkl"
    matches = sorted(CACHE_DIR.glob(pattern))
    if not matches:
        return None
    cache_file = matches[-1]
    with open(cache_file, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fade the Fakeout Backtest")
    parser.add_argument("--instruments", type=str, default="XAU_USD,NAS100_USD")
    parser.add_argument("--sessions", type=str, default="TOKYO,LONDON,NY")
    parser.add_argument("--rr-target", type=float, default=2.0)
    parser.add_argument("--fakeout-window", type=int, default=3,
                        help="Max candles after breakout to detect reversion (default: 3)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING,
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
            print(f"Unknown session: {s}")
            sys.exit(1)

    session_configs = {name: SESSIONS[name] for name in session_names}

    print()
    print("=" * 80)
    print("  FADE THE FAKEOUT BACKTEST")
    print("=" * 80)
    print(f"  Instruments:     {', '.join(instruments)}")
    print(f"  Sessions:        {', '.join(s.value for s in session_names)}")
    print(f"  RR Target:       {args.rr_target}")
    print(f"  Fakeout Window:  {args.fakeout_window} candles")
    print()

    # Load data
    instrument_candles: dict[str, list[Candle]] = {}
    for inst in instruments:
        candles = load_candles_from_cache(inst)
        if candles is None:
            print(f"  WARNING: No cached data for {inst}")
            continue
        instrument_candles[inst] = candles
        first_ny = candle_ny_time(candles[0])
        last_ny = candle_ny_time(candles[-1])
        print(f"  {inst}: {len(candles)} candles ({first_ny.date()} to {last_ny.date()})")

    if not instrument_candles:
        print("ERROR: No data. Run backtest.py first to cache data.")
        sys.exit(1)

    # Group by session day
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]] = {}
    for inst, candles in instrument_candles.items():
        groups = group_candles_by_session_day(candles, list(session_configs.values()))
        instrument_groups[inst] = groups
        print(f"  {inst}: {len(groups)} session-days")

    print()
    print("  Running simulation...")

    trades = collect_fakeout_trades(
        instrument_groups=instrument_groups,
        session_configs=session_configs,
        rr_target=args.rr_target,
        fakeout_window=args.fakeout_window,
        max_trades_per_session=TradingConfig.MAX_TRADES_PER_SESSION,
        verbose=args.verbose,
    )

    print(f"  Total fakeout trades: {len(trades)}")

    if not trades:
        print("No fakeout trades found.")
        return

    # Reports
    report_baseline(trades)
    report_instrument_session(trades)
    report_fakeout_window(trades)
    report_penetration(trades)
    report_cvd_filter(trades)
    report_volume_ramp_filter(trades)
    report_combined_filters(trades)
    report_monthly(trades)
    report_rr_sweep(instrument_groups, session_configs, args.fakeout_window, TradingConfig.MAX_TRADES_PER_SESSION)


if __name__ == "__main__":
    main()
