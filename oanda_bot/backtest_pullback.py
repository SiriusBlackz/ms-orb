#!/usr/bin/env python3
"""Strategy 2: Breakout Pullback — Continuation Entry Backtest.

Instead of entering on the breakout candle (wide SL at candle low/high),
wait for the first pullback toward the range boundary after a confirmed
breakout. Enter on the pullback for a tighter stop and better RR.

Hypothesis: The breakouts that work show CVD alignment + volume ramp.
By waiting for a pullback, we get a better entry price, tighter stop
(below the pullback low instead of below the breakout candle low),
and therefore better RR for the same TP distance.

Usage:
    python backtest_pullback.py [--instruments XAU_USD,NAS100_USD]
                                [--sessions TOKYO,LONDON,NY]
                                [--rr-target 3.0]
                                [--pullback-window 6]
                                [--pullback-depth 0.2]
                                [--verbose]
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
class PullbackTradeResult:
    """Single simulated pullback-continuation trade."""
    instrument: str
    session: SessionName
    trade_date: date
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float
    exit_reason: str
    rr_achieved: float
    entry_time: datetime
    exit_time: datetime
    # Pullback metadata
    range_high: float
    range_low: float
    range_size: float
    breakout_candle_close: float
    breakout_candle_sl: float     # What the SL would have been on breakout entry
    pullback_depth: float         # How far price pulled back (% of breakout move)
    candles_to_pullback: int      # Candles from breakout to pullback entry
    cvd_range: float
    cvd_aligned: bool
    volume_ramp: float
    original_rr: float            # RR if entered on breakout candle (for comparison)


# ─────────────────────────────────────────────────────────────
#  Time helpers
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
#  Trade simulation + order flow helpers
# ─────────────────────────────────────────────────────────────


def _calc_rr(direction: Direction, entry: float, exit_price: float, risk: float) -> float:
    if risk <= 0:
        return 0.0
    if direction == Direction.LONG:
        return (exit_price - entry) / risk
    else:
        return (entry - exit_price) / risk


def simulate_trade(
    entry_price: float, stop_loss: float, take_profit: float,
    direction: Direction, risk: float,
    candles_after_entry: list[Candle], session_cfg: SessionConfig,
) -> Optional[tuple[float, str, datetime]]:
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
            if abs(candle.open - stop_loss) <= abs(candle.open - take_profit):
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
#  Pullback detection and simulation
# ─────────────────────────────────────────────────────────────


def collect_pullback_trades(
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]],
    session_configs: dict[SessionName, SessionConfig],
    rr_target: float,
    pullback_window: int = 6,
    pullback_depth_min: float = 0.2,
    max_trades_per_session: int = 2,
    verbose: bool = False,
) -> list[PullbackTradeResult]:
    """Detect breakouts, wait for pullback, enter continuation.

    Logic:
    1. Detect breakout (candle closes beyond range)
    2. Watch next `pullback_window` candles for a pullback:
       - LONG breakout: pullback = candle that closes lower than the breakout candle close,
         pulling back at least `pullback_depth_min` of the breakout move toward range_high
       - SHORT breakout: mirror logic
    3. Enter on pullback candle close in the breakout direction
    4. SL: pullback candle low (LONG) or high (SHORT) — tighter than breakout candle SL
    5. TP: entry ± risk * rr_target
    6. If no pullback within window, skip (breakout ran away — missed it)
    """
    all_trades: list[PullbackTradeResult] = []

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

            cvd_range = compute_cvd(range_candles)

            trades_taken = 0
            breakout_up_used = False
            breakout_down_used = False
            i = 0

            while i < len(post_range_candles) and trades_taken < max_trades_per_session:
                candle = post_range_candles[i]
                ny_time = candle_ny_time(candle)

                if not is_in_entry_window(ny_time, session_cfg):
                    i += 1
                    continue

                # Detect breakout
                breakout_up = candle.close > range_high and not breakout_up_used
                breakout_down = candle.close < range_low and not breakout_down_used

                if not breakout_up and not breakout_down:
                    i += 1
                    continue

                breakout_idx = i
                breakout_candle = candle

                if breakout_up:
                    bo_direction = Direction.LONG
                    breakout_move = candle.close - range_high
                    breakout_sl = candle.low  # Original MS-ORB SL
                    breakout_up_used = True
                else:
                    bo_direction = Direction.SHORT
                    breakout_move = range_low - candle.close
                    breakout_sl = candle.high
                    breakout_down_used = True

                if breakout_move <= 0:
                    i += 1
                    continue

                # Volume ramp at breakout
                pre_bo = post_range_candles[max(0, breakout_idx - 3):breakout_idx]
                if not pre_bo:
                    pre_bo = range_candles[-3:]
                vol_ramp = compute_volume_ramp(pre_bo)

                cvd_aligned = (
                    (bo_direction == Direction.LONG and cvd_range > 0) or
                    (bo_direction == Direction.SHORT and cvd_range < 0)
                )

                # Watch for pullback
                pullback_found = False
                pb_candle = None
                candles_to_pb = 0
                best_price_since_bo = breakout_candle.close  # Track if price extended first

                for j in range(1, pullback_window + 1):
                    check_idx = breakout_idx + j
                    if check_idx >= len(post_range_candles):
                        break
                    check_candle = post_range_candles[check_idx]
                    check_ny = candle_ny_time(check_candle)
                    if is_past_session_close(check_ny, session_cfg):
                        break

                    # Track extension
                    if bo_direction == Direction.LONG:
                        best_price_since_bo = max(best_price_since_bo, check_candle.high)
                        # Pullback = candle pulls back toward range_high
                        # Depth = how far from best price back toward range boundary
                        extension = best_price_since_bo - range_high
                        if extension <= 0:
                            continue
                        retracement = best_price_since_bo - check_candle.close
                        depth = retracement / extension

                        # Must still be above range (don't enter if fakeout)
                        if check_candle.close <= range_high:
                            break  # Fakeout, abort

                        if depth >= pullback_depth_min:
                            pullback_found = True
                            pb_candle = check_candle
                            candles_to_pb = j
                            break
                    else:
                        best_price_since_bo = min(best_price_since_bo, check_candle.low)
                        extension = range_low - best_price_since_bo
                        if extension <= 0:
                            continue
                        retracement = check_candle.close - best_price_since_bo
                        depth = retracement / extension

                        if check_candle.close >= range_low:
                            break  # Fakeout, abort

                        if depth >= pullback_depth_min:
                            pullback_found = True
                            pb_candle = check_candle
                            candles_to_pb = j
                            break

                if not pullback_found:
                    i = breakout_idx + pullback_window + 1
                    continue

                # Enter on pullback
                entry_price = pb_candle.close

                if bo_direction == Direction.LONG:
                    stop_loss = pb_candle.low
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        i = breakout_idx + candles_to_pb + 1
                        continue
                    take_profit = entry_price + (risk * rr_target)
                else:
                    stop_loss = pb_candle.high
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        i = breakout_idx + candles_to_pb + 1
                        continue
                    take_profit = entry_price - (risk * rr_target)

                # Compute what original MS-ORB RR would have been
                orig_risk = abs(breakout_candle.close - breakout_sl)
                if orig_risk > 0:
                    orig_tp_dist = orig_risk * rr_target
                    original_rr = rr_target
                else:
                    original_rr = 0.0

                # Simulate trade
                entry_idx = breakout_idx + candles_to_pb
                remaining = post_range_candles[entry_idx + 1:]
                result = simulate_trade(
                    entry_price, stop_loss, take_profit,
                    bo_direction, risk, remaining, session_cfg,
                )

                if result:
                    exit_price, exit_reason, exit_time = result
                    rr = _calc_rr(bo_direction, entry_price, exit_price, risk)

                    trade = PullbackTradeResult(
                        instrument=inst,
                        session=session_name,
                        trade_date=session_date,
                        direction=bo_direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        rr_achieved=rr,
                        entry_time=pb_candle.time,
                        exit_time=exit_time,
                        range_high=range_high,
                        range_low=range_low,
                        range_size=range_size,
                        breakout_candle_close=breakout_candle.close,
                        breakout_candle_sl=breakout_sl,
                        pullback_depth=depth,
                        candles_to_pullback=candles_to_pb,
                        cvd_range=cvd_range,
                        cvd_aligned=cvd_aligned,
                        volume_ramp=vol_ramp,
                        original_rr=original_rr,
                    )
                    all_trades.append(trade)
                    trades_taken += 1

                    if verbose:
                        _print_trade(trade)

                    if exit_reason == "stop":
                        # Allow re-entry: reset the used flag
                        if bo_direction == Direction.LONG:
                            breakout_up_used = False
                        else:
                            breakout_down_used = False
                        while i < len(post_range_candles) and post_range_candles[i].time <= exit_time:
                            i += 1
                        continue
                    else:
                        break
                i = breakout_idx + candles_to_pb + 1

    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


def _print_trade(t: PullbackTradeResult) -> None:
    print(
        f"  {t.trade_date} {t.session.value:6s} {t.instrument:11s} "
        f"{t.direction.value:5s} entry={t.entry_price:>10.2f} "
        f"exit={t.exit_price:>10.2f} reason={t.exit_reason:6s} "
        f"RR={t.rr_achieved:+.2f} depth={t.pullback_depth:.0%} "
        f"ttPB={t.candles_to_pullback} cvd={'Y' if t.cvd_aligned else 'N'}"
    )


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────


def _band_stats(trades):
    n = len(trades)
    if n == 0:
        return 0, 0.0, 0.0
    wins = sum(1 for t in trades if t.rr_achieved > 0)
    avg_rr = sum(t.rr_achieved for t in trades) / n
    return n, wins / n, avg_rr

def _total_r(trades): return sum(t.rr_achieved for t in trades)

def _profit_factor(trades):
    gp = sum(t.rr_achieved for t in trades if t.rr_achieved > 0)
    gl = abs(sum(t.rr_achieved for t in trades if t.rr_achieved <= 0))
    return gp / gl if gl > 0 else float("inf")

def _max_drawdown_r(trades):
    if not trades: return 0.0
    peak = eq = max_dd = 0.0
    for t in trades:
        eq += t.rr_achieved
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > max_dd: max_dd = dd
    return max_dd


def report_baseline(trades):
    n, wr, avg_rr = _band_stats(trades)
    total = _total_r(trades)
    pf = _profit_factor(trades)
    dd = _max_drawdown_r(trades)
    stops = sum(1 for t in trades if t.exit_reason == "stop")
    targets = sum(1 for t in trades if t.exit_reason == "target")
    time_exits = sum(1 for t in trades if t.exit_reason == "time")

    print("\n" + "=" * 80)
    print("  BREAKOUT PULLBACK — BASELINE RESULTS")
    print("=" * 80)
    print(f"  Trades:     {n}")
    print(f"  Win Rate:   {wr:.1%}")
    print(f"  Avg RR:     {avg_rr:+.2f}")
    print(f"  Total R:    {total:+.2f}")
    print(f"  PF:         {pf:.2f}")
    print(f"  Max DD:     {dd:.2f}R")
    print(f"  Exits:      stops={stops}  targets={targets}  time={time_exits}")

    longs = [t for t in trades if t.direction == Direction.LONG]
    shorts = [t for t in trades if t.direction == Direction.SHORT]
    print(f"\n  {'Direction':<8s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 48)
    for label, subset in [("LONG", longs), ("SHORT", shorts)]:
        sn, swr, savg = _band_stats(subset)
        print(f"  {label:<8s} {sn:>7d} {swr:>6.1%} {savg:>+8.2f} {_total_r(subset):>+9.2f} {_profit_factor(subset):>7.2f}")
    print()


def report_instrument_session(trades):
    print("=" * 80)
    print("  BREAKDOWN: Instrument x Session")
    print("=" * 80)
    print(f"  {'Instrument':<12s} {'Session':<8s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 66)
    instruments = sorted(set(t.instrument for t in trades))
    for inst in instruments:
        for sess in [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]:
            subset = [t for t in trades if t.instrument == inst and t.session == sess]
            if not subset: continue
            n, wr, avg_rr = _band_stats(subset)
            print(f"  {inst:<12s} {sess.value:<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {_total_r(subset):>+9.2f} {_profit_factor(subset):>7.2f}")
    print("  " + "-" * 66)
    n, wr, avg_rr = _band_stats(trades)
    print(f"  {'TOTAL':<12s} {'ALL':<8s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {_total_r(trades):>+9.2f} {_profit_factor(trades):>7.2f}")
    print()


def report_filters(trades):
    print("=" * 80)
    print("  FILTER ANALYSIS")
    print("=" * 80)
    combos = [
        ("Baseline", lambda t: True),
        ("CVD-aligned", lambda t: t.cvd_aligned),
        ("Ramp > 0", lambda t: t.volume_ramp > 0),
        ("CVD + ramp > 0", lambda t: t.cvd_aligned and t.volume_ramp > 0),
        ("Depth 20-50%", lambda t: 0.2 <= t.pullback_depth < 0.5),
        ("Depth 50-80%", lambda t: 0.5 <= t.pullback_depth < 0.8),
        ("PB in 1-2 candles", lambda t: t.candles_to_pullback <= 2),
        ("PB in 3-6 candles", lambda t: 3 <= t.candles_to_pullback <= 6),
        ("CVD + ramp + depth<50%", lambda t: t.cvd_aligned and t.volume_ramp > 0 and t.pullback_depth < 0.5),
        ("CVD + ramp + PB<=2", lambda t: t.cvd_aligned and t.volume_ramp > 0 and t.candles_to_pullback <= 2),
    ]
    print(f"  {'Combination':<28s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 80)
    for label, fn in combos:
        subset = [t for t in trades if fn(t)]
        n, wr, avg_rr = _band_stats(subset)
        print(f"  {label:<28s} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {_total_r(subset):>+9.2f} {_profit_factor(subset):>7.2f} {_max_drawdown_r(subset):>6.1f}R")
    print()


def report_rr_sweep(instrument_groups, session_configs, pullback_window, pullback_depth_min, max_trades):
    print("=" * 80)
    print("  RR TARGET SWEEP")
    print("=" * 80)
    print(f"  {'RR Target':<10s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 64)
    for rr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]:
        trades = collect_pullback_trades(
            instrument_groups, session_configs, rr,
            pullback_window=pullback_window,
            pullback_depth_min=pullback_depth_min,
            max_trades_per_session=max_trades,
        )
        n, wr, avg_rr = _band_stats(trades)
        print(f"  {rr:<10.1f} {n:>7d} {wr:>6.1%} {avg_rr:>+8.2f} {_total_r(trades):>+9.2f} {_profit_factor(trades):>7.2f} {_max_drawdown_r(trades):>6.1f}R")
    print()


def report_monthly(trades):
    print("=" * 80)
    print("  MONTHLY BREAKDOWN")
    print("=" * 80)
    print(f"  {'Month':<10s} {'Trades':>7s} {'Win%':>7s} {'Total R':>9s} {'PF':>7s} {'Cum R':>9s}")
    print("  " + "-" * 52)
    months = sorted(set((t.trade_date.year, t.trade_date.month) for t in trades))
    cum = 0.0
    pos = neg = 0
    for y, m in months:
        subset = [t for t in trades if t.trade_date.year == y and t.trade_date.month == m]
        if not subset: continue
        n, wr, _ = _band_stats(subset)
        total = _total_r(subset)
        cum += total
        if total > 0: pos += 1
        elif total < 0: neg += 1
        print(f"  {y}-{m:02d}    {n:>7d} {wr:>6.1%} {total:>+9.2f} {_profit_factor(subset):>7.2f} {cum:>+9.2f}")
    print("  " + "-" * 52)
    t = pos + neg
    if t: print(f"  Positive months: {pos}/{t} ({pos/t:.0%})")
    print()


# ─────────────────────────────────────────────────────────────
#  Data loading + Main
# ─────────────────────────────────────────────────────────────


def load_candles_from_cache(instrument):
    matches = sorted(CACHE_DIR.glob(f"{instrument}_*_M5.pkl"))
    if not matches: return None
    with open(matches[-1], "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Breakout Pullback Backtest")
    parser.add_argument("--instruments", type=str, default="XAU_USD,NAS100_USD")
    parser.add_argument("--sessions", type=str, default="TOKYO,LONDON,NY")
    parser.add_argument("--rr-target", type=float, default=3.0)
    parser.add_argument("--pullback-window", type=int, default=6)
    parser.add_argument("--pullback-depth", type=float, default=0.2,
                        help="Min pullback depth as fraction of breakout move (default: 0.2 = 20%%)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    instruments = [s.strip() for s in args.instruments.split(",")]
    session_names = []
    for s in args.sessions.split(","):
        s = s.strip().upper()
        try: session_names.append(SessionName(s))
        except ValueError: print(f"Unknown session: {s}"); sys.exit(1)

    session_configs = {name: SESSIONS[name] for name in session_names}

    print(f"\n{'=' * 80}")
    print("  BREAKOUT PULLBACK BACKTEST")
    print(f"{'=' * 80}")
    print(f"  Instruments:     {', '.join(instruments)}")
    print(f"  Sessions:        {', '.join(s.value for s in session_names)}")
    print(f"  RR Target:       {args.rr_target}")
    print(f"  Pullback Window: {args.pullback_window} candles")
    print(f"  Min PB Depth:    {args.pullback_depth:.0%}")
    print()

    instrument_candles = {}
    for inst in instruments:
        candles = load_candles_from_cache(inst)
        if candles is None: print(f"  WARNING: No cached data for {inst}"); continue
        instrument_candles[inst] = candles
        print(f"  {inst}: {len(candles)} candles ({candle_ny_time(candles[0]).date()} to {candle_ny_time(candles[-1]).date()})")

    if not instrument_candles: print("ERROR: No data."); sys.exit(1)

    instrument_groups = {}
    for inst, candles in instrument_candles.items():
        groups = group_candles_by_session_day(candles, list(session_configs.values()))
        instrument_groups[inst] = groups
        print(f"  {inst}: {len(groups)} session-days")

    print(f"\n  Running simulation...")

    trades = collect_pullback_trades(
        instrument_groups, session_configs, args.rr_target,
        pullback_window=args.pullback_window,
        pullback_depth_min=args.pullback_depth,
        max_trades_per_session=TradingConfig.MAX_TRADES_PER_SESSION,
        verbose=args.verbose,
    )
    print(f"  Total pullback trades: {len(trades)}")
    if not trades: print("No pullback trades found."); return

    report_baseline(trades)
    report_instrument_session(trades)
    report_filters(trades)
    report_monthly(trades)
    report_rr_sweep(instrument_groups, session_configs, args.pullback_window, args.pullback_depth, TradingConfig.MAX_TRADES_PER_SESSION)


if __name__ == "__main__":
    main()
