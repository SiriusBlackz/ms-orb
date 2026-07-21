#!/usr/bin/env python3
"""Strategy 3: Session Momentum — Directional Bias + Pullback Entry.

Detects early session directional bias from the first N post-range candles.
If a clear majority close in the same direction, that's the session bias.
Enter on the first pullback (counter-direction candle) in the bias direction.

Hypothesis: Early breakouts with volume ramp had the best PF in MS-ORB.
Instead of requiring a range breakout, use candle direction consensus as
the bias signal. This captures trending sessions regardless of whether
price breaks a specific level.

Usage:
    python backtest_momentum.py [--instruments XAU_USD,NAS100_USD]
                                [--sessions TOKYO,LONDON,NY]
                                [--rr-target 5.0]
                                [--bias-candles 6]
                                [--bias-threshold 4]
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
from strategy import Direction, RangeRecorder

logger = logging.getLogger(__name__)

NY_TZ = pytz.timezone(TIMEZONE)
CACHE_DIR = Path(__file__).parent / "cache"


@dataclass
class MomentumTradeResult:
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
    # Momentum metadata
    bias_strength: int            # How many of N candles matched bias direction
    bias_candles_total: int       # N (window size)
    range_high: float
    range_low: float
    range_size: float
    entry_vs_range: str           # "above", "below", "inside" range at entry
    cvd_bias_candles: float       # CVD over the bias-detection candles
    pullback_candle_idx: int      # Which candle after bias window was the pullback


# ─────────────────────────────────────────────────────────────
#  Time helpers
# ─────────────────────────────────────────────────────────────


def candle_ny_time(candle: Candle) -> datetime:
    utc = candle.time
    if utc.tzinfo is None: utc = utc.replace(tzinfo=timezone.utc)
    return utc.astimezone(NY_TZ)

def assign_session_date(ny_time, cfg):
    t = ny_time.time()
    if cfg.name == SessionName.TOKYO:
        if t >= cfg.range_start: return ny_time.date()
        elif t <= cfg.session_close: return (ny_time - timedelta(days=1)).date()
        else: return ny_time.date()
    return ny_time.date()

def is_in_session_window(ny_time, cfg):
    t = ny_time.time()
    if cfg.name == SessionName.TOKYO:
        return t >= cfg.range_start or t <= cfg.session_close
    return cfg.range_start <= t <= cfg.session_close

def is_in_range_period(ny_time, cfg):
    return cfg.range_start <= ny_time.time() < cfg.range_end

def is_in_entry_window(ny_time, cfg):
    return cfg.range_end <= ny_time.time() < cfg.entry_end

def is_past_session_close(ny_time, cfg):
    t = ny_time.time()
    if cfg.name == SessionName.TOKYO:
        if t <= cfg.session_close: return False
        elif t < cfg.range_start: return True
        return False
    return t >= cfg.session_close

def group_candles_by_session_day(candles, sessions):
    groups = {}
    for c in candles:
        if not c.complete: continue
        ny = candle_ny_time(c)
        for cfg in sessions:
            if is_in_session_window(ny, cfg):
                sd = assign_session_date(ny, cfg)
                key = (cfg.name, sd)
                if key not in groups: groups[key] = []
                groups[key].append(c)
    for k in groups: groups[k].sort(key=lambda c: c.time)
    return groups


# ─────────────────────────────────────────────────────────────
#  Trade simulation
# ─────────────────────────────────────────────────────────────


def _calc_rr(direction, entry, exit_price, risk):
    if risk <= 0: return 0.0
    if direction == Direction.LONG: return (exit_price - entry) / risk
    return (entry - exit_price) / risk

def simulate_trade(entry, sl, tp, direction, risk, candles_after, cfg):
    for c in candles_after:
        ny = candle_ny_time(c)
        if is_past_session_close(ny, cfg):
            return c.open, "time", c.time
        if direction == Direction.LONG:
            sl_hit = c.low <= sl; tp_hit = c.high >= tp
        else:
            sl_hit = c.high >= sl; tp_hit = c.low <= tp
        if sl_hit and tp_hit:
            if abs(c.open - sl) <= abs(c.open - tp): return sl, "stop", c.time
            else: return tp, "target", c.time
        elif sl_hit: return sl, "stop", c.time
        elif tp_hit: return tp, "target", c.time
    if candles_after:
        return candles_after[-1].close, "time", candles_after[-1].time
    return None

def compute_cvd(candles):
    total = 0
    for c in candles:
        if c.close > c.open: total += c.volume
        elif c.close < c.open: total -= c.volume
    return total


# ─────────────────────────────────────────────────────────────
#  Momentum detection and simulation
# ─────────────────────────────────────────────────────────────


def collect_momentum_trades(
    instrument_groups, session_configs, rr_target,
    bias_candles=6, bias_threshold=4,
    max_trades_per_session=2, verbose=False,
):
    """Detect session momentum bias and enter on pullback.

    Logic:
    1. After range period ends, examine first `bias_candles` post-range candles
    2. Count bullish (close > open) vs bearish (close < open) candles
    3. If bullish count >= bias_threshold → LONG bias
       If bearish count >= bias_threshold → SHORT bias
       Otherwise → no bias, skip session
    4. After bias established, wait for first pullback candle
       (candle closing against the bias direction)
    5. Enter on pullback candle close in the bias direction
    6. SL: pullback candle low (LONG) or high (SHORT)
    7. TP: entry ± risk * rr_target
    """
    all_trades = []

    for inst, groups in instrument_groups.items():
        sorted_items = sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0].value))

        for (session_name, session_date), candles in sorted_items:
            if session_name not in session_configs:
                continue
            cfg = session_configs[session_name]

            range_candles = []
            post_range = []
            for c in candles:
                ny = candle_ny_time(c)
                if is_in_range_period(ny, cfg):
                    range_candles.append(c)
                else:
                    post_range.append(c)

            if not range_candles:
                continue

            # Build range for context
            recorder = RangeRecorder(session_name, inst)
            for c in range_candles:
                recorder.add_candle(c)
            opening_range = recorder.finalize()
            if opening_range is None:
                continue

            range_high = opening_range.high
            range_low = opening_range.low
            range_size = range_high - range_low

            # Filter post-range to entry window only for bias detection
            entry_candles = []
            for c in post_range:
                ny = candle_ny_time(c)
                if is_in_entry_window(ny, cfg):
                    entry_candles.append(c)

            if len(entry_candles) < bias_candles:
                continue

            # Detect bias from first N candles
            bias_window = entry_candles[:bias_candles]
            bullish = sum(1 for c in bias_window if c.close > c.open)
            bearish = sum(1 for c in bias_window if c.close < c.open)

            if bullish >= bias_threshold:
                bias_dir = Direction.LONG
                bias_strength = bullish
            elif bearish >= bias_threshold:
                bias_dir = Direction.SHORT
                bias_strength = bearish
            else:
                continue  # No clear bias

            cvd_bias = compute_cvd(bias_window)

            # Look for pullback after the bias window
            trades_taken = 0
            remaining_candles = entry_candles[bias_candles:]

            for pb_idx, pb_candle in enumerate(remaining_candles):
                if trades_taken >= max_trades_per_session:
                    break

                ny = candle_ny_time(pb_candle)
                if is_past_session_close(ny, cfg):
                    break

                # Pullback = candle closing against bias
                is_pullback = False
                if bias_dir == Direction.LONG and pb_candle.close < pb_candle.open:
                    is_pullback = True
                elif bias_dir == Direction.SHORT and pb_candle.close > pb_candle.open:
                    is_pullback = True

                if not is_pullback:
                    continue

                # Enter on pullback candle close
                entry_price = pb_candle.close

                if bias_dir == Direction.LONG:
                    stop_loss = pb_candle.low
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        continue
                    take_profit = entry_price + (risk * rr_target)
                else:
                    stop_loss = pb_candle.high
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        continue
                    take_profit = entry_price - (risk * rr_target)

                # Where is entry vs range?
                if entry_price > range_high:
                    entry_vs = "above"
                elif entry_price < range_low:
                    entry_vs = "below"
                else:
                    entry_vs = "inside"

                # Simulate
                sim_candles = remaining_candles[pb_idx + 1:]
                # Also include post-entry-window candles for the trade to play out
                post_entry_window = [c for c in post_range if c not in entry_candles and c.time > pb_candle.time]
                sim_candles = sim_candles + post_entry_window

                result = simulate_trade(
                    entry_price, stop_loss, take_profit,
                    bias_dir, risk, sim_candles, cfg,
                )

                if result:
                    exit_price, exit_reason, exit_time = result
                    rr = _calc_rr(bias_dir, entry_price, exit_price, risk)

                    trade = MomentumTradeResult(
                        instrument=inst,
                        session=session_name,
                        trade_date=session_date,
                        direction=bias_dir,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        rr_achieved=rr,
                        entry_time=pb_candle.time,
                        exit_time=exit_time,
                        bias_strength=bias_strength,
                        bias_candles_total=bias_candles,
                        range_high=range_high,
                        range_low=range_low,
                        range_size=range_size,
                        entry_vs_range=entry_vs,
                        cvd_bias_candles=cvd_bias,
                        pullback_candle_idx=pb_idx,
                    )
                    all_trades.append(trade)
                    trades_taken += 1

                    if verbose:
                        print(
                            f"  {session_date} {session_name.value:6s} {inst:11s} "
                            f"{bias_dir.value:5s} bias={bias_strength}/{bias_candles} "
                            f"entry={entry_price:>10.2f} exit={exit_price:>10.2f} "
                            f"reason={exit_reason:6s} RR={rr:+.2f} pos={entry_vs}"
                        )

                    if exit_reason == "stop":
                        # Skip past exit, allow re-entry
                        continue
                    else:
                        break  # Target/time → done

    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────


def _stats(trades):
    n = len(trades)
    if not n: return 0, 0.0, 0.0
    return n, sum(1 for t in trades if t.rr_achieved > 0) / n, sum(t.rr_achieved for t in trades) / n

def _total_r(trades): return sum(t.rr_achieved for t in trades)

def _pf(trades):
    gp = sum(t.rr_achieved for t in trades if t.rr_achieved > 0)
    gl = abs(sum(t.rr_achieved for t in trades if t.rr_achieved <= 0))
    return gp / gl if gl > 0 else float("inf")

def _max_dd(trades):
    if not trades: return 0.0
    peak = eq = dd = 0.0
    for t in trades:
        eq += t.rr_achieved
        if eq > peak: peak = eq
        d = peak - eq
        if d > dd: dd = d
    return dd


def report_all(trades, instrument_groups, session_configs, bias_candles, bias_threshold, max_trades):
    n, wr, avg = _stats(trades)
    total = _total_r(trades)
    pf = _pf(trades)
    dd = _max_dd(trades)
    stops = sum(1 for t in trades if t.exit_reason == "stop")
    targets = sum(1 for t in trades if t.exit_reason == "target")
    time_exits = sum(1 for t in trades if t.exit_reason == "time")

    print(f"\n{'=' * 80}")
    print(f"  SESSION MOMENTUM — BASELINE RESULTS")
    print(f"{'=' * 80}")
    print(f"  Trades:     {n}")
    print(f"  Win Rate:   {wr:.1%}")
    print(f"  Avg RR:     {avg:+.2f}")
    print(f"  Total R:    {total:+.2f}")
    print(f"  PF:         {pf:.2f}")
    print(f"  Max DD:     {dd:.2f}R")
    print(f"  Exits:      stops={stops}  targets={targets}  time={time_exits}")

    longs = [t for t in trades if t.direction == Direction.LONG]
    shorts = [t for t in trades if t.direction == Direction.SHORT]
    print(f"\n  {'Dir':<8s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 48)
    for l, s in [("LONG", longs), ("SHORT", shorts)]:
        sn, sw, sa = _stats(s)
        print(f"  {l:<8s} {sn:>7d} {sw:>6.1%} {sa:>+8.2f} {_total_r(s):>+9.2f} {_pf(s):>7.2f}")

    # Instrument x Session
    print(f"\n{'=' * 80}")
    print(f"  BREAKDOWN: Instrument x Session")
    print(f"{'=' * 80}")
    print(f"  {'Instrument':<12s} {'Session':<8s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 66)
    instruments = sorted(set(t.instrument for t in trades))
    for inst in instruments:
        for sess in [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]:
            sub = [t for t in trades if t.instrument == inst and t.session == sess]
            if not sub: continue
            sn, sw, sa = _stats(sub)
            print(f"  {inst:<12s} {sess.value:<8s} {sn:>7d} {sw:>6.1%} {sa:>+8.2f} {_total_r(sub):>+9.2f} {_pf(sub):>7.2f}")
    print("  " + "-" * 66)
    print(f"  {'TOTAL':<12s} {'ALL':<8s} {n:>7d} {wr:>6.1%} {avg:>+8.2f} {total:>+9.2f} {pf:>7.2f}")

    # Filters
    print(f"\n{'=' * 80}")
    print(f"  FILTER ANALYSIS")
    print(f"{'=' * 80}")
    combos = [
        ("Baseline", lambda t: True),
        (f"Bias {bias_candles}/{bias_candles} (perfect)", lambda t: t.bias_strength == t.bias_candles_total),
        (f"Bias {bias_threshold}+ (threshold)", lambda t: t.bias_strength >= bias_threshold),
        ("Entry above range", lambda t: t.entry_vs_range == "above"),
        ("Entry below range", lambda t: t.entry_vs_range == "below"),
        ("Entry inside range", lambda t: t.entry_vs_range == "inside"),
        ("CVD confirms bias", lambda t:
            (t.direction == Direction.LONG and t.cvd_bias_candles > 0) or
            (t.direction == Direction.SHORT and t.cvd_bias_candles < 0)),
        ("First pullback (idx 0)", lambda t: t.pullback_candle_idx == 0),
        ("Early PB (idx 0-2)", lambda t: t.pullback_candle_idx <= 2),
        ("Late PB (idx 3+)", lambda t: t.pullback_candle_idx >= 3),
        ("Perfect bias + CVD", lambda t:
            t.bias_strength == t.bias_candles_total and (
                (t.direction == Direction.LONG and t.cvd_bias_candles > 0) or
                (t.direction == Direction.SHORT and t.cvd_bias_candles < 0))),
        ("Above/below range + CVD", lambda t:
            t.entry_vs_range != "inside" and (
                (t.direction == Direction.LONG and t.cvd_bias_candles > 0) or
                (t.direction == Direction.SHORT and t.cvd_bias_candles < 0))),
    ]
    print(f"  {'Combination':<28s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 80)
    for label, fn in combos:
        sub = [t for t in trades if fn(t)]
        sn, sw, sa = _stats(sub)
        print(f"  {label:<28s} {sn:>7d} {sw:>6.1%} {sa:>+8.2f} {_total_r(sub):>+9.2f} {_pf(sub):>7.2f} {_max_dd(sub):>6.1f}R")

    # Monthly
    print(f"\n{'=' * 80}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"{'=' * 80}")
    print(f"  {'Month':<10s} {'Trades':>7s} {'Win%':>7s} {'Total R':>9s} {'PF':>7s} {'Cum R':>9s}")
    print("  " + "-" * 52)
    months = sorted(set((t.trade_date.year, t.trade_date.month) for t in trades))
    cum = 0.0; pos = neg = 0
    for y, m in months:
        sub = [t for t in trades if t.trade_date.year == y and t.trade_date.month == m]
        if not sub: continue
        sn, sw, _ = _stats(sub)
        total_m = _total_r(sub); cum += total_m
        if total_m > 0: pos += 1
        elif total_m < 0: neg += 1
        print(f"  {y}-{m:02d}    {sn:>7d} {sw:>6.1%} {total_m:>+9.2f} {_pf(sub):>7.2f} {cum:>+9.2f}")
    print("  " + "-" * 52)
    t = pos + neg
    if t: print(f"  Positive months: {pos}/{t} ({pos/t:.0%})")

    # RR sweep
    print(f"\n{'=' * 80}")
    print(f"  RR TARGET SWEEP")
    print(f"{'=' * 80}")
    print(f"  {'RR':<6s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 58)
    for rr in [1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        tr = collect_momentum_trades(
            instrument_groups, session_configs, rr,
            bias_candles=bias_candles, bias_threshold=bias_threshold,
            max_trades_per_session=max_trades,
        )
        sn, sw, sa = _stats(tr)
        print(f"  {rr:<6.1f} {sn:>7d} {sw:>6.1%} {sa:>+8.2f} {_total_r(tr):>+9.2f} {_pf(tr):>7.2f} {_max_dd(tr):>6.1f}R")

    # Bias threshold sweep
    print(f"\n{'=' * 80}")
    print(f"  BIAS THRESHOLD SWEEP (window={bias_candles})")
    print(f"{'=' * 80}")
    print(f"  {'Threshold':<12s} {'Trades':>7s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 56)
    for thresh in range(3, bias_candles + 1):
        tr = collect_momentum_trades(
            instrument_groups, session_configs, rr_target=5.0,
            bias_candles=bias_candles, bias_threshold=thresh,
            max_trades_per_session=max_trades,
        )
        sn, sw, sa = _stats(tr)
        print(f"  {thresh}/{bias_candles}        {sn:>7d} {sw:>6.1%} {sa:>+8.2f} {_total_r(tr):>+9.2f} {_pf(tr):>7.2f}")

    print()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────


def load_candles_from_cache(instrument):
    matches = sorted(CACHE_DIR.glob(f"{instrument}_*_M5.pkl"))
    if not matches: return None
    with open(matches[-1], "rb") as f: return pickle.load(f)


def main():
    p = argparse.ArgumentParser(description="Session Momentum Backtest")
    p.add_argument("--instruments", default="XAU_USD,NAS100_USD")
    p.add_argument("--sessions", default="TOKYO,LONDON,NY")
    p.add_argument("--rr-target", type=float, default=5.0)
    p.add_argument("--bias-candles", type=int, default=6, help="Candles to detect bias (default: 6 = 30 min)")
    p.add_argument("--bias-threshold", type=int, default=4, help="Min candles in same direction for bias (default: 4/6)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)

    instruments = [s.strip() for s in args.instruments.split(",")]
    session_names = []
    for s in args.sessions.split(","):
        s = s.strip().upper()
        try: session_names.append(SessionName(s))
        except ValueError: print(f"Unknown: {s}"); sys.exit(1)
    session_configs = {n: SESSIONS[n] for n in session_names}

    print(f"\n{'=' * 80}")
    print(f"  SESSION MOMENTUM BACKTEST")
    print(f"{'=' * 80}")
    print(f"  Instruments:     {', '.join(instruments)}")
    print(f"  Sessions:        {', '.join(s.value for s in session_names)}")
    print(f"  RR Target:       {args.rr_target}")
    print(f"  Bias Window:     {args.bias_candles} candles")
    print(f"  Bias Threshold:  {args.bias_threshold}/{args.bias_candles}")
    print()

    ic = {}
    for inst in instruments:
        c = load_candles_from_cache(inst)
        if c is None: print(f"  WARNING: No data for {inst}"); continue
        ic[inst] = c
        print(f"  {inst}: {len(c)} candles ({candle_ny_time(c[0]).date()} to {candle_ny_time(c[-1]).date()})")
    if not ic: print("ERROR: No data."); sys.exit(1)

    ig = {}
    for inst, candles in ic.items():
        g = group_candles_by_session_day(candles, list(session_configs.values()))
        ig[inst] = g
        print(f"  {inst}: {len(g)} session-days")

    print(f"\n  Running simulation...")
    trades = collect_momentum_trades(
        ig, session_configs, args.rr_target,
        bias_candles=args.bias_candles, bias_threshold=args.bias_threshold,
        max_trades_per_session=TradingConfig.MAX_TRADES_PER_SESSION,
        verbose=args.verbose,
    )
    print(f"  Total momentum trades: {len(trades)}")
    if not trades: print("No trades found."); return

    report_all(trades, ig, session_configs, args.bias_candles, args.bias_threshold, TradingConfig.MAX_TRADES_PER_SESSION)


if __name__ == "__main__":
    main()
