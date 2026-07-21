#!/usr/bin/env python3
"""Option B: Multi-Timeframe ORB Backtest.

Tests the opening range breakout on H1 candles (aggregated from cached M5 data).
Also tests M5 breakout with H1 range confirmation (multi-TF alignment).

Variants:
  1. Pure H1 ORB — range = first H1 candle of session, breakout on H1 close
  2. H1 range + M5 breakout — range from H1, but detect breakout on M5 for faster entry
  3. H1+M5 alignment — only take M5 breakouts when price is also beyond H1 range

Usage:
    python backtest_h1_orb.py [--instruments XAU_USD,NAS100_USD]
                              [--sessions TOKYO,LONDON,NY]
                              [--rr-target 5.0]
                              [--verbose]
"""

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import floor
from pathlib import Path
from typing import Optional

import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SESSIONS, TIMEZONE, SessionConfig, SessionName, TradingConfig
from oanda_client import Candle
from strategy import Direction, RangeRecorder

logger = logging.getLogger(__name__)
NY_TZ = pytz.timezone(TIMEZONE)
CACHE_DIR = Path(__file__).parent / "cache"


@dataclass
class H1OrbTradeResult:
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
    variant: str                  # "pure_h1", "h1_range_m5_break", "m5_h1_aligned"
    range_high: float
    range_low: float
    range_size: float
    range_candles_used: int       # 1 for H1, 6 for M5
    cvd_aligned: bool
    volume_ramp: float


# ─────────────────────────────────────────────────────────────
#  Time helpers
# ─────────────────────────────────────────────────────────────

def candle_ny_time(c):
    utc = c.time
    if utc.tzinfo is None: utc = utc.replace(tzinfo=timezone.utc)
    return utc.astimezone(NY_TZ)

def assign_session_date(ny, cfg):
    t = ny.time()
    if cfg.name == SessionName.TOKYO:
        if t >= cfg.range_start: return ny.date()
        elif t <= cfg.session_close: return (ny - timedelta(days=1)).date()
    return ny.date()

def is_in_session_window(ny, cfg):
    t = ny.time()
    if cfg.name == SessionName.TOKYO:
        return t >= cfg.range_start or t <= cfg.session_close
    return cfg.range_start <= t <= cfg.session_close

def is_in_range_period(ny, cfg):
    return cfg.range_start <= ny.time() < cfg.range_end

def is_in_entry_window(ny, cfg):
    return cfg.range_end <= ny.time() < cfg.entry_end

def is_past_session_close(ny, cfg):
    t = ny.time()
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
#  M5 → H1 aggregation
# ─────────────────────────────────────────────────────────────


def aggregate_m5_to_h1(m5_candles: list[Candle]) -> list[Candle]:
    """Aggregate M5 candles into H1 candles (12 M5 = 1 H1).

    Groups by the hour boundary of each M5 candle's time.
    """
    hourly_groups: dict[datetime, list[Candle]] = {}

    for c in m5_candles:
        # Truncate to hour boundary
        hour_key = c.time.replace(minute=0, second=0, microsecond=0)
        if hour_key not in hourly_groups:
            hourly_groups[hour_key] = []
        hourly_groups[hour_key].append(c)

    h1_candles = []
    for hour_time in sorted(hourly_groups.keys()):
        group = hourly_groups[hour_time]
        if not group:
            continue

        h1 = Candle(
            time=hour_time,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
            complete=all(c.complete for c in group) and len(group) >= 11,
        )
        h1_candles.append(h1)

    return h1_candles


# ─────────────────────────────────────────────────────────────
#  Trade simulation + helpers
# ─────────────────────────────────────────────────────────────

def _calc_rr(d, entry, exit_p, risk):
    if risk <= 0: return 0.0
    return (exit_p - entry) / risk if d == Direction.LONG else (entry - exit_p) / risk

def simulate_trade_on_candles(entry, sl, tp, direction, risk, candles, cfg):
    """Simulate on M5 candles for finer granularity exit detection."""
    for c in candles:
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
    if candles:
        return candles[-1].close, "time", candles[-1].time
    return None

def compute_cvd(candles):
    return sum(c.volume if c.close > c.open else (-c.volume if c.close < c.open else 0) for c in candles)

def compute_volume_ramp(candles):
    if len(candles) < 2: return 0.0
    vols = [c.volume for c in candles]
    avg = sum(vols) / len(vols)
    if avg <= 0: return 0.0
    return (vols[-1] - vols[0]) / avg


# ─────────────────────────────────────────────────────────────
#  Variant 1: Pure H1 ORB
# ─────────────────────────────────────────────────────────────


def collect_pure_h1_trades(
    instrument_groups_m5, instrument_groups_h1,
    session_configs, rr_target, max_trades=2,
):
    """Pure H1 ORB: range = first H1 candle, breakout detected on H1, simulated on M5."""
    all_trades = []

    for inst in instrument_groups_h1:
        h1_groups = instrument_groups_h1[inst]
        m5_groups = instrument_groups_m5.get(inst, {})

        for (session_name, session_date), h1_candles in sorted(h1_groups.items(), key=lambda kv: (kv[0][1], kv[0][0].value)):
            if session_name not in session_configs:
                continue
            cfg = session_configs[session_name]

            # Separate H1 candles into range and post-range
            range_h1 = []
            post_range_h1 = []
            for c in h1_candles:
                ny = candle_ny_time(c)
                if is_in_range_period(ny, cfg):
                    range_h1.append(c)
                elif ny.time() >= cfg.range_end:
                    post_range_h1.append(c)

            if not range_h1:
                continue

            # H1 range = first H1 candle(s) that fall in the range period
            # For 30-min range windows, only partial H1 candles may qualify
            # Use all range H1 candles to build the range
            range_high = max(c.high for c in range_h1)
            range_low = min(c.low for c in range_h1)
            range_size = range_high - range_low
            if range_size <= 0:
                continue

            cvd_range = compute_cvd(range_h1)

            # Get corresponding M5 candles for this session day (for simulation)
            m5_key = (session_name, session_date)
            m5_candles = m5_groups.get(m5_key, [])
            m5_post_range = [c for c in m5_candles if candle_ny_time(c).time() >= cfg.range_end]

            trades_taken = 0
            bo_up_used = False
            bo_down_used = False

            for h1_idx, h1_candle in enumerate(post_range_h1):
                if trades_taken >= max_trades:
                    break
                ny = candle_ny_time(h1_candle)
                if not is_in_entry_window(ny, cfg):
                    continue

                # Check H1 breakout
                bo_up = h1_candle.close > range_high and not bo_up_used
                bo_down = h1_candle.close < range_low and not bo_down_used

                if not bo_up and not bo_down:
                    continue

                if bo_up:
                    direction = Direction.LONG
                    entry = h1_candle.close
                    sl = h1_candle.low
                    risk = entry - sl
                    bo_up_used = True
                else:
                    direction = Direction.SHORT
                    entry = h1_candle.close
                    sl = h1_candle.high
                    risk = sl - entry
                    bo_down_used = True

                if risk <= 0:
                    continue

                tp = entry + (risk * rr_target) if direction == Direction.LONG else entry - (risk * rr_target)

                # Pre-breakout volume ramp (from H1 candles)
                pre_bo = post_range_h1[max(0, h1_idx - 2):h1_idx]
                if not pre_bo:
                    pre_bo = range_h1[-2:]
                vol_ramp = compute_volume_ramp(pre_bo)

                cvd_aligned = (
                    (direction == Direction.LONG and cvd_range > 0) or
                    (direction == Direction.SHORT and cvd_range < 0)
                )

                # Simulate on M5 candles after the H1 breakout candle's time
                sim_m5 = [c for c in m5_post_range if c.time > h1_candle.time]
                result = simulate_trade_on_candles(entry, sl, tp, direction, risk, sim_m5, cfg)

                if result:
                    exit_p, exit_r, exit_t = result
                    rr = _calc_rr(direction, entry, exit_p, risk)
                    all_trades.append(H1OrbTradeResult(
                        instrument=inst, session=session_name, trade_date=session_date,
                        direction=direction, entry_price=entry, stop_loss=sl,
                        take_profit=tp, exit_price=exit_p, exit_reason=exit_r,
                        rr_achieved=rr, entry_time=h1_candle.time, exit_time=exit_t,
                        variant="pure_h1", range_high=range_high, range_low=range_low,
                        range_size=range_size, range_candles_used=len(range_h1),
                        cvd_aligned=cvd_aligned, volume_ramp=vol_ramp,
                    ))
                    trades_taken += 1
                    if exit_r == "stop":
                        if direction == Direction.LONG: bo_up_used = False
                        else: bo_down_used = False
                    else:
                        break

    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


# ─────────────────────────────────────────────────────────────
#  Variant 2: H1 range + M5 breakout
# ─────────────────────────────────────────────────────────────


def collect_h1_range_m5_break_trades(
    instrument_groups_m5, instrument_groups_h1,
    session_configs, rr_target, max_trades=2,
):
    """H1 range boundaries, but breakout detected on M5 for faster entry. SL = M5 candle-based."""
    all_trades = []

    for inst in instrument_groups_h1:
        h1_groups = instrument_groups_h1[inst]
        m5_groups = instrument_groups_m5.get(inst, {})

        for (session_name, session_date), h1_candles in sorted(h1_groups.items(), key=lambda kv: (kv[0][1], kv[0][0].value)):
            if session_name not in session_configs:
                continue
            cfg = session_configs[session_name]

            range_h1 = [c for c in h1_candles if is_in_range_period(candle_ny_time(c), cfg)]
            if not range_h1:
                continue

            range_high = max(c.high for c in range_h1)
            range_low = min(c.low for c in range_h1)
            range_size = range_high - range_low
            if range_size <= 0:
                continue

            cvd_range = compute_cvd(range_h1)

            m5_key = (session_name, session_date)
            m5_candles = m5_groups.get(m5_key, [])
            m5_post_range = [c for c in m5_candles if candle_ny_time(c).time() >= cfg.range_end]

            trades_taken = 0
            bo_up_used = False
            bo_down_used = False

            for m5_idx, m5c in enumerate(m5_post_range):
                if trades_taken >= max_trades:
                    break
                ny = candle_ny_time(m5c)
                if not is_in_entry_window(ny, cfg):
                    continue

                # M5 breakout of H1 range
                bo_up = m5c.close > range_high and not bo_up_used
                bo_down = m5c.close < range_low and not bo_down_used

                if not bo_up and not bo_down:
                    continue

                if bo_up:
                    direction = Direction.LONG
                    entry = m5c.close
                    sl = m5c.low  # M5 candle-based SL (tighter)
                    risk = entry - sl
                    bo_up_used = True
                else:
                    direction = Direction.SHORT
                    entry = m5c.close
                    sl = m5c.high
                    risk = sl - entry
                    bo_down_used = True

                if risk <= 0:
                    continue

                tp = entry + (risk * rr_target) if direction == Direction.LONG else entry - (risk * rr_target)

                pre_bo = m5_post_range[max(0, m5_idx - 3):m5_idx]
                vol_ramp = compute_volume_ramp(pre_bo)

                cvd_aligned = (
                    (direction == Direction.LONG and cvd_range > 0) or
                    (direction == Direction.SHORT and cvd_range < 0)
                )

                sim = m5_post_range[m5_idx + 1:]
                result = simulate_trade_on_candles(entry, sl, tp, direction, risk, sim, cfg)

                if result:
                    exit_p, exit_r, exit_t = result
                    rr = _calc_rr(direction, entry, exit_p, risk)
                    all_trades.append(H1OrbTradeResult(
                        instrument=inst, session=session_name, trade_date=session_date,
                        direction=direction, entry_price=entry, stop_loss=sl,
                        take_profit=tp, exit_price=exit_p, exit_reason=exit_r,
                        rr_achieved=rr, entry_time=m5c.time, exit_time=exit_t,
                        variant="h1_range_m5_break", range_high=range_high, range_low=range_low,
                        range_size=range_size, range_candles_used=len(range_h1),
                        cvd_aligned=cvd_aligned, volume_ramp=vol_ramp,
                    ))
                    trades_taken += 1
                    if exit_r == "stop":
                        if direction == Direction.LONG: bo_up_used = False
                        else: bo_down_used = False
                    else:
                        break

    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────


def _s(trades):
    n = len(trades)
    if not n: return 0, 0.0, 0.0
    return n, sum(1 for t in trades if t.rr_achieved > 0) / n, sum(t.rr_achieved for t in trades) / n

def _tr(trades): return sum(t.rr_achieved for t in trades)
def _pf(trades):
    gp = sum(t.rr_achieved for t in trades if t.rr_achieved > 0)
    gl = abs(sum(t.rr_achieved for t in trades if t.rr_achieved <= 0))
    return gp / gl if gl > 0 else float("inf")
def _dd(trades):
    if not trades: return 0.0
    p = e = d = 0.0
    for t in trades:
        e += t.rr_achieved
        if e > p: p = e
        dd = p - e
        if dd > d: d = dd
    return d


def report_variant(trades, label):
    n, wr, avg = _s(trades)
    total = _tr(trades)
    pf = _pf(trades)
    dd = _dd(trades)
    stops = sum(1 for t in trades if t.exit_reason == "stop")
    targets = sum(1 for t in trades if t.exit_reason == "target")
    time_exits = sum(1 for t in trades if t.exit_reason == "time")

    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")
    print(f"  Trades: {n}  Win: {wr:.1%}  Avg RR: {avg:+.2f}  Total R: {total:+.1f}  PF: {pf:.2f}  MaxDD: {dd:.1f}R")
    print(f"  Exits: stops={stops}  targets={targets}  time={time_exits}")

    # Direction
    longs = [t for t in trades if t.direction == Direction.LONG]
    shorts = [t for t in trades if t.direction == Direction.SHORT]
    print(f"\n  {'Dir':<8s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 44)
    for l, sub in [("LONG", longs), ("SHORT", shorts)]:
        sn, sw, sa = _s(sub)
        print(f"  {l:<8s} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(sub):>+9.2f} {_pf(sub):>7.2f}")

    # Instrument x Session
    print(f"\n  {'Inst':<12s} {'Sess':<8s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 56)
    for inst in sorted(set(t.instrument for t in trades)):
        for sess in [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]:
            sub = [t for t in trades if t.instrument == inst and t.session == sess]
            if not sub: continue
            sn, sw, sa = _s(sub)
            print(f"  {inst:<12s} {sess.value:<8s} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(sub):>+9.2f} {_pf(sub):>7.2f}")

    # CVD filter
    cvd_y = [t for t in trades if t.cvd_aligned]
    cvd_n = [t for t in trades if not t.cvd_aligned]
    ramp_pos = [t for t in trades if t.volume_ramp > 0]
    cvd_ramp = [t for t in trades if t.cvd_aligned and t.volume_ramp > 0]

    print(f"\n  {'Filter':<24s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 58)
    for l, sub in [("All", trades), ("CVD-aligned", cvd_y), ("CVD-opposed", cvd_n),
                    ("Ramp > 0", ramp_pos), ("CVD + ramp > 0", cvd_ramp)]:
        sn, sw, sa = _s(sub)
        print(f"  {l:<24s} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(sub):>+9.2f} {_pf(sub):>7.2f}")

    # Monthly
    print(f"\n  {'Month':<10s} {'N':>5s} {'Win%':>7s} {'Total R':>9s} {'Cum R':>9s}")
    print("  " + "-" * 44)
    months = sorted(set((t.trade_date.year, t.trade_date.month) for t in trades))
    cum = 0.0; pos_m = 0
    for y, m in months:
        sub = [t for t in trades if t.trade_date.year == y and t.trade_date.month == m]
        if not sub: continue
        sn, sw, _ = _s(sub)
        total_m = _tr(sub); cum += total_m
        if total_m > 0: pos_m += 1
        print(f"  {y}-{m:02d}    {sn:>5d} {sw:>6.1%} {total_m:>+9.2f} {cum:>+9.2f}")
    total_m_count = len(months)
    if total_m_count: print(f"  Positive: {pos_m}/{total_m_count} ({pos_m/total_m_count:.0%})")


def report_rr_sweep(label, collect_fn, ig_m5, ig_h1, sc, max_trades):
    print(f"\n{'=' * 90}")
    print(f"  RR SWEEP — {label}")
    print(f"{'=' * 90}")
    print(f"  {'RR':<6s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 50)
    for rr in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        tr = collect_fn(ig_m5, ig_h1, sc, rr, max_trades)
        sn, sw, sa = _s(tr)
        print(f"  {rr:<6.1f} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(tr):>+9.2f} {_pf(tr):>7.2f} {_dd(tr):>6.1f}R")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def load_candles_from_cache(instrument):
    matches = sorted(CACHE_DIR.glob(f"{instrument}_*_M5.pkl"))
    if not matches: return None
    with open(matches[-1], "rb") as f: return pickle.load(f)


def main():
    p = argparse.ArgumentParser(description="H1 Multi-Timeframe ORB Backtest")
    p.add_argument("--instruments", default="XAU_USD,NAS100_USD")
    p.add_argument("--sessions", default="TOKYO,LONDON,NY")
    p.add_argument("--rr-target", type=float, default=5.0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    instruments = [s.strip() for s in args.instruments.split(",")]
    session_names = []
    for s in args.sessions.split(","):
        s = s.strip().upper()
        try: session_names.append(SessionName(s))
        except ValueError: print(f"Unknown: {s}"); sys.exit(1)
    sc = {n: SESSIONS[n] for n in session_names}

    print(f"\n{'=' * 90}")
    print(f"  H1 MULTI-TIMEFRAME ORB BACKTEST")
    print(f"{'=' * 90}")
    print(f"  Instruments: {', '.join(instruments)}")
    print(f"  Sessions:    {', '.join(s.value for s in session_names)}")
    print(f"  RR Target:   {args.rr_target}")
    print()

    # Load M5 data
    ic_m5 = {}
    for inst in instruments:
        c = load_candles_from_cache(inst)
        if c is None: print(f"  WARNING: No data for {inst}"); continue
        ic_m5[inst] = c
        print(f"  {inst}: {len(c)} M5 candles ({candle_ny_time(c[0]).date()} to {candle_ny_time(c[-1]).date()})")

    if not ic_m5: print("ERROR: No data."); sys.exit(1)

    # Aggregate to H1
    ic_h1 = {}
    for inst, m5s in ic_m5.items():
        h1s = aggregate_m5_to_h1(m5s)
        ic_h1[inst] = h1s
        print(f"  {inst}: {len(h1s)} H1 candles (aggregated)")

    # Group by session day
    ig_m5 = {}
    ig_h1 = {}
    for inst in instruments:
        if inst in ic_m5:
            ig_m5[inst] = group_candles_by_session_day(ic_m5[inst], list(sc.values()))
        if inst in ic_h1:
            ig_h1[inst] = group_candles_by_session_day(ic_h1[inst], list(sc.values()))

    print(f"\n  Running simulations...")

    # Variant 1: Pure H1 ORB
    trades_v1 = collect_pure_h1_trades(ig_m5, ig_h1, sc, args.rr_target)
    print(f"  Pure H1 ORB: {len(trades_v1)} trades")
    report_variant(trades_v1, "VARIANT 1: PURE H1 ORB (range=H1, breakout=H1, sim=M5)")

    # Variant 2: H1 range + M5 breakout
    trades_v2 = collect_h1_range_m5_break_trades(ig_m5, ig_h1, sc, args.rr_target)
    print(f"\n  H1 range + M5 breakout: {len(trades_v2)} trades")
    report_variant(trades_v2, "VARIANT 2: H1 RANGE + M5 BREAKOUT (range=H1, breakout=M5, SL=M5)")

    # Comparison table
    print(f"\n{'=' * 90}")
    print(f"  COMPARISON: All Variants at RR={args.rr_target}")
    print(f"{'=' * 90}")
    print(f"  {'Variant':<36s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 78)
    for label, tr in [
        ("Pure H1 ORB", trades_v1),
        ("H1 range + M5 breakout", trades_v2),
        ("(ref) MS-ORB M5 baseline*", []),  # placeholder
    ]:
        if not tr:
            print(f"  {label:<36s}  {'see backtest_orderflow.py':>50s}")
            continue
        sn, sw, sa = _s(tr)
        print(f"  {label:<36s} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(tr):>+9.2f} {_pf(tr):>7.2f} {_dd(tr):>6.1f}R")

    # RR sweeps
    report_rr_sweep("Pure H1 ORB", collect_pure_h1_trades, ig_m5, ig_h1, sc, TradingConfig.MAX_TRADES_PER_SESSION)
    report_rr_sweep("H1 Range + M5 Break", collect_h1_range_m5_break_trades, ig_m5, ig_h1, sc, TradingConfig.MAX_TRADES_PER_SESSION)

    print()


if __name__ == "__main__":
    main()
