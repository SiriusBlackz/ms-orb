#!/usr/bin/env python3
"""Option C: Crypto ORB Backtest — BTC, ETH, SOL.

Fetches historical M5 klines from Binance (free, no auth needed) and
runs the MS-ORB strategy with session structure adapted for crypto.

Crypto sessions (UTC-based, mapped to traditional market opens):
  - Asia:   00:00-08:00 UTC (Tokyo/HK/Singapore open)
  - Europe: 07:00-15:00 UTC (London open)
  - US:     13:00-21:00 UTC (NY open)

Range = first 30 min of each session. Same ORB logic as MS-ORB.

Usage:
    python backtest_crypto_orb.py [--symbols BTCUSDT,ETHUSDT,SOLUSDT]
                                  [--rr-target 5.0]
                                  [--months 12]
                                  [--verbose]
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time as time_mod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import Direction

logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


# ─────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────


@dataclass
class CryptoCandle:
    time: datetime        # UTC open time
    open: float
    high: float
    low: float
    close: float
    volume: float         # Base asset volume (BTC, ETH, etc.)
    complete: bool


@dataclass
class CryptoSession:
    name: str
    range_start: time     # UTC
    range_end: time       # UTC
    entry_end: time       # UTC
    session_close: time   # UTC


# Crypto sessions — all times UTC
CRYPTO_SESSIONS = {
    "ASIA": CryptoSession(
        name="ASIA",
        range_start=time(0, 0),
        range_end=time(0, 30),
        entry_end=time(3, 30),
        session_close=time(7, 55),
    ),
    "EUROPE": CryptoSession(
        name="EUROPE",
        range_start=time(7, 0),
        range_end=time(7, 30),
        entry_end=time(10, 30),
        session_close=time(12, 55),
    ),
    "US": CryptoSession(
        name="US",
        range_start=time(13, 0),
        range_end=time(13, 30),
        entry_end=time(16, 30),
        session_close=time(20, 55),
    ),
}


@dataclass
class CryptoTradeResult:
    symbol: str
    session: str
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
    range_high: float
    range_low: float
    range_size: float
    range_size_pct: float
    cvd_range: float
    cvd_aligned: bool
    volume_ramp: float


# ─────────────────────────────────────────────────────────────
#  Binance data fetching
# ─────────────────────────────────────────────────────────────


def fetch_binance_klines(
    symbol: str,
    interval: str = "5m",
    start_dt: datetime = None,
    end_dt: datetime = None,
    use_cache: bool = True,
) -> list[CryptoCandle]:
    """Fetch M5 klines from Binance with caching."""
    cache_key = f"crypto_{symbol}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}_{interval}"
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  Fetching {symbol} {interval} from Binance: {start_dt.date()} to {end_dt.date()}")

    all_candles = []
    current_start = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Binance API error: {e}")
            break

        if not data:
            break

        for k in data:
            open_time = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
            close_time = datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc)

            all_candles.append(CryptoCandle(
                time=open_time,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                complete=True,
            ))

        # Advance cursor
        last_open = data[-1][0]
        if last_open >= end_ms:
            break
        current_start = last_open + 1

        if len(data) < 1000:
            break

        time_mod.sleep(0.2)  # Rate limit

    print(f"  {symbol}: {len(all_candles)} candles fetched")

    if use_cache and all_candles:
        with open(cache_file, "wb") as f:
            pickle.dump(all_candles, f)

    return all_candles


# ─────────────────────────────────────────────────────────────
#  Session grouping
# ─────────────────────────────────────────────────────────────


def group_by_session_day(
    candles: list[CryptoCandle],
    sessions: dict[str, CryptoSession],
) -> dict[tuple[str, date], list[CryptoCandle]]:
    """Group candles into (session_name, date) buckets."""
    groups = {}
    for c in candles:
        if not c.complete:
            continue
        t = c.time.time()
        for name, sess in sessions.items():
            if sess.range_start <= t <= sess.session_close:
                key = (name, c.time.date())
                if key not in groups:
                    groups[key] = []
                groups[key].append(c)
    for k in groups:
        groups[k].sort(key=lambda c: c.time)
    return groups


# ─────────────────────────────────────────────────────────────
#  ORB logic + simulation
# ─────────────────────────────────────────────────────────────


def _calc_rr(d, entry, exit_p, risk):
    if risk <= 0: return 0.0
    return (exit_p - entry) / risk if d == Direction.LONG else (entry - exit_p) / risk


def simulate_trade(entry, sl, tp, direction, risk, candles_after, sess):
    for c in candles_after:
        t = c.time.time()
        if t >= sess.session_close:
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
    return sum(c.volume if c.close > c.open else (-c.volume if c.close < c.open else 0) for c in candles)


def compute_volume_ramp(candles):
    if len(candles) < 2: return 0.0
    vols = [c.volume for c in candles]
    avg = sum(vols) / len(vols)
    if avg <= 0: return 0.0
    return (vols[-1] - vols[0]) / avg


def collect_crypto_orb_trades(
    symbol: str,
    groups: dict[tuple[str, date], list[CryptoCandle]],
    sessions: dict[str, CryptoSession],
    rr_target: float,
    max_trades: int = 2,
    verbose: bool = False,
) -> list[CryptoTradeResult]:
    all_trades = []

    for (sess_name, sess_date), candles in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        sess = sessions[sess_name]

        range_candles = [c for c in candles if sess.range_start <= c.time.time() < sess.range_end]
        post_range = [c for c in candles if c.time.time() >= sess.range_end]

        if not range_candles:
            continue

        range_high = max(c.high for c in range_candles)
        range_low = min(c.low for c in range_candles)
        range_size = range_high - range_low
        if range_size <= 0:
            continue

        range_size_pct = (range_size / range_low) * 100

        cvd_range = compute_cvd(range_candles)

        trades_taken = 0
        bo_up = bo_down = False
        i = 0

        while i < len(post_range) and trades_taken < max_trades:
            c = post_range[i]
            t = c.time.time()

            if not (sess.range_end <= t < sess.entry_end):
                i += 1
                continue

            up = c.close > range_high and not bo_up
            down = c.close < range_low and not bo_down

            if not up and not down:
                i += 1
                continue

            if up:
                direction = Direction.LONG
                entry = c.close
                sl = c.low
                risk = entry - sl
                bo_up = True
            else:
                direction = Direction.SHORT
                entry = c.close
                sl = c.high
                risk = sl - entry
                bo_down = True

            if risk <= 0:
                i += 1
                continue

            tp = entry + (risk * rr_target) if direction == Direction.LONG else entry - (risk * rr_target)

            # Order flow
            pre_bo = post_range[max(0, i - 3):i]
            if not pre_bo: pre_bo = range_candles[-3:]
            vol_ramp = compute_volume_ramp(pre_bo)

            cvd_aligned = (
                (direction == Direction.LONG and cvd_range > 0) or
                (direction == Direction.SHORT and cvd_range < 0)
            )

            result = simulate_trade(entry, sl, tp, direction, risk, post_range[i + 1:], sess)

            if result:
                exit_p, exit_r, exit_t = result
                rr = _calc_rr(direction, entry, exit_p, risk)
                trade = CryptoTradeResult(
                    symbol=symbol, session=sess_name, trade_date=sess_date,
                    direction=direction, entry_price=entry, stop_loss=sl,
                    take_profit=tp, exit_price=exit_p, exit_reason=exit_r,
                    rr_achieved=rr, entry_time=c.time, exit_time=exit_t,
                    range_high=range_high, range_low=range_low,
                    range_size=range_size, range_size_pct=range_size_pct,
                    cvd_range=cvd_range, cvd_aligned=cvd_aligned,
                    volume_ramp=vol_ramp,
                )
                all_trades.append(trade)
                trades_taken += 1

                if verbose:
                    print(
                        f"  {sess_date} {sess_name:6s} {symbol:10s} "
                        f"{direction.value:5s} entry={entry:>12.2f} "
                        f"exit={exit_p:>12.2f} reason={exit_r:6s} RR={rr:+.2f}"
                    )

                if exit_r == "stop":
                    if direction == Direction.LONG: bo_up = False
                    else: bo_down = False
                    while i < len(post_range) and post_range[i].time <= exit_t:
                        i += 1
                    continue
                else:
                    break

            i += 1

    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────


def _s(tr):
    n = len(tr)
    if not n: return 0, 0.0, 0.0
    return n, sum(1 for t in tr if t.rr_achieved > 0) / n, sum(t.rr_achieved for t in tr) / n

def _tr(tr): return sum(t.rr_achieved for t in tr)
def _pf(tr):
    gp = sum(t.rr_achieved for t in tr if t.rr_achieved > 0)
    gl = abs(sum(t.rr_achieved for t in tr if t.rr_achieved <= 0))
    return gp / gl if gl > 0 else float("inf")
def _dd(tr):
    if not tr: return 0.0
    p = e = d = 0.0
    for t in tr:
        e += t.rr_achieved
        if e > p: p = e
        dd = p - e
        if dd > d: d = dd
    return d


def report(trades, symbol):
    n, wr, avg = _s(trades)
    total = _tr(trades)
    pf = _pf(trades)
    dd = _dd(trades)
    stops = sum(1 for t in trades if t.exit_reason == "stop")
    targets = sum(1 for t in trades if t.exit_reason == "target")
    time_exits = sum(1 for t in trades if t.exit_reason == "time")

    print(f"\n{'=' * 80}")
    print(f"  {symbol} — ORB RESULTS")
    print(f"{'=' * 80}")
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

    # By session
    print(f"\n  {'Session':<10s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 48)
    for sess_name in ["ASIA", "EUROPE", "US"]:
        sub = [t for t in trades if t.session == sess_name]
        if not sub: continue
        sn, sw, sa = _s(sub)
        print(f"  {sess_name:<10s} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(sub):>+9.2f} {_pf(sub):>7.2f}")

    # CVD + ramp filters
    cvd_y = [t for t in trades if t.cvd_aligned]
    ramp_pos = [t for t in trades if t.volume_ramp > 0]
    cvd_ramp = [t for t in trades if t.cvd_aligned and t.volume_ramp > 0]

    print(f"\n  {'Filter':<20s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
    print("  " + "-" * 52)
    for l, sub in [("All", trades), ("CVD-aligned", cvd_y), ("Ramp > 0", ramp_pos), ("CVD + ramp > 0", cvd_ramp)]:
        sn, sw, sa = _s(sub)
        print(f"  {l:<20s} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(sub):>+9.2f} {_pf(sub):>7.2f}")

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
    if months: print(f"  Positive: {pos_m}/{len(months)} ({pos_m/len(months):.0%})")


def report_rr_sweep(symbol, groups, sessions, max_trades):
    print(f"\n{'=' * 80}")
    print(f"  {symbol} — RR TARGET SWEEP")
    print(f"{'=' * 80}")
    print(f"  {'RR':<6s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s} {'MaxDD':>7s}")
    print("  " + "-" * 50)
    for rr in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        tr = collect_crypto_orb_trades(symbol, groups, sessions, rr, max_trades)
        sn, sw, sa = _s(tr)
        print(f"  {rr:<6.1f} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(tr):>+9.2f} {_pf(tr):>7.2f} {_dd(tr):>6.1f}R")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Crypto ORB Backtest")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--rr-target", type=float, default=5.0)
    p.add_argument("--months", type=int, default=12)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    symbols = [s.strip() for s in args.symbols.split(",")]

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.months * 30)

    print(f"\n{'=' * 80}")
    print(f"  CRYPTO ORB BACKTEST")
    print(f"{'=' * 80}")
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Period:     {start_dt.date()} to {end_dt.date()}")
    print(f"  RR Target:  {args.rr_target}")
    print(f"  Sessions:   ASIA (00:00-08:00 UTC), EUROPE (07:00-13:00), US (13:00-21:00)")
    print()

    all_trades_combined = []

    for symbol in symbols:
        candles = fetch_binance_klines(
            symbol, "5m", start_dt, end_dt,
            use_cache=not args.no_cache,
        )
        if not candles:
            print(f"  WARNING: No data for {symbol}")
            continue

        print(f"  {symbol}: {len(candles)} candles ({candles[0].time.date()} to {candles[-1].time.date()})")

        groups = group_by_session_day(candles, CRYPTO_SESSIONS)
        print(f"  {symbol}: {len(groups)} session-days")

        trades = collect_crypto_orb_trades(
            symbol, groups, CRYPTO_SESSIONS, args.rr_target,
            max_trades=2, verbose=args.verbose,
        )
        print(f"  {symbol}: {len(trades)} trades")

        if trades:
            report(trades, symbol)
            report_rr_sweep(symbol, groups, CRYPTO_SESSIONS, 2)

        all_trades_combined.extend(trades)

    # Combined summary
    if all_trades_combined:
        all_trades_combined.sort(key=lambda t: t.entry_time)
        print(f"\n{'=' * 80}")
        print(f"  COMBINED SUMMARY — ALL CRYPTO")
        print(f"{'=' * 80}")
        n, wr, avg = _s(all_trades_combined)
        total = _tr(all_trades_combined)
        pf = _pf(all_trades_combined)
        dd = _dd(all_trades_combined)
        print(f"  Trades: {n}  Win: {wr:.1%}  Avg RR: {avg:+.2f}  Total R: {total:+.1f}  PF: {pf:.2f}  MaxDD: {dd:.1f}R")

        # By symbol
        print(f"\n  {'Symbol':<12s} {'N':>5s} {'Win%':>7s} {'Avg RR':>8s} {'Total R':>9s} {'PF':>7s}")
        print("  " + "-" * 48)
        for sym in symbols:
            sub = [t for t in all_trades_combined if t.symbol == sym]
            if not sub: continue
            sn, sw, sa = _s(sub)
            print(f"  {sym:<12s} {sn:>5d} {sw:>6.1%} {sa:>+8.2f} {_tr(sub):>+9.2f} {_pf(sub):>7.2f}")

        # CVD + ramp filter on combined
        cvd_ramp = [t for t in all_trades_combined if t.cvd_aligned and t.volume_ramp > 0]
        sn, sw, sa = _s(cvd_ramp)
        print(f"\n  CVD+ramp>0 (combined): {sn} trades, {sw:.1%} win, {_tr(cvd_ramp):+.1f}R, {_pf(cvd_ramp):.2f} PF")

    print()


if __name__ == "__main__":
    main()
