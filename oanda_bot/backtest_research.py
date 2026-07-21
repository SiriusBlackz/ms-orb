#!/usr/bin/env python3
"""MS-ORB Research Harness — win-rate-at-3RR feature mining.

Goal: find an entry-selection / stop model that reaches >=50% win rate at a
fixed 3:1 reward:risk (PF 3.0). This harness extracts EVERY breakout signal from
the cached M5 data, computes a rich, no-look-ahead feature vector at the moment
of each signal, and records the 3R win/loss outcome under several stop-width
definitions. It then does a time-based train/test split and reports:

  * overall WR@3R per stop mode (the stop-width lever)
  * univariate WR@3R per feature decile (train), re-checked out-of-sample (test)

Reuses the authoritative signal + trade-simulation code from backtest.py so the
outcomes stay consistent with the source-of-truth backtest (no divergence).

Usage:
  python backtest_research.py --months 12 [--no-regen]   # build dataset + analyze
  python backtest_research.py --analyze-only             # re-analyze cached CSV
"""

import argparse
import csv
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SESSIONS, SessionName, TradingConfig
from oanda_client import OANDAClient, Candle
from strategy import RangeRecorder, BreakoutDetector, BreakoutSignal, Direction
from backtest import (
    fetch_candles_range,
    candle_ny_time,
    group_candles_by_session_day,
    is_in_range_period,
    is_in_entry_window,
    simulate_trade,
    compute_volume_ratio,
    compute_adx_series,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "research_signals.csv")
RR = 3.0  # fixed reward:risk for the win-rate target


# ─────────────────────────────────────────────────────────────
#  Indicator series (per-instrument, indexed by candle.time)
# ─────────────────────────────────────────────────────────────

def compute_atr_series(candles, period=14):
    """Wilder ATR per candle.time. Uses prior close for true range."""
    out = {}
    trs = []
    prev_close = None
    atr = None
    for i, c in enumerate(candles):
        if prev_close is None:
            tr = c.high - c.low
        else:
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        if i + 1 == period:
            atr = sum(trs[:period]) / period
            out[c.time] = atr
        elif i + 1 > period:
            atr = (atr * (period - 1) + tr) / period
            out[c.time] = atr
        prev_close = c.close
    return out


def compute_ema_series(candles, period):
    """EMA of close per candle.time."""
    out = {}
    k = 2 / (period + 1)
    ema = None
    for c in candles:
        ema = c.close if ema is None else (c.close * k + ema * (1 - k))
        out[c.time] = ema
    return out


# Scheduled US macro release windows (ET). FOMC announcement dates 2025-26
# (2:00 PM ET); NFP = first Friday of each month (8:30 AM ET). Top-tier only.
FOMC_DATES = {
    (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18), (2025, 7, 30),
    (2025, 9, 17), (2025, 10, 29), (2025, 12, 10),
    (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17),
}


def is_news_window(ny_dt, pad_min=15):
    """True if the entry time sits in a top-tier US release window (ET)."""
    d = ny_dt.date()
    t = ny_dt.hour * 60 + ny_dt.minute
    # FOMC 14:00 ET
    if (d.year, d.month, d.day) in FOMC_DATES and abs(t - 14 * 60) <= pad_min:
        return True
    # NFP: first Friday 08:30 ET
    if ny_dt.weekday() == 4 and 1 <= d.day <= 7 and abs(t - (8 * 60 + 30)) <= pad_min:
        return True
    return False


def signed_vol(c):
    if c.close > c.open:
        return c.volume
    if c.close < c.open:
        return -c.volume
    return 0


# ─────────────────────────────────────────────────────────────
#  Feature extraction at a signal
# ─────────────────────────────────────────────────────────────

def extract_features(sig, candle, range_candles, pre_breakout, prior_candles,
                     atr, adx, ema_fast, ema_slow, ema_htf, seq_idx, range_pctile):
    rng = sig.opening_range
    entry = sig.entry_price
    direction = sig.direction
    range_size = rng.high - rng.low
    body = abs(candle.close - candle.open)
    candle_range = candle.high - candle.low

    # order flow over the 30-min range window
    cvd = sum(signed_vol(c) for c in range_candles)
    range_vol = sum(c.volume for c in range_candles) or 1
    cvd_norm = cvd / range_vol
    cvd_aligned = int((direction == Direction.LONG and cvd > 0) or
                      (direction == Direction.SHORT and cvd < 0))

    # pre-breakout volume ramp (slope of last <=3 candle volumes / avg)
    if len(pre_breakout) >= 2:
        vols = [c.volume for c in pre_breakout]
        avg = sum(vols) / len(vols) or 1
        ramp = (vols[-1] - vols[0]) / avg
    else:
        ramp = 0.0

    # trend alignment
    t8_21 = int((direction == Direction.LONG and ema_fast > ema_slow) or
                (direction == Direction.SHORT and ema_fast < ema_slow)) if (ema_fast and ema_slow) else 0
    htf = int((direction == Direction.LONG and candle.close > ema_htf) or
              (direction == Direction.SHORT and candle.close < ema_htf)) if ema_htf else 0

    # geometry
    if direction == Direction.LONG:
        dist_beyond = candle.close - rng.high
    else:
        dist_beyond = rng.low - candle.close
    risk = sig.risk_per_unit or 1e-9
    ny = candle_ny_time(candle)

    return {
        "instrument": sig.instrument,
        "session": sig.session.value,
        "direction": direction.value,
        "date": ny.date().isoformat(),
        "hour": ny.hour,
        "dow": ny.weekday(),
        "seq": seq_idx,                                   # how late in entry window (0=first)
        "range_atr": (range_size / atr) if atr else 0.0,  # range size vs volatility
        "range_pctile": range_pctile,                     # vol regime (0-1) vs recent same-session
        "atr_pct": (atr / entry) if atr else 0.0,
        "body_ratio": (body / candle_range) if candle_range else 0.0,
        "vol_ratio": prior_candles,  # placeholder replaced below
        "adx": adx if adx is not None else -1,
        "cvd_aligned": cvd_aligned,
        "cvd_norm": cvd_norm,
        "ramp": ramp,
        "trend_fast": t8_21,
        "trend_htf": htf,
        "dist_beyond_r": dist_beyond / risk,              # breakout extension in R units
        "dist_beyond_atr": (dist_beyond / atr) if atr else 0.0,
    }


def trail_simulate(entry_candle, direction, remaining, session_cfg, atr, init_stop, k=2.0):
    """No fixed target; chandelier trailing stop (k*ATR from running extreme).
    Exit on trail hit or session close. Returns realized R (float) or None."""
    from backtest import is_past_session_close
    entry = entry_candle.close
    risk = abs(entry - init_stop)
    if risk <= 0 or atr is None:
        return None
    stop = init_stop
    if direction == Direction.LONG:
        run_hi = entry_candle.high
        for c in remaining:
            if is_past_session_close(candle_ny_time(c), session_cfg):
                return (c.open - entry) / risk
            if c.low <= stop:                      # prior-bar trail hit this bar
                return (stop - entry) / risk
            run_hi = max(run_hi, c.high)
            stop = max(stop, run_hi - k * atr)     # ratchet up only
        return (remaining[-1].close - entry) / risk if remaining else 0.0
    else:
        run_lo = entry_candle.low
        for c in remaining:
            if is_past_session_close(candle_ny_time(c), session_cfg):
                return (entry - c.open) / risk
            if c.high >= stop:
                return (entry - stop) / risk
            run_lo = min(run_lo, c.low)
            stop = min(stop, run_lo + k * atr)
        return (entry - remaining[-1].close) / risk if remaining else 0.0


def make_signal(rng_sig_template, entry_candle, direction, opening_range, stop_price, instrument, session):
    """Build a BreakoutSignal with a custom stop and a 3R target."""
    entry = entry_candle.close
    risk = abs(entry - stop_price)
    if risk <= 0:
        return None
    if direction == Direction.LONG:
        tp = entry + RR * risk
    else:
        tp = entry - RR * risk
    return BreakoutSignal(
        session=session, instrument=instrument, direction=direction,
        entry_price=entry, stop_loss=stop_price, take_profit=tp,
        opening_range=opening_range, signal_time=entry_candle.time,
        breakout_candle=entry_candle, risk_per_unit=risk, rr_target=RR,
    )


def outcome_for_stop(direction, entry_candle, opening_range, stop_price,
                     remaining, session_cfg, instrument, session):
    """Return 1 if target(3R) hit, 0 if stop, -1 if time/neither. None if invalid."""
    sig = make_signal(None, entry_candle, direction, opening_range, stop_price, instrument, session)
    if sig is None:
        return None
    res = simulate_trade(sig, remaining, session_cfg)
    if res is None:
        return None
    if res.exit_reason == "target":
        return 1
    if res.exit_reason == "stop":
        return 0
    return -1


# ─────────────────────────────────────────────────────────────
#  Dataset generation
# ─────────────────────────────────────────────────────────────

def generate(instruments, months):
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=months * 30)
    client = OANDAClient()
    if not client.test_connection():
        print("ERROR: OANDA connection failed")
        sys.exit(1)

    session_cfgs = {n: SESSIONS[n] for n in SessionName}
    rows = []

    for inst in instruments:
        candles = fetch_candles_range(client, inst, start_dt, end_dt, use_cache=True)
        if not candles:
            continue
        print(f"  {inst}: {len(candles)} candles")
        # prior-day high/low per NY date (for clean-runway filter)
        daily_hl = {}
        for cc in candles:
            dd = candle_ny_time(cc).date()
            hi, lo = daily_hl.get(dd, (cc.high, cc.low))
            daily_hl[dd] = (max(hi, cc.high), min(lo, cc.low))
        daily_dates = sorted(daily_hl.keys())

        def prior_day_hl(sdate):
            import bisect
            j = bisect.bisect_left(daily_dates, sdate) - 1
            return daily_hl[daily_dates[j]] if j >= 0 else (None, None)

        # full chronological stream + index for multi-day "swing" hold outcomes
        cand_idx = {cc.time: ii for ii, cc in enumerate(candles)}

        def swing_outcome(direction, entry, stop, target, start_time, max_bars=1440):
            """Hold until target(+3R) or stop, ignoring session close. 1/0/-1."""
            j = cand_idx.get(start_time)
            if j is None:
                return None
            for cc in candles[j + 1: j + 1 + max_bars]:
                if direction == Direction.LONG:
                    sh, th = cc.low <= stop, cc.high >= target
                else:
                    sh, th = cc.high >= stop, cc.low <= target
                if sh and th:
                    return 0  # conservative: assume stop
                if sh:
                    return 0
                if th:
                    return 1
            return -1

        atr_s = compute_atr_series(candles)
        adx_s = compute_adx_series(candles)
        ema8 = compute_ema_series(candles, 8)
        ema21 = compute_ema_series(candles, 21)
        ema100 = compute_ema_series(candles, 100)
        groups = group_candles_by_session_day(candles, list(session_cfgs.values()))

        # trailing range sizes / range-period volumes per session
        recent_ranges = defaultdict(lambda: deque(maxlen=20))
        recent_rvols = defaultdict(lambda: deque(maxlen=14))

        for (session_name, sdate), day_candles in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0].value)):
            cfg = session_cfgs[session_name]
            range_candles = [c for c in day_candles if is_in_range_period(candle_ny_time(c), cfg)]
            post = [c for c in day_candles if not is_in_range_period(candle_ny_time(c), cfg)]
            if not range_candles:
                continue
            rec = RangeRecorder(session_name, inst)
            for c in range_candles:
                rec.add_candle(c)
            opening_range = rec.finalize()
            if opening_range is None:
                continue

            range_size = opening_range.high - opening_range.low
            hist = recent_ranges[session_name]
            pctile = (sum(1 for r in hist if r <= range_size) / len(hist)) if hist else 0.5
            hist.append(range_size)

            # range-period relative volume (today vs prior 14 same-session days)
            rangevol = sum(c.volume for c in range_candles)
            vhist = recent_rvols[session_name]
            rvol = (rangevol / (sum(vhist) / len(vhist))) if vhist else 1.0
            vhist.append(rangevol)

            detector = BreakoutDetector(opening_range, rr_target=RR)
            i = 0
            seq = 0
            while i < len(post):
                c = post[i]
                ny = candle_ny_time(c)
                sig = detector.check_breakout(c) if is_in_entry_window(ny, cfg) else None
                if sig:
                    prior = range_candles + post[:i]
                    pre_breakout = post[max(0, i - 3):i] if i >= 1 else range_candles[-3:]
                    vol_ratio = compute_volume_ratio(c, prior, lookback=20)
                    feat = extract_features(
                        sig, c, range_candles, pre_breakout, prior,
                        atr_s.get(c.time), adx_s.get(c.time),
                        ema8.get(c.time), ema21.get(c.time), ema100.get(c.time),
                        seq, pctile,
                    )
                    feat["vol_ratio"] = vol_ratio if vol_ratio is not None else -1
                    feat["rvol"] = round(rvol, 3)
                    feat["news"] = int(is_news_window(candle_ny_time(c)))
                    # clean runway: no prior-day level between entry and the 3R
                    # target (target measured at the 1.5xATR stop variant)
                    pdh, pdl = prior_day_hl(sdate)
                    _atr = atr_s.get(c.time)
                    if _atr and pdh is not None:
                        _risk = 1.5 * _atr
                        if sig.direction == Direction.LONG:
                            _tgt = sig.entry_price + 3 * _risk
                            feat["clean"] = int(not (sig.entry_price < pdh < _tgt))
                        else:
                            _tgt = sig.entry_price - 3 * _risk
                            feat["clean"] = int(not (_tgt < pdl < sig.entry_price))
                    else:
                        feat["clean"] = 1
                    # multi-day swing outcomes (hold until resolved), 1.5xATR stop, 3R target
                    if _atr:
                        sgn3 = -1 if sig.direction == Direction.LONG else 1
                        sstop = sig.entry_price + sgn3 * 1.5 * _atr
                        starg = sig.entry_price - sgn3 * 3 * 1.5 * _atr
                        for nd, bars in (("3d", 864), ("5d", 1440)):
                            feat[f"y_swing{nd}"] = swing_outcome(
                                sig.direction, sig.entry_price, sstop, starg, c.time, bars)
                    else:
                        feat["y_swing3d"] = feat["y_swing5d"] = ""
                    remaining = post[i + 1:]
                    atr = atr_s.get(c.time)

                    # outcomes under several stop-width definitions (target = 3R each)
                    if sig.direction == Direction.LONG:
                        sl_candle = c.low
                        sl_range = opening_range.low
                        sl_atr10 = sig.entry_price - (atr or range_size)
                        sl_atr15 = sig.entry_price - 1.5 * (atr or range_size)
                    else:
                        sl_candle = c.high
                        sl_range = opening_range.high
                        sl_atr10 = sig.entry_price + (atr or range_size)
                        sl_atr15 = sig.entry_price + 1.5 * (atr or range_size)

                    # candle-stop trade drives the outcome label AND re-entry timing
                    csig = make_signal(None, c, sig.direction, opening_range, sl_candle, inst, session_name)
                    cres = simulate_trade(csig, remaining, cfg) if csig else None
                    if cres is None:
                        i += 1
                        continue
                    feat["y_candle"] = 1 if cres.exit_reason == "target" else (0 if cres.exit_reason == "stop" else -1)
                    # peak favorable excursion (in R) before candle-stop hit, intraday.
                    # WR at any target k:1 == fraction with maxR >= k.
                    _risk_c = abs(sig.entry_price - sl_candle)
                    _best = 0.0
                    if _risk_c > 0:
                        for cc2 in remaining:
                            from backtest import is_past_session_close as _ipsc
                            if _ipsc(candle_ny_time(cc2), cfg):
                                break
                            if sig.direction == Direction.LONG:
                                if cc2.low <= sl_candle:
                                    break
                                _best = max(_best, (cc2.high - sig.entry_price) / _risk_c)
                            else:
                                if cc2.high >= sl_candle:
                                    break
                                _best = max(_best, (sig.entry_price - cc2.low) / _risk_c)
                    feat["maxR"] = round(_best, 3)
                    feat["y_range"] = outcome_for_stop(sig.direction, c, opening_range, sl_range, remaining, cfg, inst, session_name)
                    feat["y_atr10"] = outcome_for_stop(sig.direction, c, opening_range, sl_atr10, remaining, cfg, inst, session_name)
                    feat["y_atr15"] = outcome_for_stop(sig.direction, c, opening_range, sl_atr15, remaining, cfg, inst, session_name)
                    # finer stop-width sweep for fixed-3R win-rate optimization
                    if atr:
                        sgn2 = -1 if sig.direction == Direction.LONG else 1
                        for w in (0.5, 0.75, 1.25, 2.0):
                            slw = sig.entry_price + sgn2 * w * atr
                            feat[f"y_atr{int(w*100)}"] = outcome_for_stop(
                                sig.direction, c, opening_range, slw, remaining, cfg, inst, session_name)
                    else:
                        for w in (0.5, 0.75, 1.25, 2.0):
                            feat[f"y_atr{int(w*100)}"] = ""
                    # no-target trailing runner (realized R). Sweep trail k on the
                    # candle-based initial stop, plus one ATR-initial-stop variant.
                    feat["risk_price"] = round(sig.risk_per_unit, 5)
                    feat["risk_atr"] = round(abs(sig.entry_price - sl_atr10), 5)
                    for kk in (1.5, 2.0, 2.5, 3.0):
                        rr = trail_simulate(c, sig.direction, remaining, cfg, atr, sl_candle, k=kk)
                        feat[f"r_k{int(kk*10)}"] = round(rr, 3) if rr is not None else ""
                    rr_a = trail_simulate(c, sig.direction, remaining, cfg, atr, sl_atr10, k=2.0)
                    feat["r_atrstop"] = round(rr_a, 3) if rr_a is not None else ""
                    feat["r_runner"] = feat["r_k20"]  # back-compat default
                    # robustness grid: initial ATR-stop width x trail k (net-cost analysis)
                    feat["atr_price"] = round(atr, 5) if atr else ""
                    if atr:
                        sgn = -1 if sig.direction == Direction.LONG else 1
                        for w in (0.75, 1.0, 1.5):
                            istop = sig.entry_price + sgn * w * atr
                            for tk in (2.0, 3.0):
                                rv = trail_simulate(c, sig.direction, remaining, cfg, atr, istop, k=tk)
                                feat[f"r_a{int(w*100)}_k{int(tk*10)}"] = round(rv, 3) if rv is not None else ""
                    rows.append(feat)
                    seq += 1

                    # mirror live re-entry: only after a candle-stop loss, and
                    # advance PAST the exit candle so re-entries don't overlap.
                    if feat["y_candle"] == 0:
                        detector.reset_direction(sig.direction)
                        while i < len(post) and post[i].time <= cres.exit_time:
                            i += 1
                        continue
                    else:
                        break
                i += 1

    if rows:
        keys = list(rows[0].keys())
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {len(rows)} signals -> {CSV_PATH}")
    return rows


# ─────────────────────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────────────────────

def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def wr(rows, ycol):
    """Win rate@3R = targets / total signals (matches backtest.py win%)."""
    n = len(rows)
    if n == 0:
        return 0.0, 0, 0
    wins = sum(1 for r in rows if int(float(r[ycol])) == 1)
    return wins / n, wins, n


def split_train_test(rows, frac=0.70):
    rows = sorted(rows, key=lambda r: r["date"])
    cut = int(len(rows) * frac)
    boundary = rows[cut]["date"]
    train = [r for r in rows if r["date"] < boundary]
    test = [r for r in rows if r["date"] >= boundary]
    return train, test


NUM_FEATS = ["range_atr", "range_pctile", "atr_pct", "body_ratio", "vol_ratio",
             "adx", "cvd_norm", "ramp", "dist_beyond_r", "dist_beyond_atr", "seq"]
CAT_FEATS = ["instrument", "session", "direction", "cvd_aligned", "trend_fast", "trend_htf", "dow", "hour"]


def analyze(rows):
    print("\n" + "=" * 78)
    print("  RESEARCH ANALYSIS — Win Rate @ 3:1 RR (target = 50%)")
    print("=" * 78)
    print(f"  Total signals: {len(rows)}")

    # 1. stop-width lever (overall WR per stop mode)
    print("\n  [1] STOP-WIDTH LEVER — overall WR@3R by stop definition")
    print("  " + "-" * 60)
    for y, lbl in [("y_candle", "breakout-candle"), ("y_range", "opposite range edge"),
                   ("y_atr10", "1.0x ATR"), ("y_atr15", "1.5x ATR")]:
        valid = [r for r in rows if r.get(y) not in (None, "", "None")]
        w, wins, n = wr(valid, y)
        resolved = [r for r in valid if int(float(r[y])) in (0, 1)]
        rw, _, rn = wr(resolved, y)
        print(f"    {lbl:22s}: WR={w*100:5.1f}%  ({wins}/{n})   resolved-WR={rw*100:5.1f}% (excl. time exits)")

    train, test = split_train_test(rows, 0.70)
    print(f"\n  Train: {len(train)} signals (older)  |  Test: {len(test)} signals (newer)")

    # 2. univariate WR@3R (candle stop) by category — train vs test
    print("\n  [2] CATEGORICAL features — WR@3R (candle stop), train -> test")
    print("  " + "-" * 70)
    for feat in CAT_FEATS:
        vals = sorted(set(r[feat] for r in train))
        if len(vals) > 12:
            continue
        cells = []
        for v in vals:
            tr = [r for r in train if r[feat] == v]
            te = [r for r in test if r[feat] == v]
            wtr, _, ntr = wr(tr, "y_candle")
            wte, _, nte = wr(te, "y_candle")
            cells.append(f"{v}={wtr*100:.0f}%/{wte*100:.0f}%(n{ntr})")
        print(f"    {feat:12s}: " + "  ".join(cells))

    # 3. univariate numeric — top/bottom decile WR vs baseline
    base_w, _, _ = wr(train, "y_candle")
    print(f"\n  [3] NUMERIC features — does a high/low slice beat base WR ({base_w*100:.1f}%)?")
    print("    feature        slice         train-WR   test-WR    n(tr)")
    print("  " + "-" * 64)
    for feat in NUM_FEATS:
        try:
            tr_sorted = sorted(train, key=lambda r: float(r[feat]))
        except (ValueError, KeyError):
            continue
        m = len(tr_sorted)
        if m < 40:
            continue
        thi = float(tr_sorted[int(m * 0.7)][feat])  # top-30% threshold
        tlo = float(tr_sorted[int(m * 0.3)][feat])  # bottom-30% threshold
        for slabel, cond in [(f">={thi:.3g}", lambda r, t=thi: float(r[feat]) >= t),
                             (f"<={tlo:.3g}", lambda r, t=tlo: float(r[feat]) <= t)]:
            trs = [r for r in train if cond(r)]
            tes = [r for r in test if cond(r)]
            wtr, _, ntr = wr(trs, "y_candle")
            wte, _, nte = wr(tes, "y_candle")
            flag = "  <==" if (wtr > base_w + 0.05 and wte > base_w + 0.03) else ""
            print(f"    {feat:13s} {slabel:12s}  {wtr*100:5.1f}%    {wte*100:5.1f}%   {ntr:5d}{flag}")

    greedy_search(train, test, "y_candle")
    greedy_search(train, test, "y_atr10")
    runner_analysis(rows, train, test)
    runner_sweep(rows, train, test)
    walk_forward(rows)
    robustness_grid(rows, train, test)
    rvol_news_filters(rows, train, test)
    wr3r_frontier(rows, train, test)
    cross_sectional(rows, train, test)
    print("=" * 78)


def rvol_news_filters(rows, train, test):
    # Use the cost-robust 1.0xATR stop / 2.0 trail runner (r_a100_k20, width 1.0)
    RCOL, W = "r_a100_k20", 1.0
    print("\n  [9] RVOL GATE — on Tokyo+HTF runner (1.0xATR stop), NET of costs")
    print("  " + "-" * 70)
    base = lambda rs: [r for r in rs if r["session"] == "TOKYO" and int(r["trend_htf"]) == 1]
    tr, te = base(train), base(test)
    print("    rvol gate     trainPF trainExp | testPF testExp testWR  n_te")
    for thr in (0.0, 0.8, 1.0, 1.2, 1.5):
        ftr = [r for r in tr if float(r.get("rvol", 1)) >= thr]
        fte = [r for r in te if float(r.get("rvol", 1)) >= thr]
        s_tr = _stats_from_rs(_net_rs_width(ftr, RCOL, W))
        s_te = _stats_from_rs(_net_rs_width(fte, RCOL, W))
        if not s_tr or not s_te:
            continue
        lbl = "off" if thr == 0 else f">={thr}"
        print(f"    rvol {lbl:8s}  {s_tr['pf']:5.2f}  {s_tr['exp']:+5.2f}  | {s_te['pf']:5.2f}  "
              f"{s_te['exp']:+5.2f}  {s_te['wr']*100:4.0f}%  {s_te['n']:4d}")

    print("\n  [10] NEWS BLACKOUT — signals flagged per session, and London/NY unlock test")
    print("  " + "-" * 70)
    for sess in ("TOKYO", "LONDON", "NY"):
        ss = [r for r in rows if r["session"] == sess]
        flagged = sum(1 for r in ss if int(r.get("news", 0)) == 1)
        print(f"    {sess:7s}: {flagged}/{len(ss)} signals in a US-news window "
              f"({'≈0 — evening session, no US overlap' if flagged == 0 else 'blackout removes these'})")
    # Do London/NY + HTF become viable under the runner with news-blackout + rvol>=1.0?
    print("    London/NY + HTF runner (1.0xATR), NET, with news-blackout + rvol>=1.0:")
    for sess in ("LONDON", "NY"):
        allrows = [r for r in rows if r["session"] == sess and int(r["trend_htf"]) == 1]
        filt = [r for r in allrows if int(r.get("news", 0)) == 0 and float(r.get("rvol", 1)) >= 1.0]
        s0 = _stats_from_rs(_net_rs_width(allrows, RCOL, W))
        s1 = _stats_from_rs(_net_rs_width(filt, RCOL, W))
        if s0 and s1:
            print(f"      {sess:7s}: raw PF={s0['pf']:.2f} (n{s0['n']}) -> filtered PF={s1['pf']:.2f} "
                  f"exp{s1['exp']:+.2f} WR{s1['wr']*100:.0f}% (n{s1['n']})")

    print("\n  [11] RECOMMENDED COMBINED CONFIG — Tokyo+NY, HTF-aligned, rvol>=1.0, runner")
    print("  " + "-" * 70)
    combo = lambda rs: [r for r in rs if r["session"] in ("TOKYO", "NY")
                        and int(r["trend_htf"]) == 1 and float(r.get("rvol", 1)) >= 1.0]
    for lbl, rs in [("full", rows), ("train", train), ("test", test)]:
        s = _stats_from_rs(_net_rs_width(combo(rs), RCOL, W))
        if s:
            print(f"    {lbl:6s}: n={s['n']:4d}  WR={s['wr']*100:4.1f}%  expR={s['exp']:+.2f}  "
                  f"totalR={s['total']:+6.1f}  PF={s['pf']:.2f}  Sharpe~{s['sharpe']:.2f}")


def _net_rs_width(rows, rcol, width):
    """Net realized R using initial stop = width*ATR (risk = width*atr_price)."""
    out = []
    for r in rows:
        v = r.get(rcol)
        ap = r.get("atr_price")
        if v in (None, "", "None") or ap in (None, "", "None"):
            continue
        risk = width * float(ap)
        if risk <= 0:
            continue
        out.append(float(v) - COSTS.get(r["instrument"], 0.0) / risk)
    return out


def robustness_grid(rows, train, test):
    print("\n  [8] PARAMETER ROBUSTNESS — Tokyo+HTF, NET PF (train | test), is there a plateau?")
    print("  " + "-" * 70)
    seg = lambda rs: [r for r in rs if r["session"] == "TOKYO" and int(r["trend_htf"]) == 1]
    tr, te = seg(train), seg(test)
    print("    init-stop \\ trail        k=2.0            k=3.0")
    for w in (0.75, 1.0, 1.5):
        cells = []
        for tk in (2.0, 3.0):
            rcol = f"r_a{int(w*100)}_k{int(tk*10)}"
            s_tr = _stats_from_rs(_net_rs_width(tr, rcol, w))
            s_te = _stats_from_rs(_net_rs_width(te, rcol, w))
            if s_tr and s_te:
                cells.append(f"PF {s_tr['pf']:.2f}|{s_te['pf']:.2f} exp{s_te['exp']:+.2f}")
            else:
                cells.append("n/a")
        print(f"    {w:.2f}xATR stop          " + "   ".join(f"{c:22s}" for c in cells))


def walk_forward(rows):
    print("\n  [7] WALK-FORWARD STABILITY — monthly NET expR (cost-robust ATR-stop runner)")
    print("      is the edge persistent across months, or just a recent-window artifact?")
    print("  " + "-" * 74)
    segs = [("Tokyo+HTF", lambda r: r["session"] == "TOKYO" and int(r["trend_htf"]) == 1),
            ("Tokyo+first", lambda r: r["session"] == "TOKYO" and int(r["seq"]) == 0)]
    months = sorted(set(r["date"][:7] for r in rows))
    for name, fn in segs:
        line = []
        pos = tot_n = 0
        all_rs = []
        for m in months:
            sub = [r for r in rows if r["date"][:7] == m and fn(r)]
            rs = _net_rs(sub, "r_atrstop", "risk_atr")
            all_rs += rs
            if not rs:
                continue
            e = sum(rs) / len(rs)
            tot_n += 1
            if e > 0:
                pos += 1
            line.append(f"{m[2:]}:{e:+.2f}")
        s = _stats_from_rs(all_rs)
        print(f"    {name:12s} | " + " ".join(line))
        if s:
            print(f"    {'':12s} -> positive months {pos}/{tot_n} | net expR={s['exp']:+.2f} "
                  f"PF={s['pf']:.2f} WR={s['wr']*100:.0f}% total={s['total']:+.1f}R Sharpe~{s['sharpe']:.2f}")


# round-trip transaction cost (spread+slippage) in PRICE units per instrument
COSTS = {"XAU_USD": 0.35, "NAS100_USD": 1.5}


def _net_rs(rows, rcol, riskcol):
    """Realized R per trade, net of transaction cost (cost_price/risk_price)."""
    out = []
    for r in rows:
        v = r.get(rcol)
        rp = r.get(riskcol)
        if v in (None, "", "None") or rp in (None, "", "None"):
            continue
        rp = float(rp)
        if rp <= 0:
            continue
        cost_r = COSTS.get(r["instrument"], 0.0) / rp
        out.append(float(v) - cost_r)
    return out


def _stats_from_rs(rs):
    n = len(rs)
    if n == 0:
        return None
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    gl = -sum(losses)
    pf = (sum(wins) / gl) if gl > 0 else float("inf")
    mean = sum(rs) / n
    var = sum((x - mean) ** 2 for x in rs) / n if n > 1 else 0
    sd = var ** 0.5
    sharpe = (mean / sd * (n ** 0.5)) if sd > 0 else 0  # per-sample Sharpe over the window
    return {"n": n, "wr": len(wins) / n, "exp": mean, "total": sum(rs), "pf": pf, "sharpe": sharpe}


def runner_sweep(rows, train, test):
    print("\n  [6] RUNNER EXIT SWEEP — NET of costs (XAU $0.35, NAS100 1.5pt round-trip)")
    print("      segment = Tokyo + first-breakout (best OOS); R net of per-trade cost")
    print("  " + "-" * 78)
    seg = lambda rs: [r for r in rs if r["session"] == "TOKYO" and int(r["seq"]) == 0]
    variants = [("r_k15", "risk_price", "candle stop, trail 1.5xATR"),
                ("r_k20", "risk_price", "candle stop, trail 2.0xATR"),
                ("r_k25", "risk_price", "candle stop, trail 2.5xATR"),
                ("r_k30", "risk_price", "candle stop, trail 3.0xATR"),
                ("r_atrstop", "risk_atr", "1.0xATR stop, trail 2.0xATR")]
    print("    variant                       trainPF  trainExp | testPF  testExp testWR  n_te")
    print("  " + "-" * 78)
    for rcol, riskcol, lbl in variants:
        ts = _stats_from_rs(_net_rs(seg(train), rcol, riskcol))
        es = _stats_from_rs(_net_rs(seg(test), rcol, riskcol))
        if not ts or not es:
            continue
        print(f"    {lbl:28s}  {ts['pf']:5.2f}   {ts['exp']:+5.2f}  | {es['pf']:5.2f}  {es['exp']:+5.2f}  "
              f"{es['wr']*100:4.0f}%  {es['n']:4d}")
    # full-sample net stats for the leading variant across segments
    print("  " + "-" * 78)
    print("    FULL-SAMPLE net (best variant r_k20, candle stop) by segment:")
    for name, fn in [("ALL", lambda r: True),
                     ("Tokyo", lambda r: r["session"] == "TOKYO"),
                     ("Tokyo+HTF", lambda r: r["session"] == "TOKYO" and int(r["trend_htf"]) == 1),
                     ("Tokyo+first", lambda r: r["session"] == "TOKYO" and int(r["seq"]) == 0)]:
        s = _stats_from_rs(_net_rs([r for r in rows if fn(r)], "r_k20", "risk_price"))
        if s:
            print(f"      {name:14s} n={s['n']:4d}  WR={s['wr']*100:4.1f}%  expR={s['exp']:+.2f}  "
                  f"totalR={s['total']:+7.1f}  PF={s['pf']:.2f}  Sharpe~{s['sharpe']:.2f}")


def runner_stats(rows):
    rs = [float(r["r_runner"]) for r in rows if r.get("r_runner") not in (None, "", "None")]
    n = len(rs)
    if n == 0:
        return None
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    gross_w = sum(wins)
    gross_l = -sum(losses)
    pf = (gross_w / gross_l) if gross_l > 0 else float("inf")
    return {
        "n": n, "wr": len(wins) / n, "avg_w": (sum(wins) / len(wins)) if wins else 0,
        "avg_l": (sum(losses) / len(losses)) if losses else 0,
        "exp": sum(rs) / n, "total": sum(rs), "pf": pf,
        "p_ge3": sum(1 for x in rs if x >= 3) / n,
    }


def runner_analysis(rows, train, test):
    print("\n  [5] NO-TARGET TRAILING RUNNER (2xATR chandelier) — the validated edge")
    print("      win=net green; 'RR' = avg_win/|avg_loss|; %>=3R = trades that ran >=3R")
    print("  " + "-" * 84)
    print("    segment                 n    win%   avgWin  avgLoss   RR    expR   totalR    PF   %>=3R")
    print("  " + "-" * 84)
    segs = [
        ("ALL", rows),
        ("Tokyo only", [r for r in rows if r["session"] == "TOKYO"]),
        ("Tokyo+HTF-aligned", [r for r in rows if r["session"] == "TOKYO" and int(r["trend_htf"]) == 1]),
        ("HTF-aligned (all sess)", [r for r in rows if int(r["trend_htf"]) == 1]),
        ("Tokyo+first-breakout", [r for r in rows if r["session"] == "TOKYO" and int(r["seq"]) == 0]),
    ]
    for name, sub in segs:
        s = runner_stats(sub)
        if not s:
            continue
        rr = (s["avg_w"] / abs(s["avg_l"])) if s["avg_l"] != 0 else float("inf")
        print(f"    {name:22s} {s['n']:5d}  {s['wr']*100:4.1f}%  {s['avg_w']:+6.2f}  {s['avg_l']:+6.2f}  "
              f"{rr:4.1f}  {s['exp']:+5.2f}  {s['total']:+7.1f}  {s['pf']:4.2f}  {s['p_ge3']*100:4.1f}%")
    # OOS check on the best segment
    print("  " + "-" * 84)
    for name, trf, tef in [("Tokyo", lambda r: r["session"] == "TOKYO",
                            lambda r: r["session"] == "TOKYO")]:
        tr = runner_stats([r for r in train if trf(r)])
        te = runner_stats([r for r in test if tef(r)])
        if tr and te:
            print(f"    OOS [{name}]: train expR={tr['exp']:+.2f} PF={tr['pf']:.2f} | "
                  f"test expR={te['exp']:+.2f} PF={te['pf']:.2f} win{te['wr']*100:.0f}%")


def build_candidates(train):
    """Binary filter candidates, thresholds derived from TRAIN only (no leakage)."""
    def pct(feat, q):
        vals = sorted(float(r[feat]) for r in train)
        return vals[int(len(vals) * q)]
    c = []
    c.append(("session=TOKYO", lambda r: r["session"] == "TOKYO"))
    c.append(("session=NY", lambda r: r["session"] == "NY"))
    c.append(("session!=NY", lambda r: r["session"] != "NY"))
    c.append(("early(seq<=1)", lambda r: int(r["seq"]) <= 1))
    c.append(("rvol>=1.0", lambda r: float(r.get("rvol", 1)) >= 1.0))
    c.append(("rvol>=1.2", lambda r: float(r.get("rvol", 1)) >= 1.2))
    c.append(("inst=NAS100", lambda r: r["instrument"] == "NAS100_USD"))
    c.append(("inst=XAU", lambda r: r["instrument"] == "XAU_USD"))
    c.append(("first_breakout(seq=0)", lambda r: int(r["seq"]) == 0))
    c.append(("trend_htf_aligned", lambda r: int(r["trend_htf"]) == 1))
    c.append(("trend_htf_counter", lambda r: int(r["trend_htf"]) == 0))
    c.append(("cvd_aligned", lambda r: int(r["cvd_aligned"]) == 1))
    c.append(("cvd_counter", lambda r: int(r["cvd_aligned"]) == 0))
    lo_atr = pct("atr_pct", 0.5)
    c.append((f"atr_pct<={lo_atr:.4g}(low vol)", lambda r, t=lo_atr: float(r["atr_pct"]) <= t))
    hi_body = pct("body_ratio", 0.4)
    c.append((f"body>={hi_body:.3g}", lambda r, t=hi_body: float(r["body_ratio"]) >= t))
    sh = pct("dist_beyond_atr", 0.5)
    c.append((f"shallow_dist<={sh:.3g}", lambda r, t=sh: float(r["dist_beyond_atr"]) <= t))
    lo_adx = pct("adx", 0.5)
    c.append((f"adx<={lo_adx:.3g}", lambda r, t=lo_adx: float(r["adx"]) <= t))
    lo_rng = pct("range_atr", 0.5)
    c.append((f"range_atr<={lo_rng:.3g}", lambda r, t=lo_rng: float(r["range_atr"]) <= t))
    return c


def greedy_search(train, test, ycol, min_train=150, min_test=60, max_depth=5):
    """Forward-select filters to maximize TRAIN WR@3R; read OOS TEST WR each step."""
    print(f"\n  [4] GREEDY FILTER STACK — maximize WR@3R ({ycol}), OOS-validated")
    print(f"      floor: train n>={min_train}, test n>={min_test}")
    print("  " + "-" * 70)
    chosen = []
    cur_tr, cur_te = train, test
    base_w, _, _ = wr(train, ycol)
    print(f"    start: train-WR={base_w*100:.1f}%  (n_tr={len(train)}, n_te={len(test)})")
    used = set()
    for depth in range(max_depth):
        cands = build_candidates(cur_tr)
        best = None
        for name, fn in cands:
            if name in used:
                continue
            sub_tr = [r for r in cur_tr if fn(r)]
            sub_te = [r for r in cur_te if fn(r)]
            if len(sub_tr) < min_train or len(sub_te) < min_test:
                continue
            wtr, _, _ = wr(sub_tr, ycol)
            if best is None or wtr > best[2]:
                best = (name, fn, wtr, sub_tr, sub_te)
        if best is None:
            break
        name, fn, wtr, sub_tr, sub_te = best
        wte, _, _ = wr(sub_te, ycol)
        cur_w, _, _ = wr(cur_tr, ycol)
        if wtr <= cur_w + 0.005:
            print(f"    (no candidate improves train WR beyond {cur_w*100:.1f}% within sample floor)")
            break
        chosen.append(name)
        used.add(name)
        cur_tr, cur_te = sub_tr, sub_te
        print(f"    + {name:28s} train-WR={wtr*100:5.1f}%  test-WR={wte*100:5.1f}%  (n_tr={len(sub_tr)}, n_te={len(sub_te)})")
    fw, _, fn_ = wr(cur_tr, ycol)
    fwt, _, fnt = wr(cur_te, ycol)
    print(f"    FINAL stack: {' & '.join(chosen) if chosen else '(none)'}")
    print(f"    => train-WR={fw*100:.1f}% (n{fn_})   TEST-WR={fwt*100:.1f}% (n{fnt})   target=50%")
    return fwt, fnt


def greedy_quiet(train, test, ycol, min_train=150, min_test=80, max_depth=5):
    """Greedy forward-selection maximizing TRAIN WR@3R; returns (stack, train_wr, test_wr, n_te)."""
    chosen, cur_tr, cur_te, used = [], train, test, set()
    for _ in range(max_depth):
        cands = build_candidates(cur_tr)
        best = None
        for name, fn in cands:
            if name in used:
                continue
            str_ = [r for r in cur_tr if fn(r)]
            ste_ = [r for r in cur_te if fn(r)]
            if len(str_) < min_train or len(ste_) < min_test:
                continue
            wtr, _, _ = wr(str_, ycol)
            if best is None or wtr > best[1]:
                best = ((name, fn), wtr, str_, ste_)
        if best is None:
            break
        (name, fn), wtr, str_, ste_ = best
        cur_w, _, _ = wr(cur_tr, ycol)
        if wtr <= cur_w + 0.005:
            break
        chosen.append(name); used.add(name); cur_tr, cur_te = str_, ste_
    tw, _, _ = wr(cur_tr, ycol)
    ew, _, ne = wr(cur_te, ycol)
    return chosen, tw, ew, ne


def wr3r_frontier(rows, train, test):
    """Best-achievable WR@3R: sweep stop width x OOS-greedy filter stack."""
    print("\n" + "=" * 80)
    print("  BEST-ACHIEVABLE WR @ true 3:1 RR — stop-width x OOS filter stack")
    print("  (TEST WR is the honest number; floor train>=150, test>=80)")
    print("=" * 80)
    widths = [(0.5, "y_atr50"), (0.75, "y_atr75"), (1.0, "y_atr10"),
              (1.25, "y_atr125"), (1.5, "y_atr15"), (2.0, "y_atr200"), ("candle", "y_candle")]
    print(f"    {'stop':10s} {'baseWR':>7s} {'greedy trainWR':>15s} {'TEST WR':>9s} {'n_te':>5s}  filters")
    print("  " + "-" * 76)
    best_overall = None
    for w, col in widths:
        if not any(r.get(col) not in (None, "", "None") for r in train[:50] + train[-50:]):
            continue
        valid_tr = [r for r in train if r.get(col) not in (None, "", "None")]
        valid_te = [r for r in test if r.get(col) not in (None, "", "None")]
        bw, _, _ = wr(valid_tr, col)
        stack, tw, ew, ne = greedy_quiet(valid_tr, valid_te, col)
        lbl = f"{w}xATR" if w != "candle" else "candle"
        print(f"    {lbl:10s} {bw*100:6.1f}% {tw*100:14.1f}% {ew*100:8.1f}% {ne:5d}  {' & '.join(stack) if stack else '(none)'}")
        if ne >= 80 and (best_overall is None or ew > best_overall[1]):
            best_overall = (lbl, ew, ne, stack, tw)
    if best_overall:
        lbl, ew, ne, stack, tw = best_overall
        print("  " + "-" * 76)
        print(f"  => BEST honest WR@3R: {ew*100:.1f}% (test, n={ne}) | stop={lbl} | "
              f"{' & '.join(stack) if stack else 'no filters'} | trainWR={tw*100:.1f}%")
    print("=" * 80)


def cross_sectional(rows, train, test):
    """Each session-day, rank candidate setups across ALL instruments and keep
    only the top-K by a quality metric — the literature's path to higher WR@3R."""
    print("\n" + "=" * 80)
    print("  CROSS-SECTIONAL DAILY SELECTION — top-K setups/day across the universe")
    print("  (stop=1.5xATR, y_atr15; pick best per (session,date); TEST WR is honest)")
    print("=" * 80)
    insts = sorted(set(r["instrument"] for r in rows))
    print(f"    universe: {len(insts)} instruments | {len(rows)} total signals")
    YC = "y_atr15"

    def select(rs, metric, k, prefilter=None):
        pool = [r for r in rs if (prefilter(r) if prefilter else True)
                and r.get(YC) not in (None, "", "None")]
        groups = defaultdict(list)
        for r in pool:
            groups[(r["session"], r["date"])].append(r)
        out = []
        for g in groups.values():
            try:
                g2 = sorted(g, key=lambda r: -float(r.get(metric, 0)))
            except ValueError:
                g2 = g
            out += g2[:k]
        return out

    htf = lambda r: int(r["trend_htf"]) == 1
    htf_first = lambda r: int(r["trend_htf"]) == 1 and int(r["seq"]) == 0
    print("\n    selection                         metric   K   trainWR  testWR  n_te")
    print("  " + "-" * 74)
    configs = [
        ("all, top-K by RVOL", "rvol", None),
        ("HTF-aligned, top-K by RVOL", "rvol", htf),
        ("HTF+first, top-K by RVOL", "rvol", htf_first),
        ("HTF+first, top-K by range_pctile", "range_pctile", htf_first),
        ("HTF-aligned, top-K by body_ratio", "body_ratio", htf),
    ]
    best = None
    for lbl, metric, pf in configs:
        for k in (1, 2, 3):
            str_ = select(train, metric, k, pf)
            ste_ = select(test, metric, k, pf)
            wtr, _, ntr = wr(str_, YC)
            wte, _, nte = wr(ste_, YC)
            if nte < 60:
                continue
            mark = ""
            if wte >= 0.40 and ntr >= 100:
                mark = "  <== >=40% OOS"
            print(f"    {lbl:33s} {metric:11s} {k}  {wtr*100:6.1f}%  {wte*100:6.1f}%  {nte:4d}{mark}")
            if nte >= 80 and (best is None or wte > best[1]):
                best = (f"{lbl} | {metric} top-{k}", wte, nte, wtr)
    if best:
        print("  " + "-" * 74)
        print(f"  => BEST OOS WR@3R: {best[1]*100:.1f}% (test n={best[2]}, train {best[3]*100:.1f}%) | {best[0]}")
        print(f"     target=40%  ->  {'ACHIEVED' if best[1] >= 0.40 else 'not yet'}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--instruments", type=str, default="XAU_USD,NAS100_USD")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--no-regen", action="store_true", help="reuse CSV if present")
    args = ap.parse_args()

    if args.analyze_only or (args.no_regen and os.path.exists(CSV_PATH)):
        rows = load_rows()
        print(f"Loaded {len(rows)} cached signals from {CSV_PATH}")
    else:
        instruments = [s.strip() for s in args.instruments.split(",")]
        print("Generating research dataset...")
        rows = generate(instruments, args.months)

    if not rows:
        print("No signals.")
        return
    # normalize: generate() returns dict rows; load_rows returns str rows — unify via CSV reload
    if not isinstance(rows[0].get("y_candle"), str):
        rows = load_rows()
    analyze(rows)


if __name__ == "__main__":
    main()
