"""Walk-forward validation of the Config-A regime filters.

Question: the 2026-08-04 forensics and 2026-10-21 seasonality study found
filters that rescue Config A in-sample (cost-in-R < 0.07, D1 ADX, skip July,
Mon-Wed only, NAS-only). Do any of them survive when the rule is chosen using
ONLY data available before the period it is traded on?

Method
  * Trades: authoritative oanda_bot/backtest.py Config-A machinery (first
    breakout, HTF EMA100, 1.5xATR14 stop, 3:1, 1 trade/session), XAU_USD +
    NAS100_USD, Tokyo session, fixed price costs (XAU 0.35, NAS 1.5).
  * Ex-ante features per trade (all known at entry): cost_r, D1 ADX14 of the
    last COMPLETED daily bar, calendar month, Tokyo-calendar weekday, instrument.
  * A. Baseline by year.
  * B. Frozen rules exactly as discovered, split into discovery sample vs the
       data that arrived after discovery.
  * C. Anchored walk-forward: for each half-year test block, pick the best
       rule from a fixed grid on all PRIOR trades, trade it on the block.
  * D. Each filter on its own, walk-forward.
  * E. Leave-2026-out: choose on <=2025, test on 2026.
  * F. Permutation test: is the stitched out-of-sample result better than
       keeping the same number of random trades per block?

Usage: python3 research/walkforward_filters.py [--fetch-latest]
"""
import argparse
import glob
import itertools
import logging
import math
import pickle
import random
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import pytz

sys.path.insert(0, "/Users/derianj/projects/ms-orb/oanda_bot")
from backtest import (  # noqa: E402
    run_simulation, group_candles_by_session_day, compute_adx_series,
    compute_atr_series, compute_ema_series, fetch_candles_range,
)
from config import SESSIONS, SessionName  # noqa: E402

CACHE = "/Users/derianj/projects/ms-orb/oanda_bot/cache"
NY = pytz.timezone("America/New_York")
TOKYO = pytz.timezone("Asia/Tokyo")
COSTS = {"XAU_USD": 0.35, "NAS100_USD": 1.5}
INSTRUMENTS = ("XAU_USD", "NAS100_USD")
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FORENSICS_DATE = date(2026, 8, 4)     # cost_r / ADX / July discovered here
SEASONALITY_DATE = date(2026, 10, 21)  # weekday / NAS-only discovered here
MIN_TRAIN_TRADES = 100
random.seed(20261117)


# ---------------------------------------------------------------- data
def load_candles(inst: str, fetch_latest: bool) -> list:
    merged = {}
    for f in sorted(glob.glob(f"{CACHE}/{inst}_*_M5.pkl")):
        with open(f, "rb") as fh:
            for c in pickle.load(fh):
                merged[c.time] = c
    if fetch_latest:
        try:
            from oanda_client import OANDAClient
            client = OANDAClient()
            last = max(merged)
            new = fetch_candles_range(client, inst, last - timedelta(days=1),
                                      datetime.now(timezone.utc), use_cache=False)
            for c in new:
                if getattr(c, "complete", True):
                    merged[c.time] = c
            print(f"  fetched {len(new)} recent {inst} candles")
        except Exception as e:  # network / credentials — fall back to cache
            print(f"  (could not fetch latest {inst}: {e}; using cache only)")
    return [merged[t] for t in sorted(merged)]


def d1_adx(cs: list) -> tuple[dict, list]:
    """Wilder ADX14 on FX-convention daily bars (17:00->17:00 NY, keyed by close date)."""
    bars = {}
    for c in cs:
        key = (c.time.astimezone(NY) + timedelta(hours=7)).date()
        b = bars.get(key)
        if b is None:
            bars[key] = [c.high, c.low, c.close]
        else:
            b[0] = max(b[0], c.high); b[1] = min(b[1], c.low); b[2] = c.close
    ds = sorted(bars)
    h = [bars[d][0] for d in ds]; l = [bars[d][1] for d in ds]; cl = [bars[d][2] for d in ds]
    n = len(ds)
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - cl[i-1]), abs(l[i] - cl[i-1]))
                          for i in range(1, n)]
    pdm = [0.0] * n; ndm = [0.0] * n
    for i in range(1, n):
        up, dn = h[i] - h[i-1], l[i-1] - l[i]
        pdm[i] = up if up > dn and up > 0 else 0.0
        ndm[i] = dn if dn > up and dn > 0 else 0.0
    adx = {}
    if n > 28:
        s_tr, s_p, s_n = sum(tr[1:15]), sum(pdm[1:15]), sum(ndm[1:15])
        dxs, prev = [], None
        for i in range(15, n):
            s_tr = s_tr - s_tr / 14 + tr[i]
            s_p = s_p - s_p / 14 + pdm[i]
            s_n = s_n - s_n / 14 + ndm[i]
            pdi, ndi = 100 * s_p / s_tr, 100 * s_n / s_tr
            dx = 100 * abs(pdi - ndi) / (pdi + ndi) if pdi + ndi else 0.0
            dxs.append(dx)
            j = i - 15
            if j == 13:
                prev = sum(dxs) / 14
            elif j > 13:
                prev = (prev * 13 + dx) / 14
            if prev is not None:
                adx[ds[i]] = prev
    return adx, ds


def build_trades(fetch_latest: bool) -> list[dict]:
    rows = []
    for inst in INSTRUMENTS:
        cs = load_candles(inst, fetch_latest)
        print(f"  {inst}: {len(cs)} M5 candles {cs[0].time.date()} -> {cs[-1].time.date()}")
        adx5 = compute_adx_series(cs)
        atr = compute_atr_series(cs)
        ema = compute_ema_series(cs, 100)
        groups = group_candles_by_session_day(cs, [SESSIONS[SessionName.TOKYO]])
        m = run_simulation(
            grouped_candles={k: v for k, v in groups.items() if k[0] == SessionName.TOKYO},
            session_configs={SessionName.TOKYO: SESSIONS[SessionName.TOKYO]},
            instrument=inst, rr_target=3.0, vol_threshold=None, adx_threshold=None,
            adx_series=adx5, max_trades_per_session=1, stop_mode="atr",
            atr_stop_mult=1.5, first_breakout_only=True, htf_align=True,
            atr_series=atr, ema_htf_series=ema, cost_price=COSTS[inst],
        )
        adx_d1, d1_dates = d1_adx(cs)
        for t in m.trades:
            e_ny = t.entry_time.astimezone(NY)
            cur_key = (e_ny + timedelta(hours=7)).date()   # daily bar in progress
            prior = [d for d in d1_dates if d < cur_key and d in adx_d1]
            # d1_dates is sorted; take the last completed bar
            a = adx_d1[prior[-1]] if prior else None
            sd = t.entry_time.astimezone(TOKYO).date()
            rows.append(dict(
                inst=inst, t=t.entry_time, day=e_ny.date(), sd=sd,
                year=sd.year, month=sd.month, wd=sd.weekday(),
                r=t.rr_achieved, gross=t.rr_gross, cost_r=t.cost_r, adx=a,
            ))
    rows.sort(key=lambda r: r["t"])
    return rows


# ---------------------------------------------------------------- rules
COST_TH = (None, 0.05, 0.06, 0.07, 0.08, 0.10)
ADX_TH = (None, 20.0, 25.0)
JULY = (False, True)
DAYS = ("all", "MonWed")
INST = ("both", "NAS", "XAU")
GRID = list(itertools.product(COST_TH, ADX_TH, JULY, DAYS, INST))


def passes(r: dict, rule: tuple) -> bool:
    c, a, j, w, i = rule
    if c is not None and not (r["cost_r"] < c):
        return False
    if a is not None and (r["adx"] is None or r["adx"] < a):
        return False
    if j and r["month"] == 7:
        return False
    if w == "MonWed" and r["wd"] > 2:
        return False
    if i == "NAS" and r["inst"] != "NAS100_USD":
        return False
    if i == "XAU" and r["inst"] != "XAU_USD":
        return False
    return True


def label(rule: tuple) -> str:
    c, a, j, w, i = rule
    parts = []
    if c is not None: parts.append(f"cost<{c:.2f}")
    if a is not None: parts.append(f"ADX>={a:.0f}")
    if j: parts.append("noJul")
    if w != "all": parts.append("Mon-Wed")
    if i != "both": parts.append(f"{i}only")
    return " ".join(parts) or "no filter"


def st(rs: list[float]) -> dict:
    n = len(rs)
    if n == 0:
        return dict(n=0, net=0.0, exp=0.0, t=0.0, pf=0.0, dd=0.0, wr=0.0)
    net = sum(rs); mean = net / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in rs) / (n - 1)) if n > 1 else 0.0
    gp = sum(x for x in rs if x > 0); gl = -sum(x for x in rs if x <= 0)
    eq = pk = dd = 0.0
    for x in rs:
        eq += x; pk = max(pk, eq); dd = max(dd, pk - eq)
    return dict(n=n, net=net, exp=mean, t=(mean / (sd / math.sqrt(n)) if sd else 0.0),
                pf=(gp / gl if gl else float("inf")), dd=dd,
                wr=sum(1 for x in rs if x > 0) / n)


def fmt(s: dict) -> str:
    if s["n"] == 0:
        return "n=   0"
    return (f"n={s['n']:4d}  net {s['net']:+7.1f}R  exp {s['exp']:+.3f}  t {s['t']:+5.2f}  "
            f"PF {s['pf']:4.2f}  WR {s['wr']*100:4.1f}%  maxDD {s['dd']:5.1f}R")


def best_rule(train: list[dict], grid: list[tuple], criterion: str = "net") -> tuple:
    best, best_score = grid[0], -1e9
    for rule in grid:
        rs = [r["r"] for r in train if passes(r, rule)]
        if len(rs) < MIN_TRAIN_TRADES:
            continue
        s = st(rs)
        score = s["net"] if criterion == "net" else s["t"]
        if score > best_score:
            best, best_score = rule, score
    return best


def half_year_blocks(rows: list[dict], first_test: date) -> list[tuple[date, date]]:
    end = max(r["sd"] for r in rows)
    blocks, cur = [], first_test
    while cur <= end:
        nxt = date(cur.year, 7, 1) if cur.month == 1 else date(cur.year + 1, 1, 1)
        blocks.append((cur, nxt))
        cur = nxt
    return blocks


def walk_forward(rows, grid, first_test, criterion="net", verbose=True):
    oos_f, oos_b, per_block = [], [], []
    for lo, hi in half_year_blocks(rows, first_test):
        train = [r for r in rows if r["sd"] < lo]
        test = [r for r in rows if lo <= r["sd"] < hi]
        if not test:
            continue
        rule = best_rule(train, grid, criterion)
        kept = [r for r in test if passes(r, rule)]
        oos_f += kept; oos_b += test
        per_block.append((lo, test, kept, rule))
        if verbose:
            sb, sf = st([r["r"] for r in test]), st([r["r"] for r in kept])
            print(f"  {lo} -> {hi}  train n={len(train):4d}  rule: {label(rule):<34} "
                  f"| all {sb['net']:+6.1f}R (n={sb['n']:3d})  filtered {sf['net']:+6.1f}R (n={sf['n']:3d})")
    return oos_f, oos_b, per_block


def perm_p(per_block, n_iter=5000) -> float:
    """P(random same-size subsets per block do at least as well as the filter)."""
    obs = sum(r["r"] for _, _, kept, _ in per_block for r in kept)
    ge = 0
    for _ in range(n_iter):
        tot = 0.0
        for _, test, kept, _ in per_block:
            tot += sum(r["r"] for r in random.sample(test, len(kept)))
        ge += tot >= obs
    return ge / n_iter


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-latest", action="store_true",
                    help="top up the cached candles from OANDA to today")
    ap.add_argument("--dump", default=None,
                    help="directory to write trade series for the FTMO Monte Carlo")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    print("Loading candles and running Config-A (Tokyo, XAU + NAS100)...")
    rows = build_trades(args.fetch_latest)
    first, last = rows[0]["sd"], rows[-1]["sd"]
    print(f"\n{len(rows)} trades, {first} -> {last}")

    # ---- A. baseline -----------------------------------------------------
    print("\n" + "=" * 100 + "\nA. BASELINE (no filter) BY YEAR\n" + "=" * 100)
    years = sorted({r["year"] for r in rows})
    for y in years:
        for inst in INSTRUMENTS + ("ALL",):
            sel = [r["r"] for r in rows if r["year"] == y and (inst == "ALL" or r["inst"] == inst)]
            print(f"  {y} {inst[:6]:<6} {fmt(st(sel))}")
    print(f"  TOTAL        {fmt(st([r['r'] for r in rows]))}")
    print("\n  Share of trades with cost_r < 0.07, by year (fixed price cost, so this"
          "\n  rises mechanically as prices and ATR rise):")
    for y in years:
        sel = [r for r in rows if r["year"] == y]
        for inst in INSTRUMENTS:
            s2 = [r for r in sel if r["inst"] == inst]
            if s2:
                share = sum(1 for r in s2 if r["cost_r"] < 0.07) / len(s2)
                med = sorted(r["cost_r"] for r in s2)[len(s2) // 2]
                print(f"    {y} {inst[:6]}: {share*100:5.1f}% pass   median cost_r {med:.3f}")

    # ---- B. frozen rules, discovery vs after ------------------------------
    print("\n" + "=" * 100 + "\nB. FROZEN RULES AS DISCOVERED: discovery sample vs data that arrived afterwards\n" + "=" * 100)
    frozen = [
        ("cost<0.07", (0.07, None, False, "all", "both"), FORENSICS_DATE),
        ("cost<0.07 + noJul", (0.07, None, True, "all", "both"), FORENSICS_DATE),
        ("cost<0.07 + ADX>=20 + noJul", (0.07, 20.0, True, "all", "both"), FORENSICS_DATE),
        ("Mon-Wed only", (None, None, False, "MonWed", "both"), SEASONALITY_DATE),
        ("NAS only", (None, None, False, "all", "NAS"), SEASONALITY_DATE),
    ]
    for name, rule, disc in frozen:
        ins = [r for r in rows if r["day"] <= disc]
        out = [r for r in rows if r["day"] > disc]
        print(f"\n  {name}   (discovered {disc})")
        print(f"    discovery sample  filtered: {fmt(st([r['r'] for r in ins if passes(r, rule)]))}")
        print(f"                      all     : {fmt(st([r['r'] for r in ins]))}")
        print(f"    AFTER discovery   filtered: {fmt(st([r['r'] for r in out if passes(r, rule)]))}")
        print(f"                      all     : {fmt(st([r['r'] for r in out]))}")

    # ---- C. full-grid walk-forward ----------------------------------------
    first_test = date(2024, 1, 1)
    print("\n" + "=" * 100)
    print(f"C. ANCHORED WALK-FORWARD, full grid ({len(GRID)} rules), half-year blocks from {first_test}")
    print("   Rule for each block = highest net R on ALL PRIOR trades (min "
          f"{MIN_TRAIN_TRADES} training trades)\n" + "=" * 100)
    for crit in ("net", "t"):
        print(f"\n  Selection criterion: {'net R' if crit == 'net' else 't-statistic'}")
        f, b, pb = walk_forward(rows, GRID, first_test, crit)
        print(f"  OUT-OF-SAMPLE filtered: {fmt(st([r['r'] for r in f]))}")
        print(f"  OUT-OF-SAMPLE baseline: {fmt(st([r['r'] for r in b]))}")
        print(f"  Permutation p (random same-size subsets do as well): {perm_p(pb):.3f}")
        for yr in sorted({r['year'] for r in b}):
            fy = st([r["r"] for r in f if r["year"] == yr]); by = st([r["r"] for r in b if r["year"] == yr])
            print(f"    {yr}: filtered {fy['net']:+6.1f}R (n={fy['n']:3d})   baseline {by['net']:+6.1f}R (n={by['n']:3d})")

    # ---- D. single-filter walk-forward ------------------------------------
    print("\n" + "=" * 100 + "\nD. EACH FILTER ON ITS OWN, walk-forward (threshold re-chosen on prior data each block;"
          "\n   'no filter' is always an allowed choice)\n" + "=" * 100)
    none = (None, None, False, "all", "both")
    families = {
        "cost_r threshold": [(c, None, False, "all", "both") for c in COST_TH],
        "D1 ADX minimum": [(None, a, False, "all", "both") for a in ADX_TH],
        "skip July": [none, (None, None, True, "all", "both")],
        "Mon-Wed only": [none, (None, None, False, "MonWed", "both")],
        "instrument choice": [(None, None, False, "all", i) for i in INST],
    }
    for name, grid in families.items():
        f, b, pb = walk_forward(rows, grid, first_test, "net", verbose=False)
        chosen = ", ".join(label(rule) for _, _, _, rule in pb)
        sf, sb = st([r["r"] for r in f]), st([r["r"] for r in b])
        print(f"\n  {name}")
        print(f"    chosen per block: {chosen}")
        print(f"    OOS filtered: {fmt(sf)}")
        print(f"    OOS baseline: {fmt(sb)}   | improvement {sf['net'] - sb['net']:+.1f}R, permutation p {perm_p(pb):.3f}")

    # ---- E. leave-2026-out -------------------------------------------------
    print("\n" + "=" * 100 + "\nE. LEAVE-2026-OUT: rule chosen on trades up to 2025-12-31, tested on 2026\n" + "=" * 100)
    train = [r for r in rows if r["sd"] < date(2026, 1, 1)]
    test = [r for r in rows if r["sd"] >= date(2026, 1, 1)]
    ranked = []
    for rule in GRID:
        rs = [r["r"] for r in train if passes(r, rule)]
        if len(rs) >= MIN_TRAIN_TRADES:
            ranked.append((st(rs)["net"], rule))
    ranked.sort(reverse=True)
    print(f"  Training (to end-2025) baseline: {fmt(st([r['r'] for r in train]))}")
    print(f"  2026 baseline                  : {fmt(st([r['r'] for r in test]))}")
    print("\n  Top 8 rules on training data, and how each did in 2026:")
    for net, rule in ranked[:8]:
        tr = st([r["r"] for r in train if passes(r, rule)])
        te = st([r["r"] for r in test if passes(r, rule)])
        print(f"    {label(rule):<36} train {tr['net']:+6.1f}R (n={tr['n']:3d}, t {tr['t']:+.2f})"
              f"   2026 {te['net']:+6.1f}R (n={te['n']:3d}, exp {te['exp']:+.3f})")
    n_pos = sum(1 for net, _ in ranked if net > 0)
    print(f"\n  Rules with positive training net R: {n_pos} of {len(ranked)} eligible")
    # how do the best-on-2026 rules look pre-2026 (the reverse direction)
    print("\n  Reverse check - rule chosen on 2026 only, applied to 2022-2025:")
    r26 = []
    for rule in GRID:
        rs = [r["r"] for r in test if passes(r, rule)]
        if len(rs) >= 50:
            r26.append((st(rs)["net"], rule))
    r26.sort(reverse=True)
    for net, rule in r26[:4]:
        tr = st([r["r"] for r in train if passes(r, rule)])
        print(f"    {label(rule):<36} 2026 {net:+6.1f}R   2022-2025 {tr['net']:+6.1f}R (n={tr['n']:3d}, exp {tr['exp']:+.3f})")

    # ---- F. yearly table for the headline frozen rule ----------------------
    print("\n" + "=" * 100 + "\nF. YEAR-BY-YEAR for the main candidate rules (descriptive, in-sample)\n" + "=" * 100)
    cands = [none, (0.07, None, False, "all", "both"), (0.07, 20.0, True, "all", "both"),
             (None, None, False, "all", "NAS"), (None, None, False, "MonWed", "both"),
             (None, None, True, "MonWed", "NAS")]
    print(f"  {'rule':<34}" + "".join(f"{y:>9}" for y in years) + f"{'total':>9}{'n':>6}{'maxDD':>7}")
    for rule in cands:
        sel = [r for r in rows if passes(r, rule)]
        cells = "".join(f"{sum(r['r'] for r in sel if r['year'] == y):>+9.1f}" for y in years)
        s = st([r["r"] for r in sel])
        print(f"  {label(rule):<34}{cells}{s['net']:>+9.1f}{s['n']:>6}{s['dd']:>7.1f}")

    # ---- G. NAS100 detail: is the edge gross, or just falling cost? --------
    print("\n" + "=" * 100 + "\nG. NAS100 TOKYO DETAIL (the only leg that survives): gross vs net by year\n" + "=" * 100)
    nas = [r for r in rows if r["inst"] == "NAS100_USD"]
    print(f"  {'year':<6}{'n':>5}{'gross R':>10}{'cost R':>9}{'net R':>9}{'net exp':>9}{'gross exp':>10}{'Jul net':>9}")
    for y in years + ["ALL"]:
        sel = [r for r in nas if y == "ALL" or r["year"] == y]
        g = sum(r["gross"] for r in sel); n_ = sum(r["r"] for r in sel)
        jul = sum(r["r"] for r in sel if r["month"] == 7)
        print(f"  {str(y):<6}{len(sel):>5}{g:>+10.1f}{g - n_:>9.1f}{n_:>+9.1f}{n_/len(sel):>+9.3f}{g/len(sel):>+10.3f}{jul:>+9.1f}")
    print(f"  NAS full sample      : {fmt(st([r['r'] for r in nas]))}")
    print(f"  NAS full, skip July  : {fmt(st([r['r'] for r in nas if r['month'] != 7]))}")
    print(f"  NAS since 2024-01    : {fmt(st([r['r'] for r in nas if r['sd'] >= first_test]))}")
    print(f"  NAS since 2024, noJul: {fmt(st([r['r'] for r in nas if r['sd'] >= first_test and r['month'] != 7]))}")
    xau = [r for r in rows if r["inst"] == "XAU_USD"]
    print(f"  XAU full sample      : {fmt(st([r['r'] for r in xau]))}")
    print(f"  XAU gross by year    : " + "  ".join(
        f"{y}: {sum(r['gross'] for r in xau if r['year'] == y):+.0f}" for y in years))
    # cost sensitivity for NAS
    print("  NAS net R if the true cost were a multiple of the assumed 1.5 points:")
    for mult in (1.0, 1.5, 2.0):
        full = sum(r["gross"] - mult * r["cost_r"] for r in nas)
        oos = sum(r["gross"] - mult * r["cost_r"] for r in nas if r["sd"] >= first_test)
        print(f"    {mult:.1f}x cost: full sample {full:+7.1f}R   since 2024 {oos:+7.1f}R")

    if args.dump:
        import os
        os.makedirs(args.dump, exist_ok=True)
        series = {
            "both_full": rows,
            "nas_full": nas,
            "nas_full_nojul": [r for r in nas if r["month"] != 7],
            "nas_since2024": [r for r in nas if r["sd"] >= first_test],
            "nas_since2024_nojul": [r for r in nas if r["sd"] >= first_test and r["month"] != 7],
        }
        for name, sel in series.items():
            with open(f"{args.dump}/{name}.txt", "w") as fh:
                for r in sel:
                    fh.write(f"{r['day']} {r['inst']} RR={r['r']:+.4f}\n")
        print(f"\n  Daily trade series written to {args.dump}/")


if __name__ == "__main__":
    main()
