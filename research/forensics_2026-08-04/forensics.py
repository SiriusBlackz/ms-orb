"""Drawdown forensics on the 48-month config-A backtest.

Sections:
  A. Equity curve + all drawdown episodes (depth >= 8R)
  B. Seasonality: R by (year x calendar month) + month-of-year aggregate
  C. Loss-clustering statistics (runs test, bootstrap streak/variance tests)
  D. XAU/NAS same-day dependence
  E. Ex-ante regime features (D1 ATR/ADX/CHOP/EMA200, OR width) vs outcomes
  F. Loser anatomy: MFE before stop (instant fakeout vs near-miss)
"""
import glob
import math
import pickle
import random
import re
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

import pytz

sys.path.insert(0, "/Users/derianj/projects/ms-orb/oanda_bot")

NY = pytz.timezone("America/New_York")
SCRATCH = "/private/tmp/claude-501/-Users-derianj-projects-ms-orb/d48b00a9-a814-4b41-a31d-fd8c76689e9b/scratchpad"
CACHE = "/Users/derianj/projects/ms-orb/oanda_bot/cache"
DD_THRESHOLD = 8.0
random.seed(20260804)

# ---------- parse trades ----------
LINE = re.compile(
    r"^\s+(\d{4}-\d{2}-\d{2}) TOKYO\s+(\S+)\s+(LONG|SHORT)\s+entry=\s*([\d.]+) "
    r"sl=\s*([\d.]+) tp=\s*([\d.]+) exit=\s*([\d.]+) reason=(\w+)\s+RR=([+-][\d.]+) "
    r"vol=([\d.]+|N/A) ADX=([\d.]+|N/A)")

trades = []
for raw in open(f"{SCRATCH}/configa_48mo.txt"):
    m = LINE.match(raw)
    if m:
        trades.append(dict(
            date=date.fromisoformat(m.group(1)), inst=m.group(2),
            dir=m.group(3), entry=float(m.group(4)), sl=float(m.group(5)),
            tp=float(m.group(6)), exit=float(m.group(7)), reason=m.group(8),
            rr=float(m.group(9)),
            rvol=None if m.group(10) == "N/A" else float(m.group(10)),
            adx5=None if m.group(11) == "N/A" else float(m.group(11))))
print(f"Parsed {len(trades)} trades, {trades[0]['date']} -> {trades[-1]['date']}")

daily = defaultdict(float)
daily_by_inst = defaultdict(lambda: defaultdict(float))
for t in trades:
    daily[t["date"]] += t["rr"]
    daily_by_inst[t["inst"]][t["date"]] += t["rr"]
dates = sorted(daily)
rs = [daily[d] for d in dates]

# ---------- A. equity + DD episodes ----------
print("\n" + "=" * 78)
print("A. DRAWDOWN EPISODES (combined equity, depth >= %.0fR)" % DD_THRESHOLD)
print("=" * 78)
eq = peak = 0.0
peak_d = dates[0]
episodes = []          # (peak_date, trough_date, depth, recovered_date|None)
cur_trough, cur_trough_d = 0.0, None
open_ep = False
for d, r in zip(dates, rs):
    eq += r
    if eq >= peak:
        if open_ep and peak - cur_trough >= DD_THRESHOLD:
            episodes.append((peak_d, cur_trough_d, peak - cur_trough, d))
        peak, peak_d, open_ep = eq, d, False
    else:
        if not open_ep or eq < cur_trough:
            cur_trough, cur_trough_d = eq, d
            open_ep = True
if open_ep and peak - cur_trough >= DD_THRESHOLD:
    episodes.append((peak_d, cur_trough_d, peak - cur_trough, None))

ep_days = set()
for pk, tr, depth, rec in episodes:
    span = [d for d in dates if pk < d <= tr]
    ep_days.update(span)
    n_tr = sum(1 for t in trades if pk < t["date"] <= tr)
    wr = (sum(1 for t in trades if pk < t["date"] <= tr and t["rr"] > 0)
          / n_tr * 100 if n_tr else 0)
    rec_s = rec.isoformat() if rec else "NOT RECOVERED"
    print(f"  {pk} -> {tr}  depth {depth:5.1f}R  {len(span):3d} act.days "
          f"{n_tr:3d} trades  WR {wr:4.1f}%  recovered {rec_s}")
print(f"\nTotal R: {eq:+.1f} over {len(dates)} active days "
      f"({len(ep_days)} inside episodes = {len(ep_days)/len(dates):.0%})")

# ---------- B. seasonality ----------
print("\n" + "=" * 78)
print("B. SEASONALITY  (sum R per year x month)")
print("=" * 78)
ym = defaultdict(float)
for d, r in zip(dates, rs):
    ym[(d.year, d.month)] += r
years = sorted({y for y, _ in ym})
print("      " + "".join(f"{m:>7}" for m in range(1, 13)))
for y in years:
    row = "".join(f"{ym.get((y, m), float('nan')):>+7.1f}"
                  if (y, m) in ym else f"{'--':>7}" for m in range(1, 13))
    print(f"{y}  {row}")
print("-" * 90)
avg_row = []
for m in range(1, 13):
    vals = [ym[(y, m)] for y in years if (y, m) in ym]
    avg_row.append(f"{statistics.mean(vals):>+7.1f}" if vals else f"{'--':>7}")
print("mean  " + "".join(avg_row))

# ---------- C. loss clustering ----------
print("\n" + "=" * 78)
print("C. LOSS-DAY CLUSTERING")
print("=" * 78)
signs = [1 if r > 0 else 0 for r in rs]
n1, n0 = sum(signs), len(signs) - sum(signs)
runs = 1 + sum(1 for a, b in zip(signs, signs[1:]) if a != b)
mu = 2 * n1 * n0 / (n1 + n0) + 1
var = (mu - 1) * (mu - 2) / (n1 + n0 - 1)
z = (runs - mu) / math.sqrt(var)
print(f"  Win days {n1}/{len(signs)} ({n1/len(signs):.0%}). Runs test: "
      f"observed {runs} vs expected {mu:.0f}, z = {z:+.2f} "
      f"({'CLUSTERED' if z < -1.64 else 'not significantly clustered'})")

def max_loss_streak(seq):
    best = cur = 0
    for s in seq:
        cur = cur + 1 if s == 0 else 0
        best = max(best, cur)
    return best

def var10(seq):
    sums = [sum(seq[i:i + 10]) for i in range(0, len(seq) - 9)]
    return statistics.pvariance(sums)

obs_streak, obs_v10 = max_loss_streak(signs), var10(rs)
ge_streak = ge_v = 0
N = 2000
pool = rs[:]
for _ in range(N):
    random.shuffle(pool)
    if max_loss_streak([1 if r > 0 else 0 for r in pool]) >= obs_streak:
        ge_streak += 1
    if var10(pool) >= obs_v10:
        ge_v += 1
print(f"  Max losing-day streak: {obs_streak} (shuffle p = {ge_streak/N:.3f})")
print(f"  10-day-sum variance: {obs_v10:.1f} (shuffle p = {ge_v/N:.3f}) "
      f"-> {'losses CLUSTER beyond iid' if ge_v/N < 0.05 else 'consistent with iid'}")

# ---------- D. XAU/NAS dependence ----------
print("\n" + "=" * 78)
print("D. XAU vs NAS SAME-DAY DEPENDENCE")
print("=" * 78)
both = [d for d in dates
        if d in daily_by_inst["XAU_USD"] and d in daily_by_inst["NAS100_USD"]]
xs = [daily_by_inst["XAU_USD"][d] for d in both]
ns = [daily_by_inst["NAS100_USD"][d] for d in both]
if len(both) > 2:
    corr = statistics.correlation(xs, ns)
    jl = sum(1 for a, b in zip(xs, ns) if a < 0 and b < 0) / len(both)
    px = sum(1 for a in xs if a < 0) / len(both)
    pn = sum(1 for b in ns if b < 0) / len(both)
    print(f"  {len(both)} days with both traded. Daily-R corr = {corr:+.2f}. "
          f"Joint-loss {jl:.0%} vs {px*pn:.0%} if independent.")

# ---------- load candles ----------
def load(inst):
    files = sorted(glob.glob(f"{CACHE}/{inst}_*_M5.pkl"))
    spans = [(f, (datetime.strptime(f.split('_')[-2], '%Y%m%d')
                  - datetime.strptime(f.split('_')[-3], '%Y%m%d')).days)
             for f in files]
    f = max(spans, key=lambda x: x[1])[0]
    with open(f, "rb") as fh:
        return pickle.load(fh)

candles = {i: load(i) for i in ("XAU_USD", "NAS100_USD")}
for i, cs in candles.items():
    print(f"\nLoaded {i}: {len(cs)} M5 candles "
          f"{cs[0].time.date()} -> {cs[-1].time.date()}")

# ---------- D1 features ----------
def d1_features(cs):
    """FX-convention daily bars (17:00->17:00 NY, labeled by close date)."""
    bars = {}
    for c in cs:
        ny = c.time.astimezone(NY)
        key = (ny + timedelta(hours=7)).date()
        b = bars.get(key)
        if b is None:
            bars[key] = [c.open, c.high, c.low, c.close]
        else:
            b[1] = max(b[1], c.high); b[2] = min(b[2], c.low); b[3] = c.close
    ds = sorted(bars)
    o = [bars[d][0] for d in ds]; h = [bars[d][1] for d in ds]
    l = [bars[d][2] for d in ds]; cl = [bars[d][3] for d in ds]
    n = len(ds)
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - cl[i-1]),
                              abs(l[i] - cl[i-1])) for i in range(1, n)]
    atr = [None] * n
    if n > 14:
        a = sum(tr[1:15]) / 14
        atr[14] = a
        for i in range(15, n):
            a = (a * 13 + tr[i]) / 14
            atr[i] = a
    # Wilder ADX14
    pdm = [0.0] * n; ndm = [0.0] * n
    for i in range(1, n):
        up, dn = h[i] - h[i-1], l[i-1] - l[i]
        pdm[i] = up if up > dn and up > 0 else 0.0
        ndm[i] = dn if dn > up and dn > 0 else 0.0
    adx = [None] * n
    if n > 28:
        str_, spd, snd = sum(tr[1:15]), sum(pdm[1:15]), sum(ndm[1:15])
        dxs = []
        for i in range(15, n):
            str_ = str_ - str_ / 14 + tr[i]
            spd = spd - spd / 14 + pdm[i]
            snd = snd - snd / 14 + ndm[i]
            pdi, ndi = 100 * spd / str_, 100 * snd / str_
            dx = 100 * abs(pdi - ndi) / (pdi + ndi) if pdi + ndi else 0
            dxs.append(dx)
            j = i - 15
            if j == 13:
                adx[i] = sum(dxs) / 14
            elif j > 13:
                adx[i] = (adx[i-1] * 13 + dx) / 14
    chop = [None] * n
    for i in range(14, n):
        rng = max(h[i-13:i+1]) - min(l[i-13:i+1])
        s = sum(tr[i-13:i+1])
        chop[i] = 100 * math.log10(s / rng) / math.log10(14) if rng > 0 else None
    ema = [None] * n
    k = 2 / 201
    e = cl[0]
    for i in range(n):
        e = cl[i] * k + e * (1 - k)
        if i >= 199:
            ema[i] = e
    atr_pct = [None] * n
    for i in range(n):
        if atr[i] is None:
            continue
        win = [a for a in atr[max(0, i - 251):i + 1] if a is not None]
        if len(win) >= 60:
            atr_pct[i] = sum(1 for a in win if a <= atr[i]) / len(win)
    feats = {}
    for i, d in enumerate(ds):
        feats[d] = dict(atr=atr[i], atr_pct=atr_pct[i], adx=adx[i],
                        chop=chop[i],
                        ema_dist=(abs(cl[i] - ema[i]) / atr[i]
                                  if ema[i] and atr[i] else None))
    return feats, ds

feats = {}; d1_dates = {}
for inst in candles:
    feats[inst], d1_dates[inst] = d1_features(candles[inst])

def feat_asof(inst, session_date):
    """Latest completed D1 bar at 18:00 NY on session_date = bar labeled session_date."""
    ds = d1_dates[inst]
    prior = [d for d in ds if d <= session_date]
    return feats[inst][prior[-1]] if prior else None

# Tokyo opening range width per (inst, session_date)
or_width = defaultdict(dict)
for inst, cs in candles.items():
    for c in cs:
        ny = c.time.astimezone(NY)
        if ny.hour == 18 and ny.minute < 30:
            d = ny.date()
            w = or_width[inst].get(d)
            if w is None:
                or_width[inst][d] = [c.high, c.low]
            else:
                w[0] = max(w[0], c.high); w[1] = min(w[1], c.low)

# ---------- E. feature terciles + episode comparison ----------
print("\n" + "=" * 78)
print("E. EX-ANTE REGIME FEATURES vs TRADE OUTCOME")
print("=" * 78)
rows = []
for t in trades:
    f = feat_asof(t["inst"], t["date"])
    if not f:
        continue
    w = or_width[t["inst"]].get(t["date"])
    orw = (w[0] - w[1]) / f["atr"] if w and f["atr"] else None
    rows.append(dict(t, **f, or_atr=orw, in_ep=t["date"] in ep_days))

def tercile_table(key, label):
    vals = [(r[key], r["rr"]) for r in rows if r.get(key) is not None]
    if len(vals) < 60:
        return
    vs = sorted(v for v, _ in vals)
    lo, hi = vs[len(vs) // 3], vs[2 * len(vs) // 3]
    buckets = {"low": [], "mid": [], "high": []}
    for v, r in vals:
        buckets["low" if v <= lo else "high" if v > hi else "mid"].append(r)
    line = f"  {label:<26}"
    for b in ("low", "mid", "high"):
        rr = buckets[b]
        line += (f"  {b}: {statistics.mean(rr):+.3f}R/tr "
                 f"(n={len(rr)}, WR {sum(1 for x in rr if x>0)/len(rr):.0%})")
    print(line + f"   [cuts: {lo:.2f} / {hi:.2f}]")

for r in rows:
    risk = abs(r["entry"] - r["sl"])
    r["cost_r"] = (0.35 if r["inst"] == "XAU_USD" else 1.5) / risk if risk else None

print("\nExpectancy by feature tercile (ex-ante at session start):")
tercile_table("cost_r", "Spread cost in R (ex-ante)")
tercile_table("atr_pct", "D1 ATR 1-yr percentile")
tercile_table("adx", "D1 ADX14")
tercile_table("chop", "D1 CHOP14")
tercile_table("ema_dist", "|close-EMA200|/ATR (D1)")
tercile_table("or_atr", "OR width / D1 ATR")
tercile_table("rvol", "RVOL at signal (M5)")

print("\nNet R if trading ONLY when ex-ante cost_r < threshold "
      "(descriptive, not OOS):")
for th in (0.05, 0.06, 0.07, 0.08, 0.10, 1.0):
    sel = [r for r in rows if r.get("cost_r") is not None and r["cost_r"] < th]
    if sel:
        yr = defaultdict(float)
        for r in sel:
            yr[r["date"].year] += r["rr"]
        yrs = " ".join(f"{y}:{v:+.0f}" for y, v in sorted(yr.items()))
        print(f"  cost_r<{th:4.2f}: n={len(sel):4d}  total {sum(r['rr'] for r in sel):+7.1f}R  "
              f"exp {statistics.mean(r['rr'] for r in sel):+.3f}R/tr   [{yrs}]")

print("\nFeature means: inside DD episodes vs outside:")
for key, label in (("atr_pct", "D1 ATR percentile"), ("adx", "D1 ADX14"),
                   ("chop", "D1 CHOP14"), ("ema_dist", "|c-EMA200|/ATR"),
                   ("or_atr", "OR/ATR"), ("rvol", "RVOL")):
    ins = [r[key] for r in rows if r["in_ep"] and r.get(key) is not None]
    outs = [r[key] for r in rows if not r["in_ep"] and r.get(key) is not None]
    if ins and outs:
        print(f"  {label:<22} in-DD {statistics.mean(ins):6.2f}  "
              f"outside {statistics.mean(outs):6.2f}")

# ---------- F. loser anatomy (MFE before stop) ----------
print("\n" + "=" * 78)
print("F. LOSER ANATOMY - max favorable excursion before the stop hit")
print("=" * 78)
by_time = {i: sorted(cs, key=lambda c: c.time) for i, cs in candles.items()}

def mfe_for(t):
    cs = by_time[t["inst"]]
    d = t["date"]
    start = NY.localize(datetime(d.year, d.month, d.day, 18, 30))
    end = start + timedelta(hours=8, minutes=24)
    win = [c for c in cs if start <= c.time.astimezone(NY) <= end]
    ei = next((i for i, c in enumerate(win)
               if abs(c.close - t["entry"]) < 0.011), None)
    if ei is None:
        return None
    risk = abs(t["entry"] - t["sl"])
    best = 0.0
    for c in win[ei + 1:]:
        if t["dir"] == "LONG":
            best = max(best, c.high - t["entry"])
            if c.low <= t["sl"]:
                return best / risk
        else:
            best = max(best, t["entry"] - c.low)
            if c.high >= t["sl"]:
                return best / risk
    return best / risk

stopped = [t for t in trades if t["reason"] == "stop"]
mfes = []
for t in stopped:
    m = mfe_for(t)
    if m is not None:
        mfes.append((t, m))
print(f"  Matched {len(mfes)}/{len(stopped)} stopped trades to candles.")
for scope, sel in (("ALL", mfes),
                   ("in-DD", [(t, m) for t, m in mfes if t["date"] in ep_days]),
                   ("outside", [(t, m) for t, m in mfes if t["date"] not in ep_days])):
    if not sel:
        continue
    n = len(sel)
    b1 = sum(1 for _, m in sel if m < 0.25) / n
    b2 = sum(1 for _, m in sel if 0.25 <= m < 1.0) / n
    b3 = sum(1 for _, m in sel if m >= 1.0) / n
    print(f"  {scope:<8} n={n:3d}  instant fakeout (<0.25R): {b1:.0%}   "
          f"partial (0.25-1R): {b2:.0%}   near-miss (>=1R): {b3:.0%}   "
          f"median MFE {statistics.median(m for _, m in sel):.2f}R")
