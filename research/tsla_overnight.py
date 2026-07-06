"""Overnight vs intraday return decomposition — TSLA, 4 years of RTH 5-min data.

The classic flow effect: overnight (prior close -> open) captures news +
opening-auction flow; intraday (open -> close) is where retail pays spread.
Measures: cumulative log-return split by year, t-stats, and two cheap
conditional variants. Measurement only — no strategy code.
"""
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime

CSV_PATH = "/Users/derianj/projects/orb-trading-bot/data/TSLA_20220101_20260125_5min_alpaca.csv"

days = {}  # date -> (open_first, close_last)
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        d = row["date"][:10]
        o, c = float(row["open"]), float(row["close"])
        if d not in days:
            days[d] = [o, c]
        else:
            days[d][1] = c

dates = sorted(days)
# drop the unadjusted split gap (2022-08-25) from overnight legs
SPLIT_DAYS = {"2022-08-25"}

overnight, intraday = [], []   # (date, log-ret)
for i in range(1, len(dates)):
    d_prev, d = dates[i - 1], dates[i]
    if d in SPLIT_DAYS:
        continue
    prev_close = days[d_prev][1]
    o, c = days[d]
    overnight.append((d, math.log(o / prev_close)))
    intraday.append((d, math.log(c / o)))

def stats(rets):
    xs = [r for _, r in rets]
    n = len(xs)
    mu, sd = statistics.mean(xs), statistics.pstdev(xs)
    t = mu / (sd / math.sqrt(n)) if sd else 0
    ann = mu * 252
    return n, ann, t, sum(xs)

print(f"{'leg':22s} {'n':>5s} {'ann.logret':>11s} {'t-stat':>7s} {'cum':>8s}")
for name, rets in (("OVERNIGHT (c->o)", overnight), ("INTRADAY (o->c)", intraday)):
    n, ann, t, cum = stats(rets)
    print(f"{name:22s} {n:5d} {ann*100:>10.1f}% {t:>7.2f} {cum*100:>7.1f}%")

print("\nPer-year split (cum log-ret %):")
by = defaultdict(lambda: [0.0, 0.0])
for (d, r_on), (_, r_id) in zip(overnight, intraday):
    y = d[:4]
    by[y][0] += r_on
    by[y][1] += r_id
print(f"{'year':6s} {'overnight':>10s} {'intraday':>10s} {'total':>10s}")
for y in sorted(by):
    on, iday = by[y]
    print(f"{y:6s} {on*100:>9.1f}% {iday*100:>9.1f}% {(on+iday)*100:>9.1f}%")

# conditional variants
print("\nConditional overnight (held only after condition on prior day):")
conds = {
    "after UP intraday day": lambda i: intraday[i - 1][1] > 0,
    "after DOWN intraday day": lambda i: intraday[i - 1][1] < 0,
    "after UP overnight": lambda i: overnight[i - 1][1] > 0,
}
for name, cond in conds.items():
    sel = [overnight[i] for i in range(1, len(overnight)) if cond(i)]
    n, ann, t, cum = stats(sel)
    print(f"  {name:26s} n={n:4d}  ann={ann*100:+7.1f}%  t={t:+5.2f}  cum={cum*100:+7.1f}%")

# cost reality: one round trip per day (MOC buy + MOO sell or vice versa)
print("\nCost hurdle: 2 crossings/day. TSLA spread ~1.5bp + fees ~0.4bp")
print(f"  => ~{2*1.9*252/100:.1f}%/yr cost drag on an every-day overnight hold")
