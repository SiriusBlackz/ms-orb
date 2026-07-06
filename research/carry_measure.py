"""First-pass measurement of short-perp funding carry (delta-neutral construction).

Position model: long spot + short perp, equal notional -> collect funding.
Gross carry over a window = sum of funding rates (return on notional,
non-compounded). Costs modeled explicitly:
  - entry+exit: spot taker 0.10% x2 + perp taker 0.05% x2 = 0.30% round trip
  - always-on = one round trip per year; conditional = round trip per flip
No prediction anywhere; the only 'signal' tested is a trailing-average gate.
"""
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FUND = Path(__file__).parent / "funding"
FEE_RT = 0.0030          # spot 10bp x2 + perp 5bp x2, taker both legs
YEAR_HOURS = 365 * 24

data = {}
for f in sorted(FUND.glob("*.csv")):
    rows = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["fundingTime"]), float(r["fundingRate"])))
    rows.sort()
    data[f.stem] = rows

print(f"{'symbol':9s} {'events':>6s} {'span':>19s}")
for s, rows in data.items():
    t0 = datetime.fromtimestamp(rows[0][0]/1000, tz=timezone.utc).date()
    t1 = datetime.fromtimestamp(rows[-1][0]/1000, tz=timezone.utc).date()
    print(f"{s:9s} {len(rows):6d}  {t0} .. {t1}")

# ── per-symbol, per-year gross & net carry (always-on) ──────
print("\n=== ANNUAL GROSS CARRY, % of notional (short perp collects positive funding) ===")
years = list(range(2022, 2027))
header = f"{'symbol':9s}" + "".join(f"{y:>9d}" for y in years) + f"{'pos.8h%':>9s}{'worst30d':>9s}"
print(header)
portfolio_by_period = defaultdict(list)   # rounded 8h bucket -> list of rates
for s, rows in data.items():
    by_year = defaultdict(float)
    pos = 0
    rates = []
    for t, r in rows:
        y = datetime.fromtimestamp(t/1000, tz=timezone.utc).year
        by_year[y] += r
        pos += r > 0
        rates.append((t, r))
        portfolio_by_period[t // (8*3600*1000)].append(r)
    # worst rolling 30d funding sum
    worst = 0.0
    ts = [t for t, _ in rates]
    import bisect
    cum = [0.0]
    for _, r in rates:
        cum.append(cum[-1] + r)
    for i, t in enumerate(ts):
        j = bisect.bisect_right(ts, t + 30*24*3600*1000)
        worst = min(worst, cum[j] - cum[i])
    cells = "".join(f"{by_year.get(y, 0)*100:>9.1f}" for y in years)
    print(f"{s:9s}{cells}{pos/len(rows)*100:>9.1f}{worst*100:>9.2f}")

# ── equal-weight portfolio (all 8), always-on, net of fees ──
print("\n=== EQUAL-WEIGHT 8-SYMBOL PORTFOLIO (delta-neutral, always on) ===")
period_returns = sorted(
    (k, sum(v)/len(v)) for k, v in portfolio_by_period.items() if len(v) >= 4)
by_year_p = defaultdict(float)
for k, r in period_returns:
    y = datetime.fromtimestamp(k*8*3600, tz=timezone.utc).year
    by_year_p[y] += r
total_days = (period_returns[-1][0] - period_returns[0][0]) * 8 / 24
for y in years:
    if by_year_p.get(y) is None:
        continue
    gross = by_year_p[y]*100
    print(f"  {y}: gross {gross:+6.2f}%   net(1 RT fee/yr) {gross-FEE_RT*100:+6.2f}%")
g = sum(r for _, r in period_returns)
ann = g / (total_days/365)
# rough vol of the 8h portfolio funding stream -> Sharpe of the carry stream
import statistics
rets = [r for _, r in period_returns]
mu = statistics.mean(rets); sd = statistics.pstdev(rets)
periods_per_year = 3*365
sharpe = (mu*periods_per_year - FEE_RT) / (sd*(periods_per_year**0.5)) if sd else 0
print(f"\n  full span: gross {g*100:+.1f}% over {total_days/365:.2f}y "
      f"= {ann*100:+.2f}%/yr annualized; net ~{(ann-FEE_RT)*100:+.2f}%/yr")
print(f"  funding-stream Sharpe (excl. basis/price noise): ~{sharpe:.1f}")
print(f"  NOTE: Sharpe here is of the FUNDING stream only — execution basis noise,")
print(f"  spot-perp tracking and exchange risk are NOT in this number.")

# ── conditional gate: hold only when trailing 7d avg funding > hurdle ──
print("\n=== CONDITIONAL: hold only when trailing 7-day avg 8h-funding > hurdle ===")
print(f"{'symbol':9s} {'gross%/yr':>10s} {'flips/yr':>9s} {'net%/yr':>9s}   (hurdle=1bp/8h)")
HURDLE = 0.0001
for s, rows in data.items():
    window = []
    holding = False
    total = 0.0
    flips = 0
    for i, (t, r) in enumerate(rows):
        if holding:
            total += r
        window.append((t, r))
        cutoff = t - 7*24*3600*1000
        while window and window[0][0] < cutoff:
            window.pop(0)
        avg = sum(x for _, x in window)/len(window)
        want = avg > HURDLE
        if want != holding:
            flips += 1
            holding = want
    span_y = (rows[-1][0]-rows[0][0])/(365*24*3600*1000)
    gross_yr = total/span_y*100
    net_yr = gross_yr - (flips/span_y/2)*FEE_RT*100
    print(f"{s:9s} {gross_yr:>10.2f} {flips/span_y:>9.1f} {net_yr:>9.2f}")
