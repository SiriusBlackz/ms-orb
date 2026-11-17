"""FTMO 2-Step Swing Monte Carlo on any trade series (day-block bootstrap).

Same model as research/forensics_2026-08-04/mc_ftmo_rerun.py, parametrised:
  python3 research/mc_ftmo.py <trades.txt> [<trades.txt> ...]
Each input line: "<YYYY-MM-DD> <instrument> RR=<signed R>".

Terms: $100k, Phase 1 +10%, Phase 2 +5%, max total loss 10% static, max daily
loss 5%, no time limit, $600 fee refunded with first payout, 80% split, monthly
payout with balance reset, funded horizon 24 months, fixed % of initial per R.
"""
import random
import statistics
import sys
from collections import defaultdict

FEE, BAL = 600.0, 100_000.0
P1_TARGET, P2_TARGET = 0.10, 0.05
MAX_TOTAL_DD, MAX_DAILY_DD = 0.10, 0.05
SPLIT, N_SIMS, BLOCK, FUNDED_MONTHS = 0.80, 4000, 10, 24
RISKS = (0.0025, 0.0035, 0.005, 0.0075, 0.01)


def load(path):
    daily = defaultdict(float)
    for line in open(path):
        parts = line.split()
        daily[parts[0]] += float(next(p for p in parts if p.startswith("RR="))[3:])
    keys = sorted(daily)
    days = [daily[k] for k in keys]
    span = ((int(keys[-1][:4]) - int(keys[0][:4])) * 12
            + int(keys[-1][5:7]) - int(keys[0][5:7]) + 1)
    return days, len(days) / span, keys[0], keys[-1]


def run(path):
    days, dpm, first, last = load(path)
    n = len(days)

    def gen():
        while True:
            s = random.randrange(n)
            for i in range(s, min(s + BLOCK, n)):
                yield days[i]

    def phase(g, risk, target):
        eq, d = BAL, 0
        while True:
            pnl = next(g) * risk * BAL
            d += 1
            if pnl <= -MAX_DAILY_DD * BAL:
                return False, d
            eq += pnl
            if eq <= BAL * (1 - MAX_TOTAL_DD):
                return False, d
            if eq >= BAL * (1 + target):
                return True, d
            if d > 5000:
                return False, d

    def funded(g, risk):
        total, eq, md, months, paid = 0.0, BAL, 0, 0, False
        while months < FUNDED_MONTHS:
            pnl = next(g) * risk * BAL
            md += 1
            if pnl <= -MAX_DAILY_DD * BAL:
                return total, False
            eq += pnl
            if eq <= BAL * (1 - MAX_TOTAL_DD):
                return total, False
            if md >= dpm:
                months += 1; md = 0
                if eq > BAL:
                    total += SPLIT * (eq - BAL)
                    if not paid:
                        total += FEE; paid = True
                    eq = BAL
        return total, True

    print(f"\n{path.split('/')[-1]}: {n} active days {first} -> {last} "
          f"({dpm:.1f}/month)  sum {sum(days):+.1f}R  mean {statistics.mean(days):+.3f}R/day")
    print(f"  {'risk':>6} {'P(pass P1)':>11} {'P(funded)':>10} {'months to fund':>15} "
          f"{'P(survive 2y)':>14} {'EV per attempt':>15}")
    for risk in RISKS:
        random.seed(20261117)
        p1 = p2 = surv = 0
        pays, dtf = [], []
        for _ in range(N_SIMS):
            g = gen()
            ok, d1 = phase(g, risk, P1_TARGET)
            if not ok:
                pays.append(0.0); continue
            p1 += 1
            ok, d2 = phase(g, risk, P2_TARGET)
            if not ok:
                pays.append(0.0); continue
            p2 += 1; dtf.append(d1 + d2)
            pay, alive = funded(g, risk)
            pays.append(pay); surv += alive
        mo = statistics.median(dtf) / dpm if dtf else float("nan")
        print(f"  {risk:>6.2%} {p1/N_SIMS:>11.1%} {p2/N_SIMS:>10.1%} {mo:>14.1f}m "
              f"{(surv/p2 if p2 else 0):>14.1%} ${statistics.mean(pays) - FEE:>+14,.0f}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        run(f)
