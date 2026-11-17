"""FTMO Monte Carlo re-run with the updated 13-month config-A daily R series.

Reconstruction of the 2026-07-05 viability Monte Carlo (day-block bootstrap,
FTMO 2-Step Swing $100k terms), now fed with the Aug-2025..Aug-2026 window that
includes the worst-ever July 2026 month (-23.4R) / 35R max drawdown.

Terms modeled (FTMO 2-Step Swing $100k, 2026):
  Phase 1: +10% target | Phase 2: +5% target
  Max total loss: 10% static from initial | Max daily loss: 5%
  No phase time limit. Fee $600, refunded with first funded payout.
  Funded: 80% profit split, monthly payout, balance resets to initial after payout.
  Funded horizon capped at 2 years. Sizing = fixed % of INITIAL balance per R.
"""
import random
import statistics
from collections import defaultdict

random.seed(20260804)

TRADES_FILE = "configa_trades.txt"
FEE = 600.0
BAL = 100_000.0
P1_TARGET, P2_TARGET = 0.10, 0.05
MAX_TOTAL_DD, MAX_DAILY_DD = 0.10, 0.05
SPLIT = 0.80
N_SIMS = 5000
BLOCK = 10          # day-block length (2 trading weeks)
FUNDED_MONTHS = 24  # 2-year funded horizon

# ---- load daily R series ----
daily = defaultdict(float)
for line in open(TRADES_FILE):
    parts = line.split()
    date = parts[0]
    rr = next(p for p in parts if p.startswith("RR="))
    daily[date] += float(rr[3:])
days = [daily[d] for d in sorted(daily)]
n_days = len(days)

# active trading days per calendar month, measured from the data span
first, last = min(daily), max(daily)
span_months = ((int(last[:4]) - int(first[:4])) * 12
               + int(last[5:7]) - int(first[5:7]) + 1)
days_per_month = n_days / span_months

print(f"Daily R series: {n_days} active days over ~{span_months} months "
      f"({days_per_month:.1f} active days/month)")
print(f"  mean {statistics.mean(days):+.3f}R/day  sd {statistics.pstdev(days):.3f}  "
      f"min {min(days):+.2f}  max {max(days):+.2f}  sumR {sum(days):+.1f}")


def path():
    """Infinite daily-R generator via day-block bootstrap."""
    while True:
        start = random.randrange(n_days)
        for i in range(start, min(start + BLOCK, n_days)):
            yield days[i]


def run_phase(gen, risk, target):
    """Return (passed, days_used). Breach on -10% static or -5% daily."""
    eq = BAL
    d = 0
    while True:
        r = next(gen)
        d += 1
        pnl = r * risk * BAL
        if pnl <= -MAX_DAILY_DD * BAL:
            return False, d
        eq += pnl
        if eq <= BAL * (1 - MAX_TOTAL_DD):
            return False, d
        if eq >= BAL * (1 + target):
            return True, d
        if d > 5000:   # pathological non-mover
            return False, d


def run_funded(gen, risk):
    """Return total trader payouts over <=2yr. Monthly payout+reset."""
    total = 0.0
    eq = BAL
    month_days = 0
    months = 0
    first_payout_done = False
    while months < FUNDED_MONTHS:
        r = next(gen)
        month_days += 1
        pnl = r * risk * BAL
        if pnl <= -MAX_DAILY_DD * BAL:
            return total, months, False
        eq += pnl
        if eq <= BAL * (1 - MAX_TOTAL_DD):
            return total, months, False
        if month_days >= days_per_month:
            months += 1
            month_days = 0
            if eq > BAL:
                total += SPLIT * (eq - BAL)
                if not first_payout_done:
                    total += FEE  # fee refund with first payout
                    first_payout_done = True
                eq = BAL
    return total, months, True


def simulate(risk):
    p1_pass = p2_pass = survived = 0
    days_to_fund = []
    payouts = []
    for _ in range(N_SIMS):
        gen = path()
        ok1, d1 = run_phase(gen, risk, P1_TARGET)
        if not ok1:
            payouts.append(0.0)
            continue
        p1_pass += 1
        ok2, d2 = run_phase(gen, risk, P2_TARGET)
        if not ok2:
            payouts.append(0.0)
            continue
        p2_pass += 1
        days_to_fund.append(d1 + d2)
        pay, months, alive = run_funded(gen, risk)
        payouts.append(pay)
        if alive:
            survived += 1
    ev = statistics.mean(payouts) - FEE
    med_pay = statistics.median(payouts)
    funded_rate = p2_pass / N_SIMS
    mo_to_fund = (statistics.median(days_to_fund) / days_per_month
                  if days_to_fund else float("nan"))
    surv = survived / p2_pass if p2_pass else 0.0
    paying = [p for p in payouts if p > 0]
    mean_pay_if_funded = statistics.mean(paying) if paying else 0.0
    return (risk, p1_pass / N_SIMS, funded_rate, mo_to_fund, surv,
            mean_pay_if_funded, med_pay, ev)


print(f"\nFTMO 2-Step Swing $100k | fee ${FEE:.0f} (refunded) | split {SPLIT:.0%} | "
      f"block={BLOCK}d | {N_SIMS} sims/config | funded horizon {FUNDED_MONTHS}mo")
print(f"\n{'risk':>6} {'P(P1)':>7} {'P(fund)':>8} {'mo->fund':>9} "
      f"{'P(srv 2y)':>10} {'E[pay|fund]':>12} {'med pay':>9} {'EV/attempt':>11}")
for risk in (0.0025, 0.003, 0.004, 0.005, 0.0075, 0.01):
    r, p1, pf, mo, sv, mpf, med, ev = simulate(risk)
    print(f"{r:>6.2%} {p1:>7.1%} {pf:>8.1%} {mo:>8.1f}m {sv:>10.1%} "
          f"${mpf:>11,.0f} ${med:>8,.0f} ${ev:>+10,.0f}")

# block-length sensitivity at 0.25% and 0.5%
print("\nBlock-length sensitivity (EV/attempt):")
for bl in (5, 21):
    BLOCK = bl
    for risk in (0.0025, 0.005):
        _, p1, pf, mo, sv, mpf, med, ev = simulate(risk)
        print(f"  block={bl:>2}d risk={risk:.2%}: P(fund)={pf:.1%} "
              f"P(srv 2y)={sv:.1%} EV=${ev:+,.0f}")
