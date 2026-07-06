"""Config A on TSLA, NY session, 4 years of Alpaca 5-min RTH data.

Reuses the authoritative oanda_bot/backtest.py machinery (same RangeRecorder/
BreakoutDetector/simulate_trade/run_simulation code paths as --config-a).
Rules: NY session, first breakout only, HTF EMA100-aligned, 1.5xATR14 stop,
3:1 target. Cost model: proportional round-trip (bps of entry) since TSLA's
price spans ~$100-$1,100 across the unadjusted 3:1 split.
"""
import csv
import sys
from datetime import datetime, timedelta

import pytz

sys.path.insert(0, "/Users/derianj/projects/ms-orb/oanda_bot")
from backtest import (  # noqa: E402
    SimulationMetrics, run_simulation, group_candles_by_session_day,
    compute_atr_series, compute_ema_series,
)
from oanda_client import Candle  # noqa: E402
from config import SESSIONS, SessionName  # noqa: E402

CSV_PATH = "/Users/derianj/projects/orb-trading-bot/data/TSLA_20220101_20260125_5min_alpaca.csv"
NY = pytz.timezone("America/New_York")
COST_BPS_RT = 2.0  # round-trip cost as basis points of entry (spread+comm+slip)

# ── load candles (naive ET -> aware UTC) ─────────────────────
candles = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        t = NY.localize(datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S"))
        candles.append(Candle(
            time=t.astimezone(pytz.UTC),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=int(float(row["volume"])), complete=True,
        ))
candles.sort(key=lambda c: c.time)
print(f"TSLA candles: {len(candles)}  {candles[0].time.date()} .. {candles[-1].time.date()}")

# ── drop split-gap days (unadjusted 3:1 split 2022-08-25 etc.) ───────────────
# Exclude any session day whose open gaps >20% vs prior close, plus the next
# day (ATR-14 recovers within ~a day; EMA100 bias is small vs a 200% gap).
by_day: dict = {}
for c in candles:
    by_day.setdefault(c.time.astimezone(NY).date(), []).append(c)
days = sorted(by_day)
skip_days = set()
prev_close = None
for d in days:
    day_open = by_day[d][0].open
    if prev_close and abs(day_open / prev_close - 1) > 0.20:
        skip_days.add(d)
        i = days.index(d)
        if i + 1 < len(days):
            skip_days.add(days[i + 1])
    prev_close = by_day[d][-1].close
if skip_days:
    print(f"Skipping gap days (split/unadjusted): {sorted(skip_days)}")
candles = [c for c in candles if c.time.astimezone(NY).date() not in skip_days]

# ── indicators + session grouping (NY session only) ──────────
atr_s = compute_atr_series(candles)
ema_s = compute_ema_series(candles, 100)
session_cfgs = {SessionName.NY: SESSIONS[SessionName.NY]}
groups = group_candles_by_session_day(candles, list(session_cfgs.values()))
print(f"NY session-days: {len(groups)}")


def run(rr, stop_mode, first, htf, label):
    m = run_simulation(
        grouped_candles=groups, session_configs=session_cfgs,
        instrument="TSLA", rr_target=rr,
        vol_threshold=None, adx_threshold=None, adx_series={},
        max_trades_per_session=1 if first else 2,
        stop_mode=stop_mode, atr_stop_mult=1.5,
        first_breakout_only=first, htf_align=htf,
        atr_series=atr_s, ema_htf_series=ema_s, cost_price=0.0,
    )
    m.trades.sort(key=lambda t: t.entry_time)
    for mult_label, mult in (("net 1x", 1.0), ("net 2x", 2.0)):
        rs = []
        for t in m.trades:
            cost_r = (COST_BPS_RT / 10_000 * t.entry_price * mult) / abs(t.entry_price - t.stop_loss)
            rs.append(t.rr_gross if t.rr_gross else t.rr_achieved - 0)  # rr_achieved==gross (cost 0)
            rs[-1] = t.rr_achieved - cost_r
        n = len(rs)
        wins = sum(1 for r in rs if r > 0)
        gp = sum(r for r in rs if r > 0); gl = abs(sum(r for r in rs if r <= 0))
        pf = gp / gl if gl else float("inf")
        eq = pk = mdd = 0.0
        for r in rs:
            eq += r; pk = max(pk, eq); mdd = max(mdd, pk - eq)
        if mult == 1.0:
            # yearly breakdown at base cost
            yearly = {}
            for t, r in zip(m.trades, rs):
                yearly.setdefault(t.entry_time.year, []).append(r)
            ytxt = "  ".join(f"{y}:{sum(v):+.0f}R({len(v)})" for y, v in sorted(yearly.items()))
        print(f"{label:38s} {mult_label}: n={n:4d} WR={wins/max(n,1)*100:4.1f}% "
              f"expR={sum(rs)/max(n,1):+.3f} total={sum(rs):+7.1f}R PF={pf:4.2f} maxDD={mdd:5.1f}R")
    print(f"{'':38s} yearly(1x): {ytxt}")
    avg_cost = sum((COST_BPS_RT/10_000*t.entry_price)/abs(t.entry_price-t.stop_loss) for t in m.trades)/max(len(m.trades),1)
    print(f"{'':38s} mean cost_r={avg_cost:.3f}  exits="
          f"{m.exit_reason_counts()}")
    print()


print(f"\nCost model: {COST_BPS_RT}bp of entry round-trip (~$0.02 spread + IBKR comm on a $430 stock)\n")
run(3.0, "atr", True, True,  "CONFIG A (ATR 1.5x, 3:1, first, HTF)")
run(3.0, "atr", True, False, "  - without HTF filter")
run(3.0, "atr", False, False, "  - no filters (all breakouts, 2/day)")
run(5.0, "candle", False, False, "DEPLOYED ibkr style (candle SL, 5:1)")
run(5.0, "candle", True, True, "  - candle 5:1 + first + HTF")
