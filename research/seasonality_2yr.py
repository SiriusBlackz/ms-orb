"""Config-A seasonality study: NAS100 vs XAU vs WTI over 24 months.

Questions:
  1. Best ticker (and best session per ticker) under Config-A rules
     (first breakout, HTF EMA100-aligned, 1.5xATR14 stop, 3:1, costed).
  2. Best / worst days of the week.
  3. Months or seasons where trading should be reduced or stopped.

Reuses the authoritative oanda_bot/backtest.py machinery verbatim. Weekday
and month are keyed on the ENTRY time (backtest.TradeResult.trade_date is
stamped from the exit candle, which rolls past midnight NY for Tokyo trades).
Tokyo trades are labelled by the Tokyo-calendar weekday (Sunday 18:00 NY =
Monday Tokyo open); London/NY by the NY-local date.

Usage:  python3 research/seasonality_2yr.py [--months 24] [--wti-cost 0.04]
"""
import argparse
import logging
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import pytz

sys.path.insert(0, "/Users/derianj/projects/ms-orb/oanda_bot")
from backtest import (  # noqa: E402
    SimulationMetrics, TradeResult, run_simulation, group_candles_by_session_day,
    compute_adx_series, compute_atr_series, compute_ema_series, fetch_candles_range,
)
from oanda_client import OANDAClient  # noqa: E402
from config import SESSIONS, SessionName  # noqa: E402

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NY_TZ = pytz.timezone("America/New_York")
TOKYO_TZ = pytz.timezone("Asia/Tokyo")


def session_day(t: TradeResult) -> date:
    """Calendar day the trade belongs to, keyed on entry time."""
    tz = TOKYO_TZ if t.session == SessionName.TOKYO else NY_TZ
    return t.entry_time.astimezone(tz).date()

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = {"Q1 Jan-Mar": (1, 2, 3), "Q2 Apr-Jun": (4, 5, 6),
           "Q3 Jul-Sep": (7, 8, 9), "Q4 Oct-Dec": (10, 11, 12)}


def stats(trades: list[TradeResult]) -> dict:
    n = len(trades)
    if n == 0:
        return dict(n=0, net=0.0, gross=0.0, wr=0.0, pf=0.0, exp=0.0, se=0.0,
                    dd=0.0, cost=0.0)
    rs = [t.rr_achieved for t in trades]
    net = sum(rs)
    gp = sum(r for r in rs if r > 0)
    gl = abs(sum(r for r in rs if r <= 0))
    mean = net / n
    var = sum((r - mean) ** 2 for r in rs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    peak = eq = dd = 0.0
    for t in sorted(trades, key=lambda t: t.entry_time):
        eq += t.rr_achieved
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return dict(
        n=n, net=net, gross=sum(t.rr_gross for t in trades),
        wr=sum(1 for r in rs if r > 0) / n,
        pf=(gp / gl) if gl > 0 else float("inf"),
        exp=mean, se=se, dd=dd,
        cost=sum(t.cost_r for t in trades) / n,
    )


def fmt_row(label: str, s: dict, width: int = 14) -> str:
    if s["n"] == 0:
        return f"{label:<{width}} │ {'—':>6}"
    tstat = s["exp"] / s["se"] if s["se"] > 0 else 0.0
    return (f"{label:<{width}} │ {s['n']:>5} │ {s['wr']*100:5.1f}% │ "
            f"{s['net']:+8.2f} │ {s['gross']:+8.2f} │ {s['exp']:+.3f} │ "
            f"{tstat:+5.2f} │ {s['pf']:5.2f} │ {s['dd']:6.1f} │ {s['cost']:.3f}")


HEADER = (f"{'':<14} │ {'n':>5} │ {'WR':>6} │ {'Net R':>8} │ {'Gross R':>8} │ "
          f"{'exp/t':>6} │ {'t':>5} │ {'PF':>5} │ {'MaxDD':>6} │ cost_r")


def print_table(title: str, rows: list[tuple[str, list[TradeResult]]]) -> None:
    print(f"\n{title}")
    print(HEADER)
    print("─" * len(HEADER))
    for label, trades in rows:
        print(fmt_row(label, stats(trades)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--instruments", default="NAS100_USD,XAU_USD,WTICO_USD")
    ap.add_argument("--wti-cost", type=float, default=0.04,
                    help="WTICO_USD round-trip cost in price units (default 0.04)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--days", default=None,
                    help="Only trade these weekdays, e.g. Mon,Tue,Wed (Tokyo-calendar for Tokyo)")
    ap.add_argument("--skip-months", default=None,
                    help="Calendar months to skip, e.g. 7 or 7,12")
    ap.add_argument("--timeline", action="store_true",
                    help="Print month-by-month equity timeline + cost sensitivity")
    args = ap.parse_args()
    keep_days = ({WEEKDAYS.index(d.strip()[:3].title()) for d in args.days.split(",")}
                 if args.days else None)
    skip_months = ({int(m) for m in args.skip_months.split(",")}
                   if args.skip_months else set())

    logging.basicConfig(level=logging.WARNING)
    costs = {"XAU_USD": 0.35, "NAS100_USD": 1.5, "WTICO_USD": args.wti_cost}
    instruments = [s.strip() for s in args.instruments.split(",")]
    sessions = [SessionName.TOKYO, SessionName.LONDON, SessionName.NY]
    session_configs = {s: SESSIONS[s] for s in sessions}

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.months * 30)
    print(f"Config-A seasonality study  {start_dt.date()} → {end_dt.date()}  "
          f"({args.months} months)")
    print(f"Rules: first breakout, HTF EMA100, 1.5xATR14 stop, 3:1, 1 trade/session")
    print(f"Costs (round-trip, price units): {costs}")
    if keep_days is not None or skip_months:
        print(f"FILTERS: days={sorted(WEEKDAYS[d] for d in keep_days) if keep_days else 'all'}  "
              f"skip_months={sorted(skip_months) or 'none'}  (post-hoc trade filter; "
              f"exact for 1-trade/session first-breakout rules)")

    client = OANDAClient()
    if not client.test_connection():
        sys.exit("Cannot connect to OANDA")

    # instrument -> session -> trades
    results: dict[str, dict[SessionName, list[TradeResult]]] = {}
    for inst in instruments:
        candles = fetch_candles_range(client, inst, start_dt, end_dt,
                                      use_cache=not args.no_cache)
        if not candles:
            print(f"!! no candles for {inst}")
            continue
        adx = compute_adx_series(candles)
        atr = compute_atr_series(candles)
        ema = compute_ema_series(candles, 100)
        groups = group_candles_by_session_day(candles, list(session_configs.values()))
        results[inst] = {}
        for sess in sessions:
            m = run_simulation(
                grouped_candles={k: v for k, v in groups.items() if k[0] == sess},
                session_configs={sess: SESSIONS[sess]},
                instrument=inst, rr_target=3.0,
                vol_threshold=None, adx_threshold=None, adx_series=adx,
                max_trades_per_session=1, stop_mode="atr", atr_stop_mult=1.5,
                first_breakout_only=True, htf_align=True,
                atr_series=atr, ema_htf_series=ema, cost_price=costs.get(inst, 0.0),
            )
            trades = sorted(m.trades, key=lambda t: t.entry_time)
            if keep_days is not None:
                trades = [t for t in trades if session_day(t).weekday() in keep_days]
            if skip_months:
                trades = [t for t in trades if session_day(t).month not in skip_months]
            results[inst][sess] = trades
        print(f"  {inst}: {len(candles)} candles, "
              f"{candles[0].time.date()} → {candles[-1].time.date()}, "
              f"trades " + ", ".join(f"{s.value}={len(results[inst][s])}" for s in sessions))

    # ── 1. Instrument × session ─────────────────────────────────────────────
    rows = []
    for inst in results:
        for sess in sessions:
            rows.append((f"{inst[:6]} {sess.value}", results[inst][sess]))
    print_table("1. INSTRUMENT × SESSION (Config-A rules on every session)", rows)

    tokyo_rows = [(inst, results[inst][SessionName.TOKYO]) for inst in results]
    tokyo_rows.append(("ALL Tokyo", [t for inst in results for t in results[inst][SessionName.TOKYO]]))
    print_table("1b. CONFIG-A HEADLINE (Tokyo only)", tokyo_rows)

    # ── 2. Day of week ─────────────────────────────────────────────────────
    for inst in results:
        for sess in sessions:
            trades = results[inst][sess]
            if len(trades) < 60:
                continue
            by_dow = defaultdict(list)
            for t in trades:
                by_dow[session_day(t).weekday()].append(t)
            rows = [(WEEKDAYS[d], by_dow[d]) for d in range(7) if by_dow[d]]
            note = "  (Tokyo-calendar weekday)" if sess == SessionName.TOKYO else "  (NY-local weekday)"
            print_table(f"2. DAY OF WEEK — {inst} {sess.value}{note}", rows)

    # Pooled Tokyo weekday across instruments
    by_dow = defaultdict(list)
    for inst in results:
        for t in results[inst][SessionName.TOKYO]:
            by_dow[session_day(t).weekday()].append(t)
    print_table("2b. DAY OF WEEK — ALL instruments, Tokyo pooled",
                [(WEEKDAYS[d], by_dow[d]) for d in range(7) if by_dow[d]])

    # ── 3. Month of year (each year separately + pooled) ──────────────────
    for inst in results:
        for sess in sessions:
            trades = results[inst][sess]
            if len(trades) < 60:
                continue
            by_ym = defaultdict(list)
            years = sorted({session_day(t).year for t in trades})
            for t in trades:
                by_ym[(session_day(t).year, session_day(t).month)].append(t)
            print(f"\n3. MONTH OF YEAR — {inst} {sess.value}  (net R per calendar month)")
            hdr = f"{'Month':<6} │ " + " │ ".join(f"{y:>7}" for y in years) + f" │ {'Pooled':>7} │ {'n':>4} │ {'+yrs':>4}"
            print(hdr)
            print("─" * len(hdr))
            for mo in range(1, 13):
                cells, pooled, pos = [], [], 0
                for y in years:
                    tr = by_ym.get((y, mo), [])
                    if tr:
                        r = sum(t.rr_achieved for t in tr)
                        cells.append(f"{r:+7.1f}")
                        pooled.extend(tr)
                        pos += r > 0
                    else:
                        cells.append(f"{'·':>7}")
                tot = sum(t.rr_achieved for t in pooled)
                nyrs = sum(1 for y in years if by_ym.get((y, mo)))
                print(f"{MONTHS[mo-1]:<6} │ " + " │ ".join(cells)
                      + f" │ {tot:+7.1f} │ {len(pooled):>4} │ {pos}/{nyrs}")

    # Pooled Tokyo by season
    print("\n3b. SEASON — Tokyo, per instrument and pooled (net R)")
    hdr = f"{'Season':<12} │ " + " │ ".join(f"{i[:6]:>8}" for i in results) + f" │ {'ALL':>8} │ {'n':>5} │ {'exp/t':>6}"
    print(hdr)
    print("─" * len(hdr))
    for label, mos in SEASONS.items():
        cells, pooled = [], []
        for inst in results:
            tr = [t for t in results[inst][SessionName.TOKYO] if session_day(t).month in mos]
            cells.append(f"{sum(t.rr_achieved for t in tr):+8.1f}")
            pooled.extend(tr)
        s = stats(pooled)
        print(f"{label:<12} │ " + " │ ".join(cells) + f" │ {s['net']:+8.1f} │ {s['n']:>5} │ {s['exp']:+.3f}")

    # Pooled Tokyo month-of-year with both years
    print("\n3c. MONTH OF YEAR — Tokyo, all instruments pooled (net R)")
    by_ym = defaultdict(list)
    for inst in results:
        for t in results[inst][SessionName.TOKYO]:
            by_ym[(session_day(t).year, session_day(t).month)].append(t)
    years = sorted({y for y, _ in by_ym})
    hdr = f"{'Month':<6} │ " + " │ ".join(f"{y:>7}" for y in years) + f" │ {'Pooled':>7} │ {'n':>4} │ {'exp/t':>6} │ {'cost_r':>6}"
    print(hdr)
    print("─" * len(hdr))
    for mo in range(1, 13):
        cells, pooled = [], []
        for y in years:
            tr = by_ym.get((y, mo), [])
            cells.append(f"{sum(t.rr_achieved for t in tr):+7.1f}" if tr else f"{'·':>7}")
            pooled.extend(tr)
        s = stats(pooled)
        print(f"{MONTHS[mo-1]:<6} │ " + " │ ".join(cells)
              + f" │ {s['net']:+7.1f} │ {s['n']:>4} │ {s['exp']:+.3f} │ {s['cost']:.3f}")


    if args.timeline:
        for inst in results:
            for sess in sessions:
                trades = results[inst][sess]
                if len(trades) < 30:
                    continue
                print(f"\n4. EQUITY TIMELINE — {inst} {sess.value}")
                s_all = stats(trades)
                by_ym = defaultdict(list)
                for t in trades:
                    by_ym[(session_day(t).year, session_day(t).month)].append(t)
                eq = peak = 0.0
                print(f"{'Month':<8} │ {'n':>3} │ {'net R':>7} │ {'cum R':>7} │ {'DD':>6}")
                for ym in sorted(by_ym):
                    r = sum(t.rr_achieved for t in by_ym[ym])
                    eq += r
                    peak = max(peak, eq)
                    print(f"{ym[0]}-{ym[1]:02d}  │ {len(by_ym[ym]):>3} │ {r:+7.2f} │ {eq:+7.2f} │ {peak-eq:6.2f}")
                pos = sum(1 for ym in by_ym if sum(t.rr_achieved for t in by_ym[ym]) > 0)
                print(f"Positive months: {pos}/{len(by_ym)}")
                years = defaultdict(float)
                for t in trades:
                    years[session_day(t).year] += t.rr_achieved
                print("By year: " + "  ".join(f"{y}: {r:+.1f}R" for y, r in sorted(years.items())))
                tstat = s_all['exp'] / s_all['se'] if s_all['se'] else 0
                print(f"Net {s_all['net']:+.2f}R  gross {s_all['gross']:+.2f}R  n={s_all['n']}  "
                      f"WR {s_all['wr']*100:.1f}%  exp {s_all['exp']:+.3f}R/t (t={tstat:+.2f})  "
                      f"PF {s_all['pf']:.2f}  maxDD {s_all['dd']:.1f}R")
                print("Cost sensitivity:")
                for mult in (1.0, 1.5, 2.0):
                    rs = [t.rr_gross - mult * t.cost_r for t in trades]
                    gp = sum(r for r in rs if r > 0); gl = abs(sum(r for r in rs if r <= 0))
                    e = pk = dd = 0.0
                    for r in rs:
                        e += r; pk = max(pk, e); dd = max(dd, pk - e)
                    print(f"  {mult:.1f}x cost: net {sum(rs):+7.2f}R  PF {gp/gl if gl else 0:.2f}  maxDD {dd:.1f}R")


if __name__ == "__main__":
    main()
