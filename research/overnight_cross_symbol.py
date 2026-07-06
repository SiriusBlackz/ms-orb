"""Cross-symbol validation of the conditional overnight effect.

Long overnight after an UP intraday session, short overnight after a DOWN
one. Daily OHLC (split-adjusted) is sufficient: overnight = prior close ->
open, intraday = open -> close. Tests whether the TSLA finding generalizes,
and whether magnitude scales with volatility. IEX-feed caveat: free-tier
closes can differ slightly from official SIP auction prices — good enough
for a kill/continue verdict, not for final sizing.
"""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import dotenv_values

env = dotenv_values("/Users/derianj/projects/orb-trading-bot/.env")
HEADERS = {
    "APCA-API-KEY-ID": env["ALPACA_API_KEY"],
    "APCA-API-SECRET-KEY": env["ALPACA_SECRET_KEY"],
}
URL = "https://data.alpaca.markets/v2/stocks/bars"

SYMBOLS = ["TSLA", "NVDA", "AMD", "COIN", "MSTR", "PLTR", "SMCI", "RIOT",
           "META", "NFLX", "AMZN", "AAPL", "MSFT", "GOOGL",
           "SPY", "QQQ", "IWM", "ARKK"]
CACHE = Path(__file__).parent / "daily_bars.json"

if CACHE.exists():
    bars = json.loads(CACHE.read_text())
else:
    bars = {}
    for chunk in (SYMBOLS[:9], SYMBOLS[9:]):
        params = {
            "symbols": ",".join(chunk), "timeframe": "1Day",
            "start": "2022-01-01", "end": "2026-07-04",
            "adjustment": "split", "feed": "iex", "limit": 10000,
        }
        page = None
        while True:
            if page:
                params["page_token"] = page
            r = requests.get(URL, headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            for sym, rows in (data.get("bars") or {}).items():
                bars.setdefault(sym, []).extend(rows)
            page = data.get("next_page_token")
            if not page:
                break
    CACHE.write_text(json.dumps(bars))

print(f"{'symbol':7s} {'n':>5s} {'ann.vol':>8s} {'L/S ann%':>9s} {'t':>6s} "
      f"{'2022':>7s} {'2023':>7s} {'2024':>7s} {'2025':>7s} {'26H1':>7s}")

summary = []
for sym in SYMBOLS:
    rows = sorted(bars.get(sym, []), key=lambda r: r["t"])
    if len(rows) < 500:
        print(f"{sym:7s}  insufficient data ({len(rows)})")
        continue
    rets, years = [], defaultdict(float)
    daily_moves = []
    for i in range(1, len(rows)):
        pc, o, c = rows[i-1]["c"], rows[i]["o"], rows[i]["c"]
        if not (pc and o and c):
            continue
        prev_intraday = math.log(rows[i-1]["c"] / rows[i-1]["o"])
        r_on = math.log(o / pc)
        daily_moves.append(math.log(c / pc))
        if abs(r_on) > 0.35:   # unadjusted split residue / bad print guard
            continue
        pl = r_on if prev_intraday > 0 else (-r_on if prev_intraday < 0 else 0)
        rets.append(pl)
        years[rows[i]["t"][:4]] += pl
    mu, sd = statistics.mean(rets), statistics.pstdev(rets)
    t = mu / (sd / math.sqrt(len(rets))) if sd else 0
    vol = statistics.pstdev(daily_moves) * math.sqrt(252)
    ycells = "".join(f"{years.get(y, 0)*100:>7.1f}" for y in
                     ("2022", "2023", "2024", "2025", "2026"))
    print(f"{sym:7s} {len(rets):5d} {vol*100:>7.0f}% {mu*252*100:>8.1f} {t:>6.2f}{ycells}")
    summary.append((sym, vol, mu * 252, t))

print("\nCorrelation of effect size with volatility:")
vols = [v for _, v, _, _ in summary]
anns = [a for _, _, a, _ in summary]
mv, ma = statistics.mean(vols), statistics.mean(anns)
cov = sum((v-mv)*(a-ma) for v, a in zip(vols, anns)) / len(vols)
corr = cov / (statistics.pstdev(vols) * statistics.pstdev(anns))
print(f"  corr(ann.vol, L/S annual return) = {corr:+.2f} across {len(summary)} symbols")
sig = [s for s, _, a, t in summary if t > 2]
print(f"  symbols with t>2: {sig}")
