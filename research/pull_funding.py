"""Pull perpetual funding-rate history (Binance USD-M primary, Bybit fallback).

Binance fapi: GET /fapi/v1/fundingRate — public, 1000 rows/page, paginate by
startTime. Funding every 8h -> ~1095 rows/year/symbol.
Output: one CSV per symbol in ./funding/ (symbol, fundingTime(ms), fundingRate).
"""
import csv
import json
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).parent / "funding"
OUT.mkdir(exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
           "LINKUSDT", "AVAXUSDT", "LTCUSDT"]
START_MS = 1640995200000  # 2022-01-01 UTC — spans bear, chop, bull regimes

BINANCE = "https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={start}&limit=1000"
BYBIT = ("https://api.bybit.com/v5/market/funding/history?category=linear"
         "&symbol={sym}&startTime={start}&endTime={end}&limit=200")


def fetch(url):
    r = requests.get(url, headers={"User-Agent": "carry-research/0.1"}, timeout=20)
    r.raise_for_status()
    return r.json()


def pull_binance(sym):
    rows, start = [], START_MS
    while True:
        data = fetch(BINANCE.format(sym=sym, start=start))
        if not isinstance(data, list) or not data:
            break
        for d in data:
            rows.append((int(d["fundingTime"]), float(d["fundingRate"])))
        if len(data) < 1000:
            break
        start = rows[-1][0] + 1
        time.sleep(0.3)
    return rows


def pull_bybit(sym):
    rows, start = [], START_MS
    now = int(time.time() * 1000)
    while start < now:
        end = min(start + 200 * 8 * 3600 * 1000, now)
        data = fetch(BYBIT.format(sym=sym, start=start, end=end))
        lst = data.get("result", {}).get("list", [])
        for d in lst:
            rows.append((int(d["fundingRateTimestamp"]), float(d["fundingRate"])))
        start = end + 1
        time.sleep(0.3)
    return sorted(set(rows))


for sym in SYMBOLS:
    out_file = OUT / f"{sym}.csv"
    if out_file.exists():
        print(f"{sym}: cached")
        continue
    source = "binance"
    try:
        rows = pull_binance(sym)
    except Exception as e:
        print(f"{sym}: binance failed ({e}); trying bybit")
        source = "bybit"
        try:
            rows = pull_bybit(sym)
        except Exception as e2:
            print(f"{sym}: bybit failed too ({e2}); SKIP")
            continue
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fundingTime", "fundingRate", "source"])
        for t, r in rows:
            w.writerow([t, r, source])
    print(f"{sym}: {len(rows)} funding events ({source})")
