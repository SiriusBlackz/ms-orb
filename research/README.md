# Research scripts (measurement provenance)

One-off measurement drivers behind decisions recorded in the project memory
and `docs/VIABILITY_REPORT_2026-07.md`. Not production code; rerunnable.

| Script | Question it answered | Verdict (2026-07-06) |
|---|---|---|
| `tsla_config_a.py` | Does Config A (ATR 1.5x, 3:1, first, HTF) work on TSLA NY? Uses the authoritative `oanda_bot/backtest.py` machinery on 4yrs of Alpaca 5-min RTH data (`orb-trading-bot/data/TSLA_*_5min_alpaca.csv`). | YES — +119R/4yr net, PF 1.26, all years positive → ibkr_bot migrated (commit 2231668) |
| `tsla_overnight.py` | Overnight vs intraday split on TSLA; conditional variants. | Unconditional dead; conditional strong but decaying |
| `overnight_cross_symbol.py` | Does the conditional overnight effect generalize? 18 symbols via Alpaca daily bars. | Lives only in high-vol retail names; basket t=3.0 but monotonic decay → PARKED |
| `pull_funding.py` / `carry_measure.py` | Crypto perp funding carry (8 symbols, 4.5yrs Binance). | NO-GO — nets ~T-bill rates in current regime |

Data dependencies: Alpaca keys in `orb-trading-bot/.env` (equity pulls);
Binance fapi public (funding). XAU/NAS100 Config-A comparisons need no script —
run `python3 oanda_bot/backtest.py --config-a` and flag variants (see backtest.py --help).
