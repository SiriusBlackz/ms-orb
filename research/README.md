# Research scripts (measurement provenance)

One-off measurement drivers behind decisions recorded in the project memory
and `docs/VIABILITY_REPORT_2026-07.md`. Not production code; rerunnable.

| Script | Question it answered | Verdict (2026-07-06) |
|---|---|---|
| `tsla_config_a.py` | Does Config A (ATR 1.5x, 3:1, first, HTF) work on TSLA NY? Uses the authoritative `oanda_bot/backtest.py` machinery on 4yrs of Alpaca 5-min RTH data (`orb-trading-bot/data/TSLA_*_5min_alpaca.csv`). | YES — +119R/4yr net, PF 1.26, all years positive → ibkr_bot migrated (commit 2231668) |
| `tsla_overnight.py` | Overnight vs intraday split on TSLA; conditional variants. | Unconditional dead; conditional strong but decaying |
| `overnight_cross_symbol.py` | Does the conditional overnight effect generalize? 18 symbols via Alpaca daily bars. | Lives only in high-vol retail names; basket t=3.0 but monotonic decay → PARKED |
| `seasonality_2yr.py` | Config A on NAS100 / XAU / WTI (WTICO_USD), 24mo, all 3 sessions: best ticker, weekday (Tokyo-calendar, keyed on entry time), month/season. | 2026-10-21: NAS Tokyo only positive leg (+53R, PF 1.21); XAU Tokyo flat; WTI dead everywhere (cost 0.15–0.44R/trade). Tokyo Mon best (+52R pooled, t=+2.0), Thu/Fri negative (−57/−73R). July negative both years; Jul–Sep and Dec worst. |
| `walkforward_filters.py` + `mc_ftmo.py` | Do the regime filters (cost-in-R, D1 ADX, skip July, Mon–Wed, NAS-only) survive when chosen only on prior data? 4.2yrs Config-A Tokyo, half-year anchored walk-forward, permutation tests; FTMO Monte Carlo on the survivor. Outputs in `walkforward_2026-11-17/`. | 2026-11-17: filter COMBO FAILS out of sample (−24R vs +11R unfiltered, p=0.90). cost-in-R, ADX, Mon–Wed fail individually. Only "drop gold" survives: NAS100 Tokyo +86R since 2024 (PF 1.24, t=2.1, maxDD 20R; p=0.002 vs random), +67R full sample (t=1.5). Skip-July suggestive (p=0.02, 4 Julys). XAU Tokyo −113R over 4yrs (t=−2.3). |
| `pull_funding.py` / `carry_measure.py` | Crypto perp funding carry (8 symbols, 4.5yrs Binance). | NO-GO — nets ~T-bill rates in current regime |

Data dependencies: Alpaca keys in `orb-trading-bot/.env` (equity pulls);
Binance fapi public (funding). XAU/NAS100 Config-A comparisons need no script —
run `python3 oanda_bot/backtest.py --config-a` and flag variants (see backtest.py --help).
