# MS-ORB Trading Bot

Live trading bots implementing the **Multi-Session Opening Range Breakout** (MS-ORB)
strategy across OANDA (FX/commodities/indices) and IBKR (US equities).

## Repo layout

- `oanda_bot/` — OANDA bot, multi-instrument (`XAU_USD`, `NAS100_USD`), Tokyo + NY sessions (London disabled 2026-07-21). **Deployed** as `ms-orb-oanda`.
- `oanda_bot_filtered/` — variant with extra filters. **Deployed** as `ms-orb-filtered` (at `/opt/ms-orb-oanda-filtered/`).
- `oanda_bot_orderflow/` — **CVD+ramp order flow variant**. Filters breakouts by range-period CVD alignment + pre-breakout volume ramp. **Not deployed** (local + git only).
- `ibkr_bot/` — IBKR bot, TSLA only, NY session only. **Deployed** as `ms-orb-ibkr`.
- `ms_orb_filter.pine` — TradingView Pine indicator (volume/ADX filter) used for manual chart overlay. Not consumed by the bots.

### Backtests (all in `oanda_bot/`)

- `backtest.py` + `backtest_filters.py` — **authoritative** strategy backtest. Live bots must stay synchronized with this.
- `backtest_orderflow.py` — Order flow metrics backtest (CVD, body ratio, volume skew, ramp, absorption, divergence). Source of the CVD+ramp filter.
- `backtest_fakeout.py` — Strategy 1: Fade failed breakouts (40% win rate but not profitable)
- `backtest_pullback.py` — Strategy 2: Breakout pullback entries (failed — tight stops)
- `backtest_momentum.py` — Strategy 3: Session momentum bias (failed — tight stops)
- `backtest_h1_orb.py` — H1 multi-timeframe ORB (failed — doesn't suit session structure)
- `backtest_crypto_orb.py` — Crypto ORB on BTC/ETH/SOL via Binance data (BTC profitable at +94R)
- `backtest_research.py` — **WR-vs-RR research harness.** Mines every breakout into a no-look-ahead feature vector + outcome under multiple stop widths/exits; stop-width × OOS greedy-filter frontier, no-target trailing runner, cross-sectional selection, walk-forward. Writes `cache/research_signals.csv`.
- `ml_classifier.py` — sklearn gradient-boosting trade-selection model on `research_signals.csv` (walk-forward CV, leakage-free, per-instrument costs). Verdict: features have weak signal (AUC ~0.55), no hidden alpha. Requires `scikit-learn`.

## Architecture (shared across all bots)

| File | Responsibility |
|---|---|
| `config.py` | Session times (NY-local), instrument config, env-driven trading params |
| `{oanda,ibkr}_client.py` | Broker API wrapper — `Candle`, `Price`, `Position`, `OrderResult` dataclasses |
| `strategy.py` | `RangeRecorder`, `BreakoutDetector`, `BreakoutSignal` (broker-agnostic) |
| `session_manager.py` | State machine per (instrument × session) |
| `trade_logger.py` | SQLite logging (`trades.db`, identical schema across bots) |
| `main.py` | `TradingBot` orchestrator, polling loop |

### Session state machine (`session_manager.py`)

```
WAITING_FOR_SESSION
        │  range_start ≤ now < range_end
        ▼
RECORDING_RANGE
        │  now ≥ range_end → finalize range
        ▼
WATCHING_FOR_BREAKOUT
        │  5-min close breaks range → signal
        ▼
PENDING_ORDER (limit order live)
        │  fill confirmed
        ▼
IN_TRADE
        │  SL/TP/time exit
        ▼
SESSION_CLOSED
        │  next day's range_start → reset()
        └──────────────→ (loops back to RECORDING_RANGE)
```

On startup, `SessionManager.recover_session_state()` back-fills the opening range
from historical M5 candles if the bot was started *after* `range_end` but still
within `session_close`, so the session can still trade that day.

## Strategy rules (canonical — match the backtest)

- **Opening range**: first 30 minutes of each session (6 × M5 candles). Membership is decided by each **candle's own NY-local start time** in `[range_start, range_end)` via `SessionManager._candle_in_range_window()` — *never* by wall-clock, so the pre-range candle that completes exactly at `range_start` (e.g. 17:55–18:00 for Tokyo) cannot leak in.
- **Entry trigger**: a 5-minute candle *closes* above `range_high` (LONG) or below `range_low` (SHORT)
- **Entry price**: `candle.close` of the breakout candle. Placed as a **LIMIT** order (not market), so a shallow breakout sits just barely above the range edge.
- **Stop loss**: `candle.low` (LONG) / `candle.high` (SHORT) — **breakout-candle-based, not range-based**. Because the breakout candle's wick usually dips back into the range before closing out of it, **SL typically lands inside the range box**. This is by design and matches `backtest.py`.
- **Take profit**: `entry ± risk × RR_TARGET` (default `RR_TARGET=5.0`)
- **Risk**: 1% of account balance per trade (`RISK_PER_TRADE=0.01`)
- **Max trades per session**: 1 (`MAX_TRADES_PER_SESSION=1`, default since 2026-07-21 — the second entry was gross-flat but cost ~−86R/yr net of spread over 12mo; first-only = −45R vs −131R net, Tokyo+NY-only first-only = **+50.7R net, PF 1.06**)
- **Re-entry (disabled by default via the cap of 1, machinery retained)**: allowed ONLY after a stop-out (then same-direction re-break, or opposite-direction break), and only while under the trade cap. A **target or time exit closes the session** for that instrument — no re-entry. `handle_trade_exit()`/`_close_trade()` gate both the state transition and `reset_direction()` on `reason == "stop"` (matches `backtest.py`). Verified by `oanda_bot/test_session_manager.py`. Revert to two entries with env `MAX_TRADES_PER_SESSION=2`.
- **Sessions traded**: Tokyo + NY only. **London disabled 2026-07-21** (negative gross AND net over 12mo: −95R net even first-entry-only, driven by XAU London). Controlled by `ENABLED_SESSIONS` (config.py, env-overridable, default `TOKYO,NY`); filters `get_active_session()` only — `infer_session_from_time()` stays unfiltered so position adoption can still attribute a leftover London trade. Revert with env `ENABLED_SESSIONS=TOKYO,LONDON,NY`. `backtest.py` is unaffected (session choice via `--sessions`), but it DOES inherit the cap default — plain runs are now single-entry; use env `MAX_TRADES_PER_SESSION=2` to backtest the old two-entry rule.
- **Time-based auto-close**: 5 minutes before session end

### Session windows (NY-local, `config.py:41-64`)

| Session | Range | Entry window | Session close |
|---|---|---|---|
| Tokyo | 18:00–18:30 | 18:30–21:30 | 02:54 (next day) |
| London (**not traded** since 2026-07-21) | 03:00–03:30 | 03:30–06:30 | 08:25 |
| NY | 09:30–10:00 | 10:00–13:00 | 15:55 |

## OANDA vs IBKR differences

| | OANDA | IBKR |
|---|---|---|
| API | `oandapyV20` REST | `ib_insync` |
| Instruments | `XAU_USD`, `NAS100_USD` | `TSLA` |
| Sessions | Tokyo, London, NY | NY only |
| Order type | LIMIT + attached SL/TP | Bracket (MKT + SL stop + TP limit) |
| Sizing | Pip-based units | Share-based |
| Sleep pattern | `time.sleep()` | `ib.sleep()` (keeps event loop alive) |
| Ports (IBKR) | — | 7497 TWS paper / 7496 live / 4002 GW paper / 4001 GW live |

## Deployment

Bots run on a **DigitalOcean droplet at `146.190.76.14`** (root login), all OANDA
on **demo/practice** accounts. `trades.db` and `logs/bot.log` on the local machine
are stale — SSH into the droplet to see live state. Supervisor is **systemd**.

| Service (systemd) | Remote dir | Source dir |
|---|---|---|
| `ms-orb-oanda` | `/opt/ms-orb-oanda/` | `oanda_bot/` |
| `ms-orb-filtered` | `/opt/ms-orb-oanda-filtered/` | `oanda_bot_filtered/` |
| `ms-orb-ibkr` | `/opt/ms-orb-ibkr/` | `ibkr_bot/` |
| `ms-orb-oanda-orderflow` | — (not deployed) | `oanda_bot_orderflow/` |

**Deploy = rsync, NOT git pull.** Each bot has `deploy/deploy.sh` that rsyncs the
local dir to its `/opt` target and `systemctl restart`s its service (`.git`, `.env`,
`trades.db`, `logs/` are excluded). A GitHub push does **not** update the droplet.

```bash
# from local machine, per bot:
SERVER=146.190.76.14 bash oanda_bot/deploy/deploy.sh
ssh root@146.190.76.14 'journalctl -u ms-orb-oanda -f'   # tail logs
```

> `oanda_bot_filtered/deploy/` targets the filtered deployment separately:
> `/opt/ms-orb-oanda-filtered`, service `ms-orb-filtered`, and log
> `/var/log/ms-orb-filtered.log`.

> Claude can't SSH to the droplet non-interactively (its key isn't in root's
> `authorized_keys`); the droplet accepts password auth, so the user runs droplet
> commands via the `!` prefix.

## Debugging tips

- **Check actual placed orders**: `sqlite3 oanda_bot/trades.db "SELECT session, instrument, entry_time, entry_price, stop_loss, take_profit, opening_range_high, opening_range_low FROM trades ORDER BY entry_time DESC LIMIT 10;"` on the droplet.
- **Verify RR_TARGET**: `(take_profit - entry_price) / (entry_price - stop_loss)` should equal `RR_TARGET` (default 5.0).
- **Verify entry above range**: for longs, `entry_price > opening_range_high`; for shorts, `entry_price < opening_range_low`. Because entries are shallow (often only 1–2 pts outside the range), they can *look* like they're in-range on a chart even when they technically aren't.
- **Session state dump**: every 5 minutes `main.py` logs all session states + ranges via `_log_status()`. Grep the log for `=== Status` lines.

## Research findings (win-rate vs reward-risk)

Established via `backtest_research.py` + `ml_classifier.py` (12mo, OOS-validated, cost-aware):

- **50% WR at ≥3RR is infeasible** on liquid markets — it equals profit factor 3.0, which doesn't exist out-of-sample without overfitting. Proven across 6 approaches + literature + a crypto cross-check. WR and RR trade off mechanically (`WR × RR ≈ PF`, and robust PF caps ~1.1–1.6).
- **WR-vs-RR curve** (XAU/NAS baseline): 1:1 → ~53% WR, 2:1 → ~35%, 3:1 → ~28%, 5:1 → ~23%. 50%+ WR is only reachable at ~1:1.
- **Best honest ≥3RR config**: XAU/NAS · Tokyo · first-breakout · HTF-trend aligned · **1.5×ATR stop** · 3:1 → **~35% WR, PF ~1.60** (gross). Or a **no-target 2×ATR trailing runner** (Tokyo+NY, HTF, RVOL≥1.0) → realized RR ≈ 3.0, **PF ~1.22 net of costs**.
- **Costs matter most**: tight candle stops die to spread; intraday **FX majors are unprofitable** (spread ≈ 16–27% of a tight ATR stop) — trade **indices/metals** only.
- **ML**: gradient boosting on the feature set tops out at **AUC 0.55** and doesn't beat the simple manual rules — no hidden alpha in M5 OHLCV+tick-volume. Beating ~PF 1.6 needs richer data (order book, cross-asset, alt-data), not more modeling.

See memory `project_wr3r_research.md`, `project_ml_classifier.md` for detail.
