# Execution System

This repo is the current highest-leverage execution target because it combines
live trading bots, research harnesses, deployment scripts, and active ORB
strategy work. The operating rule is simple: each session should leave behind a
small shipped increment with an explicit verifier.

## Operating Loop

1. Pick the highest-leverage project from current evidence, not preference.
2. Define one increment that removes a real blocker or creates a reusable tool.
3. Keep the edit scoped to the affected surface.
4. Run a verifier that proves the increment's key invariant.
5. Record the increment, evidence, and next runnable step here.

## Current Project

`ms-orb`

Why this project:
- It contains deployed OANDA and IBKR trading bots.
- It has active MS-ORB research variants and backtests.
- It has deployment scripts where small mistakes can affect live services.
- It has enough local testable surface to make verified progress without
  needing broker or droplet access for every increment.

## Increment Ledger

### 2026-06-19: Filtered OANDA Deploy Safety

Problem:
`oanda_bot_filtered/deploy/` was copied from the base OANDA bot and targeted the
base service paths. Running the filtered deploy as written could overwrite or
restart the base OANDA bot.

Shipped:
- Retargeted filtered deploy to `/opt/ms-orb-oanda-filtered`.
- Retargeted filtered systemd service to `ms-orb-filtered`.
- Retargeted filtered logs to `/var/log/ms-orb-filtered.log`.
- Renamed the filtered service file to `ms-orb-filtered.service`.
- Updated the filtered deployment guide.
- Added `tools/verify_deploy_safety.sh` as the repeatable local verifier.

Verifier:

```bash
bash tools/verify_deploy_safety.sh
```

Last local result:
Passed on 2026-06-19.

Evidence required:
- `deploy.sh` and `setup.sh` parse with `bash -n`.
- `oanda_bot_filtered/deploy/` contains no base-service operations such as
  `systemctl restart ms-orb-oanda`.
- The filtered service points at the filtered app dir, filtered `.env`, and
  filtered log file.

### 2026-06-19: Strategy Re-entry Parity Verifier

Problem:
The canonical MS-ORB rule says live bots must match `oanda_bot/backtest.py`:
re-entry is allowed only after a stop-out. A target or time exit closes that
session for the instrument. Existing tests covered this behavior, but there was
no single repo-level command that proved the parity invariant.

Shipped:
- Added `tools/verify_strategy_parity.sh`.
- The verifier runs the base and orderflow hermetic `SessionManager` tests.
- The verifier also checks the authoritative backtest stop-only re-entry gate
  and the matching live session-manager gates.

Verifier:

```bash
bash tools/verify_strategy_parity.sh
```

Last local result:
Passed on 2026-06-19.

Evidence required:
- Base `oanda_bot/test_session_manager.py` passes.
- Orderflow `oanda_bot_orderflow/test_session_manager.py` passes.
- `oanda_bot/backtest.py` still resets breakout direction only after stop-out.
- Base and orderflow `SessionManager.handle_trade_exit()` still return to
  `WATCHING_FOR_BREAKOUT` only for stop exits under the trade cap, otherwise
  `SESSION_CLOSED`.

### 2026-06-19: Offline Aggregate Verifier

Problem:
Individual verifiers are useful, but the execution system needs one local gate
that answers whether the current offline-safe checks pass before shipping,
committing, or deploying.

Shipped:
- Added `tools/verify_all.sh`.
- The aggregate runner calls the deploy safety verifier and strategy parity
  verifier.
- The runner intentionally excludes broker, OANDA, IBKR, droplet, and data-fetch
  checks so it remains reliable without credentials or network.

Verifier:

```bash
bash tools/verify_all.sh
```

Last local result:
Passed on 2026-06-19.

Evidence required:
- Filtered deploy remains isolated from the base OANDA service.
- Base and orderflow session-manager tests pass.
- Backtest/live stop-only re-entry parity checks pass.

### 2026-07-05: Config-A Forward-Test Bot

Problem:
The July 2026 viability analysis validated the Tokyo bracket portfolio
(`oanda_bot/backtest.py --config-a`: XAU+NAS Tokyo, first breakout, HTF-aligned,
1.5xATR stop, 3:1) but it had never traded forward, and the deployed bots run a
different config (5:1 candle-stop). The prop-firm decision is gated on a 2-3
month forward test that must not disturb `ms-orb-oanda` or `ms-orb-filtered`.

Shipped:
- `oanda_bot_configa/` — new bot variant: ENABLED_SESSIONS (Tokyo-only),
  first-breakout-only (rejected first signal or any exit closes the session),
  HTF EMA100 gate, 1.5xATR stop / 3:1 TP rebuilt at signal time from a
  1000-candle indicator fetch (`indicators.py` copied verbatim from the
  authoritative backtest). Indicator-fetch failure rejects the signal.
- All strategy parameters pinned explicitly in `.env.example`, which also
  documents the separate-practice-sub-account requirement.
- Isolated deploy: service `ms-orb-configa`, dir `/opt/ms-orb-configa`, log
  `/var/log/ms-orb-configa.log`.
- Extended `tools/verify_deploy_safety.sh` (Config-A isolation checks) and
  `tools/verify_strategy_parity.sh` (runs the Config-A hermetic tests).

Verifier:

```bash
bash tools/verify_all.sh
```

Last local result:
Passed on 2026-07-05 (10/10 Config-A tests; deploy isolation checks green).

Evidence required:
- Config-A tests prove: Tokyo-only contexts, HTF-misaligned first breakout ->
  SESSION_CLOSED with no trade, ATR stop = 1.5x Wilder ATR14 with 3:1 TP,
  fetch failure -> no trade, stop exit -> SESSION_CLOSED (no re-entry).
- Config-A deploy references no base or filtered service targets.

## Next Runnable Increment

Add an opt-in deployment preflight command for remote checks that should not run
inside `verify_all.sh`, for example:

```bash
bash tools/preflight_remote_deploy.sh <server-ip> ms-orb-filtered
```

It should check service names, remote app dirs, `.env` presence, and systemd
unit installation without restarting anything. Keep it read-only by default.
