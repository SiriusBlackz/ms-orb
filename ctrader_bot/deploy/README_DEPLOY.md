# cTrader Bot — Setup & Deployment

The Config-A strategy (XAU + NAS100, Tokyo only, first breakout, HTF-aligned,
1.5×ATR stop, 3:1) executed via the **cTrader Open API** — the prop-firm port
for FTMO (cTrader) and FundingPips. Strategy files are identical to
`oanda_bot_configa/`; only the client layer differs (`ctrader_client.py`,
JSON over WebSocket, no SDK).

## Why this exists

No prop firm speaks OANDA's REST API. This bot is step 3 of the viability
plan (`docs/VIABILITY_REPORT_2026-07.md`): build and demo-prove the cTrader
execution path *during* the OANDA forward test, so an evaluation can be
purchased the week the forward-test gate passes.

## One-time setup (interactive, ~15 minutes)

1. **cTrader ID**: create at [id.ctrader.com](https://id.ctrader.com).
2. **Demo account**: open any cTrader broker demo (IC Markets demo is free
   and instant) — the Open API works with every cTrader broker. Later, the
   FTMO/FundingPips account slots in by changing `CTRADER_ACCOUNT_ID` and
   `SYMBOL_MAP` only.
3. **Register an app** at [openapi.ctrader.com](https://openapi.ctrader.com):
   add redirect URI `http://localhost/`, copy the client ID + secret.
4. **Tokens**: put ID+secret in `.env`, run `python3 get_token.py`, follow
   the prompts (browser grant → paste code), paste the printed
   `CTRADER_ACCESS_TOKEN` / `CTRADER_REFRESH_TOKEN` / `CTRADER_ACCOUNT_ID`
   into `.env`. The bot refreshes tokens automatically afterwards and
   persists rotations to `ctrader_tokens.json` — **the interactive step
   never repeats** unless access is revoked.
5. **Symbols**: set `SYMBOL_MAP` to the broker's names (check the cTrader
   platform UI). The bot fails loudly at startup if a name doesn't resolve,
   and logs `Symbol resolved: XAU_USD -> XAUUSD (id=..., digits=...)` when it does.

## Local smoke test

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python3 main.py --dry-run          # connects, resolves symbols, no orders
python3 test_session_manager.py             # hermetic strategy tests
python3 test_ctrader_client.py              # hermetic client-adapter tests
```

## Droplet deployment

Service `ms-orb-ctrader` · dir `/opt/ms-orb-ctrader` · log
`/var/log/ms-orb-ctrader.log` — disjoint from all other bot services.

```bash
# once:
scp -r deploy root@146.190.76.14:/tmp/ctrader-deploy && ssh root@146.190.76.14 'bash /tmp/ctrader-deploy/setup.sh'
# .env on the droplet (copy your locally-tested one):
scp .env root@146.190.76.14:/opt/ms-orb-ctrader/.env
# each deploy:
SERVER=146.190.76.14 bash deploy/deploy.sh
```

## Prop-firm notes (from the July 2026 research)

- **FTMO**: buy the **Swing** account type (no news/weekend restrictions —
  Tokyo overnight holds need it). cTrader is a supported platform. Own-built
  bots explicitly allowed; keep git history as provenance.
- **FundingPips**: cTrader costs ~$20 extra at checkout. Their algo policy is
  proof-of-ownership — be ready to show source/git history on request.
- Re-verify each firm's rules the week of purchase; CFD prop terms drift.
- **Confirm Open API access works on the firm's cTrader environment before
  paying** — prop firms run white-label servers; test with their free trial.
