# Config-A Bot — Deployment (forward test)

Deploys the Tokyo bracket portfolio validated by `oanda_bot/backtest.py
--config-a`: XAU_USD + NAS100_USD, **Tokyo session only**, first breakout only
(no re-entry), HTF-aligned (close vs EMA100), **1.5×ATR stop**, **3:1 RR**.

This is a **forward-test deployment on a practice account**. Its purpose is to
measure, over 2–3 months, whether live fills match the backtest before any
prop-firm evaluation is purchased. Gate (per the July 2026 viability report):
forward PF net > ~1.2 and fills within ~0.05R of the cost model, judged per
instrument.

## Isolation guarantees

- Service: `ms-orb-configa` · dir `/opt/ms-orb-configa` · log
  `/var/log/ms-orb-configa.log` — disjoint from `ms-orb-oanda` and
  `ms-orb-filtered`. `tools/verify_deploy_safety.sh` checks this.
- **Requires its own OANDA practice sub-account** (create one in the OANDA
  dashboard). Sharing an account with the other bots would net positions
  across bots and contaminate the measurement. `.env.example` documents this.

## First-time setup

```bash
# 1. On the droplet (as root):
scp -r deploy root@146.190.76.14:/tmp/configa-deploy
ssh root@146.190.76.14 'bash /tmp/configa-deploy/setup.sh'

# 2. Create the env file on the droplet from .env.example
ssh root@146.190.76.14 'nano /opt/ms-orb-configa/.env'

# 3. From this directory on the local machine:
SERVER=146.190.76.14 bash deploy/deploy.sh
```

## Update deploy

```bash
SERVER=146.190.76.14 bash deploy/deploy.sh   # rsync + restart, .env untouched
```

## Monitoring the forward test

```bash
ssh root@146.190.76.14 'journalctl -u ms-orb-configa -f'          # live logs
ssh root@146.190.76.14 'sqlite3 /opt/ms-orb-configa/trades.db \
  "SELECT session,instrument,entry_time,entry_price,stop_loss,take_profit,
          exit_price,exit_reason FROM trades ORDER BY entry_time DESC LIMIT 10;"'
# Pull the DB monthly for fill-quality analysis vs the backtest cost model:
scp root@146.190.76.14:/opt/ms-orb-configa/trades.db ./forward_test_$(date +%Y%m).db
```

Sanity checks per trade (should hold by construction):
- `(take_profit - entry_price) / (entry_price - stop_loss)` = 3.0 for longs.
- Stop distance = 1.5 × ATR14 at the breakout candle — NOT the candle low/high.
- At most one trade per (instrument × day); Tokyo session only (entries
  18:30–21:30 NY).
