# MS-ORB OANDA Bot (Order Flow Variant) — Deployment Guide

## Architecture

```
Local machine                     DigitalOcean Droplet (Ubuntu 22/24)
┌────────────────────┐  rsync    ┌──────────────────────────────────┐
│ oanda_bot_orderflow│ ───────>  │ /opt/ms-orb-oanda-orderflow/     │
│   deploy.sh        │           │   main.py, *.py, venv/           │
└────────────────────┘           │   .env (DEMO account secrets)    │
                                 │                                   │
                                 │ systemd: ms-orb-oanda-orderflow   │
                                 │   runs as: msorbbot user          │
                                 │   logs: /var/log/ms-orb-oanda-*   │
                                 │   db:   trades.db (separate)      │
                                 ├──────────────────────────────────┤
                                 │ /opt/ms-orb-oanda/  (base bot)   │
                                 │ systemd: ms-orb-oanda (LIVE)     │
                                 └──────────────────────────────────┘
```

**All three bots run independently.** Different OANDA demo accounts,
different systemd services, different trade databases. No interference.

Existing services on the droplet:
- `ms-orb-oanda` — base bot (demo)
- `ms-orb-oandafilter` — filtered variant (demo)
- `ms-orb-oanda-orderflow` — this bot (demo, new)

## First-Time Server Setup

### 1. Copy deploy files and run setup

From your **local machine**:

```bash
cd ~/projects/ms-orb/oanda_bot_orderflow/deploy
scp setup.sh ms-orb-oanda-orderflow.service root@YOUR_SERVER_IP:/tmp/
ssh root@YOUR_SERVER_IP 'cd /tmp && bash setup.sh'
```

### 2. Create .env on the server

```bash
ssh root@YOUR_SERVER_IP
nano /opt/ms-orb-oanda-orderflow/.env
```

Paste your DEMO account configuration:

```env
OANDA_API_KEY=your-demo-api-key
OANDA_ACCOUNT_ID=your-demo-account-id
OANDA_ENVIRONMENT=practice
RISK_PER_TRADE=0.01
RR_TARGET=5.0
MAX_TRADES_PER_SESSION=2
PRICE_POLL_INTERVAL=10
LOG_LEVEL=INFO

# Order flow filter settings
USE_ORDER_FLOW_FILTER=true
REQUIRE_CVD_ALIGNED=true
MIN_VOLUME_RAMP=0.0
```

Then fix ownership:

```bash
chown msorbbot:msorbbot /opt/ms-orb-oanda-orderflow/.env
chmod 600 /opt/ms-orb-oanda-orderflow/.env
```

### 3. Deploy the code

From your **local machine**:

```bash
SERVER=YOUR_SERVER_IP bash deploy.sh
```

## Deploying Updates

```bash
cd ~/projects/ms-orb/oanda_bot_orderflow/deploy
SERVER=YOUR_SERVER_IP bash deploy.sh
```

## Managing the Bot

```bash
# Status
ssh root@YOUR_SERVER_IP 'systemctl status ms-orb-oanda-orderflow'

# Live logs
ssh root@YOUR_SERVER_IP 'journalctl -u ms-orb-oanda-orderflow -f'

# Last 100 lines
ssh root@YOUR_SERVER_IP 'journalctl -u ms-orb-oanda-orderflow -n 100 --no-pager'

# Stop / Start / Restart
ssh root@YOUR_SERVER_IP 'systemctl stop ms-orb-oanda-orderflow'
ssh root@YOUR_SERVER_IP 'systemctl start ms-orb-oanda-orderflow'
ssh root@YOUR_SERVER_IP 'systemctl restart ms-orb-oanda-orderflow'
```

## Checking Trade Database

```bash
ssh root@YOUR_SERVER_IP 'sqlite3 /opt/ms-orb-oanda-orderflow/trades.db "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;"'
```

## Comparing All Three Bots

```bash
# Side-by-side status
ssh root@YOUR_SERVER_IP 'for s in ms-orb-oanda ms-orb-oandafilter ms-orb-oanda-orderflow; do echo "=== $s ==="; systemctl status $s --no-pager -l 2>/dev/null || echo "not installed"; echo; done'

# Compare trade counts
ssh root@YOUR_SERVER_IP 'for d in ms-orb-oanda ms-orb-oandafilter ms-orb-oanda-orderflow; do echo "=== $d ==="; sqlite3 /opt/$d/trades.db "SELECT COUNT(*) as total, SUM(CASE WHEN exit_reason='\''target'\'' THEN 1 ELSE 0 END) as wins FROM trades;" 2>/dev/null || echo "no db yet"; done'
```

## Order Flow Filter Tuning

The filter is controlled by env vars — no code change needed to adjust:

| Env Var | Default | Description |
|---|---|---|
| `USE_ORDER_FLOW_FILTER` | `true` | Master toggle. Set `false` to disable entirely |
| `REQUIRE_CVD_ALIGNED` | `true` | Require range-period CVD to confirm direction |
| `MIN_VOLUME_RAMP` | `0.0` | Minimum volume ramp (0.0 = non-negative passes) |

After changing `.env`, restart the service:

```bash
ssh root@YOUR_SERVER_IP 'systemctl restart ms-orb-oanda-orderflow'
```

## Troubleshooting

**"ORDER FLOW REJECT" in logs:**
This is expected. The filter is working — it's rejecting breakouts that
don't have confirming volume. Grep for `ORDER FLOW PASS` to see accepted signals.

```bash
ssh root@YOUR_SERVER_IP 'journalctl -u ms-orb-oanda-orderflow --no-pager | grep "ORDER FLOW"'
```
