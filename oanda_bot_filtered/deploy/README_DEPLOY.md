# MS-ORB Filtered OANDA Bot — Deployment Guide

## Architecture

```
Local:  oanda_bot_filtered/
Remote: /opt/ms-orb-oanda-filtered/
Service: ms-orb-filtered
User:    msorbbot
Logs:    /var/log/ms-orb-filtered.log
Deploy:  rsync over SSH, then restart the filtered service
```

## First-Time Server Setup

### 1. SSH into your droplet

```bash
ssh root@YOUR_SERVER_IP
```

### 2. Copy deploy files and run setup

From your **local machine**:

```bash
cd ~/projects/ms-orb/oanda_bot_filtered/deploy
scp setup.sh ms-orb-filtered.service root@YOUR_SERVER_IP:/tmp/
ssh root@YOUR_SERVER_IP 'cd /tmp && bash setup.sh'
```

This will:
- Install Python 3, pip, venv, sqlite3
- Create `msorbbot` service user (non-root, no login)
- Create `/opt/ms-orb-oanda-filtered/` directory
- Set up Python virtual environment
- Install and enable the systemd service

### 3. Create .env on the server

```bash
ssh root@YOUR_SERVER_IP
nano /opt/ms-orb-oanda-filtered/.env
```

Paste your configuration (use `.env.example` as reference):

```env
OANDA_API_KEY=your-api-key-here
OANDA_ACCOUNT_ID=your-account-id-here
OANDA_ENVIRONMENT=practice
RISK_PER_TRADE=0.01
RR_TARGET=5.0
MAX_TRADES_PER_SESSION=2
PRICE_POLL_INTERVAL=10
LOG_LEVEL=INFO
```

Then fix ownership:

```bash
chown msorbbot:msorbbot /opt/ms-orb-oanda-filtered/.env
chmod 600 /opt/ms-orb-oanda-filtered/.env
```

### 4. Deploy the code

From your **local machine**:

```bash
SERVER=YOUR_SERVER_IP bash deploy.sh
```

## Deploying Updates

After making code changes locally:

```bash
cd ~/projects/ms-orb/oanda_bot_filtered/deploy
SERVER=YOUR_SERVER_IP bash deploy.sh
```

This rsyncs the code, installs any new dependencies, and restarts the service.

## Managing the Bot

### Check status

```bash
ssh root@YOUR_SERVER_IP 'systemctl status ms-orb-filtered'
```

### View live logs

```bash
# Systemd journal (structured, recommended)
ssh root@YOUR_SERVER_IP 'journalctl -u ms-orb-filtered -f'

# Or the log file directly
ssh root@YOUR_SERVER_IP 'tail -f /var/log/ms-orb-filtered.log'

# Last 100 lines
ssh root@YOUR_SERVER_IP 'journalctl -u ms-orb-filtered -n 100 --no-pager'
```

### Stop the bot

```bash
ssh root@YOUR_SERVER_IP 'systemctl stop ms-orb-filtered'
```

### Start the bot

```bash
ssh root@YOUR_SERVER_IP 'systemctl start ms-orb-filtered'
```

### Restart the bot

```bash
ssh root@YOUR_SERVER_IP 'systemctl restart ms-orb-filtered'
```

### Disable auto-start on boot

```bash
ssh root@YOUR_SERVER_IP 'systemctl disable ms-orb-filtered'
```

## Checking the Trade Database

```bash
ssh root@YOUR_SERVER_IP 'sqlite3 /opt/ms-orb-oanda-filtered/trades.db "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;"'
```

## Troubleshooting

**Bot won't start — check logs first:**
```bash
journalctl -u ms-orb-filtered -n 50 --no-pager
```

**"OANDA_API_KEY is not set" error:**
```bash
# Verify .env exists and is readable
ls -la /opt/ms-orb-oanda-filtered/.env
cat /opt/ms-orb-oanda-filtered/.env
```

**Service keeps restarting:**
The service auto-restarts on failure with a 30s delay. Check logs for the root cause.

**Permission errors:**
```bash
chown -R msorbbot:msorbbot /opt/ms-orb-oanda-filtered
chown msorbbot:msorbbot /var/log/ms-orb-filtered.log
```
