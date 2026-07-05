#!/usr/bin/env bash
# One-command first-time deploy of the MS-ORB Config-A bot.
#
# Does everything: server setup (user, dirs, venv, systemd unit), writes
# /opt/ms-orb-configa/.env from your values (no nano), rsyncs the code,
# installs deps, and starts the service. Idempotent — safe to re-run.
#
# Usage (prompts for anything not provided):
#   SERVER=146.190.76.14 bash deploy/bootstrap.sh
# or fully non-interactive:
#   SERVER=146.190.76.14 OANDA_API_KEY=xxx OANDA_ACCOUNT_ID=101-004-xxx-002 \
#     bash deploy/bootstrap.sh
#
# Uses a shared SSH connection, so password auth prompts at most once.
set -euo pipefail

SERVER="${1:-${SERVER:-}}"
SSH_USER="${SSH_USER:-root}"
APP_DIR="/opt/ms-orb-configa"
SERVICE_NAME="ms-orb-configa"
LOG_FILE="/var/log/ms-orb-configa.log"
BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$SERVER" ]; then
    echo "Error: No server IP provided. Usage: SERVER=1.2.3.4 bash deploy/bootstrap.sh"
    exit 1
fi

# ── Collect credentials (prompt only for what's missing) ─────────────────────
if [ -z "${OANDA_API_KEY:-}" ]; then
    read -r -s -p "OANDA API key: " OANDA_API_KEY; echo
fi
if [ -z "${OANDA_ACCOUNT_ID:-}" ]; then
    echo "Reminder: use a SEPARATE practice sub-account — not the one"
    echo "ms-orb-oanda or ms-orb-filtered trade on."
    read -r -p "OANDA practice account ID (e.g. 101-004-1234567-002): " OANDA_ACCOUNT_ID
fi
if [ -z "$OANDA_API_KEY" ] || [ -z "$OANDA_ACCOUNT_ID" ]; then
    echo "Error: API key and account ID are required." >&2
    exit 1
fi

# ── Shared SSH connection: one password prompt for everything ────────────────
CTRL="/tmp/ssh-configa-$$"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$CTRL" -o ControlPersist=120)
cleanup() { ssh -O exit -o ControlPath="$CTRL" "$SSH_USER@$SERVER" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== [1/5] Server setup (user, dirs, venv, service) ==="
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SERVER" bash -s <<REMOTE
set -euo pipefail
id msorbbot >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin msorbbot
mkdir -p $APP_DIR && chown msorbbot:msorbbot $APP_DIR
touch $LOG_FILE && chown msorbbot:msorbbot $LOG_FILE
test -x $APP_DIR/venv/bin/python3 || sudo -u msorbbot python3 -m venv $APP_DIR/venv
REMOTE

echo "=== [2/5] Writing $APP_DIR/.env ==="
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SERVER" \
    "cat > $APP_DIR/.env && chown msorbbot:msorbbot $APP_DIR/.env && chmod 600 $APP_DIR/.env" <<ENV
OANDA_API_KEY=$OANDA_API_KEY
OANDA_ACCOUNT_ID=$OANDA_ACCOUNT_ID
OANDA_ENVIRONMENT=practice

# Config-A strategy parameters — must match backtest.py --config-a
RISK_PER_TRADE=0.01
RR_TARGET=3.0
MAX_TRADES_PER_SESSION=1
ENABLED_SESSIONS=TOKYO
STOP_MODE=atr
ATR_STOP_MULT=1.5
ATR_PERIOD=14
HTF_ALIGN=true
HTF_EMA_PERIOD=100
FIRST_BREAKOUT_ONLY=true
INDICATOR_CANDLE_COUNT=1000

PRICE_POLL_INTERVAL=10
LOG_LEVEL=INFO
ENV

echo "=== [3/5] Syncing code ==="
rsync -az --delete \
    -e "ssh -o ControlMaster=auto -o ControlPath=$CTRL -o ControlPersist=120" \
    --exclude '.env' --exclude 'venv/' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'deploy/' --exclude 'trades.db' --exclude 'logs/' --exclude '.git' \
    "$BOT_DIR/" "$SSH_USER@$SERVER:$APP_DIR/"

echo "=== [4/5] Installing deps + systemd unit ==="
scp -o ControlMaster=auto -o ControlPath="$CTRL" -o ControlPersist=120 \
    "$BOT_DIR/deploy/$SERVICE_NAME.service" "$SSH_USER@$SERVER:/etc/systemd/system/"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SERVER" bash -s <<REMOTE
set -euo pipefail
chown -R msorbbot:msorbbot $APP_DIR
sudo -u msorbbot $APP_DIR/venv/bin/pip install -q -r $APP_DIR/requirements.txt
systemctl daemon-reload
systemctl enable $SERVICE_NAME.service
systemctl restart $SERVICE_NAME
REMOTE

echo "=== [5/5] Status ==="
sleep 2
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SERVER" \
    "systemctl status $SERVICE_NAME --no-pager -l | head -12; echo; tail -5 $LOG_FILE 2>/dev/null || true"

echo ""
echo "Done. Follow logs with:"
echo "  ssh $SSH_USER@$SERVER 'journalctl -u $SERVICE_NAME -f'"
