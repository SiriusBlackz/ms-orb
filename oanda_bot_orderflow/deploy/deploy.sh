#!/usr/bin/env bash
# Deploy (or update) MS-ORB OANDA bot (Order Flow variant) to DigitalOcean droplet.
# Usage: SERVER=1.2.3.4 bash deploy.sh
#    or: bash deploy.sh 1.2.3.4
set -euo pipefail

SERVER="${1:-${SERVER:-}}"
SSH_USER="${SSH_USER:-root}"
APP_DIR="/opt/ms-orb-oanda-orderflow"
SERVICE_NAME="ms-orb-oanda-orderflow"
BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$SERVER" ]; then
    echo "Error: No server IP provided."
    echo "Usage: SERVER=1.2.3.4 bash deploy.sh"
    echo "   or: bash deploy.sh 1.2.3.4"
    exit 1
fi

echo "=== Deploying MS-ORB OANDA Bot (Order Flow) ==="
echo "Server:  $SSH_USER@$SERVER"
echo "Source:  $BOT_DIR"
echo "Remote:  $APP_DIR"
echo "Service: $SERVICE_NAME"
echo ""

# Sync bot code (exclude .env, __pycache__, deploy scripts, local DB)
echo "[1/4] Syncing code to server..."
rsync -avz --delete \
    --exclude '.env' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'deploy/' \
    --exclude 'trades.db' \
    --exclude 'logs/' \
    --exclude '.git' \
    --exclude '.DS_Store' \
    --exclude '.claude' \
    "$BOT_DIR/" "$SSH_USER@$SERVER:$APP_DIR/"

# Fix ownership
echo "[2/4] Fixing file ownership..."
ssh "$SSH_USER@$SERVER" "chown -R msorbbot:msorbbot $APP_DIR"

# Install/update Python dependencies
echo "[3/4] Installing Python dependencies..."
ssh "$SSH_USER@$SERVER" "sudo -u msorbbot $APP_DIR/venv/bin/pip install -q -r $APP_DIR/requirements.txt"

# Check if .env exists on server
if ! ssh "$SSH_USER@$SERVER" "test -f $APP_DIR/.env"; then
    echo ""
    echo "WARNING: No .env file found on server!"
    echo "Create it now:  ssh $SSH_USER@$SERVER 'nano $APP_DIR/.env'"
    echo "(use .env.example as template — use DEMO account credentials)"
    echo ""
    echo "Skipping service restart until .env is configured."
    exit 0
fi

# Restart service
echo "[4/4] Restarting service..."
ssh "$SSH_USER@$SERVER" "systemctl daemon-reload && systemctl restart $SERVICE_NAME"

# Quick status check
echo ""
echo "=== Deploy complete! Checking status... ==="
ssh "$SSH_USER@$SERVER" "sleep 2 && systemctl status $SERVICE_NAME --no-pager -l" || true

echo ""
echo "Useful commands:"
echo "  Logs:    ssh $SSH_USER@$SERVER 'journalctl -u $SERVICE_NAME -f'"
echo "  Status:  ssh $SSH_USER@$SERVER 'systemctl status $SERVICE_NAME'"
echo "  Stop:    ssh $SSH_USER@$SERVER 'systemctl stop $SERVICE_NAME'"
echo "  DB:      ssh $SSH_USER@$SERVER 'sqlite3 $APP_DIR/trades.db \"SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;\"'"
