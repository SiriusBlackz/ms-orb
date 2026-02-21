#!/usr/bin/env bash
# Deploy (or update) MS-ORB IBKR bot to DigitalOcean droplet.
# Usage: SERVER=1.2.3.4 bash deploy.sh
#    or: bash deploy.sh 1.2.3.4
set -euo pipefail

SERVER="${1:-${SERVER:-}}"
SSH_USER="${SSH_USER:-root}"
APP_DIR="/opt/ms-orb-ibkr"
DEPLOY_DIR="$APP_DIR/deploy"
BOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$SERVER" ]; then
    echo "Error: No server IP provided."
    echo "Usage: SERVER=1.2.3.4 bash deploy.sh"
    echo "   or: bash deploy.sh 1.2.3.4"
    exit 1
fi

echo "=== Deploying MS-ORB IBKR Bot ==="
echo "Server:  $SSH_USER@$SERVER"
echo "Source:  $BOT_DIR"
echo "Remote:  $APP_DIR"
echo ""

# Sync bot code (exclude .env, __pycache__, deploy scripts, local DB)
echo "[1/5] Syncing code to server..."
rsync -avz --delete \
    --exclude '.env' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'deploy/' \
    --exclude 'trades.db' \
    --exclude 'logs/' \
    --exclude '.git' \
    "$BOT_DIR/" "$SSH_USER@$SERVER:$APP_DIR/"

# Sync deploy files (docker-compose, etc.) separately
echo "[2/5] Syncing deploy files..."
rsync -avz \
    --exclude '.env' \
    --exclude 'tws_settings/' \
    "$BOT_DIR/deploy/" "$SSH_USER@$SERVER:$DEPLOY_DIR/"

# Fix ownership
echo "[3/5] Fixing file ownership..."
ssh "$SSH_USER@$SERVER" "chown -R msorbbot:msorbbot $APP_DIR"

# Install/update Python dependencies
echo "[4/5] Installing Python dependencies..."
ssh "$SSH_USER@$SERVER" "sudo -u msorbbot $APP_DIR/venv/bin/pip install -q -r $APP_DIR/requirements.txt"

# Check if .env exists on server
if ! ssh "$SSH_USER@$SERVER" "test -f $APP_DIR/.env"; then
    echo ""
    echo "WARNING: No bot .env file found on server!"
    echo "Create it now:  ssh $SSH_USER@$SERVER 'nano $APP_DIR/.env'"
    echo "(use .env.example as template)"
    echo ""
    echo "Skipping service restart until .env is configured."
    exit 0
fi

# Start/update IB Gateway container + restart bot service
echo "[5/5] Starting IB Gateway and restarting bot..."
if ssh "$SSH_USER@$SERVER" "test -f $DEPLOY_DIR/.env"; then
    ssh "$SSH_USER@$SERVER" "cd $DEPLOY_DIR && docker compose up -d"
else
    echo "  WARNING: No Gateway .env — skipping docker compose."
    echo "  Create it:  ssh $SSH_USER@$SERVER 'nano $DEPLOY_DIR/.env'"
fi

ssh "$SSH_USER@$SERVER" "systemctl daemon-reload && systemctl restart ms-orb-ibkr"

# Quick status check
echo ""
echo "=== Deploy complete! Checking status... ==="
ssh "$SSH_USER@$SERVER" "sleep 2 && systemctl status ms-orb-ibkr --no-pager -l" || true

echo ""
echo "Useful commands:"
echo "  Bot logs:      ssh $SSH_USER@$SERVER 'journalctl -u ms-orb-ibkr -f'"
echo "  Bot status:    ssh $SSH_USER@$SERVER 'systemctl status ms-orb-ibkr'"
echo "  Bot stop:      ssh $SSH_USER@$SERVER 'systemctl stop ms-orb-ibkr'"
echo "  Gateway logs:  ssh $SSH_USER@$SERVER 'cd $DEPLOY_DIR && docker compose logs -f'"
echo "  Gateway stop:  ssh $SSH_USER@$SERVER 'cd $DEPLOY_DIR && docker compose stop'"
echo "  VNC debug:     ssh -L 5900:localhost:5900 $SSH_USER@$SERVER"
