#!/usr/bin/env bash
# First-time server setup for MS-ORB IBKR bot.
# Run this on the droplet as root: bash setup.sh
set -euo pipefail

APP_USER="msorbbot"
APP_DIR="/opt/ms-orb-ibkr"
DEPLOY_DIR="$APP_DIR/deploy"
LOG_FILE="/var/log/ms-orb-ibkr.log"

echo "=== MS-ORB IBKR Bot - Server Setup ==="

# Install system dependencies
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv sqlite3 docker.io docker-compose-plugin > /dev/null

# Enable Docker
echo "[2/7] Enabling Docker..."
systemctl enable --now docker

# Create non-root service user (no login shell), add to docker group
echo "[3/7] Creating service user '$APP_USER'..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
    echo "  Created user: $APP_USER"
else
    echo "  User $APP_USER already exists, skipping."
fi
usermod -aG docker "$APP_USER"

# Create application directory
echo "[4/7] Setting up app directory at $APP_DIR..."
mkdir -p "$APP_DIR"
chown "$APP_USER":"$APP_USER" "$APP_DIR"

# Create log file with correct permissions
echo "[5/7] Setting up log file..."
touch "$LOG_FILE"
chown "$APP_USER":"$APP_USER" "$LOG_FILE"

# Set up Python virtual environment
echo "[6/7] Creating Python virtual environment..."
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/venv"

# Install systemd service
echo "[7/7] Installing systemd service..."
cp "$(dirname "$0")/ms-orb-ibkr.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable ms-orb-ibkr.service

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Create bot .env:      nano $APP_DIR/.env"
echo "     (use ibkr_bot/.env.example as template)"
echo "  2. Create Gateway .env:  nano $DEPLOY_DIR/.env"
echo "     (use deploy/.env.example as template)"
echo "  3. Deploy code:          bash deploy.sh <server-ip>"
echo "  4. Start Gateway:        cd $DEPLOY_DIR && docker compose up -d"
echo "  5. Start bot:            systemctl start ms-orb-ibkr"
echo "  6. Check status:         systemctl status ms-orb-ibkr"
echo "  7. View logs:            journalctl -u ms-orb-ibkr -f"
