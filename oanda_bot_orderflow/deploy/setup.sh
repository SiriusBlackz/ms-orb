#!/usr/bin/env bash
# First-time server setup for MS-ORB OANDA bot (Order Flow variant).
# Run this on the droplet as root: bash setup.sh
#
# Runs alongside the base ms-orb-oanda service — separate directory,
# separate service, separate DB and logs. Same user (msorbbot).
set -euo pipefail

APP_USER="msorbbot"
APP_DIR="/opt/ms-orb-oanda-orderflow"
LOG_FILE="/var/log/ms-orb-oanda-orderflow.log"
SERVICE_NAME="ms-orb-oanda-orderflow"

echo "=== MS-ORB OANDA Bot (Order Flow) - Server Setup ==="

# Install system dependencies (likely already present from base bot)
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv sqlite3 > /dev/null

# Ensure service user exists (shared with base bot)
echo "[2/6] Checking service user '$APP_USER'..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
    echo "  Created user: $APP_USER"
else
    echo "  User $APP_USER already exists."
fi

# Create application directory
echo "[3/6] Setting up app directory at $APP_DIR..."
mkdir -p "$APP_DIR"
chown "$APP_USER":"$APP_USER" "$APP_DIR"

# Create log file with correct permissions
echo "[4/6] Setting up log file..."
touch "$LOG_FILE"
chown "$APP_USER":"$APP_USER" "$LOG_FILE"

# Set up Python virtual environment
echo "[5/6] Creating Python virtual environment..."
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/venv"

# Install systemd service
echo "[6/6] Installing systemd service..."
cp "$(dirname "$0")/$SERVICE_NAME.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable "$SERVICE_NAME.service"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Create .env file:  nano $APP_DIR/.env"
echo "     (use .env.example as template)"
echo "     IMPORTANT: Use your DEMO account credentials, not live!"
echo "  2. Deploy code:       bash deploy.sh <server-ip>"
echo "  3. Start the bot:     systemctl start $SERVICE_NAME"
echo "  4. Check status:      systemctl status $SERVICE_NAME"
echo "  5. View logs:         journalctl -u $SERVICE_NAME -f"
echo ""
echo "Both bots side by side:"
echo "  Base bot:       systemctl status ms-orb-oanda"
echo "  Order flow bot: systemctl status $SERVICE_NAME"
