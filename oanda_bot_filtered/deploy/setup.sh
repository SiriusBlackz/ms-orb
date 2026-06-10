#!/usr/bin/env bash
# First-time server setup for MS-ORB OANDA bot.
# Run this on the droplet as root: bash setup.sh
set -euo pipefail

APP_USER="msorbbot"
APP_DIR="/opt/ms-orb-oanda"
LOG_FILE="/var/log/ms-orb-oanda.log"

echo "=== MS-ORB OANDA Bot - Server Setup ==="

# Install system dependencies
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv sqlite3 > /dev/null

# Create non-root service user (no login shell)
echo "[2/6] Creating service user '$APP_USER'..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
    echo "  Created user: $APP_USER"
else
    echo "  User $APP_USER already exists, skipping."
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
cp "$(dirname "$0")/ms-orb-oanda.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable ms-orb-oanda.service

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Create .env file:  nano $APP_DIR/.env"
echo "     (use .env.example as template)"
echo "  2. Deploy code:       bash deploy.sh"
echo "  3. Start the bot:     systemctl start ms-orb-oanda"
echo "  4. Check status:      systemctl status ms-orb-oanda"
echo "  5. View logs:         journalctl -u ms-orb-oanda -f"
