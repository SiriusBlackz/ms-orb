#!/usr/bin/env bash
# First-time server setup for the MS-ORB cTrader bot.
# Run this on the droplet as root: bash setup.sh
#
# Safe to run alongside the existing ms-orb-oanda / ms-orb-filtered services —
# everything here is scoped to the Config-A paths.
set -euo pipefail

APP_USER="msorbbot"
APP_DIR="/opt/ms-orb-ctrader"
SERVICE_NAME="ms-orb-ctrader"
LOG_FILE="/var/log/ms-orb-ctrader.log"

echo "=== MS-ORB cTrader Bot - Server Setup ==="

# Install system dependencies (no-op if already present from the base bot).
# Non-fatal: this droplet has a broken apt post-invoke hook (command-not-found
# missing apt_pkg) that fails `apt-get update` even when the package work
# succeeded. The venv step below fails loudly if python3-venv is truly absent.
echo "[1/6] Installing system packages..."
if ! apt-get update -qq; then
    echo "  WARNING: apt-get update exited non-zero (continuing; deps likely already installed)"
fi
if ! apt-get install -y -qq python3 python3-pip python3-venv sqlite3 > /dev/null; then
    echo "  WARNING: apt-get install exited non-zero (continuing; venv step will catch real gaps)"
fi

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
cp "$(dirname "$0")/$SERVICE_NAME.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable "$SERVICE_NAME.service"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Create .env file:  nano $APP_DIR/.env"
echo "     (use .env.example as template — needs its OWN practice sub-account)"
echo "  2. Deploy code:       bash deploy.sh"
echo "  3. Start the bot:     systemctl start $SERVICE_NAME"
echo "  4. Check status:      systemctl status $SERVICE_NAME"
echo "  5. View logs:         journalctl -u $SERVICE_NAME -f"
