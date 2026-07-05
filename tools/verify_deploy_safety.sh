#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILTERED_DEPLOY="$ROOT_DIR/oanda_bot_filtered/deploy"
SERVICE_FILE="$FILTERED_DEPLOY/ms-orb-filtered.service"

if ! command -v rg >/dev/null 2>&1; then
    echo "ERROR: rg is required for deploy safety verification." >&2
    exit 1
fi

echo "[1/4] Checking shell syntax..."
bash -n "$FILTERED_DEPLOY/deploy.sh"
bash -n "$FILTERED_DEPLOY/setup.sh"

echo "[2/4] Checking filtered service file exists..."
test -f "$SERVICE_FILE"
test ! -f "$FILTERED_DEPLOY/ms-orb-oanda.service"

echo "[3/4] Checking filtered deploy targets..."
rg -q 'APP_DIR="/opt/ms-orb-oanda-filtered"' "$FILTERED_DEPLOY/deploy.sh"
rg -q 'SERVICE_NAME="ms-orb-filtered"' "$FILTERED_DEPLOY/deploy.sh"
rg -q 'APP_DIR="/opt/ms-orb-oanda-filtered"' "$FILTERED_DEPLOY/setup.sh"
rg -q 'SERVICE_NAME="ms-orb-filtered"' "$FILTERED_DEPLOY/setup.sh"
rg -q 'LOG_FILE="/var/log/ms-orb-filtered.log"' "$FILTERED_DEPLOY/setup.sh"
rg -q 'WorkingDirectory=/opt/ms-orb-oanda-filtered' "$SERVICE_FILE"
rg -q 'EnvironmentFile=/opt/ms-orb-oanda-filtered/.env' "$SERVICE_FILE"
rg -q 'ExecStart=/opt/ms-orb-oanda-filtered/venv/bin/python3 main.py' "$SERVICE_FILE"
rg -q 'StandardOutput=append:/var/log/ms-orb-filtered.log' "$SERVICE_FILE"
rg -q 'StandardError=append:/var/log/ms-orb-filtered.log' "$SERVICE_FILE"
rg -q 'ReadWritePaths=/opt/ms-orb-oanda-filtered /var/log/ms-orb-filtered.log' "$SERVICE_FILE"

echo "[4/4] Checking filtered deploy does not target the base service..."
if rg -n \
    'systemctl (start|stop|restart|status|enable|disable) ms-orb-oanda|journalctl -u ms-orb-oanda|/opt/ms-orb-oanda/|/var/log/ms-orb-oanda\.log|ms-orb-oanda\.service' \
    "$FILTERED_DEPLOY"; then
    echo "ERROR: filtered deploy still references base OANDA deployment targets." >&2
    exit 1
fi

echo "OK: filtered OANDA deploy targets are isolated from the base service."

# ── Config-A bot deploy isolation ────────────────────────────────────────────
CONFIGA_DEPLOY="$ROOT_DIR/oanda_bot_configa/deploy"
CONFIGA_SERVICE="$CONFIGA_DEPLOY/ms-orb-configa.service"

echo "[5/8] Checking Config-A shell syntax..."
bash -n "$CONFIGA_DEPLOY/deploy.sh"
bash -n "$CONFIGA_DEPLOY/setup.sh"

echo "[6/8] Checking Config-A service file exists..."
test -f "$CONFIGA_SERVICE"
test ! -f "$CONFIGA_DEPLOY/ms-orb-oanda.service"
test ! -f "$CONFIGA_DEPLOY/ms-orb-filtered.service"

echo "[7/8] Checking Config-A deploy targets..."
rg -q 'APP_DIR="/opt/ms-orb-configa"' "$CONFIGA_DEPLOY/deploy.sh"
rg -q 'SERVICE_NAME="ms-orb-configa"' "$CONFIGA_DEPLOY/deploy.sh"
rg -q 'APP_DIR="/opt/ms-orb-configa"' "$CONFIGA_DEPLOY/setup.sh"
rg -q 'SERVICE_NAME="ms-orb-configa"' "$CONFIGA_DEPLOY/setup.sh"
rg -q 'LOG_FILE="/var/log/ms-orb-configa.log"' "$CONFIGA_DEPLOY/setup.sh"
rg -q 'WorkingDirectory=/opt/ms-orb-configa' "$CONFIGA_SERVICE"
rg -q 'EnvironmentFile=/opt/ms-orb-configa/.env' "$CONFIGA_SERVICE"
rg -q 'ExecStart=/opt/ms-orb-configa/venv/bin/python3 main.py' "$CONFIGA_SERVICE"
rg -q 'StandardOutput=append:/var/log/ms-orb-configa.log' "$CONFIGA_SERVICE"
rg -q 'StandardError=append:/var/log/ms-orb-configa.log' "$CONFIGA_SERVICE"
rg -q 'ReadWritePaths=/opt/ms-orb-configa /var/log/ms-orb-configa.log' "$CONFIGA_SERVICE"

echo "[8/8] Checking Config-A deploy does not target the base or filtered services..."
if rg -n \
    'systemctl (start|stop|restart|status|enable|disable) (ms-orb-oanda|ms-orb-filtered)|journalctl -u (ms-orb-oanda|ms-orb-filtered)|/opt/ms-orb-oanda(-filtered)?/|/var/log/ms-orb-(oanda|filtered)\.log|ms-orb-(oanda|filtered)\.service' \
    "$CONFIGA_DEPLOY/deploy.sh" "$CONFIGA_DEPLOY/setup.sh" "$CONFIGA_SERVICE"; then
    echo "ERROR: Config-A deploy references base or filtered deployment targets." >&2
    exit 1
fi

echo "OK: Config-A deploy targets are isolated from the base and filtered services."
