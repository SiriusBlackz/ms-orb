#!/usr/bin/env bash
# Set RISK_PER_TRADE in the droplet's cTrader bot .env and restart the bot.
# Usage: bash ctrader_bot/deploy/set_risk.sh [risk]     (default 0.0035 = 0.35%)
# One ssh session, one password prompt. Touches no other setting.
set -euo pipefail
SERVER="${SERVER:-146.190.76.14}"
RISK="${1:-0.0035}"
[[ "$RISK" =~ ^0\.[0-9]+$ ]] || { echo "Risk must look like 0.0035"; exit 1; }

ssh "root@$SERVER" 'bash -s' <<EOF
set -euo pipefail
cd /opt/ms-orb-ctrader
cp -p .env ".env.bak.\$(date +%Y%m%d-%H%M%S)"
echo "Before: \$(grep -o 'RISK_PER_TRADE=.*' .env || echo 'not set') (at line start: \$(grep -c '^RISK_PER_TRADE=' .env || true))"
# Remove any existing setting, including one glued onto the end of another line.
sed -i -e 's/RISK_PER_TRADE=.*\$//' -e '/^[[:space:]]*\$/d' .env
# Make sure the file ends with a newline, then add the setting on its own line.
[ -z "\$(tail -c1 .env)" ] || echo >> .env
echo "RISK_PER_TRADE=$RISK" >> .env
chown msorbbot:msorbbot .env
echo "After:  \$(grep '^RISK_PER_TRADE=' .env)"
systemctl restart ms-orb-ctrader
echo "Restarted; waiting 20s..."
sleep 20
systemctl is-active ms-orb-ctrader || true
tail -40 /var/log/ms-orb-ctrader.log | grep -E "Risk per trade|authorized|Balance|ERROR|WARNING" | tail -8 | cut -c1-200
EOF
