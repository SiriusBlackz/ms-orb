#!/usr/bin/env bash
# Bring the TSLA bot back on the new IBKR paper account. Run on the Mac:
#   bash tsla_start.sh
# Asks for the droplet password several times (once per step).
set -uo pipefail
cd "$(dirname "$0")"
S=root@146.190.76.14

echo "=== 1/3 Set the old account's trade record aside ==="
ssh "$S" 'bash -s' <<'EOF'
cd /opt/ms-orb-ibkr || exit 1
if [ -f ibkr_trades.db ]; then
  mv -n ibkr_trades.db ibkr_trades_old_account.db && echo "moved ibkr_trades.db -> ibkr_trades_old_account.db"
else
  echo "no ibkr_trades.db here (nothing to move)"
fi
ls -la *.db 2>/dev/null
EOF

echo
echo "=== 2/3 Deploy the new code (this also starts the bot) ==="
SERVER=146.190.76.14 bash ibkr_bot/deploy/deploy.sh || { echo "DEPLOY FAILED - stopping here"; exit 1; }

echo
echo "=== 3/3 Start on boot, then show the bot's startup log ==="
ssh "$S" 'bash -s' <<'EOF'
systemctl enable ms-orb-ibkr
sleep 20
echo "service: $(systemctl is-active ms-orb-ibkr), on boot: $(systemctl is-enabled ms-orb-ibkr)"
tail -n 60 /var/log/ms-orb-ibkr.log | cut -c1-240
EOF
