#!/usr/bin/env bash
# The gateway came up in Read-Only API mode on the new account, so the bot
# cannot place orders. This restarts the gateway (so it re-applies its
# "read-only = no" setting), restarts the bot, and shows the result.
# Run on the Mac:  bash tsla_readonly_fix.sh     (one password prompt, ~3 min)
ssh root@146.190.76.14 'bash -s' <<'EOF'
cd /opt/ms-orb-ibkr/deploy || exit 1
echo "=== gateway setting requested (want READ_ONLY_API=no) ==="
docker compose exec -T ib-gateway env < /dev/null | grep -E "READ_ONLY_API|TRADING_MODE"
echo "=== what the gateway said about read-only BEFORE restart ==="
docker compose logs --no-color 2>&1 | grep -iE "read-only|readonly" | tail -6 | cut -c1-200

echo "=== restarting gateway, waiting 100s for login ==="
systemctl stop ms-orb-ibkr
docker compose restart < /dev/null
sleep 100
echo "=== what the gateway said about read-only AFTER restart ==="
docker compose logs --no-color --since 2m 2>&1 | grep -iE "read-only|readonly|Login has completed" | tail -6 | cut -c1-200

echo "=== restarting bot, waiting 40s ==="
systemctl start ms-orb-ibkr
sleep 40
echo "service: $(systemctl is-active ms-orb-ibkr)"
echo "=== bot startup log (this start only) ==="
awk '/MS-ORB IBKR Trading Bot Starting/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' \
  /var/log/ms-orb-ibkr.log | cut -c1-230 | tail -40
echo "=== trade record files ==="
ls -la /opt/ms-orb-ibkr/*.db
EOF
