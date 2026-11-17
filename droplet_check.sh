#!/usr/bin/env bash
# Read-only health check of the droplet bots. Usage: bash droplet_check.sh
# Asks for the root password once. Changes nothing on the droplet.
ssh root@146.190.76.14 'bash -s' <<'EOF'
echo "=== Services (oanda, filtered, ibkr, ctrader) ==="
systemctl is-active ms-orb-oanda ms-orb-filtered ms-orb-ibkr ms-orb-ctrader

for svc in ms-orb-oanda ms-orb-filtered; do
  echo
  echo "=== $svc: started ==="
  systemctl show "$svc" -p ActiveEnterTimestamp --value
  echo "=== $svc: OANDA 401 errors per day, last 7 days ==="
  journalctl -u "$svc" --since "-7d" --no-pager -o short-iso \
    | grep "failed \[401" | cut -c1-10 | sort | uniq -c
  echo "=== $svc: warnings, errors, position take-over (last 60 min) ==="
  journalctl -u "$svc" --since "-60min" --no-pager \
    | grep -E "WARNING|ERROR|ADOPT|Adopt|Existing position|recover" \
    | cut -c1-260 | tail -12
  echo "=== $svc: latest status block ==="
  journalctl -u "$svc" --since "-60min" --no-pager \
    | grep -A 8 "=== Status" | cut -c1-260 | tail -9
done
EOF
