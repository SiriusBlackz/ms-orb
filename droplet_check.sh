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

# FTMO trial bot. Reads the journal, or the log file if the journal is empty.
ct() {
  if [ -n "$(journalctl -u ms-orb-ctrader --since "-7d" --no-pager -n 1 -q 2>/dev/null)" ]; then
    journalctl -u ms-orb-ctrader --since "-7d" --no-pager -o cat
  else
    tail -n 300000 /var/log/ms-orb-ctrader.log
  fi
}
echo
echo "=== ms-orb-ctrader: started ==="
systemctl show ms-orb-ctrader -p ActiveEnterTimestamp --value
echo "=== ms-orb-ctrader: startup settings ==="
ct | grep -E "Risk per trade|RR Target|Max trades per session|Environment:|Balance" | tail -6 | cut -c1-200
echo "=== ms-orb-ctrader: orders, fills, exits (last 7 days) ==="
ct | grep -E "ORDER PLACED|ORDER FILLED|ORDER FAILED|EXIT LOGGED|Position closed" | cut -c1-240 | tail -40
echo "=== ms-orb-ctrader: WARNING/ERROR lines per day ==="
ct | grep -E "\| (WARNING|ERROR) " | cut -c1-10 | sort | uniq -c
echo "=== ms-orb-ctrader: most frequent WARNING/ERROR messages ==="
ct | grep -E "\| (WARNING|ERROR) " | cut -d'|' -f3- | cut -c1-200 | sort | uniq -c | sort -rn | head -8
echo "=== ms-orb-ctrader: latest status block ==="
ct | grep -A 8 "=== Status" | cut -c1-240 | tail -9
EOF
