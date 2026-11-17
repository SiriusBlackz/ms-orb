#!/usr/bin/env bash
# Read-only: show what IB said around the most recent failed TSLA orders.
# Usage: bash tsla_check3.sh > tsla_record3.txt   (root password once). Changes nothing.
ssh root@146.190.76.14 'bash -s' <<'EOF'
export LC_ALL=C
L=/var/log/ms-orb-ibkr.log
T=$(mktemp); trap 'rm -f "$T"' EXIT
tail -c 40000000 "$L" > "$T"
echo "=== service ==="; systemctl is-active ms-orb-ibkr
echo "=== last 3 failed orders, with the 14 log lines before and 6 after each ==="
grep -n "ORDER FAILED" "$T" | tail -3 | cut -d: -f1 | while read n; do
  echo "--- around line $n"
  sed -n "$((n-14)),$((n+6))p" "$T" | cut -c1-330
done
echo "=== distinct IB order-related error codes in the sample (count, code, text) ==="
grep -E "Error [0-9]+, reqId [0-9]+" "$T" | grep -viE "historical|market data" \
  | sed -E 's/.*(Error [0-9]+), reqId [0-9]+: (.{0,110}).*/\1: \2/' | sort | uniq -c | sort -rn | head -10
echo "=== TSLA executions seen today ==="
grep "^$(date -u +%Y-%m-%d)" "$T" | grep -E "execDetails|ORDER FILLED|commissionReport" | cut -c1-80,150-420 | head -12
echo "=== working TSLA orders mentioned most recently ==="
grep -E "openOrder|orderStatus" "$T" | tail -4 | cut -c1-420
EOF
