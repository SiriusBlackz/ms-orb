#!/usr/bin/env bash
# Read-only follow-up diagnostics for the TSLA (IBKR paper) bot.
# Usage: bash tsla_check2.sh > tsla_record2.txt   (root password once). Changes nothing
# on the bot; uses two temporary files on the droplet and removes them.
# The bot log is ~900 MB, so this works on the last 60 MB plus one pass for two dates.
# Progress is shown in the terminal; expect about a minute.
ssh root@146.190.76.14 'bash -s' <<'EOF'
export LC_ALL=C
L=/var/log/ms-orb-ibkr.log
T=$(mktemp); D=$(mktemp); trap 'rm -f "$T" "$D"' EXIT
echo "step 1/4: sampling the last 60 MB of the log..." >&2
tail -c 60000000 "$L" > "$T"
echo "=== disk space ==="
df -h / | tail -1
echo "=== sample window ==="
sed -n '2p' "$T" | cut -c1-23; tail -1 "$T" | cut -c1-23
echo "=== top 12 distinct ERROR/WARNING messages in the sample ==="
grep -E "\| (ERROR|WARNING|CRITICAL) " "$T" | cut -c25-200 | sed -E 's/[0-9]+(\.[0-9]+)?/N/g' | sort | uniq -c | sort -rn | head -12
echo "=== last ValueError with the 14 lines before it ==="
n=$(grep -n "ValueError" "$T" | tail -1 | cut -d: -f1)
[ -n "$n" ] && sed -n "$((n>14?n-14:1)),$((n+1))p" "$T" | cut -c1-260
echo "=== today's session: state changes, signals, orders, errors ==="
grep "^$(date -u +%Y-%m-%d) 1[3-9]" "$T" | grep -E "\->|SIGNAL|ORDER|rejected|ERROR|WARNING|Range|recovered|Existing" | cut -c1-240 | sed -E 's/^(.{240}).*/\1/' | awk '!seen[substr($0,25,120)]++' | head -25
echo "step 2/4: finding the first ValueError (scans from the start)..." >&2
echo "=== first ValueError in the whole log ==="
grep -m1 -n "ValueError" "$L" | cut -c1-260
echo "step 3/4: extracting 13-14 August (one full pass)..." >&2
grep -E "^2026-08-1[34]" "$L" > "$D"
echo "=== 13 August (last trading day): trade lifecycle ==="
grep "^2026-08-13" "$D" | grep -E "SIGNAL|ORDER|FILL|Fill|filled|racket|EXIT|TIME|CLOSED|closed detected|rejected|Reject|Error [0-9]+|ERROR" | cut -c1-240 | awk '!seen[substr($0,25,120)]++' | head -40
echo "=== 14 August 13:25-20:10 UTC: state changes, signals, errors ==="
grep -E "^2026-08-14 (1[3-9]|20:0)" "$D" | grep -E "\->|SIGNAL|ORDER|rejected|ERROR|WARNING|Range|Existing" | cut -c1-240 | awk '!seen[substr($0,25,120)]++' | head -30
echo "step 4/4: done" >&2
EOF
