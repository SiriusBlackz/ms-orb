#!/usr/bin/env bash
# Read-only pull of the TSLA (IBKR paper) bot's forward record from the droplet.
# Usage: bash tsla_check.sh > tsla_record.txt   (asks for the root password once)
# Prints service state, the settings the bot logged at its last start, recent
# connection problems, and every trade in its database. Changes nothing.
ssh root@146.190.76.14 'bash -s' <<'EOF'
echo "=== service ==="
systemctl is-active ms-orb-ibkr
systemctl show ms-orb-ibkr -p ActiveEnterTimestamp --value
echo "=== settings logged at last start ==="
grep -E "Risk per trade|RR Target|Max trades|Stop mode|STOP|HTF|First breakout|Instruments|Dry run|Account|Connected" \
  /var/log/ms-orb-ibkr.log | tail -14 | cut -c1-200
echo "=== connection errors by month (log lines) ==="
grep -iE "ERROR|disconnect|refused|timeout" /var/log/ms-orb-ibkr.log | cut -c1-7 | sort | uniq -c | tail -8
echo "=== last status block ==="
grep -A 4 "=== Status" /var/log/ms-orb-ibkr.log | tail -5 | cut -c1-200
echo "=== trades ==="
cd /opt/ms-orb-ibkr
ls -la *.db 2>/dev/null
python3 - <<'PY'
import glob, sqlite3
for db in sorted(glob.glob("*.db")):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT instrument, session, direction, entry_time, entry_price, stop_loss, "
            "take_profit, units, exit_time, exit_price, exit_reason, pnl, rr_achieved, "
            "opening_range_high, opening_range_low FROM trades ORDER BY entry_time").fetchall()
    except Exception as e:
        print(f"{db}: {e}"); continue
    print(f"--- {db}: {len(rows)} trades")
    print("instrument,session,direction,entry_time,entry,sl,tp,units,exit_time,exit,reason,pnl,rr,or_high,or_low")
    for r in rows:
        print(",".join("" if v is None else str(v) for v in r))
PY
EOF
