#!/usr/bin/env bash
# IB Gateway watchdog — restarts the gateway container when the API is dead.
#
# The failure mode this guards against (observed 2026-07-06): the container
# runs for months, socat still accepts TCP on the mapped port, but the
# Gateway java process inside is dead — so a plain port check passes while
# every API handshake times out. The probe therefore performs a REAL
# ib_insync handshake, not a TCP connect.
#
# Install (droplet): see setup_watchdog.sh. Runs from a systemd timer every
# 5 minutes; restarts the container only after 2 consecutive probe failures
# (10 minutes), and never inside the gateway's own nightly auto-restart
# window (23:50-00:15 ET).
set -uo pipefail

CONTAINER="${GW_CONTAINER:-deploy_ib-gateway_1}"
HOST="${GW_HOST:-127.0.0.1}"
PORT="${GW_PORT:-4002}"
PYTHON="${GW_PYTHON:-/opt/ms-orb-ibkr/venv/bin/python3}"
STATE_FILE="/var/run/ib-gateway-watchdog.fails"
LOG_FILE="/var/log/ib-gateway-watchdog.log"
# Optional: a healthchecks.io/UptimeRobot heartbeat URL pinged on success —
# their side alerts you if pings stop (i.e. droplet or watchdog itself died).
PING_URL="${GW_PING_URL:-}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"; }

# Skip the nightly IBC auto-restart window (23:50-00:15 ET) — the gateway is
# legitimately down then and a docker restart would fight the auto-restart.
et_hm=$(TZ=America/New_York date '+%H%M')
if [ "$et_hm" -ge 2350 ] || [ "$et_hm" -le 0015 ]; then
    log "SKIP probe (nightly auto-restart window)"
    exit 0
fi

# Real API handshake with a dedicated clientId (won't clash with the bot's).
if "$PYTHON" - <<EOF
import sys
from ib_insync import IB
ib = IB()
try:
    ib.connect("$HOST", $PORT, clientId=97, timeout=20, readonly=True)
    ok = ib.isConnected()
    ib.disconnect()
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
EOF
then
    # healthy
    rm -f "$STATE_FILE"
    if [ -n "$PING_URL" ]; then
        curl -fsS -m 10 "$PING_URL" > /dev/null 2>&1 || true
    fi
    exit 0
fi

# unhealthy — count consecutive failures
fails=$(( $(cat "$STATE_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$STATE_FILE"
log "PROBE FAILED (consecutive: $fails)"

if [ "$fails" -ge 2 ]; then
    log "RESTARTING $CONTAINER after $fails consecutive failures"
    if docker restart "$CONTAINER" >> "$LOG_FILE" 2>&1; then
        log "restart issued; clearing failure counter"
        rm -f "$STATE_FILE"
    else
        log "ERROR: docker restart failed"
    fi
fi
