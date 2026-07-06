#!/usr/bin/env bash
# Install the IB Gateway watchdog (systemd service + 5-minute timer).
# Run on the droplet as root from this directory:  bash setup_watchdog.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

install -m 755 "$DIR/gateway_watchdog.sh" /usr/local/bin/ib-gateway-watchdog.sh
touch /var/log/ib-gateway-watchdog.log

cat > /etc/systemd/system/ib-gateway-watchdog.service <<'EOF'
[Unit]
Description=IB Gateway API watchdog (probe + restart on failure)
After=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/ib-gateway-watchdog.sh
# Optional heartbeat URL (healthchecks.io etc.):
# Environment=GW_PING_URL=https://hc-ping.com/your-uuid
EOF

cat > /etc/systemd/system/ib-gateway-watchdog.timer <<'EOF'
[Unit]
Description=Run IB Gateway watchdog every 5 minutes

[Timer]
OnBootSec=3min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ib-gateway-watchdog.timer

echo "Watchdog installed. Check: systemctl list-timers ib-gateway-watchdog.timer"
echo "Log: tail -f /var/log/ib-gateway-watchdog.log"
