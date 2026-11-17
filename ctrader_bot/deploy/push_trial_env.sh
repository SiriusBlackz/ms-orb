#!/usr/bin/env bash
# Point the droplet's cTrader bot at a new FTMO trial account.
#
# Takes four values (typed at the prompts, or kept from the LOCAL
# ctrader_bot/.env by pressing Enter) and applies them to the
# droplet's /opt/ms-orb-ctrader/.env over one ssh session (one password prompt):
#   CTRADER_ACCESS_TOKEN, CTRADER_REFRESH_TOKEN, CTRADER_ACCOUNT_ID, RISK_PER_TRADE
# On the droplet it also: stops the bot, backs up .env, archives trades.db,
# removes the stale saved tokens, starts the bot and shows the log.
# Tokens are never printed.
#
# Usage: bash ctrader_bot/deploy/push_trial_env.sh
set -euo pipefail

SERVER="${SERVER:-146.190.76.14}"
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"

[ -f "$ENV_FILE" ] || { echo "No local .env at $ENV_FILE"; exit 1; }

get() {
    grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- \
        | tr -d '\r' | sed -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//'
}

ACCESS="$(get CTRADER_ACCESS_TOKEN || true)"
REFRESH="$(get CTRADER_REFRESH_TOKEN || true)"
ACCOUNT="$(get CTRADER_ACCOUNT_ID || true)"
RISK="$(get RISK_PER_TRADE || true)"
RISK="${RISK:-0.0035}"

echo "Press Enter at any prompt to keep the value from the local .env."
read -r -p "New trial ctidTraderAccountId [${ACCOUNT:-none}]: " in; ACCOUNT="${in:-$ACCOUNT}"
read -r -p "Risk per trade [${RISK}]: " in; RISK="${in:-$RISK}"
read -r -s -p "New access token (hidden) [keep local]: " in; echo; ACCESS="${in:-$ACCESS}"
read -r -s -p "New refresh token (hidden) [keep local]: " in; echo; REFRESH="${in:-$REFRESH}"

ASSIGN=""
add() {
    local k="$1" v
    v="$(printf '%s' "$2" | tr -d '[:space:]')"
    if [ -z "$v" ]; then echo "No value for $k — nothing changed."; exit 1; fi
    if ! [[ "$v" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "$k contains unexpected characters — nothing changed."; exit 1
    fi
    ASSIGN+="$(printf 'V_%s=%q' "$k" "$v")"$'\n'
}
add CTRADER_ACCESS_TOKEN "$ACCESS"
add CTRADER_REFRESH_TOKEN "$REFRESH"
add CTRADER_ACCOUNT_ID "$ACCOUNT"
add RISK_PER_TRADE "$RISK"

echo
echo "Will send to $SERVER:"
echo "  CTRADER_ACCOUNT_ID    = $ACCOUNT"
echo "  RISK_PER_TRADE        = $RISK"
echo "  access/refresh tokens = (hidden)"
read -r -p "Apply these to the droplet and restart ms-orb-ctrader? [y/N] " ok
[ "$ok" = "y" ] || [ "$ok" = "Y" ] || { echo "Cancelled; nothing changed."; exit 0; }

ssh "root@$SERVER" 'bash -s' <<EOF
set -euo pipefail
$ASSIGN
cd /opt/ms-orb-ctrader
stamp=\$(date +%Y%m%d-%H%M%S)
systemctl stop ms-orb-ctrader
cp -p .env ".env.bak.\$stamp"
if [ -f trades.db ]; then mv trades.db "trades_archived_\$stamp.db"; echo "Archived trades.db"; fi
rm -f ctrader_tokens.json
setkv() {
    if grep -q "^\$1=" .env; then
        sed -i "s|^\$1=.*|\$1=\$2|" .env
    else
        [ -z "\$(tail -c1 .env)" ] || echo >> .env   # ensure trailing newline first
        printf '%s=%s\n' "\$1" "\$2" >> .env
    fi
}
setkv CTRADER_ACCESS_TOKEN  "\$V_CTRADER_ACCESS_TOKEN"
setkv CTRADER_REFRESH_TOKEN "\$V_CTRADER_REFRESH_TOKEN"
setkv CTRADER_ACCOUNT_ID    "\$V_CTRADER_ACCOUNT_ID"
setkv RISK_PER_TRADE        "\$V_RISK_PER_TRADE"
chown msorbbot:msorbbot .env
echo "Updated .env (backup: .env.bak.\$stamp). Non-secret settings now:"
grep -E "^(CTRADER_ACCOUNT_ID|CTRADER_ENVIRONMENT|RISK_PER_TRADE|ENABLED_SESSIONS)=" .env || true
systemctl start ms-orb-ctrader
echo "Started; waiting 25s for the bot to connect..."
sleep 25
systemctl is-active ms-orb-ctrader || true
tail -25 /var/log/ms-orb-ctrader.log | cut -c1-240
EOF
