#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v rg >/dev/null 2>&1; then
    echo "ERROR: rg is required for strategy parity verification." >&2
    exit 1
fi

echo "[1/3] Running hermetic SessionManager tests..."
python3 "$ROOT_DIR/oanda_bot/test_session_manager.py"
python3 "$ROOT_DIR/oanda_bot_orderflow/test_session_manager.py"
python3 "$ROOT_DIR/oanda_bot_configa/test_session_manager.py"
python3 "$ROOT_DIR/ctrader_bot/test_session_manager.py"
python3 "$ROOT_DIR/ctrader_bot/test_ctrader_client.py"
python3 "$ROOT_DIR/ibkr_bot/test_session_manager.py"

echo "[2/3] Checking authoritative backtest re-entry gate..."
rg -q 'if result\.exit_reason == "stop":' "$ROOT_DIR/oanda_bot/backtest.py"
rg -q 'detector\.reset_direction\(result\.direction\)' "$ROOT_DIR/oanda_bot/backtest.py"
rg -q 'Target hit or time exit -> done for this session day' "$ROOT_DIR/oanda_bot/backtest.py"

echo "[3/3] Checking live SessionManager re-entry gates..."
for manager in \
    "$ROOT_DIR/oanda_bot/session_manager.py" \
    "$ROOT_DIR/oanda_bot_orderflow/session_manager.py"
do
    rg -q 'if reason == "stop" and ctx\.can_trade and ctx\.breakout_detector:' "$manager"
    rg -q 'ctx\.state = SessionState\.WATCHING_FOR_BREAKOUT' "$manager"
    rg -q 'ctx\.state = SessionState\.SESSION_CLOSED' "$manager"
done

echo "OK: base and orderflow session managers match the backtest stop-only re-entry invariant."
