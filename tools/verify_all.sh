#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== MS-ORB offline verification =="
bash "$ROOT_DIR/tools/verify_deploy_safety.sh"
bash "$ROOT_DIR/tools/verify_strategy_parity.sh"
echo "OK: all offline verifiers passed."
