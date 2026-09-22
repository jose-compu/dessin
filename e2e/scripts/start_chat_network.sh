#!/bin/bash
# Start DeSSIN with web UI (run from anywhere; uses repo venv if present).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================================"
echo "Starting DeSSIN network with web UI"
echo "================================================================================"
echo ""
echo "Press Enter to start..."
read -r

if [ -x "./bin/python" ]; then
  exec ./bin/python e2e/scripts/run_network_with_web_ui.py
fi
exec python3 e2e/scripts/run_network_with_web_ui.py
