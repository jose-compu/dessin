#!/bin/bash
# Clone and pin nanochat under external/nanochat (run from anywhere).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NANOCHAT_DIR="external/nanochat"
NANOCHAT_COMMIT="90442de35f860226ccec6d64ab1829bdc1fad55a"
NANOCHAT_REPO="https://github.com/karpathy/nanochat.git"

echo "Setting up nanochat integration for DeSSIN"
echo "=========================================="
echo ""

mkdir -p external

if [ -d "$NANOCHAT_DIR" ]; then
    echo "Nanochat directory exists"
    cd "$NANOCHAT_DIR"
    if [ -d ".git" ]; then
        echo "Updating nanochat..."
        git fetch origin
        git checkout "$NANOCHAT_COMMIT"
        echo "Nanochat pinned to commit $NANOCHAT_COMMIT"
    else
        echo "Error: $NANOCHAT_DIR exists but is not a git repository"
        exit 1
    fi
else
    echo "Cloning nanochat repository..."
    git clone "$NANOCHAT_REPO" "$NANOCHAT_DIR"
    cd "$NANOCHAT_DIR"
    git checkout "$NANOCHAT_COMMIT"
    echo "Nanochat cloned and pinned to commit $NANOCHAT_COMMIT"
fi

cd "$REPO_ROOT"
echo ""
echo "Done. Pinned revision is documented in dessin/nanochat_wrapper.py."
