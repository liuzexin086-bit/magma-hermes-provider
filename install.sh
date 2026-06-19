#!/usr/bin/env bash
# Install MAGMA Memory Provider for Hermes Agent
# Usage: bash install.sh [hermes_home]
#   hermes_home defaults to ~/.hermes

set -euo pipefail

HERMES_HOME="${1:-$HOME/.hermes}"
PLUGIN_DIR="$HERMES_HOME/plugins/magma"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing MAGMA Memory Provider..."
echo "    Target: $PLUGIN_DIR"

mkdir -p "$HERMES_HOME/plugins"

if [ -d "$PLUGIN_DIR" ]; then
    echo "    Warning: plugin already exists at $PLUGIN_DIR"
    read -p "    Overwrite? [y/N] " yn
    if [[ ! "$yn" =~ ^[yY] ]]; then
        echo "    Cancelled."
        exit 1
    fi
    rm -rf "$PLUGIN_DIR"
fi

cp -r "$SCRIPT_DIR/magma" "$PLUGIN_DIR"
echo "    Files copied."

echo "    Installing Python dependencies..."
pip install numpy networkx 2>/dev/null || pip3 install numpy networkx 2>/dev/null || {
    echo "    Warning: pip install failed. Run manually: pip install numpy networkx"
}

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the provider:"
echo "     hermes config set memory.provider magma"
echo ""
echo "  2. Restart Hermes (or /reset in an active session)"
echo ""
echo "  3. For better embeddings: pip install sentence-transformers"
echo ""
echo "See README.md for full documentation."
