#!/bin/bash

# This script runs pyright with the project's virtual environment
# Only checks network/ files (backend has its own pyright config)
# Works from VS Code, CLI, or any git client
# Usage: pyright_check.sh file1.py file2.py ...

set -e

# Get the repository root (where this script's parent lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$REPO_ROOT/venv"

# Check if any network files are being checked
HAS_NETWORK_FILES=false
for file in "$@"; do
    if [[ "$file" == network/* ]]; then
        HAS_NETWORK_FILES=true
        break
    fi
done

# If no network files, exit successfully
if [ "$HAS_NETWORK_FILES" = false ]; then
    exit 0
fi

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r network/requirements.txt"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run pyright from network directory with its config
cd "$REPO_ROOT/network"
pyright .
