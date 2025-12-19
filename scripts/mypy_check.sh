#!/bin/bash

# This script runs mypy with the project's virtual environment
# Works from VS Code, CLI, or any git client
# Usage: mypy_check.sh file1.py file2.py ...

set -e

# Get the repository root (where this script's parent lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$REPO_ROOT/venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r network/requirements.txt"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run mypy with all passed files
mypy --ignore-missing-imports "$@"
