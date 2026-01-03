#!/bin/bash
# Hook script that formats files after Claude modifies them
# Reads JSON from stdin and extracts the file path

# Read JSON input from stdin
INPUT=$(cat)

# Extract file_path from tool_input using jq
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Exit early if no file path (shouldn't happen for Edit/Write)
if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(dirname "$(dirname "$(dirname "$0")")")}"

cd "$PROJECT_DIR" || exit 0

# Determine which formatter to run based on the file path
if [[ "$FILE_PATH" == *"/backend/"* ]] || [[ "$FILE_PATH" == "$PROJECT_DIR/backend/"* ]]; then
    make format-backend 2>/dev/null
elif [[ "$FILE_PATH" == *"/network/"* ]] || [[ "$FILE_PATH" == "$PROJECT_DIR/network/"* ]]; then
    make format-network 2>/dev/null
elif [[ "$FILE_PATH" == *"/frontend/"* ]] || [[ "$FILE_PATH" == "$PROJECT_DIR/frontend/"* ]]; then
    make format-frontend 2>/dev/null
else
    # For other files (like root-level Python scripts), run full format
    make format 2>/dev/null
fi

# Always exit 0 to not block Claude
exit 0
