#!/bin/bash

# This script lints Bicep files using Azure CLI
# Usage: bicep_lint.sh file1.bicep file2.bicep ...

set -e

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI is not installed. Please install it first."
    echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Ensure Bicep CLI is installed
if ! az bicep version &> /dev/null; then
    echo "Installing Bicep CLI..."
    az bicep install
fi

# Process each file
EXIT_CODE=0
for file in "$@"; do
    echo "Linting $file..."
    if ! az bicep build --file "$file" --stdout > /dev/null; then
        echo "Error in $file"
        EXIT_CODE=1
    fi
done

exit $EXIT_CODE
