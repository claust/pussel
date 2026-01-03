#!/usr/bin/env python3
"""Generate a test JWT token for e2e testing.

This script generates a valid JWT token for a test user that can be used
in e2e tests to authenticate API requests.

Usage:
    python scripts/generate_test_token.py

The token is printed to stdout and can be captured in CI:
    TEST_TOKEN=$(python scripts/generate_test_token.py)
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.auth.service import AuthService  # noqa: E402
from app.models.user_model import User  # noqa: E402

# Create a test user
test_user = User(
    id="test-user-id-for-e2e",
    email="test@example.com",
    name="E2E Test User",
    picture=None,
)

# Create auth service and generate token
auth_service = AuthService()
token, _ = auth_service.create_access_token(test_user)

# Print token to stdout (for capture in CI)
print(token)
