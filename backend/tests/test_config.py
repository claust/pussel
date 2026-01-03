"""Test module for configuration validation."""

import os
import sys
from typing import Generator

import pytest


@pytest.fixture
def cleanup_imports() -> Generator[None, None, None]:
    """Clean up app.config imports after each test."""
    yield
    # Remove app.config modules from sys.modules to allow fresh imports
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith("app.config")]
    for module in modules_to_remove:
        del sys.modules[module]


def test_jwt_secret_validation_in_production(cleanup_imports: None) -> None:
    """Test that default JWT_SECRET raises error in production."""
    # Set production environment BEFORE importing config
    os.environ["ENVIRONMENT"] = "production"
    os.environ["JWT_SECRET"] = "change-me-in-production-use-a-strong-random-secret"

    try:
        # Import and create Settings - should raise ValidationError
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="JWT_SECRET is using the default value"):
            from app.config import Settings

            Settings()
    finally:
        # Cleanup
        os.environ.pop("ENVIRONMENT", None)
        os.environ.pop("JWT_SECRET", None)


def test_jwt_secret_validation_in_development(cleanup_imports: None) -> None:
    """Test that default JWT_SECRET is allowed in development."""
    # Set development environment BEFORE importing config
    os.environ["ENVIRONMENT"] = "development"
    os.environ["JWT_SECRET"] = "change-me-in-production-use-a-strong-random-secret"

    try:
        # Import and create Settings - should NOT raise error
        from app.config import Settings

        settings = Settings()
        assert settings.JWT_SECRET == "change-me-in-production-use-a-strong-random-secret"
        assert settings.ENVIRONMENT == "development"
    finally:
        # Cleanup
        os.environ.pop("ENVIRONMENT", None)
        os.environ.pop("JWT_SECRET", None)


def test_jwt_secret_validation_in_test(cleanup_imports: None) -> None:
    """Test that default JWT_SECRET is allowed in test environment."""
    # Set test environment BEFORE importing config
    os.environ["ENVIRONMENT"] = "test"
    os.environ["JWT_SECRET"] = "change-me-in-production-use-a-strong-random-secret"

    try:
        # Import and create Settings - should NOT raise error
        from app.config import Settings

        settings = Settings()
        assert settings.JWT_SECRET == "change-me-in-production-use-a-strong-random-secret"
        assert settings.ENVIRONMENT == "test"
    finally:
        # Cleanup
        os.environ.pop("ENVIRONMENT", None)
        os.environ.pop("JWT_SECRET", None)


def test_jwt_secret_custom_value_in_production(cleanup_imports: None) -> None:
    """Test that custom JWT_SECRET is allowed in production."""
    # Set production environment with custom secret BEFORE importing config
    os.environ["ENVIRONMENT"] = "production"
    os.environ["JWT_SECRET"] = "my-secure-custom-secret-key-12345"

    try:
        # Import and create Settings - should NOT raise error
        from app.config import Settings

        settings = Settings()
        assert settings.JWT_SECRET == "my-secure-custom-secret-key-12345"
        assert settings.ENVIRONMENT == "production"
    finally:
        # Cleanup
        os.environ.pop("ENVIRONMENT", None)
        os.environ.pop("JWT_SECRET", None)
