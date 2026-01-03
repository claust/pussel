"""Configuration module for the puzzle solver application."""

import os
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings configuration."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Puzzle Solver"

    # File upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Azure settings
    USE_AZURE_STORAGE: bool = False
    AZURE_STORAGE_CONNECTION_STRING: str = ""

    # Authentication settings
    JWT_SECRET: str = "change-me-in-production-use-a-strong-random-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60
    GOOGLE_CLIENT_ID: str = ""

    # Background removal settings
    ENABLE_BACKGROUND_REMOVAL: bool = True
    REMBG_MODEL: str = "u2net"  # u2net, u2netp, isnet-general-use

    # Environment setting (used for validation)
    ENVIRONMENT: str = "development"  # development, test, or production

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")

    @model_validator(mode="after")
    def validate_jwt_secret(self) -> "Settings":
        """Validate that JWT_SECRET is not using the default value in production.

        Raises:
            ValueError: If JWT_SECRET is the default value in production environment.

        Returns:
            The validated Settings instance.
        """
        default_secret = "change-me-in-production-use-a-strong-random-secret"
        if self.ENVIRONMENT == "production" and self.JWT_SECRET == default_secret:
            raise ValueError(
                "JWT_SECRET is using the default value in production environment. "
                "This is a security risk. Please set a secure random secret. "
                "Generate one with: openssl rand -base64 32"
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create instance
settings = get_settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
