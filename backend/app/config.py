"""Configuration module for the puzzle solver application."""

import os
from functools import lru_cache

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

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create instance
settings = get_settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
