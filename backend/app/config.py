import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings configuration."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Puzzle Solver API"

    # File upload settings
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    class Config:
        """Pydantic configuration class."""

        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create instance
settings = get_settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
