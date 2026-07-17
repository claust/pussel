"""Configuration module for the puzzle solver application."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
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

    # Dev-only: save accepted piece-preview crops under UPLOAD_DIR/preview_crops
    # so real-world false positives can be harvested as classifier hard negatives
    SAVE_PREVIEW_CROPS: bool = False

    # Piece matcher backend: "classical" (SIFT->NCC hybrid, exp25) or "cnn"
    MATCHER: Literal["classical", "cnn"] = "classical"
    # Grid size used only for the NCC fallback's nominal cell-size estimate
    # (matches the north-star evaluation puzzles).
    CLASSICAL_GRID_ROWS: int = Field(default=4, ge=1)
    CLASSICAL_GRID_COLS: int = Field(default=4, ge=1)

    # Piece geometry (M8) scan-lock thresholds on the combined shape+spatial-color
    # z-score (exp28 M7, frozen on north_star): accept as a match below
    # PIECE_GEOMETRY_T_ACCEPT, declare a new piece above PIECE_GEOMETRY_T_NEW, and
    # treat anything in between as a gray zone (ask the user to rescan).
    # t_accept sits at M7's FMR=1% ROC point (-3.98) rather than the strictest
    # frozen setting (-4.78): the M10 device runs showed the strict point sends
    # ~26-30% of genuine re-scans to the gray zone, and with the scanner
    # auto-enrolling gray-zone pieces those re-scans would duplicate. At -3.98
    # gray-zone re-scans drop to ~8% for a 1% wrong-lock risk (see exp28
    # HANDOFF M7/M10).
    PIECE_GEOMETRY_T_ACCEPT: float = -3.98
    PIECE_GEOMETRY_T_NEW: float = -0.80

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
