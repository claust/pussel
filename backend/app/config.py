"""Configuration module for the puzzle solver application."""

import logging
import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings configuration."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Puzzle Solver"

    # File upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # Azure settings
    USE_AZURE_STORAGE: bool = False
    AZURE_STORAGE_CONNECTION_STRING: str = ""

    # Authentication settings
    JWT_SECRET: str = "change-me-in-production-use-a-strong-random-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60
    GOOGLE_CLIENT_ID: str = ""

    # Email allowlist for login: any successfully-verified Google account whose
    # email appears here (case-insensitively, whitespace-trimmed) may sign in.
    # An EMPTY list means "allow any Google account" — this preserves today's
    # behavior and is convenient for local dev, but means anyone with a Gmail
    # account gets a full account since there's no user table to otherwise
    # gate access. Set this in deployed environments to restrict access.
    ALLOWED_EMAILS: list[str] = []

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

    # Rate limiting (in-memory, per-process — see app/rate_limit.py for the
    # per-process caveat). Both are requests-per-minute per caller identity.
    # Set either to 0 to disable that limit entirely, e.g. for tests and
    # local dev where hammering an endpoint on purpose shouldn't 429.
    #
    # Per-IP, guards POST /api/v1/auth/google (unauthenticated, does a
    # network call to Google's cert endpoint per request): generous enough
    # for real logins/retries, low enough to blunt brute force and free DoS
    # amplification.
    RATE_LIMIT_AUTH_PER_MINUTE: int = Field(default=10, ge=0)
    # Per-user, guards POST /api/v1/piece/preview, which the client polls in
    # a loop while the piece camera is open. Measured client cadences: the
    # web frontend polls at a 400ms floor between requests (~150/min in
    # steady state; see DETECT_INTERVAL_MS in
    # frontend/src/components/camera/live-piece-capture.tsx), and the iOS
    # app targets ~4Hz / a 250ms minimum interval, serialized on one
    # in-flight request at a time (~240/min; see
    # PiecePreviewThrottle.minInterval in
    # ios/Pussel/Features/Solve/PiecePreviewThrottle.swift). 300/min sits
    # comfortably above the faster (iOS) cadence so normal live-preview use
    # never gets 429'd, while still bounding a runaway or malicious client.
    RATE_LIMIT_PREVIEW_PER_MINUTE: int = Field(default=300, ge=0)

    # Whether to trust the inbound X-Forwarded-For header for per-IP rate
    # limiting (see app/rate_limit.py::_client_ip). Defaults to False:
    # nothing in this deployment (no Uvicorn --proxy-headers, no
    # ProxyHeadersMiddleware, no infra-level stripping) removes or rewrites
    # an inbound X-Forwarded-For, so a direct caller can set it to an
    # arbitrary/random value on every request and land in a fresh rate-limit
    # bucket each time -- trusting it by default would make the auth
    # brute-force limiter trivially bypassable by exactly the attacker it
    # exists to stop. Only flip this to True when the app is actually
    # deployed behind a reverse proxy that overwrites/appends to
    # X-Forwarded-For itself (e.g. Azure App Service's front end), so the
    # header's rightmost entry is guaranteed proxy-authored rather than
    # attacker-authored.
    TRUST_PROXY_HEADERS: bool = False

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

    def is_email_allowed(self, email: str) -> bool:
        """Check whether an email may authenticate, per ALLOWED_EMAILS.

        Comparison is case-insensitive and tolerant of surrounding whitespace
        on the configured values, since Google emails are case-insensitive in
        practice and a comma-separated env var may include stray spaces.

        Args:
            email: The candidate email address, as returned by Google.

        Returns:
            True if ALLOWED_EMAILS is empty (allow any account) or the email
            matches an entry in ALLOWED_EMAILS, False otherwise.
        """
        if not self.ALLOWED_EMAILS:
            return True
        normalized_allowed = {allowed.strip().lower() for allowed in self.ALLOWED_EMAILS}
        return email.strip().lower() in normalized_allowed


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create instance
settings = get_settings()

if not settings.ALLOWED_EMAILS:
    logger.warning(
        "ALLOWED_EMAILS is empty: any Google account with a verified email can sign in. "
        "Set ALLOWED_EMAILS to restrict access to this application."
    )

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
