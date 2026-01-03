"""Authentication module for the puzzle solver application."""

from app.auth.dependencies import get_current_user, get_optional_user
from app.auth.service import AuthService, get_auth_service

__all__ = ["AuthService", "get_auth_service", "get_current_user", "get_optional_user"]
