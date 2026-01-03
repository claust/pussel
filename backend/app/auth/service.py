"""Authentication service for Google OAuth and JWT management."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from jose import JWTError, jwt

from app.config import settings
from app.models.user_model import TokenPayload, User


class AuthService:
    """Service for handling authentication operations."""

    def __init__(self) -> None:
        """Initialize the auth service."""
        self._google_request = google_requests.Request()

    def verify_google_token(self, token: str) -> Optional[dict]:
        """Verify a Google ID token and extract user information.

        Args:
            token: The Google ID token to verify.

        Returns:
            Dictionary containing user info if valid, None otherwise.
        """
        try:
            # Verify the token with Google
            idinfo = google_id_token.verify_oauth2_token(
                token,
                self._google_request,
                settings.GOOGLE_CLIENT_ID,
            )

            # Verify the issuer
            if idinfo["iss"] not in ["accounts.google.com", "https://accounts.google.com"]:
                return None

            # Verify email is verified
            if not idinfo.get("email_verified", False):
                return None

            return {
                "sub": idinfo["sub"],
                "email": idinfo["email"],
                "name": idinfo.get("name", idinfo["email"].split("@")[0]),
                "picture": idinfo.get("picture"),
            }
        except ValueError:
            # Invalid token
            return None

    def create_access_token(self, user: User) -> tuple[str, int]:
        """Create a JWT access token for a user.

        Args:
            user: The user to create a token for.

        Returns:
            Tuple of (token string, expiration time in seconds).
        """
        expires_delta = timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
        expire = datetime.now(timezone.utc) + expires_delta

        payload = {
            "sub": user.id,
            "email": user.email,
            "name": user.name,
            "picture": user.picture,
            "exp": expire,
        }

        token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
        return token, settings.JWT_EXPIRE_MINUTES * 60

    def decode_token(self, token: str) -> Optional[TokenPayload]:
        """Decode and validate a JWT token.

        Args:
            token: The JWT token to decode.

        Returns:
            TokenPayload if valid, None otherwise.
        """
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
            )
            return TokenPayload(**payload)
        except JWTError:
            return None

    def get_user_from_token(self, token: str) -> Optional[User]:
        """Extract user information from a valid JWT token.

        Args:
            token: The JWT token to extract user from.

        Returns:
            User if token is valid, None otherwise.
        """
        payload = self.decode_token(token)
        if payload is None:
            return None

        return User(
            id=payload.sub,
            email=payload.email,
            name=payload.name,
            picture=payload.picture,
        )


@lru_cache()
def get_auth_service() -> AuthService:
    """Get cached auth service instance."""
    return AuthService()
