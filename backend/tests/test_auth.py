"""Test module for authentication functionality."""

import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from google.auth import exceptions as google_exceptions
from jose import jwt

from app.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_and_cleanup() -> Generator[None, None, None]:
    """Set up test environment and clean up after tests."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    yield
    if os.path.exists(settings.UPLOAD_DIR):
        shutil.rmtree(settings.UPLOAD_DIR)


def create_test_token(
    user_id: str = "test-user-id",
    email: str = "test@example.com",
    name: str = "Test User",
    picture: str | None = "https://example.com/picture.jpg",
    expired: bool = False,
) -> str:
    """Create a test JWT token.

    Args:
        user_id: The user ID to include in the token.
        email: The user email to include in the token.
        name: The user name to include in the token.
        picture: The user picture URL to include in the token.
        expired: Whether to create an expired token.

    Returns:
        A JWT token string.
    """
    if expired:
        expire = datetime.now(timezone.utc) - timedelta(hours=1)
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=1)

    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "picture": picture,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def get_auth_header(token: str) -> dict[str, str]:
    """Create an authorization header with the given token.

    Args:
        token: The JWT token to include.

    Returns:
        A dictionary with the Authorization header.
    """
    return {"Authorization": f"Bearer {token}"}


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_google_auth_success(self) -> None:
        """Test successful Google authentication."""
        mock_user_info = {
            "sub": "google-user-123",
            "email": "user@example.com",
            "name": "Google User",
            "picture": "https://example.com/avatar.jpg",
        }

        with patch("app.auth.service.AuthService.verify_google_token", return_value=mock_user_info):
            response = client.post(
                "/api/v1/auth/google",
                json={"id_token": "fake-google-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert data["user"]["email"] == "user@example.com"
        assert data["user"]["name"] == "Google User"

    def test_google_auth_invalid_token(self) -> None:
        """Test Google authentication with invalid token."""
        with patch("app.auth.service.AuthService.verify_google_token", return_value=None):
            response = client.post(
                "/api/v1/auth/google",
                json={"id_token": "invalid-google-token"},
            )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid Google token"

    def test_get_current_user_profile_success(self) -> None:
        """Test getting current user profile with valid token."""
        token = create_test_token()
        response = client.get(
            "/api/v1/auth/me",
            headers=get_auth_header(token),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["name"] == "Test User"

    def test_get_current_user_profile_no_token(self) -> None:
        """Test getting current user profile without token."""
        response = client.get("/api/v1/auth/me")

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_get_current_user_profile_expired_token(self) -> None:
        """Test getting current user profile with expired token."""
        token = create_test_token(expired=True)
        response = client.get(
            "/api/v1/auth/me",
            headers=get_auth_header(token),
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired token"


class TestProtectedEndpoints:
    """Tests for protected puzzle endpoints."""

    def test_upload_puzzle_requires_auth(self) -> None:
        """Test that puzzle upload requires authentication."""
        with open("test_puzzle.jpg", "wb") as f:
            f.write(b"fake image content")

        try:
            with open("test_puzzle.jpg", "rb") as f:
                files = {"file": ("test_puzzle.jpg", f, "image/jpeg")}
                response = client.post("/api/v1/puzzle/upload", files=files)

            assert response.status_code == 401
        finally:
            if os.path.exists("test_puzzle.jpg"):
                os.remove("test_puzzle.jpg")

    def test_upload_puzzle_with_auth(self) -> None:
        """Test puzzle upload with valid authentication."""
        token = create_test_token()

        with open("test_puzzle.jpg", "wb") as f:
            f.write(b"fake image content")

        try:
            with open("test_puzzle.jpg", "rb") as f:
                files = {"file": ("test_puzzle.jpg", f, "image/jpeg")}
                response = client.post(
                    "/api/v1/puzzle/upload",
                    files=files,
                    headers=get_auth_header(token),
                )

            assert response.status_code == 200
            assert "puzzle_id" in response.json()
        finally:
            if os.path.exists("test_puzzle.jpg"):
                os.remove("test_puzzle.jpg")

    def test_process_piece_requires_auth(self) -> None:
        """Test that piece processing requires authentication."""
        with open("test_piece.jpg", "wb") as f:
            f.write(b"fake piece content")

        try:
            with open("test_piece.jpg", "rb") as f:
                files = {"file": ("test_piece.jpg", f, "image/jpeg")}
                response = client.post("/api/v1/puzzle/test-id/piece", files=files)

            assert response.status_code == 401
        finally:
            if os.path.exists("test_piece.jpg"):
                os.remove("test_piece.jpg")

    def test_generate_piece_requires_auth(self) -> None:
        """Test that piece generation requires authentication."""
        response = client.post(
            "/api/v1/puzzle/test-id/generate-piece",
            json={"center_x": 0.5, "center_y": 0.5},
        )

        assert response.status_code == 401


class TestAuthService:
    """Tests for the AuthService class."""

    def test_create_and_decode_token(self) -> None:
        """Test creating and decoding a token."""
        from app.auth.service import get_auth_service
        from app.models.user_model import User

        auth_service = get_auth_service()
        user = User(
            id="test-id",
            email="test@example.com",
            name="Test User",
            picture="https://example.com/pic.jpg",
        )

        token, expires_in = auth_service.create_access_token(user)
        assert token is not None
        assert expires_in == settings.JWT_EXPIRE_MINUTES * 60

        decoded_user = auth_service.get_user_from_token(token)
        assert decoded_user is not None
        assert decoded_user.id == user.id
        assert decoded_user.email == user.email
        assert decoded_user.name == user.name

    def test_decode_invalid_token(self) -> None:
        """Test decoding an invalid token."""
        from app.auth.service import get_auth_service

        auth_service = get_auth_service()
        result = auth_service.decode_token("invalid-token")
        assert result is None

    def test_verify_google_token_invalid_issuer(self) -> None:
        """Test that tokens with invalid issuer are rejected."""
        from app.auth.service import get_auth_service

        auth_service = get_auth_service()

        # Mock google id_token verification to return a token with wrong issuer
        mock_idinfo = {
            "iss": "invalid-issuer",
            "sub": "123",
            "email": "test@example.com",
            "email_verified": True,
        }

        with patch("google.oauth2.id_token.verify_oauth2_token", return_value=mock_idinfo):
            result = auth_service.verify_google_token("fake-token")
            assert result is None

    def test_verify_google_token_unverified_email(self) -> None:
        """Test that tokens with unverified email are rejected."""
        from app.auth.service import get_auth_service

        auth_service = get_auth_service()

        mock_idinfo = {
            "iss": "accounts.google.com",
            "sub": "123",
            "email": "test@example.com",
            "email_verified": False,
        }

        with patch("google.oauth2.id_token.verify_oauth2_token", return_value=mock_idinfo):
            result = auth_service.verify_google_token("fake-token")
            assert result is None

    def test_verify_google_token_handles_value_error(self) -> None:
        """Test that ValueError exceptions are handled gracefully."""
        from app.auth.service import get_auth_service

        auth_service = get_auth_service()

        with patch(
            "google.oauth2.id_token.verify_oauth2_token",
            side_effect=ValueError("Invalid token format"),
        ):
            result = auth_service.verify_google_token("fake-token")
            assert result is None

    def test_verify_google_token_handles_google_auth_error(self) -> None:
        """Test that GoogleAuthError exceptions are handled gracefully."""
        from app.auth.service import get_auth_service

        auth_service = get_auth_service()

        with patch(
            "google.oauth2.id_token.verify_oauth2_token",
            side_effect=google_exceptions.GoogleAuthError("Authentication error"),
        ):
            result = auth_service.verify_google_token("fake-token")
            assert result is None
