"""User and authentication models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """User model representing an authenticated user."""

    id: str = Field(..., description="Unique user identifier (Google sub)")
    email: EmailStr = Field(..., description="User's email address")
    name: str = Field(..., description="User's display name")
    picture: Optional[str] = Field(None, description="URL to user's profile picture")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")


class TokenPayload(BaseModel):
    """JWT token payload model."""

    sub: str = Field(..., description="Subject (user ID)")
    email: str = Field(..., description="User's email")
    name: str = Field(..., description="User's name")
    picture: Optional[str] = Field(None, description="User's profile picture URL")
    exp: Optional[int] = Field(None, description="Token expiration timestamp")


class GoogleAuthRequest(BaseModel):
    """Request model for Google authentication."""

    id_token: str = Field(..., description="Google ID token from frontend")


class TokenResponse(BaseModel):
    """Response model for authentication endpoints."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: User = Field(..., description="Authenticated user information")


class AuthError(BaseModel):
    """Authentication error response model."""

    detail: str = Field(..., description="Error message")
