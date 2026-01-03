"""FastAPI dependencies for authentication."""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.service import AuthService, get_auth_service
from app.models.user_model import User

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> User:
    """Get the current authenticated user from the request.

    Args:
        credentials: The HTTP Bearer credentials from the request.
        auth_service: The authentication service.

    Returns:
        The authenticated user.

    Raises:
        HTTPException: If authentication fails.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = auth_service.get_user_from_token(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_optional_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> Optional[User]:
    """Get the current user if authenticated, None otherwise.

    Args:
        credentials: The HTTP Bearer credentials from the request.
        auth_service: The authentication service.

    Returns:
        The authenticated user or None if not authenticated.
    """
    if credentials is None:
        return None

    return auth_service.get_user_from_token(credentials.credentials)
