"""FastAPI dependencies for authentication."""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.service import AuthService, get_auth_service
from app.models.user_model import User
from app.services.puzzle_store import PuzzleRecord, get_puzzle_store

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


async def get_owned_puzzle(
    puzzle_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
) -> PuzzleRecord:
    """Look up a puzzle and verify the current user owns it.

    Args:
        puzzle_id: The puzzle's id, taken from the request path.
        current_user: The authenticated user.

    Returns:
        The requested puzzle's record.

    Raises:
        HTTPException: 404 when the puzzle doesn't exist, or exists but is
            owned by someone else. A 403 would confirm the id exists (just
            not for this caller), leaking information about other users'
            puzzles — so both cases are indistinguishable 404s.
    """
    puzzle = get_puzzle_store().get(puzzle_id)
    if puzzle is None or puzzle.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Puzzle not found")
    return puzzle
