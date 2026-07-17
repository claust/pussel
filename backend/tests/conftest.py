"""Shared pytest fixtures for the backend test suite."""

from typing import Generator

import pytest

from app.main import auth_rate_limiter, piece_preview_rate_limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiters() -> Generator[None, None, None]:
    """Reset the process-global rate limiters before every test.

    `auth_rate_limiter` and `piece_preview_rate_limiter` (app/main.py) hold
    process-wide state, same as `get_puzzle_store()` and other singletons
    used across this suite. Without this, one test file's calls to
    `/api/v1/auth/google` or `/api/v1/piece/preview` would leak counters into
    the next test (and TestClient requests share a single "IP"/user
    identity), causing order-dependent 429s.
    """
    auth_rate_limiter.reset()
    piece_preview_rate_limiter.reset()
    yield
    auth_rate_limiter.reset()
    piece_preview_rate_limiter.reset()
