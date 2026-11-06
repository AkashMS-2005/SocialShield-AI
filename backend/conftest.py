"""
conftest.py – pytest fixtures for the SocialShield backend test suite.

Creates a test-scoped FastAPI app with the lifespan skipped so that
aiosqlite never tries to initialise the DB during unit tests.
"""
from __future__ import annotations

import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport

from app.main import app as _real_app


@asynccontextmanager
async def _noop_lifespan(app):
    """Replace DB-initialising lifespan with a no-op for tests."""
    yield


@pytest.fixture()
async def client():
    """
    Async HTTP client wired directly to the FastAPI app.
    The lifespan (DB init) is bypassed so tests are self-contained.
    """
    _real_app.router.lifespan_context = _noop_lifespan
    async with AsyncClient(
        transport=ASGITransport(app=_real_app),
        base_url="http://test",
    ) as c:
        yield c
