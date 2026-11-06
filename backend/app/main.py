"""
main.py – FastAPI application entry point.

Mounts all routers, configures CORS, and handles startup/shutdown events.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db_sync
from app.routes import health, scan, analytics, history

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB tables on startup (sync, avoids aiosqlite event-loop issues)."""
    init_db_sync()
    yield
    # shutdown logic goes here if needed


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Multi-Platform Fake Social Media Account Detection System. "
        "Outputs are **risk indicators only** and must not be treated as proof of identity."
    ),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router,    tags=["Health"])
app.include_router(scan.router,      prefix="/api", tags=["Scan"])
app.include_router(history.router,   prefix="/api", tags=["History"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", summary="Root – redirect hint")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
    }

