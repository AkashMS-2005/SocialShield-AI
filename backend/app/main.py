"""
main.py – FastAPI application entry point with global exception handlers and CORS.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import sqlite3

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import init_db_sync
from app.ml.prediction import ModelArtifactMissingError
from app.routes import health, scan, analytics, history

settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
init_db_sync()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB tables on startup."""
    init_db_sync()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Multi-Platform Fake Social Media Account Detection System.\n\n"
        "Outputs represent **probabilistic suspicious indicators** and must **never** "
        "be treated as proof of identity or malicious intent (Working Rule #3)."
    ),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Global Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ModelArtifactMissingError)
async def model_missing_exception_handler(request: Request, exc: ModelArtifactMissingError):
    """Returns 503 Service Unavailable when the ML model has not yet been trained."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "ModelUnavailable",
            "message": str(exc),
            "detail": "The machine learning classifier has not been trained or the artifact is missing on disk.",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Returns 422 Unprocessable Entity with clear validation message."""
    errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err["loc"] if loc != "body")
        errors.append({
            "field": field,
            "message": err["msg"],
            "type": err["type"],
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Input validation failed. Please review the highlighted fields.",
            "details": errors,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Returns 400 Bad Request for unknown platforms or illegal domain values."""
    msg = str(exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "BadRequest",
            "message": msg,
        },
    )


@app.exception_handler(sqlite3.Error)
async def database_exception_handler(request: Request, exc: sqlite3.Error):
    """Returns 500 Internal Server Error for database faults."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "DatabaseError",
            "message": "A database operation failed.",
            "detail": str(exc),
        },
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
@app.get("/", summary="Root – system status hint")
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
    }
