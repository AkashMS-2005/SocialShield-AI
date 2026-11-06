"""
routes/health.py – /api/health endpoint.
"""
import sqlite3
from fastapi import APIRouter
from app.config import get_settings
from app.schemas import HealthResponse

router = APIRouter()
settings = get_settings()


@router.get("/api/health", response_model=HealthResponse, summary="Health check")
def health_check():
    """Returns service status and DB connectivity."""
    db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        db=db_status,
    )


