"""
routes/history.py – GET /api/history

Returns paginated scan history from the database.
Currently returns mock data; will be backed by a real ORM model in a later step.
"""
from datetime import datetime
from fastapi import APIRouter, Query

from app.schemas import HistoryResponse, HistoryItem, Classification

router = APIRouter()


@router.get("/history", response_model=HistoryResponse, summary="Scan history (paginated)")
async def get_history(
    page: int  = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
):
    """
    Returns paginated scan history.
    **mock=true** – real data will come from the SQLite scan_history table once
    the ML pipeline is wired.
    """
    # MOCK – clearly labelled, schema-valid placeholder
    mock_items = [
        HistoryItem(
            id=1,
            username="mock_user_1",
            platform="Instagram",
            classification=Classification.suspicious,
            risk_score=0.72,
            scanned_at=datetime(2026, 11, 6, 10, 0, 0),
        ),
        HistoryItem(
            id=2,
            username="mock_user_2",
            platform="X",
            classification=Classification.low_risk,
            risk_score=0.18,
            scanned_at=datetime(2026, 11, 6, 10, 15, 0),
        ),
    ]
    return HistoryResponse(
        items=mock_items,
        total=2,
        page=page,
        limit=limit,
    )

