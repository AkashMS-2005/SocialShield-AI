"""
routes/history.py – Paginated scan history endpoint backed by SQLite.
"""
from fastapi import APIRouter, Query
from typing import Optional

from app.database import query_scan_history
from app.schemas import HistoryResponse, HistoryItem, Classification

router = APIRouter()


@router.get("/history", response_model=HistoryResponse, summary="Scan history (paginated & filtered)")
async def get_history(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    platform: Optional[str] = Query(None, description="Filter by platform (Instagram, Facebook, X, YouTube)"),
    classification: Optional[str] = Query(None, description="Filter by classification tier"),
):
    """
    Returns real paginated scan records saved to SQLite.
    Supports filtering by platform and classification.
    """
    items_raw, total = query_scan_history(
        page=page,
        limit=limit,
        platform=platform,
        classification=classification,
    )

    items = []
    for r in items_raw:
        try:
            class_enum = Classification(r["classification"])
        except ValueError:
            class_enum = Classification.suspicious

        items.append(
            HistoryItem(
                id=r["id"],
                username=r["username"],
                platform=r["platform"],
                classification=class_enum,
                risk_score=r["risk_score"],
                scanned_at=r["scanned_at"],
            )
        )

    return HistoryResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
    )
