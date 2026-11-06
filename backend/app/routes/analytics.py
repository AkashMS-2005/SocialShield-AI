"""
routes/analytics.py – Aggregate statistics across all real scans from SQLite.
"""
from fastapi import APIRouter
from app.database import query_analytics
from app.schemas import AnalyticsResponse

router = APIRouter()


@router.get("/analytics", response_model=AnalyticsResponse, summary="Aggregate scan analytics")
async def get_analytics():
    """
    Returns real aggregate counts, platform distributions, and average risk score
    from the SQLite scans database.
    """
    data = query_analytics()
    return AnalyticsResponse(
        total_scans=data["total_scans"],
        classification_counts=data["classification_counts"],
        platform_counts=data["platform_counts"],
        average_risk_score=data["average_risk_score"],
        mock=False,
    )
