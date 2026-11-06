"""
routes/analytics.py – GET /api/analytics

Aggregate statistics over all scans.
Currently returns mock data; will query the DB in a later step.
"""
from fastapi import APIRouter
from app.schemas import AnalyticsResponse

router = APIRouter()


@router.get("/analytics", response_model=AnalyticsResponse, summary="Aggregate scan analytics")
async def get_analytics():
    """
    Returns aggregate statistics across all scans.
    **mock=true** – will be powered by real DB aggregates in a later step.
    """
    return AnalyticsResponse(
        total_scans=42,
        classification_counts={
            "Low risk": 18,
            "Suspicious": 14,
            "High risk": 7,
            "Insufficient evidence": 3,
        },
        platform_counts={
            "Instagram": 20,
            "Facebook": 10,
            "X": 8,
            "YouTube": 4,
        },
        average_risk_score=0.48,
        mock=True,
    )

