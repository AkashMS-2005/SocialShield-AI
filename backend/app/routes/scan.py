"""
routes/scan.py – POST /api/scan, POST /api/analyze-text, POST /api/bulk-scan

All endpoints currently return MOCK, schema-valid responses labelled mock=true.
They will be replaced with real ML/NLP pipeline calls in later steps.
"""
from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter

from app.schemas import (
    AccountInput,
    BulkScanInput,
    BulkScanResponse,
    Classification,
    ExplainabilityResult,
    MLResult,
    NLPResult,
    ScanResponse,
    TextAnalysisInput,
    TextAnalysisResponse,
)

router = APIRouter()


def _mock_scan_response(account: AccountInput) -> ScanResponse:
    """
    Builds a schema-valid, clearly-labelled mock ScanResponse.

    Data availability is computed honestly from the input fields.
    Classification is 'Insufficient evidence' when fewer than 3 fields are provided,
    so we never claim false certainty.
    """
    # Count how many optional fields were actually provided
    optional_fields = [
        account.followers,
        account.following,
        account.posts,
        account.account_age_days,
        account.has_profile_pic,
        account.bio,
        account.post_text,
    ]
    provided = sum(1 for f in optional_fields if f is not None)
    data_availability = round(provided / len(optional_fields), 2)

    # Mock classification based on data availability
    if provided < 2:
        classification = Classification.insufficient_evidence
        risk_score = None
        limitations = [
            "Fewer than 2 profile fields provided – cannot compute a reliable risk score.",
            "Account age unknown – recency signals unavailable.",
            "No post text provided – NLP analysis skipped.",
        ]
        indicators: list[str] = []
    else:
        # Placeholder heuristics – NOT real ML
        classification = Classification.suspicious
        risk_score = 0.61
        limitations = [
            "Risk score is a placeholder – real ML pipeline not yet connected.",
            "Post content not provided – NLP analysis skipped." if not account.post_text else "",
        ]
        limitations = [l for l in limitations if l]
        indicators = [
            "MOCK: This is a placeholder response – real analysis coming in Step 2.",
        ]

    return ScanResponse(
        username=account.username,
        platform=account.platform.value,
        classification=classification,
        risk_score=risk_score,
        ml_result=MLResult(
            model_used=None,
            probability=None,
            confidence=None,
            mock=True,
        ),
        nlp_result=NLPResult(
            spam_score=None,
            sentiment=None,
            suspicious_phrases=[],
            mock=True,
        ),
        suspicious_indicators=indicators,
        explainability=ExplainabilityResult(top_features=[], mock=True),
        recommendations=[
            "Provide more profile fields to enable a more accurate assessment.",
        ] if provided < 4 else [
            "Review the suspicious indicators listed above.",
            "Cross-reference with platform reporting tools.",
        ],
        data_availability=data_availability,
        limitations=limitations,
        scanned_at=datetime.utcnow(),
        mock=True,
    )


@router.post("/scan", response_model=ScanResponse, summary="Scan a single account")
async def scan_account(account: AccountInput) -> ScanResponse:
    """
    Analyses a single social media account for suspicious indicators.

    **Currently returns mock data** (mock=true in response).  
    The real ML + NLP pipeline will be connected in a later step.

    Outputs are risk *indicators* only. A high risk score does not prove
    an account is inauthentic.
    """
    return _mock_scan_response(account)


@router.post(
    "/analyze-text",
    response_model=TextAnalysisResponse,
    summary="Standalone NLP text analysis",
)
async def analyze_text(payload: TextAnalysisInput) -> TextAnalysisResponse:
    """
    Runs NLP analysis on a text snippet without requiring account metadata.

    **Currently returns mock data** (mock=true in response).
    """
    preview = payload.text[:120] + ("…" if len(payload.text) > 120 else "")
    return TextAnalysisResponse(
        text_preview=preview,
        nlp_result=NLPResult(
            spam_score=None,
            sentiment=None,
            suspicious_phrases=[],
            mock=True,
        ),
        suspicious_indicators=["MOCK: NLP pipeline not yet connected."],
        limitations=["TF-IDF model not trained yet – text analysis unavailable."],
        mock=True,
    )


@router.post("/bulk-scan", response_model=BulkScanResponse, summary="Bulk scan up to 100 accounts")
async def bulk_scan(payload: BulkScanInput) -> BulkScanResponse:
    """
    Accepts a list of AccountInput records and returns a ScanResponse for each.

    **Currently returns mock data** (mock=true in response).
    Max 100 accounts per request.
    """
    results = [_mock_scan_response(acc) for acc in payload.accounts]
    return BulkScanResponse(
        total=len(results),
        results=results,
        mock=True,
    )

