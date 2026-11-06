"""
routes/scan.py – Production implementation of scan endpoints.
Replaces all mocks with real ML, NLP, and SHAP execution.
"""
from __future__ import annotations

import csv
import io
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import (
    AccountInput,
    BulkScanInput,
    BulkScanResponse,
    NLPResult,
    ScanResponse,
    TextAnalysisInput,
    TextAnalysisResponse,
)
from app.services.nlp_service import analyze_text
from app.services.scan_service import execute_account_scan

router = APIRouter()


@router.post(
    "/scan",
    response_model=ScanResponse,
    summary="Scan a single account (Production ML + NLP + SHAP)",
    description="Validates account input, runs tabular ML, NLP text analysis, SHAP explainability, and the composite risk engine.",
)
async def scan_account(account: AccountInput) -> ScanResponse:
    """Executes a full multi-modal scan on a single account."""
    return execute_account_scan(account, persist=True)


@router.post(
    "/analyze-text",
    response_model=TextAnalysisResponse,
    summary="Standalone NLP text analysis",
    description="Evaluates bio or post content using TF-IDF + Logistic Regression and heuristic phishing/promotional rules.",
)
async def analyze_text_endpoint(payload: TextAnalysisInput) -> TextAnalysisResponse:
    """Performs isolated text analysis without requiring account metadata."""
    preview = payload.text[:120] + ("…" if len(payload.text) > 120 else "")
    nlp_res = analyze_text(bio=payload.text, posts=None)

    top_phrases = [p["phrase"] for p in nlp_res.get("top_suspicious_phrases", [])]

    nlp_result = NLPResult(
        spam_score=nlp_res.get("text_risk_score"),
        sentiment=nlp_res.get("text_risk_level"),
        suspicious_phrases=top_phrases,
        mock=False,
    )

    indicators = []
    for flag in nlp_res.get("rule_based_flags", []):
        indicators.append(flag)

    return TextAnalysisResponse(
        text_preview=preview,
        nlp_result=nlp_result,
        suspicious_indicators=indicators,
        limitations=nlp_res.get("limitations", []),
        mock=False,
    )


@router.post(
    "/bulk-scan",
    response_model=BulkScanResponse,
    summary="Bulk scan JSON list of up to 100 accounts",
)
async def bulk_scan_endpoint(payload: BulkScanInput) -> BulkScanResponse:
    """Executes scans across a list of accounts."""
    results = [execute_account_scan(acc, persist=True) for acc in payload.accounts]
    return BulkScanResponse(
        total=len(results),
        results=results,
        mock=False,
    )


@router.post(
    "/bulk-scan-csv",
    response_model=BulkScanResponse,
    summary="Bulk scan via CSV file upload",
    description="Upload a CSV file containing social media accounts. Automatically maps platform-specific headers and handles malformed rows gracefully.",
)
async def bulk_scan_csv_endpoint(
    file: UploadFile = File(..., description="CSV file with social media account details"),
) -> BulkScanResponse:
    """Uploads and scans accounts from a CSV file."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a valid .csv format.",
        )

    content = await file.read()
    try:
        text_stream = io.StringIO(content.decode("utf-8"))
    except UnicodeDecodeError:
        text_stream = io.StringIO(content.decode("latin-1"))

    reader = csv.DictReader(text_stream)
    if not reader.fieldnames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file contains no headers or is empty.",
        )

    results: List[ScanResponse] = []
    for row in reader:
        # Skip completely empty lines
        if not any(row.values()):
            continue
        try:
            # execute_account_scan uses platform adapters to safely normalise headers
            res = execute_account_scan(row, persist=True)
            results.append(res)
        except Exception:
            # Gracefully handle bad rows without crashing the batch
            continue

    if not results:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid account records could be processed from the CSV.",
        )

    return BulkScanResponse(
        total=len(results),
        results=results,
        mock=False,
    )
