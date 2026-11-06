"""
services/scan_service.py – Core orchestration pipeline for scanning accounts.

Coordinates:
  1. Input normalisation (via platform adapters)
  2. Tabular ML model inference (calibrated probability)
  3. NLP text & bio analysis (spam probability & keyword flags)
  4. SHAP explainability (local feature attributions & explanation sentences)
  5. Composite risk engine (weight re-normalisation, behavioral rules, final score 0-100)
  6. Persistence to database
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from app.adapters.platforms import get_adapter_for_platform
from app.database import insert_scan_record
from app.ml.prediction import predict
from app.schemas import (
    AccountInput,
    Classification,
    ExplainabilityResult,
    MLResult,
    NLPResult,
    Platform,
    ScanResponse,
)
from app.services.explain_service import explain_account
from app.services.nlp_service import analyze_text
from app.services.risk_engine import risk_engine


def execute_account_scan(
    account_input: Union[AccountInput, Dict[str, Any]],
    persist: bool = True,
) -> ScanResponse:
    """
    Executes an end-to-end multi-modal scan of a single social media account.
    """
    # 1. Normalize input using appropriate platform adapter
    if isinstance(account_input, AccountInput):
        plat = account_input.platform
        adapter = get_adapter_for_platform(plat)
        account = adapter.normalize_input(account_input.model_dump())
    else:
        plat_str = account_input.get("platform", "Instagram")
        adapter = get_adapter_for_platform(plat_str)
        account = adapter.normalize_input(account_input)

    # 2. Run Tabular ML Model Prediction
    ml_output = predict(account)
    ml_prob = ml_output.get("calibrated_probability")
    model_name = ml_output.get("model_used")
    is_synthetic = ml_output.get("is_synthetic", False)

    ml_result = MLResult(
        model_used=model_name,
        probability=ml_prob,
        confidence="high" if ml_prob is not None and (ml_prob > 0.8 or ml_prob < 0.2) else "medium",
        mock=False,
    )

    # 3. Run NLP Analysis on Bio & Sample Posts
    nlp_output = analyze_text(bio=account.bio, posts=account.post_text)
    nlp_score = nlp_output.get("text_risk_score")
    top_phrases_dicts = nlp_output.get("top_suspicious_phrases", [])
    top_phrases_list = [p["phrase"] for p in top_phrases_dicts] if top_phrases_dicts else []

    nlp_result = NLPResult(
        spam_score=nlp_score,
        sentiment=nlp_output.get("text_risk_level"),
        suspicious_phrases=top_phrases_list,
        mock=False,
    )

    # 4. Run SHAP Explainability
    shap_output = explain_account(account, top_k=5)
    explain_result = ExplainabilityResult(
        top_features=shap_output.get("top_features", []),
        mock=False,
    )

    # 5. Composite Risk Engine Evaluation
    risk_assessment = risk_engine.compute_composite_risk(
        account=account,
        ml_probability=ml_prob,
        nlp_score=nlp_score,
    )

    # Combine behavioral and NLP indicators
    suspicious_indicators = []
    for ind in risk_assessment.get("indicators", []):
        suspicious_indicators.append(f"{ind['name']}: {ind['explanation']}")
    for flag in nlp_output.get("rule_based_flags", []):
        suspicious_indicators.append(f"Content indicator: {flag}")

    # Build ScanResponse
    response = ScanResponse(
        username=account.username,
        platform=account.platform.value,
        classification=risk_assessment["classification"],
        risk_score=risk_assessment["risk_score"],
        ml_result=ml_result,
        nlp_result=nlp_result,
        suspicious_indicators=suspicious_indicators,
        explainability=explain_result,
        recommendations=risk_assessment.get("recommendations", []),
        data_availability=risk_assessment.get("data_availability", 0.0),
        limitations=risk_assessment.get("limitations", []),
        scanned_at=datetime.utcnow(),
        mock=False,
    )

    # 6. Persist to SQLite Database
    if persist:
        try:
            insert_scan_record(
                platform=response.platform,
                username=response.username,
                classification=response.classification.value,
                risk_score=response.risk_score,
                result_dict=response.model_dump(),
            )
        except Exception:
            # Persistence failure should not crash the user-facing scan response
            pass

    return response

