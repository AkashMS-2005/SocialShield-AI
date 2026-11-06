"""
schemas.py – Pydantic v2 models for API contracts and internal feature schema.

Design principles (from project rules):
  - All profile/numeric fields are Optional; None means "not provided / unknown".
  - Negative numbers are rejected with explicit validation messages.
  - Classification is an enum, never a free string.
  - Risk score is always accompanied by data_availability and limitations.
  - Language avoids identity claims ("suspicious indicators", not "this is fake").
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
import math

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Platform(str, Enum):
    instagram = "Instagram"
    facebook  = "Facebook"
    x         = "X"
    youtube   = "YouTube"


class Classification(str, Enum):
    low_risk             = "Low risk"
    suspicious           = "Suspicious"
    high_risk            = "High risk"
    insufficient_evidence = "Insufficient evidence"


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class AccountInput(BaseModel):
    """
    Manual-entry or CSV-row input for a single account.

    All numeric/profile fields are Optional so that partial data is accepted
    instead of rejected.  Missing fields are treated as *unknown*, not zero-risk.
    """
    platform: Platform = Field(..., description="Social platform")
    username: str      = Field(..., min_length=1, max_length=200, description="Account handle or URL")

    # Numeric profile signals – all optional
    followers:        Optional[int]   = Field(None, ge=0, description="Follower count")
    following:        Optional[int]   = Field(None, ge=0, description="Following count")
    posts:            Optional[int]   = Field(None, ge=0, description="Total post count")
    account_age_days: Optional[int]   = Field(None, ge=0, description="Days since account creation")

    # Boolean signals – optional
    has_profile_pic:  Optional[bool]  = Field(None, description="Whether a profile picture is set")

    # Text fields – optional
    bio:       Optional[str] = Field(None, max_length=2000, description="Profile bio text")
    post_text: Optional[str] = Field(None, max_length=10000, description="Sample post content for NLP")

    @field_validator("followers", "following", "posts", "account_age_days", mode="before")
    @classmethod
    def reject_negative(cls, v: Any) -> Any:
        """Reject negative numbers with an explicit message."""
        if v is not None:
            try:
                val = int(v)
            except (TypeError, ValueError):
                raise ValueError("Must be an integer or null.")
            if val < 0:
                raise ValueError("Value must be 0 or greater – negative counts are invalid.")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "platform": "Instagram",
            "username": "example_user",
            "followers": 120,
            "following": 980,
            "posts": 5,
            "account_age_days": 14,
            "has_profile_pic": False,
            "bio": "DM for promos",
            "post_text": "Win a free iPhone! Click the link in bio now!",
        }
    }}


class TextAnalysisInput(BaseModel):
    """Input for standalone text/NLP analysis (no account metadata required)."""
    text: str    = Field(..., min_length=1, max_length=10000)
    platform: Optional[Platform] = Field(None)


class BulkScanInput(BaseModel):
    """
    Bulk scan: list of AccountInput records.
    Max 100 per request to keep response times reasonable.
    """
    accounts: list[AccountInput] = Field(..., min_length=1, max_length=100)


# ---------------------------------------------------------------------------
# Output / response schemas
# ---------------------------------------------------------------------------

class MLResult(BaseModel):
    """Results from the ML ensemble (populated when model is trained)."""
    model_used:  Optional[str]  = None
    probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence:  Optional[str]  = None   # "low" | "medium" | "high"
    mock:        bool           = False


class NLPResult(BaseModel):
    """Results from TF-IDF / text analysis."""
    spam_score:         Optional[float] = Field(None, ge=0.0, le=1.0)
    sentiment:          Optional[str]   = None
    suspicious_phrases: list[str]       = Field(default_factory=list)
    mock:               bool            = False


class ExplainabilityResult(BaseModel):
    """SHAP feature contributions (populated when model is available)."""
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    # e.g. [{"feature": "following_ratio", "contribution": 0.42, "direction": "increases_risk"}]
    mock:         bool                 = False


class ScanResponse(BaseModel):
    """
    Full response for a single account scan.

    Important fields:
      - classification: enum, never a free string
      - risk_score: float 0–1, always accompanied by data_availability
      - limitations: explicit list of what we could NOT assess
      - data_availability: fraction of fields that had data (0.0–1.0)
    """
    username:              str
    platform:              str
    classification:        Classification
    risk_score:            Optional[float] = Field(None, ge=0.0, le=100.0,
                                                   description="Composite risk score 0–100. "
                                                               "Null when evidence is insufficient.")
    ml_result:             MLResult
    nlp_result:            NLPResult
    suspicious_indicators: list[str]       = Field(default_factory=list,
                                                   description="Observed signals – NOT proof of inauthenticity.")
    explainability:        ExplainabilityResult
    recommendations:       list[str]       = Field(default_factory=list)
    data_availability:     float           = Field(..., ge=0.0, le=1.0,
                                                   description="Fraction of input fields that contained data.")
    limitations:           list[str]       = Field(default_factory=list,
                                                   description="What could not be assessed due to missing data.")
    scanned_at:            datetime        = Field(default_factory=datetime.utcnow)
    mock:                  bool            = Field(False, description="True while real ML pipeline is not wired up.")


class TextAnalysisResponse(BaseModel):
    text_preview:       str
    nlp_result:         NLPResult
    suspicious_indicators: list[str] = Field(default_factory=list)
    limitations:        list[str]    = Field(default_factory=list)
    mock:               bool         = False


class BulkScanResponse(BaseModel):
    total:   int
    results: list[ScanResponse]
    mock:    bool = False


class HistoryItem(BaseModel):
    id:             int
    username:       str
    platform:       str
    classification: Classification
    risk_score:     Optional[float]
    scanned_at:     datetime


class HistoryResponse(BaseModel):
    items:  list[HistoryItem]
    total:  int
    page:   int
    limit:  int


class AnalyticsResponse(BaseModel):
    total_scans:          int
    classification_counts: dict[str, int]
    platform_counts:      dict[str, int]
    average_risk_score:   Optional[float]
    mock:                 bool = False


class HealthResponse(BaseModel):
    status:  str
    version: str
    db:      str

