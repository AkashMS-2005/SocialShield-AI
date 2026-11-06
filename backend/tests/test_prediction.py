"""
tests/test_prediction.py – Unit tests for production model loader and prediction function.
"""
import pytest
from app.schemas import AccountInput, Platform
from app.ml.prediction import predict, ModelArtifactMissingError


def test_prediction_suspicious_account():
    sample = {
        "username": "bot_998877",
        "followers": 1,
        "following": 4800,
        "posts": 2,
        "account_age_days": 1,
        "has_profile_pic": False,
        "bio": "Win money now! https://bit.ly/prize",
    }
    result = predict(sample)
    assert "calibrated_probability" in result
    assert result["calibrated_probability"] is not None
    assert result["classification"] in ["Suspicious", "High risk"]
    assert result["predicted_label"] == 1
    assert result["model_used"] is not None
    assert isinstance(result["missing_features"], list)


def test_prediction_genuine_account():
    sample = AccountInput(
        platform=Platform.instagram,
        username="authentic_photographer",
        followers=1200,
        following=350,
        posts=95,
        account_age_days=400,
        has_profile_pic=True,
        bio="Photographer in Seattle.",
    )
    result = predict(sample)
    assert result["classification"] == "Low risk"
    assert result["predicted_label"] == 0
    assert result["calibrated_probability"] < 0.35


def test_prediction_insufficient_evidence():
    """Sparse profile with almost all fields missing."""
    sample = AccountInput(
        platform=Platform.x,
        username="ghost_user",
    )
    result = predict(sample)
    assert result["classification"] == "Insufficient evidence"
    assert result["calibrated_probability"] is None
    assert result["predicted_label"] is None
    assert len(result["missing_features"]) >= 5

