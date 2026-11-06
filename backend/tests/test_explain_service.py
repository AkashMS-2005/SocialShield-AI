"""
tests/test_explain_service.py – Unit tests for explainability service.

Tests:
  1. Suspicious account receives positive contributions towards suspicious.
  2. Genuine account receives negative contributions towards genuine.
  3. Method label correctly indicates "SHAP on base estimator".
  4. Latency benchmark (< 1.0s for cached inference).
  5. Insufficient evidence input handled safely.
"""
import time
from app.schemas import AccountInput, Platform
from app.services.explain_service import explain_account, init_explainer


def test_init_and_latency():
    # Warm up cache
    init_explainer()

    sample = AccountInput(
        platform=Platform.instagram,
        username="speed_test",
        followers=500,
        following=300,
        posts=40,
        account_age_days=100,
        has_profile_pic=True,
    )
    start_time = time.time()
    res = explain_account(sample)
    duration = time.time() - start_time

    assert duration < 1.0, f"Explanation took {duration:.2f}s, expected < 1.0s"
    assert "SHAP" in res["method"]
    assert res["is_sufficient_evidence"] is True


def test_suspicious_account_explanation():
    sample = {
        "username": "spambot_999999",
        "followers": 2,
        "following": 4900,
        "posts": 1,
        "account_age_days": 1,
        "has_profile_pic": False,
        "bio": "Click: https://bit.ly/prize",
    }
    res = explain_account(sample, top_k=5)

    assert res["method"].startswith("SHAP")
    assert len(res["top_features"]) <= 5
    directions = [f["direction"] for f in res["top_features"]]
    # At least some features must push toward suspicious
    assert "toward suspicious" in directions
    assert all("sentence" in f and len(f["sentence"]) > 0 for f in res["top_features"])


def test_genuine_account_explanation():
    sample = AccountInput(
        platform=Platform.instagram,
        username="legit_artist",
        followers=3500,
        following=200,
        posts=150,
        account_age_days=800,
        has_profile_pic=True,
        bio="Digital artist and creator.",
    )
    res = explain_account(sample, top_k=5)

    assert res["method"].startswith("SHAP")
    directions = [f["direction"] for f in res["top_features"]]
    # At least some features must push toward genuine
    assert "toward genuine" in directions


def test_sparse_account_insufficient_evidence():
    sample = AccountInput(
        platform=Platform.x,
        username="empty_user",
    )
    res = explain_account(sample)
    assert res["is_sufficient_evidence"] is False
    assert len(res["top_features"]) == 0

