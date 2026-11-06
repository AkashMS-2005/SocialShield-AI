"""
tests/test_risk_engine.py – Test suite for the composite risk engine.

Covers the 6 required cases:
  1. Newly created genuine-looking account (age 1-2 days, authentic bio, low followers, but never High Risk).
  2. Established authentic account (healthy ratios, low risk score).
  3. Obvious spam bot profile (extreme ratio, zero followers, 5000 following, shorteners, high risk score).
  4. Missing NLP (NLP unavailable, ML & behavioral re-normalized properly, limitation noted).
  5. Nearly empty input (only username provided -> Insufficient evidence, risk_score is None).
  6. Conflicting signals (high followers like an influencer, but suspicious link in bio).
"""
import pytest
from app.schemas import AccountInput, Classification, Platform
from app.services.risk_engine import risk_engine


def test_1_newly_created_genuine_looking_account():
    """A new account alone must NEVER be pushed into High Risk."""
    account = AccountInput(
        platform=Platform.x,
        username="sarah_student",
        followers=5,
        following=15,
        posts=2,
        account_age_days=1,
        has_profile_pic=True,
        bio="Student at UW. Learning web development.",
    )
    # ML probability is low, NLP is safe, behavioral detects only new account
    res = risk_engine.compute_composite_risk(
        account=account,
        ml_probability=0.20,
        nlp_score=0.10,
    )
    assert res["classification"] != Classification.high_risk
    assert res["risk_score"] < 70.0
    assert res["classification"] in [Classification.low_risk, Classification.suspicious]
    # Check that new account indicator is recorded
    indicator_names = [i["name"] for i in res["indicators"]]
    assert "Newly created account" in indicator_names


def test_2_established_authentic_account():
    """Established account with balanced ratios and genuine text should be Low Risk."""
    account = AccountInput(
        platform=Platform.instagram,
        username="art_studio_la",
        followers=2500,
        following=400,
        posts=180,
        account_age_days=650,
        has_profile_pic=True,
        bio="Contemporary art in Los Angeles. Gallery open Wed-Sun.",
    )
    res = risk_engine.compute_composite_risk(
        account=account,
        ml_probability=0.05,
        nlp_score=0.02,
    )
    assert res["classification"] == Classification.low_risk
    assert res["risk_score"] < 30.0
    assert len(res["indicators"]) == 0
    assert "No immediate suspicious activity" in res["recommendations"][0]


def test_3_obvious_spam_bot_profile():
    """Extreme ratio, 0 followers, 4900 following, missing avatar, shorteners -> High Risk."""
    account = AccountInput(
        platform=Platform.instagram,
        username="free_cash_bot_8877",
        followers=1,
        following=4900,
        posts=2,
        account_age_days=2,
        has_profile_pic=False,
        bio="WIN $5000 FAST! Click: https://bit.ly/cash-deal",
    )
    res = risk_engine.compute_composite_risk(
        account=account,
        ml_probability=0.95,
        nlp_score=0.90,
    )
    assert res["classification"] == Classification.high_risk
    assert res["risk_score"] >= 70.0
    indicator_names = [i["name"] for i in res["indicators"]]
    assert "Extreme following-to-followers ratio" in indicator_names
    assert "Suspicious URL in profile bio" in indicator_names
    assert res["calibrated_ml_probability"] == 0.95


def test_4_missing_nlp_renormalization():
    """When NLP is absent, weights are re-normalized across remaining modalities and noted."""
    account = AccountInput(
        platform=Platform.facebook,
        username="john_doe_99",
        followers=150,
        following=200,
        posts=20,
        account_age_days=100,
        has_profile_pic=True,
        bio=None,
    )
    # nlp_score is None
    res = risk_engine.compute_composite_risk(
        account=account,
        ml_probability=0.30,
        nlp_score=None,
    )
    assert res["risk_score"] is not None
    assert "nlp" not in res["available_modalities"]
    assert "ml" in res["available_modalities"]
    assert "behavioral" in res["available_modalities"]
    # Check that limitation mentions NLP unavailability
    assert any("NLP post/bio analysis was unavailable" in l for l in res["limitations"])


def test_5_nearly_empty_input_insufficient_evidence():
    """Only username provided -> Insufficient evidence, risk_score is None."""
    account = AccountInput(
        platform=Platform.instagram,
        username="mystery_person",
    )
    res = risk_engine.compute_composite_risk(
        account=account,
        ml_probability=None,
        nlp_score=None,
    )
    assert res["classification"] == Classification.insufficient_evidence
    assert res["risk_score"] is None
    assert any("Insufficient evidence" in l for l in res["limitations"])
    assert any("Missing data was NOT treated as zero risk" in l for l in res["limitations"])


def test_6_conflicting_signals():
    """High follower count (authentic indicator) but suspicious link shortener in bio."""
    account = AccountInput(
        platform=Platform.instagram,
        username="crypto_promoter_top",
        followers=50000,
        following=150,
        posts=450,
        account_age_days=800,
        has_profile_pic=True,
        bio="Crypto insights! Free tokens here: http://bit.ly/token-drop",
    )
    # ML is low (0.15) due to high followers and age, but NLP / behavioral flag the link (0.80)
    res = risk_engine.compute_composite_risk(
        account=account,
        ml_probability=0.15,
        nlp_score=0.85,
    )
    # Should land in Suspicious due to the conflicting high-risk text/link signal
    assert res["classification"] == Classification.suspicious
    assert 30.0 <= res["risk_score"] < 70.0
    indicator_names = [i["name"] for i in res["indicators"]]
    assert "Suspicious URL in profile bio" in indicator_names

