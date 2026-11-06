"""
tests/test_api_v2.py – Complete 12-test suite for production FastAPI backend.

Required Test Cases:
  1. Valid input scan (full realistic profile)
  2. Missing fields input (gracefully accepted, not zero risk)
  3. Negative numbers rejection (followers < 0 returns 422)
  4. Brand new account (age 0-1 days, weak signal rule respected)
  5. Established authentic account (healthy ratios, Low risk)
  6. Promotional / spam text NLP analysis (phishing indicators flagged)
  7. Normal conversational text NLP analysis (Low risk score)
  8. CSV upload bulk scan (valid CSV with YouTube/Instagram headers)
  9. Missing ML model error handling (ModelArtifactMissingError returns 503)
  10. Unknown platform rejection (returns 422 validation error)
  11. Missing optional features in CSV (imputation without crash)
  12. Health check endpoint (status ok, db connected)
"""
import io
import pytest
from starlette.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.ml.prediction import ModelArtifactMissingError
from app.schemas import Classification

client = TestClient(app, raise_server_exceptions=False)


# 1. Valid input scan
def test_case_01_valid_input_scan():
    payload = {
        "platform": "Instagram",
        "username": "authentic_traveler",
        "followers": 1500,
        "following": 320,
        "posts": 110,
        "account_age_days": 450,
        "has_profile_pic": True,
        "bio": "Travel writer and photographer. Exploring the world.",
        "post_text": "Beautiful morning in the Swiss Alps! Hiking trail was scenic and quiet.",
    }
    r = client.post("/api/scan", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["mock"] is False
    assert data["username"] == "authentic_traveler"
    assert data["platform"] == "Instagram"
    assert "classification" in data
    assert "risk_score" in data
    assert len(data["explainability"]["top_features"]) > 0
    assert data["data_availability"] > 0.8


# 2. Missing fields input (gracefully accepted, not zero risk)
def test_case_02_missing_fields_input():
    payload = {
        "platform": "X",
        "username": "sparse_user_99",
        # followers, following, posts all null
    }
    r = client.post("/api/scan", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["classification"] == "Insufficient evidence"
    assert data["risk_score"] is None
    assert any("not treated as zero risk" in l.lower() for l in data["limitations"])


# 3. Negative numbers rejection
def test_case_03_negative_numbers_rejection():
    payload = {
        "platform": "Facebook",
        "username": "negative_user",
        "followers": -50,
    }
    r = client.post("/api/scan", json=payload)
    assert r.status_code == 422
    data = r.json()
    assert data["error"] == "ValidationError"


# 4. Brand new account (age 0-1 days, weak signal rule)
def test_case_04_brand_new_account():
    payload = {
        "platform": "Instagram",
        "username": "fresh_person_2026",
        "followers": 10,
        "following": 25,
        "posts": 2,
        "account_age_days": 1,
        "has_profile_pic": True,
        "bio": "New account for university projects.",
    }
    r = client.post("/api/scan", json=payload)
    assert r.status_code == 200
    data = r.json()
    # Weak signal rule: age 1 day alone must NEVER push into High risk
    assert data["classification"] != "High risk"


# 5. Established authentic account
def test_case_05_established_authentic_account():
    payload = {
        "platform": "YouTube",
        "username": "tech_tutorials_official",
        "followers": 55000,
        "following": 120,
        "posts": 450,
        "account_age_days": 1200,
        "has_profile_pic": True,
        "bio": "Weekly Python and machine learning tutorials.",
    }
    r = client.post("/api/scan", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["classification"] == "Low risk"
    assert data["risk_score"] < 30.0


# 6. Promotional / spam text NLP analysis
def test_case_06_promotional_spam_text():
    payload = {
        "text": "WINNER SELECTED! DM to earn fast cash! Free crypto giveaway: http://bit.ly/cash-win",
        "platform": "Instagram",
    }
    r = client.post("/api/analyze-text", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["nlp_result"]["mock"] is False
    assert data["nlp_result"]["spam_score"] > 0.60
    assert len(data["suspicious_indicators"]) > 0


# 7. Normal conversational text NLP analysis
def test_case_07_normal_conversational_text():
    payload = {
        "text": "Finished reading a great biography on Alan Turing today. Fascinating look at early computation.",
        "platform": "X",
    }
    r = client.post("/api/analyze-text", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["nlp_result"]["spam_score"] < 0.40
    assert data["nlp_result"]["sentiment"] == "Low"


# 8. CSV upload bulk scan
def test_case_08_csv_upload_bulk_scan():
    csv_content = """platform,username,followers,following,posts,account_age_days,has_profile_pic,bio
Instagram,user_one,1200,300,50,200,1,Normal travel bio
YouTube,creator_two,500,50,30,150,true,Educational channel
"""
    files = {"file": ("accounts.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    r = client.post("/api/bulk-scan-csv", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["username"] == "user_one"
    assert data["results"][1]["platform"] == "YouTube"


# 9. Missing ML model error handling (503)
def test_case_09_missing_ml_model_503():
    with patch("app.services.scan_service.predict", side_effect=ModelArtifactMissingError("Artifact missing")):
        payload = {"platform": "Instagram", "username": "any_user", "followers": 100}
        r = client.post("/api/scan", json=payload)
        assert r.status_code == 503
        data = r.json()
        assert data["error"] == "ModelUnavailable"


# 10. Unknown platform rejection
def test_case_10_unknown_platform_rejection():
    payload = {
        "platform": "UnknownSocialNetwork123",
        "username": "test_user",
    }
    r = client.post("/api/scan", json=payload)
    assert r.status_code == 422


# 11. Missing optional features in CSV (graceful handling of sparse rows)
def test_case_11_missing_optional_features_in_csv():
    csv_content = """platform,username,followers,following
Facebook,sparse_fb_user,,
Instagram,minimal_user,450,
"""
    files = {"file": ("sparse.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    r = client.post("/api/bulk-scan-csv", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert data["results"][0]["classification"] == "Insufficient evidence"


# 12. Health check endpoint & SQLite integration
def test_case_12_health_and_analytics():
    # Health check
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["db"] == "ok"

    # Analytics reflects saved scans
    r_analytics = client.get("/api/analytics")
    assert r_analytics.status_code == 200
    data = r_analytics.json()
    assert data["mock"] is False
    assert data["total_scans"] >= 1
    assert isinstance(data["classification_counts"], dict)

    # History pagination
    r_history = client.get("/api/history?page=1&limit=5")
    assert r_history.status_code == 200
    hist = r_history.json()
    assert hist["total"] >= 1
    assert len(hist["items"]) <= 5

