"""
tests/test_api.py – Smoke tests using Starlette's synchronous TestClient.

TestClient properly manages the FastAPI lifespan (including DB init) in
a background thread, avoiding asyncio event-loop conflicts with pytest.

Run with:
    cd backend
    venv\\Scripts\\python -m pytest tests/ -v
"""
import pytest
from starlette.testclient import TestClient
from app.main import app

# Single shared client for the whole module (lifespan fires once)
client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Root & Health
# ---------------------------------------------------------------------------

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "app" in data
    assert data["docs"] == "/docs"


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "db" in data


# ---------------------------------------------------------------------------
# POST /api/scan
# ---------------------------------------------------------------------------

def test_scan_mock_full_data():
    payload = {
        "platform": "Instagram",
        "username": "test_user",
        "followers": 120,
        "following": 980,
        "posts": 5,
        "account_age_days": 14,
        "has_profile_pic": False,
        "bio": "DM for promos",
        "post_text": "Win a free iPhone! Click the link in bio now!",
    }
    r = client.post("/api/scan", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["mock"] is True
    assert data["username"] == "test_user"
    assert data["platform"] == "Instagram"
    assert "classification" in data
    assert 0.0 <= data["data_availability"] <= 1.0
    assert isinstance(data["limitations"], list)
    assert data["ml_result"]["mock"] is True
    assert data["nlp_result"]["mock"] is True


def test_scan_mock_insufficient_data():
    """Only platform + username → Insufficient evidence, no risk score."""
    r = client.post("/api/scan", json={"platform": "X", "username": "sparse_user"})
    assert r.status_code == 200
    data = r.json()
    assert data["classification"] == "Insufficient evidence"
    assert data["risk_score"] is None
    assert data["data_availability"] == 0.0


def test_scan_rejects_negative_followers():
    r = client.post("/api/scan", json={
        "platform": "Facebook", "username": "bad", "followers": -5
    })
    assert r.status_code == 422


def test_scan_rejects_unknown_platform():
    r = client.post("/api/scan", json={
        "platform": "TikTok", "username": "bad"
    })
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/analyze-text
# ---------------------------------------------------------------------------

def test_analyze_text_mock():
    r = client.post("/api/analyze-text", json={
        "text": "Click here to win a prize! Limited time offer!"
    })
    assert r.status_code == 200
    data = r.json()
    assert data["mock"] is True
    assert "text_preview" in data
    assert isinstance(data["suspicious_indicators"], list)


# ---------------------------------------------------------------------------
# POST /api/bulk-scan
# ---------------------------------------------------------------------------

def test_bulk_scan_mock():
    payload = {"accounts": [
        {"platform": "Instagram", "username": "user_a", "followers": 50},
        {"platform": "YouTube",   "username": "user_b", "posts": 2},
    ]}
    r = client.post("/api/bulk-scan", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["mock"] is True


def test_bulk_scan_rejects_empty():
    r = client.post("/api/bulk-scan", json={"accounts": []})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------

def test_history_mock():
    r = client.get("/api/history")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data


# ---------------------------------------------------------------------------
# GET /api/analytics
# ---------------------------------------------------------------------------

def test_analytics_mock():
    r = client.get("/api/analytics")
    assert r.status_code == 200
    data = r.json()
    assert data["mock"] is True
    assert "total_scans" in data
    assert "classification_counts" in data
    assert "platform_counts" in data
