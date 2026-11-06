"""
database.py – SQLite storage for scan history and bulk jobs using SQLite & SQLAlchemy.
Auto-creates tables on application startup.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings

settings = get_settings()


def get_db_file_path() -> Path:
    raw_url = settings.database_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    p = Path(raw_url)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_file_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db_sync() -> None:
    """Creates the SQLite tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Scans Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            platform TEXT NOT NULL,
            username TEXT NOT NULL,
            classification TEXT NOT NULL,
            risk_score REAL,
            result_json TEXT NOT NULL
        )
    """)

    # 2. Bulk Jobs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bulk_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            total_accounts INTEGER NOT NULL,
            processed_accounts INTEGER NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT
        )
    """)

    # Create indexes for fast filtering
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scans_platform ON scans(platform)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scans_classification ON scans(classification)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON scans(timestamp)")

    conn.commit()
    conn.close()


def insert_scan_record(
    platform: str,
    username: str,
    classification: str,
    risk_score: Optional[float],
    result_dict: Dict[str, Any],
) -> int:
    """Inserts a completed scan into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.utcnow().isoformat()
    json_str = json.dumps(result_dict, default=str)

    cursor.execute("""
        INSERT INTO scans (timestamp, platform, username, classification, risk_score, result_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (now_str, platform, username, classification, risk_score, json_str))

    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id


def query_scan_history(
    page: int = 1,
    limit: int = 20,
    platform: Optional[str] = None,
    classification: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Retrieves paginated scan history with optional platform and classification filters."""
    conn = get_connection()
    cursor = conn.cursor()

    where_clauses = []
    params = []

    if platform:
        where_clauses.append("platform = ?")
        params.append(platform)
    if classification:
        where_clauses.append("classification = ?")
        params.append(classification)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Count total
    cursor.execute(f"SELECT COUNT(*) FROM scans {where_sql}", params)
    total = cursor.fetchone()[0]

    # Fetch rows
    offset = (page - 1) * limit
    cursor.execute(f"""
        SELECT id, timestamp, platform, username, classification, risk_score, result_json
        FROM scans
        {where_sql}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """, params + [limit, offset])

    rows = cursor.fetchall()
    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "scanned_at": r["timestamp"],
            "platform": r["platform"],
            "username": r["username"],
            "classification": r["classification"],
            "risk_score": r["risk_score"],
            "result": json.loads(r["result_json"]) if r["result_json"] else {},
        })

    conn.close()
    return items, total


def query_analytics() -> Dict[str, Any]:
    """Computes aggregate metrics across scan records."""
    conn = get_connection()
    cursor = conn.cursor()

    # Total scans
    cursor.execute("SELECT COUNT(*) FROM scans")
    total_scans = cursor.fetchone()[0]

    # Classification counts
    cursor.execute("SELECT classification, COUNT(*) as cnt FROM scans GROUP BY classification")
    classification_counts = {row["classification"]: row["cnt"] for row in cursor.fetchall()}

    # Platform counts
    cursor.execute("SELECT platform, COUNT(*) as cnt FROM scans GROUP BY platform")
    platform_counts = {row["platform"]: row["cnt"] for row in cursor.fetchall()}

    # Average risk score (ignoring NULL scores for insufficient evidence)
    cursor.execute("SELECT AVG(risk_score) FROM scans WHERE risk_score IS NOT NULL")
    avg_score_row = cursor.fetchone()[0]
    avg_risk_score = round(float(avg_score_row), 1) if avg_score_row is not None else None

    # Daily scan counts (last 7 days)
    cursor.execute("""
        SELECT substr(timestamp, 1, 10) as day, COUNT(*) as cnt
        FROM scans
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
    """)
    daily_activity = [{"date": r["day"], "count": r["cnt"]} for r in cursor.fetchall()]

    conn.close()
    return {
        "total_scans": total_scans,
        "classification_counts": classification_counts,
        "platform_counts": platform_counts,
        "average_risk_score": avg_risk_score,
        "daily_activity": daily_activity,
        "mock": False,
    }
