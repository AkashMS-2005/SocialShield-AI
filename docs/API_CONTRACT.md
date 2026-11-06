# SocialShield AI – API Contract

> **Version:** 0.1.0 | **Base URL:** `http://localhost:8000`  
> All endpoints return JSON. `mock: true` indicates the response is a placeholder pending ML pipeline integration.

---

## Ethical Notice

All outputs are **risk indicators only**. A high risk score does not prove an account is inauthentic. Missing data is surfaced explicitly in the `limitations` array — it is never silently treated as low risk.

---

## Endpoints

### `GET /`

Root info.

**Response 200**
```json
{
  "app": "SocialShield AI",
  "version": "0.1.0",
  "docs": "/docs",
  "health": "/api/health"
}
```

---

### `GET /api/health`

Health check with DB connectivity.

**Response 200**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "db": "ok"
}
```

---

### `POST /api/scan`

Analyse a single social media account.

**Request body**

| Field | Type | Required | Notes |
|---|---|---|---|
| `platform` | `"Instagram" \| "Facebook" \| "X" \| "YouTube"` | ✅ | Unknown values → 422 |
| `username` | `string` | ✅ | 1–200 chars |
| `followers` | `int \| null` | ❌ | ≥ 0; negative → 422 |
| `following` | `int \| null` | ❌ | ≥ 0 |
| `posts` | `int \| null` | ❌ | ≥ 0 |
| `account_age_days` | `int \| null` | ❌ | ≥ 0 |
| `has_profile_pic` | `bool \| null` | ❌ | |
| `bio` | `string \| null` | ❌ | max 2 000 chars |
| `post_text` | `string \| null` | ❌ | max 10 000 chars |

**Example request**
```json
{
  "platform": "Instagram",
  "username": "example_user",
  "followers": 120,
  "following": 980,
  "posts": 5,
  "account_age_days": 14,
  "has_profile_pic": false,
  "bio": "DM for promos",
  "post_text": "Win a free iPhone! Click the link in bio now!"
}
```

**Response 200**
```json
{
  "username": "example_user",
  "platform": "Instagram",
  "classification": "Suspicious",
  "risk_score": 0.61,
  "ml_result": {
    "model_used": null,
    "probability": null,
    "confidence": null,
    "mock": true
  },
  "nlp_result": {
    "spam_score": null,
    "sentiment": null,
    "suspicious_phrases": [],
    "mock": true
  },
  "suspicious_indicators": [
    "MOCK: This is a placeholder response – real analysis coming in Step 2."
  ],
  "explainability": {
    "top_features": [],
    "mock": true
  },
  "recommendations": [
    "Review the suspicious indicators listed above.",
    "Cross-reference with platform reporting tools."
  ],
  "data_availability": 1.0,
  "limitations": [],
  "scanned_at": "2026-11-06T06:00:00",
  "mock": true
}
```

**Validation error (422)** – negative followers:
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "followers"],
      "msg": "Value error, Value must be 0 or greater – negative counts are invalid.",
      "input": -5
    }
  ]
}
```

**Insufficient evidence example** (only platform + username provided):
```json
{
  "username": "sparse_user",
  "platform": "X",
  "classification": "Insufficient evidence",
  "risk_score": null,
  "data_availability": 0.0,
  "limitations": [
    "Fewer than 2 profile fields provided – cannot compute a reliable risk score.",
    "Account age unknown – recency signals unavailable.",
    "No post text provided – NLP analysis skipped."
  ],
  "mock": true
}
```

---

### `POST /api/analyze-text`

Standalone NLP analysis (no account metadata required).

**Request body**
```json
{
  "text": "Click here to win a prize! Limited time offer!",
  "platform": "Instagram"
}
```

**Response 200**
```json
{
  "text_preview": "Click here to win a prize! Limited time offer!",
  "nlp_result": {
    "spam_score": null,
    "sentiment": null,
    "suspicious_phrases": [],
    "mock": true
  },
  "suspicious_indicators": ["MOCK: NLP pipeline not yet connected."],
  "limitations": ["TF-IDF model not trained yet – text analysis unavailable."],
  "mock": true
}
```

---

### `POST /api/bulk-scan`

Scan up to 100 accounts in one request. Accepts the same `AccountInput` fields per account.

**Request body**
```json
{
  "accounts": [
    {
      "platform": "Instagram",
      "username": "user_a",
      "followers": 50,
      "following": 400
    },
    {
      "platform": "YouTube",
      "username": "user_b",
      "posts": 2,
      "account_age_days": 7
    }
  ]
}
```

**Response 200**
```json
{
  "total": 2,
  "results": [
    { "username": "user_a", "platform": "Instagram", "classification": "Suspicious", "mock": true, "..." : "..." },
    { "username": "user_b", "platform": "YouTube",   "classification": "Suspicious", "mock": true, "..." : "..." }
  ],
  "mock": true
}
```

---

### `GET /api/history?page=1&limit=20`

Paginated scan history.

**Query params**

| Param | Default | Range |
|---|---|---|
| `page` | 1 | ≥ 1 |
| `limit` | 20 | 1–100 |

**Response 200**
```json
{
  "items": [
    {
      "id": 1,
      "username": "mock_user_1",
      "platform": "Instagram",
      "classification": "Suspicious",
      "risk_score": 0.72,
      "scanned_at": "2026-11-06T10:00:00"
    }
  ],
  "total": 2,
  "page": 1,
  "limit": 20
}
```

---

### `GET /api/analytics`

Aggregate statistics over all scans.

**Response 200**
```json
{
  "total_scans": 42,
  "classification_counts": {
    "Low risk": 18,
    "Suspicious": 14,
    "High risk": 7,
    "Insufficient evidence": 3
  },
  "platform_counts": {
    "Instagram": 20,
    "Facebook": 10,
    "X": 8,
    "YouTube": 4
  },
  "average_risk_score": 0.48,
  "mock": true
}
```

---

## Classification Enum

| Value | Meaning |
|---|---|
| `Low risk` | No significant suspicious indicators detected |
| `Suspicious` | Some suspicious indicators – warrants closer inspection |
| `High risk` | Multiple strong suspicious indicators present |
| `Insufficient evidence` | Too little data to produce a meaningful assessment |

> **Note:** These labels describe *risk signals*, not identity. They must never be presented to end users as a verdict on whether an account is real or fake.

---

## Error Codes

| HTTP | Meaning |
|---|---|
| 422 | Validation error (negative numbers, unknown platform, missing required fields) |
| 500 | Internal server error – check `/api/health` |

