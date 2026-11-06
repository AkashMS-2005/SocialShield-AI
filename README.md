# SocialShield AI

**Multi-Platform Fake Social Media Account Detection System**

A hackathon prototype that uses machine learning, NLP, and behavioural heuristics to surface *suspicious indicators* in social media accounts. It does **not** make identity claims — outputs are risk signals, not verdicts.

## Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite + Tailwind + Recharts |
| Backend | Python + FastAPI + Pydantic |
| ML | scikit-learn · XGBoost · Random Forest · Logistic Regression |
| NLP | TF-IDF |
| Explainability | SHAP |
| Data | Pandas · NumPy · NetworkX · SQLite · Joblib |

## Quick Start

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (Step 3+)
cd frontend
pnpm install
pnpm dev
```

## Project Structure

```
SocialShield AI/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── schemas.py       # Pydantic models & API contracts
│   │   ├── database.py      # SQLite session management
│   │   ├── config.py        # Settings
│   │   ├── routes/          # Endpoint routers
│   │   ├── services/        # Business logic
│   │   ├── ml/              # ML pipeline modules
│   │   ├── adapters/        # Platform input adapters
│   │   └── models/          # Saved model artefacts
│   ├── data/
│   │   ├── raw/             # Ingested CSVs
│   │   └── processed/       # Feature matrices
│   └── tests/               # pytest suite
├── frontend/                # React + Vite app (Step 3+)
├── docs/
│   └── API_CONTRACT.md      # Endpoint reference
└── .gitignore
```

## Ethical Notice

Outputs are **risk indicators only**. A bot-like behaviour pattern does not prove an account is fake. A new account is not automatically suspicious. Missing data is flagged as *insufficient evidence*, never treated as low risk.

