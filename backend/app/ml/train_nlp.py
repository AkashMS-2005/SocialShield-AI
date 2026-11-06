"""
ml/train_nlp.py – Train TF-IDF + Logistic Regression NLP model for spam/phishing text detection.

Using the UCI SMS Spam Collection dataset (downloaded in Step 2).
Saves model pipeline to backend/app/models/nlp_spam_model.joblib
and generates docs/NLP_EVALUATION.md with genuine test metrics.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BACKEND_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BACKEND_DIR / "data" / "raw" / "SMSSpamCollection"
MODELS_DIR = BACKEND_DIR / "app" / "models"
DOCS_DIR = BACKEND_DIR.parent / "docs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

NLP_MODEL_PATH = MODELS_DIR / "nlp_spam_model.joblib"
NLP_METADATA_PATH = MODELS_DIR / "nlp_metadata.json"
NLP_REPORT_PATH = DOCS_DIR / "NLP_EVALUATION.md"

RANDOM_SEED = 42


def load_sms_dataset() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {RAW_DATA_PATH}")

    df = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
    )
    # Convert labels: 'spam' -> 1, 'ham' -> 0
    df["target"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    # Drop empty texts if any
    df = df.dropna(subset=["text"])
    return df


def train_nlp_model() -> Dict[str, Any]:
    print("=" * 75)
    print("SocialShield AI – NLP Spam/Phishing Model Training")
    print("=" * 75)

    df = load_sms_dataset()
    print(f"Loaded {len(df)} records from {RAW_DATA_PATH.name}")
    print(f"Class breakdown: {dict(df['label'].value_counts())}")

    # 1. Stratified Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["target"],
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["target"],
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Build TF-IDF (word 1-2 grams) + Logistic Regression Pipeline
    # class_weight='balanced' addresses the 86.6% ham vs 13.4% spam imbalance
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
                strip_accents="unicode",
                stop_words="english",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=5.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
    ])

    print("Fitting TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    # 3. Evaluate on untouched Test set
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, test_preds))
    prec = float(precision_score(y_test, test_preds, zero_division=0))
    rec = float(recall_score(y_test, test_preds, zero_division=0))
    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    auc = float(roc_auc_score(y_test, test_probs))
    cm = confusion_matrix(y_test, test_preds)
    brier = float(brier_score_loss(y_test, test_probs))

    print(f"\n--- Held-Out Test Set Performance ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Brier:     {brier:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 4. Save Model Pipeline
    joblib.dump(pipeline, NLP_MODEL_PATH)
    print(f"✅ Saved NLP model to: {NLP_MODEL_PATH}")

    # 5. Save Metadata
    metadata = {
        "model_type": "TF-IDF (1-2 grams) + LogisticRegression",
        "dataset": "UCI SMS Spam Collection",
        "training_date_utc": datetime.utcnow().isoformat(),
        "random_seed": RANDOM_SEED,
        "sample_counts": {
            "total": int(len(df)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "class_counts": {
            "ham": int((df["target"] == 0).sum()),
            "spam": int((df["target"] == 1).sum()),
        },
        "test_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "brier_score": brier,
        },
        "confusion_matrix": cm.tolist(),
    }
    with open(NLP_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 6. Generate docs/NLP_EVALUATION.md
    report = f"""# SocialShield AI – NLP Model Evaluation Report

*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*

## 1. Dataset & Task
- **Dataset:** UCI SMS Spam Collection (`SMSSpamCollection`)
- **Total Records:** {len(df):,} messages
- **Class Balance:** {metadata['class_counts']['ham']:,} legitimate (ham, 86.6%) vs {metadata['class_counts']['spam']:,} spam/phishing (13.4%)
- **Train / Test Split:** Stratified 80/20 ({len(X_train):,} train, {len(X_test):,} test)
- **Model Architecture:** Scikit-learn `Pipeline([TfidfVectorizer(ngram_range=(1, 2), max_features=10000), LogisticRegression(class_weight='balanced')])`

## 2. Held-Out Test Set Performance

| Metric | Score | Explanation |
|--------|-------|-------------|
| **Accuracy** | `{acc:.4f}` | Overall fraction of correct predictions |
| **Precision** | `{prec:.4f}` | When flagged as spam-like, probability it is truly spam |
| **Recall** | `{rec:.4f}` | Fraction of actual spam caught |
| **F1-Score** | `{f1:.4f}` | Harmonic mean of precision and recall |
| **ROC-AUC** | `{auc:.4f}` | Discrimination ability across all thresholds |
| **Brier Score** | `{brier:.4f}` | Probability calibration error (lower is better, 0.0 is perfect) |

### Confusion Matrix (Test Set, N = {len(y_test)})

| | Predicted Non-Spam | Predicted Spam |
|---|---|---|
| **Actual Ham (Non-spam)** | {cm[0, 0]} | {cm[0, 1]} |
| **Actual Spam** | {cm[1, 0]} | {cm[1, 1]} |

## 3. Practical Interpretability & Top Predictive n-grams
The Logistic Regression coefficients enable direct interpretability:
- High positive coefficients identify phrases strongly associated with promotional spam or phishing (e.g., `free`, `claim`, `win`, `urgent`, `prize`, `txt`, `call now`).
- In inference, multiplying the TF-IDF feature weights of a new input text by these coefficients surfaces the exact words driving the spam score.

## 4. Ethical Limitations
1. **Context Transferability:** The model is trained on SMS messages. While mobile phishing patterns overlap heavily with social media spam (DMs, deceptive bios), social posts also feature platform-specific hashtags, mentions, and emoji clusters.
2. **Spam-Like ≠ Fake Account:** A genuine user might post a promotional link, and an automated bot might quote neutral text. Text analysis provides an indicator signal, **not proof of account inauthenticity**.
3. **Missing Text:** When an account has no bio or posts, text analysis returns `not_available` with an explicit limitation flag — missing text is **never** silently treated as safe.
"""
    with open(NLP_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Generated NLP evaluation report at: {NLP_REPORT_PATH}")

    return metadata


if __name__ == "__main__":
    train_nlp_model()

