"""
ml/train_model.py – Model training, hyperparameter tuning, model comparison,
calibration, and artifact persistence.

Rules Compliance:
  - Stratified train/val/test splits (70% train, 15% validation, 15% test).
  - Preprocessing pipeline fitted strictly on train split only (NO data leakage).
  - Train 1) Logistic Regression, 2) Random Forest, 3) XGBoost with Stratified K-Fold CV.
  - Handles class imbalance appropriately (via compute_class_weight / scale_pos_weight).
  - Model selection based on validation PR-AUC / ROC-AUC (never an assumed winner).
  - Calibrate probabilities with CalibratedClassifierCV.
  - Save pipeline + calibrated model + metadata JSON to backend/app/models/.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from app.ml.features import (
    to_feature_dataframe,
    build_preprocessing_pipeline,
    ALL_FEATURE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Directories & Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
MODELS_DIR = BACKEND_DIR / "app" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "account_classifier.joblib"
METADATA_OUTPUT_PATH = MODELS_DIR / "model_metadata.json"
DATASET_SPLITS_PATH = MODELS_DIR / "test_data.joblib"

RANDOM_SEED = 42


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, str, bool]:
    """
    Loads dataset: uses train.csv if available; otherwise falls back to synthetic_SAMPLE_ONLY.csv.
    Returns (raw_features_df, y, dataset_name, is_synthetic).
    """
    insta_train = RAW_DATA_DIR / "train.csv"
    synthetic_sample = RAW_DATA_DIR / "synthetic_SAMPLE_ONLY.csv"

    if insta_train.exists():
        df_raw = pd.read_csv(insta_train)
        dataset_name = "Kaggle Instagram Fake Spammer Genuine Accounts (train.csv)"
        is_synthetic = False
    elif synthetic_sample.exists():
        df_raw = pd.read_csv(synthetic_sample)
        dataset_name = "SYNTHETIC SAMPLE (backend/data/raw/synthetic_SAMPLE_ONLY.csv)"
        is_synthetic = True
    else:
        raise FileNotFoundError("No raw dataset found in backend/data/raw/. Please provide train.csv or run inspect_data.py")

    # Map column names if from Instagram format
    # Columns in raw: profile pic, nums/length username, #posts, #followers, #follows, description length, external URL, fake
    records = []
    labels = []
    
    for _, row in df_raw.iterrows():
        # Derive dict compatible with extract_raw_features
        rec = {
            "username": f"user_{row.get('nums/length username', 0)}",
            "followers": row.get("#followers", row.get("followers")),
            "following": row.get("#follows", row.get("following")),
            "posts": row.get("#posts", row.get("posts")),
            "account_age_days": row.get("account_age_days", None),
            "has_profile_pic": bool(row.get("profile pic", row.get("has_profile_pic", 0))),
            "bio": " " * int(row.get("description length", 0)) if "description length" in row else row.get("bio", ""),
            "post_text": row.get("post_text", None),
        }
        records.append(rec)
        labels.append(int(row["fake"]))

    y = pd.Series(labels, name="fake")
    raw_df = to_feature_dataframe(records)
    return raw_df, y, dataset_name, is_synthetic


def train_and_select_best_model() -> Dict[str, Any]:
    print("=" * 75)
    print("SocialShield AI – Model Training & Model Selection")
    print("=" * 75)

    X_raw, y, dataset_name, is_synthetic = load_dataset()
    print(f"Loaded: {dataset_name}")
    if is_synthetic:
        print("⚠️ NOTICE: SYNTHETIC DATA – Results reflect pipeline validity, not real-world performance.")

    print(f"Dataset Size: {len(y)} samples | Class Distribution: {dict(y.value_counts())}")

    # 1. Stratified 70/15/15 Split
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"Train set: {len(y_train)} | Val set: {len(y_val)} | Test set: {len(y_test)}")

    # 2. Fit Preprocessing Pipeline ONLY on Train
    preprocessor = build_preprocessing_pipeline()
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Compute class weights for imbalance handling
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {classes[i]: weights[i] for i in range(len(classes))}
    pos_scale = float(sum(y_train == 0) / max(sum(y_train == 1), 1))
    print(f"Class imbalance handling: balanced weights = {class_weight_dict}, scale_pos_weight = {pos_scale:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 3. Model Grids (fast, sensible parameter search)
    models_to_train = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
            "params": {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs"]},
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
            "params": {"n_estimators": [50, 100], "max_depth": [4, 8, None], "min_samples_split": [2, 5]},
        },
        "XGBoost": {
            "estimator": XGBClassifier(random_state=RANDOM_SEED, scale_pos_weight=pos_scale, eval_metric="logloss"),
            "params": {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        },
    }

    validation_results = {}
    best_estimators = {}

    print("\n--- Hyperparameter Tuning (5-Fold Stratified CV on Train) ---")
    for name, config in models_to_train.items():
        grid = GridSearchCV(
            config["estimator"],
            config["params"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid.fit(X_train_trans, y_train)
        best_est = grid.best_estimator_
        best_estimators[name] = best_est

        # Evaluate on held-out validation set
        val_probs = best_est.predict_proba(X_val_trans)[:, 1]
        val_preds = best_est.predict(X_val_trans)
        val_auc = roc_auc_score(y_val, val_probs)
        val_f1 = f1_score(y_val, val_preds)

        # Precision-Recall AUC (critical for imbalanced fraud/spam tasks)
        precision, recall, _ = precision_recall_curve(y_val, val_probs)
        val_pr_auc = auc(recall, precision)

        validation_results[name] = {
            "best_params": grid.best_params_,
            "val_roc_auc": float(val_auc),
            "val_pr_auc": float(val_pr_auc),
            "val_f1": float(val_f1),
        }
        print(f"[{name}] Best params: {grid.best_params_} | Val ROC-AUC: {val_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Val F1: {val_f1:.4f}")

    # 4. Model Selection based on validation PR-AUC + ROC-AUC
    # We choose the model with the highest combined rank of PR-AUC and ROC-AUC
    winner_name = max(validation_results.keys(), key=lambda m: validation_results[m]["val_pr_auc"] + validation_results[m]["val_roc_auc"])
    raw_best_model = best_estimators[winner_name]
    print(f"\n🏆 Selected Best Model: {winner_name} (based on highest validation PR-AUC & ROC-AUC)")

    # 5. Probability Calibration
    # CalibratedClassifierCV using sigmoid calibration fitted on CV folds
    print("Calibrating probabilities via CalibratedClassifierCV...")
    calibrated_model = CalibratedClassifierCV(estimator=raw_best_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train_trans, y_train)

    # Check calibration quality on validation set (Brier score: lower is better, 0 = perfect)
    val_calibrated_probs = calibrated_model.predict_proba(X_val_trans)[:, 1]
    brier = brier_score_loss(y_val, val_calibrated_probs)
    print(f"Validation Brier Score (Calibration Quality): {brier:.4f}")

    # 6. Save Full Inference Artifacts
    full_pipeline = {
        "preprocessor": preprocessor,
        "classifier": calibrated_model,
        "model_name": winner_name,
        "feature_names": ALL_FEATURE_COLUMNS,
    }
    joblib.dump(full_pipeline, MODEL_OUTPUT_PATH)
    print(f"✅ Saved trained pipeline artifact to: {MODEL_OUTPUT_PATH}")

    # Save held-out test set & all candidate models for evaluate_model.py
    test_artifacts = {
        "X_test_raw": X_test_raw,
        "X_test_trans": X_test_trans,
        "y_test": y_test,
        "y_val": y_val,
        "val_calibrated_probs": val_calibrated_probs,
        "best_estimators": best_estimators,
        "winner_name": winner_name,
        "calibrated_model": calibrated_model,
        "dataset_name": dataset_name,
        "is_synthetic": is_synthetic,
    }
    joblib.dump(test_artifacts, DATASET_SPLITS_PATH)

    # 7. Write Metadata JSON
    metadata = {
        "model_type": winner_name,
        "calibrated": True,
        "calibration_method": "sigmoid",
        "training_date_utc": datetime.utcnow().isoformat(),
        "random_seed": RANDOM_SEED,
        "dataset_name": dataset_name,
        "is_synthetic": is_synthetic,
        "sample_counts": {
            "total": int(len(y)),
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "class_balance_train": {int(k): int(v) for k, v in y_train.value_counts().items()},
        "validation_metrics": validation_results,
        "validation_brier_score": float(brier),
        "feature_list": ALL_FEATURE_COLUMNS,
    }
    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata JSON to: {METADATA_OUTPUT_PATH}")

    return metadata


if __name__ == "__main__":
    train_and_select_best_model()
