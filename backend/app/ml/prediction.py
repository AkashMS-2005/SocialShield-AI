"""
ml/prediction.py – Production model loader and inference engine.

Features:
  - Singleton-style model loading with caching.
  - Custom ModelArtifactMissingError if artifact has not been trained yet.
  - Returns calibrated probability, predicted label, and explicit list of missing inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from app.schemas import AccountInput
from app.ml.features import (
    extract_raw_features,
    to_feature_dataframe,
    ALL_FEATURE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Exceptions & Paths
# ---------------------------------------------------------------------------

class ModelArtifactMissingError(RuntimeError):
    """Raised when the trained model artifact cannot be found on disk."""
    pass


BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BACKEND_DIR / "app" / "models" / "account_classifier.joblib"
METADATA_PATH = BACKEND_DIR / "app" / "models" / "model_metadata.json"

_CACHED_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_CACHED_METADATA: Optional[Dict[str, Any]] = None


def load_model_bundle() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads and caches the model pipeline and metadata from disk."""
    global _CACHED_MODEL_BUNDLE, _CACHED_METADATA
    if _CACHED_MODEL_BUNDLE is not None and _CACHED_METADATA is not None:
        return _CACHED_MODEL_BUNDLE, _CACHED_METADATA

    if not MODEL_PATH.exists():
        raise ModelArtifactMissingError(
            f"Trained model artifact not found at '{MODEL_PATH}'. "
            "Please run 'python -m app.ml.train_model' to generate it."
        )

    _CACHED_MODEL_BUNDLE = joblib.load(MODEL_PATH)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _CACHED_METADATA = json.load(f)
    else:
        _CACHED_METADATA = {}

    return _CACHED_MODEL_BUNDLE, _CACHED_METADATA


def predict(input_data: Union[AccountInput, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs inference on a single account.

    Returns:
      {
        "calibrated_probability": float (0.0 to 1.0),
        "predicted_label": int (0 or 1),
        "classification": str ("Low risk", "Suspicious", "High risk", or "Insufficient evidence"),
        "model_used": str,
        "missing_features": List[str],
        "is_synthetic": bool
      }
    """
    bundle, metadata = load_model_bundle()
    preprocessor = bundle["preprocessor"]
    classifier = bundle["classifier"]
    model_name = bundle.get("model_name", "Ensemble Classifier")
    is_synthetic = metadata.get("is_synthetic", False)

    # 1. Extract raw features & identify missing fields
    if isinstance(input_data, AccountInput):
        d = input_data.model_dump()
    else:
        d = dict(input_data)

    missing_fields = []
    for field in ["followers", "following", "posts", "account_age_days", "has_profile_pic", "bio"]:
        if d.get(field) is None:
            missing_fields.append(field)

    raw_features = extract_raw_features(d)
    raw_df = pd.DataFrame([raw_features])

    # 2. Check for Insufficient Evidence edge-case
    # If more than 4 primary signals are completely missing, don't pretend high certainty
    if len(missing_fields) >= 5:
        return {
            "calibrated_probability": None,
            "predicted_label": None,
            "classification": "Insufficient evidence",
            "model_used": model_name,
            "missing_features": missing_fields,
            "is_synthetic": is_synthetic,
        }

    # 3. Transform via preprocessor pipeline
    X_trans = preprocessor.transform(raw_df)

    # 4. Predict calibrated probability
    # class 1 = suspicious/fake
    calibrated_prob = float(classifier.predict_proba(X_trans)[0, 1])
    predicted_label = int(calibrated_prob >= 0.5)

    # Map probability to ethical risk tiers
    if calibrated_prob < 0.35:
        classification = "Low risk"
    elif calibrated_prob < 0.70:
        classification = "Suspicious"
    else:
        classification = "High risk"

    return {
        "calibrated_probability": round(calibrated_prob, 4),
        "predicted_label": predicted_label,
        "classification": classification,
        "model_used": model_name,
        "missing_features": missing_fields,
        "is_synthetic": is_synthetic,
    }

