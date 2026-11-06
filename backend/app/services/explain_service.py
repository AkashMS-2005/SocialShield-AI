"""
services/explain_service.py – Model interpretability service.

Rules & Architecture Compliance:
  1. SHAP TreeExplainer on fitted tree-based models (RandomForest / XGBoost)
     OR LinearExplainer on Logistic Regression base estimator.
     Explicitly labels the output method ("SHAP (TreeExplainer on base estimator)"
     or "SHAP (LinearExplainer on base estimator)").
  2. Clear separation between:
     - Global Feature Importance (overall model behavior across dataset)
     - Local SHAP Values (per-instance decision breakdown)
  3. Returns top 5 features with:
     - feature: human-readable name
     - raw_value: unscaled human-readable value
     - shap_value: signed impact
     - direction: "toward suspicious" or "toward genuine"
     - explanation: natural human-readable sentence
  4. Caches explainer at startup for sub-second (< 1s) prediction latency.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap

from app.schemas import AccountInput
from app.ml.features import (
    extract_raw_features,
    ALL_FEATURE_COLUMNS,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_DIR / "app" / "models"
MODEL_PATH = MODELS_DIR / "account_classifier.joblib"
TEST_DATA_PATH = MODELS_DIR / "test_data.joblib"
DOCS_DIR = BACKEND_DIR.parent / "docs"

# Human-readable label dictionary
FEATURE_DISPLAY_NAMES = {
    "followers": "Follower count",
    "following": "Following count",
    "posts": "Total posts",
    "account_age_days": "Account age (days)",
    "following_to_followers_ratio": "Following-to-Followers ratio",
    "followers_to_following_ratio": "Followers-to-Following ratio",
    "posts_per_day": "Posting frequency (posts/day)",
    "username_len": "Username character length",
    "username_digit_ratio": "Username numerical digit percentage",
    "has_profile_pic": "Has profile picture",
    "has_bio": "Has bio biography",
    "bio_len": "Bio text length",
    "has_url_in_bio": "Has link in bio",
    "url_count_in_bio": "Number of links in bio",
    "suspicious_url_flag": "Suspicious URL / shortener flag",
    "duplicate_content_ratio": "Duplicate content ratio across posts",
    "missing_followers": "Unreported follower count",
    "missing_following": "Unreported following count",
    "missing_posts": "Unreported post count",
    "missing_age": "Unreported account age",
    "missing_profile_pic": "Unreported profile picture",
    "missing_bio": "Unreported bio",
}

# Explainer Cache Singleton
_CACHED_EXPLAINER_BUNDLE = None


def get_base_estimator(calibrated_clf):
    """Extracts the underlying unfitted or CV-calibrated base estimator from CalibratedClassifierCV."""
    if hasattr(calibrated_clf, "calibrated_classifiers_"):
        # Scikit-learn calibrated ensemble has .estimator on each fold
        first_calib = calibrated_clf.calibrated_classifiers_[0]
        if hasattr(first_calib, "estimator"):
            return first_calib.estimator
    if hasattr(calibrated_clf, "estimator"):
        return calibrated_clf.estimator
    return calibrated_clf


def init_explainer():
    """Initializes and caches the SHAP explainer and background reference distribution."""
    global _CACHED_EXPLAINER_BUNDLE
    if _CACHED_EXPLAINER_BUNDLE is not None:
        return _CACHED_EXPLAINER_BUNDLE

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact missing at {MODEL_PATH}")

    model_bundle = joblib.load(MODEL_PATH)
    preprocessor = model_bundle["preprocessor"]
    calibrated_classifier = model_bundle["classifier"]
    model_name = model_bundle.get("model_name", "Classifier")

    # Retrieve base estimator
    base_clf = get_base_estimator(calibrated_classifier)

    # Use a small background dataset for SHAP if test_data is available
    background_data = None
    if TEST_DATA_PATH.exists():
        try:
            test_bundle = joblib.load(TEST_DATA_PATH)
            background_data = test_bundle["X_test_trans"][:50]
        except Exception:
            background_data = None

    if background_data is None:
        # Fallback synthetic background zero matrix
        background_data = np.zeros((10, len(ALL_FEATURE_COLUMNS)))

    # Choose appropriate SHAP explainer
    # TreeExplainer for tree models; LinearExplainer for linear models
    clf_type_str = str(type(base_clf)).lower()
    if "forest" in clf_type_str or "tree" in clf_type_str or "xgb" in clf_type_str:
        explainer = shap.TreeExplainer(base_clf)
        method_label = "SHAP (TreeExplainer on base estimator)"
    elif "logistic" in clf_type_str or "linear" in clf_type_str:
        explainer = shap.LinearExplainer(base_clf, background_data)
        method_label = "SHAP (LinearExplainer on base estimator)"
    else:
        explainer = shap.Explainer(base_clf, background_data)
        method_label = "SHAP (Permutation/KernelExplainer on base estimator)"

    _CACHED_EXPLAINER_BUNDLE = {
        "preprocessor": preprocessor,
        "explainer": explainer,
        "base_clf": base_clf,
        "method_label": method_label,
        "model_name": model_name,
        "feature_names": ALL_FEATURE_COLUMNS,
    }
    return _CACHED_EXPLAINER_BUNDLE


def generate_feature_explanation_sentence(
    feature_name: str,
    raw_val: Any,
    shap_val: float,
    direction: str,
) -> str:
    """Produces clear, user-facing, ethical sentences explaining the impact of a feature."""
    display = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
    towards = "toward suspicious indicators" if direction == "toward suspicious" else "toward authentic/genuine signals"

    if feature_name == "following_to_followers_ratio":
        if shap_val > 0:
            return f"Following count is significantly higher than followers ({raw_val:.1f}x), pushing the score {towards}."
        else:
            return f"Balanced or high follower-to-following ratio ({raw_val:.1f}x) pushes the score {towards}."

    elif feature_name == "followers":
        if shap_val > 0:
            return f"Very low follower count ({int(raw_val) if not pd.isna(raw_val) else 'unknown'}) pushes the score {towards}."
        else:
            return f"Established follower count ({int(raw_val) if not pd.isna(raw_val) else 'unknown'}) pushes the score {towards}."

    elif feature_name == "following":
        if shap_val > 0:
            return f"High number of accounts followed ({int(raw_val) if not pd.isna(raw_val) else 'unknown'}) pushes the score {towards}."
        else:
            return f"Moderate or restrained following count ({int(raw_val) if not pd.isna(raw_val) else 'unknown'}) pushes the score {towards}."

    elif feature_name == "has_profile_pic":
        if raw_val == 0 or raw_val is False:
            return f"Absence of an avatar/profile picture pushes the score {towards}."
        else:
            return f"Profile avatar is present, contributing {towards}."

    elif feature_name == "username_digit_ratio":
        pct = int(raw_val * 100) if not pd.isna(raw_val) else 0
        if shap_val > 0:
            return f"Handle contains a high proportion of numbers ({pct}%), which often appears in automated accounts."
        else:
            return f"Clean, natural handle structure with few numbers ({pct}%), contributing {towards}."

    elif feature_name == "suspicious_url_flag":
        if raw_val == 1.0:
            return f"Detected shortener or high-risk link in bio, pushing the score {towards}."
        else:
            return f"No suspicious or shortened links detected in bio, contributing {towards}."

    elif feature_name.startswith("missing_"):
        clean_field = feature_name.replace("missing_", "")
        return f"Information for '{clean_field}' was omitted, treated as unknown risk rather than safe."

    # Generic fallback
    if shap_val > 0:
        return f"Metric '{display}' ({raw_val}) increases the risk score {towards}."
    else:
        return f"Metric '{display}' ({raw_val}) lowers the risk score {towards}."


def explain_account(
    input_data: Union[AccountInput, Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Computes local SHAP explanation for a single social account.

    Returns:
      {
        "method": "SHAP (LinearExplainer on base estimator)",
        "scope": "local",
        "top_features": [
           {
              "feature": "following_to_followers_ratio",
              "display_name": "Following-to-Followers ratio",
              "value": 1633.3,
              "shap_value": 0.4215,
              "direction": "toward suspicious",
              "sentence": "..."
           },
           ...
        ],
        "base_value": float,
        "is_sufficient_evidence": bool
      }
    """
    bundle = init_explainer()
    preprocessor = bundle["preprocessor"]
    explainer = bundle["explainer"]
    method_label = bundle["method_label"]
    feature_names = bundle["feature_names"]

    # 1. Extract raw features
    if isinstance(input_data, AccountInput):
        d = input_data.model_dump()
    else:
        d = dict(input_data)

    raw_features = extract_raw_features(d)
    raw_df = pd.DataFrame([raw_features])

    # Check for insufficient evidence (too many missing fields)
    missing_count = sum(raw_features[k] for k in raw_features if k.startswith("missing_"))
    if missing_count >= 5:
        return {
            "method": method_label,
            "scope": "local",
            "top_features": [],
            "base_value": 0.0,
            "is_sufficient_evidence": False,
            "message": "Insufficient evidence: too few profile fields provided to compute meaningful SHAP explanations.",
        }

    # 2. Transform through fitted preprocessing pipeline
    X_trans = preprocessor.transform(raw_df)

    # 3. Compute local SHAP values
    shap_vals = explainer.shap_values(X_trans)

    # Handle SHAP output dimensions (binary classification: take positive class 1)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        # Class 1 (suspicious)
        instance_shap = shap_vals[1][0]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        # (samples, features, classes)
        instance_shap = shap_vals[0, :, 1]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
        instance_shap = shap_vals[0]
    else:
        instance_shap = np.array(shap_vals).flatten()

    base_val = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)

    # 4. Rank features by absolute magnitude of contribution
    ranked_indices = np.argsort(np.abs(instance_shap))[::-1][:top_k]

    top_features = []
    for idx in ranked_indices:
        feat_name = feature_names[idx]
        s_val = float(instance_shap[idx])
        raw_val = raw_features.get(feat_name, np.nan)
        direction = "toward suspicious" if s_val > 0 else "toward genuine"
        sentence = generate_feature_explanation_sentence(feat_name, raw_val, s_val, direction)

        top_features.append({
            "feature": feat_name,
            "display_name": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
            "value": None if pd.isna(raw_val) else (round(raw_val, 3) if isinstance(raw_val, float) else raw_val),
            "shap_value": round(s_val, 4),
            "direction": direction,
            "sentence": sentence,
        })

    return {
        "method": method_label,
        "scope": "local",
        "top_features": top_features,
        "base_value": round(base_val, 4),
        "is_sufficient_evidence": True,
    }


def compute_global_feature_importance() -> List[Dict[str, Any]]:
    """
    Computes global feature importance across a reference test distribution.
    Kept strictly separate from local explanations.
    """
    bundle = init_explainer()
    explainer = bundle["explainer"]
    feature_names = bundle["feature_names"]

    if not TEST_DATA_PATH.exists():
        return []

    test_bundle = joblib.load(TEST_DATA_PATH)
    X_test_trans = test_bundle["X_test_trans"]

    shap_vals = explainer.shap_values(X_test_trans)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        vals = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        vals = shap_vals[:, :, 1]
    else:
        vals = np.array(shap_vals)

    mean_abs = np.mean(np.abs(vals), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    global_importance = []
    for idx in sorted_idx:
        fname = feature_names[idx]
        global_importance.append({
            "feature": fname,
            "display_name": FEATURE_DISPLAY_NAMES.get(fname, fname),
            "mean_abs_shap": round(float(mean_abs[idx]), 4),
        })

    return global_importance


def generate_global_summary_plot(output_path: Optional[Path] = None) -> Path:
    """Generates and saves the SHAP global summary bar plot to docs/shap_summary.png."""
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = DOCS_DIR / "shap_summary.png"

    importance = compute_global_feature_importance()
    if not importance:
        return output_path

    # Top 10 global features
    top_items = importance[:10][::-1]
    names = [item["display_name"] for item in top_items]
    scores = [item["mean_abs_shap"] for item in top_items]

    plt.figure(figsize=(9, 5))
    plt.barh(names, scores, color="#2563eb", alpha=0.85)
    plt.xlabel("Mean |SHAP Value| (Average Model Impact)")
    plt.title("Global Feature Importance (SHAP)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path

