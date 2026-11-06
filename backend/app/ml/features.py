"""
ml/features.py – Feature extraction, engineering, and scikit-learn preprocessing pipeline.

Working Rules Compliance:
  - Missing data is NEVER treated as zero risk (nullable fields preserved with missing indicators).
  - Safe mathematical operations (zero division prevention, clipping).
  - Independent, replaceable module.
  - Strict prevention of data leakage.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from app.schemas import AccountInput


# ---------------------------------------------------------------------------
# Constants & Regex patterns
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Suspicious URL shorteners or typical spam redirects
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "buff.ly",
    "ow.ly", "cutt.ly", "linktr.ee", "rebrand.ly", "tiny.cc", "t.ly"
}

# Suspicious / high-risk TLDs often seen in automated phishing or throwaway domains
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".loan", ".win", ".bid", ".click",
    ".link", ".gq", ".cf", ".ml", ".ga", ".tk", ".cc"
}


# ---------------------------------------------------------------------------
# Feature Extraction from AccountInput or raw dict
# ---------------------------------------------------------------------------

def extract_raw_features(data: Union[AccountInput, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Maps an AccountInput instance or a raw row/dict into a flat, standardized
    dictionary of unscaled numerical/boolean features.

    Preserves missing values as None / np.nan. Never silently defaults to 0.
    """
    if isinstance(data, AccountInput):
        d = data.model_dump()
    else:
        d = dict(data)

    username: str = str(d.get("username") or "")
    bio: Optional[str] = d.get("bio")
    post_text: Optional[str] = d.get("post_text")
    
    # 1. Base counts & profile fields (preserve None if missing)
    followers = d.get("followers")
    following = d.get("following")
    posts = d.get("posts")
    account_age_days = d.get("account_age_days")
    has_profile_pic = d.get("has_profile_pic")

    # Coerce to float / None
    followers_val = float(followers) if followers is not None else np.nan
    following_val = float(following) if following is not None else np.nan
    posts_val = float(posts) if posts is not None else np.nan
    age_val = float(account_age_days) if account_age_days is not None else np.nan
    
    if has_profile_pic is None:
        profile_pic_val = np.nan
    else:
        profile_pic_val = 1.0 if bool(has_profile_pic) else 0.0

    # 2. Username engineering
    username_len = float(len(username))
    digits_in_user = sum(1 for c in username if c.isdigit())
    username_digit_ratio = (digits_in_user / username_len) if username_len > 0 else 0.0

    # 3. Ratios & activity metrics (safe division)
    if not np.isnan(followers_val) and not np.isnan(following_val):
        # High following with low followers is a classic spam pattern.
        # Safe division using +1 denominator
        following_to_followers_ratio = following_val / (followers_val + 1.0)
        followers_to_following_ratio = followers_val / (following_val + 1.0)
    else:
        following_to_followers_ratio = np.nan
        followers_to_following_ratio = np.nan

    if not np.isnan(posts_val) and not np.isnan(age_val):
        # For age 0 or 1 days, add 1.0 day minimum to prevent zero division
        safe_age = max(age_val, 1.0)
        posts_per_day = posts_val / safe_age
    else:
        posts_per_day = np.nan

    # 4. Bio features
    if bio is not None:
        has_bio = 1.0 if len(bio.strip()) > 0 else 0.0
        bio_len = float(len(bio))
        urls = URL_PATTERN.findall(bio)
        url_count_in_bio = float(len(urls))
        has_url_in_bio = 1.0 if url_count_in_bio > 0 else 0.0
        
        # Suspicious URL check in bio
        suspicious_url_flag = 0.0
        for u in urls:
            u_lower = u.lower()
            if any(domain in u_lower for domain in SHORTENER_DOMAINS) or any(u_lower.endswith(tld) or (tld + "/") in u_lower for tld in SUSPICIOUS_TLDS):
                suspicious_url_flag = 1.0
                break
    else:
        has_bio = np.nan
        bio_len = np.nan
        url_count_in_bio = np.nan
        has_url_in_bio = np.nan
        suspicious_url_flag = np.nan

    # 5. Content repetition / duplicate ratio
    # If multiple posts or sample text provided
    if post_text is not None and len(post_text.strip()) > 0:
        lines = [line.strip() for line in post_text.split("\n") if line.strip()]
        if len(lines) > 1:
            unique_lines = len(set(lines))
            duplicate_content_ratio = 1.0 - (unique_lines / len(lines))
        else:
            duplicate_content_ratio = 0.0
    else:
        duplicate_content_ratio = np.nan

    # 6. Explicit Missing Indicators (1 if missing, 0 if present)
    return {
        # Core tabular features
        "has_profile_pic": profile_pic_val,
        "followers": followers_val,
        "following": following_val,
        "posts": posts_val,
        "account_age_days": age_val,
        "following_to_followers_ratio": following_to_followers_ratio,
        "followers_to_following_ratio": followers_to_following_ratio,
        "posts_per_day": posts_per_day,
        "username_len": username_len,
        "username_digit_ratio": username_digit_ratio,
        "has_bio": has_bio,
        "bio_len": bio_len,
        "has_url_in_bio": has_url_in_bio,
        "url_count_in_bio": url_count_in_bio,
        "suspicious_url_flag": suspicious_url_flag,
        "duplicate_content_ratio": duplicate_content_ratio,
        # Missing indicator flags
        "missing_followers": 1.0 if np.isnan(followers_val) else 0.0,
        "missing_following": 1.0 if np.isnan(following_val) else 0.0,
        "missing_posts": 1.0 if np.isnan(posts_val) else 0.0,
        "missing_age": 1.0 if np.isnan(age_val) else 0.0,
        "missing_profile_pic": 1.0 if np.isnan(profile_pic_val) else 0.0,
        "missing_bio": 1.0 if np.isnan(bio_len) else 0.0,
    }


def to_feature_dataframe(records: List[Union[AccountInput, Dict[str, Any]]]) -> pd.DataFrame:
    """Converts a collection of inputs to a DataFrame with all raw features extracted."""
    rows = [extract_raw_features(r) for r in records]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature Lists & Schema Definitions
# ---------------------------------------------------------------------------

CONTINUOUS_NUMERIC_FEATURES = [
    "followers",
    "following",
    "posts",
    "account_age_days",
    "following_to_followers_ratio",
    "followers_to_following_ratio",
    "posts_per_day",
    "username_len",
    "username_digit_ratio",
    "bio_len",
    "url_count_in_bio",
    "duplicate_content_ratio",
]

BINARY_NUMERIC_FEATURES = [
    "has_profile_pic",
    "has_bio",
    "has_url_in_bio",
    "suspicious_url_flag",
]

MISSING_INDICATOR_FEATURES = [
    "missing_followers",
    "missing_following",
    "missing_posts",
    "missing_age",
    "missing_profile_pic",
    "missing_bio",
]

ALL_FEATURE_COLUMNS = (
    CONTINUOUS_NUMERIC_FEATURES
    + BINARY_NUMERIC_FEATURES
    + MISSING_INDICATOR_FEATURES
)


# ---------------------------------------------------------------------------
# Scikit-Learn Preprocessing Pipeline Factory
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Builds a ColumnTransformer that:
      1. Imputes continuous numerical features with the training median and standardizes them.
      2. Imputes binary features with the most frequent value (mode) without scaling.
      3. Passes missing indicators directly through (they are already non-null 0/1 flags).

    Crucially, all medians and scalers are fitted ONLY on the training split
    to prevent data leakage.
    """
    continuous_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median", fill_value=0.0, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])

    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", fill_value=0.0, keep_empty_features=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", continuous_transformer, CONTINUOUS_NUMERIC_FEATURES),
            ("bin", binary_transformer, BINARY_NUMERIC_FEATURES),
            ("ind", "passthrough", MISSING_INDICATOR_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
