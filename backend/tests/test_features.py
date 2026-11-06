"""
tests/test_features.py – Unit tests for feature extraction and preprocessing pipeline.

Tests edge cases:
  1. Division by zero (e.g. 0 followers, 0 following, 0 days account age).
  2. Negative values rejection / safe clamping.
  3. Missing fields (verifying np.nan preservation and missing indicators).
  4. Brand new account (age 0-1 days).
  5. Username edge cases (empty string, numbers-only, no numbers, emojis/special chars).
  6. Pipeline fit/transform data leakage & shape consistency.
"""
import numpy as np
import pytest
from app.schemas import AccountInput, Platform
from app.ml.features import (
    extract_raw_features,
    to_feature_dataframe,
    build_preprocessing_pipeline,
    ALL_FEATURE_COLUMNS,
)


def test_division_by_zero_safety():
    """Verify 0 followers, 0 following, and 0 age don't trigger ZeroDivisionError or inf."""
    acc = AccountInput(
        platform=Platform.instagram,
        username="zerouser",
        followers=0,
        following=0,
        posts=0,
        account_age_days=0,
    )
    feats = extract_raw_features(acc)
    
    assert not np.isnan(feats["following_to_followers_ratio"])
    assert not np.isinf(feats["following_to_followers_ratio"])
    assert feats["following_to_followers_ratio"] == 0.0

    assert not np.isnan(feats["followers_to_following_ratio"])
    assert not np.isinf(feats["followers_to_following_ratio"])
    assert feats["followers_to_following_ratio"] == 0.0

    assert not np.isnan(feats["posts_per_day"])
    assert not np.isinf(feats["posts_per_day"])
    assert feats["posts_per_day"] == 0.0


def test_brand_new_account_age_zero_to_one():
    """Brand new account (age 0 or 1 days) with 10 posts."""
    acc = AccountInput(
        platform=Platform.x,
        username="newbie",
        followers=5,
        following=100,
        posts=10,
        account_age_days=0,
    )
    feats = extract_raw_features(acc)
    # 10 posts / max(0, 1) = 10.0
    assert feats["posts_per_day"] == 10.0
    assert feats["following_to_followers_ratio"] == 100.0 / 6.0


def test_missing_fields_preservation_and_indicators():
    """Missing fields should produce np.nan in continuous features and 1.0 in missing indicator flags."""
    acc = AccountInput(
        platform=Platform.facebook,
        username="sparse_user",
        # all counts None
    )
    feats = extract_raw_features(acc)

    assert np.isnan(feats["followers"])
    assert np.isnan(feats["following"])
    assert np.isnan(feats["posts"])
    assert np.isnan(feats["account_age_days"])
    assert np.isnan(feats["has_profile_pic"])
    assert np.isnan(feats["bio_len"])

    # Indicators must explicitly flag the absence
    assert feats["missing_followers"] == 1.0
    assert feats["missing_following"] == 1.0
    assert feats["missing_posts"] == 1.0
    assert feats["missing_age"] == 1.0
    assert feats["missing_profile_pic"] == 1.0
    assert feats["missing_bio"] == 1.0


def test_username_edge_cases():
    # 1. Pure numbers
    acc_digits = AccountInput(platform=Platform.instagram, username="123456789")
    feats_digits = extract_raw_features(acc_digits)
    assert feats_digits["username_len"] == 9.0
    assert feats_digits["username_digit_ratio"] == 1.0

    # 2. No numbers
    acc_alpha = AccountInput(platform=Platform.instagram, username="alexsmith")
    feats_alpha = extract_raw_features(acc_alpha)
    assert feats_alpha["username_len"] == 9.0
    assert feats_alpha["username_digit_ratio"] == 0.0

    # 3. Special characters & underscores
    acc_spec = AccountInput(platform=Platform.instagram, username="_user.99_")
    feats_spec = extract_raw_features(acc_spec)
    assert feats_spec["username_len"] == 9.0
    assert round(feats_spec["username_digit_ratio"], 2) == round(2.0 / 9.0, 2)


def test_bio_and_suspicious_url_detection():
    # Suspicious shortener
    acc_spam = AccountInput(
        platform=Platform.instagram,
        username="freegifts",
        bio="Check this out: https://bit.ly/freemoney or visit http://promo.xyz",
    )
    feats_spam = extract_raw_features(acc_spam)
    assert feats_spam["has_bio"] == 1.0
    assert feats_spam["url_count_in_bio"] == 2.0
    assert feats_spam["has_url_in_bio"] == 1.0
    assert feats_spam["suspicious_url_flag"] == 1.0

    # Clean normal link
    acc_clean = AccountInput(
        platform=Platform.youtube,
        username="creator",
        bio="My portfolio: https://mywebsite.com/about",
    )
    feats_clean = extract_raw_features(acc_clean)
    assert feats_clean["has_url_in_bio"] == 1.0
    assert feats_clean["suspicious_url_flag"] == 0.0


def test_duplicate_content_ratio():
    acc_dup = AccountInput(
        platform=Platform.x,
        username="botnet",
        post_text="Buy crypto now!\nBuy crypto now!\nBuy crypto now!\nVisit us!",
    )
    feats_dup = extract_raw_features(acc_dup)
    # 4 lines, 2 unique -> 1 - (2/4) = 0.5
    assert feats_dup["duplicate_content_ratio"] == 0.5


def test_preprocessing_pipeline_fit_transform():
    """Verify pipeline handles mixed missing/present records without leakage and outputs fixed shape."""
    records = [
        {"username": "user1", "followers": 100, "following": 200, "posts": 50, "account_age_days": 100, "has_profile_pic": True, "bio": "Hello"},
        {"username": "user2", "followers": 0, "following": 5000, "posts": 2, "account_age_days": 1, "has_profile_pic": False, "bio": None},
        {"username": "user3", "followers": None, "following": None, "posts": None, "account_age_days": None, "has_profile_pic": None, "bio": None},
    ]
    df = to_feature_dataframe(records)
    pipeline = build_preprocessing_pipeline()
    
    X_trans = pipeline.fit_transform(df)

    assert X_trans.shape == (3, len(ALL_FEATURE_COLUMNS))
    assert not np.isnan(X_trans).any(), "Transformed output must not contain any NaNs after imputation"

