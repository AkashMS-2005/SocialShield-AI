"""
scripts/demo_features.py – Demonstrates the feature extraction and preprocessing
pipeline on 5 distinct sample accounts, including sparse / missing data cases.
"""
from __future__ import annotations

import pandas as pd
from app.schemas import AccountInput, Platform
from app.ml.features import (
    to_feature_dataframe,
    build_preprocessing_pipeline,
    ALL_FEATURE_COLUMNS,
)

sample_accounts = [
    # 1. Normal active user
    AccountInput(
        platform=Platform.instagram,
        username="jane_doe_photo",
        followers=1450,
        following=420,
        posts=120,
        account_age_days=600,
        has_profile_pic=True,
        bio="Travel and landscape photographer. Contact me: https://janedoe.com",
        post_text="New sunset photos from our hiking trip in Oregon! Enjoying the scenic views.",
    ),
    # 2. Classic bot/spammer pattern (high following, 0 followers, suspicious shortener)
    AccountInput(
        platform=Platform.instagram,
        username="get_rich_987654",
        followers=2,
        following=4900,
        posts=3,
        account_age_days=2,
        has_profile_pic=False,
        bio="FREE BITCOIN GIVEAWAY! Click: https://bit.ly/claim-btc-now and win big!",
        post_text="Claim your free coins now!\nClaim your free coins now!\nLimited time offer!",
    ),
    # 3. Brand new authentic account (age 0-1 days)
    AccountInput(
        platform=Platform.x,
        username="charlie_dev",
        followers=10,
        following=25,
        posts=2,
        account_age_days=1,
        has_profile_pic=True,
        bio="Just joined X. Learning machine learning and backend engineering.",
        post_text="Hello world! My first post on this platform.",
    ),
    # 4. Incomplete data (sparse entry - followers unknown, bio absent)
    AccountInput(
        platform=Platform.facebook,
        username="anon_profile_42",
        followers=None,
        following=150,
        posts=None,
        account_age_days=15,
        has_profile_pic=None,
        bio=None,
        post_text=None,
    ),
    # 5. Zero-risk edge case (brand new creator, 0 following, 0 followers)
    AccountInput(
        platform=Platform.youtube,
        username="fresh_channel",
        followers=0,
        following=0,
        posts=0,
        account_age_days=0,
        has_profile_pic=True,
        bio="",
        post_text="",
    ),
]

def main():
    print("=" * 80)
    print("SocialShield AI – Feature Matrix Demo (5 Sample Accounts)")
    print("=" * 80)

    # 1. Raw Extracted Feature DataFrame (before imputation/scaling)
    raw_df = to_feature_dataframe(sample_accounts)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 120)
    print("\n--- 1. Raw Engineered Features (Preserving Nulls & Missing Indicators) ---")
    print(raw_df[["followers", "following", "following_to_followers_ratio", "posts_per_day", "username_digit_ratio", "suspicious_url_flag", "missing_followers", "missing_bio"]])

    # 2. Fit and Transform Pipeline
    pipeline = build_preprocessing_pipeline()
    transformed_matrix = pipeline.fit_transform(raw_df)
    transformed_df = pd.DataFrame(transformed_matrix, columns=ALL_FEATURE_COLUMNS)

    print("\n--- 2. Preprocessed & Scaled Feature Matrix (Ready for ML/XGBoost/RF) ---")
    print(transformed_df[["followers", "following", "following_to_followers_ratio", "posts_per_day", "username_digit_ratio", "has_profile_pic", "missing_followers", "missing_bio"]])
    print(f"\nFinal Matrix Shape: {transformed_matrix.shape} (Rows: {transformed_matrix.shape[0]}, Features: {transformed_matrix.shape[1]})")
    print(f"NaNs present in processed matrix: {pd.isna(transformed_df).any().any()}")
    print("=" * 80)

if __name__ == "__main__":
    main()

