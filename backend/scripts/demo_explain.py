"""
scripts/demo_explain.py – Demonstrates SHAP explanations for:
  1. A genuine-looking account
  2. A suspicious-looking account
Also generates and saves docs/shap_summary.png.
"""
from __future__ import annotations

import json
from app.schemas import AccountInput, Platform
from app.services.explain_service import (
    explain_account,
    generate_global_summary_plot,
    compute_global_feature_importance,
)

samples = [
    {
        "type": "Genuine-Looking Account",
        "data": AccountInput(
            platform=Platform.instagram,
            username="travel_alex_99",
            followers=2400,
            following=450,
            posts=130,
            account_age_days=720,
            has_profile_pic=True,
            bio="Outdoor photographer based in Seattle. Landscapes and wildlife.",
        ),
    },
    {
        "type": "Suspicious-Looking Account",
        "data": {
            "username": "free_crypto_998877",
            "followers": 2,
            "following": 4900,
            "posts": 1,
            "account_age_days": 1,
            "has_profile_pic": False,
            "bio": "WIN BIG NOW! Click: https://bit.ly/claim-crypto",
        },
    },
]

def main():
    print("=" * 80)
    print("SocialShield AI – SHAP Model Explainability Demonstration")
    print("=" * 80)

    for item in samples:
        print(f"\n>>> Explaining {item['type']} ({item['data'].get('username') if isinstance(item['data'], dict) else item['data'].username})")
        res = explain_account(item["data"], top_k=5)
        print(f"Method: {res['method']}")
        print(f"Scope:  {res['scope']}")
        print(f"Base expected value: {res['base_value']}")
        print("\nTop 5 Influential Features:")
        print(f"{'Feature':<35} | {'Value':<10} | {'SHAP Impact':<12} | {'Direction'}")
        print("-" * 75)
        for f in res["top_features"]:
            val_str = str(f['value']) if f['value'] is not None else "None"
            print(f"{f['display_name']:<35} | {val_str:<10} | {f['shap_value']:<+12.4f} | {f['direction']}")
            print(f"  ↳ Sentence: \"{f['sentence']}\"")
        print("-" * 75)

    print("\n--- Global Feature Importance (Top 5 Overall) ---")
    global_imp = compute_global_feature_importance()
    for g in global_imp[:5]:
        print(f"- {g['display_name']}: mean |SHAP| = {g['mean_abs_shap']:.4f}")

    plot_path = generate_global_summary_plot()
    print(f"\n✅ Global summary plot saved to: {plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

