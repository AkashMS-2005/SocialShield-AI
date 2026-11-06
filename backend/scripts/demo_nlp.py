"""
scripts/demo_nlp.py – Runs NLP analysis on three distinct sample inputs:
  1. Aggressive promotional / phishing bio with shorteners.
  2. Normal conversational profile and post.
  3. Sparse profile with missing text (demonstrates 'not_available' and safe limitation handling).
"""
import json
from app.services.nlp_service import analyze_text

samples = [
    {
        "title": "Sample 1: Aggressive Phishing / Promotional Pattern",
        "bio": "DM to earn fast cash! FREE CRYPTO GIVEAWAY! Claim now: https://bit.ly/btc-win",
        "posts": [
            "WINNER SELECTED! Tap link in bio now to claim your £1000 prize!",
            "WINNER SELECTED! Tap link in bio now to claim your £1000 prize!",
        ],
    },
    {
        "title": "Sample 2: Authentic Conversational Profile",
        "bio": "Backpacking through Southeast Asia. Photographer and coffee enthusiast. Portfolio: https://phototravels.org",
        "posts": [
            "Sunrise over Bagan temples was breathtaking today! Sharing some unedited shots.",
            "Trying local street food in Chiang Mai. Highly recommend the khao soi.",
        ],
    },
    {
        "title": "Sample 3: Incomplete / Missing Text Profile",
        "bio": None,
        "posts": None,
    },
]

def main():
    print("=" * 80)
    print("SocialShield AI – NLP Analysis Demonstration (3 Sample Inputs)")
    print("=" * 80)

    for s in samples:
        print(f"\n>>> {s['title']}")
        if s["bio"] is not None:
            print(f"Bio: {s['bio']}")
        if s["posts"] is not None:
            print(f"Posts: {s['posts']}")

        result = analyze_text(bio=s["bio"], posts=s["posts"])
        print("\n--- Output ---")
        print(f"Label:           {result['label']}")
        print(f"Risk Score:      {result['text_risk_score']}")
        print(f"Risk Level:      {result['text_risk_level']}")
        print(f"Top Phrases:     {result['top_suspicious_phrases']}")
        print(f"Rule Flags:      {result['rule_based_flags']}")
        print(f"Duplicate Ratio: {result['duplicate_ratio']}")
        print(f"Explanation:     {result['explanation']}")
        print(f"Limitations:     {result['limitations']}")
        print("-" * 80)

if __name__ == "__main__":
    main()

