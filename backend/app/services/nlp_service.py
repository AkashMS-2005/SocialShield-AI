"""
services/nlp_service.py – Hybrid NLP analysis for social account text (bio & posts).

Combines:
  1. Statistical TF-IDF + Logistic Regression model to output calibrated spam probability
     and token-level contributions (LR weights × TF-IDF scores).
  2. Rule-based heuristic indicators (phishing/promotional keywords, suspicious URL patterns,
     IP URLs, shorteners, duplicate content across posts).
  3. Ethical handling: when no text is provided, returns 'not_available' with explicit
     limitations. Never masks missing text as safe (Rule #4).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
NLP_MODEL_PATH = BACKEND_DIR / "app" / "models" / "nlp_spam_model.joblib"

# ---------------------------------------------------------------------------
# Regex & Heuristic Keyword Patterns
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
IP_URL_PATTERN = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/\S*)?", re.IGNORECASE)

PROMOTIONAL_PHRASES = [
    "click here", "click the link", "free followers", "dm to earn", "dm for promo",
    "dm for collab", "fast cash", "earn from home", "make money online",
    "crypto giveaway", "free crypto", "claim now", "limited time offer",
    "winner selected", "congratulations you won", "guaranteed profit",
    "check bio link", "tap link in bio", "100% free", "whatsapp me",
    "telegram channel", "investment scheme", "double your money"
]

SUSPICIOUS_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "buff.ly", "ow.ly",
    "cutt.ly", "linktr.ee", "rebrand.ly", "tiny.cc", "t.ly", "shorte.st"
}

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".loan", ".win", ".bid", ".click",
    ".link", ".gq", ".cf", ".ml", ".ga", ".tk", ".cc", ".icu", ".buzz"
}

_CACHED_NLP_PIPELINE = None


def load_nlp_pipeline():
    """Loads the serialized TF-IDF + Logistic Regression pipeline with caching."""
    global _CACHED_NLP_PIPELINE
    if _CACHED_NLP_PIPELINE is not None:
        return _CACHED_NLP_PIPELINE

    if not NLP_MODEL_PATH.exists():
        return None

    _CACHED_NLP_PIPELINE = joblib.load(NLP_MODEL_PATH)
    return _CACHED_NLP_PIPELINE


def get_top_predictive_phrases(pipeline, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Computes local feature contributions for the input text using:
      contribution = tfidf_value * logistic_regression_weight
    Surfaces the most spam-indicative words/ngrams for this specific text.
    """
    if not pipeline or not text.strip():
        return []

    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]

        feature_names = np.array(tfidf.get_feature_names_out())
        coefs = clf.coef_[0]

        # Vectorize single document
        vec = tfidf.transform([text]).tocsr()
        non_zero_indices = vec.indices
        values = vec.data

        contributions = []
        for idx, val in zip(non_zero_indices, values):
            weight = coefs[idx]
            impact = float(val * weight)
            # Only record terms that push probability toward spam (positive contribution)
            if impact > 0.05:
                contributions.append({
                    "phrase": str(feature_names[idx]),
                    "contribution": round(impact, 4),
                })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions[:top_k]
    except Exception:
        return []


def analyze_text(
    bio: Optional[str] = None,
    posts: Optional[Union[List[str], str]] = None,
) -> Dict[str, Any]:
    """
    Analyses account text content across bio and sample posts.

    Returns:
      {
        "label": "spam-like" | "non-spam-like" | "not_available",
        "text_risk_score": float (0.0 to 1.0) | None,
        "text_risk_level": "Low" | "Medium" | "High" | "Insufficient evidence",
        "top_suspicious_phrases": list of {"phrase": str, "contribution": float},
        "rule_based_flags": list of str,
        "duplicate_ratio": float | None,
        "explanation": str,
        "limitations": list of str
      }
    """
    # 1. Normalize post list
    post_list: List[str] = []
    if isinstance(posts, str):
        post_list = [p.strip() for p in posts.split("\n") if p.strip()]
    elif isinstance(posts, list):
        post_list = [str(p).strip() for p in posts if str(p).strip()]

    clean_bio = bio.strip() if bio and bio.strip() else None

    # 2. Check for empty / unavailable input
    if clean_bio is None and len(post_list) == 0:
        return {
            "label": "not_available",
            "text_risk_score": None,
            "text_risk_level": "Insufficient evidence",
            "top_suspicious_phrases": [],
            "rule_based_flags": [],
            "duplicate_ratio": None,
            "explanation": "No bio text or post content provided. Missing text cannot be assessed as safe.",
            "limitations": [
                "No profile text available – text-based spam heuristics skipped.",
                "Missing text was NOT treated as zero risk.",
            ],
        }

    # Aggregate full corpus for textual inspection
    text_blocks = []
    if clean_bio:
        text_blocks.append(clean_bio)
    text_blocks.extend(post_list)
    combined_text = " ".join(text_blocks)

    # 3. Rule-Based Heuristic Evaluation
    rule_flags: List[str] = []
    lower_text = combined_text.lower()

    # (a) Promotional / phishing keyword match
    matched_phrases = [p for p in PROMOTIONAL_PHRASES if p in lower_text]
    if matched_phrases:
        rule_flags.append(f"Contains promotional / phishing phrase(s): {', '.join(matched_phrases[:3])}")

    # (b) Suspicious URLs: IP address links, shorteners, high-risk TLDs
    all_urls = URL_PATTERN.findall(combined_text)
    ip_urls = IP_URL_PATTERN.findall(combined_text)
    if ip_urls:
        rule_flags.append(f"Raw IP address link detected in text: {ip_urls[0]}")

    shortener_matches = []
    suspicious_tld_matches = []
    for u in all_urls:
        u_low = u.lower()
        for short in SUSPICIOUS_SHORTENERS:
            if short in u_low:
                shortener_matches.append(short)
        for tld in SUSPICIOUS_TLDS:
            if tld in u_low:
                suspicious_tld_matches.append(tld)

    if shortener_matches:
        rule_flags.append(f"Uses link shortener: {', '.join(set(shortener_matches))}")
    if suspicious_tld_matches:
        rule_flags.append(f"Uses unusual / high-risk TLD: {', '.join(set(suspicious_tld_matches))}")

    if len(all_urls) >= 3:
        rule_flags.append(f"High density of external links ({len(all_urls)} URLs found)")

    # (c) Duplicate content across multiple posts
    duplicate_ratio: Optional[float] = None
    if len(post_list) > 1:
        unique_posts = len(set(post_list))
        duplicate_ratio = round(1.0 - (unique_posts / len(post_list)), 2)
        if duplicate_ratio >= 0.5:
            rule_flags.append(f"High post duplication rate ({int(duplicate_ratio * 100)}% identical lines)")

    # 4. Statistical ML Model Scoring (TF-IDF + LR)
    pipeline = load_nlp_pipeline()
    top_phrases: List[Dict[str, Any]] = []

    if pipeline is not None:
        try:
            model_prob = float(pipeline.predict_proba([combined_text])[0, 1])
            top_phrases = get_top_predictive_phrases(pipeline, combined_text, top_k=5)
        except Exception:
            model_prob = 0.5
    else:
        # Fallback heuristic score if model artifact missing
        model_prob = min(0.3 * len(rule_flags), 1.0)

    # 5. Hybrid Composite Text Risk Score
    # Combine model probability with rule flags
    rule_bonus = min(len(rule_flags) * 0.15, 0.40)
    composite_score = min(max(model_prob * 0.70 + rule_bonus, 0.0), 1.0)
    composite_score = round(composite_score, 4)

    # Risk level categorization
    if composite_score >= 0.70:
        label = "spam-like"
        risk_level = "High"
    elif composite_score >= 0.40:
        label = "spam-like" if len(rule_flags) > 0 else "non-spam-like"
        risk_level = "Medium"
    else:
        label = "non-spam-like"
        risk_level = "Low"

    # 6. Human-Readable Explanation
    if risk_level == "High":
        explanation = (
            f"Text exhibits multiple strong spam/phishing characteristics (score: {composite_score:.2f}). "
            f"Indicators triggered: {len(rule_flags)} rule-based flags."
        )
    elif risk_level == "Medium":
        explanation = (
            f"Moderate spam indicators detected (score: {composite_score:.2f}). "
            f"Some promotional or link patterns warrant caution."
        )
    else:
        explanation = (
            f"Text content appears largely natural and conversational (score: {composite_score:.2f}). "
            "No aggressive spam patterns detected."
        )

    limitations = [
        "Text analysis indicates spam-like patterns but cannot prove account inauthenticity or ownership.",
        "Short texts (<10 words) have lower statistical certainty.",
    ]

    return {
        "label": label,
        "text_risk_score": composite_score,
        "text_risk_level": risk_level,
        "top_suspicious_phrases": top_phrases,
        "rule_based_flags": rule_flags,
        "duplicate_ratio": duplicate_ratio,
        "explanation": explanation,
        "limitations": limitations,
    }

