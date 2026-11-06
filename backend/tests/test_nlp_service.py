"""
tests/test_nlp_service.py – Unit tests for NLP service.

Tests:
  1. Spam text (promotional phrasing, high score).
  2. Normal conversational text (low score).
  3. Empty / missing text (not_available, insufficient evidence, explicit limitations).
  4. Repeated posts (duplicate ratio calculation and flag).
  5. URL-heavy text with link shorteners and suspicious TLDs.
"""
from app.services.nlp_service import analyze_text


def test_nlp_spam_text():
    bio = "DM to earn fast cash! FREE CRYPTO GIVEAWAY!"
    posts = ["Click here now to claim your prize! Guaranteed profit."]
    res = analyze_text(bio=bio, posts=posts)

    assert res["label"] == "spam-like"
    assert res["text_risk_score"] > 0.60
    assert res["text_risk_level"] in ["Medium", "High"]
    assert len(res["rule_based_flags"]) >= 1
    assert any("promotional" in f.lower() for f in res["rule_based_flags"])
    assert "explanation" in res
    assert len(res["limitations"]) > 0


def test_nlp_normal_text():
    bio = "Software developer from Boston. Reading sci-fi and brewing coffee."
    posts = ["Enjoyed the open source conference today. Great discussions on distributed databases."]
    res = analyze_text(bio=bio, posts=posts)

    assert res["label"] == "non-spam-like"
    assert res["text_risk_score"] < 0.40
    assert res["text_risk_level"] == "Low"
    assert len(res["rule_based_flags"]) == 0


def test_nlp_empty_or_missing_text():
    """Missing text must NEVER be treated as safe."""
    res = analyze_text(bio=None, posts=None)
    assert res["label"] == "not_available"
    assert res["text_risk_score"] is None
    assert res["text_risk_level"] == "Insufficient evidence"
    assert len(res["limitations"]) >= 2
    assert any("NOT treated as zero risk" in l for l in res["limitations"])

    # Empty whitespace strings
    res_ws = analyze_text(bio="   ", posts=["", "   "])
    assert res_ws["label"] == "not_available"
    assert res_ws["text_risk_score"] is None


def test_nlp_repeated_posts_duplicate_ratio():
    posts = [
        "Follow me back for free followers!",
        "Follow me back for free followers!",
        "Follow me back for free followers!",
        "Different message here.",
    ]
    res = analyze_text(bio=None, posts=posts)
    # 4 posts, 2 unique -> duplicate ratio = 0.5 (50%)
    assert res["duplicate_ratio"] == 0.5
    assert any("duplication" in f.lower() for f in res["rule_based_flags"])


def test_nlp_url_heavy_and_suspicious_tld():
    bio = "Special discounts: http://special-offer.xyz and http://bit.ly/deal99 and http://192.168.1.1/free"
    res = analyze_text(bio=bio, posts=None)
    flags = [f.lower() for f in res["rule_based_flags"]]

    assert any("shortener" in f for f in flags)
    assert any("tld" in f for f in flags)
    assert any("ip address" in f for f in flags)
    assert res["text_risk_score"] >= 0.40

