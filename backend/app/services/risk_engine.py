"""
services/risk_engine.py – Multi-modal risk combination and decision engine.

Core Architectural Requirements:
  1. Combines available evidence:
     - Calibrated ML Probability (0.0 - 1.0)
     - NLP Spam/Text Risk Score (0.0 - 1.0)
     - Behavioural Rule Indicators (0.0 - 1.0)
     - Network Analysis Result (0.0 - 1.0, optional)
  2. Dynamically re-normalizes modality weights over AVAILABLE modalities only.
  3. Never treats missing modalities as safe or zero risk.
  4. Insufficient Evidence handling: if available modalities < minimum threshold, returns
     classification = 'Insufficient evidence' and risk_score = None.
  5. The final composite metric is called 'risk_score' (scale 0 - 100), NOT a probability.
  6. Calibrated ML probability is shown separately and labeled as such.
  7. A single weak signal (e.g. newly created account alone) NEVER pushes beyond 'Suspicious'.
  8. Generates plain-English explanations, actionable recommendations, and explicit limitations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np

from app.config import get_settings
from app.schemas import AccountInput, Classification


class RiskEngine:
    def __init__(self):
        self.settings = get_settings()

    def evaluate_behavioral_rules(
        self,
        account: Union[AccountInput, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Evaluates heuristic behavioral rules.
        Returns a list of triggered indicator dictionaries and a normalized score (0.0 to 1.0).
        """
        if isinstance(account, AccountInput):
            d = account.model_dump()
        else:
            d = dict(account)

        indicators = []

        followers = d.get("followers")
        following = d.get("following")
        posts = d.get("posts")
        age = d.get("account_age_days")
        has_pic = d.get("has_profile_pic")
        bio = d.get("bio")
        username = str(d.get("username") or "")

        # 1. Very high following / followers ratio
        if followers is not None and following is not None:
            ratio = following / (followers + 1.0)
            if ratio >= 50.0 and following >= 100:
                indicators.append({
                    "name": "Extreme following-to-followers ratio",
                    "severity": "high",
                    "weight": 0.35,
                    "explanation": f"Account follows {ratio:.1f}x more users than follow it back ({following} following vs {followers} followers).",
                })
            elif ratio >= 10.0 and following >= 50:
                indicators.append({
                    "name": "Elevated following-to-followers ratio",
                    "severity": "medium",
                    "weight": 0.20,
                    "explanation": f"Account follows {ratio:.1f}x more users than follow it back.",
                })

        # 2. Extremely high posting rate
        if posts is not None and age is not None:
            safe_age = max(float(age), 1.0)
            rate = posts / safe_age
            if rate >= 20.0 and posts >= 30:
                indicators.append({
                    "name": "Extremely high posting rate",
                    "severity": "high",
                    "weight": 0.30,
                    "explanation": f"Account has published an average of {rate:.1f} posts/day since creation.",
                })

        # 3. Very new account combined with high activity (spambot blast pattern)
        if age is not None and age <= 3:
            if posts is not None and posts >= 15:
                indicators.append({
                    "name": "High activity on brand new account",
                    "severity": "high",
                    "weight": 0.35,
                    "explanation": f"Account is only {age} day(s) old but has already created {posts} posts.",
                })
            else:
                # Weak signal on its own
                indicators.append({
                    "name": "Newly created account",
                    "severity": "low",
                    "weight": 0.10,
                    "explanation": f"Account was created within the last {age} day(s). New accounts are not inherently suspicious.",
                })

        # 4. No profile picture plus empty bio
        has_no_pic = has_pic is False
        has_no_bio = bio is None or len(bio.strip()) == 0
        if has_no_pic and has_no_bio:
            indicators.append({
                "name": "Default avatar with empty bio",
                "severity": "medium",
                "weight": 0.25,
                "explanation": "Account lacks both a personalized profile picture and biography details.",
            })

        # 5. High digit ratio in username
        if username:
            digits = sum(1 for c in username if c.isdigit())
            digit_ratio = digits / len(username)
            if digit_ratio >= 0.35 and digits >= 4:
                indicators.append({
                    "name": "High digit ratio in username",
                    "severity": "medium",
                    "weight": 0.20,
                    "explanation": f"Username contains {int(digit_ratio * 100)}% numbers ({digits} digits), typical of programmatic batch registration.",
                })

        # 6. Suspicious URLs in bio
        if bio:
            bio_low = bio.lower()
            shorteners = ["bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly"]
            odd_tlds = [".xyz", ".top", ".club", ".click", ".win"]
            if any(s in bio_low for s in shorteners) or any(t in bio_low for t in odd_tlds):
                indicators.append({
                    "name": "Suspicious URL in profile bio",
                    "severity": "high",
                    "weight": 0.30,
                    "explanation": "Bio contains a link shortener or high-risk domain extension commonly used to evade spam filters.",
                })

        # Calculate combined behavioral score (clamped 0.0 - 1.0)
        total_weight = sum(ind["weight"] for ind in indicators)
        behavioral_score = min(total_weight, 1.0)
        return indicators, behavioral_score

    def compute_composite_risk(
        self,
        account: Union[AccountInput, Dict[str, Any]],
        ml_probability: Optional[float] = None,
        nlp_score: Optional[float] = None,
        network_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes ML, NLP, Behavioral, and Network evidence into an integrated assessment.
        """
        if isinstance(account, AccountInput):
            d = account.model_dump()
        else:
            d = dict(account)

        # 1. Behavioral evaluation
        indicators, behavioral_score = self.evaluate_behavioral_rules(d)

        # 2. Track modality availability
        available_modalities = {}
        missing_modalities = []

        # Modality: Profile metadata
        profile_fields_present = sum(1 for f in ["followers", "following", "posts", "account_age_days", "has_profile_pic", "bio"] if d.get(f) is not None)
        has_profile_modality = profile_fields_present >= 2

        # Modality: ML
        if ml_probability is not None:
            available_modalities["ml"] = {
                "score": ml_probability,
                "base_weight": self.settings.weight_ml,
            }
        else:
            missing_modalities.append("ML model scoring")

        # Modality: NLP
        if nlp_score is not None:
            available_modalities["nlp"] = {
                "score": nlp_score,
                "base_weight": self.settings.weight_nlp,
            }
        else:
            missing_modalities.append("NLP text/post analysis")

        # Modality: Behavioral
        if has_profile_modality:
            available_modalities["behavioral"] = {
                "score": behavioral_score,
                "base_weight": self.settings.weight_behavioral,
            }
        else:
            missing_modalities.append("Behavioral heuristics")

        # Modality: Network (optional)
        if network_score is not None:
            available_modalities["network"] = {
                "score": network_score,
                "base_weight": self.settings.weight_network,
            }

        # 3. Check for Insufficient Evidence
        # If fewer than 2 modalities are available or less than 2 profile fields exist
        if len(available_modalities) < 2 or not has_profile_modality:
            limitations = [
                "Insufficient evidence to produce an authentic risk assessment.",
                f"Missing modalities: {', '.join(missing_modalities)}.",
                "Missing data was NOT treated as zero risk (Rule #4).",
            ]
            return {
                "classification": Classification.insufficient_evidence,
                "risk_score": None,
                "calibrated_ml_probability": ml_probability,
                "data_availability": round(profile_fields_present / 7.0, 2),
                "indicators": indicators,
                "recommendations": [
                    "Provide additional profile data (follower counts, bio, posts) to allow a complete scan.",
                    "Verify user identity manually through external official channels.",
                ],
                "explanation": "Insufficient evidence: too few profile fields or signal modalities provided to evaluate authenticity.",
                "limitations": limitations,
                "available_modalities": list(available_modalities.keys()),
            }

        # 4. Renormalize weights across AVAILABLE modalities
        total_base_weight = sum(m["base_weight"] for m in available_modalities.values())
        weighted_sum = sum(
            (m["score"] * (m["base_weight"] / total_base_weight))
            for m in available_modalities.values()
        )

        # Scale to 0 - 100 risk score
        risk_score = round(weighted_sum * 100.0, 1)

        # 5. Apply Weak Signal Rule: A single weak indicator alone (e.g. new account)
        # must NEVER push an account into High risk
        is_only_new_account = (
            len(indicators) == 1
            and indicators[0]["name"] == "Newly created account"
            and (nlp_score is None or nlp_score < 0.4)
            and (ml_probability is None or ml_probability < 0.6)
        )
        if is_only_new_account and risk_score >= self.settings.risk_high_threshold:
            risk_score = self.settings.risk_high_threshold - 5.0

        # 6. Determine Classification Tier
        if risk_score >= self.settings.risk_high_threshold:
            classification = Classification.high_risk
        elif risk_score >= self.settings.risk_low_threshold:
            classification = Classification.suspicious
        else:
            classification = Classification.low_risk

        # 7. Formulate Recommendations & Explanation
        recommendations = []
        if classification == Classification.high_risk:
            recommendations.extend([
                "Exercise extreme caution: do not engage with unsolicited requests, financial pitches, or links.",
                "Cross-check handle across verified company or creator registries.",
                "Consider submitting a platform inauthentic behavior report if impersonation is suspected.",
            ])
            explanation = (
                f"Account displays multiple high-risk indicators across {len(available_modalities)} modalities "
                f"(composite risk score: {risk_score}/100). Calibrated ML probability is {ml_probability if ml_probability is not None else 'N/A'}."
            )
        elif classification == Classification.suspicious:
            recommendations.extend([
                "Perform manual review of profile details before trusting communications or transactions.",
                "Verify through secondary official communication channels (e.g. official email or website).",
            ])
            explanation = (
                f"Account exhibits moderate suspicious indicators (composite risk score: {risk_score}/100). "
                "Some anomalies warrant human review, though this does not prove inauthenticity."
            )
        else:
            recommendations.extend([
                "No immediate suspicious activity detected based on provided evidence.",
                "Standard online safety practices still apply.",
            ])
            explanation = (
                f"Account metrics align with standard authentic usage patterns (composite risk score: {risk_score}/100)."
            )

        limitations = [
            "Risk scores represent probabilistic suspicious indicators, NOT proof of identity or malicious intent.",
        ]
        if nlp_score is None:
            limitations.append("NLP post/bio analysis was unavailable; risk score was computed with renormalized weights on remaining modalities.")
        if "network" not in available_modalities:
            limitations.append("Social graph / network neighborhood data was not evaluated.")

        return {
            "classification": classification,
            "risk_score": risk_score,
            "calibrated_ml_probability": ml_probability,
            "data_availability": round(profile_fields_present / 7.0, 2),
            "indicators": indicators,
            "recommendations": recommendations,
            "explanation": explanation,
            "limitations": limitations,
            "available_modalities": list(available_modalities.keys()),
        }


# Singleton instance
risk_engine = RiskEngine()

