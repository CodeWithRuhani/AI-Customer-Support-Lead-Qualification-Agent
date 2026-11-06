"""
lead_scorer.py
--------------
LeadScorer: assigns an explainable 0-100 lead score using a
documented, rule-based weighted-feature method (not a black box).

Why rule-based rather than a trained classifier here?
The assignment asks for a *documented, explainable* method, and we
have no historical conversion-outcome data for this agent (that
labelled dataset lives in Assignment 2's CRM). A transparent weighted
scorecard is the right, honest choice here, and every point awarded is
traceable to a specific input feature - which directly satisfies
requirement "Explain which input features influenced the score".
"""

from __future__ import annotations
import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Weights are intentionally documented inline so the method is fully
# auditable by a sales manager who is not a data scientist.
INTENT_WEIGHTS = {
    "Sales": 40,
    "Complaint": 10,
    "Support": 5,
    "General": 0,
}

HIGH_VALUE_KEYWORDS = ["enterprise", "bulk", "urgent", "budget", "demo", "quote", "purchase"]
KEYWORD_BONUS = 5          # per matched keyword, capped below
KEYWORD_BONUS_CAP = 15

CONFIDENCE_WEIGHT = 15     # scaled by classifier confidence (0-1)

CONTACT_COMPLETENESS_WEIGHT = 10   # customer supplied phone number
ENGAGEMENT_WEIGHT_PER_MSG = 3      # repeated engagement signals real interest
ENGAGEMENT_CAP = 15

PRODUCT_SPECIFIC_BONUS = 15        # customer named a specific product, not "General Inquiry"


class LeadScorer:
    """Computes an explainable lead score and category for a customer."""

    def score(
        self,
        intent: str,
        confidence: float,
        message: str,
        product_interest: str,
        has_phone: bool,
        previous_message_count: int,
    ) -> Dict:
        breakdown: List[Tuple[str, int]] = []

        # 1. Intent base score
        intent_score = INTENT_WEIGHTS.get(intent, 0)
        breakdown.append((f"Intent='{intent}'", intent_score))

        # 2. Keyword signal
        lowered = message.lower()
        matched = [
            keyword
            for keyword in HIGH_VALUE_KEYWORDS
            if re.search(rf"\b{re.escape(keyword)}\b", lowered)
        ]
        keyword_score = min(len(matched) * KEYWORD_BONUS, KEYWORD_BONUS_CAP)
        if matched:
            breakdown.append((f"High-value keywords {matched}", keyword_score))

        # 3. Classifier confidence
        confidence_score = round(confidence * CONFIDENCE_WEIGHT)
        breakdown.append((f"Classifier confidence={confidence}", confidence_score))

        # 4. Contact completeness
        contact_score = CONTACT_COMPLETENESS_WEIGHT if has_phone else 0
        breakdown.append(("Phone number provided" if has_phone else "No phone number", contact_score))

        # 5. Engagement (repeat contact = real interest)
        engagement_score = min(previous_message_count * ENGAGEMENT_WEIGHT_PER_MSG, ENGAGEMENT_CAP)
        breakdown.append((f"Previous messages={previous_message_count}", engagement_score))

        # 6. Named a specific product/service
        product_score = PRODUCT_SPECIFIC_BONUS if product_interest != "General Inquiry" else 0
        breakdown.append((f"Product interest='{product_interest}'", product_score))

        raw_total = sum(v for _, v in breakdown)
        total = max(0, min(100, raw_total))
        if total != raw_total:
            breakdown.append(("100-point score cap", total - raw_total))

        category = self._categorize(total)
        next_action = self._recommend_action(category, intent)
        logger.info(
            "Lead score calculated (intent=%s, score=%s, category=%s)",
            intent,
            total,
            category,
        )

        return {
            "score": total,
            "category": category,
            "next_action": next_action,
            "explanation": [{"feature": f, "points": p} for f, p in breakdown],
        }

    @staticmethod
    def _categorize(score: int) -> str:
        if score >= 65:
            return "Hot"
        elif score >= 35:
            return "Warm"
        return "Cold"

    @staticmethod
    def _recommend_action(category: str, intent: str) -> str:
        if category == "Hot":
            return "Assign to sales rep for immediate follow-up call within 24 hours"
        if category == "Warm":
            return "Send tailored product information and schedule a follow-up in 3 days"
        if intent == "Complaint":
            return "Route to customer success for service recovery"
        return "Add to nurture email sequence, re-engage in 2 weeks"
