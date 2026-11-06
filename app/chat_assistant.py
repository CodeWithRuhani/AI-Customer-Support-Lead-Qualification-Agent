"""Route chat to grounded local replies or the optional local Ollama model."""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from app.ai_classifier import AIClassifier
from app.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = (
    r"\bignore (?:all )?(?:previous|prior|above) instructions\b",
    r"\bdisregard (?:all )?(?:previous|prior|above) instructions\b",
    r"\breveal (?:the )?(?:system|developer) prompt\b",
    r"\bshow (?:me )?(?:the )?(?:hidden )?(?:system prompt|instructions)\b",
    r"\b(?:what is|repeat|print|display|tell me) (?:your )?(?:system|developer) prompt\b",
    r"\b(?:developer|admin) mode\b",
    r"\boverride your instructions\b",
    r"\bforget (?:all )?(?:rules|instructions)\b",
    r"\bact as (?:an? )?(?:unrestricted|uncensored|different|new)\b",
    r"\b(?:jailbreak|bypass your safety|ignore safety rules)\b",
)

INJECTION_REPLY = (
    "I can’t change my operating rules or reveal hidden instructions. I can "
    "help with the supported products, pricing enquiries, technical support, "
    "billing concerns, and company information."
)

HELP_REPLY = (
    "I can help with CRM software, analytics, API integrations, the mobile app, "
    "enterprise plans, pricing enquiries, and technical or billing issues. "
    "What would you like to know?"
)

PORTFOLIO_REPLY = (
    "The services documented for this prototype are CRM software for lead "
    "management and sales pipelines; an analytics platform for dashboards, "
    "reporting, and insights; API/webhook integrations; a mobile app; and "
    "enterprise plans. Which one would you like details about?"
)

PRODUCT_ANSWERS = {
    "CRM Software": (
        "The CRM software is associated with lead management and sales "
        "pipelines. The project does not define a fuller feature list. Are "
        "you looking for a particular CRM capability?"
    ),
    "Analytics Platform": (
        "The analytics platform is associated with dashboards, reporting, "
        "and insights. The project does not document specific integrations "
        "or plan limits. What would you like to know about?"
    ),
    "API / Integration": (
        "The project identifies API, webhook, and system integration as "
        "supported areas, but does not publish endpoint or setup details. "
        "Which system are you trying to connect?"
    ),
    "Mobile App": (
        "The project lists a mobile app and references iOS and Android. It "
        "does not include app-store links or a detailed feature list."
    ),
    "Enterprise Plan": (
        "The project lists enterprise plans and team or bulk licensing. "
        "Specific limits and prices are not documented. How many team members "
        "or licenses are you considering?"
    ),
}

UNKNOWN_FACT_REPLY = (
    "I don’t have verified information about that in this service’s available "
    "details, so I don’t want to guess. I can help with products, pricing "
    "enquiries, technical support, billing concerns, or company information."
)

FOLLOW_UP_REPLY = (
    "Could you clarify what you mean? For a product question, tell me which "
    "service you mean; for a technical issue, describe what you tried and the "
    "error you saw. You can ask without providing personal details."
)

PRICING_REPLY = (
    "I don’t have a verified price list in the available information. Tell me "
    "which product or plan you mean and, if relevant, your approximate team "
    "size; I can identify the product interest, but I can’t invent a quote."
)

SUPPORT_REPLY = (
    "I can help narrow down the issue, but I can’t access accounts or change "
    "passwords. Please describe what you were trying to do and include the "
    "exact error message. Don’t send passwords, verification codes, or "
    "payment-card details."
)

COMPLAINT_REPLY = (
    "I’m sorry you’re dealing with that. Please describe the service or "
    "billing issue without sharing passwords, verification codes, or full "
    "payment-card details. I can record the concern for a registered "
    "customer, but I can’t issue refunds or promise a resolution."
)

VAGUE_PATTERNS = (
    r"^(?:what|which|how) (?:kind|type) of (?:details?|information|help)\??$",
    r"^(?:tell me )?(?:more|more details|explain|elaborate)\.?$",
    r"^(?:what do you mean|how so|can you explain)\??$",
)

PRICING_PATTERNS = (
    r"\b(?:price|prices|pricing|cost|how much|quote|discount)\b",
)
BUSINESS_FACT_PATTERNS = {
    "hours": r"\b(?:business hours|opening hours|open today|working hours)\b",
    "location": r"\b(?:office|address|location|located)\b",
    "language": r"\b(?:languages?|supported language)\b",
    "case studies": r"\b(?:case studies|customer stories|references)\b",
    "industry": r"\b(?:industries|which industry|sectors served)\b",
    "refund policy": r"\b(?:refund policy|refund eligibility|money-back guarantee)\b",
    "support SLA": r"\b(?:support sla|response time|service-level agreement)\b",
}
BUSINESS_FACT_LABELS = {
    "hours": "business hours",
    "location": "office address/location",
    "language": "supported languages",
    "case studies": "case studies",
    "industry": "industries served",
    "refund policy": "refund policy",
    "support SLA": "support response times or SLA",
}


class ChatAssistant:
    """Classifies messages and answers only from documented prototype facts."""

    def __init__(
        self,
        classifier: AIClassifier,
        ollama_client: OllamaClient | None = None,
    ):
        self._classifier = classifier
        self._ollama_client = ollama_client or OllamaClient()

    @staticmethod
    def is_prompt_injection(message: str) -> bool:
        return any(re.search(pattern, message, re.IGNORECASE) for pattern in INJECTION_PATTERNS)

    def respond(
        self,
        message: str,
        context: List[str] | None = None,
        history: List[Dict[str, str]] | None = None,
    ) -> Dict:
        context = context or []
        history = history or []
        if self.is_prompt_injection(message):
            logger.warning("Prompt-injection-like text blocked by fixed response policy")
            return {
                "reply": INJECTION_REPLY,
                "intent": "General",
                "confidence": 1.0,
                "product_interest": "General Inquiry",
                "blocked": True,
                "answer_source": "local",
            }

        lowered = message.strip().lower()
        is_vague_follow_up = any(
            re.search(pattern, lowered) for pattern in VAGUE_PATTERNS
        )
        unknown_topics = [
            BUSINESS_FACT_LABELS[topic]
            for topic, pattern in BUSINESS_FACT_PATTERNS.items()
            if re.search(pattern, lowered)
        ]
        asks_pricing = any(
            re.search(pattern, lowered) for pattern in PRICING_PATTERNS
        )
        classification_text = message
        if is_vague_follow_up and context:
            classification_text = context[-1]
        elif is_vague_follow_up and history:
            previous_user_messages = [
                turn["content"] for turn in history if turn.get("role") == "user"
            ]
            if previous_user_messages:
                classification_text = previous_user_messages[-1]
        classification = self._classifier.classify(classification_text)
        intent = classification["intent"]
        product_interest = classification["product_interest"]

        is_greeting = bool(
            re.fullmatch(
                r"(?:hi|hello|hey|good morning|good afternoon)[!. ]*", lowered
            )
        )
        asks_for_help = bool(
            re.fullmatch(r"how can you help(?: me)?[?.! ]*", lowered)
        )
        asks_for_portfolio = any(
            re.search(pattern, lowered)
            for pattern in (
                r"\bproducts and services\b",
                r"\bwhat (?:products|services) (?:do you offer|are available)\b",
                r"\bwhat do you offer\b",
                r"\bwhat (?:do you|does your company) provide\b",
                r"\byour products\b",
                r"\byour services\b",
                r"\bproduct service\b",
            )
        )

        rule_reply = None
        if is_vague_follow_up:
            if intent == "Sales" and product_interest == "Enterprise Plan":
                rule_reply = (
                    "For the Enterprise Plan enquiry, how many team members or "
                    "licenses should the quote cover? Specific pricing is not "
                    "documented."
                )
            elif intent == "Sales" and product_interest != "General Inquiry":
                rule_reply = (
                    f"Which details about {product_interest} would help you "
                    "decide? I can clarify documented features, but I don’t "
                    "have a verified price list."
                )
            elif intent == "Support":
                rule_reply = (
                    "Could you share the exact error and what you were trying "
                    "to do? Please don’t include passwords or verification codes."
                )
            elif intent == "Complaint":
                rule_reply = (
                    "Could you clarify whether the concern is about billing, "
                    "a product issue, or a previous support request? Please "
                    "leave out passwords and full payment-card details."
                )
            elif product_interest != "General Inquiry":
                rule_reply = (
                    f"Which aspect of {product_interest} would you like me to "
                    "explain?"
                )
            else:
                rule_reply = FOLLOW_UP_REPLY
        elif is_greeting:
            rule_reply = HELP_REPLY
        elif asks_for_help:
            rule_reply = HELP_REPLY
        elif intent == "Complaint":
            rule_reply = COMPLAINT_REPLY
        elif intent == "Support":
            rule_reply = SUPPORT_REPLY
        elif unknown_topics and asks_pricing:
            topics = ", ".join(unknown_topics + ["pricing"])
            rule_reply = (
                f"I don’t have verified information about {topics} in the "
                "available details, so I won’t guess. I can help with the "
                "documented products and identify the details you need."
            )
        elif asks_pricing:
            rule_reply = PRICING_REPLY
        elif unknown_topics:
            topics = ", ".join(unknown_topics)
            rule_reply = (
                f"I don’t have verified information about {topics} in the "
                "available details, so I won’t guess. I can help with the "
                "documented products and services instead."
            )
        elif intent == "Sales":
            rule_reply = PRODUCT_ANSWERS.get(product_interest, PRICING_REPLY)
        elif product_interest != "General Inquiry":
            rule_reply = PRODUCT_ANSWERS[product_interest]
        elif asks_for_portfolio:
            rule_reply = PORTFOLIO_REPLY
        elif intent == "General" and any(
            term in lowered for term in ("help", "can you", "what can", "who are you")
        ):
            rule_reply = HELP_REPLY
        elif intent == "General":
            rule_reply = UNKNOWN_FACT_REPLY
        else:
            rule_reply = FOLLOW_UP_REPLY

        can_use_ollama = (
            intent == "General"
            and product_interest == "General Inquiry"
            and not asks_pricing
            and not unknown_topics
            and not is_greeting
            and not asks_for_help
            and not asks_for_portfolio
            and (not is_vague_follow_up or bool(context or history))
        )
        reply = (
            self._ollama_client.generate_reply(
                message,
                [
                    turn
                    for turn in history
                    if turn.get("role") != "user"
                    or not self.is_prompt_injection(turn.get("content", ""))
                ],
            )
            if can_use_ollama
            else None
        )
        answer_source = "ollama" if reply else "local"
        if reply is None:
            reply = rule_reply or FOLLOW_UP_REPLY
            logger.info("Using deterministic local response")

        logger.info(
            "Prepared response (source=%s, intent=%s, product_interest=%s)",
            answer_source,
            intent,
            product_interest,
        )
        return {
            "reply": reply,
            "intent": intent,
            "confidence": classification["confidence"],
            "product_interest": product_interest,
            "blocked": False,
            "answer_source": answer_source,
        }
