"""Dispatch logic for starting workflows from user messages."""

import re
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("workflows.engine.dispatch")

WORKFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(build|create|make|develop)\s+(me\s+)?(an?\s+)?(\w+\s+)*(product|app|application|saas|platform|tool|service|website|startup)",
        re.IGNORECASE,
    ),
    re.compile(
        r"idea\s+(for|to|about)\s+.+\s+(app|product|saas|platform|service)",
        re.IGNORECASE,
    ),
    re.compile(r"idea\.to\.product", re.IGNORECASE),
    re.compile(r"full\s+product", re.IGNORECASE),
    re.compile(r"from\s+scratch\s+.+\s+(app|product|platform)", re.IGNORECASE),
    re.compile(r"mvp\s+.+\s+(build|create|develop)", re.IGNORECASE),
    re.compile(r"launch\s+.+\s+(product|app|startup)", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Shopping workflow dispatch patterns
# ---------------------------------------------------------------------------
SHOPPING_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "quick_search": [
        # Price queries
        re.compile(r"\b(fiyat|price|ne\s+kadar|how\s+much)\b", re.IGNORECASE),
        re.compile(r"\b(en\s+ucuz|cheapest|en\s+uygun)\b", re.IGNORECASE),
    ],
    "combo_research": [
        # Comparison queries
        re.compile(r"\b(karşılaştır|kıyasla|compare)\b", re.IGNORECASE),
        re.compile(r"\bvs\.?\b", re.IGNORECASE),
    ],
    "shopping": [
        # General shopping / buying intent
        re.compile(r"\b(almak\s+istiyorum|satın\s+al|want\s+to\s+buy)\b", re.IGNORECASE),
        re.compile(r"\b(should\s+i\s+buy|almalı\s+mıyım|alsam\s+mı)\b", re.IGNORECASE),
        re.compile(r"\b(öner|tavsiye|recommend)\b", re.IGNORECASE),
    ],
    "exploration": [
        # Deal hunting
        re.compile(r"\b(indirim|kampanya|deal|discount)\b", re.IGNORECASE),
        re.compile(r"\b(fırsat|kupon|coupon|promosyon)\b", re.IGNORECASE),
    ],
    "gift_recommendation": [
        # Gift queries
        re.compile(r"\b(hediye|gift)\b", re.IGNORECASE),
    ],
}

_ONBOARD_RE = re.compile(
    r"^(?:/(?:product|mission)\s+)?onboard\s+(.+)$", re.IGNORECASE
)


def should_start_workflow(message: str) -> bool:
    """Return True if the message matches any workflow trigger pattern."""
    text = extract_idea_text(message)
    return any(pattern.search(text) for pattern in WORKFLOW_PATTERNS)


def detect_shopping_intent(message: str) -> bool:
    """Return True if the message has any shopping-related intent."""
    text = message.strip()
    for patterns in SHOPPING_PATTERNS.values():
        for pattern in patterns:
            if pattern.search(text):
                return True
    return False


def should_start_shopping_workflow(message: str) -> Optional[str]:
    """Return the shopping workflow name to use, or None if not a shopping query.

    Returns one of: "shopping", "quick_search", "combo_research",
    "gift_recommendation", "exploration", or None.
    """
    text = message.strip()
    for workflow_name, patterns in SHOPPING_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                return workflow_name
    return None


def detect_onboarding_path(message: str) -> Optional[str]:
    """If message starts with 'onboard <path>', return the path. Otherwise None."""
    m = _ONBOARD_RE.match(message.strip())
    if m:
        return m.group(1).strip()
    return None


def extract_idea_text(message: str) -> str:
    """Strip /product or /mission prefix if present, return remaining text."""
    text = message.strip()
    lower = text.lower()
    if lower.startswith("/mission"):
        text = text[len("/mission"):].strip()
    elif lower.startswith("/product"):
        text = text[len("/product"):].strip()
    return text
