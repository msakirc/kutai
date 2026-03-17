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

_ONBOARD_RE = re.compile(
    r"^(?:/product\s+)?onboard\s+(.+)$", re.IGNORECASE
)


def should_start_workflow(message: str) -> bool:
    """Return True if the message matches any workflow trigger pattern."""
    text = extract_idea_text(message)
    return any(pattern.search(text) for pattern in WORKFLOW_PATTERNS)


def detect_onboarding_path(message: str) -> Optional[str]:
    """If message starts with 'onboard <path>', return the path. Otherwise None."""
    m = _ONBOARD_RE.match(message.strip())
    if m:
        return m.group(1).strip()
    return None


def extract_idea_text(message: str) -> str:
    """Strip /product prefix if present, return remaining text."""
    text = message.strip()
    if text.lower().startswith("/product"):
        text = text[len("/product"):].strip()
    return text
