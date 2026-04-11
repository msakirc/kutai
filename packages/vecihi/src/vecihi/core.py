"""Core types and block detection for vecihi."""

import enum
from dataclasses import dataclass, field


class ScrapeTier(enum.IntEnum):
    HTTP = 0
    TLS = 1
    STEALTH = 2
    BROWSER = 3


@dataclass
class ScrapeResult:
    html: str
    status: int
    tier: ScrapeTier
    url: str
    error: str | None = None
    headers: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error and bool(self.html)


# Phrases in HTML that indicate a Cloudflare/WAF challenge page
CHALLENGE_MARKERS = [
    "just a moment",
    "checking your browser",
    "cdn-cgi/challenge-platform",
    "cf-browser-verification",
    "attention required",
    "ray id",
]


def detect_block(status: int, html: str, headers: dict) -> bool:
    """Detect if a response is blocked by WAF/anti-bot."""
    if status in (403, 429, 402, 451):
        return True
    if status == 503 and "cloudflare" in str(headers.get("server", "")).lower():
        return True
    if status == 200 and html:
        html_lower = html[:2000].lower()
        if any(marker in html_lower for marker in CHALLENGE_MARKERS):
            return True
    return False
