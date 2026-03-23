# security/sensitivity.py
"""
Phase 10.3 — Sensitivity Detection

Regex-based scanner to detect sensitive data in task content.
Returns a sensitivity level: public, private, secret.

Used by the router to restrict model selection:
  - public  → any model (cloud or local)
  - private → prefer local models (ollama, llamacpp)
  - secret  → local only; block if no local model available
"""
import re
from enum import Enum
from typing import NamedTuple


class SensitivityLevel(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    SECRET = "secret"


class SensitivityResult(NamedTuple):
    level: SensitivityLevel
    matches: list[str]   # descriptions of what was detected


# ─── Pattern Definitions ────────────────────────────────────────────────────

# API Keys — common provider prefixes (SECRET)
_API_KEY_PATTERNS = [
    (r'\bsk-[a-zA-Z0-9]{20,}', "OpenAI API key (sk-...)"),
    (r'\bsk-proj-[a-zA-Z0-9]{20,}', "OpenAI project key (sk-proj-...)"),
    (r'\bghp_[a-zA-Z0-9]{36,}', "GitHub personal access token (ghp_)"),
    (r'\bgho_[a-zA-Z0-9]{36,}', "GitHub OAuth token (gho_)"),
    (r'\bghu_[a-zA-Z0-9]{36,}', "GitHub user-to-server token (ghu_)"),
    (r'\bghs_[a-zA-Z0-9]{36,}', "GitHub server-to-server token (ghs_)"),
    (r'\bghr_[a-zA-Z0-9]{36,}', "GitHub refresh token (ghr_)"),
    (r'\bAKIA[0-9A-Z]{16}\b', "AWS access key ID (AKIA)"),
    (r'\bxoxb-[0-9a-zA-Z-]+', "Slack bot token (xoxb-)"),
    (r'\bxoxp-[0-9a-zA-Z-]+', "Slack user token (xoxp-)"),
    (r'\bxoxs-[0-9a-zA-Z-]+', "Slack session token (xoxs-)"),
    (r'\bnpm_[a-zA-Z0-9]{36,}', "npm token"),
    (r'\bpypi-[a-zA-Z0-9]{50,}', "PyPI API token"),
    (r'\bglpat-[a-zA-Z0-9_-]{20,}', "GitLab personal access token"),
    (r'\bSG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}', "SendGrid API key"),
    (r'\brk_live_[a-zA-Z0-9]{20,}', "Stripe restricted key"),
    (r'\bsk_live_[a-zA-Z0-9]{20,}', "Stripe secret key"),
    (r'\bBearer\s+[a-zA-Z0-9._-]{20,}', "Bearer token"),
]

# Private key blocks (SECRET)
_PRIVATE_KEY_PATTERNS = [
    (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', "Private key block"),
    (r'-----BEGIN\s+(?:EC\s+)?PRIVATE\s+KEY-----', "EC private key block"),
    (r'-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----', "OpenSSH private key"),
    (r'-----BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----', "PGP private key"),
]

# Credit card numbers (SECRET)
_CREDIT_CARD_PATTERNS = [
    # Visa: starts with 4, 13-16 digits
    (r'\b4[0-9]{12}(?:[0-9]{3})?\b', "Visa card number"),
    # Mastercard: starts with 51-55 or 2221-2720
    (r'\b5[1-5][0-9]{14}\b', "Mastercard number"),
    # Amex: starts with 34 or 37, 15 digits
    (r'\b3[47][0-9]{13}\b', "American Express card number"),
    # Card with separators: 1234-5678-9012-3456 or 1234 5678 9012 3456
    (r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b', "Card number with separators"),
]

# SSN patterns (SECRET)
_SSN_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', "SSN pattern (XXX-XX-XXXX)"),
]

# Email patterns (PRIVATE)
_EMAIL_PATTERNS = [
    (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', "Email address"),
]

# Password patterns (SECRET)
_PASSWORD_PATTERNS = [
    (r'(?:password|passwd|pwd)\s*[:=]\s*\S+', "Password assignment"),
    (r'(?:secret|token|api_key|apikey)\s*[:=]\s*["\']?\S+', "Secret/token assignment"),
]


def detect_sensitivity(text: str) -> SensitivityResult:
    """
    Scan text for sensitive data patterns.

    Returns (SensitivityLevel, list_of_match_descriptions).

    Escalation logic:
      - Any SECRET pattern → SensitivityLevel.SECRET
      - Any PRIVATE pattern → SensitivityLevel.PRIVATE
      - No matches → SensitivityLevel.PUBLIC
    """
    if not text:
        return SensitivityResult(SensitivityLevel.PUBLIC, [])

    secret_matches: list[str] = []
    private_matches: list[str] = []

    # ── Check SECRET patterns ──
    for pattern, desc in _API_KEY_PATTERNS:
        if re.search(pattern, text):
            secret_matches.append(desc)

    for pattern, desc in _PRIVATE_KEY_PATTERNS:
        if re.search(pattern, text):
            secret_matches.append(desc)

    for pattern, desc in _CREDIT_CARD_PATTERNS:
        if re.search(pattern, text):
            secret_matches.append(desc)

    for pattern, desc in _SSN_PATTERNS:
        if re.search(pattern, text):
            secret_matches.append(desc)

    for pattern, desc in _PASSWORD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            secret_matches.append(desc)

    # ── Check PRIVATE patterns ──
    for pattern, desc in _EMAIL_PATTERNS:
        if re.search(pattern, text):
            private_matches.append(desc)

    # ── Determine level ──
    if secret_matches:
        return SensitivityResult(
            SensitivityLevel.SECRET,
            secret_matches + private_matches,
        )
    elif private_matches:
        return SensitivityResult(SensitivityLevel.PRIVATE, private_matches)
    else:
        return SensitivityResult(SensitivityLevel.PUBLIC, [])


# ─── Secret Redaction ──────────────────────────────────────────────────────

# Compiled patterns for redaction (SECRET-level only)
_REDACT_PATTERNS: list[tuple[re.Pattern, str]] = []

def _build_redact_patterns() -> list[tuple[re.Pattern, str]]:
    """Compile all SECRET-level patterns once for reuse."""
    if _REDACT_PATTERNS:
        return _REDACT_PATTERNS
    all_secret = (
        _API_KEY_PATTERNS
        + _PRIVATE_KEY_PATTERNS
        + _CREDIT_CARD_PATTERNS
        + _SSN_PATTERNS
        + _PASSWORD_PATTERNS
    )
    for pattern_str, desc in all_secret:
        _REDACT_PATTERNS.append((re.compile(pattern_str, re.IGNORECASE), desc))
    return _REDACT_PATTERNS


def redact_secrets(text: str, placeholder: str = "[REDACTED]") -> str:
    """
    Replace all SECRET-level sensitive patterns in text with a placeholder.

    Returns the redacted text. Safe to call on any outgoing message.
    """
    if not text:
        return text
    result = text
    for compiled, _desc in _build_redact_patterns():
        result = compiled.sub(placeholder, result)
    return result


def scan_task(
    title: str,
    description: str,
    context: str | dict | None = None,
) -> SensitivityResult:
    """
    Convenience wrapper: scan all task fields at once.

    Combines title + description + serialized context into one text block
    and runs detect_sensitivity on it.
    """
    parts = [title or "", description or ""]
    if context:
        if isinstance(context, dict):
            import json
            parts.append(json.dumps(context, default=str))
        else:
            parts.append(str(context))
    combined = "\n".join(parts)
    return detect_sensitivity(combined)
