# security/risk_assessor.py
"""
Phase 14.3 — Delegation Intelligence / Risk Assessor

Assesses task risk: reversibility, cost, external impact, novelty.
Score 0-10. Below threshold → auto-execute. Above → approval request.
Learns from approval patterns.
"""
from __future__ import annotations
import re
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("security.risk_assessor")

# Default auto-execute threshold (0-10). Tasks scoring above this need approval.
DEFAULT_AUTO_THRESHOLD = 6.0

# Adjustable at runtime via /autonomy command
_current_threshold: float = DEFAULT_AUTO_THRESHOLD

# Risk factors: patterns and their base scores
_HIGH_RISK_PATTERNS = [
    (r"delete|remove|drop|truncate|rm\b|purge", 3.0, "destructive operation"),
    (r"deploy|publish|release|push.*prod|production", 2.5, "production deployment"),
    (r"send.*email|send.*message|post.*twitter|post.*social", 2.0, "external communication"),
    (r"payment|charge|billing|stripe|paypal", 3.0, "financial operation"),
    (r"credential|password|secret|api.?key", 2.5, "credential handling"),
    (r"database.*migration|alter.*table|schema.*change", 2.0, "schema change"),
    (r"git.*force|reset.*hard|rebase", 1.5, "destructive git operation"),
]

_LOW_RISK_PATTERNS = [
    (r"read|list|show|display|get|fetch|search|query", -1.5, "read-only"),
    (r"analyze|review|check|inspect|audit", -1.0, "analysis only"),
    (r"test|pytest|jest|unit test", -0.5, "testing"),
    (r"document|readme|comment", -0.5, "documentation"),
]


def assess_risk(task_title: str, task_description: str) -> dict:
    """
    Assess task risk score and whether it needs human approval.

    Returns:
        {
            "score": float (0-10),
            "needs_approval": bool,
            "risk_factors": [str],
            "threshold": float,
        }
    """
    text = f"{task_title} {task_description}".lower()
    score = 3.0  # baseline medium risk
    factors = []

    for pattern, delta, label in _HIGH_RISK_PATTERNS:
        if re.search(pattern, text):
            score += delta
            factors.append(f"+{delta:.1f} {label}")

    for pattern, delta, label in _LOW_RISK_PATTERNS:
        if re.search(pattern, text):
            score += delta
            factors.append(f"{delta:.1f} {label}")

    score = max(0.0, min(10.0, score))
    needs_approval = score >= _current_threshold

    return {
        "score": round(score, 1),
        "needs_approval": needs_approval,
        "risk_factors": factors,
        "threshold": _current_threshold,
    }


def set_autonomy_threshold(threshold: float) -> None:
    """Set the global auto-execute threshold. Higher = more autonomous."""
    global _current_threshold
    _current_threshold = max(0.0, min(10.0, threshold))
    logger.info(f"Autonomy threshold set to {_current_threshold}")


def get_autonomy_threshold() -> float:
    """Get the current auto-execute threshold."""
    return _current_threshold


def format_risk_assessment(assessment: dict) -> str:
    """Format a risk assessment for display."""
    score = assessment["score"]
    needs = assessment["needs_approval"]
    factors = assessment.get("risk_factors", [])
    threshold = assessment.get("threshold", DEFAULT_AUTO_THRESHOLD)

    emoji = "🔴" if score >= 7 else "🟡" if score >= 4 else "🟢"
    verdict = "⚠️ Needs approval" if needs else "✅ Auto-execute"

    lines = [
        f"{emoji} Risk Score: {score:.1f}/10 (threshold: {threshold:.1f})",
        verdict,
    ]
    if factors:
        lines.append("Factors: " + ", ".join(factors[:4]))
    return "\n".join(lines)
