"""Z9 T2D — ``growth_anti_patterns`` agent tool.

Thin async wrapper around :func:`src.growth.anti_patterns.detect_all` so the
``growth_digest_synthesizer`` agent can run the three deterministic growth
anti-pattern detectors (vanity metric / engagement vampire / insufficient-N)
without re-deriving the math in prose.

The agent passes the ``digest_input`` bundle (the same dict the
``analytics_digest`` mechanical executor placed on ``task.context``); a
JSON string or an already-decoded dict are both accepted.
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("tools.growth_anti_patterns")


async def growth_anti_patterns(digest_input: str = "{}") -> str:
    """Run the three growth anti-pattern detectors over ``digest_input``.

    Args:
        digest_input: the analytics bundle — a JSON string (preferred, the
            shape the LLM emits) or a dict. Relevant keys: ``north_star``,
            ``event_count``, ``retention_curve``, ``experiments``.

    Returns:
        A readable summary of every warning found, or a "no anti-patterns"
        line when the digest is clean.
    """
    from src.growth.anti_patterns import detect_all

    parsed: dict[str, Any]
    if isinstance(digest_input, dict):
        parsed = digest_input
    elif isinstance(digest_input, str):
        try:
            loaded = json.loads(digest_input or "{}")
            parsed = loaded if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return (
                "growth_anti_patterns: could not parse digest_input — "
                "pass the digest_input dict as a JSON object string."
            )
    else:
        parsed = {}

    try:
        findings = detect_all(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("detect_all raised", error=str(exc))
        return f"growth_anti_patterns: detector error — {exc}"

    if not findings:
        return "growth_anti_patterns: no anti-patterns detected (0 warnings)."

    lines = [f"growth_anti_patterns: {len(findings)} warning(s) detected:"]
    for f in findings:
        lines.append(f"- [{f['code']}/{f['severity']}] {f['message']}")
    return "\n".join(lines)


__all__ = ["growth_anti_patterns"]
