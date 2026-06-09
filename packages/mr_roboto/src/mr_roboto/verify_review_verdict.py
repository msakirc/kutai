"""Read a reviewer's *_review_result verdict and classify it.

pass-class  -> the reviewer accepted the artifact; the step completes.
fail-class  -> route to general_beckman.review_routing.route_review_failure.
malformed   -> the reviewer task itself failed (no parseable verdict): normal
               DLQ, NOT the routing path.

Two reviewer output shapes are accepted:

1. STATUS shape (the 7 standard reviewers): a top-level
   ``{"status": pass|approved|needs_minor_fixes|fail, "issues": [...]}``.
   ``status`` drives the verdict; ``issues`` rides along to the router.

2. FINDINGS shape (step 10.5 encryption_and_logging_review): the reviewer has
   no pass/fail status. Instead it emits one or two findings arrays
   (``findings: [{check, severity, description, remediation, target_artifact?}]``)
   keyed under its artifact names (e.g. ``encryption_verification_result`` and
   ``logging_security_result``), OR a single top-level ``findings`` array.
   A finding at severity ``critical`` or ``high`` is a blocker → ``fail``;
   anything else (medium/low/info only) → ``pass``. Blocking findings are
   mapped to the standard issue shape the router consumes
   (``{target_artifact, severity, problem}``) so routing works unchanged.
"""
from __future__ import annotations

from typing import Any

_PASS_CLASS = {"pass", "approved", "needs_minor_fixes"}
_FAIL_CLASS = {"fail"}

# Findings at these severities block the gate (mirrors the artifact_schema
# blockers declaration: {field: severity, levels: [critical, high]}).
_BLOCKING_SEVERITIES = {"critical", "high"}


def _iter_findings(review_result: dict) -> list[dict]:
    """Collect every finding dict from a findings-shape review_result.

    Accepts both a single top-level ``findings`` array and per-artifact
    wrappers (``{<artifact_name>: {"findings": [...]}, ...}``). Non-dict
    findings are ignored. Returns a flat list preserving order.
    """
    out: list[dict] = []
    top = review_result.get("findings")
    if isinstance(top, list):
        out.extend(f for f in top if isinstance(f, dict))
    for value in review_result.values():
        if isinstance(value, dict):
            nested = value.get("findings")
            if isinstance(nested, list):
                out.extend(f for f in nested if isinstance(f, dict))
    return out


def _has_findings_shape(review_result: dict) -> bool:
    """True iff review_result carries a findings array (top-level or nested)
    and no top-level ``status`` (status takes precedence when present)."""
    if "status" in review_result:
        return False
    if isinstance(review_result.get("findings"), list):
        return True
    return any(
        isinstance(v, dict) and isinstance(v.get("findings"), list)
        for v in review_result.values()
    )


def findings_to_issues(findings: list[dict]) -> list[dict]:
    """Map blocking (critical/high) findings to the router's issue shape.

    Each blocking finding becomes ``{target_artifact, severity, problem}``:
      * ``problem``        <- finding ``description``
      * ``severity``       <- finding ``severity`` (critical|high)
      * ``target_artifact``<- finding ``target_artifact`` (None if absent →
                              the router treats it as unresolved → founder-halt)

    Pure / deterministic — non-blocking findings are dropped (they do not
    reject the artifact). Order is preserved.
    """
    issues: list[dict] = []
    for f in findings:
        sev = str(f.get("severity") or "").lower()
        if sev not in _BLOCKING_SEVERITIES:
            continue
        issues.append({
            "target_artifact": f.get("target_artifact"),
            "severity": sev,
            "problem": f.get("description") or f.get("check") or "",
        })
    return issues


def verify_review_verdict(*, review_result: Any) -> dict[str, Any]:
    if not isinstance(review_result, dict):
        return {"ok": False, "verdict_class": "malformed",
                "error": "no parseable review verdict", "issues": []}

    # FINDINGS shape (10.5): no status, but one or more findings arrays.
    if _has_findings_shape(review_result):
        findings = _iter_findings(review_result)
        blocking = findings_to_issues(findings)
        if blocking:
            return {"ok": False, "verdict_class": "fail", "issues": blocking}
        return {"ok": True, "verdict_class": "pass", "issues": []}

    # STATUS shape (the 7 standard reviewers).
    if "status" not in review_result:
        return {"ok": False, "verdict_class": "malformed",
                "error": "no parseable review verdict", "issues": []}
    status = str(review_result.get("status") or "").lower()
    issues = review_result.get("issues") or []
    if status in _FAIL_CLASS:
        return {"ok": False, "verdict_class": "fail", "issues": issues}
    if status in _PASS_CLASS:
        return {"ok": True, "verdict_class": "pass", "issues": issues}
    return {"ok": False, "verdict_class": "malformed",
            "error": f"unknown verdict status {status!r}", "issues": issues}
