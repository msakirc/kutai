"""Read a reviewer's *_review_result verdict and classify it.

pass-class  -> the reviewer accepted the artifact; the step completes.
fail-class  -> route to general_beckman.review_routing.route_review_failure.
malformed   -> the reviewer task itself failed (no parseable verdict): normal
               DLQ, NOT the routing path."""
from __future__ import annotations

from typing import Any

_PASS_CLASS = {"pass", "approved", "needs_minor_fixes"}
_FAIL_CLASS = {"fail"}


def verify_review_verdict(*, review_result: Any) -> dict[str, Any]:
    if not isinstance(review_result, dict) or "status" not in review_result:
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
