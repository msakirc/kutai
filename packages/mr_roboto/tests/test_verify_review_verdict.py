import asyncio
import pytest

@pytest.mark.parametrize("status", ["pass", "approved"])
def test_pass_class_ok(status):
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={"status": status, "issues": []})
    assert res["ok"] is True
    assert res["verdict_class"] == "pass"

def test_fail_class_flags_route():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={
        "status": "fail",
        "issues": [{"target_artifact": "x", "severity": "blocker", "problem": "p"}],
    })
    assert res["ok"] is False
    assert res["verdict_class"] == "fail"
    assert res["issues"]

def test_unparseable_is_task_failure_not_route():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result=None)
    assert res["ok"] is False
    assert res["verdict_class"] == "malformed"

def test_unknown_status_is_malformed():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={"status": "weird"})
    assert res["verdict_class"] == "malformed"


# ── verdict-key shape (step 1.13 research_quality_review uses `verdict`) ───

def test_verdict_key_fail_routes():
    """Step 1.13 emits {"verdict": "fail", ...} — must classify as fail, not
    malformed (its artifact_schema field is `verdict`, not `status`)."""
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={
        "verdict": "fail",
        "issues": [{"target_artifact": "market_research_report",
                    "severity": "blocker", "problem": "no evidence"}],
    })
    assert res["ok"] is False
    assert res["verdict_class"] == "fail"
    assert res["issues"]


def test_verdict_key_pass_ok():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={"verdict": "pass"})
    assert res["ok"] is True
    assert res["verdict_class"] == "pass"


def test_verdict_key_needs_minor_fixes_passes():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={"verdict": "needs_minor_fixes"})
    assert res["verdict_class"] == "pass"


def test_neither_status_nor_verdict_is_malformed():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result={"foo": "bar"})
    assert res["verdict_class"] == "malformed"


def test_dispatch_pass_completes():
    from mr_roboto import run as mr_run
    task = {"id": 0, "mission_id": 0,
            "payload": {"action": "verify_review_verdict",
                        "review_result": {"status": "pass", "issues": []}}}
    res = asyncio.run(mr_run(task))
    assert res.status == "completed"

def test_dispatch_fail_surfaces_failed():
    from mr_roboto import run as mr_run
    task = {"id": 0, "mission_id": 0,
            "payload": {"action": "verify_review_verdict",
                        "review_result": {"status": "fail", "issues": [{"problem": "p"}]}}}
    res = asyncio.run(mr_run(task))
    assert res.status == "failed"
    assert (res.result or {}).get("verdict_class") == "fail"


# ── findings shape (step 10.5 encryption_and_logging_review) ──────────────

def _enc_log_result(enc_findings, log_findings):
    """Build 10.5's native combined two-artifact result object."""
    return {
        "encryption_verification_result": {"findings": enc_findings},
        "logging_security_result": {"findings": log_findings},
    }


def test_findings_critical_high_fail_with_mapped_issues():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    rr = _enc_log_result(
        enc_findings=[{
            "check": "tls", "severity": "critical",
            "description": "DB connection not using SSL",
            "remediation": "enable sslmode=require",
            "target_artifact": "owasp_audit_result",
        }],
        log_findings=[{
            "check": "pii", "severity": "high",
            "description": "user email logged in plaintext",
            "remediation": "mask PII",
            "target_artifact": "observability_strategy",
        }],
    )
    res = verify_review_verdict(review_result=rr)
    assert res["ok"] is False
    assert res["verdict_class"] == "fail"
    # Both blocking findings mapped to the router's issue shape.
    issues = res["issues"]
    assert len(issues) == 2
    targets = {i["target_artifact"] for i in issues}
    assert targets == {"owasp_audit_result", "observability_strategy"}
    by_target = {i["target_artifact"]: i for i in issues}
    assert by_target["owasp_audit_result"]["severity"] == "critical"
    assert by_target["owasp_audit_result"]["problem"] == "DB connection not using SSL"
    assert by_target["observability_strategy"]["severity"] == "high"


def test_findings_all_low_info_pass():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    rr = _enc_log_result(
        enc_findings=[{"check": "rotation", "severity": "low",
                       "description": "key rotation cadence undocumented",
                       "remediation": "document"}],
        log_findings=[{"check": "perms", "severity": "info",
                       "description": "log dir is 0644", "remediation": "n/a"}],
    )
    res = verify_review_verdict(review_result=rr)
    assert res["ok"] is True
    assert res["verdict_class"] == "pass"
    assert res["issues"] == []


def test_findings_without_target_artifact_routes_to_founder():
    """A blocking finding with no target_artifact -> issue.target_artifact=None.
    map_tagged_issues then leaves it unresolved -> founder-halt."""
    from mr_roboto.verify_review_verdict import verify_review_verdict
    rr = _enc_log_result(
        enc_findings=[{"check": "systemic", "severity": "high",
                       "description": "no central secrets manager",
                       "remediation": "introduce one"}],  # no target_artifact
        log_findings=[],
    )
    res = verify_review_verdict(review_result=rr)
    assert res["verdict_class"] == "fail"
    assert len(res["issues"]) == 1
    assert res["issues"][0]["target_artifact"] is None


def test_findings_empty_both_artifacts_pass():
    from mr_roboto.verify_review_verdict import verify_review_verdict
    res = verify_review_verdict(review_result=_enc_log_result([], []))
    assert res["ok"] is True
    assert res["verdict_class"] == "pass"


def test_findings_top_level_array_also_accepted():
    """Single top-level findings array (not nested per-artifact) still works."""
    from mr_roboto.verify_review_verdict import verify_review_verdict
    rr = {"findings": [{"check": "tls", "severity": "critical",
                        "description": "weak cipher", "remediation": "fix",
                        "target_artifact": "owasp_audit_result"}]}
    res = verify_review_verdict(review_result=rr)
    assert res["verdict_class"] == "fail"
    assert res["issues"][0]["target_artifact"] == "owasp_audit_result"


def test_status_takes_precedence_over_findings():
    """If a reviewer somehow emits BOTH status and findings, status wins
    (keeps the 7 standard reviewers' contract stable)."""
    from mr_roboto.verify_review_verdict import verify_review_verdict
    rr = {"status": "pass", "issues": [],
          "findings": [{"severity": "critical", "description": "x"}]}
    res = verify_review_verdict(review_result=rr)
    assert res["verdict_class"] == "pass"


def test_findings_to_issues_helper_is_pure():
    from mr_roboto.verify_review_verdict import findings_to_issues
    findings = [
        {"severity": "critical", "description": "a", "target_artifact": "t1"},
        {"severity": "medium", "description": "b"},          # dropped
        {"severity": "high", "check": "c-check"},            # no description → check
        {"severity": "info", "description": "d"},            # dropped
    ]
    issues = findings_to_issues(findings)
    assert len(issues) == 2
    assert issues[0] == {"target_artifact": "t1", "severity": "critical", "problem": "a"}
    assert issues[1]["problem"] == "c-check"
    assert issues[1]["target_artifact"] is None


def test_findings_dispatch_fail_surfaces_failed():
    from mr_roboto import run as mr_run
    rr = {"encryption_verification_result": {"findings": [
        {"severity": "critical", "description": "boom",
         "target_artifact": "owasp_audit_result"}]},
        "logging_security_result": {"findings": []}}
    task = {"id": 0, "mission_id": 0,
            "payload": {"action": "verify_review_verdict", "review_result": rr}}
    res = asyncio.run(mr_run(task))
    assert res.status == "failed"
    assert (res.result or {}).get("verdict_class") == "fail"


def test_findings_dispatch_pass_completes():
    from mr_roboto import run as mr_run
    rr = {"encryption_verification_result": {"findings": [
        {"severity": "low", "description": "minor"}]},
        "logging_security_result": {"findings": []}}
    task = {"id": 0, "mission_id": 0,
            "payload": {"action": "verify_review_verdict", "review_result": rr}}
    res = asyncio.run(mr_run(task))
    assert res.status == "completed"
