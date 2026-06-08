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
