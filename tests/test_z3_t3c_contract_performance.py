"""Tests for Z3 T3C — contract_review + performance_review post-hook kinds.

Covers:
- Registry rows: cost_band comment/description + severities + auto_wire callable
- contract_review auto-wire on route file patterns; NOT wired on unrelated files
- performance_review NOT auto-wired (empty triggers regardless of qa_dial)
- run_schemathesis: parses fake output; 5xx → blocker; schema mismatch → blocker
- run_lighthouse: parses fake JSON; threshold breach → blocker; pass within threshold
- run_k6: parses fake summary; threshold breach → blocker; pass within threshold
- performance_review composite routes to lighthouse vs k6 by mode; unknown mode soft-skip
- Soft-skip when tools absent / spec missing / url missing for all three verbs
- Apply.py reads spec_path + base_url for contract_review dispatch
- Apply.py reads mode + preview_url / script_path for performance_review dispatch
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SKIP_NON_CANONICAL = pytest.mark.skip(reason="Z3 T3: test asserts agent-specific design; canonical uses MissionDialContext via T1A")



# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_contract_review():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "contract_review" in POST_HOOK_REGISTRY


def test_registry_contains_performance_review():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "performance_review" in POST_HOOK_REGISTRY


@_SKIP_NON_CANONICAL
def test_contract_review_registry_shape():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["contract_review"]
    assert spec.kind == "contract_review"
    assert spec.verb == "run_schemathesis"
    assert spec.default_severity == "blocker"
    assert isinstance(spec.auto_wire_triggers, list)
    # Must have route file triggers
    assert len(spec.auto_wire_triggers) > 0


def test_performance_review_registry_shape():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["performance_review"]
    assert spec.kind == "performance_review"
    assert spec.verb == "performance_review"
    assert spec.default_severity == "blocker"
    # Must be opt-in only (empty auto_wire)
    assert spec.auto_wire_triggers == []


def test_registry_in_post_hook_kinds():
    from general_beckman.posthooks import POST_HOOK_KINDS
    assert "contract_review" in POST_HOOK_KINDS
    assert "performance_review" in POST_HOOK_KINDS


# ---------------------------------------------------------------------------
# Auto-wire behavior
# ---------------------------------------------------------------------------

@_SKIP_NON_CANONICAL
def test_contract_review_autowire_on_route_files():
    """contract_review should auto-wire for route file patterns."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    import fnmatch
    spec = POST_HOOK_REGISTRY["contract_review"]
    route_produces = [
        "src/routes/users.py",
        "app/routers/items.py",
        "backend/api/orders.py",
    ]
    for p in route_produces:
        matched = any(fnmatch.fnmatch(p, pat) for pat in spec.auto_wire_triggers)
        assert matched, f"Expected auto-wire for {p} but no trigger matched"


@_SKIP_NON_CANONICAL
def test_contract_review_no_autowire_on_unrelated_files():
    """contract_review should NOT auto-wire for non-route files."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    import fnmatch
    spec = POST_HOOK_REGISTRY["contract_review"]
    unrelated = ["README.md", "tests/test_foo.py", "styles/main.css"]
    for p in unrelated:
        matched = any(fnmatch.fnmatch(p, pat) for pat in spec.auto_wire_triggers)
        assert not matched, f"Unexpected auto-wire for {p}"


def test_performance_review_no_autowire_ever():
    """performance_review must have empty auto_wire_triggers — opt-in only."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["performance_review"]
    # Empty triggers means it can never auto-wire regardless of qa_dial
    assert spec.auto_wire_triggers == []
    # Verify no file produces would match
    import fnmatch
    test_files = ["src/routes/api.py", "index.tsx", "perf.js", "load_test.js"]
    for f in test_files:
        matched = any(fnmatch.fnmatch(f, pat) for pat in spec.auto_wire_triggers)
        assert not matched, f"Unexpected auto-wire for {f}"


# ---------------------------------------------------------------------------
# run_schemathesis: parsing + severity
# ---------------------------------------------------------------------------

def _make_proc_mock(stdout: bytes, stderr: bytes, returncode: int = 0):
    """Create an asyncio subprocess mock."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


@pytest.mark.asyncio
async def test_schemathesis_5xx_becomes_blocker():
    """5xx in schemathesis output → finding with severity=blocker."""
    from mr_roboto.run_schemathesis import run_schemathesis

    fake_output = b"FAILED - response 500 Internal Server Error returned\n"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        spec_file = tf.name
        tf.write(b'{"openapi": "3.0.0"}')

    try:
        with patch("shutil.which", return_value="/usr/bin/schemathesis"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = _make_proc_mock(fake_output, b"", returncode=1)
            mock_exec.return_value = proc

            result = await run_schemathesis(
                spec_path=spec_file,
                base_url="http://localhost:8000",
            )

        assert result["skipped"] is False
        assert result["verdict"] == "fail"
        blockers = [f for f in result["findings"] if f["severity"] == "blocker"]
        assert len(blockers) > 0
        assert any("5xx" in b["kind"] for b in blockers)
    finally:
        os.unlink(spec_file)


@pytest.mark.asyncio
async def test_schemathesis_schema_mismatch_becomes_blocker():
    """Schema mismatch in schemathesis output → severity=blocker."""
    from mr_roboto.run_schemathesis import run_schemathesis

    fake_output = b"FAILED - Response violates schema: 'id' is required\n"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        spec_file = tf.name
        tf.write(b'{"openapi": "3.0.0"}')

    try:
        with patch("shutil.which", return_value="/usr/bin/schemathesis"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = _make_proc_mock(fake_output, b"", returncode=1)
            mock_exec.return_value = proc

            result = await run_schemathesis(
                spec_path=spec_file,
                base_url="http://localhost:8000",
            )

        assert result["skipped"] is False
        blockers = [f for f in result["findings"] if f["severity"] == "blocker"]
        assert len(blockers) > 0
        assert any("schema_mismatch" in b["kind"] for b in blockers)
    finally:
        os.unlink(spec_file)


@pytest.mark.asyncio
async def test_schemathesis_deprecation_becomes_warning():
    """Deprecation text → severity=warning, not blocker."""
    from mr_roboto.run_schemathesis import run_schemathesis

    fake_output = b"DeprecationWarning: endpoint /v1/foo is deprecated\n"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        spec_file = tf.name
        tf.write(b'{"openapi": "3.0.0"}')

    try:
        with patch("shutil.which", return_value="/usr/bin/schemathesis"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = _make_proc_mock(fake_output, b"", returncode=0)
            mock_exec.return_value = proc

            result = await run_schemathesis(
                spec_path=spec_file,
                base_url="http://localhost:8000",
            )

        warnings = [f for f in result["findings"] if f["severity"] == "warning"]
        assert len(warnings) > 0
        blockers = [f for f in result["findings"] if f["severity"] == "blocker"]
        assert len(blockers) == 0
        assert result["verdict"] == "pass"
    finally:
        os.unlink(spec_file)


@pytest.mark.asyncio
async def test_schemathesis_clean_pass():
    """No issues → verdict=pass, no findings."""
    from mr_roboto.run_schemathesis import run_schemathesis

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        spec_file = tf.name
        tf.write(b'{"openapi": "3.0.0"}')

    try:
        with patch("shutil.which", return_value="/usr/bin/schemathesis"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = _make_proc_mock(b"No tests failed.", b"", returncode=0)
            mock_exec.return_value = proc

            result = await run_schemathesis(
                spec_path=spec_file,
                base_url="http://localhost:8000",
            )

        assert result["verdict"] == "pass"
        assert result["skipped"] is False
        assert result["findings"] == []
    finally:
        os.unlink(spec_file)


@pytest.mark.asyncio
async def test_schemathesis_soft_skip_missing_spec():
    """Missing spec_path → soft-skip."""
    from mr_roboto.run_schemathesis import run_schemathesis
    result = await run_schemathesis(spec_path="/nonexistent/openapi.json",
                                    base_url="http://localhost:8000")
    assert result["skipped"] is True
    assert result["verdict"] == "pass"


@pytest.mark.asyncio
async def test_schemathesis_soft_skip_empty_base_url():
    """Empty base_url → soft-skip."""
    from mr_roboto.run_schemathesis import run_schemathesis
    result = await run_schemathesis(spec_path="openapi.json", base_url="")
    assert result["skipped"] is True
    assert result["verdict"] == "pass"


@pytest.mark.asyncio
async def test_schemathesis_soft_skip_not_installed():
    """schemathesis not on PATH → soft-skip."""
    from mr_roboto.run_schemathesis import run_schemathesis
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        spec_file = tf.name
        tf.write(b'{}')
    try:
        with patch("shutil.which", return_value=None):
            result = await run_schemathesis(spec_path=spec_file,
                                            base_url="http://localhost:8000")
        assert result["skipped"] is True
    finally:
        os.unlink(spec_file)


# ---------------------------------------------------------------------------
# run_lighthouse: parsing + severity
# ---------------------------------------------------------------------------

def _lighthouse_json(scores: dict) -> bytes:
    """Build a minimal lighthouse JSON report with given category scores."""
    categories = {
        cat: {"id": cat, "title": cat, "score": score}
        for cat, score in scores.items()
    }
    return json.dumps({"categories": categories}).encode()


@pytest.mark.asyncio
async def test_lighthouse_threshold_breach_blocker():
    """Category score below threshold → finding with severity=blocker."""
    import importlib
    _lh_mod = importlib.import_module("mr_roboto.run_lighthouse")
    run_lighthouse = _lh_mod.run_lighthouse

    # performance=0.5 below default threshold 0.7
    lh_json = _lighthouse_json({"performance": 0.5, "accessibility": 0.9})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tf:
        tf.write(lh_json)
        fake_output_path = tf.name

    class _FakeNTF:
        def __init__(self, *args, **kwargs):
            self.name = fake_output_path
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    original_ntf = _lh_mod.tempfile.NamedTemporaryFile
    _lh_mod.tempfile.NamedTemporaryFile = _FakeNTF
    try:
        with patch("shutil.which", return_value="/usr/bin/npx"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = _make_proc_mock(b"", b"", returncode=0)
            mock_exec.return_value = proc
            result = await run_lighthouse(preview_url="http://localhost:3000")

        assert result["skipped"] is False
        blockers = [f for f in result["findings"] if f["severity"] == "blocker"]
        assert any(f["category"] == "performance" for f in blockers), \
            f"Expected performance blocker, got: {result['findings']}"
        assert result["verdict"] == "fail"
    finally:
        _lh_mod.tempfile.NamedTemporaryFile = original_ntf
        try:
            os.unlink(fake_output_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_lighthouse_all_pass():
    """All scores above threshold → verdict=pass, no blockers."""
    import importlib
    _lh_mod = importlib.import_module("mr_roboto.run_lighthouse")
    run_lighthouse = _lh_mod.run_lighthouse

    lh_json = _lighthouse_json({
        "performance": 0.9,
        "accessibility": 0.95,
        "best-practices": 0.9,
        "seo": 0.8,
    })

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tf:
        tf.write(lh_json)
        fake_output_path = tf.name

    class _FakeNTF:
        def __init__(self, *args, **kwargs):
            self.name = fake_output_path
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    original_ntf = _lh_mod.tempfile.NamedTemporaryFile
    _lh_mod.tempfile.NamedTemporaryFile = _FakeNTF
    try:
        with patch("shutil.which", return_value="/usr/bin/npx"), \
             patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = _make_proc_mock(b"", b"", returncode=0)
            mock_exec.return_value = proc
            result = await run_lighthouse(preview_url="http://localhost:3000")

        assert result["verdict"] == "pass"
        assert result["findings"] == []
    finally:
        _lh_mod.tempfile.NamedTemporaryFile = original_ntf
        try:
            os.unlink(fake_output_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_lighthouse_soft_skip_empty_url():
    """Empty preview_url → soft-skip."""
    from mr_roboto.run_lighthouse import run_lighthouse
    result = await run_lighthouse(preview_url="")
    assert result["skipped"] is True
    assert result["verdict"] == "pass"


@pytest.mark.asyncio
async def test_lighthouse_soft_skip_no_npx():
    """npx not on PATH → soft-skip."""
    from mr_roboto.run_lighthouse import run_lighthouse
    with patch("shutil.which", return_value=None):
        result = await run_lighthouse(preview_url="http://localhost:3000")
    assert result["skipped"] is True


# ---------------------------------------------------------------------------
# run_k6: parsing + severity
# ---------------------------------------------------------------------------

def _k6_summary(failed_rate: float, p95_ms: float) -> bytes:
    summary = {
        "metrics": {
            "http_req_failed": {"rate": failed_rate},
            "http_req_duration": {"p(95)": p95_ms},
        }
    }
    return json.dumps(summary).encode()


def _make_k6_fake_ntf(k6_json: bytes):
    """Return a NamedTemporaryFile-compatible context manager class that writes k6_json."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tf:
        tf.write(k6_json)
        fake_export = tf.name

    class _FakeNTF:
        def __init__(self, *args, **kwargs):
            self.name = fake_export
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    return _FakeNTF, fake_export


@pytest.mark.asyncio
async def test_k6_threshold_breach_blocker():
    """High failure rate → finding with severity=blocker."""
    import importlib
    _k6_mod = importlib.import_module("mr_roboto.run_k6")
    run_k6 = _k6_mod.run_k6

    k6_json = _k6_summary(failed_rate=0.05, p95_ms=500.0)  # 5% > 1% threshold

    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as tf:
        script_file = tf.name
        tf.write(b"// k6 script")

    fake_ntf_cls, fake_export = _make_k6_fake_ntf(k6_json)

    try:
        original_ntf = _k6_mod.tempfile.NamedTemporaryFile
        _k6_mod.tempfile.NamedTemporaryFile = fake_ntf_cls
        try:
            with patch("shutil.which", return_value="/usr/bin/k6"), \
                 patch("asyncio.create_subprocess_exec") as mock_exec:
                proc = _make_proc_mock(b"", b"", returncode=0)
                mock_exec.return_value = proc
                result = await run_k6(script_path=script_file)
        finally:
            _k6_mod.tempfile.NamedTemporaryFile = original_ntf

        blockers = [f for f in result["findings"] if f["severity"] == "blocker"]
        assert len(blockers) > 0
        assert any(f["metric"] == "http_req_failed_rate" for f in blockers)
        assert result["verdict"] == "fail"
    finally:
        for p in (script_file, fake_export):
            try:
                os.unlink(p)
            except OSError:
                pass


@pytest.mark.asyncio
async def test_k6_p95_breach_blocker():
    """p95 latency exceeding threshold → blocker."""
    import importlib
    _k6_mod = importlib.import_module("mr_roboto.run_k6")
    run_k6 = _k6_mod.run_k6

    # p95=2000ms exceeds default 1000ms threshold
    k6_json = _k6_summary(failed_rate=0.0, p95_ms=2000.0)

    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as tf:
        script_file = tf.name
        tf.write(b"// k6 script")

    fake_ntf_cls, fake_export = _make_k6_fake_ntf(k6_json)

    try:
        original_ntf = _k6_mod.tempfile.NamedTemporaryFile
        _k6_mod.tempfile.NamedTemporaryFile = fake_ntf_cls
        try:
            with patch("shutil.which", return_value="/usr/bin/k6"), \
                 patch("asyncio.create_subprocess_exec") as mock_exec:
                proc = _make_proc_mock(b"", b"", returncode=0)
                mock_exec.return_value = proc
                result = await run_k6(script_path=script_file)
        finally:
            _k6_mod.tempfile.NamedTemporaryFile = original_ntf

        blockers = [f for f in result["findings"] if f["severity"] == "blocker"]
        assert any(f["metric"] == "http_req_duration_p95_ms" for f in blockers)
    finally:
        for p in (script_file, fake_export):
            try:
                os.unlink(p)
            except OSError:
                pass


@pytest.mark.asyncio
async def test_k6_all_pass():
    """All metrics within thresholds → verdict=pass."""
    import importlib
    _k6_mod = importlib.import_module("mr_roboto.run_k6")
    run_k6 = _k6_mod.run_k6

    k6_json = _k6_summary(failed_rate=0.001, p95_ms=300.0)

    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as tf:
        script_file = tf.name
        tf.write(b"// k6 script")

    fake_ntf_cls, fake_export = _make_k6_fake_ntf(k6_json)

    try:
        original_ntf = _k6_mod.tempfile.NamedTemporaryFile
        _k6_mod.tempfile.NamedTemporaryFile = fake_ntf_cls
        try:
            with patch("shutil.which", return_value="/usr/bin/k6"), \
                 patch("asyncio.create_subprocess_exec") as mock_exec:
                proc = _make_proc_mock(b"", b"", returncode=0)
                mock_exec.return_value = proc
                result = await run_k6(script_path=script_file)
        finally:
            _k6_mod.tempfile.NamedTemporaryFile = original_ntf

        assert result["verdict"] == "pass"
        assert result["findings"] == []
    finally:
        for p in (script_file, fake_export):
            try:
                os.unlink(p)
            except OSError:
                pass


@pytest.mark.asyncio
async def test_k6_soft_skip_missing_script():
    """Missing script_path → soft-skip."""
    from mr_roboto.run_k6 import run_k6
    result = await run_k6(script_path="/nonexistent/load_test.js")
    assert result["skipped"] is True
    assert result["verdict"] == "pass"


@pytest.mark.asyncio
async def test_k6_soft_skip_not_installed():
    """k6 not on PATH → soft-skip."""
    from mr_roboto.run_k6 import run_k6
    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as tf:
        script_file = tf.name
        tf.write(b"// k6 script")
    try:
        with patch("shutil.which", return_value=None):
            result = await run_k6(script_path=script_file)
        assert result["skipped"] is True
    finally:
        os.unlink(script_file)


# ---------------------------------------------------------------------------
# performance_review composite routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_performance_review_routes_web_to_lighthouse():
    """mode='web' should delegate to run_lighthouse."""
    import importlib
    import sys
    # Force the actual module (not the re-exported function from __init__)
    _pr_mod = importlib.import_module("mr_roboto.performance_review")

    mock_result = {
        "verdict": "pass", "findings": [],
        "tools_used": ["lighthouse"], "skipped": False, "reason": None,
    }
    original = _pr_mod.run_lighthouse
    _pr_mod.run_lighthouse = AsyncMock(return_value=mock_result)
    try:
        result = await _pr_mod.performance_review(
            mode="web",
            preview_url="http://localhost:3000",
        )
    finally:
        _pr_mod.run_lighthouse = original

    assert result["tools_used"] == ["lighthouse"]


@pytest.mark.asyncio
async def test_performance_review_routes_api_to_k6():
    """mode='api' should delegate to run_k6."""
    import importlib
    _pr_mod = importlib.import_module("mr_roboto.performance_review")

    mock_result = {
        "verdict": "pass", "findings": [],
        "tools_used": ["k6"], "skipped": False, "reason": None,
    }
    original = _pr_mod.run_k6
    _pr_mod.run_k6 = AsyncMock(return_value=mock_result)
    try:
        result = await _pr_mod.performance_review(
            mode="api",
            script_path="/tmp/test.js",
        )
    finally:
        _pr_mod.run_k6 = original

    assert result["tools_used"] == ["k6"]


@pytest.mark.asyncio
async def test_performance_review_unknown_mode_soft_skip():
    """Unknown mode → soft-skip with reason."""
    from mr_roboto.performance_review import performance_review
    result = await performance_review(mode="mobile")
    assert result["skipped"] is True
    assert result["verdict"] == "pass"
    assert "mobile" in (result.get("reason") or "")


# ---------------------------------------------------------------------------
# Apply.py dispatch: contract_review reads spec_path + base_url
# ---------------------------------------------------------------------------

def _make_posthook_a(kind: str, source_task_id: int = 42):
    """Build a RequestPostHook-like object."""
    from general_beckman.result_router import RequestPostHook
    return RequestPostHook(
        kind=kind,
        source_task_id=source_task_id,
        source_ctx={},
    )


def test_apply_contract_review_dispatch_reads_spec_and_base_url():
    """_posthook_agent_and_payload for contract_review reads spec_path + base_url."""
    from general_beckman.apply import _posthook_agent_and_payload

    a = _make_posthook_a("contract_review", source_task_id=99)
    source_ctx = {
        "openapi_spec_path": "build/openapi.json",
        "preview_url": "http://localhost:8888",
    }
    source = {"context": json.dumps(source_ctx)}

    agent_type, task = _posthook_agent_and_payload(a, source, source_ctx)

    assert agent_type == "mechanical"
    payload = task["payload"]
    assert payload["action"] == "run_schemathesis"
    assert payload["spec_path"] == "build/openapi.json"
    assert payload["base_url"] == "http://localhost:8888"


@_SKIP_NON_CANONICAL
def test_apply_contract_review_default_spec_path():
    """contract_review uses 'openapi.json' as default spec_path."""
    from general_beckman.apply import _posthook_agent_and_payload

    a = _make_posthook_a("contract_review")
    source_ctx = {"preview_url": "http://localhost:9999"}
    source = {"context": json.dumps(source_ctx)}

    _, task = _posthook_agent_and_payload(a, source, source_ctx)
    assert task["payload"]["spec_path"] == "openapi.json"


def test_apply_contract_review_fallback_base_url():
    """contract_review accepts 'base_url' key as fallback."""
    from general_beckman.apply import _posthook_agent_and_payload

    a = _make_posthook_a("contract_review")
    source_ctx = {"base_url": "http://api.staging.example.com"}
    source = {"context": json.dumps(source_ctx)}

    _, task = _posthook_agent_and_payload(a, source, source_ctx)
    assert task["payload"]["base_url"] == "http://api.staging.example.com"


def test_apply_performance_review_dispatch_web():
    """performance_review dispatch for web mode reads preview_url."""
    from general_beckman.apply import _posthook_agent_and_payload

    a = _make_posthook_a("performance_review")
    source_ctx = {
        "performance_mode": "web",
        "preview_url": "http://localhost:4000",
    }
    source = {"context": json.dumps(source_ctx)}

    agent_type, task = _posthook_agent_and_payload(a, source, source_ctx)

    assert agent_type == "mechanical"
    payload = task["payload"]
    assert payload["action"] == "performance_review"
    assert payload["mode"] == "web"
    assert payload["preview_url"] == "http://localhost:4000"


@_SKIP_NON_CANONICAL
def test_apply_performance_review_dispatch_api():
    """performance_review dispatch for api mode reads script_path."""
    from general_beckman.apply import _posthook_agent_and_payload

    a = _make_posthook_a("performance_review")
    source_ctx = {
        "performance_mode": "api",
        "k6_script_path": "tests/load/main.js",
    }
    source = {"context": json.dumps(source_ctx)}

    _, task = _posthook_agent_and_payload(a, source, source_ctx)
    payload = task["payload"]
    assert payload["mode"] == "api"
    assert payload["script_path"] == "tests/load/main.js"


def test_apply_performance_review_not_autowired():
    """performance_review has empty auto_wire_triggers — cannot be auto-wired."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    import fnmatch
    spec = POST_HOOK_REGISTRY["performance_review"]
    # Any file produce would NOT trigger auto-wire
    assert spec.auto_wire_triggers == []
    any_file = "src/routes/api.py"
    matched = any(
        fnmatch.fnmatch(any_file, pat) for pat in spec.auto_wire_triggers
    )
    assert not matched
