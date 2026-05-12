"""Tests for Z3 T4B — adr_drift_check post-hook (mechanical violation gate).

Covers:
- check_adr_drift reads register + ADRs correctly
- v2 ADR with forbidden_imports → import statement detected → finding emitted
- v2 ADR with forbidden_patterns → regex hit → finding
- v2 ADR with required_test_coverage=True + no test file → finding
- v2 ADR with all signals + no violations → verdict=pass
- v1 ADR (string falsification_signal) → mechanical skip; judgment_only_adr_ids
- falsification_signal=null → judgment-only; no mechanical fail
- Missing register + missing .adr/ → soft-skip (skipped=True, verdict=pass)
- Registry entry shape (cost_band, severity, verb, callable trigger)
- Auto-wire callable returns [] when qa_dial=off
- apply.py dispatch returns mechanical executor
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_adr(adr_dir: Path, adr_id: str, falsification_signal) -> Path:
    """Write an ADR JSON file to adr_dir and return its path."""
    adr_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": adr_id,
        "title": f"Test ADR {adr_id}",
        "falsification_signal": falsification_signal,
    }
    p = adr_dir / f"{adr_id}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_file(directory: Path, name: str, content: str) -> Path:
    """Write a file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Verb: check_adr_drift
# ---------------------------------------------------------------------------

class TestCheckAdrDrift:
    """Unit tests for the check_adr_drift verb."""

    def test_soft_skip_no_adr_dir(self):
        """When no register and no .adr/ dir exist, skipped=True, verdict=pass."""
        from mr_roboto.check_adr_drift import check_adr_drift
        result = asyncio.run(check_adr_drift(
            adr_register_path="/nonexistent/.adr/register.md",
            produced_files=[],
            workspace_path=None,
        ))
        assert result["skipped"] is True
        assert result["verdict"] == "pass"
        assert result["findings"] == []

    def test_soft_skip_empty_workspace(self, tmp_path):
        """Workspace without .adr/ → soft-skip."""
        from mr_roboto.check_adr_drift import check_adr_drift
        result = asyncio.run(check_adr_drift(
            adr_register_path=str(tmp_path / ".adr" / "register.md"),
            produced_files=["src/foo.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["skipped"] is True
        assert result["verdict"] == "pass"

    def test_null_falsification_signal_judgment_only(self, tmp_path):
        """null falsification_signal → judgment-only; verdict=pass."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-001"
        _write_adr(adr_dir, adr_id, None)
        # produce a dummy py file
        src = tmp_path / "src"
        src.mkdir()
        (src / "foo.py").write_text("import os\n", encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/foo.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "pass"
        assert result["skipped"] is False
        assert adr_id in result["judgment_only_adr_ids"]
        assert result["findings"] == []

    def test_v1_string_signal_judgment_only(self, tmp_path):
        """v1 string falsification_signal → mechanical skip; id in judgment_only_adr_ids."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-002"
        _write_adr(adr_dir, adr_id, "If the feature is not used in production within 6 months, revisit.")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=[],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "pass"
        assert adr_id in result["judgment_only_adr_ids"]
        assert result["findings"] == []

    def test_v2_forbidden_imports_hit(self, tmp_path):
        """v2 ADR with forbidden_imports detects import statement."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-003"
        _write_adr(adr_dir, adr_id, {
            "forbidden_imports": ["requests"],
        })
        src = tmp_path / "src"
        src.mkdir()
        (src / "client.py").write_text("import requests\n\ndef get():\n    pass\n", encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/client.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "fail"
        assert len(result["findings"]) >= 1
        f = result["findings"][0]
        assert f["severity"] == "blocker"
        assert f["adr_id"] == adr_id
        assert f["signal_type"] == "forbidden_imports"
        assert "requests" in f["why"]

    def test_v2_forbidden_imports_from_form(self, tmp_path):
        """v2 ADR with forbidden_imports detects `from X import` form."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-003b"
        _write_adr(adr_dir, adr_id, {
            "forbidden_imports": ["pickle"],
        })
        src = tmp_path / "src"
        src.mkdir()
        (src / "util.py").write_text("from pickle import dumps\n", encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/util.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "fail"
        assert any(f["signal_type"] == "forbidden_imports" for f in result["findings"])

    def test_v2_forbidden_patterns_hit(self, tmp_path):
        """v2 ADR with forbidden_patterns regex hit → finding."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-004"
        _write_adr(adr_dir, adr_id, {
            "forbidden_patterns": [r"eval\s*\("],
        })
        src = tmp_path / "src"
        src.mkdir()
        (src / "runner.py").write_text("result = eval(code)\n", encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/runner.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "fail"
        f = result["findings"][0]
        assert f["signal_type"] == "forbidden_patterns"
        assert f["severity"] == "blocker"

    def test_v2_required_test_coverage_no_test_file(self, tmp_path):
        """v2 ADR with required_test_coverage=True but no test file → finding."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-005"
        _write_adr(adr_dir, adr_id, {
            "required_test_coverage": True,
        })
        src = tmp_path / "src"
        src.mkdir()
        (src / "logic.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/logic.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "fail"
        f = result["findings"][0]
        assert f["signal_type"] == "required_test_coverage"
        assert f["severity"] == "blocker"

    def test_v2_required_test_coverage_with_test_file(self, tmp_path):
        """v2 ADR with required_test_coverage=True + test file → pass."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-006"
        _write_adr(adr_dir, adr_id, {
            "required_test_coverage": True,
        })
        src = tmp_path / "src"
        src.mkdir()
        (src / "logic.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_logic.py").write_text("def test_add(): assert 1 + 1 == 2\n", encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/logic.py", "tests/test_logic.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "pass"
        assert result["findings"] == []

    def test_v2_all_signals_no_violations(self, tmp_path):
        """v2 ADR with all signals but no violations → verdict=pass."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-007"
        _write_adr(adr_dir, adr_id, {
            "forbidden_imports": ["requests"],
            "forbidden_patterns": [r"eval\s*\("],
            "required_test_coverage": True,
        })
        src = tmp_path / "src"
        src.mkdir()
        (src / "safe.py").write_text("import httpx\n\ndef fetch(): pass\n", encoding="utf-8")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_safe.py").write_text("def test_fetch(): pass\n", encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/safe.py", "tests/test_safe.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "pass"
        assert result["findings"] == []
        assert result["skipped"] is False

    def test_mixed_v2_and_null_adrs(self, tmp_path):
        """Mix of v2 (violation) and null ADR → fail from v2; null in judgment_only."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id_v2 = "ADR-2026-05-11-010"
        adr_id_null = "ADR-2026-05-11-011"
        _write_adr(adr_dir, adr_id_v2, {"forbidden_imports": ["bad_lib"]})
        _write_adr(adr_dir, adr_id_null, None)
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("import bad_lib\n", encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/app.py"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "fail"
        assert any(f["adr_id"] == adr_id_v2 for f in result["findings"])
        assert adr_id_null in result["judgment_only_adr_ids"]

    def test_unreadable_produced_file_does_not_crash(self, tmp_path):
        """Produced file that doesn't exist → skipped gracefully; no crash."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-012"
        _write_adr(adr_dir, adr_id, {"forbidden_imports": ["requests"]})

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["nonexistent/file.py"],
            workspace_path=str(tmp_path),
        ))
        # No findings since file doesn't exist to scan
        assert result["verdict"] == "pass"

    def test_ts_forbidden_imports(self, tmp_path):
        """v2 ADR with forbidden_imports detects TS import statement."""
        from mr_roboto.check_adr_drift import check_adr_drift
        adr_dir = tmp_path / ".adr"
        adr_id = "ADR-2026-05-11-013"
        _write_adr(adr_dir, adr_id, {"forbidden_imports": ["lodash"]})
        src = tmp_path / "src"
        src.mkdir()
        (src / "util.ts").write_text('import { get } from "lodash";\n', encoding="utf-8")

        result = asyncio.run(check_adr_drift(
            adr_register_path=str(adr_dir / "register.md"),
            produced_files=["src/util.ts"],
            workspace_path=str(tmp_path),
        ))
        assert result["verdict"] == "fail"
        assert any(f["signal_type"] == "forbidden_imports" for f in result["findings"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_contains_adr_drift_check(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        assert "adr_drift_check" in POST_HOOK_REGISTRY

    def test_registry_row_shape(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["adr_drift_check"]
        assert spec.kind == "adr_drift_check"
        assert spec.verb == "check_adr_drift"
        assert spec.default_severity == "blocker"
        assert spec.cost_band == "cheap"

    def test_registry_in_post_hook_kinds(self):
        from general_beckman.posthooks import POST_HOOK_KINDS
        assert "adr_drift_check" in POST_HOOK_KINDS

    def test_auto_wire_callable_qa_dial_standard(self):
        """Callable trigger returns globs when qa_dial=standard."""
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["adr_drift_check"]
        assert callable(spec.auto_wire_triggers)
        globs = spec.auto_wire_triggers({"qa_dial": "standard"})
        assert isinstance(globs, list)
        assert len(globs) > 0
        assert "**/*.py" in globs

    def test_auto_wire_callable_qa_dial_off(self):
        """Callable trigger returns [] when qa_dial=off."""
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["adr_drift_check"]
        assert callable(spec.auto_wire_triggers)
        globs = spec.auto_wire_triggers({"qa_dial": "off"})
        assert globs == []

    def test_auto_wire_callable_no_dial(self):
        """Callable trigger returns globs when qa_dial not set (default=standard)."""
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["adr_drift_check"]
        globs = spec.auto_wire_triggers({})
        assert isinstance(globs, list)
        assert len(globs) > 0

    def test_dial_get_helper(self):
        """_dial_get reads values correctly with defaults."""
        from general_beckman.posthooks import _dial_get
        assert _dial_get({"qa_dial": "strict"}, "qa_dial", "standard") == "strict"
        assert _dial_get({}, "qa_dial", "standard") == "standard"
        assert _dial_get({"qa_dial": None}, "qa_dial", "standard") == "standard"
        assert _dial_get(None, "qa_dial", "standard") == "standard"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# apply.py dispatch
# ---------------------------------------------------------------------------

class TestApplyDispatch:
    def test_posthook_agent_and_payload_returns_mechanical(self):
        """_posthook_agent_and_payload for adr_drift_check returns mechanical executor."""
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import RequestPostHook

        source_ctx = {
            "workspace_path": "/workspace/mission_7",
            "produces": ["src/foo.py", "tests/test_foo.py"],
        }
        a = RequestPostHook(
            source_task_id=42,
            kind="adr_drift_check",
            source_ctx=source_ctx,
        )
        source = {"id": 42, "mission_id": 7}
        agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)
        assert agent_type == "mechanical"
        assert payload["executor"] == "mechanical"
        assert payload["posthook_kind"] == "adr_drift_check"
        assert payload["payload"]["action"] == "check_adr_drift"
        assert "adr_register_path" in payload["payload"]
        assert payload["payload"]["produced_files"] == ["src/foo.py", "tests/test_foo.py"]
        # adr_register_path derived from workspace_path
        assert ".adr/register.md" in payload["payload"]["adr_register_path"]

    def test_posthook_agent_and_payload_custom_register_path(self):
        """Custom adr_register_path from source_ctx is forwarded."""
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import RequestPostHook

        source_ctx = {
            "workspace_path": "/ws",
            "adr_register_path": "/ws/custom/.adr/register.md",
            "produces": [],
        }
        a = RequestPostHook(source_task_id=1, kind="adr_drift_check", source_ctx=source_ctx)
        source = {"id": 1, "mission_id": 3}
        _, payload = _posthook_agent_and_payload(a, source, source_ctx)
        assert payload["payload"]["adr_register_path"] == "/ws/custom/.adr/register.md"


# ---------------------------------------------------------------------------
# mr_roboto dispatch (integration stub)
# ---------------------------------------------------------------------------

class TestMrRobotoDispatch:
    def test_check_adr_drift_registered_in_init(self):
        """check_adr_drift is importable from mr_roboto."""
        from mr_roboto import check_adr_drift
        assert callable(check_adr_drift)

    def test_run_dispatch_soft_skip(self):
        """mr_roboto.run with check_adr_drift on empty workspace → completed (skipped)."""
        import asyncio
        from mr_roboto import run as mr_run

        task = {
            "id": 99,
            "mission_id": 1,
            "payload": {
                "action": "check_adr_drift",
                "adr_register_path": "/nonexistent/.adr/register.md",
                "produced_files": [],
                "workspace_path": None,
            },
        }

        # Patch the audit log to avoid DB dependency
        with patch(
            "mr_roboto._log_action_event",
            new=AsyncMock(),
        ):
            action = asyncio.run(mr_run(task))

        assert action.status == "completed"
        assert action.result is not None
        assert action.result.get("skipped") is True
