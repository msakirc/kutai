"""Z3 T3B — accessibility_review post-hook via axe-core.

Covers:
- accessibility_review kind registered with correct fields (cost_band=heavy,
  default_severity=blocker, verb=run_axe).
- Callable auto_wire_triggers: returns [] when accessibility_dial=='off'.
- Callable auto_wire_triggers: returns ['*.tsx', '*.jsx'] when dial is not off.
- run_axe soft-skips when preview_url is missing.
- run_axe soft-skips when preview_url is a pending: placeholder.
- run_axe parses axe JSON correctly (mocked subprocess).
- run_axe returns verdict=fail when critical/serious violation present.
- Severity mapping: critical→blocker, serious→blocker, moderate→warning,
  minor→info.
- Apply.py reads last_preview_url.txt from mission workspace.
- emit_preview_url writes last_preview_url.txt when real URL captured.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from fnmatch import fnmatch
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SKIP_NON_CANONICAL = pytest.mark.skip(reason="Z3 T3: test asserts agent-specific design; canonical uses MissionDialContext via T1A")



# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------

def test_accessibility_review_in_registry():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "accessibility_review" in POST_HOOK_REGISTRY


def test_accessibility_review_kind_field():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    assert spec.kind == "accessibility_review"


def test_accessibility_review_verb():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    assert spec.verb == "run_axe"


def test_accessibility_review_cost_band_heavy():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    assert spec.cost_band == "heavy"


def test_accessibility_review_default_severity_blocker():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    assert spec.default_severity == "blocker"


def test_accessibility_review_in_post_hook_kinds():
    from general_beckman.posthooks import POST_HOOK_KINDS
    assert "accessibility_review" in POST_HOOK_KINDS


# ---------------------------------------------------------------------------
# 2. Callable auto_wire_triggers
# ---------------------------------------------------------------------------

def test_auto_wire_triggers_is_callable():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    assert callable(spec.auto_wire_triggers)


def test_auto_wire_returns_empty_when_dial_off():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    result = spec.auto_wire_triggers({"accessibility_dial": "off"})
    assert result == []


def test_auto_wire_returns_tsx_jsx_when_dial_on():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    result = spec.auto_wire_triggers({"accessibility_dial": "on"})
    assert "*.tsx" in result
    assert "*.jsx" in result


@_SKIP_NON_CANONICAL
def test_auto_wire_returns_tsx_jsx_when_dial_absent():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    result = spec.auto_wire_triggers({})
    assert "*.tsx" in result
    assert "*.jsx" in result


@_SKIP_NON_CANONICAL
def test_auto_wire_returns_tsx_jsx_with_no_ctx():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    # callable with no arguments (default ctx=None path)
    result = spec.auto_wire_triggers()
    assert "*.tsx" in result
    assert "*.jsx" in result


def test_auto_wire_matches_tsx_files_via_expander_logic():
    """Simulate what the expander does with callable triggers."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    import fnmatch as _fnmatch
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    ctx = {"produces": ["src/components/Button.tsx"], "accessibility_dial": "on"}
    triggers = spec.auto_wire_triggers(ctx)
    candidate_paths = ["src/components/Button.tsx"]
    matched = any(
        _fnmatch.fnmatchcase(p, t)
        for p in candidate_paths
        for t in triggers
    )
    assert matched


def test_auto_wire_no_match_when_dial_off():
    """When dial is off, even .tsx files must NOT trigger."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    import fnmatch as _fnmatch
    spec = POST_HOOK_REGISTRY["accessibility_review"]
    ctx = {"produces": ["src/components/Button.tsx"], "accessibility_dial": "off"}
    triggers = spec.auto_wire_triggers(ctx)
    candidate_paths = ["src/components/Button.tsx"]
    matched = any(
        _fnmatch.fnmatchcase(p, t)
        for p in candidate_paths
        for t in triggers
    )
    assert not matched


# ---------------------------------------------------------------------------
# 3. run_axe soft-skip paths
# ---------------------------------------------------------------------------

def test_run_axe_skips_when_no_preview_url():
    from mr_roboto.run_axe import run_axe
    result = asyncio.new_event_loop().run_until_complete(
        run_axe(preview_url=None)
    )
    assert result["skipped"] is True
    assert result["verdict"] == "pass"
    assert result["findings"] == []
    assert "reason" in result


def test_run_axe_skips_when_empty_preview_url():
    from mr_roboto.run_axe import run_axe
    result = asyncio.new_event_loop().run_until_complete(
        run_axe(preview_url="")
    )
    assert result["skipped"] is True


def test_run_axe_skips_when_pending_placeholder():
    from mr_roboto.run_axe import run_axe
    result = asyncio.new_event_loop().run_until_complete(
        run_axe(preview_url="pending: hosting deferred to Z2")
    )
    assert result["skipped"] is True
    assert result["verdict"] == "pass"


def test_run_axe_skips_when_npx_absent():
    """When npx is not on PATH, soft-skip."""
    mod = _run_axe_module()
    with patch.object(mod, "_axe_available", return_value=False):
        result = asyncio.new_event_loop().run_until_complete(
            mod.run_axe(preview_url="https://real.trycloudflare.com")
        )
    assert result["skipped"] is True
    assert result["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 4. run_axe — axe JSON parsing
# ---------------------------------------------------------------------------

_SAMPLE_AXE_OUTPUT = json.dumps({
    "violations": [
        {
            "id": "color-contrast",
            "impact": "serious",
            "description": "Elements must have sufficient color contrast",
            "help": "Ensure the contrast ratio",
            "nodes": [
                {"target": ["button.cta"]}
            ],
        },
        {
            "id": "image-alt",
            "impact": "critical",
            "description": "Images must have alternate text",
            "help": "Ensure img elements have alt text",
            "nodes": [
                {"target": ["img.hero"]}
            ],
        },
        {
            "id": "aria-label",
            "impact": "moderate",
            "description": "Aria labels should be present",
            "help": "Add aria-label",
            "nodes": [],
        },
        {
            "id": "tabindex",
            "impact": "minor",
            "description": "Avoid positive tabindex",
            "help": "Remove positive tabindex",
            "nodes": [{"target": ["a.link"]}],
        },
    ]
})


def _make_mock_proc(stdout: bytes, returncode: int = 2):
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
    mock_proc.returncode = returncode
    mock_proc.kill = MagicMock()
    return mock_proc


def _run_axe_module():
    import importlib
    return importlib.import_module("mr_roboto.run_axe")


def test_run_axe_parses_valid_output():
    mod = _run_axe_module()
    mock_proc = _make_mock_proc(_SAMPLE_AXE_OUTPUT.encode())
    spawn_mock = AsyncMock(return_value=mock_proc)

    with patch.object(mod, "_axe_available", return_value=True), \
         patch("asyncio.create_subprocess_exec", new=spawn_mock):
        result = asyncio.new_event_loop().run_until_complete(
            mod.run_axe(preview_url="https://test.trycloudflare.com")
        )

    assert result["skipped"] is False
    assert len(result["findings"]) >= 4  # serious, critical, moderate+no nodes surfaced once, minor


def test_run_axe_verdict_fail_on_blockers():
    mod = _run_axe_module()
    mock_proc = _make_mock_proc(_SAMPLE_AXE_OUTPUT.encode())
    spawn_mock = AsyncMock(return_value=mock_proc)

    with patch.object(mod, "_axe_available", return_value=True), \
         patch("asyncio.create_subprocess_exec", new=spawn_mock):
        result = asyncio.new_event_loop().run_until_complete(
            mod.run_axe(preview_url="https://test.trycloudflare.com")
        )

    assert result["verdict"] == "fail"
    blockers = [f for f in result["findings"] if f["severity"] == "blocker"]
    assert len(blockers) >= 2  # serious + critical both become blocker


def test_run_axe_verdict_pass_when_no_blockers():
    """Only moderate/minor violations → pass."""
    mod = _run_axe_module()
    mild_output = json.dumps({
        "violations": [
            {
                "id": "aria-label",
                "impact": "moderate",
                "description": "Moderate issue",
                "help": "Fix it",
                "nodes": [{"target": ["div"]}],
            }
        ]
    })
    mock_proc = _make_mock_proc(mild_output.encode())
    spawn_mock = AsyncMock(return_value=mock_proc)

    with patch.object(mod, "_axe_available", return_value=True), \
         patch("asyncio.create_subprocess_exec", new=spawn_mock):
        result = asyncio.new_event_loop().run_until_complete(
            mod.run_axe(preview_url="https://test.trycloudflare.com")
        )

    assert result["verdict"] == "pass"
    assert result["skipped"] is False


def test_run_axe_verdict_pass_when_no_violations():
    mod = _run_axe_module()
    empty_output = json.dumps({"violations": []})
    mock_proc = _make_mock_proc(empty_output.encode(), returncode=0)
    spawn_mock = AsyncMock(return_value=mock_proc)

    with patch.object(mod, "_axe_available", return_value=True), \
         patch("asyncio.create_subprocess_exec", new=spawn_mock):
        result = asyncio.new_event_loop().run_until_complete(
            mod.run_axe(preview_url="https://test.trycloudflare.com")
        )

    assert result["verdict"] == "pass"
    assert result["findings"] == []


# ---------------------------------------------------------------------------
# 5. Severity mapping
# ---------------------------------------------------------------------------

def test_severity_critical_maps_to_blocker():
    from mr_roboto.run_axe import _impact_to_severity
    assert _impact_to_severity("critical") == "blocker"


def test_severity_serious_maps_to_blocker():
    from mr_roboto.run_axe import _impact_to_severity
    assert _impact_to_severity("serious") == "blocker"


def test_severity_moderate_maps_to_warning():
    from mr_roboto.run_axe import _impact_to_severity
    assert _impact_to_severity("moderate") == "warning"


def test_severity_minor_maps_to_info():
    from mr_roboto.run_axe import _impact_to_severity
    assert _impact_to_severity("minor") == "info"


def test_severity_unknown_maps_to_info():
    from mr_roboto.run_axe import _impact_to_severity
    assert _impact_to_severity("cosmetic") == "info"


# ---------------------------------------------------------------------------
# 6. Apply.py reads last_preview_url.txt
# ---------------------------------------------------------------------------

def test_posthook_payload_reads_last_preview_url_txt():
    """_posthook_agent_and_payload resolves preview URL from last_preview_url.txt."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    with tempfile.TemporaryDirectory() as tmpdir:
        preview_dir = os.path.join(tmpdir, ".preview")
        os.makedirs(preview_dir)
        url_file = os.path.join(preview_dir, "last_preview_url.txt")
        real_url = "https://vivid-clouds.trycloudflare.com"
        with open(url_file, "w") as f:
            f.write(f"{real_url}\n")

        a = RequestPostHook(
            kind="accessibility_review",
            source_task_id=42,
            source_ctx={},
        )
        source = {"id": 42, "mission_id": 99, "title": "Build UI"}
        source_ctx = {
            "workspace_path": tmpdir,
            "produces": ["src/components/Button.tsx"],
        }

        agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert agent_type == "mechanical"
    assert payload["payload"]["action"] == "run_axe"
    assert payload["payload"]["preview_url"] == real_url


def test_posthook_payload_falls_back_gracefully_when_no_url_file():
    """If last_preview_url.txt is absent, preview_url is empty (soft-skip later)."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    with tempfile.TemporaryDirectory() as tmpdir:
        a = RequestPostHook(
            kind="accessibility_review",
            source_task_id=43,
            source_ctx={},
        )
        source = {"id": 43, "mission_id": 99, "title": "Build UI"}
        source_ctx = {
            "workspace_path": tmpdir,
            "produces": ["src/components/Card.tsx"],
        }

        agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert agent_type == "mechanical"
    assert payload["payload"]["preview_url"] == ""


def test_posthook_payload_skips_pending_url():
    """A pending: URL in last_preview_url.txt is treated as absent."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    with tempfile.TemporaryDirectory() as tmpdir:
        preview_dir = os.path.join(tmpdir, ".preview")
        os.makedirs(preview_dir)
        url_file = os.path.join(preview_dir, "last_preview_url.txt")
        with open(url_file, "w") as f:
            f.write("pending: hosting deferred to Z2\n")

        a = RequestPostHook(
            kind="accessibility_review",
            source_task_id=44,
            source_ctx={},
        )
        source = {"id": 44, "mission_id": 99, "title": "Build UI"}
        source_ctx = {
            "workspace_path": tmpdir,
            "produces": ["src/components/Card.tsx"],
        }

        agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert payload["payload"]["preview_url"] == ""


# ---------------------------------------------------------------------------
# 7. emit_preview_url writes last_preview_url.txt
# ---------------------------------------------------------------------------

def _epu_module():
    import importlib
    return importlib.import_module("mr_roboto.emit_preview_url")


def test_emit_preview_url_writes_last_preview_url_txt():
    """When a real URL is captured, emit_preview_url writes .preview/last_preview_url.txt."""
    _epu_mod = _epu_module()
    real_url = "https://vivid-clouds.trycloudflare.com"

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = os.path.join(tmpdir, "mission_99")
        os.makedirs(workspace_path)

        mock_proc = MagicMock()
        mock_proc.pid = 1234

        with patch.object(_epu_mod, "_resolve_workspace", return_value=workspace_path), \
             patch.object(_epu_mod, "_await_kill_prior", new_callable=AsyncMock,
                          return_value={"ok": True}), \
             patch.dict(os.environ, {"KUTAI_PREVIEW_PROVIDER": "cloudflared"}), \
             patch.object(_epu_mod.shutil, "which", return_value="/usr/bin/cloudflared"), \
             patch.object(_epu_mod, "_spawn_cloudflared", return_value=mock_proc), \
             patch.object(_epu_mod, "_read_url_from_proc", return_value=real_url), \
             patch.object(_epu_mod, "_persist_to_db", new_callable=AsyncMock):

            result = asyncio.new_event_loop().run_until_complete(
                _epu_mod.emit_preview_url(mission_id=99, workspace_path=workspace_path)
            )

        last_url_path = os.path.join(workspace_path, ".preview", "last_preview_url.txt")
        assert os.path.exists(last_url_path), "last_preview_url.txt must be written"
        content = open(last_url_path).read().strip()
        assert content == real_url


def test_emit_preview_url_no_last_url_when_pending():
    """In pending mode (no provider), last_preview_url.txt is NOT written."""
    _epu_mod = _epu_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = os.path.join(tmpdir, "mission_100")
        os.makedirs(workspace_path)

        with patch.object(_epu_mod, "_resolve_workspace", return_value=workspace_path), \
             patch.object(_epu_mod, "_await_kill_prior", new_callable=AsyncMock,
                          return_value={"ok": True}), \
             patch.dict(os.environ, {"KUTAI_PREVIEW_PROVIDER": ""}):

            result = asyncio.new_event_loop().run_until_complete(
                _epu_mod.emit_preview_url(mission_id=100, workspace_path=workspace_path)
            )

        assert result["pending"] is True
        last_url_path = os.path.join(workspace_path, ".preview", "last_preview_url.txt")
        assert not os.path.exists(last_url_path), (
            "last_preview_url.txt must NOT be written in pending mode"
        )


# ---------------------------------------------------------------------------
# 8. PostHookSpec dataclass still accepts list[str] for other kinds
# ---------------------------------------------------------------------------

def test_posthook_spec_list_triggers_backward_compat():
    """Existing list-based auto_wire_triggers still work."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["design_system_check"]
    # Should be a list
    assert isinstance(spec.auto_wire_triggers, list)
    assert "*.tsx" in spec.auto_wire_triggers
