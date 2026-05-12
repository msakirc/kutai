"""Z3 T4C — layer-aware tooling tests.

Covers:
- inspect_layer heuristic on each layer's path conventions
- inspect_layer spec-override reads layer_map.json and wins over heuristic
- inspect_layer returns "unknown" when no match
- LAYER_BLOCKS dict has all 5 named keys (domain/adapter/infra/ui/test)
- build_reflection_prompt appends layer block when layer is resolvable
- forbidden_in_domain.yml parses as valid YAML and has the 5 expected rule IDs
- run_semgrep_layer_filtered filters files before invoking semgrep (mock subprocess)
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml


# ────────────────────────────────────────────────────────────────────────────
# inspect_layer — heuristic table
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected_layer", [
    # domain
    ("src/domain/user.py", "domain"),
    ("packages/core/domain/entities.py", "domain"),
    # adapter
    ("src/adapter/http_gateway.py", "adapter"),
    ("src/gateway/stripe_client.py", "adapter"),
    ("src/client/api_client.py", "adapter"),
    # infra
    ("src/infra/db.py", "infra"),
    ("src/storage/s3.py", "infra"),
    ("src/repo/user_repo.py", "infra"),
    ("src/repository/order_repo.py", "infra"),
    ("src/db/models.py", "infra"),
    # test
    ("tests/test_user.py", "test"),
    ("test_utils.py", "test"),
    ("src/user_test.go", "test"),
    ("src/user.test.ts", "test"),
    ("src/user.spec.ts", "test"),
    # ui
    ("src/components/Button.tsx", "ui"),
    ("src/pages/index.tsx", "ui"),
    ("src/views/Home.vue", "ui"),
    ("src/ui/sidebar.tsx", "ui"),
    # unknown
    ("main.py", "unknown"),
    ("utils/helpers.py", "unknown"),
    ("config.yaml", "unknown"),
])
def test_inspect_layer_heuristic(path, expected_layer):
    from src.tools.inspect_layer import inspect_layer

    result = asyncio.new_event_loop().run_until_complete(inspect_layer(path))
    assert result == expected_layer, (
        f"inspect_layer({path!r}) = {result!r}, expected {expected_layer!r}"
    )


def test_inspect_layer_unknown_path():
    from src.tools.inspect_layer import inspect_layer

    result = asyncio.new_event_loop().run_until_complete(inspect_layer("random_file.py"))
    assert result == "unknown"


def test_inspect_layer_empty_path():
    from src.tools.inspect_layer import inspect_layer

    result = asyncio.new_event_loop().run_until_complete(inspect_layer(""))
    assert result == "unknown"


# ────────────────────────────────────────────────────────────────────────────
# inspect_layer — spec-override via layer_map.json
# ────────────────────────────────────────────────────────────────────────────

def test_inspect_layer_spec_override_wins(tmp_path):
    """Spec-override glob takes precedence over heuristic."""
    from src.tools.inspect_layer import inspect_layer

    # Create a workspace with a layer_map.json that overrides
    spec_dir = tmp_path / ".spec"
    spec_dir.mkdir()
    layer_map = {
        "globs": {
            "services/**": "domain",      # override: services/ → domain (not adapter)
            "adapters/**": "adapter",
            "infrastructure/**": "infra",
        }
    }
    (spec_dir / "layer_map.json").write_text(json.dumps(layer_map), encoding="utf-8")

    # "services/user_service.py" would be "unknown" by heuristic but "domain" by spec
    result = asyncio.new_event_loop().run_until_complete(
        inspect_layer("services/user_service.py", workspace_path=str(tmp_path))
    )
    assert result == "domain", f"Expected 'domain' from spec override, got {result!r}"


def test_inspect_layer_spec_override_first_match_wins(tmp_path):
    """When multiple globs match, the first one wins."""
    from src.tools.inspect_layer import inspect_layer

    spec_dir = tmp_path / ".spec"
    spec_dir.mkdir()
    # Two patterns that could match "src/app/ui/button.py"
    layer_map = {
        "globs": {
            "src/app/**": "infra",    # first match
            "src/app/ui/**": "ui",    # second match — should NOT win
        }
    }
    (spec_dir / "layer_map.json").write_text(json.dumps(layer_map), encoding="utf-8")

    result = asyncio.new_event_loop().run_until_complete(
        inspect_layer("src/app/ui/button.py", workspace_path=str(tmp_path))
    )
    assert result == "infra"


def test_inspect_layer_spec_override_falls_back_to_heuristic_on_miss(tmp_path):
    """Files not matched by any spec glob fall back to heuristic."""
    from src.tools.inspect_layer import inspect_layer

    spec_dir = tmp_path / ".spec"
    spec_dir.mkdir()
    layer_map = {"globs": {"src/special/**": "domain"}}
    (spec_dir / "layer_map.json").write_text(json.dumps(layer_map), encoding="utf-8")

    # This path matches the heuristic (infra/) but not the spec glob
    result = asyncio.new_event_loop().run_until_complete(
        inspect_layer("src/infra/db.py", workspace_path=str(tmp_path))
    )
    assert result == "infra"


def test_inspect_layer_spec_override_invalid_json_degrades_gracefully(tmp_path):
    """Invalid layer_map.json falls back to heuristic without crashing."""
    from src.tools.inspect_layer import inspect_layer

    spec_dir = tmp_path / ".spec"
    spec_dir.mkdir()
    (spec_dir / "layer_map.json").write_text("not valid json!!!", encoding="utf-8")

    result = asyncio.new_event_loop().run_until_complete(
        inspect_layer("src/domain/user.py", workspace_path=str(tmp_path))
    )
    # Falls back to heuristic
    assert result == "domain"


def test_inspect_layer_no_workspace_path_uses_heuristic():
    """When workspace_path is None, spec-override is skipped; heuristic applies."""
    from src.tools.inspect_layer import inspect_layer

    result = asyncio.new_event_loop().run_until_complete(
        inspect_layer("src/domain/user.py", workspace_path=None)
    )
    assert result == "domain"


# ────────────────────────────────────────────────────────────────────────────
# LAYER_BLOCKS
# ────────────────────────────────────────────────────────────────────────────

EXPECTED_LAYER_KEYS = {"domain", "adapter", "infra", "ui", "test"}


def test_layer_blocks_has_all_expected_keys():
    from coulson.reflection import LAYER_BLOCKS

    for key in EXPECTED_LAYER_KEYS:
        assert key in LAYER_BLOCKS, f"LAYER_BLOCKS missing key: {key!r}"


def test_layer_blocks_named_layers_are_strings():
    from coulson.reflection import LAYER_BLOCKS

    for key, val in LAYER_BLOCKS.items():
        assert isinstance(val, str), f"LAYER_BLOCKS[{key!r}] is not a string"


def test_layer_blocks_content_bearing_layers_nonempty():
    """domain, adapter, infra, ui must have meaningful content."""
    from coulson.reflection import LAYER_BLOCKS

    for key in ("domain", "adapter", "infra", "ui"):
        assert len(LAYER_BLOCKS[key].strip()) > 30, (
            f"LAYER_BLOCKS[{key!r}] looks like a placeholder (<30 chars)"
        )


def test_layer_blocks_test_and_unknown_are_empty_or_missing():
    """test and unknown produce no guidance (empty string)."""
    from coulson.reflection import LAYER_BLOCKS

    assert LAYER_BLOCKS.get("test", "") == ""
    # unknown may or may not be present; if present must be empty
    if "unknown" in LAYER_BLOCKS:
        assert LAYER_BLOCKS["unknown"] == ""


# ────────────────────────────────────────────────────────────────────────────
# build_reflection_prompt — layer parameter
# ────────────────────────────────────────────────────────────────────────────

def test_build_reflection_prompt_appends_layer_block():
    from coulson.reflection import build_reflection_prompt, LAYER_BLOCKS

    result = build_reflection_prompt("coder", iteration=1, layer="domain")
    assert "domain" in result.lower()
    layer_fragment = LAYER_BLOCKS["domain"][:40]
    assert layer_fragment in result


def test_build_reflection_prompt_no_layer_no_crash():
    from coulson.reflection import build_reflection_prompt

    result = build_reflection_prompt("coder", iteration=1)
    assert "[iteration 1]" in result


def test_build_reflection_prompt_unknown_layer_no_block():
    from coulson.reflection import build_reflection_prompt

    result = build_reflection_prompt("coder", iteration=1, layer="unknown")
    assert "## Layer reminder" not in result


def test_build_reflection_prompt_test_layer_no_block():
    from coulson.reflection import build_reflection_prompt

    result = build_reflection_prompt("coder", iteration=1, layer="test")
    assert "## Layer reminder" not in result


def test_build_reflection_prompt_layer_and_stack_both_appended():
    from coulson.reflection import build_reflection_prompt, LAYER_BLOCKS, STACK_BLOCKS

    result = build_reflection_prompt("coder", iteration=2, stack="fastapi", layer="domain")
    # Both blocks present
    assert STACK_BLOCKS["fastapi"][:30] in result
    assert LAYER_BLOCKS["domain"][:30] in result


def test_build_reflection_prompt_layer_no_crash_unknown_agent():
    from coulson.reflection import build_reflection_prompt

    result = build_reflection_prompt("some_future_agent", iteration=1, layer="adapter")
    assert "adapter" in result.lower()


def test_build_reflection_prompt_layer_infra():
    from coulson.reflection import build_reflection_prompt, LAYER_BLOCKS

    result = build_reflection_prompt("implementer", iteration=3, layer="infra")
    assert LAYER_BLOCKS["infra"][:30] in result


def test_build_reflection_prompt_layer_ui():
    from coulson.reflection import build_reflection_prompt, LAYER_BLOCKS

    result = build_reflection_prompt("coder", iteration=1, layer="ui")
    assert LAYER_BLOCKS["ui"][:30] in result


# ────────────────────────────────────────────────────────────────────────────
# forbidden_in_domain.yml — YAML validity + rule IDs
# ────────────────────────────────────────────────────────────────────────────

def _load_forbidden_in_domain_yml():
    pack_path = (
        Path(__file__).parent.parent
        / "packages" / "mr_roboto" / "src" / "mr_roboto"
        / "rule_packs" / "forbidden_in_domain.yml"
    )
    with open(pack_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_forbidden_in_domain_yml_parses():
    data = _load_forbidden_in_domain_yml()
    assert isinstance(data, dict), "YAML must parse to a dict"
    assert "rules" in data, "YAML must have a top-level 'rules' key"
    assert isinstance(data["rules"], list)


EXPECTED_RULE_IDS = {
    "fastapi-import-in-domain",
    "flask-import-in-domain",
    "django-import-in-domain",
    "sqlalchemy-import-in-domain",
    "requests-import-in-domain",
}


def test_forbidden_in_domain_yml_has_all_rule_ids():
    data = _load_forbidden_in_domain_yml()
    found_ids = {r["id"] for r in data["rules"]}
    for rule_id in EXPECTED_RULE_IDS:
        assert rule_id in found_ids, f"Missing rule ID: {rule_id!r}"


def test_forbidden_in_domain_yml_all_rules_error_severity():
    data = _load_forbidden_in_domain_yml()
    for rule in data["rules"]:
        assert rule.get("severity") == "ERROR", (
            f"Rule {rule['id']!r} has severity {rule.get('severity')!r}, expected 'ERROR'"
        )


def test_forbidden_in_domain_yml_all_rules_python_language():
    data = _load_forbidden_in_domain_yml()
    for rule in data["rules"]:
        langs = rule.get("languages", [])
        assert "python" in langs, (
            f"Rule {rule['id']!r} missing 'python' in languages: {langs}"
        )


# ────────────────────────────────────────────────────────────────────────────
# run_semgrep_layer_filtered — filters files before invoking semgrep
# ────────────────────────────────────────────────────────────────────────────

def _make_fake_run_semgrep_calls():
    """Return a list that records calls to the mock run_semgrep."""
    calls = []

    async def mock_run_semgrep(
        mission_id,
        target_files=None,
        rule_pack_path=None,
        workspace_path=None,
        timeout_s=120.0,
    ):
        calls.append({
            "mission_id": mission_id,
            "target_files": list(target_files or []),
            "rule_pack_path": rule_pack_path,
        })
        return {
            "ok": True,
            "skipped": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.1,
            "error": None,
        }

    return calls, mock_run_semgrep


def test_run_semgrep_layer_filtered_only_domain_files_scanned(monkeypatch, tmp_path):
    """Files not in the domain layer are excluded before semgrep is invoked."""
    calls, mock_semgrep = _make_fake_run_semgrep_calls()

    import mr_roboto.run_semgrep_layer_filtered as _mod
    monkeypatch.setattr(_mod, "run_semgrep", mock_semgrep, raising=False)

    # Also patch the import inside the function
    import sys
    import types
    fake_run_semgrep_mod = types.ModuleType("mr_roboto.run_semgrep")
    fake_run_semgrep_mod.run_semgrep = mock_semgrep
    monkeypatch.setitem(sys.modules, "mr_roboto.run_semgrep", fake_run_semgrep_mod)

    target_files = [
        "src/domain/user.py",          # domain — should be included
        "src/domain/order.py",         # domain — should be included
        "src/infra/db.py",             # infra — should be excluded
        "src/adapter/http_gw.py",      # adapter — should be excluded
        "tests/test_domain.py",        # test — should be excluded
    ]

    rule_pack = str(
        Path(__file__).parent.parent
        / "packages" / "mr_roboto" / "src" / "mr_roboto"
        / "rule_packs" / "forbidden_in_domain.yml"
    )

    async def run():
        return await _mod.run_semgrep_layer_filtered(
            mission_id=1,
            target_files=target_files,
            rule_pack_path=rule_pack,
            required_layer="domain",
        )

    result = asyncio.new_event_loop().run_until_complete(run())
    assert result["ok"] is True

    # run_semgrep must have been called exactly once
    assert len(calls) == 1, f"Expected 1 semgrep call, got {len(calls)}"
    scanned = calls[0]["target_files"]
    assert "src/domain/user.py" in scanned
    assert "src/domain/order.py" in scanned
    assert "src/infra/db.py" not in scanned
    assert "src/adapter/http_gw.py" not in scanned
    assert "tests/test_domain.py" not in scanned


def test_run_semgrep_layer_filtered_no_matching_files_skips(monkeypatch):
    """When no files match the required layer, semgrep is NOT invoked."""
    calls, mock_semgrep = _make_fake_run_semgrep_calls()

    import mr_roboto.run_semgrep_layer_filtered as _mod
    import sys, types
    fake_mod = types.ModuleType("mr_roboto.run_semgrep")
    fake_mod.run_semgrep = mock_semgrep
    monkeypatch.setitem(sys.modules, "mr_roboto.run_semgrep", fake_mod)

    target_files = [
        "src/infra/db.py",
        "src/adapter/gw.py",
    ]

    async def run():
        return await _mod.run_semgrep_layer_filtered(
            mission_id=1,
            target_files=target_files,
            rule_pack_path="forbidden_in_domain.yml",
            required_layer="domain",
        )

    result = asyncio.new_event_loop().run_until_complete(run())
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["findings"] == []
    # semgrep must NOT have been called
    assert len(calls) == 0, "Expected no semgrep call when no files match"


def test_run_semgrep_layer_filtered_spec_override_respected(monkeypatch, tmp_path):
    """Spec-override layer_map.json influences which files are scanned."""
    calls, mock_semgrep = _make_fake_run_semgrep_calls()

    import mr_roboto.run_semgrep_layer_filtered as _mod
    import sys, types
    fake_mod = types.ModuleType("mr_roboto.run_semgrep")
    fake_mod.run_semgrep = mock_semgrep
    monkeypatch.setitem(sys.modules, "mr_roboto.run_semgrep", fake_mod)

    # Override: services/** → domain
    spec_dir = tmp_path / ".spec"
    spec_dir.mkdir()
    (spec_dir / "layer_map.json").write_text(
        json.dumps({"globs": {"services/**": "domain"}}), encoding="utf-8"
    )

    target_files = [
        "services/user_service.py",   # → domain via spec override
        "src/infra/db.py",            # → infra by heuristic
    ]

    rule_pack = str(
        Path(__file__).parent.parent
        / "packages" / "mr_roboto" / "src" / "mr_roboto"
        / "rule_packs" / "forbidden_in_domain.yml"
    )

    async def run():
        return await _mod.run_semgrep_layer_filtered(
            mission_id=1,
            target_files=target_files,
            rule_pack_path=rule_pack,
            required_layer="domain",
            workspace_path=str(tmp_path),
        )

    result = asyncio.new_event_loop().run_until_complete(run())
    assert result["ok"] is True
    assert len(calls) == 1
    scanned = calls[0]["target_files"]
    assert "services/user_service.py" in scanned
    assert "src/infra/db.py" not in scanned


# ────────────────────────────────────────────────────────────────────────────
# i2p_v3.json — layer_map artifact wired in phase_4
# ────────────────────────────────────────────────────────────────────────────

def test_i2p_v3_phase4_step_produces_layer_map():
    """At least one phase_4 step produces the layer_map.json artifact."""
    wf_path = (
        Path(__file__).parent.parent
        / "src" / "workflows" / "i2p" / "i2p_v3.json"
    )
    with open(wf_path, encoding="utf-8") as fh:
        data = json.load(fh)

    phase4_steps = [s for s in data["steps"] if s.get("phase") == "phase_4"]
    layer_map_producers = [
        s for s in phase4_steps
        if any(
            "layer_map.json" in (p or "")
            for p in (s.get("produces") or [])
        )
    ]
    assert layer_map_producers, (
        "No phase_4 step produces layer_map.json. "
        "Expected at least one step with 'mission_{mission_id}/.spec/layer_map.json' in produces."
    )
