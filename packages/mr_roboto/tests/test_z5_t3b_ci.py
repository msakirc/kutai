"""Z5 T3b — free-first GitHub Actions mobile CI + Fastlane adapter tests.

Covers the two T3b verbs (``gen_mobile_ci``, ``fastlane``) and the
``mobile_ci`` recipe:

  * ``gen_mobile_ci`` writes a workflow file that is valid YAML, with the
    expected iOS (macOS runner) + Android (Linux runner) jobs.
  * ``fastlane`` dispatches via ``mr_roboto.run`` and soft-skips when the
    CLI is missing — never a hard failure.
  * ``fastlane`` lane reversibility: ``pilot``/``supply`` →
    ``irreversible`` (binary lands on a store track), ``build``/``match``
    → ``full`` (local + re-runnable).
  * the ``mobile_ci`` recipe loads, matches the ``expo`` stack, and
    instantiates with no unresolved ``<<...>>`` tokens.

``run_cmd`` is monkeypatched at the *module* level of the adapter (the
adapter does ``from mr_roboto.run_cmd import run_cmd`` at import time, so
the bound name lives on the adapter module). The monkeypatch-``run_cmd``
fixture style mirrors ``test_z5_mobile_adapters.py``.
"""
from __future__ import annotations

import importlib

import pytest

import mr_roboto
from mr_roboto.reversibility import get_reversibility

from src.infra.recipes import (
    instantiate_recipe,
    list_recipes,
    load_recipe,
    match_recipe,
)

# Fetch the actual submodule object so monkeypatching the module-level
# `run_cmd` name works (mr_roboto/__init__.py rebinds attributes).
fastlane_mod = importlib.import_module("mr_roboto.fastlane_run")
gen_mobile_ci_mod = importlib.import_module("mr_roboto.gen_mobile_ci")


# --------------------------------------------------------------------------
# Fake run_cmd factories
# --------------------------------------------------------------------------

def _ok_run_cmd(stdout: str = "", stderr: str = "", exit_code: int = 0):
    """Return an async fake run_cmd simulating a successful subprocess."""
    async def _fake(*args, **kwargs):
        return {
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "duration_s": 1.5,
            "timed_out": False,
            "ok": exit_code == 0,
        }
    return _fake


def _missing_exe_run_cmd():
    """Return an async fake run_cmd simulating a missing executable."""
    async def _fake(*args, **kwargs):
        return {
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "timed_out": False,
            "ok": False,
            "error": (
                "executable not found: [Errno 2] "
                "No such file or directory: 'fastlane'"
            ),
        }
    return _fake


def _timeout_run_cmd():
    async def _fake(*args, **kwargs):
        return {
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 99.0,
            "timed_out": True,
            "ok": False,
        }
    return _fake


# ==========================================================================
# gen_mobile_ci
# ==========================================================================

@pytest.mark.asyncio
async def test_gen_mobile_ci_produces_valid_yaml(tmp_path):
    res = await gen_mobile_ci_mod.gen_mobile_ci(
        mission_id=None,
        workspace_path=str(tmp_path),
        platforms=["ios", "android"],
        bundle_id="com.acme.demo",
    )
    assert res["ok"] is True
    assert res["skipped"] is False
    assert res["error"] is None
    assert sorted(res["jobs"]) == ["android", "ios"]

    workflow_path = res["workflow_path"]
    assert workflow_path.endswith("mobile.yml")
    assert ".github" in workflow_path

    import yaml
    with open(workflow_path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    assert loaded["name"] == "Mobile CI"
    assert "jobs" in loaded
    assert set(loaded["jobs"].keys()) == {"ios", "android"}
    # iOS job runs on a macOS runner; Android on Linux.
    assert loaded["jobs"]["ios"]["runs-on"] == "macos-latest"
    assert loaded["jobs"]["android"]["runs-on"] == "ubuntu-latest"
    # `on` is a YAML 1.1 truthy token — safe_load gives the bool True key.
    assert (True in loaded) or ("on" in loaded)


@pytest.mark.asyncio
async def test_gen_mobile_ci_ios_only(tmp_path):
    res = await gen_mobile_ci_mod.gen_mobile_ci(
        mission_id=None,
        workspace_path=str(tmp_path),
        platforms=["ios"],
        bundle_id="com.acme.demo",
    )
    assert res["ok"] is True
    assert res["jobs"] == ["ios"]

    import yaml
    with open(res["workflow_path"], encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    assert set(loaded["jobs"].keys()) == {"ios"}


@pytest.mark.asyncio
async def test_gen_mobile_ci_rejects_unknown_platform(tmp_path):
    res = await gen_mobile_ci_mod.gen_mobile_ci(
        mission_id=None,
        workspace_path=str(tmp_path),
        platforms=["ios", "blackberry"],
    )
    assert res["ok"] is False
    assert "unsupported" in (res["error"] or "")


@pytest.mark.asyncio
async def test_gen_mobile_ci_dispatch_via_run(tmp_path):
    task = {
        "id": 10,
        "mission_id": 42,
        "payload": {
            "action": "gen_mobile_ci",
            "workspace_path": str(tmp_path),
            "platforms": ["ios", "android"],
            "bundle_id": "com.acme.demo",
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert sorted(action.result["jobs"]) == ["android", "ios"]
    # gen_mobile_ci writes a single local file — fully reversible.
    assert action.reversibility == "full"


# ==========================================================================
# fastlane
# ==========================================================================

@pytest.mark.asyncio
async def test_fastlane_success_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fastlane_mod, "run_cmd", _ok_run_cmd(stdout="fastlane build done"),
    )
    res = await fastlane_mod.fastlane(
        mission_id=None,
        lane="build",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["skipped"] is False
    assert res["lane"] == "build"
    assert res["reversibility"] == "full"
    assert res["exit"] == 0
    assert "fastlane build done" in res["stdout_tail"]
    assert res["error"] is None


@pytest.mark.asyncio
async def test_fastlane_missing_cli_soft_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(fastlane_mod, "run_cmd", _missing_exe_run_cmd())
    res = await fastlane_mod.fastlane(
        mission_id=None,
        lane="pilot",
        workspace_path=str(tmp_path),
    )
    # Missing CLI is a soft-skip — never a hard failure.
    assert res["skipped"] is True
    assert res["ok"] is True
    assert res["error"] is None


@pytest.mark.asyncio
async def test_fastlane_rejects_unknown_lane(tmp_path):
    res = await fastlane_mod.fastlane(
        mission_id=None,
        lane="deploy-the-thing",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert res["skipped"] is False
    assert "unsupported" in (res["error"] or "")


@pytest.mark.asyncio
async def test_fastlane_nonzero_exit_is_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fastlane_mod, "run_cmd", _ok_run_cmd(stderr="boom", exit_code=1),
    )
    res = await fastlane_mod.fastlane(
        mission_id=None,
        lane="match",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert res["skipped"] is False
    assert res["exit"] == 1


@pytest.mark.asyncio
async def test_fastlane_timeout_is_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(fastlane_mod, "run_cmd", _timeout_run_cmd())
    res = await fastlane_mod.fastlane(
        mission_id=None,
        lane="build",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "timed out" in (res["error"] or "")


@pytest.mark.asyncio
async def test_fastlane_dispatch_via_run(monkeypatch, tmp_path):
    monkeypatch.setattr(fastlane_mod, "run_cmd", _ok_run_cmd(stdout="ok"))
    task = {
        "id": 11,
        "mission_id": 42,
        "payload": {
            "action": "fastlane",
            "lane": "build",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["lane"] == "build"


@pytest.mark.asyncio
async def test_fastlane_dispatch_missing_cli_not_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(fastlane_mod, "run_cmd", _missing_exe_run_cmd())
    task = {
        "id": 11,
        "mission_id": 42,
        "payload": {
            "action": "fastlane",
            "lane": "supply",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["skipped"] is True


# ==========================================================================
# fastlane per-lane reversibility
# ==========================================================================

@pytest.mark.asyncio
async def test_fastlane_pilot_lane_dispatch_is_irreversible(
    monkeypatch, tmp_path,
):
    """`pilot` uploads to TestFlight — the Action must tag irreversible."""
    monkeypatch.setattr(fastlane_mod, "run_cmd", _ok_run_cmd(stdout="ok"))
    task = {
        "id": 12,
        "mission_id": 42,
        "payload": {
            "action": "fastlane",
            "lane": "pilot",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.reversibility == "irreversible"


@pytest.mark.asyncio
async def test_fastlane_supply_lane_dispatch_is_irreversible(
    monkeypatch, tmp_path,
):
    """`supply` uploads to the Play internal track — irreversible."""
    monkeypatch.setattr(fastlane_mod, "run_cmd", _ok_run_cmd(stdout="ok"))
    task = {
        "id": 13,
        "mission_id": 42,
        "payload": {
            "action": "fastlane",
            "lane": "supply",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.reversibility == "irreversible"


@pytest.mark.asyncio
async def test_fastlane_build_lane_dispatch_is_full(monkeypatch, tmp_path):
    """`build` is a local, re-runnable compile — fully reversible."""
    monkeypatch.setattr(fastlane_mod, "run_cmd", _ok_run_cmd(stdout="ok"))
    task = {
        "id": 14,
        "mission_id": 42,
        "payload": {
            "action": "fastlane",
            "lane": "build",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.reversibility == "full"


@pytest.mark.asyncio
async def test_fastlane_match_lane_dispatch_is_full(monkeypatch, tmp_path):
    """`match` syncs signing material locally — fully reversible."""
    monkeypatch.setattr(fastlane_mod, "run_cmd", _ok_run_cmd(stdout="ok"))
    task = {
        "id": 15,
        "mission_id": 42,
        "payload": {
            "action": "fastlane",
            "lane": "match",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.reversibility == "full"


def test_fastlane_lane_reversibility_helper():
    """The pure lane→tag helper used by the dispatcher override."""
    from mr_roboto.fastlane_run import lane_reversibility

    assert lane_reversibility("build") == "full"
    assert lane_reversibility("match") == "full"
    assert lane_reversibility("pilot") == "irreversible"
    assert lane_reversibility("supply") == "irreversible"
    # Unknown / missing → conservative default.
    assert lane_reversibility("nonsense") == "irreversible"
    assert lane_reversibility(None) == "irreversible"


def test_fastlane_base_reversibility_is_conservative():
    """The base VERB_REVERSIBILITY entry is the conservative default."""
    assert get_reversibility("fastlane") == "irreversible"


def test_gen_mobile_ci_reversibility_is_full():
    """gen_mobile_ci writes one local file — full."""
    assert get_reversibility("gen_mobile_ci") == "full"


# ==========================================================================
# mobile_ci recipe
# ==========================================================================

def test_mobile_ci_recipe_loads_and_matches():
    """The mobile_ci recipe is discoverable and matches the expo stack."""
    recipes = list_recipes("recipes")
    names = {r.name for r in recipes}
    assert "mobile_ci" in names, f"mobile_ci not in {sorted(names)}"

    matches = match_recipe("expo", recipes)
    matched_names = {r.name for r, _score in matches}
    assert "mobile_ci" in matched_names, (
        f"mobile_ci did not match 'expo' stack; matched: {matched_names}"
    )


def test_mobile_ci_recipe_instantiates_cleanly(tmp_path):
    """instantiate_recipe round-trips with no unresolved <<...>> tokens."""
    recipe = load_recipe("recipes/mobile_ci/v1/recipe.yaml")
    assert recipe.lessons_domain == "mobile_ci"
    assert "expo" in " ".join(recipe.requires.get("tech_stack") or [])

    result = instantiate_recipe(recipe, str(tmp_path), params={})
    assert result["ok"] is True
    assert result["files_written"], "no files written"

    # No unresolved <<KEY>> tokens should remain in any instantiated file.
    import os
    import re

    token_re = re.compile(r"<<[A-Z_][A-Z0-9_]*>>")
    for root, _dirs, files in os.walk(tmp_path):
        for fname in files:
            path = os.path.join(root, fname)
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            leftover = token_re.findall(text)
            assert not leftover, f"unresolved tokens {leftover} in {path}"
