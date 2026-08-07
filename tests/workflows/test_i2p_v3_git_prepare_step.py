"""7.12b — git_prepare_repo mechanical step runs before 7.13 staging deploy.

git_prepare_repo creates the mission's GitHub repo + pushes the scaffold and
persists missions.github_repo_url, which deploy_staging (7.13) reads via
_repo_from_mission. The step must therefore exist and 7.13 must depend on it.

Opened with encoding="utf-8" — i2p_v3.json has non-cp1252 bytes; a bare open()
crashes on Windows (cp1252 default).
"""
import json
from pathlib import Path

_WF = Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"


def _load():
    with open(_WF, encoding="utf-8") as f:
        return json.load(f)


def test_git_prepare_repo_step_exists_and_is_mechanical():
    wf = _load()
    step = next(s for s in wf["steps"] if s.get("id") == "7.12b")
    assert step["agent"] == "mechanical"
    assert step["executor"] == "git_prepare_repo"
    assert step["payload"]["action"] == "git_prepare_repo"


def test_staging_deploy_depends_on_git_prepare_repo():
    wf = _load()
    staging = next(s for s in wf["steps"] if s.get("id") == "7.13")
    assert "7.12b" in staging["depends_on"]
