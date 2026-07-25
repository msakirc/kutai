"""write_file produces-sandbox — a step may only write its declared outputs.

Root: m90 5.20b (task 567455). The ``analyst`` step (produces = .screens/) had
``write_file`` and overwrote its READ-ONLY input ``.flow/screen_inventory.md``,
corrupting 5.0d's gated YAML inventory → the scaffold materialize read 0 chunk
routes → shape-gate DLQ. Fix: when the executing task declares ``produces``,
``write_file`` rejects any target outside those prefixes.
"""
from __future__ import annotations

import pytest

from src.core.heartbeat import current_task_produces
from src.tools import workspace


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "WORKSPACE_DIR", str(tmp_path), raising=False)
    monkeypatch.delenv("KUTAI_WRITE_SANDBOX", raising=False)
    return tmp_path


def _reset_produces():
    # Best-effort: the async test's set() lives in the event-loop task context,
    # which is discarded after the test; the finalizer runs in the main context
    # where reset(token) is cross-context. Just clear to the default.
    try:
        current_task_produces.set(None)
    except Exception:
        pass


def _set_produces(request, produces):
    current_task_produces.set(produces)
    request.addfinalizer(_reset_produces)


@pytest.mark.asyncio
async def test_write_outside_produces_is_rejected(ws, request):
    _set_produces(request, ["mission_90/.screens/"])
    res = await workspace.write_file(
        "mission_90/.flow/screen_inventory.md", "# clobbered\n")
    assert res.startswith("❌")                       # rejected
    assert not (ws / "mission_90/.flow/screen_inventory.md").exists()  # not written


@pytest.mark.asyncio
async def test_write_inside_produces_dir_is_allowed(ws, request):
    _set_produces(request, ["mission_90/.screens/"])
    res = await workspace.write_file(
        "mission_90/.screens/habits/screen_plan.md", "---\nx: 1\n---\nbody\n")
    assert res.startswith("✅")
    assert (ws / "mission_90/.screens/habits/screen_plan.md").exists()


@pytest.mark.asyncio
async def test_write_to_produces_file_exact_is_allowed(ws, request):
    _set_produces(request, ["mission_90/.adr/decision.json"])
    res = await workspace.write_file("mission_90/.adr/decision.json", '{"adr_id":"A"}')
    assert res.startswith("✅")
    assert (ws / "mission_90/.adr/decision.json").exists()


@pytest.mark.asyncio
async def test_no_produces_declared_allows_any_write(ws, request):
    _set_produces(request, [])                        # ad-hoc /task: no produces
    res = await workspace.write_file("scratch/notes.md", "free\n")
    assert res.startswith("✅")
    assert (ws / "scratch/notes.md").exists()


@pytest.mark.asyncio
async def test_code_phase_produces_are_not_sandboxed(ws, request):
    """Code-build steps produce real repo paths (backend/, frontend/), not the
    mission_<id> artifact namespace. The shared codebase is built collaboratively
    across many code steps that write files beyond a rigid produces list — the
    sandbox must NOT fire there, or the whole build phase false-rejects."""
    _set_produces(request, ["backend/app/models/habit.py",
                            "backend/app/services/habit_service.py"])
    res = await workspace.write_file("backend/app/routes/habit.py", "# router\n")
    assert res.startswith("✅")                        # sibling code file allowed
    assert (ws / "backend/app/routes/habit.py").exists()


@pytest.mark.asyncio
async def test_killswitch_disables_sandbox(ws, request, monkeypatch):
    monkeypatch.setenv("KUTAI_WRITE_SANDBOX", "off")
    _set_produces(request, ["mission_90/.screens/"])
    res = await workspace.write_file("mission_90/.flow/screen_inventory.md", "x\n")
    assert res.startswith("✅")                        # sandbox bypassed
    assert (ws / "mission_90/.flow/screen_inventory.md").exists()
