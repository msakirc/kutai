import os
import sys
import pytest

import mr_roboto
from mr_roboto.run_cmd import run_cmd


PY = sys.executable  # use the same interpreter the tests run under


@pytest.mark.asyncio
async def test_exit_zero_ok(tmp_path):
    res = await run_cmd(
        mission_id=None,
        cmd=[PY, "-c", "print('hi')"],
        workspace_path=str(tmp_path),
        require_exit_zero=True,
    )
    assert res["exit"] == 0
    assert res["ok"] is True
    assert "hi" in res["stdout_tail"]
    assert res["timed_out"] is False


@pytest.mark.asyncio
async def test_nonzero_exit_marks_not_ok_when_required(tmp_path):
    res = await run_cmd(
        mission_id=None,
        cmd=[PY, "-c", "import sys; sys.exit(3)"],
        workspace_path=str(tmp_path),
        require_exit_zero=True,
    )
    assert res["exit"] == 3
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_nonzero_exit_ok_when_not_required(tmp_path):
    res = await run_cmd(
        mission_id=None,
        cmd=[PY, "-c", "import sys; sys.exit(2)"],
        workspace_path=str(tmp_path),
        require_exit_zero=False,
    )
    assert res["exit"] == 2
    # Without require_exit_zero, finished-without-timeout counts as ok=True.
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_stderr_captured(tmp_path):
    res = await run_cmd(
        mission_id=None,
        cmd=[PY, "-c", "import sys; sys.stderr.write('boom')"],
        workspace_path=str(tmp_path),
    )
    assert "boom" in res["stderr_tail"]


@pytest.mark.asyncio
async def test_timeout_kills_process(tmp_path):
    res = await run_cmd(
        mission_id=None,
        cmd=[PY, "-c", "import time; time.sleep(5)"],
        workspace_path=str(tmp_path),
        timeout_s=0.5,
        require_exit_zero=True,
    )
    assert res["timed_out"] is True
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_cwd_resolved_under_workspace(tmp_path):
    sub = tmp_path / "inner"
    sub.mkdir()
    res = await run_cmd(
        mission_id=None,
        cmd=[PY, "-c", "import os; print(os.getcwd())"],
        cwd="inner",
        workspace_path=str(tmp_path),
    )
    assert res["exit"] == 0
    # Realpath equality (Windows + macOS may symlink temp dirs)
    assert os.path.realpath(res["stdout_tail"].strip()) == os.path.realpath(str(sub))


@pytest.mark.asyncio
async def test_cwd_traversal_rejected(tmp_path):
    inner = tmp_path / "ws"
    inner.mkdir()
    res = await run_cmd(
        mission_id=None,
        cmd=[PY, "-c", "print(1)"],
        cwd="../outside",
        workspace_path=str(inner),
    )
    assert res["ok"] is False
    assert "rejected" in (res.get("error") or "")


@pytest.mark.asyncio
async def test_absolute_cwd_rejected(tmp_path):
    res = await run_cmd(
        mission_id=None,
        cmd=[PY, "-c", "print(1)"],
        cwd=str(tmp_path),
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "rejected" in (res.get("error") or "")


@pytest.mark.asyncio
async def test_cmd_must_be_list(tmp_path):
    res = await run_cmd(
        mission_id=None,
        cmd="echo hi",  # not a list
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "non-empty list" in (res.get("error") or "")


@pytest.mark.asyncio
async def test_executable_not_found(tmp_path):
    res = await run_cmd(
        mission_id=None,
        cmd=["definitely_not_a_real_binary_xyz123"],
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "not found" in (res.get("error") or "")


@pytest.mark.asyncio
async def test_run_router_run_cmd_completed(tmp_path):
    task = {
        "id": 1,
        "mission_id": 99,
        "payload": {
            "action": "run_cmd",
            "cmd": [PY, "-c", "print('ok')"],
            "workspace_path": str(tmp_path),
            "require_exit_zero": True,
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["exit"] == 0


@pytest.mark.asyncio
async def test_run_router_run_cmd_failed_on_nonzero_when_required(tmp_path):
    task = {
        "id": 1,
        "mission_id": 99,
        "payload": {
            "action": "run_cmd",
            "cmd": [PY, "-c", "import sys; sys.exit(1)"],
            "workspace_path": str(tmp_path),
            "require_exit_zero": True,
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert action.result["exit"] == 1
