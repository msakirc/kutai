import os
import pytest

import salako
from salako.verify_artifacts import verify_artifacts


@pytest.mark.asyncio
async def test_all_paths_present_passes(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")

    res = await verify_artifacts(
        mission_id=None,
        paths=["a.py", "sub/b.py"],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is True
    assert res["missing"] == []
    assert res["failed"] == []
    paths = {v["path"] for v in res["verified"]}
    assert paths == {"a.py", "sub/b.py"}
    for v in res["verified"]:
        assert v["bytes"] > 0
        assert len(v["sha256"]) == 64


@pytest.mark.asyncio
async def test_missing_file_reported(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    res = await verify_artifacts(
        mission_id=None,
        paths=["a.py", "ghost.py"],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False
    assert "ghost.py" in res["missing"]
    assert any(v["path"] == "a.py" for v in res["verified"])


@pytest.mark.asyncio
async def test_zero_byte_file_fails_min_bytes(tmp_path):
    (tmp_path / "empty.py").write_text("")
    res = await verify_artifacts(
        mission_id=None,
        paths=["empty.py"],
        min_bytes=1,
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False
    assert any(f["path"] == "empty.py" for f in res["failed"])


@pytest.mark.asyncio
async def test_absolute_path_rejected(tmp_path):
    res = await verify_artifacts(
        mission_id=None,
        paths=[str(tmp_path / "x.py")],  # absolute
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False
    assert any("rejected" in f["reason"] for f in res["failed"])


@pytest.mark.asyncio
async def test_traversal_rejected(tmp_path):
    inner = tmp_path / "ws"
    inner.mkdir()
    (tmp_path / "outside.py").write_text("x = 1\n")
    res = await verify_artifacts(
        mission_id=None,
        paths=["../outside.py"],
        workspace_path=str(inner),
    )
    assert res["all_ok"] is False
    assert any("rejected" in f["reason"] for f in res["failed"])


@pytest.mark.asyncio
async def test_compile_check_python_pass(tmp_path):
    (tmp_path / "good.py").write_text("def f():\n    return 1\n")
    res = await verify_artifacts(
        mission_id=None,
        paths=["good.py"],
        compile_check=True,
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is True


@pytest.mark.asyncio
async def test_compile_check_python_fail(tmp_path):
    (tmp_path / "bad.py").write_text("def f(:\n  return\n")  # syntax error
    res = await verify_artifacts(
        mission_id=None,
        paths=["bad.py"],
        compile_check=True,
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False
    assert any("compile check" in f["reason"] for f in res["failed"])


@pytest.mark.asyncio
async def test_compile_check_json_fail(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    res = await verify_artifacts(
        mission_id=None,
        paths=["broken.json"],
        compile_check=True,
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False


@pytest.mark.asyncio
async def test_compile_check_skips_unsupported_extension(tmp_path):
    # .tsx has no checker registered → must not fail just because ext unknown
    (tmp_path / "page.tsx").write_text("garbage that wouldnt parse anywhere")
    res = await verify_artifacts(
        mission_id=None,
        paths=["page.tsx"],
        compile_check=True,
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is True


@pytest.mark.asyncio
async def test_empty_paths_returns_failure(tmp_path):
    res = await verify_artifacts(
        mission_id=None,
        paths=[],
        workspace_path=str(tmp_path),
    )
    assert res["all_ok"] is False


@pytest.mark.asyncio
async def test_run_router_verify_artifacts_completed(tmp_path):
    (tmp_path / "ok.py").write_text("a = 1\n")
    task = {
        "id": 1,
        "mission_id": 99,
        "payload": {
            "action": "verify_artifacts",
            "paths": ["ok.py"],
            "workspace_path": str(tmp_path),
        },
    }
    action = await salako.run(task)
    assert action.status == "completed"
    assert action.result["all_ok"] is True


@pytest.mark.asyncio
async def test_run_router_verify_artifacts_failed_when_missing(tmp_path):
    task = {
        "id": 1,
        "mission_id": 99,
        "payload": {
            "action": "verify_artifacts",
            "paths": ["ghost.py"],
            "workspace_path": str(tmp_path),
        },
    }
    action = await salako.run(task)
    assert action.status == "failed"
    assert "ghost.py" in (action.error or "")
    assert action.result["missing"] == ["ghost.py"]
