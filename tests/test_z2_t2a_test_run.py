"""Z2 T2A — test_run post-hook kind tests.

Covers:
1. Registry contains test_run with correct fields + triggers.
2. Expander auto-wires test_run when produces includes a test file.
3. _run_dispatch picks run_pytest for .py targets, run_jest for jest-hint,
   run_vitest for .ts without jest, and "no runner" for unknown extensions.
4. Failure surface: red pytest → ok=False; green → ok=True.
   (Uses tmp_path with a tiny real pytest file.)
5. _posthook_agent_and_payload maps test_run → mechanical run_tests payload.
"""
from __future__ import annotations

import asyncio
import json
import shutil

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_node() -> bool:
    return shutil.which("node") is not None


def _make_step(produces: list[str], post_hooks: list[str] | None = None) -> dict:
    step: dict = {
        "id": "1.1",
        "phase": "phase_1",
        "name": "write_tests",
        "agent": "coder",
        "instruction": "Write tests.",
        "depends_on": [],
        "input_artifacts": [],
        "output_artifacts": [],
        "produces": produces,
    }
    if post_hooks is not None:
        step["post_hooks"] = post_hooks
    return step


def _make_task(action: str, **payload_fields) -> dict:
    return {
        "id": 1,
        "mission_id": None,
        "payload": {"action": action, **payload_fields},
    }


# ---------------------------------------------------------------------------
# 1.  Registry
# ---------------------------------------------------------------------------

def test_registry_has_test_run():
    from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec
    assert "test_run" in POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["test_run"]
    assert isinstance(spec, PostHookSpec)
    assert spec.kind == "test_run"
    assert spec.verb == "run_tests"
    assert spec.default_severity == "blocker"


def test_test_run_triggers():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["test_run"]
    triggers = spec.auto_wire_triggers
    assert "tests/*" in triggers
    assert "test_*.py" in triggers
    assert "*.test.ts" in triggers
    assert "*.test.tsx" in triggers
    assert "*.spec.ts" in triggers
    assert "*.spec.tsx" in triggers


def test_test_run_in_post_hook_kinds():
    from general_beckman.posthooks import POST_HOOK_KINDS
    assert "test_run" in POST_HOOK_KINDS


def test_post_hook_kinds_derived_from_registry():
    from general_beckman.posthooks import POST_HOOK_REGISTRY, POST_HOOK_KINDS
    assert POST_HOOK_KINDS == frozenset(POST_HOOK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 2.  Expander auto-wire
# ---------------------------------------------------------------------------

def _get_post_hooks(task: dict) -> list[str]:
    ctx = task.get("context") or {}
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    return ctx.get("post_hooks") or []


def test_expander_autowires_test_run_for_py_test():
    from src.workflows.engine.expander import expand_steps_to_tasks
    steps = [_make_step(produces=["tests/test_foo.py"])]
    tasks = expand_steps_to_tasks(steps, mission_id="1")
    assert "test_run" in _get_post_hooks(tasks[0])


def test_expander_autowires_test_run_for_ts_test():
    from src.workflows.engine.expander import expand_steps_to_tasks
    steps = [_make_step(produces=["src/foo.test.ts"])]
    tasks = expand_steps_to_tasks(steps, mission_id="1")
    assert "test_run" in _get_post_hooks(tasks[0])


def test_expander_does_not_duplicate_test_run():
    from src.workflows.engine.expander import expand_steps_to_tasks
    steps = [_make_step(
        produces=["tests/test_foo.py"],
        post_hooks=["test_run"],
    )]
    tasks = expand_steps_to_tasks(steps, mission_id="1")
    hooks = _get_post_hooks(tasks[0])
    assert hooks.count("test_run") == 1


def test_expander_does_not_wire_test_run_for_non_test_file():
    from src.workflows.engine.expander import expand_steps_to_tasks
    steps = [_make_step(produces=["src/main.py"])]
    tasks = expand_steps_to_tasks(steps, mission_id="1")
    assert "test_run" not in _get_post_hooks(tasks[0])


# ---------------------------------------------------------------------------
# 3.  Runner dispatch in _run_dispatch
# ---------------------------------------------------------------------------

def _make_capturing_run_cmd(cmd_calls: list, exit_code: int = 0, stdout: str = "1 passed"):
    """Return an async run_cmd replacement that records argv and returns canned result."""
    async def _mock(**kwargs):
        cmd_calls.append(list(kwargs.get("cmd") or []))
        return {
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": "",
            "duration_s": 0.1,
            "timed_out": False,
            "error": None,
        }
    return _mock


def _patch_run_cmd(module_dotted: str, cmd_calls: list):
    """Context manager: patch run_cmd inside a runner submodule via sys.modules."""
    import sys
    import importlib
    from unittest.mock import patch

    # Ensure the real submodule is loaded into sys.modules
    importlib.import_module(module_dotted)
    return patch.object(sys.modules[module_dotted], "run_cmd",
                        _make_capturing_run_cmd(cmd_calls))


def test_run_tests_dispatches_pytest_for_py():
    """action=run_tests with *.py target → run_cmd called with pytest argv."""
    cmd_calls: list = []
    import mr_roboto as mr

    with _patch_run_cmd("mr_roboto.run_pytest", cmd_calls):
        task = _make_task("run_tests", target_files=["tests/test_foo.py"], stack_hint="")
        action = asyncio.run(mr._run_dispatch(task))

    assert action.status == "completed"
    assert any("pytest" in " ".join(c) for c in cmd_calls)


def test_run_tests_dispatches_jest_when_stack_hint():
    """action=run_tests + stack_hint=jest → run_cmd called with jest argv."""
    cmd_calls: list = []
    import mr_roboto as mr

    with _patch_run_cmd("mr_roboto.run_jest", cmd_calls):
        task = _make_task(
            "run_tests", target_files=["src/foo.test.ts"], stack_hint="jest",
        )
        action = asyncio.run(mr._run_dispatch(task))

    assert action.status == "completed"
    assert any("jest" in " ".join(c) for c in cmd_calls)


def test_run_tests_dispatches_vitest_when_no_jest_hint():
    """action=run_tests + .ts target + no jest hint → vitest argv."""
    cmd_calls: list = []
    import mr_roboto as mr

    with _patch_run_cmd("mr_roboto.run_vitest", cmd_calls):
        task = _make_task(
            "run_tests", target_files=["src/foo.spec.ts"], stack_hint="",
            workspace_path="/nonexistent_path_xyz",  # no package.json here
        )
        action = asyncio.run(mr._run_dispatch(task))

    assert action.status == "completed"
    assert any("vitest" in " ".join(c) for c in cmd_calls)


def test_run_tests_no_runner_for_unknown_extension():
    """action=run_tests with *.rb → completed with no_runner_detected warning."""
    import mr_roboto as mr
    task = _make_task("run_tests", target_files=["spec/foo_spec.rb"])
    action = asyncio.run(mr._run_dispatch(task))
    assert action.status == "completed"
    assert (action.result or {}).get("warning") == "no_runner_detected"


def test_run_tests_empty_targets():
    """action=run_tests with no targets → no_runner_detected, not a crash."""
    import mr_roboto as mr
    task = _make_task("run_tests", target_files=[])
    action = asyncio.run(mr._run_dispatch(task))
    assert action.status == "completed"
    assert (action.result or {}).get("warning") == "no_runner_detected"


# ---------------------------------------------------------------------------
# 4.  Green / red pytest via tmp_path (real subprocess)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pytest_green(tmp_path):
    from mr_roboto.run_pytest import run_pytest

    test_file = tmp_path / "test_pass.py"
    test_file.write_text("def test_ok():\n    assert 1 + 1 == 2\n")

    result = await run_pytest(
        mission_id=None,
        target=[str(test_file)],
        workspace_path=str(tmp_path),
    )
    assert result["ok"] is True
    assert result["passed"] >= 1
    assert result["failed"] == 0


@pytest.mark.asyncio
async def test_run_pytest_red(tmp_path):
    from mr_roboto.run_pytest import run_pytest

    test_file = tmp_path / "test_fail.py"
    test_file.write_text("def test_bad():\n    assert 1 == 2\n")

    result = await run_pytest(
        mission_id=None,
        target=[str(test_file)],
        workspace_path=str(tmp_path),
    )
    assert result["ok"] is False
    assert result["failed"] >= 1


@pytest.mark.asyncio
async def test_run_pytest_import_error(tmp_path):
    """Import error → ok=False (zero collected counts as red)."""
    from mr_roboto.run_pytest import run_pytest

    test_file = tmp_path / "test_broken.py"
    test_file.write_text(
        "import nonexistent_module_xyz_12345\n\ndef test_x():\n    pass\n"
    )

    result = await run_pytest(
        mission_id=None,
        target=[str(test_file)],
        workspace_path=str(tmp_path),
    )
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# 5.  _posthook_agent_and_payload for test_run
# ---------------------------------------------------------------------------

def test_posthook_payload_for_test_run():
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    sc = {
        "produces": ["tests/test_foo.py", "tests/test_bar.py"],
        "stack_hint": "fastapi",
    }
    a = RequestPostHook(source_task_id=42, kind="test_run", source_ctx=sc)
    source = {"id": 42, "title": "write tests"}
    agent_type, payload = _posthook_agent_and_payload(a, source, sc)

    assert agent_type == "mechanical"
    assert payload["posthook_kind"] == "test_run"
    inner = payload["payload"]
    assert inner["action"] == "run_tests"
    assert "tests/test_foo.py" in inner["target_files"]
    assert inner["stack_hint"] == "fastapi"


def test_posthook_payload_test_run_no_stack_hint():
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    sc = {"produces": ["test_something.py"]}
    a = RequestPostHook(source_task_id=7, kind="test_run", source_ctx=sc)
    source = {"id": 7}
    _, payload = _posthook_agent_and_payload(a, source, sc)
    assert payload["payload"]["stack_hint"] == ""
