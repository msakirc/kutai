"""SP3b Task 7 — purity tests: coulson is react-only after removing inline callers.

After Task 7:
- ``coulson.execute`` must NOT call ``maybe_apply`` inline (constrained_emit is a post-hook).
- ``coulson.react.run`` must NOT call ``self_reflect(`` inline (self_reflect is a post-hook).
- Importing coulson must NOT pull in ``husam`` (no SP3 husam worker coupling).
"""
import inspect
import os
import pathlib
import subprocess
import sys

# Worktree root — three levels up from packages/coulson/tests/
_WORKTREE = pathlib.Path(__file__).parent.parent.parent.parent


def _make_subprocess_pythonpath() -> str:
    """Build PYTHONPATH that mirrors conftest.py path injections."""
    packages = [
        "fatih_hoca", "nerd_herd", "kuleden_donen_var", "general_beckman",
        "hallederiz_kadir", "dallama", "dogru_mu_samet", "vecihi",
        "yasar_usta", "yazbunu", "mr_roboto", "coulson", "sade_kalsin",
        "c21_paraflow_diff", "intersect", "yalayut", "safety_guard", "husam",
    ]
    paths = [str(_WORKTREE)]
    for pkg in packages:
        p = _WORKTREE / "packages" / pkg / "src"
        if p.is_dir():
            paths.append(str(p))
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def test_coulson_execute_has_no_inline_emit():
    import coulson
    src = inspect.getsource(coulson.execute)
    assert "maybe_apply" not in src, "constrained_emit still inline in coulson.execute"


def test_react_run_has_no_inline_self_reflect():
    import coulson.react as react
    src = inspect.getsource(react.run)
    assert "self_reflect(" not in src, "self_reflect still inline in react.run"


def test_coulson_does_not_import_husam():
    """Purity: importing coulson must not drag husam into the module graph.

    coulson is the ReAct worker; husam is the non-agentic single-call worker.
    The two must never share a transitive import dependency at load time.

    This test uses a subprocess to avoid cross-test session pollution: if
    another test already imported husam into sys.modules before this test
    runs, checking sys.modules in-process gives a false positive.  The
    subprocess starts with a clean module graph regardless of test order.
    """
    code = (
        "import coulson, sys; "
        "bad = [m for m in sys.modules if m == 'husam' or m.startswith('husam.')]; "
        "assert not bad, f'coulson import pulled in husam modules: {bad}'"
    )
    env = {**os.environ, "PYTHONPATH": _make_subprocess_pythonpath()}
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
        env=env,
    )
    assert r.returncode == 0, (
        f"subprocess purity check failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"
    )
