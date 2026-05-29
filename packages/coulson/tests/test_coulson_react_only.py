"""SP3b Task 7 — purity tests: coulson is react-only after removing inline callers.

After Task 7:
- ``coulson.execute`` must NOT call ``maybe_apply`` inline (constrained_emit is a post-hook).
- ``coulson.react.run`` must NOT call ``self_reflect(`` inline (self_reflect is a post-hook).
- Importing coulson must NOT pull in ``husam`` (no SP3 husam worker coupling).
"""
import inspect
import importlib


def test_coulson_execute_has_no_inline_emit():
    import coulson
    src = inspect.getsource(coulson.execute)
    assert "maybe_apply" not in src, "constrained_emit still inline in coulson.execute"


def test_react_run_has_no_inline_self_reflect():
    import coulson.react as react
    src = inspect.getsource(react.run)
    assert "self_reflect(" not in src, "self_reflect still inline in react.run"


def test_coulson_does_not_import_husam():
    import sys
    # Re-import to be sure module is loaded
    import coulson  # noqa: F401
    assert not any(
        m == "husam" or m.startswith("husam.")
        for m in sys.modules
    ), "husam was imported as a side-effect of importing coulson"
