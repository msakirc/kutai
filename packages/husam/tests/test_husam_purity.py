"""SP3b Task 9 — husam structural purity assertions.

Three invariants proved here:
  1. Importing husam does NOT pull in coulson (coulson is the ReAct worker;
     husam is the single-call dumb worker — no dependency permitted).
  2. LLMDispatcher has NO ``dispatch``, ``_do_dispatch``, or
     ``_result_to_dict`` attributes (moved to husam in Task 2); HAS ``execute``
     (the dumb-pipe primitive husam calls).
  3. The husam.worker source contains NO reference to ``await_inline`` — the
     worker must never block the Beckman queue on an inline waiter.
"""
from __future__ import annotations

import inspect
import sys


# ---------------------------------------------------------------------------
# 1. Purity: husam import must NOT pull coulson into sys.modules
# ---------------------------------------------------------------------------

def test_husam_does_not_import_coulson():
    """Purity: importing husam must not drag coulson into the module graph.

    husam is the non-agentic single-call worker; coulson owns the ReAct loop.
    The two must never share a transitive import dependency at load time.
    """
    # Import husam (may already be cached in the test session — that is fine).
    import husam  # noqa: F401

    coulson_mods = [m for m in sys.modules if m == "coulson" or m.startswith("coulson.")]
    assert not coulson_mods, (
        f"husam import pulled in coulson modules: {coulson_mods}"
    )


# ---------------------------------------------------------------------------
# 2. Dumb-pipe assertion: LLMDispatcher shape after SP3b Task 2
# ---------------------------------------------------------------------------

def test_llm_dispatcher_is_dumb_pipe():
    """LLMDispatcher must be a dumb pipe: execute exists, old dispatch methods gone.

    SP3b Task 2 moved dispatch/_do_dispatch/_result_to_dict to husam.
    Asserting their absence here prevents silent re-introduction.
    """
    from src.core.llm_dispatcher import LLMDispatcher

    d = LLMDispatcher()

    # Must have the dumb-pipe primitive (single attempt, no selection, no retry)
    assert hasattr(d, "execute") and callable(d.execute), (
        "LLMDispatcher.execute is missing — dumb-pipe primitive not wired"
    )

    # Must NOT have the deleted orchestration methods (moved to husam)
    for dead_attr in ("dispatch", "_do_dispatch", "_result_to_dict"):
        assert not hasattr(d, dead_attr), (
            f"LLMDispatcher still has .{dead_attr} — it should have been "
            f"deleted in SP3b Task 2 (orchestration moved to husam.worker)"
        )


# ---------------------------------------------------------------------------
# 3. husam.worker source contains no await_inline
# ---------------------------------------------------------------------------

def test_husam_worker_source_has_no_await_inline():
    """husam.worker must never use await_inline (the blocking queue primitive).

    await_inline is the mechanism that caused the old deadlock: a parent task
    holding a lane slot while blocking on a child needing the SAME lane.
    husam.worker is the worker for raw_dispatch tasks — it runs ONE LLM call
    via the dumb-pipe dispatcher.execute() and returns.  It must NEVER
    enqueue a child with await_inline=True (that would recreate the deadlock).
    """
    import husam.worker as _worker_mod

    src = inspect.getsource(_worker_mod)
    assert "await_inline" not in src, (
        "husam.worker source contains 'await_inline' — this would recreate "
        "the lane-deadlock that SP3b was designed to eliminate. "
        "The worker must not block on any Beckman inline waiter."
    )
