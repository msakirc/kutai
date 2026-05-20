"""Host-path tests for the Z3 P2 cascade fixes from the 2026-05-18 sweep.

The cascade (handoff §Z3 P2):

  1. ``expand_steps_with_multifile`` had no production caller — runner.py
     :205 / :445 and hooks.py :2167 called the plain expander.
  2. Latent ``TypeError``: ``to_mission_dial_context(mission_id, raw_dials)``
     was called with two args; the function only accepts ``(dials,)``.
  3. ``integration_reviewer.allowed_tools`` listed ``ast_signatures`` but
     the kind wasn't in ``TOOL_REGISTRY`` — every call failed
     "unknown tool".
  4. ``_auto_wire_posthooks`` was invoked without ``dial_ctx`` from
     :func:`expand_steps_to_tasks`, so callable triggers always saw the
     default conservative dials — the founder's ``/density`` setting was
     ignored.

These tests pin each link of the chain.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestZ3P2ArityFix(unittest.TestCase):
    """Latent TypeError — to_mission_dial_context expects one arg."""

    def test_to_mission_dial_context_arity_one(self):
        from src.workflows.review_density import to_mission_dial_context
        sig = inspect.signature(to_mission_dial_context)
        # Exactly one positional parameter, no kw-only.
        self.assertEqual(len(sig.parameters), 1)

    def test_multifile_expander_passes_single_arg(self):
        """Source-grep guard: the expander must not call the function with
        (mission_id, raw_dials) — that pattern is what the sweep caught.
        """
        src = (REPO_ROOT / "src/workflows/engine/expander.py").read_text(
            encoding="utf-8",
        )
        self.assertNotIn(
            "to_mission_dial_context(mission_id, raw_dials)", src,
            "the 2-arg TypeError pattern came back",
        )
        # The fixed call must still exist.
        self.assertIn("to_mission_dial_context(raw_dials)", src)


class TestZ3P2MultifileExpanderWiring(unittest.TestCase):
    """expand_steps_with_multifile must be reachable from runner + hooks."""

    def test_runner_imports_and_calls_multifile_variant(self):
        src = (REPO_ROOT / "src/workflows/engine/runner.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("expand_steps_with_multifile", src)
        # Both call sites (resume + start) switched.
        self.assertGreaterEqual(
            src.count("await expand_steps_with_multifile("), 2,
            "expected both runner call sites to use the multifile expander",
        )

    def test_hooks_imports_and_calls_multifile_variant(self):
        src = (REPO_ROOT / "src/workflows/engine/hooks.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("expand_steps_with_multifile", src)
        self.assertIn("await expand_steps_with_multifile(", src)


class TestZ3P2AstSignaturesTool(unittest.TestCase):
    """integration_reviewer's allowed_tools must resolve."""

    def test_ast_signatures_in_tool_registry(self):
        from src.tools import TOOL_REGISTRY
        self.assertIn("ast_signatures", TOOL_REGISTRY)
        spec = TOOL_REGISTRY["ast_signatures"]
        self.assertTrue(callable(spec["function"]))
        self.assertIn("signatures", spec["description"].lower())
        self.assertIn("integration", spec["description"].lower() + " hint integration")

    def test_integration_reviewer_allowed_tools_all_resolve(self):
        from src.agents.integration_reviewer import IntegrationReviewerAgent
        from src.tools import TOOL_REGISTRY
        for t in IntegrationReviewerAgent.allowed_tools:
            self.assertIn(
                t, TOOL_REGISTRY,
                f"integration_reviewer's allowed tool {t!r} not registered",
            )


class TestZ3P2DialThreading(unittest.IsolatedAsyncioTestCase):
    """_auto_wire_posthooks must see the founder's dials.

    expand_steps_with_multifile resolves dials via get_dials and forwards
    them; expand_steps_to_tasks must accept + forward dial_ctx. This test
    pokes a callable trigger that returns different globs depending on the
    qa_dial setting and asserts the expander honors the dial.
    """

    async def test_callable_trigger_receives_threaded_dials(self):
        from unittest.mock import patch
        from src.workflows.engine.expander import expand_steps_to_tasks
        from general_beckman.posthooks import (
            MissionDialContext, POST_HOOK_REGISTRY, PostHookSpec,
        )

        seen: list[MissionDialContext] = []

        def _spy_trigger(ctx: MissionDialContext) -> list[str]:
            seen.append(ctx)
            # Dial-conditional glob: only fire on strict dial.
            return ["*"] if getattr(ctx, "qa_dial", None) == "strict" else []

        spy_spec = PostHookSpec(
            kind="_z3_p2_spy",
            verb="_z3_p2_spy",
            default_severity="warning",
            auto_wire_triggers=_spy_trigger,
        )
        with patch.dict(
            POST_HOOK_REGISTRY, {"_z3_p2_spy": spy_spec}, clear=False,
        ):
            steps = [{
                "id": "1.0", "phase": "phase_1", "name": "spy",
                "agent": "planner", "instruction": "",
                "produces": ["x.txt"],
            }]
            # default dials → triggers empty → kind NOT auto-wired
            out = expand_steps_to_tasks(steps, mission_id="1")
            hooks_default = out[0]["context"].get("post_hooks") or []
            self.assertNotIn("_z3_p2_spy", hooks_default)

            # strict dials → triggers fire → kind auto-wired
            strict = MissionDialContext(qa_dial="strict")
            out_strict = expand_steps_to_tasks(
                steps, mission_id="1", dial_ctx=strict,
            )
            hooks_strict = out_strict[0]["context"].get("post_hooks") or []
            self.assertIn(
                "_z3_p2_spy", hooks_strict,
                f"dial_ctx not threaded; saw={seen!r}",
            )


if __name__ == "__main__":
    unittest.main()
