"""Tests for workflow execution hooks (pre/post orchestrator integration)."""

import asyncio
import json
import unittest
from unittest.mock import patch, AsyncMock

from src.workflows.engine.hooks import (
    get_artifact_store,
    is_workflow_step,
    extract_output_artifact_names,
    post_execute_workflow_step,
    collect_empty_exemption_inputs,
)
# enrich_task_description + pre_execute_workflow_step removed
# 2026-04-27 (handoff item D). Tests for them deleted with the
# functions; live prompt-build path covered by tests/test_missing_
# artifact_note.py + tests/test_recency_reorder.py.
from src.workflows.engine.artifacts import ArtifactStore


class TestCollectEmptyExemptionInputs(unittest.TestCase):
    """The lazy loader behind the empty_ok_when_input_empty exemption.

    Anchors the exemption to the REAL upstream artifact file (authored by a
    different task), never the producer's own output — so a lazy producer
    cannot fake an empty-scope exemption.
    """

    _SCHEMA = {
        "compliance_overlay": {
            "type": "object",
            "fields": {
                "required_documents": {
                    "type": "array",
                    "empty_ok_when_input_empty":
                        "compliance_fingerprint.jurisdictions",
                },
            },
        }
    }

    def test_no_marker_returns_none_without_touching_fs(self):
        # Schema with no marker must short-circuit (zero file IO).
        plain = {"prd": {"type": "object", "required_fields": ["a"]}}
        with patch("src.workflows.engine.hooks.get_mission_workspace") as gw:
            out = collect_empty_exemption_inputs(plain, 87)
        self.assertIsNone(out)
        gw.assert_not_called()

    def test_loads_referenced_artifact_file(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "compliance_fingerprint.json"),
                      "w", encoding="utf-8") as f:
                json.dump({"jurisdictions": [], "user_classes": []}, f)
            with patch("src.workflows.engine.hooks.get_mission_workspace",
                       return_value=d):
                out = collect_empty_exemption_inputs(self._SCHEMA, 87)
        self.assertEqual(out, {"compliance_fingerprint":
                               {"jurisdictions": [], "user_classes": []}})

    def test_missing_file_yields_empty_inputs_not_crash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:  # no fingerprint file
            with patch("src.workflows.engine.hooks.get_mission_workspace",
                       return_value=d):
                out = collect_empty_exemption_inputs(self._SCHEMA, 87)
        # Nothing loaded → None (gate then has no proof → rejects empties).
        self.assertIsNone(out)


class TestIsWorkflowStep(unittest.TestCase):
    """Tests for is_workflow_step detection."""

    def test_is_workflow_step(self):
        """Returns True when is_workflow_step is set."""
        self.assertTrue(is_workflow_step({"is_workflow_step": True}))

    def test_is_workflow_step_false(self):
        """Returns False for empty dict or missing key."""
        self.assertFalse(is_workflow_step({}))
        self.assertFalse(is_workflow_step({"other_key": True}))

    def test_is_workflow_step_explicit_false(self):
        """Returns False when explicitly set to False."""
        self.assertFalse(is_workflow_step({"is_workflow_step": False}))


class TestExtractOutputArtifactNames(unittest.TestCase):
    """Tests for extract_output_artifact_names."""

    def test_extract_output_artifact_names(self):
        """Returns list from context output_artifacts."""
        ctx = {"output_artifacts": ["spec", "plan"]}
        self.assertEqual(extract_output_artifact_names(ctx), ["spec", "plan"])

    def test_extract_output_artifact_names_missing(self):
        """Returns empty list when key is absent."""
        self.assertEqual(extract_output_artifact_names({}), [])

    def test_extract_single_artifact(self):
        """Single item list works."""
        ctx = {"output_artifacts": ["result"]}
        self.assertEqual(extract_output_artifact_names(ctx), ["result"])



class TestGetArtifactStore(unittest.TestCase):
    """Tests for the module-level singleton."""

    def test_get_artifact_store_returns_instance(self):
        """Returns an ArtifactStore instance."""
        store = get_artifact_store()
        self.assertIsInstance(store, ArtifactStore)

    def test_get_artifact_store_singleton(self):
        """Returns the same instance on repeated calls."""
        store1 = get_artifact_store()
        store2 = get_artifact_store()
        self.assertIs(store1, store2)



class TestPostExecuteWorkflowStep(unittest.TestCase):
    """Tests for post_execute_workflow_step."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_non_workflow_step_noop(self):
        """Non-workflow tasks are ignored."""
        task = {"context": "{}"}
        # Should not raise
        self._run(post_execute_workflow_step(task, {"result": "result"}))

    def test_stores_single_output_artifact(self):
        """Single output artifact stores the full result."""
        store = get_artifact_store()
        task = {
            "context": json.dumps({
                "is_workflow_step": True,
                "mission_id": 100,
                "output_artifacts": ["analysis"],
            }),
        }
        result = {"result": "Analysis result text"}
        self._run(post_execute_workflow_step(task, result))

        stored = self._run(store.retrieve(100, "analysis"))
        self.assertEqual(stored, "Analysis result text")

    def test_stores_multiple_output_artifacts(self):
        """Multiple output artifacts each get the full result."""
        store = get_artifact_store()
        task = {
            "context": json.dumps({
                "is_workflow_step": True,
                "mission_id": 101,
                "output_artifacts": ["doc_a", "doc_b"],
            }),
        }
        result = {"result": "Combined output"}
        self._run(post_execute_workflow_step(task, result))

        stored_a = self._run(store.retrieve(101, "doc_a"))
        stored_b = self._run(store.retrieve(101, "doc_b"))
        self.assertEqual(stored_a, "Combined output")
        self.assertEqual(stored_b, "Combined output")

    # Candidate-state JSON repeats the same keys per candidate, which the
    # degeneracy detector reads as repetition (mission #84: groups_state 64915
    # chars flagged). Legit data, not LLM garbage — the deterministic dispatcher
    # must bypass the check; LLM agents must not.
    _REPETITIVE = '{"title":"X","site":"a","price":1,"url":"u"}\n' * 3000

    def test_deterministic_dispatcher_skips_degeneracy_gate(self):
        store = get_artifact_store()
        task = {
            "context": json.dumps({
                "is_workflow_step": True, "mission_id": 200,
                "output_artifacts": ["groups_state"],
            }),
            "agent_type": "shopping_pipeline_v2",
        }
        result = {"result": self._REPETITIVE}
        self._run(post_execute_workflow_step(task, result))
        self.assertNotEqual(result.get("status"), "failed")
        self.assertTrue(self._run(store.retrieve(200, "groups_state")))

    def test_llm_step_still_degeneracy_gated(self):
        task = {
            "context": json.dumps({
                "is_workflow_step": True, "mission_id": 201,
                "output_artifacts": ["analysis"],
            }),
            "agent_type": "researcher",
        }
        result = {"result": self._REPETITIVE}
        self._run(post_execute_workflow_step(task, result))
        self.assertEqual(result.get("status"), "failed")


if __name__ == "__main__":
    unittest.main()
