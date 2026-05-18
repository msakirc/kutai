"""Z9 T2A — analytics_instrumentation/v1 recipe tests.

Covers:
- recipe.yaml loads via the recipe loader
- recipe is discoverable by list_recipes (nested recipes/<name>/<version>/ layout)
- dependencies declare posthog-js (frontend) + posthog (backend)
- lessons_domain is analytics_instrumentation
- template files exist
- the events template carries the full standard AARRR event taxonomy
- i2p step 13.4 wires the recipe + depends on success_metrics (step 2.9)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

WORKTREE_ROOT = Path(__file__).parent.parent
RECIPE_V1 = WORKTREE_ROOT / "recipes" / "analytics_instrumentation" / "v1"


# The standard AARRR taxonomy from docs/i2p-evolution/09-growth-v2.md.
EXPECTED_TAXONOMY = {
    "acquisition": ["landing_view", "signup_started", "signup_completed"],
    "activation": ["first_value_event"],
    "retention": ["session_started"],
    "revenue": [
        "checkout_started",
        "checkout_completed",
        "subscription_created",
        "subscription_cancelled",
    ],
    "referral": ["share_initiated", "share_completed", "invite_redeemed"],
}


# ---------------------------------------------------------------------------
# recipe.yaml
# ---------------------------------------------------------------------------

class TestRecipeYaml:
    def _load(self):
        from src.infra.recipes import load_recipe
        return load_recipe(str(RECIPE_V1 / "recipe.yaml"))

    def test_loads_without_error(self):
        recipe = self._load()
        assert recipe.name == "analytics_instrumentation"
        assert recipe.version == "v1"
        assert recipe.description

    def test_lessons_domain(self):
        assert self._load().lessons_domain == "analytics_instrumentation"

    def test_dependencies(self):
        deps = self._load().dependencies
        assert "posthog-js" in deps.get("frontend", []), deps
        assert "posthog" in deps.get("backend", []), deps

    def test_templates_present(self):
        templates = self._load().templates
        for role in ("client", "server", "events", "notes"):
            assert role in templates, f"{role} missing from templates"

    def test_template_files_exist_on_disk(self):
        recipe = self._load()
        for rel in recipe.templates.values():
            assert (RECIPE_V1 / rel).exists(), f"template file missing: {rel}"

    def test_param_defaults(self):
        pd = self._load().param_defaults
        assert "POSTHOG_API_KEY_ENV" in pd
        assert "POSTHOG_HOST_ENV" in pd
        assert "ACTIVATION_EVENT" in pd


# ---------------------------------------------------------------------------
# Discoverability
# ---------------------------------------------------------------------------

class TestDiscoverability:
    def test_discovered_by_list_recipes(self):
        from src.infra.recipes import list_recipes
        names = {r.name for r in list_recipes(str(WORKTREE_ROOT / "recipes"))}
        assert "analytics_instrumentation" in names, names


# ---------------------------------------------------------------------------
# Event taxonomy carried by the events template
# ---------------------------------------------------------------------------

class TestEventTaxonomy:
    def test_events_template_carries_full_aarrr_taxonomy(self):
        text = (RECIPE_V1 / "events.template.ts").read_text(encoding="utf-8")
        for stage, events in EXPECTED_TAXONOMY.items():
            assert stage in text, f"AARRR stage {stage!r} missing from events template"
            for ev in events:
                assert ev in text, f"event {ev!r} ({stage}) missing from events template"

    def test_server_shim_is_valid_python(self):
        text = (RECIPE_V1 / "server.template.py").read_text(encoding="utf-8")
        ast.parse(text)  # raises SyntaxError if the shim is malformed

    def test_track_event_helper_present_in_both_shims(self):
        client = (RECIPE_V1 / "client.template.ts").read_text(encoding="utf-8")
        server = (RECIPE_V1 / "server.template.py").read_text(encoding="utf-8")
        assert "track_event" in client
        assert "def track_event" in server


# ---------------------------------------------------------------------------
# i2p step 13.4 wiring
# ---------------------------------------------------------------------------

class TestI2pStep:
    def _step_13_4(self):
        path = WORKTREE_ROOT / "src" / "workflows" / "i2p" / "i2p_v3.json"
        d = json.loads(path.read_text(encoding="utf-8"))
        steps = [s for s in d["steps"] if str(s.get("id")) == "13.4"]
        assert len(steps) == 1, "expected exactly one step 13.4"
        return steps[0]

    def test_depends_on_success_metrics_step(self):
        # success_metrics artifact is produced by step 2.9
        assert "2.9" in self._step_13_4()["depends_on"]

    def test_produces_instrumentation_artifact(self):
        assert "analytics_instrumentation" in self._step_13_4()["output_artifacts"]

    def test_instruction_references_recipe(self):
        instr = self._step_13_4()["instruction"]
        assert "analytics_instrumentation/v1" in instr
        assert "aarrr_metrics" in instr

    def test_tools_hint_points_at_recipe_tools(self):
        hints = self._step_13_4()["tools_hint"]
        assert "pick_recipe" in hints or "instantiate_recipe" in hints
