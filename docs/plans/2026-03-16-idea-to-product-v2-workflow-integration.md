# Idea-to-Product v2 Workflow Engine Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate `idea_to_product_v2.json` as a first-class workflow engine into the existing orchestrator, enabling 24/7 autonomous execution of the complete product lifecycle (idea → research → design → build → launch → ops) while leveraging `CodingPipeline` for Phase 8 implementation work.

**Architecture:** Build a generic `WorkflowEngine` in `src/workflows/engine/` that loads JSON workflow definitions, evaluates conditional groups, expands templates, and inserts tasks into the existing DB with proper dependency chains. The orchestrator's existing `get_ready_tasks()` → `claim_task()` → `execute` loop handles execution. Artifacts flow between steps via an extended blackboard store. Phase 8 delegates each feature's coding work to `CodingPipeline` (nested invocation). Phase 15 recurring steps map to the existing `scheduled_tasks` system. The engine is workflow-agnostic — any JSON definition following the schema can be loaded.

**Tech Stack:** Python 3.12, aiosqlite, asyncio, JSON workflow definitions

---

## Revised Analysis: Pipeline vs Idea-to-Product Relationship

### Key Decision: Pipeline works WITHIN Idea-to-Product, not alongside

After analyzing both systems, the relationship is clear:

| Dimension | CodingPipeline | Idea-to-Product v2 |
|-----------|---------------|---------------------|
| Scope | Single coding task (feature/bugfix) | Full product lifecycle (17 phases, 150+ steps) |
| Duration | Minutes to hours | Days to weeks |
| Steps | 5-8 hardcoded stages | 150+ JSON-defined + 31-step template per feature |
| State | In-memory `PipelineContext` | Persistent artifact store (blackboard-backed) |
| Dependencies | Linear stages | Complex DAG with parallelism + conditional groups |
| Templates | None | `feature_implementation_template` (31 steps) |
| Human-in-loop | None (auto review loop) | `may_need_clarification` on ~15 steps |
| Recurring | None | Phase 15: 20+ recurring ops tasks |
| Conditionals | None | 6 conditional groups (competitors, realtime, payments, mobile, SEO, email) |
| Onboarding | None | Phase -1: existing project analysis (7 steps) |

**Why nested, not peers:**

1. **Phase 8's `feature_implementation_template`** has 31 steps per feature (feat.1 through feat.31) covering spec review → db migration → backend → frontend → tests → review → deploy. Steps feat.3 through feat.9 (backend model, service, endpoints, middleware, unit tests, integration tests) and feat.10-feat.22 (frontend types through accessibility) map directly to what `CodingPipeline` does.

2. **The right delegation model:** When the workflow engine hits a template step tagged for coding (feat.3-feat.22), it packages the step's instruction + artifact context into a `CodingPipeline` task. Pipeline handles the architect → implement → test → review → fix loop. Result flows back as the step's output artifact.

3. **Standalone pipeline unchanged:** Direct coding tasks ("fix this bug", "add a button") still go through `CodingPipeline` directly via the orchestrator's existing routing. Zero changes to current behavior.

4. **They should NOT run as parallel peers** because:
   - There's no scenario where you'd run a full product lifecycle AND a standalone pipeline for unrelated coding at the same time on the same goal
   - Pipeline is a low-level execution engine; Idea-to-Product is a high-level orchestration plan
   - Making them peers would require a "meta-orchestrator" above both — unnecessary complexity

### v2 Changes from v1

Key differences in v2 that affect implementation:

1. **Phase -1 (Existing Project Onboarding)** — 7 new steps for reverse-engineering existing codebases. Conditional on `existing_codebase_path` input.
2. **6 Conditional Groups** — Steps that are included/excluded based on artifact state (competitor count, realtime needs, payment model, mobile apps, SEO, email lists). Engine must evaluate conditions at runtime.
3. **Revision Policy** — Phase 8 can revise earlier artifacts (tech design, schema, API spec) via mini-ADRs at sprint boundaries.
4. **Onboarding Policy** — When working on existing repos: never rewrite unless asked, match existing patterns, use branch-and-PR strategy, require human approval for schema/dependency/architecture/deletion changes.
5. **Review Policy** — Max 3 review cycles per step, then escalate to `needs_clarification`.
6. **`context_strategy`** on templates — Primary vs reference vs full-only-if-needed artifact loading to manage context window.
7. **Many v2 steps reference v1** — Instructions like "Execute per v1 plan specification" for later phases. Engine must handle this by loading full instruction text, not just the reference.
8. **31-step template** (up from ~25 in v1) — Added feature-level review gates, responsive, accessibility, visual review steps.

### 24/7 Operation Model

The existing orchestrator runs 24/7. The workflow engine just needs to:
1. Load the JSON definition once at startup
2. On workflow start: expand non-template steps into tasks in the DB with proper `depends_on` chains
3. Evaluate conditional groups as their condition artifacts become available
4. Expand templates dynamically when Phase 8 backlog is initialized
5. Map Phase 15 recurring steps to `scheduled_tasks` with cron expressions
6. Let the orchestrator's existing `get_ready_tasks()` loop handle everything else

No new event loop, scheduler, or daemon needed — the orchestrator IS the engine.

---

## Task 1: Workflow Definition Loader (v2-aware)

**Files:**
- Create: `src/workflows/engine/__init__.py`
- Create: `src/workflows/engine/loader.py`
- Test: `tests/test_workflow_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_workflow_loader.py
import unittest
from pathlib import Path

class TestWorkflowLoader(unittest.TestCase):
    """Load and validate JSON workflow definitions."""

    def test_load_v2_workflow(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        self.assertEqual(wf.plan_id, "idea_to_product_v2")
        self.assertEqual(wf.version, "2.0")
        self.assertGreater(len(wf.phases), 0)
        self.assertGreater(len(wf.steps), 0)

    def test_workflow_has_17_phases(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        # Phase -1 through Phase 15 = 17 phases
        self.assertEqual(len(wf.phases), 17)

    def test_steps_have_required_fields(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        for step in wf.steps:
            self.assertIn("id", step)
            self.assertIn("agent", step)
            self.assertIn("instruction", step)
            self.assertIn("depends_on", step)

    def test_workflow_has_templates(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        self.assertGreater(len(wf.templates), 0)
        tmpl = wf.templates[0]
        self.assertEqual(tmpl["template_id"], "feature_implementation_template")
        self.assertEqual(len(tmpl["steps"]), 31)

    def test_workflow_has_conditional_groups(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        self.assertEqual(len(wf.conditional_groups), 6)
        group_ids = {g["group_id"] for g in wf.conditional_groups}
        self.assertIn("competitor_deep_dive", group_ids)
        self.assertIn("realtime_features", group_ids)
        self.assertIn("payment_flow", group_ids)

    def test_workflow_has_review_and_revision_policies(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        self.assertIn("review_policy", wf.metadata)
        self.assertIn("revision_policy", wf.metadata)
        self.assertEqual(wf.metadata["review_policy"]["max_review_cycles"], 3)

    def test_workflow_has_onboarding_policy(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        self.assertIn("onboarding_policy", wf.metadata)
        self.assertTrue(wf.metadata["onboarding_policy"]["never_rewrite_existing_code_unless_asked"])

    def test_load_nonexistent_raises(self):
        from src.workflows.engine.loader import load_workflow
        with self.assertRaises(FileNotFoundError):
            load_workflow("nonexistent_workflow")

    def test_dependency_graph_valid(self):
        from src.workflows.engine.loader import load_workflow, validate_dependencies
        wf = load_workflow("idea_to_product_v2")
        errors = validate_dependencies(wf)
        self.assertEqual(errors, [])

    def test_get_recurring_steps(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        recurring = [s for s in wf.steps if s.get("type") == "recurring"]
        self.assertGreater(len(recurring), 10)  # Phase 15 has 20+ recurring steps
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_loader.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/workflows/engine/__init__.py
"""Workflow engine — loads and executes JSON-defined workflows."""

# src/workflows/engine/loader.py
"""Load and validate JSON workflow definitions (v2-aware)."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).parent.parent  # src/workflows/


@dataclass
class WorkflowDefinition:
    """Parsed workflow definition with v2 features."""
    plan_id: str
    version: str
    metadata: dict
    phases: list[dict]
    steps: list[dict]
    templates: list[dict] = field(default_factory=list)
    conditional_groups: list[dict] = field(default_factory=list)

    def get_step(self, step_id: str) -> Optional[dict]:
        for s in self.steps:
            if s["id"] == step_id:
                return s
        return None

    def get_phase_steps(self, phase_id: str) -> list[dict]:
        return [s for s in self.steps if s.get("phase") == phase_id]

    def get_template(self, template_id: str) -> Optional[dict]:
        for t in self.templates:
            if t["template_id"] == template_id:
                return t
        return None

    def get_conditional_group(self, group_id: str) -> Optional[dict]:
        for g in self.conditional_groups:
            if g["group_id"] == group_id:
                return g
        return None

    def get_recurring_steps(self) -> list[dict]:
        return [s for s in self.steps if s.get("type") == "recurring"]

    def get_phase(self, phase_id: str) -> Optional[dict]:
        for p in self.phases:
            if p["id"] == phase_id:
                return p
        return None


def load_workflow(workflow_name: str) -> WorkflowDefinition:
    """Load a workflow JSON definition by name.

    Searches for <name>.json in workflow subdirectories.
    Handles both 'idea_to_product_v2' and 'idea_to_product' naming.
    """
    # Normalize: strip version suffix for directory lookup
    base_name = workflow_name
    for suffix in ("_v1", "_v2", "_v3"):
        base_name = base_name.replace(suffix, "")

    wf_dir = WORKFLOW_DIR / base_name
    candidates = []

    if wf_dir.exists():
        # Look for exact match first
        exact = wf_dir / f"{workflow_name}.json"
        if exact.exists():
            candidates = [exact]
        else:
            candidates = sorted(wf_dir.glob(f"{workflow_name}*.json"))

    if not candidates:
        # Search all subdirectories
        for subdir in WORKFLOW_DIR.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                found = list(subdir.glob(f"{workflow_name}*.json"))
                if found:
                    candidates = found
                    break

    if not candidates:
        raise FileNotFoundError(
            f"Workflow '{workflow_name}' not found in {WORKFLOW_DIR}"
        )

    path = candidates[0]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return WorkflowDefinition(
        plan_id=data["plan_id"],
        version=data.get("version", "1.0"),
        metadata=data.get("metadata", {}),
        phases=data.get("phases", []),
        steps=data.get("steps", []),
        templates=data.get("templates", []),
        conditional_groups=data.get("metadata", {}).get("conditional_groups", []),
    )


def validate_dependencies(wf: WorkflowDefinition) -> list[str]:
    """Validate that all depends_on references exist as step IDs."""
    step_ids = {s["id"] for s in wf.steps}
    # Also include fallback step IDs from conditional groups
    for group in wf.conditional_groups:
        for fb_step in group.get("fallback_steps", []):
            step_ids.add(fb_step["id"])

    errors = []
    for step in wf.steps:
        for dep_id in step.get("depends_on", []):
            if dep_id not in step_ids:
                errors.append(
                    f"Step {step['id']} depends on '{dep_id}' which does not exist"
                )
    return errors
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/ tests/test_workflow_loader.py
git commit -m "feat: add v2-aware workflow definition loader with conditional groups"
```

---

## Task 2: Artifact Store (Blackboard Extension)

**Files:**
- Create: `src/workflows/engine/artifacts.py`
- Test: `tests/test_workflow_artifacts.py`

The v2 workflow passes artifacts between steps. Each step reads `input_artifacts` from prior steps and produces `output_artifacts`. We extend the existing blackboard system to store these keyed by `(goal_id, artifact_name)`. The v2 template `context_strategy` (primary/reference/full_only_if_needed) controls how much artifact context is injected.

**Step 1: Write the failing test**

```python
# tests/test_workflow_artifacts.py
import unittest
import asyncio

class TestArtifactStore(unittest.TestCase):
    """Workflow artifact storage on top of blackboard."""

    def test_store_and_retrieve(self):
        from src.workflows.engine.artifacts import ArtifactStore
        store = ArtifactStore(use_db=False)
        asyncio.run(store.store("goal_1", "raw_idea_document", "The idea is..."))
        result = asyncio.run(store.retrieve("goal_1", "raw_idea_document"))
        self.assertEqual(result, "The idea is...")

    def test_retrieve_missing_returns_none(self):
        from src.workflows.engine.artifacts import ArtifactStore
        store = ArtifactStore(use_db=False)
        result = asyncio.run(store.retrieve("goal_1", "nonexistent"))
        self.assertIsNone(result)

    def test_collect_multiple(self):
        from src.workflows.engine.artifacts import ArtifactStore
        store = ArtifactStore(use_db=False)
        asyncio.run(store.store("goal_1", "doc_a", "Content A"))
        asyncio.run(store.store("goal_1", "doc_b", "Content B"))
        result = asyncio.run(
            store.collect("goal_1", ["doc_a", "doc_b", "doc_missing"])
        )
        self.assertEqual(result["doc_a"], "Content A")
        self.assertEqual(result["doc_b"], "Content B")
        self.assertIsNone(result["doc_missing"])

    def test_format_artifacts_for_prompt(self):
        from src.workflows.engine.artifacts import format_artifacts_for_prompt
        artifacts = {"doc_a": "Content A", "doc_b": "Content B"}
        prompt = format_artifacts_for_prompt(artifacts)
        self.assertIn("doc_a", prompt)
        self.assertIn("Content A", prompt)

    def test_format_with_context_strategy(self):
        from src.workflows.engine.artifacts import format_artifacts_for_prompt
        artifacts = {
            "implementation_context": "Primary content here " * 200,
            "openapi_spec": "Reference spec " * 200,
            "prd_final": "Full PRD " * 200,
        }
        strategy = {
            "primary": ["implementation_context"],
            "reference": ["openapi_spec"],
            "full_only_if_needed": ["prd_final"],
        }
        prompt = format_artifacts_for_prompt(artifacts, context_strategy=strategy)
        # Primary gets more space than reference
        self.assertIn("implementation_context", prompt)
        self.assertIn("openapi_spec", prompt)
        # full_only_if_needed should be summarized/truncated more aggressively
        self.assertIn("prd_final", prompt)

    def test_has_artifact(self):
        from src.workflows.engine.artifacts import ArtifactStore
        store = ArtifactStore(use_db=False)
        asyncio.run(store.store("goal_1", "doc_a", "Content"))
        self.assertTrue(asyncio.run(store.has("goal_1", "doc_a")))
        self.assertFalse(asyncio.run(store.has("goal_1", "doc_b")))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_artifacts.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/workflows/engine/artifacts.py
"""Artifact store for workflow step inputs/outputs.

Artifacts are stored on the goal's blackboard under the "artifacts" key.
In-memory cache for fast access. Supports v2 context_strategy for
tiered artifact loading (primary > reference > full_only_if_needed).
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Context budget allocation per strategy tier
CONTEXT_BUDGETS = {
    "primary": 8000,        # Full content, generous space
    "reference": 3000,      # Moderate truncation
    "full_only_if_needed": 1500,  # Aggressive truncation
    "default": 6000,        # No strategy specified
}


class ArtifactStore:
    """Store and retrieve artifacts for a workflow execution."""

    def __init__(self, use_db: bool = True):
        self._cache: dict[str, dict[str, str]] = {}
        self._use_db = use_db

    async def store(self, goal_id: str | int, name: str, value: str) -> None:
        """Store an artifact."""
        gid = str(goal_id)
        if gid not in self._cache:
            self._cache[gid] = {}
        self._cache[gid][name] = value

        if self._use_db:
            try:
                from src.collaboration.blackboard import update_blackboard_entry
                await update_blackboard_entry(int(goal_id), "artifacts", name, value)
            except Exception as e:
                logger.warning(f"Failed to persist artifact {name}: {e}")

    async def retrieve(self, goal_id: str | int, name: str) -> Optional[str]:
        """Retrieve a single artifact."""
        gid = str(goal_id)
        cached = self._cache.get(gid, {}).get(name)
        if cached is not None:
            return cached

        if self._use_db:
            try:
                from src.collaboration.blackboard import read_blackboard
                artifacts = await read_blackboard(int(goal_id), "artifacts")
                if artifacts and name in artifacts:
                    if gid not in self._cache:
                        self._cache[gid] = {}
                    self._cache[gid][name] = artifacts[name]
                    return artifacts[name]
            except Exception:
                pass
        return None

    async def has(self, goal_id: str | int, name: str) -> bool:
        """Check if an artifact exists."""
        return await self.retrieve(goal_id, name) is not None

    async def collect(
        self, goal_id: str | int, names: list[str]
    ) -> dict[str, Optional[str]]:
        """Collect multiple artifacts by name."""
        return {name: await self.retrieve(goal_id, name) for name in names}

    async def list_artifacts(self, goal_id: str | int) -> list[str]:
        """List all artifact names for a goal."""
        gid = str(goal_id)
        return list(self._cache.get(gid, {}).keys())


def format_artifacts_for_prompt(
    artifacts: dict[str, Optional[str]],
    context_strategy: Optional[dict] = None,
    max_total: int = 20000,
) -> str:
    """Format artifacts as context for an agent prompt.

    If context_strategy is provided (v2 template feature), allocates
    context budget per tier: primary > reference > full_only_if_needed.
    """
    if not any(v for v in artifacts.values() if v is not None):
        return ""

    sections = []

    if context_strategy:
        # Tiered loading
        for tier in ("primary", "reference", "full_only_if_needed"):
            tier_names = context_strategy.get(tier, [])
            budget = CONTEXT_BUDGETS.get(tier, CONTEXT_BUDGETS["default"])
            for name in tier_names:
                value = artifacts.get(name)
                if value is None:
                    continue
                truncated = value[:budget]
                if len(value) > budget:
                    truncated += f"\n... [{len(value)} chars total, truncated for {tier}]"
                sections.append(f"### {name}\n\n{truncated}")
        # Any artifacts not in strategy get default budget
        categorized = set()
        for tier_names in context_strategy.values():
            categorized.update(tier_names)
        for name, value in artifacts.items():
            if name in categorized or value is None:
                continue
            budget = CONTEXT_BUDGETS["default"]
            truncated = value[:budget]
            if len(value) > budget:
                truncated += f"\n... [truncated]"
            sections.append(f"### {name}\n\n{truncated}")
    else:
        # Flat loading with default budget
        budget = CONTEXT_BUDGETS["default"]
        for name, value in artifacts.items():
            if value is None:
                continue
            truncated = value[:budget]
            if len(value) > budget:
                truncated += f"\n... [truncated, {len(value)} chars total]"
            sections.append(f"### {name}\n\n{truncated}")

    combined = "\n\n---\n\n".join(sections)
    if len(combined) > max_total:
        combined = combined[:max_total] + "\n\n... [context truncated to fit]"
    return combined
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_artifacts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/artifacts.py tests/test_workflow_artifacts.py
git commit -m "feat: add artifact store with v2 context_strategy support"
```

---

## Task 3: Conditional Group Evaluator

**Files:**
- Create: `src/workflows/engine/conditions.py`
- Test: `tests/test_workflow_conditions.py`

v2 has 6 conditional groups that include/exclude steps based on artifact state. The engine must evaluate these at runtime when the condition artifact becomes available.

**Step 1: Write the failing test**

```python
# tests/test_workflow_conditions.py
import unittest
import asyncio

class TestConditionalGroups(unittest.TestCase):

    def test_evaluate_competitor_count_true(self):
        from src.workflows.engine.conditions import evaluate_condition
        # 3+ competitors → full research path
        artifact = '[{"name": "A"}, {"name": "B"}, {"name": "C"}]'
        result = evaluate_condition("length(competitors) >= 3", artifact)
        self.assertTrue(result)

    def test_evaluate_competitor_count_false(self):
        from src.workflows.engine.conditions import evaluate_condition
        artifact = '[{"name": "A"}]'
        result = evaluate_condition("length(competitors) >= 3", artifact)
        self.assertFalse(result)

    def test_evaluate_any_category(self):
        from src.workflows.engine.conditions import evaluate_condition
        artifact = '[{"category": "realtime"}, {"category": "crud"}]'
        result = evaluate_condition("any(req.category == 'realtime')", artifact)
        self.assertTrue(result)

    def test_evaluate_any_category_false(self):
        from src.workflows.engine.conditions import evaluate_condition
        artifact = '[{"category": "crud"}, {"category": "auth"}]'
        result = evaluate_condition("any(req.category == 'realtime')", artifact)
        self.assertFalse(result)

    def test_evaluate_pricing_model(self):
        from src.workflows.engine.conditions import evaluate_condition
        artifact = '{"pricing_model": "freemium"}'
        result = evaluate_condition("pricing_model != 'free'", artifact)
        self.assertTrue(result)

    def test_evaluate_pricing_model_free(self):
        from src.workflows.engine.conditions import evaluate_condition
        artifact = '{"pricing_model": "free"}'
        result = evaluate_condition("pricing_model != 'free'", artifact)
        self.assertFalse(result)

    def test_evaluate_platforms_include(self):
        from src.workflows.engine.conditions import evaluate_condition
        artifact = '{"platforms": ["web", "ios"]}'
        result = evaluate_condition(
            "platforms_include('ios') OR platforms_include('android')", artifact
        )
        self.assertTrue(result)

    def test_evaluate_boolean_field(self):
        from src.workflows.engine.conditions import evaluate_condition
        artifact = '{"has_public_web_pages": true}'
        result = evaluate_condition("has_public_web_pages == true", artifact)
        self.assertTrue(result)

    def test_resolve_conditional_group(self):
        from src.workflows.engine.conditions import resolve_group
        group = {
            "group_id": "payment_flow",
            "condition_check": "pricing_model != 'free'",
            "if_true": ["13.30", "14.4"],
            "if_false": [],
        }
        included, excluded = resolve_group(group, '{"pricing_model": "freemium"}')
        self.assertEqual(included, ["13.30", "14.4"])
        self.assertEqual(excluded, [])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_conditions.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/workflows/engine/conditions.py
"""Evaluate v2 conditional groups based on artifact state.

The condition expressions are a simple DSL:
- length(field) >= N
- any(item.field == 'value')
- field != 'value'
- field == true/false
- platforms_include('value')
- expr OR expr
"""
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _parse_artifact(artifact_str: str) -> Any:
    """Parse artifact value (JSON string or raw)."""
    if not artifact_str:
        return None
    try:
        return json.loads(artifact_str)
    except (json.JSONDecodeError, TypeError):
        return artifact_str


def evaluate_condition(condition_check: str, artifact_value: str) -> bool:
    """Evaluate a condition expression against an artifact value.

    Returns True if condition is met, False otherwise.
    On parse error, returns False (safe default: skip optional steps).
    """
    try:
        data = _parse_artifact(artifact_value)
        if data is None:
            return False

        # Handle OR expressions
        if " OR " in condition_check:
            parts = condition_check.split(" OR ")
            return any(evaluate_condition(p.strip(), artifact_value) for p in parts)

        # length(field) >= N
        m = re.match(r"length\((\w+)\)\s*>=\s*(\d+)", condition_check)
        if m:
            field, threshold = m.group(1), int(m.group(2))
            if isinstance(data, list):
                return len(data) >= threshold
            if isinstance(data, dict) and field in data:
                return len(data[field]) >= threshold
            return False

        # any(item.field == 'value')
        m = re.match(r"any\((\w+)\.(\w+)\s*==\s*'([^']+)'\)", condition_check)
        if m:
            _, field, value = m.group(1), m.group(2), m.group(3)
            if isinstance(data, list):
                return any(
                    item.get(field) == value for item in data
                    if isinstance(item, dict)
                )
            return False

        # field != 'value'
        m = re.match(r"(\w+)\s*!=\s*'([^']+)'", condition_check)
        if m:
            field, value = m.group(1), m.group(2)
            if isinstance(data, dict):
                return data.get(field) != value
            return True  # If not a dict, condition trivially holds

        # field == true/false
        m = re.match(r"(\w+)\s*==\s*(true|false)", condition_check)
        if m:
            field, value = m.group(1), m.group(2) == "true"
            if isinstance(data, dict):
                return data.get(field) == value
            return False

        # platforms_include('value')
        m = re.match(r"platforms_include\('([^']+)'\)", condition_check)
        if m:
            value = m.group(1)
            if isinstance(data, dict):
                platforms = data.get("platforms", [])
                return value in platforms
            if isinstance(data, list):
                return value in data
            return False

        # email_list_exists == true (simple boolean)
        m = re.match(r"(\w+)\s*==\s*true", condition_check)
        if m:
            field = m.group(1)
            if isinstance(data, dict):
                return bool(data.get(field))
            return False

        logger.warning(f"Unknown condition expression: {condition_check}")
        return False

    except Exception as e:
        logger.warning(f"Condition evaluation error for '{condition_check}': {e}")
        return False


def resolve_group(
    group: dict, artifact_value: str
) -> tuple[list[str], list[str]]:
    """Resolve a conditional group: returns (included_step_ids, excluded_step_ids)."""
    condition = group.get("condition_check", "")
    if_true = group.get("if_true", [])
    if_false = group.get("if_false", [])

    result = evaluate_condition(condition, artifact_value)

    if result:
        return if_true, if_false
    else:
        # if_false may contain fallback step IDs
        fallback_ids = [s["id"] for s in group.get("fallback_steps", [])]
        return if_false + fallback_ids, if_true
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_conditions.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/conditions.py tests/test_workflow_conditions.py
git commit -m "feat: add conditional group evaluator for v2 workflow"
```

---

## Task 4: Workflow Expander — Steps to DB Tasks (v2-aware)

**Files:**
- Create: `src/workflows/engine/expander.py`
- Test: `tests/test_workflow_expander.py`

Core piece: converts `WorkflowDefinition` steps into task dicts for DB insertion. Handles agent mapping, conditional Phase -1 inclusion, v2's step types (`recurring`), `may_need_clarification` flag, and template expansion with `context_strategy`.

**Step 1: Write the failing test**

```python
# tests/test_workflow_expander.py
import unittest

class TestWorkflowExpander(unittest.TestCase):

    def test_expand_single_phase(self):
        from src.workflows.engine.expander import expand_steps_to_tasks
        steps = [
            {
                "id": "0.1", "phase": "phase_0", "name": "raw_idea_intake",
                "agent": "writer", "depends_on": [],
                "instruction": "Document the idea.",
                "input_artifacts": [], "output_artifacts": ["raw_idea_document"],
            },
            {
                "id": "0.2", "phase": "phase_0", "name": "problem_extraction",
                "agent": "analyst", "depends_on": ["0.1"],
                "instruction": "Extract problem.",
                "input_artifacts": ["raw_idea_document"],
                "output_artifacts": ["problem_statement"],
            },
        ]
        tasks = expand_steps_to_tasks(steps, goal_id=1)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["depends_on_steps"], [])
        self.assertEqual(tasks[1]["depends_on_steps"], ["0.1"])
        self.assertEqual(tasks[0]["agent_type"], "writer")

    def test_expand_maps_v2_agents(self):
        from src.workflows.engine.expander import map_agent_type
        # v2 has agents that exist in system: analyst, summarizer, error_recovery
        self.assertEqual(map_agent_type("analyst"), "analyst")
        self.assertEqual(map_agent_type("summarizer"), "summarizer")
        self.assertEqual(map_agent_type("error_recovery"), "error_recovery")
        self.assertEqual(map_agent_type("coder"), "coder")
        # System agents fallback
        self.assertEqual(map_agent_type("router"), "executor")
        self.assertEqual(map_agent_type("assistant"), "assistant")

    def test_expand_preserves_step_type(self):
        from src.workflows.engine.expander import expand_steps_to_tasks
        steps = [{
            "id": "15.1", "phase": "phase_15", "name": "error_monitoring",
            "agent": "error_recovery", "depends_on": ["14.5"],
            "instruction": "Monitor errors.",
            "input_artifacts": [], "output_artifacts": ["error_log"],
            "type": "recurring", "trigger": "Daily",
        }]
        tasks = expand_steps_to_tasks(steps, goal_id=1)
        self.assertEqual(tasks[0]["context"]["step_type"], "recurring")
        self.assertEqual(tasks[0]["context"]["trigger"], "Daily")

    def test_expand_preserves_may_need_clarification(self):
        from src.workflows.engine.expander import expand_steps_to_tasks
        steps = [{
            "id": "0.5", "phase": "phase_0", "name": "human_clarification",
            "agent": "planner", "depends_on": ["0.4"],
            "instruction": "Ask human.", "may_need_clarification": True,
            "input_artifacts": [], "output_artifacts": ["answers"],
        }]
        tasks = expand_steps_to_tasks(steps, goal_id=1)
        self.assertTrue(tasks[0]["context"]["may_need_clarification"])

    def test_expand_template_produces_concrete_steps(self):
        from src.workflows.engine.expander import expand_template
        template = {
            "template_id": "feature_implementation_template",
            "parameters": {"feature_id": "", "feature_name": ""},
            "context_strategy": {
                "primary": ["implementation_context"],
                "reference": ["openapi_spec"],
            },
            "steps": [
                {
                    "template_step_id": "feat.1",
                    "name": "feature_spec_review",
                    "agent": "planner",
                    "instruction": "Review spec for '{feature_name}'.",
                    "output_artifacts": ["feature_spec_summary"],
                },
                {
                    "template_step_id": "feat.2",
                    "name": "feature_impl_plan",
                    "agent": "planner",
                    "instruction": "Plan impl for '{feature_name}'.",
                    "output_artifacts": ["feature_impl_plan"],
                },
            ],
        }
        params = {"feature_id": "F-001", "feature_name": "User Auth"}
        concrete = expand_template(template, params, prefix="8.F-001")
        self.assertEqual(len(concrete), 2)
        self.assertEqual(concrete[0]["id"], "8.F-001.feat.1")
        self.assertIn("User Auth", concrete[0]["instruction"])
        # context_strategy propagated
        self.assertIn("context_strategy", concrete[0])

    def test_expand_template_with_conditions(self):
        from src.workflows.engine.expander import expand_template
        template = {
            "template_id": "test",
            "parameters": {"feature_name": ""},
            "steps": [
                {
                    "template_step_id": "feat.3",
                    "name": "db_migration",
                    "agent": "implementer",
                    "condition": "Only if feature_impl_plan includes db_migration",
                    "instruction": "Create migration for {feature_name}.",
                    "output_artifacts": ["migration_file"],
                },
            ],
        }
        concrete = expand_template(template, {"feature_name": "Auth"}, prefix="8.F-001")
        self.assertIn("condition", concrete[0])

    def test_phase_minus1_excluded_without_codebase(self):
        from src.workflows.engine.expander import filter_steps_for_context
        steps = [
            {"id": "-1.1", "phase": "phase_-1", "name": "discovery",
             "agent": "analyst", "depends_on": [], "instruction": "Scan code.",
             "input_artifacts": ["existing_codebase_path"], "output_artifacts": ["inventory"]},
            {"id": "0.1", "phase": "phase_0", "name": "intake",
             "agent": "writer", "depends_on": [], "instruction": "Doc idea.",
             "input_artifacts": [], "output_artifacts": ["raw_idea"]},
        ]
        # No codebase → Phase -1 excluded
        filtered = filter_steps_for_context(steps, has_existing_codebase=False)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["id"], "0.1")

        # With codebase → all included
        filtered = filter_steps_for_context(steps, has_existing_codebase=True)
        self.assertEqual(len(filtered), 2)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_expander.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/workflows/engine/expander.py
"""Expand workflow definitions into concrete tasks for the orchestrator.

Handles v2 features: Phase -1 conditional inclusion, recurring step types,
template expansion with context_strategy, and agent name mapping.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Agent name mapping: workflow JSON agent name → system agent type
# v2 defines 16 agents. Most map 1:1 to our system agents.
# System agents (router, assistant) need special handling.
AGENT_MAP = {
    "router": "executor",   # System agent, not assigned directly
    # All others pass through: planner, architect, coder, implementer,
    # fixer, test_generator, reviewer, researcher, writer, analyst,
    # executor, error_recovery, visual_reviewer, summarizer, assistant
}


def map_agent_type(agent_name: str) -> str:
    """Map workflow agent names to system agent types."""
    if agent_name in AGENT_MAP:
        return AGENT_MAP[agent_name]
    return agent_name


def filter_steps_for_context(
    steps: list[dict],
    has_existing_codebase: bool = False,
) -> list[dict]:
    """Filter steps based on execution context.

    Phase -1 (Existing Project Onboarding) only runs if existing_codebase_path is provided.
    """
    if has_existing_codebase:
        return steps
    return [s for s in steps if s.get("phase") != "phase_-1"]


def expand_steps_to_tasks(
    steps: list[dict],
    goal_id: int,
    initial_context: Optional[dict] = None,
) -> list[dict]:
    """Convert workflow steps into task dicts for DB insertion.

    Each task dict contains: title, description, agent_type,
    depends_on_steps (step IDs, resolved to task IDs later),
    context, priority, tier.
    """
    tasks = []
    for step in steps:
        step_id = step["id"]
        agent = map_agent_type(step.get("agent", "executor"))
        instruction = step.get("instruction", "")
        input_artifacts = step.get("input_artifacts", [])
        output_artifacts = step.get("output_artifacts", [])
        phase = step.get("phase", "")
        name = step.get("name", step_id)
        may_need_clarification = step.get("may_need_clarification", False)
        condition = step.get("condition")
        step_type = step.get("type")
        trigger = step.get("trigger")
        done_when = step.get("done_when", "")

        ctx: dict[str, Any] = {
            "workflow_step_id": step_id,
            "workflow_phase": phase,
            "input_artifacts": input_artifacts,
            "output_artifacts": output_artifacts,
            "may_need_clarification": may_need_clarification,
            "is_workflow_step": True,
        }
        if condition:
            ctx["condition"] = condition
        if step_type:
            ctx["step_type"] = step_type
        if trigger:
            ctx["trigger"] = trigger
        if done_when:
            ctx["done_when"] = done_when
        if initial_context:
            ctx["workflow_context"] = initial_context

        # Priority: earlier phases get higher priority
        phase_num = 5
        try:
            phase_str = phase.replace("phase_", "")
            phase_num = int(phase_str) if phase_str not in ("", "-1") else (
                -1 if phase_str == "-1" else 5
            )
        except (ValueError, AttributeError):
            pass
        # Phase -1 = priority 10 (highest), Phase 0 = 10, Phase 15 = 1
        priority = max(1, min(10, 10 - phase_num))

        task = {
            "title": f"[{step_id}] {name}",
            "description": instruction,
            "agent_type": agent,
            "goal_id": goal_id,
            "depends_on_steps": step.get("depends_on", []),
            "context": ctx,
            "priority": priority,
            "tier": "auto",
        }
        tasks.append(task)

    return tasks


def expand_template(
    template: dict,
    params: dict[str, str],
    prefix: str = "",
) -> list[dict]:
    """Expand a template into concrete steps with parameter substitution.

    Template steps get IDs: {prefix}.{template_step_id}
    {param_name} placeholders in instructions are replaced.
    context_strategy is propagated to each step.
    """
    context_strategy = template.get("context_strategy")
    context_artifacts = template.get("context_artifacts", [])
    concrete_steps = []

    for tmpl_step in template.get("steps", []):
        step_id = tmpl_step["template_step_id"]
        full_id = f"{prefix}.{step_id}" if prefix else step_id

        instruction = tmpl_step.get("instruction", "")
        for key, value in params.items():
            instruction = instruction.replace(f"{{{key}}}", str(value))

        step: dict[str, Any] = {
            "id": full_id,
            "phase": "phase_8",
            "name": tmpl_step.get("name", step_id),
            "agent": tmpl_step.get("agent", "executor"),
            "depends_on": [],  # Resolved after all features expanded
            "instruction": instruction,
            "input_artifacts": tmpl_step.get("input_artifacts", context_artifacts),
            "output_artifacts": tmpl_step.get("output_artifacts", []),
            "may_need_clarification": tmpl_step.get("may_need_clarification", False),
        }
        if tmpl_step.get("condition"):
            step["condition"] = tmpl_step["condition"]
        if context_strategy:
            step["context_strategy"] = context_strategy

        concrete_steps.append(step)

    return concrete_steps
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_expander.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/expander.py tests/test_workflow_expander.py
git commit -m "feat: add v2-aware workflow expander with conditionals and templates"
```

---

## Task 5: Workflow Runner — DB Integration and Lifecycle

**Files:**
- Create: `src/workflows/engine/runner.py`
- Test: `tests/test_workflow_runner.py`

Ties everything together: loads definition → creates goal → filters steps → expands to tasks → inserts into DB. Handles Phase -1 conditional inclusion and stores initial inputs as artifacts.

**Step 1: Write the failing test**

```python
# tests/test_workflow_runner.py
import unittest

class TestWorkflowRunner(unittest.TestCase):

    def test_resolve_dependencies(self):
        from src.workflows.engine.runner import resolve_dependencies
        mapping = {"0.1": 101, "0.2": 102, "0.3": 103}
        deps = resolve_dependencies(["0.1", "0.3"], mapping)
        self.assertEqual(deps, [101, 103])

    def test_resolve_skips_missing(self):
        from src.workflows.engine.runner import resolve_dependencies
        mapping = {"0.1": 101}
        deps = resolve_dependencies(["0.1", "0.99"], mapping)
        self.assertEqual(deps, [101])

    def test_build_step_description(self):
        from src.workflows.engine.runner import build_step_description
        desc = build_step_description(
            instruction="Do the thing.",
            input_artifacts=["doc_a", "doc_b"],
            artifact_contents={"doc_a": "Content A", "doc_b": None},
        )
        self.assertIn("Do the thing.", desc)
        self.assertIn("doc_a", desc)
        self.assertIn("Content A", desc)

    def test_build_step_description_with_done_when(self):
        from src.workflows.engine.runner import build_step_description
        desc = build_step_description(
            instruction="Do the thing.",
            input_artifacts=[],
            artifact_contents={},
            done_when="Output exists and is complete.",
        )
        self.assertIn("Do the thing.", desc)
        self.assertIn("Output exists and is complete.", desc)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_runner.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/workflows/engine/runner.py
"""Workflow runner — creates goals and tasks from workflow definitions.

Usage:
    runner = WorkflowRunner()
    goal_id = await runner.start(
        "idea_to_product_v2",
        initial_input={"raw_idea": "Build a ..."},
    )
"""
import json
import logging
from typing import Any, Optional

from .loader import load_workflow, WorkflowDefinition
from .expander import (
    expand_steps_to_tasks,
    expand_template,
    filter_steps_for_context,
)
from .artifacts import ArtifactStore, format_artifacts_for_prompt

logger = logging.getLogger(__name__)


def resolve_dependencies(
    step_dep_ids: list[str],
    step_to_task_map: dict[str, int],
) -> list[int]:
    """Convert workflow step IDs to DB task IDs."""
    task_ids = []
    for sid in step_dep_ids:
        if sid in step_to_task_map:
            task_ids.append(step_to_task_map[sid])
        else:
            logger.warning(f"Dependency step '{sid}' not in task map, skipping")
    return task_ids


def build_step_description(
    instruction: str,
    input_artifacts: list[str],
    artifact_contents: dict[str, Optional[str]],
    done_when: str = "",
) -> str:
    """Build the full task description with instruction + artifact context + done_when."""
    parts = [instruction]

    available = {k: v for k, v in artifact_contents.items() if v is not None}
    if available:
        parts.append("\n\n## Input Artifacts\n")
        parts.append(format_artifacts_for_prompt(available))

    missing = [k for k in input_artifacts if artifact_contents.get(k) is None]
    if missing:
        parts.append(f"\n\nNote: These artifacts are not yet available: {missing}")

    if done_when:
        parts.append(f"\n\n## Done When\n{done_when}")

    return "\n".join(parts)


class WorkflowRunner:
    """Starts and manages workflow executions."""

    def __init__(self):
        self.artifact_store = ArtifactStore(use_db=True)

    async def start(
        self,
        workflow_name: str,
        initial_input: Optional[dict[str, str]] = None,
        title: Optional[str] = None,
        existing_codebase_path: Optional[str] = None,
    ) -> int:
        """Start a workflow execution.

        Creates a goal, filters steps (Phase -1 conditional),
        expands to tasks, and inserts into DB.
        Returns the goal_id.
        """
        from src.infra.db import add_goal, add_task

        wf = load_workflow(workflow_name)

        goal_title = title or f"Workflow: {wf.plan_id}"
        goal_desc = wf.metadata.get("description", "")
        goal_id = await add_goal(
            title=goal_title,
            description=goal_desc,
            priority=8,
            context={
                "workflow_id": wf.plan_id,
                "workflow_version": wf.version,
                "has_existing_codebase": existing_codebase_path is not None,
            },
        )

        # Store initial inputs as artifacts
        if initial_input:
            for name, value in initial_input.items():
                await self.artifact_store.store(goal_id, name, value)

        if existing_codebase_path:
            await self.artifact_store.store(
                goal_id, "existing_codebase_path", existing_codebase_path
            )

        # Filter steps: exclude Phase -1 if no existing codebase
        has_codebase = existing_codebase_path is not None
        filtered_steps = filter_steps_for_context(wf.steps, has_codebase)

        # Exclude recurring steps from initial expansion
        # (they'll be registered as scheduled tasks separately)
        non_recurring = [s for s in filtered_steps if s.get("type") != "recurring"]
        recurring = [s for s in filtered_steps if s.get("type") == "recurring"]

        # Expand non-recurring steps into tasks
        tasks = expand_steps_to_tasks(
            non_recurring,
            goal_id=goal_id,
            initial_context={"workflow_id": wf.plan_id},
        )

        # Insert tasks and build step_id → task_id mapping
        step_to_task: dict[str, int] = {}
        for task_dict in tasks:
            step_id = task_dict["context"]["workflow_step_id"]
            step_deps = task_dict.pop("depends_on_steps")
            resolved_deps = resolve_dependencies(step_deps, step_to_task)

            task_id = await add_task(
                title=task_dict["title"],
                description=task_dict["description"],
                goal_id=goal_id,
                agent_type=task_dict["agent_type"],
                tier=task_dict.get("tier", "auto"),
                priority=task_dict.get("priority", 5),
                depends_on=resolved_deps if resolved_deps else None,
                context=task_dict["context"],
            )

            if task_id:
                step_to_task[step_id] = task_id

        # Register recurring steps as scheduled tasks
        if recurring:
            await self._register_recurring_steps(
                recurring, goal_id, wf.plan_id, step_to_task
            )

        # Store workflow metadata for conditional evaluation later
        await self.artifact_store.store(
            goal_id, "_workflow_metadata",
            json.dumps({
                "conditional_groups": wf.conditional_groups,
                "templates": [t["template_id"] for t in wf.templates],
                "step_to_task": step_to_task,
            }),
        )

        logger.info(
            f"Workflow '{wf.plan_id}' started: goal_id={goal_id}, "
            f"tasks={len(step_to_task)}, recurring={len(recurring)}"
        )
        return goal_id

    async def _register_recurring_steps(
        self,
        recurring_steps: list[dict],
        goal_id: int,
        workflow_id: str,
        step_to_task: dict[str, int],
    ) -> None:
        """Register recurring steps as scheduled tasks.

        Maps trigger descriptions to cron expressions:
        - "Daily" → "0 9 * * *"
        - "Weekly" → "0 9 * * 1"
        - "Continuous during first week" → "0 */4 * * *" (every 4h)
        - "Triggered by alerts" → no cron, manual trigger
        """
        from src.infra.db import add_scheduled_task

        TRIGGER_TO_CRON = {
            "daily": "0 9 * * *",
            "weekly": "0 9 * * 1",
            "continuous": "0 */4 * * *",
        }

        for step in recurring_steps:
            trigger = step.get("trigger", "").lower()
            cron = None
            for key, expr in TRIGGER_TO_CRON.items():
                if key in trigger:
                    cron = expr
                    break

            if cron:
                try:
                    await add_scheduled_task(
                        name=f"wf_{workflow_id}_{step['id']}",
                        description=step.get("name", step["id"]),
                        cron_expression=cron,
                        task_template={
                            "title": f"[{step['id']}] {step.get('name', '')}",
                            "description": step.get("instruction", ""),
                            "agent_type": step.get("agent", "executor"),
                            "goal_id": goal_id,
                            "context": {
                                "is_workflow_step": True,
                                "workflow_step_id": step["id"],
                                "workflow_phase": step.get("phase", ""),
                                "input_artifacts": step.get("input_artifacts", []),
                                "output_artifacts": step.get("output_artifacts", []),
                                "step_type": "recurring",
                            },
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to register recurring step {step['id']}: {e}")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/runner.py tests/test_workflow_runner.py
git commit -m "feat: add workflow runner with Phase -1 filtering and recurring step registration"
```

---

## Task 6: Workflow Execution Hooks in Orchestrator

**Files:**
- Create: `src/workflows/engine/hooks.py`
- Modify: `src/core/orchestrator.py` (add pre/post hooks for workflow steps)
- Test: `tests/test_workflow_hooks.py`

When the orchestrator executes a task with `is_workflow_step: true`, it must:
- **Pre-execution:** Fetch input artifacts, inject into task description, apply `context_strategy` if present
- **Post-execution:** Store agent output as output artifacts, evaluate conditional groups if condition artifact just became available, trigger template expansion when implementation backlog completes
- **Clarification handling:** If `may_need_clarification` is true, the step may set `needs_clarification` status → Telegram flow handles it (existing system)
- **Review policy:** Track review cycles per step, escalate after 3 failures

**Step 1: Write the failing test**

```python
# tests/test_workflow_hooks.py
import unittest

class TestWorkflowHooks(unittest.TestCase):

    def test_is_workflow_step(self):
        from src.workflows.engine.hooks import is_workflow_step
        self.assertTrue(is_workflow_step({"is_workflow_step": True}))
        self.assertFalse(is_workflow_step({}))
        self.assertFalse(is_workflow_step({"is_workflow_step": False}))

    def test_enrich_task_description(self):
        from src.workflows.engine.hooks import enrich_task_description
        task = {
            "description": "Do the analysis.",
            "context": {
                "is_workflow_step": True,
                "input_artifacts": ["doc_a"],
                "output_artifacts": ["result"],
            },
        }
        enriched = enrich_task_description(
            task, {"doc_a": "Some document content"}
        )
        self.assertIn("Do the analysis.", enriched)
        self.assertIn("Some document content", enriched)

    def test_enrich_with_context_strategy(self):
        from src.workflows.engine.hooks import enrich_task_description
        task = {
            "description": "Implement feature.",
            "context": {
                "is_workflow_step": True,
                "input_artifacts": ["impl_ctx", "api_spec"],
                "context_strategy": {
                    "primary": ["impl_ctx"],
                    "reference": ["api_spec"],
                },
            },
        }
        enriched = enrich_task_description(
            task,
            {"impl_ctx": "Primary content", "api_spec": "Reference spec"},
        )
        self.assertIn("Primary content", enriched)
        self.assertIn("Reference spec", enriched)

    def test_enrich_with_done_when(self):
        from src.workflows.engine.hooks import enrich_task_description
        task = {
            "description": "Build the thing.",
            "context": {
                "is_workflow_step": True,
                "input_artifacts": [],
                "done_when": "Output exists.",
            },
        }
        enriched = enrich_task_description(task, {})
        self.assertIn("Done When", enriched)
        self.assertIn("Output exists.", enriched)

    def test_extract_output_artifact_names(self):
        from src.workflows.engine.hooks import extract_output_artifact_names
        ctx = {"output_artifacts": ["raw_idea_document", "summary"]}
        self.assertEqual(
            extract_output_artifact_names(ctx),
            ["raw_idea_document", "summary"],
        )

    def test_should_delegate_to_pipeline(self):
        from src.workflows.engine.hooks import should_delegate_to_pipeline
        # Backend implementation steps delegate to pipeline
        self.assertTrue(should_delegate_to_pipeline("feat.3", "implementer"))
        self.assertTrue(should_delegate_to_pipeline("feat.5", "implementer"))
        self.assertTrue(should_delegate_to_pipeline("feat.13", "implementer"))
        # Test and review steps don't
        self.assertFalse(should_delegate_to_pipeline("feat.1", "planner"))
        self.assertFalse(should_delegate_to_pipeline("feat.8", "test_generator"))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_hooks.py -v`
Expected: FAIL

**Step 3: Write hooks module**

```python
# src/workflows/engine/hooks.py
"""Pre/post execution hooks for workflow steps in the orchestrator.

Handles artifact injection, output storage, conditional evaluation,
template expansion triggers, and CodingPipeline delegation.
"""
import json
import logging
from typing import Any, Optional

from .artifacts import ArtifactStore, format_artifacts_for_prompt

logger = logging.getLogger(__name__)

_artifact_store: Optional[ArtifactStore] = None

# Template steps that should delegate to CodingPipeline
# These are the coding-heavy steps in the feature_implementation_template
PIPELINE_DELEGATE_STEPS = {
    "feat.3", "feat.4", "feat.5", "feat.6", "feat.7",   # backend
    "feat.10", "feat.11", "feat.12", "feat.13", "feat.14",  # frontend core
    "feat.15", "feat.16", "feat.17", "feat.18",  # frontend polish
}
PIPELINE_DELEGATE_AGENTS = {"implementer", "coder"}


def get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(use_db=True)
    return _artifact_store


def is_workflow_step(context: dict) -> bool:
    return bool(context.get("is_workflow_step"))


def extract_output_artifact_names(context: dict) -> list[str]:
    return context.get("output_artifacts", [])


def should_delegate_to_pipeline(
    template_step_id: str, agent_type: str
) -> bool:
    """Check if a template step should be delegated to CodingPipeline."""
    step_base = template_step_id.split(".")[-1] if "." in template_step_id else template_step_id
    full_id = f"feat.{step_base}" if not step_base.startswith("feat.") else step_base
    return (
        full_id in PIPELINE_DELEGATE_STEPS
        and agent_type in PIPELINE_DELEGATE_AGENTS
    )


def enrich_task_description(
    task: dict,
    artifact_contents: dict[str, Optional[str]],
) -> str:
    """Enrich task description with artifact context and done_when."""
    instruction = task.get("description", "")
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        ctx = json.loads(ctx)

    context_strategy = ctx.get("context_strategy")
    done_when = ctx.get("done_when", "")

    # Inject artifacts
    available = {k: v for k, v in artifact_contents.items() if v is not None}
    if available:
        artifact_section = format_artifacts_for_prompt(
            available, context_strategy=context_strategy
        )
        if artifact_section:
            instruction = f"{instruction}\n\n## Input Artifacts\n\n{artifact_section}"

    # Inject done_when criteria
    if done_when:
        instruction = f"{instruction}\n\n## Done When\n{done_when}"

    return instruction


async def pre_execute_workflow_step(task: dict) -> dict:
    """Pre-execution hook: fetch artifacts and enrich task."""
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        ctx = json.loads(ctx)

    if not is_workflow_step(ctx):
        return task

    goal_id = task.get("goal_id")
    input_names = ctx.get("input_artifacts", [])

    if not goal_id or not input_names:
        # Still inject done_when even without artifacts
        if ctx.get("done_when"):
            task["description"] = enrich_task_description(task, {})
        return task

    store = get_artifact_store()
    contents = await store.collect(goal_id, input_names)
    task["description"] = enrich_task_description(task, contents)
    return task


async def post_execute_workflow_step(
    task: dict, result: dict
) -> None:
    """Post-execution hook: store output artifacts and trigger downstream logic."""
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        ctx = json.loads(ctx)

    if not is_workflow_step(ctx):
        return

    goal_id = task.get("goal_id")
    output_names = ctx.get("output_artifacts", [])
    result_text = result.get("result", "")

    if not goal_id or not output_names or not result_text:
        return

    store = get_artifact_store()

    # Store output artifacts
    if len(output_names) == 1:
        await store.store(goal_id, output_names[0], result_text)
    else:
        # Store full result under each artifact name
        # Agents should structure output with headers for multi-artifact steps
        for name in output_names:
            await store.store(goal_id, name, result_text)

    logger.info(
        f"Stored artifacts {output_names} for step "
        f"{ctx.get('workflow_step_id', '?')}"
    )

    # Check if this step's output triggers conditional group evaluation
    await _check_conditional_triggers(goal_id, output_names, store)

    # Check if this is the implementation_backlog step → trigger template expansion
    step_id = ctx.get("workflow_step_id", "")
    if step_id == "8.0" and "implementation_backlog" in output_names:
        await _trigger_template_expansion(goal_id, result_text, store)


async def _check_conditional_triggers(
    goal_id: int,
    artifact_names: list[str],
    store: ArtifactStore,
) -> None:
    """Check if a newly stored artifact triggers conditional group evaluation."""
    try:
        meta_str = await store.retrieve(goal_id, "_workflow_metadata")
        if not meta_str:
            return
        meta = json.loads(meta_str)

        from .conditions import resolve_group
        from .loader import load_workflow

        for group in meta.get("conditional_groups", []):
            condition_artifact = group.get("condition_artifact", "")
            if condition_artifact not in artifact_names:
                continue

            artifact_value = await store.retrieve(goal_id, condition_artifact)
            if not artifact_value:
                continue

            included, excluded = resolve_group(group, artifact_value)

            logger.info(
                f"Conditional group '{group['group_id']}' resolved: "
                f"include={included}, exclude={excluded}"
            )

            # TODO: Create tasks for fallback steps if needed,
            # and cancel/skip excluded steps
            # This requires DB operations and is wired in Task 8

    except Exception as e:
        logger.warning(f"Conditional trigger check failed: {e}")


async def _trigger_template_expansion(
    goal_id: int,
    backlog_text: str,
    store: ArtifactStore,
) -> None:
    """When implementation_backlog is produced, expand feature templates."""
    try:
        # Parse backlog (JSON list of features)
        backlog = json.loads(backlog_text)
        if not isinstance(backlog, list):
            logger.warning("implementation_backlog is not a list, skipping expansion")
            return

        from .loader import load_workflow
        from .expander import expand_template, expand_steps_to_tasks
        from .runner import resolve_dependencies
        from src.infra.db import add_task

        # Load workflow to get template
        meta_str = await store.retrieve(goal_id, "_workflow_metadata")
        meta = json.loads(meta_str) if meta_str else {}
        step_to_task = meta.get("step_to_task", {})

        # We need the full workflow definition for the template
        wf = load_workflow("idea_to_product_v2")
        template = wf.get_template("feature_implementation_template")
        if not template:
            logger.error("feature_implementation_template not found")
            return

        for feature in backlog:
            feature_id = feature.get("feature_id", "unknown")
            feature_name = feature.get("feature_name", "Unknown Feature")
            params = {
                "feature_id": feature_id,
                "feature_name": feature_name,
                "user_story_ids": str(feature.get("user_story_ids", [])),
                "epic_id": str(feature.get("epic_id", "")),
                "sprint_id": str(feature.get("sprint_id", "")),
            }

            prefix = f"8.{feature_id}"
            concrete_steps = expand_template(template, params, prefix=prefix)
            tasks = expand_steps_to_tasks(concrete_steps, goal_id=goal_id)

            # Insert tasks with dependency on 8.0
            backlog_task_id = step_to_task.get("8.0")
            for task_dict in tasks:
                step_id = task_dict["context"]["workflow_step_id"]
                step_deps = task_dict.pop("depends_on_steps")
                resolved = resolve_dependencies(step_deps, step_to_task)
                if backlog_task_id and not resolved:
                    resolved = [backlog_task_id]

                task_id = await add_task(
                    title=task_dict["title"],
                    description=task_dict["description"],
                    goal_id=goal_id,
                    agent_type=task_dict["agent_type"],
                    tier="auto",
                    priority=task_dict.get("priority", 5),
                    depends_on=resolved if resolved else None,
                    context=task_dict["context"],
                )
                if task_id:
                    step_to_task[step_id] = task_id

        # Update metadata with new step_to_task mapping
        meta["step_to_task"] = step_to_task
        await store.store(goal_id, "_workflow_metadata", json.dumps(meta))

        logger.info(
            f"Expanded {len(backlog)} features into template tasks for goal {goal_id}"
        )

    except Exception as e:
        logger.error(f"Template expansion failed: {e}")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_hooks.py -v`
Expected: PASS

**Step 5: Wire hooks into orchestrator.py**

In `src/core/orchestrator.py`, modify `process_task()`:

1. Before agent execution (after context injection, ~line 637):

```python
# Workflow step pre-hook: inject artifact context
from ..workflows.engine.hooks import (
    pre_execute_workflow_step,
    post_execute_workflow_step,
    is_workflow_step,
    should_delegate_to_pipeline,
)

task_ctx = task.get("context", {})
if isinstance(task_ctx, str):
    task_ctx = json.loads(task_ctx)

if is_workflow_step(task_ctx):
    task = await pre_execute_workflow_step(task)

    # Check if this template step should delegate to CodingPipeline
    wf_step_id = task_ctx.get("workflow_step_id", "")
    if should_delegate_to_pipeline(wf_step_id, agent_type):
        agent_type = "pipeline"
```

2. After successful execution (~line 674), before `_handle_complete`:

```python
if status == "completed":
    task_ctx_post = task.get("context", {})
    if isinstance(task_ctx_post, str):
        task_ctx_post = json.loads(task_ctx_post)
    if is_workflow_step(task_ctx_post):
        await post_execute_workflow_step(task, result)
```

3. Add timeout for workflow steps in `AGENT_TIMEOUTS`:

```python
"workflow": 900,  # 15 min — workflow steps can be lengthy (research, analysis)
```

**Step 6: Commit**

```bash
git add src/workflows/engine/hooks.py tests/test_workflow_hooks.py src/core/orchestrator.py
git commit -m "feat: add workflow execution hooks with pipeline delegation and conditional triggers"
```

---

## Task 7: Workflow Dispatch — Telegram and API Integration

**Files:**
- Create: `src/workflows/engine/dispatch.py`
- Modify: `src/app/telegram_bot.py` (add `/product` command)
- Test: `tests/test_workflow_dispatch.py`

Users trigger workflows via Telegram: `/product Build me a task management app` or natural language detection. Also supports existing project onboarding: `/product onboard /path/to/repo`.

**Step 1: Write the failing test**

```python
# tests/test_workflow_dispatch.py
import unittest

class TestWorkflowDispatch(unittest.TestCase):

    def test_workflow_keywords_classify(self):
        from src.workflows.engine.dispatch import should_start_workflow
        self.assertTrue(should_start_workflow("Build me a product that does X"))
        self.assertTrue(should_start_workflow("I have an idea for a SaaS app"))
        self.assertTrue(should_start_workflow("Create an MVP for task management"))
        self.assertFalse(should_start_workflow("Fix the bug in auth module"))
        self.assertFalse(should_start_workflow("Add a logout button"))

    def test_detect_onboarding(self):
        from src.workflows.engine.dispatch import detect_onboarding_path
        path = detect_onboarding_path("onboard /home/user/myproject")
        self.assertEqual(path, "/home/user/myproject")
        path = detect_onboarding_path("Build me a new app")
        self.assertIsNone(path)

    def test_extract_idea_text(self):
        from src.workflows.engine.dispatch import extract_idea_text
        text = extract_idea_text("/product Build me a task management app")
        self.assertEqual(text, "Build me a task management app")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_dispatch.py -v`
Expected: FAIL

**Step 3: Write dispatch module**

```python
# src/workflows/engine/dispatch.py
"""Dispatch logic for starting workflows from user messages."""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

WORKFLOW_PATTERNS = [
    r"\b(build|create|make|develop)\s+(me\s+)?(a\s+)?(product|app|application|saas|platform|tool|service|website|startup)\b",
    r"\bidea\s+(for|to|about)\b.*\b(app|product|saas|platform|service)\b",
    r"\bidea.to.product\b",
    r"\bfull\s+product\b",
    r"\bfrom\s+scratch\b.*\b(app|product|platform)\b",
    r"\bmvp\b.*\b(build|create|develop)\b",
    r"\blaunch\b.*\b(product|app|startup)\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in WORKFLOW_PATTERNS]


def should_start_workflow(message: str) -> bool:
    """Check if a user message should trigger a workflow."""
    return any(p.search(message) for p in _compiled)


def detect_onboarding_path(message: str) -> Optional[str]:
    """Detect if the message requests onboarding an existing project."""
    m = re.match(r"onboard\s+(.+)", message.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_idea_text(message: str) -> str:
    """Extract the idea text from a /product command."""
    # Strip the /product prefix if present
    text = re.sub(r"^/product\s*", "", message.strip())
    return text.strip()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_dispatch.py -v`
Expected: PASS

**Step 5: Add /product command to telegram_bot.py**

In `src/app/telegram_bot.py`, add a new command handler:

```python
async def cmd_product(update, context):
    """Start an idea-to-product workflow."""
    message_text = " ".join(context.args) if context.args else ""
    if not message_text:
        await update.message.reply_text(
            "Usage:\n"
            "/product <your idea description>\n"
            "/product onboard /path/to/existing/project"
        )
        return

    from src.workflows.engine.dispatch import detect_onboarding_path, extract_idea_text
    from src.workflows.engine.runner import WorkflowRunner

    runner = WorkflowRunner()
    onboard_path = detect_onboarding_path(message_text)

    if onboard_path:
        goal_id = await runner.start(
            "idea_to_product_v2",
            initial_input={},
            title=f"Onboard: {onboard_path}",
            existing_codebase_path=onboard_path,
        )
        await update.message.reply_text(
            f"🏗️ Started project onboarding workflow (goal #{goal_id})\n"
            f"Analyzing existing codebase at: {onboard_path}"
        )
    else:
        idea = extract_idea_text(message_text)
        goal_id = await runner.start(
            "idea_to_product_v2",
            initial_input={"raw_idea": idea},
            title=f"Product: {idea[:80]}",
        )
        await update.message.reply_text(
            f"🚀 Started idea-to-product workflow (goal #{goal_id})\n"
            f"Phase 0: Idea Capture & Clarification starting..."
        )

# Register in the handler setup:
# app.add_handler(CommandHandler("product", cmd_product))
```

**Step 6: Commit**

```bash
git add src/workflows/engine/dispatch.py tests/test_workflow_dispatch.py src/app/telegram_bot.py
git commit -m "feat: add /product command and workflow dispatch for idea-to-product v2"
```

---

## Task 8: Pipeline Delegation Bridge

**Files:**
- Create: `src/workflows/engine/pipeline_bridge.py`
- Test: `tests/test_pipeline_bridge.py`

When the workflow engine encounters a template step tagged for pipeline delegation (feat.3-feat.22 with implementer/coder agent), it packages the step's context into a `CodingPipeline` task. This bridges the high-level workflow with the low-level coding pipeline.

**Step 1: Write the failing test**

```python
# tests/test_pipeline_bridge.py
import unittest

class TestPipelineBridge(unittest.TestCase):

    def test_build_pipeline_task_from_workflow_step(self):
        from src.workflows.engine.pipeline_bridge import build_pipeline_task
        task = build_pipeline_task(
            step_title="[8.F-001.feat.5] backend_service",
            step_instruction="Implement business logic service layer.",
            goal_id=42,
            feature_name="User Auth",
            artifact_context="OpenAPI spec: POST /auth/login ...",
        )
        self.assertEqual(task["goal_id"], 42)
        self.assertIn("Implement business logic", task["description"])
        self.assertIn("User Auth", task["description"])
        self.assertIn("OpenAPI spec", task["description"])
        # Pipeline mode should be 'feature' for template steps
        ctx = task["context"]
        self.assertEqual(ctx["pipeline_mode"], "feature")

    def test_extract_feature_context_from_step(self):
        from src.workflows.engine.pipeline_bridge import extract_feature_context
        step_ctx = {
            "workflow_step_id": "8.F-001.feat.5",
            "workflow_context": {"workflow_id": "idea_to_product_v2"},
        }
        feature_id, feature_name = extract_feature_context(step_ctx)
        self.assertEqual(feature_id, "F-001")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_bridge.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/workflows/engine/pipeline_bridge.py
"""Bridge between workflow template steps and CodingPipeline.

When a feature_implementation_template step is tagged for pipeline delegation,
this module packages the step's context into a CodingPipeline-compatible task.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def extract_feature_context(step_context: dict) -> tuple[str, str]:
    """Extract feature_id and feature_name from step context.

    Step IDs follow pattern: 8.<feature_id>.feat.<N>
    """
    step_id = step_context.get("workflow_step_id", "")
    # Parse: "8.F-001.feat.5" → feature_id = "F-001"
    m = re.match(r"8\.([^.]+)\.feat\.\d+", step_id)
    feature_id = m.group(1) if m else "unknown"

    # Feature name from workflow context or step title
    feature_name = step_context.get("workflow_context", {}).get(
        "feature_name", feature_id
    )
    return feature_id, feature_name


def build_pipeline_task(
    step_title: str,
    step_instruction: str,
    goal_id: int,
    feature_name: str,
    artifact_context: str = "",
) -> dict:
    """Build a CodingPipeline-compatible task from a workflow step."""
    description = f"Feature: {feature_name}\n\n{step_instruction}"
    if artifact_context:
        description += f"\n\n## Context\n{artifact_context}"

    return {
        "title": step_title,
        "description": description,
        "goal_id": goal_id,
        "context": {
            "pipeline_mode": "feature",
            "prefer_quality": True,
        },
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_bridge.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/pipeline_bridge.py tests/test_pipeline_bridge.py
git commit -m "feat: add pipeline bridge for workflow template step delegation"
```

---

## Task 9: Workflow Status Tracking and Telegram Notifications

**Files:**
- Create: `src/workflows/engine/status.py`
- Modify: `src/app/telegram_bot.py` (add `/wfstatus` command)
- Test: `tests/test_workflow_status.py`

Track workflow progress and notify user via Telegram at phase transitions.

**Step 1: Write the failing test**

```python
# tests/test_workflow_status.py
import unittest
import asyncio

class TestWorkflowStatus(unittest.TestCase):

    def test_compute_phase_progress(self):
        from src.workflows.engine.status import compute_phase_progress
        tasks = [
            {"context": {"workflow_phase": "phase_0"}, "status": "completed"},
            {"context": {"workflow_phase": "phase_0"}, "status": "completed"},
            {"context": {"workflow_phase": "phase_0"}, "status": "pending"},
            {"context": {"workflow_phase": "phase_1"}, "status": "pending"},
        ]
        progress = compute_phase_progress(tasks)
        self.assertEqual(progress["phase_0"]["completed"], 2)
        self.assertEqual(progress["phase_0"]["total"], 3)
        self.assertEqual(progress["phase_1"]["completed"], 0)

    def test_format_status_message(self):
        from src.workflows.engine.status import format_status_message
        progress = {
            "phase_0": {"completed": 8, "total": 8, "name": "Idea Capture"},
            "phase_1": {"completed": 3, "total": 12, "name": "Market Research"},
        }
        msg = format_status_message("idea_to_product_v2", 42, progress)
        self.assertIn("phase_0", msg.lower() or msg)
        self.assertIn("42", msg)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_status.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/workflows/engine/status.py
"""Workflow progress tracking and status reporting."""
import json
import logging
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

PHASE_NAMES = {
    "phase_-1": "Existing Project Onboarding",
    "phase_0": "Idea Capture & Clarification",
    "phase_1": "Market & Competitive Research",
    "phase_2": "Product Strategy & Definition",
    "phase_3": "Requirements Engineering",
    "phase_4": "Architecture & Technical Design",
    "phase_5": "UX/UI Design Specification",
    "phase_6": "Project Planning & Sprint Setup",
    "phase_7": "Development Environment Setup",
    "phase_8": "Core Implementation",
    "phase_9": "Comprehensive Testing",
    "phase_10": "Security Hardening",
    "phase_11": "Documentation",
    "phase_12": "Legal & Compliance",
    "phase_13": "Pre-Launch Preparation",
    "phase_14": "Launch",
    "phase_15": "Post-Launch Operations",
}


def compute_phase_progress(tasks: list[dict]) -> dict:
    """Compute per-phase progress from task list."""
    phases: dict[str, dict] = defaultdict(
        lambda: {"completed": 0, "total": 0, "name": ""}
    )

    for task in tasks:
        ctx = task.get("context", {})
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                continue

        phase = ctx.get("workflow_phase", "")
        if not phase:
            continue

        phases[phase]["total"] += 1
        phases[phase]["name"] = PHASE_NAMES.get(phase, phase)
        if task.get("status") == "completed":
            phases[phase]["completed"] += 1

    return dict(phases)


def format_status_message(
    workflow_id: str,
    goal_id: int,
    progress: dict,
) -> str:
    """Format a human-readable status message."""
    lines = [f"📊 Workflow Status: {workflow_id} (goal #{goal_id})\n"]

    # Sort phases by ID
    sorted_phases = sorted(
        progress.items(),
        key=lambda x: int(x[0].replace("phase_", "").replace("-", ""))
        if x[0].replace("phase_", "").replace("-", "").lstrip("-").isdigit()
        else 99,
    )

    for phase_id, data in sorted_phases:
        total = data["total"]
        completed = data["completed"]
        name = data.get("name", phase_id)
        pct = (completed / total * 100) if total > 0 else 0

        if completed == total and total > 0:
            icon = "✅"
        elif completed > 0:
            icon = "🔄"
        else:
            icon = "⬜"

        bar_len = 10
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines.append(f"{icon} {name}: {bar} {completed}/{total}")

    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_status.py -v`
Expected: PASS

**Step 5: Add /wfstatus command to telegram_bot.py**

```python
async def cmd_wfstatus(update, context):
    """Show workflow progress."""
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /wfstatus <goal_id>")
        return

    goal_id = int(args[0])
    from src.infra.db import get_tasks_for_goal
    from src.workflows.engine.status import compute_phase_progress, format_status_message

    tasks = await get_tasks_for_goal(goal_id)
    if not tasks:
        await update.message.reply_text(f"No tasks found for goal #{goal_id}")
        return

    # Detect workflow_id from first task's context
    first_ctx = tasks[0].get("context", {})
    if isinstance(first_ctx, str):
        first_ctx = json.loads(first_ctx)
    workflow_id = first_ctx.get("workflow_context", {}).get("workflow_id", "unknown")

    progress = compute_phase_progress(tasks)
    msg = format_status_message(workflow_id, goal_id, progress)
    await update.message.reply_text(msg)

# Register: app.add_handler(CommandHandler("wfstatus", cmd_wfstatus))
```

**Step 6: Commit**

```bash
git add src/workflows/engine/status.py tests/test_workflow_status.py src/app/telegram_bot.py
git commit -m "feat: add workflow status tracking and /wfstatus command"
```

---

## Task 10: Review Policy and Revision Policy Implementation

**Files:**
- Create: `src/workflows/engine/policies.py`
- Test: `tests/test_workflow_policies.py`

v2's `review_policy` (max 3 review cycles, then escalate) and `revision_policy` (Phase 8 can revise earlier artifacts via mini-ADRs) need enforcement in the hooks.

**Step 1: Write the failing test**

```python
# tests/test_workflow_policies.py
import unittest

class TestWorkflowPolicies(unittest.TestCase):

    def test_review_cycle_tracking(self):
        from src.workflows.engine.policies import ReviewTracker
        tracker = ReviewTracker(max_cycles=3)
        self.assertFalse(tracker.should_escalate("step_0.7"))
        tracker.record_failure("step_0.7")
        tracker.record_failure("step_0.7")
        self.assertFalse(tracker.should_escalate("step_0.7"))
        tracker.record_failure("step_0.7")
        self.assertTrue(tracker.should_escalate("step_0.7"))

    def test_escalation_message(self):
        from src.workflows.engine.policies import ReviewTracker
        tracker = ReviewTracker(max_cycles=3)
        for _ in range(3):
            tracker.record_failure("step_0.7")
        msg = tracker.escalation_message("step_0.7", ["issue1", "issue2"])
        self.assertIn("step_0.7", msg)
        self.assertIn("3", msg)
        self.assertIn("issue1", msg)

    def test_onboarding_policy_checks(self):
        from src.workflows.engine.policies import check_onboarding_policy
        # Schema change requires approval
        self.assertTrue(
            check_onboarding_policy("database_schema_changes", is_existing_project=True)
        )
        # Greenfield doesn't need approval
        self.assertFalse(
            check_onboarding_policy("database_schema_changes", is_existing_project=False)
        )
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_policies.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/workflows/engine/policies.py
"""v2 workflow policies: review cycles, revision, and onboarding."""
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

APPROVAL_REQUIRED_ACTIONS = {
    "database_schema_changes",
    "dependency_major_upgrades",
    "architecture_pattern_changes",
    "deletion_of_existing_code",
}


class ReviewTracker:
    """Track review cycles per step and enforce max_review_cycles."""

    def __init__(self, max_cycles: int = 3):
        self.max_cycles = max_cycles
        self._failures: dict[str, int] = defaultdict(int)

    def record_failure(self, step_id: str) -> None:
        self._failures[step_id] += 1

    def should_escalate(self, step_id: str) -> bool:
        return self._failures[step_id] >= self.max_cycles

    def escalation_message(
        self, step_id: str, issues: list[str]
    ) -> str:
        issues_str = "; ".join(issues)
        return (
            f"Review cycle for step '{step_id}' has failed "
            f"{self.max_cycles} times. Issues: {issues_str}. "
            f"Human decision needed: fix manually, accept as-is, "
            f"or provide direction."
        )

    def get_cycle_count(self, step_id: str) -> int:
        return self._failures[step_id]


def check_onboarding_policy(
    action_type: str,
    is_existing_project: bool = False,
) -> bool:
    """Check if an action requires human approval under onboarding policy.

    Returns True if approval is needed.
    """
    if not is_existing_project:
        return False
    return action_type in APPROVAL_REQUIRED_ACTIONS
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_policies.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/policies.py tests/test_workflow_policies.py
git commit -m "feat: add review policy and onboarding policy enforcement"
```

---

## Task 11: Integration Test — End-to-End Workflow Start

**Files:**
- Test: `tests/test_workflow_e2e.py`

Verify the complete flow: load v2 definition → create goal → expand steps → insert tasks → verify dependency chain → verify Phase -1 exclusion for greenfield.

**Step 1: Write the integration test**

```python
# tests/test_workflow_e2e.py
import unittest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

class TestWorkflowE2E(unittest.TestCase):
    """End-to-end workflow start integration test."""

    @patch("src.infra.db.add_goal", new_callable=AsyncMock)
    @patch("src.infra.db.add_task", new_callable=AsyncMock)
    def test_greenfield_workflow_start(self, mock_add_task, mock_add_goal):
        """Starting a greenfield workflow excludes Phase -1 and creates tasks."""
        mock_add_goal.return_value = 1
        mock_add_task.side_effect = lambda **kw: len(mock_add_task.call_args_list)

        from src.workflows.engine.runner import WorkflowRunner
        runner = WorkflowRunner()
        runner.artifact_store._use_db = False

        goal_id = asyncio.run(runner.start(
            "idea_to_product_v2",
            initial_input={"raw_idea": "Build a task management app"},
            title="Test Product",
        ))

        self.assertEqual(goal_id, 1)
        mock_add_goal.assert_called_once()

        # Verify tasks were created (Phase -1 excluded, so no -1.x steps)
        created_titles = [
            call.kwargs.get("title", "") or call.args[0] if call.args else ""
            for call in mock_add_task.call_args_list
        ]
        # No Phase -1 steps
        self.assertFalse(
            any("-1." in t for t in created_titles),
            "Phase -1 steps should be excluded for greenfield"
        )
        # Phase 0 steps should exist
        self.assertTrue(
            any("[0.1]" in str(call) for call in mock_add_task.call_args_list),
            "Phase 0 steps should be present"
        )

    @patch("src.infra.db.add_goal", new_callable=AsyncMock)
    @patch("src.infra.db.add_task", new_callable=AsyncMock)
    def test_onboarding_workflow_includes_phase_minus1(
        self, mock_add_task, mock_add_goal
    ):
        """Starting with existing codebase includes Phase -1."""
        mock_add_goal.return_value = 2
        mock_add_task.side_effect = lambda **kw: len(mock_add_task.call_args_list)

        from src.workflows.engine.runner import WorkflowRunner
        runner = WorkflowRunner()
        runner.artifact_store._use_db = False

        goal_id = asyncio.run(runner.start(
            "idea_to_product_v2",
            initial_input={},
            title="Onboard Existing",
            existing_codebase_path="/home/user/myproject",
        ))

        self.assertEqual(goal_id, 2)
        # Phase -1 steps should exist
        self.assertTrue(
            any("[-1.1]" in str(call) for call in mock_add_task.call_args_list),
            "Phase -1 steps should be present for onboarding"
        )

    def test_loader_plus_expander_step_count(self):
        """Verify step count matches v2 JSON."""
        from src.workflows.engine.loader import load_workflow
        from src.workflows.engine.expander import filter_steps_for_context

        wf = load_workflow("idea_to_product_v2")
        # All steps
        all_steps = wf.steps
        self.assertGreater(len(all_steps), 100)

        # Greenfield (no Phase -1)
        greenfield = filter_steps_for_context(all_steps, has_existing_codebase=False)
        phase_minus1_count = len([s for s in all_steps if s.get("phase") == "phase_-1"])
        self.assertEqual(len(greenfield), len(all_steps) - phase_minus1_count)

    def test_conditional_group_evaluation(self):
        """Verify conditional groups resolve correctly."""
        from src.workflows.engine.conditions import evaluate_condition

        # Competitor deep dive: 3+ competitors
        self.assertTrue(evaluate_condition(
            "length(competitors) >= 3",
            '[{"name":"A"},{"name":"B"},{"name":"C"}]',
        ))
        self.assertFalse(evaluate_condition(
            "length(competitors) >= 3",
            '[{"name":"A"}]',
        ))

        # Payment flow: not free
        self.assertTrue(evaluate_condition(
            "pricing_model != 'free'",
            '{"pricing_model": "freemium"}',
        ))
        self.assertFalse(evaluate_condition(
            "pricing_model != 'free'",
            '{"pricing_model": "free"}',
        ))
```

**Step 2: Run test**

Run: `python -m pytest tests/test_workflow_e2e.py -v`
Expected: PASS (after all prior tasks are implemented)

**Step 3: Commit**

```bash
git add tests/test_workflow_e2e.py
git commit -m "test: add end-to-end workflow integration tests"
```

---

## Summary: File Inventory

### New Files (10)

| File | Purpose |
|------|---------|
| `src/workflows/engine/__init__.py` | Package init |
| `src/workflows/engine/loader.py` | Load + validate v2 JSON definitions |
| `src/workflows/engine/artifacts.py` | Artifact store with context_strategy |
| `src/workflows/engine/conditions.py` | Conditional group evaluator |
| `src/workflows/engine/expander.py` | Steps → task dicts with template expansion |
| `src/workflows/engine/runner.py` | DB integration: goal + task creation |
| `src/workflows/engine/hooks.py` | Pre/post execution hooks for orchestrator |
| `src/workflows/engine/dispatch.py` | Telegram /product command dispatch |
| `src/workflows/engine/pipeline_bridge.py` | CodingPipeline delegation for template steps |
| `src/workflows/engine/status.py` | Progress tracking + /wfstatus command |
| `src/workflows/engine/policies.py` | Review + onboarding policy enforcement |

### Modified Files (2)

| File | Changes |
|------|---------|
| `src/core/orchestrator.py` | Pre/post hooks for workflow steps, pipeline delegation, timeout |
| `src/app/telegram_bot.py` | `/product` and `/wfstatus` commands |

### Test Files (9)

| File | Tests |
|------|-------|
| `tests/test_workflow_loader.py` | Loading, validation, v2 features |
| `tests/test_workflow_artifacts.py` | Store, retrieve, context_strategy |
| `tests/test_workflow_conditions.py` | All 6 conditional group types |
| `tests/test_workflow_expander.py` | Step expansion, template, Phase -1 |
| `tests/test_workflow_runner.py` | Dependency resolution, description building |
| `tests/test_workflow_hooks.py` | Pre/post hooks, pipeline delegation |
| `tests/test_workflow_dispatch.py` | Keyword detection, onboarding |
| `tests/test_pipeline_bridge.py` | Pipeline task packaging |
| `tests/test_workflow_status.py` | Phase progress, status formatting |
| `tests/test_workflow_policies.py` | Review tracking, onboarding policy |
| `tests/test_workflow_e2e.py` | End-to-end integration |

---

## Architectural Decisions

### 1. CodingPipeline lives INSIDE Idea-to-Product (nested, not competing)
- Phase 8 template steps (feat.3-feat.22) delegate to CodingPipeline
- Pipeline handles architect → implement → test → review → fix → commit
- Standalone coding tasks bypass workflows entirely (unchanged behavior)

### 2. Orchestrator IS the engine (no new daemon)
- Workflow steps become regular tasks in the DB
- Existing `get_ready_tasks()` → `claim_task()` → `execute` loop runs them
- Hooks inject artifacts pre-execution and store outputs post-execution

### 3. Conditional groups evaluated lazily
- When a condition artifact is stored, the post-hook evaluates all groups watching that artifact
- Included steps get their tasks created; excluded steps are skipped

### 4. Phase 15 recurring steps use existing scheduled_tasks
- Mapped to cron expressions based on trigger descriptions
- Alert-triggered steps are manual-only (no cron)

### 5. Phase -1 is conditional on input
- `existing_codebase_path` presence toggles Phase -1 inclusion
- Onboarding policy (branch-and-PR, approval requirements) injected into agent context
