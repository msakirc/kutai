# Idea-to-Product Workflow Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate the `idea_to_product_v1.json` workflow definition as a first-class workflow alongside the existing `CodingPipeline`, creating a generic workflow engine that can run any JSON-defined workflow 24/7 through the existing orchestrator.

**Architecture:** Build a `WorkflowEngine` that loads JSON workflow definitions, expands them into goals+tasks with proper dependency chains in the DB, and lets the existing orchestrator poll/execute them like any other task. The CodingPipeline stays as a specialized fast-path for coding-only tasks. Idea-to-Product becomes a high-level workflow that can *invoke* the CodingPipeline for its Phase 8 implementation steps. An artifact store (blackboard-backed) passes outputs between steps. Template expansion creates concrete tasks from reusable templates.

**Tech Stack:** Python 3.12, aiosqlite, asyncio, JSON workflow definitions

---

## Analysis: Pipeline vs Idea-to-Product Relationship

### Key Differences

| Dimension | CodingPipeline | Idea-to-Product |
|-----------|---------------|-----------------|
| Scope | Single coding task (feature/bugfix) | Entire product lifecycle (idea → launch → ops) |
| Duration | Minutes to hours | Days to weeks |
| Steps | 5-8 hardcoded stages | 150+ JSON-defined steps across 16 phases |
| State | In-memory PipelineContext | Needs persistent artifact store |
| Dependencies | Linear/sequential stages | Complex DAG with parallelism |
| Templates | None | Feature implementation template (25 steps) |
| Human-in-loop | None (auto review loop) | `may_need_clarification` on ~15 steps |
| Recurring steps | None | Phase 15 has recurring ops tasks |

### Relationship Decision: Nested, Not Competing

The CodingPipeline should work **within** Idea-to-Product, not alongside it as peers:

- **Idea-to-Product Phase 8** (Core Implementation) uses a `feature_implementation_template` with 28 steps per feature — this overlaps heavily with what CodingPipeline already does (architect → implement → test → review → fix → commit)
- The right design: Phase 8's template steps delegate to CodingPipeline for the actual coding work, while the workflow engine handles the broader lifecycle (research, design, planning, testing, launch, ops)
- For standalone coding tasks (user sends "fix this bug"), the orchestrator routes to CodingPipeline directly (current behavior, unchanged)
- For "build me a product from this idea", the orchestrator starts an Idea-to-Product workflow, which eventually spawns CodingPipeline runs for each feature

### 24/7 Operation Model

The existing orchestrator already runs 24/7 polling for ready tasks. The workflow engine just needs to:
1. Load the JSON definition once
2. Expand it into goals+tasks in the DB with proper `depends_on`
3. Let the orchestrator's existing `get_ready_tasks()` → `claim_task()` → `execute` loop handle everything
4. Store artifacts on the blackboard so downstream steps can read upstream outputs

No new event loop or scheduler needed — the orchestrator IS the engine.

---

## Task 1: Workflow Definition Loader

**Files:**
- Create: `src/workflows/engine/__init__.py`
- Create: `src/workflows/engine/loader.py`
- Test: `tests/test_workflow_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_workflow_loader.py
import unittest
import json
from pathlib import Path

class TestWorkflowLoader(unittest.TestCase):
    """Load and validate JSON workflow definitions."""

    def test_load_workflow_definition(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v1")
        self.assertEqual(wf.plan_id, "idea_to_product_v1")
        self.assertEqual(wf.version, "1.0")
        self.assertGreater(len(wf.phases), 0)
        self.assertGreater(len(wf.steps), 0)

    def test_workflow_has_phases_and_steps(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v1")
        # 16 phases
        self.assertEqual(len(wf.phases), 16)
        # Steps have required fields
        for step in wf.steps:
            self.assertIn("id", step)
            self.assertIn("agent", step)
            self.assertIn("instruction", step)
            self.assertIn("depends_on", step)

    def test_workflow_has_templates(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v1")
        self.assertGreater(len(wf.templates), 0)
        tmpl = wf.templates[0]
        self.assertEqual(tmpl["template_id"], "feature_implementation_template")
        self.assertGreater(len(tmpl["steps"]), 0)

    def test_load_nonexistent_workflow_raises(self):
        from src.workflows.engine.loader import load_workflow
        with self.assertRaises(FileNotFoundError):
            load_workflow("nonexistent_workflow")

    def test_dependency_graph_is_valid(self):
        from src.workflows.engine.loader import load_workflow, validate_dependencies
        wf = load_workflow("idea_to_product_v1")
        # All depends_on references exist as step IDs
        errors = validate_dependencies(wf)
        self.assertEqual(errors, [])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_loader.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/workflows/engine/__init__.py
# Workflow engine package

# src/workflows/engine/loader.py
"""Load and validate JSON workflow definitions."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKFLOW_DIR = Path(__file__).parent.parent  # src/workflows/


@dataclass
class WorkflowDefinition:
    """Parsed workflow definition."""
    plan_id: str
    version: str
    metadata: dict
    phases: list[dict]
    steps: list[dict]
    templates: list[dict] = field(default_factory=list)

    def get_step(self, step_id: str) -> dict | None:
        for s in self.steps:
            if s["id"] == step_id:
                return s
        return None

    def get_phase_steps(self, phase_id: str) -> list[dict]:
        return [s for s in self.steps if s.get("phase") == phase_id]

    def get_template(self, template_id: str) -> dict | None:
        for t in self.templates:
            if t["template_id"] == template_id:
                return t
        return None


def load_workflow(workflow_name: str) -> WorkflowDefinition:
    """Load a workflow JSON definition by name.

    Searches for <name>/<name>*.json in the workflows directory.
    """
    wf_dir = WORKFLOW_DIR / workflow_name.replace("_v1", "").replace("_v2", "")
    # Try exact match first, then glob
    candidates = list(wf_dir.glob(f"{workflow_name}*.json")) if wf_dir.exists() else []

    if not candidates:
        # Try subdirectories
        for subdir in WORKFLOW_DIR.iterdir():
            if subdir.is_dir():
                candidates = list(subdir.glob(f"{workflow_name}*.json"))
                if candidates:
                    break

    if not candidates:
        raise FileNotFoundError(f"Workflow '{workflow_name}' not found in {WORKFLOW_DIR}")

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
    )


def validate_dependencies(wf: WorkflowDefinition) -> list[str]:
    """Validate that all depends_on references exist as step IDs."""
    step_ids = {s["id"] for s in wf.steps}
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
git commit -m "feat: add workflow definition loader for JSON workflows"
```

---

## Task 2: Artifact Store (Blackboard Extension)

**Files:**
- Create: `src/workflows/engine/artifacts.py`
- Test: `tests/test_workflow_artifacts.py`

The idea-to-product workflow passes artifacts between steps (e.g., step 0.1 produces `raw_idea_document`, step 0.2 consumes it). We extend the existing blackboard system to store these artifacts keyed by `(goal_id, artifact_name)`.

**Step 1: Write the failing test**

```python
# tests/test_workflow_artifacts.py
import unittest
import asyncio

class TestArtifactStore(unittest.TestCase):
    """Workflow artifact storage on top of blackboard."""

    def test_store_and_retrieve_artifact(self):
        from src.workflows.engine.artifacts import ArtifactStore
        store = ArtifactStore.__new__(ArtifactStore)
        store._cache = {}
        # In-memory test
        asyncio.run(store.store("goal_1", "raw_idea_document", "The idea is..."))
        result = asyncio.run(store.retrieve("goal_1", "raw_idea_document"))
        self.assertEqual(result, "The idea is...")

    def test_retrieve_missing_artifact_returns_none(self):
        from src.workflows.engine.artifacts import ArtifactStore
        store = ArtifactStore.__new__(ArtifactStore)
        store._cache = {}
        result = asyncio.run(store.retrieve("goal_1", "nonexistent"))
        self.assertIsNone(result)

    def test_collect_artifacts_for_step(self):
        from src.workflows.engine.artifacts import ArtifactStore
        store = ArtifactStore.__new__(ArtifactStore)
        store._cache = {}
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
        self.assertIn("doc_b", prompt)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_artifacts.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/workflows/engine/artifacts.py
"""Artifact store for workflow step inputs/outputs.

Artifacts are stored on the goal's blackboard under the "artifacts" key.
In-memory cache for fast access during workflow execution.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Max artifact size in prompt context (truncate large artifacts)
MAX_ARTIFACT_PROMPT_CHARS = 6000


class ArtifactStore:
    """Store and retrieve artifacts for a workflow execution.

    Backed by the blackboard system for persistence,
    with an in-memory cache for fast reads.
    """

    def __init__(self, use_db: bool = True):
        self._cache: dict[str, dict[str, str]] = {}  # {goal_id: {name: value}}
        self._use_db = use_db

    async def store(
        self, goal_id: str | int, name: str, value: str
    ) -> None:
        """Store an artifact."""
        gid = str(goal_id)
        if gid not in self._cache:
            self._cache[gid] = {}
        self._cache[gid][name] = value

        if self._use_db:
            try:
                from src.collaboration.blackboard import (
                    update_blackboard_entry,
                )
                await update_blackboard_entry(
                    int(goal_id), "artifacts", name, value
                )
            except Exception as e:
                logger.warning(f"Failed to persist artifact {name}: {e}")

    async def retrieve(
        self, goal_id: str | int, name: str
    ) -> Optional[str]:
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
                    # Populate cache
                    if gid not in self._cache:
                        self._cache[gid] = {}
                    self._cache[gid][name] = artifacts[name]
                    return artifacts[name]
            except Exception:
                pass

        return None

    async def collect(
        self, goal_id: str | int, names: list[str]
    ) -> dict[str, Optional[str]]:
        """Collect multiple artifacts by name."""
        result = {}
        for name in names:
            result[name] = await self.retrieve(goal_id, name)
        return result


def format_artifacts_for_prompt(
    artifacts: dict[str, Optional[str]],
    max_per_artifact: int = MAX_ARTIFACT_PROMPT_CHARS,
) -> str:
    """Format artifacts as context for an agent prompt."""
    sections = []
    for name, value in artifacts.items():
        if value is None:
            continue
        truncated = value[:max_per_artifact]
        if len(value) > max_per_artifact:
            truncated += f"\n... [truncated, {len(value)} chars total]"
        sections.append(f"### {name}\n\n{truncated}")
    return "\n\n---\n\n".join(sections)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_artifacts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/artifacts.py tests/test_workflow_artifacts.py
git commit -m "feat: add artifact store for workflow step I/O"
```

---

## Task 3: Workflow Expander — Convert JSON Steps to DB Tasks

**Files:**
- Create: `src/workflows/engine/expander.py`
- Test: `tests/test_workflow_expander.py`

This is the core piece: takes a `WorkflowDefinition` + user input, creates a goal and tasks in the DB with proper `depends_on` chains. The orchestrator's existing polling loop will pick them up.

**Step 1: Write the failing test**

```python
# tests/test_workflow_expander.py
import unittest
import asyncio

class TestWorkflowExpander(unittest.TestCase):
    """Expand workflow definition into DB tasks."""

    def test_expand_single_phase(self):
        from src.workflows.engine.expander import expand_steps_to_tasks
        steps = [
            {
                "id": "0.1", "phase": "phase_0", "name": "raw_idea_intake",
                "agent": "writer", "depends_on": [],
                "instruction": "Document the idea.", "input_artifacts": [],
                "output_artifacts": ["raw_idea_document"],
            },
            {
                "id": "0.2", "phase": "phase_0", "name": "problem_extraction",
                "agent": "analyst", "depends_on": ["0.1"],
                "instruction": "Extract problem.", "input_artifacts": ["raw_idea_document"],
                "output_artifacts": ["problem_statement"],
            },
        ]
        tasks = expand_steps_to_tasks(steps, goal_id=1)
        self.assertEqual(len(tasks), 2)
        # First task has no depends_on
        self.assertEqual(tasks[0]["depends_on_steps"], [])
        # Second task depends on first
        self.assertEqual(tasks[1]["depends_on_steps"], ["0.1"])
        # Agent types mapped
        self.assertEqual(tasks[0]["agent_type"], "writer")
        self.assertEqual(tasks[1]["agent_type"], "analyst")

    def test_expand_preserves_instruction_and_artifacts(self):
        from src.workflows.engine.expander import expand_steps_to_tasks
        steps = [
            {
                "id": "0.1", "phase": "phase_0", "name": "test_step",
                "agent": "writer", "depends_on": [],
                "instruction": "Do the thing.",
                "input_artifacts": ["input_a"],
                "output_artifacts": ["output_b"],
            },
        ]
        tasks = expand_steps_to_tasks(steps, goal_id=1)
        ctx = tasks[0]["context"]
        self.assertEqual(ctx["workflow_step_id"], "0.1")
        self.assertEqual(ctx["input_artifacts"], ["input_a"])
        self.assertEqual(ctx["output_artifacts"], ["output_b"])
        self.assertIn("Do the thing.", tasks[0]["description"])

    def test_expand_template_produces_concrete_steps(self):
        from src.workflows.engine.expander import expand_template
        template = {
            "template_id": "feature_implementation_template",
            "parameters": {"feature_id": "", "feature_name": ""},
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
                    "name": "feature_implementation_plan",
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_expander.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/workflows/engine/expander.py
"""Expand workflow definitions into concrete tasks for the orchestrator.

Takes a WorkflowDefinition (phases, steps, templates) and creates
task dicts ready for insertion into the DB via add_subtasks_atomically.
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Agent name mapping (workflow JSON → system agent types)
AGENT_MAP = {
    "analyst": "planner",      # No dedicated analyst agent; planner handles analysis
    "summarizer": "writer",    # Writer handles summarization
    "router": None,            # System agent, not assigned directly
    "assistant": None,         # System agent
}


def map_agent_type(agent_name: str) -> str:
    """Map workflow agent names to system agent types."""
    if agent_name in AGENT_MAP:
        mapped = AGENT_MAP[agent_name]
        if mapped is None:
            return "executor"  # Fallback for system agents
        return mapped
    return agent_name  # Pass through if already a valid agent type


def expand_steps_to_tasks(
    steps: list[dict],
    goal_id: int,
    initial_context: dict | None = None,
) -> list[dict]:
    """Convert workflow steps into task dicts.

    Each task dict contains all info needed for add_task():
    title, description, agent_type, depends_on_steps, context, priority.
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
        step_type = step.get("type")  # "recurring" or None

        # Build context for the executing agent
        ctx = {
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
        if initial_context:
            ctx["workflow_context"] = initial_context

        # Priority: earlier phases get higher priority
        phase_num = 5
        try:
            phase_num = int(phase.replace("phase_", "")) if phase else 5
        except (ValueError, AttributeError):
            pass
        priority = max(1, 10 - phase_num)

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

    Template steps get IDs like: {prefix}.{template_step_id}
    Instructions have {param_name} placeholders replaced.
    """
    concrete_steps = []
    for tmpl_step in template.get("steps", []):
        step_id = tmpl_step["template_step_id"]
        full_id = f"{prefix}.{step_id}" if prefix else step_id

        # Substitute parameters in instruction
        instruction = tmpl_step.get("instruction", "")
        for key, value in params.items():
            instruction = instruction.replace(f"{{{key}}}", str(value))

        step = {
            "id": full_id,
            "phase": "phase_8",  # Templates are implementation phase
            "name": tmpl_step.get("name", step_id),
            "agent": tmpl_step.get("agent", "executor"),
            "depends_on": [],  # Will be resolved after expansion
            "instruction": instruction,
            "input_artifacts": tmpl_step.get("input_artifacts",
                                              template.get("context_artifacts", [])),
            "output_artifacts": tmpl_step.get("output_artifacts", []),
            "may_need_clarification": tmpl_step.get("may_need_clarification", False),
            "condition": tmpl_step.get("condition"),
        }
        concrete_steps.append(step)

    return concrete_steps
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_expander.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/expander.py tests/test_workflow_expander.py
git commit -m "feat: add workflow expander to convert JSON steps to tasks"
```

---

## Task 4: Workflow Runner — DB Integration and Lifecycle

**Files:**
- Create: `src/workflows/engine/runner.py`
- Test: `tests/test_workflow_runner.py`

The runner ties everything together: loads definition → creates goal → expands steps to tasks → inserts into DB. The orchestrator's existing loop handles execution.

**Step 1: Write the failing test**

```python
# tests/test_workflow_runner.py
import unittest
import asyncio

class TestWorkflowRunner(unittest.TestCase):
    """Workflow runner creates goal + tasks in DB."""

    def test_runner_creates_goal_and_tasks(self):
        from src.workflows.engine.runner import WorkflowRunner

        runner = WorkflowRunner.__new__(WorkflowRunner)
        # Test the step→task ID mapping logic
        step_ids = ["0.1", "0.2", "0.3"]
        task_ids = [101, 102, 103]
        mapping = dict(zip(step_ids, task_ids))

        # Resolve depends_on from step IDs to task IDs
        from src.workflows.engine.runner import resolve_dependencies
        deps = resolve_dependencies(["0.1", "0.3"], mapping)
        self.assertEqual(deps, [101, 103])

    def test_resolve_dependencies_skips_missing(self):
        from src.workflows.engine.runner import resolve_dependencies
        mapping = {"0.1": 101}
        deps = resolve_dependencies(["0.1", "0.99"], mapping)
        self.assertEqual(deps, [101])

    def test_build_workflow_description(self):
        from src.workflows.engine.runner import build_step_description
        desc = build_step_description(
            instruction="Do the thing.",
            input_artifacts=["doc_a", "doc_b"],
            artifact_contents={"doc_a": "Content A", "doc_b": None},
        )
        self.assertIn("Do the thing.", desc)
        self.assertIn("doc_a", desc)
        self.assertIn("Content A", desc)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_runner.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/workflows/engine/runner.py
"""Workflow runner — creates goals and tasks from workflow definitions.

Usage:
    runner = WorkflowRunner()
    goal_id = await runner.start("idea_to_product_v1", raw_idea="Build a ...")
"""
import json
import logging
from typing import Any, Optional

from .loader import load_workflow, WorkflowDefinition
from .expander import expand_steps_to_tasks, expand_template
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
) -> str:
    """Build the full task description with instruction + artifact context."""
    parts = [instruction]

    available = {k: v for k, v in artifact_contents.items() if v is not None}
    if available:
        parts.append("\n\n## Input Artifacts\n")
        parts.append(format_artifacts_for_prompt(available))

    missing = [k for k in input_artifacts if artifact_contents.get(k) is None]
    if missing:
        parts.append(f"\n\nNote: These artifacts are not yet available: {missing}")

    return "\n".join(parts)


class WorkflowRunner:
    """Starts and manages workflow executions."""

    def __init__(self):
        self.artifact_store = ArtifactStore(use_db=True)

    async def start(
        self,
        workflow_name: str,
        initial_input: dict[str, str] | None = None,
        title: str | None = None,
    ) -> int:
        """Start a workflow execution.

        Creates a goal and all initial tasks (non-template steps).
        Returns the goal_id.
        """
        from src.infra.db import add_goal, add_task

        # Load definition
        wf = load_workflow(workflow_name)

        # Create goal
        goal_title = title or f"Workflow: {wf.plan_id}"
        goal_desc = wf.metadata.get("description", "")
        goal_id = await add_goal(
            title=goal_title,
            description=goal_desc,
            priority=8,
            context={
                "workflow_id": wf.plan_id,
                "workflow_version": wf.version,
            },
        )

        # Store initial inputs as artifacts
        if initial_input:
            for name, value in initial_input.items():
                await self.artifact_store.store(goal_id, name, value)

        # Expand all non-template steps into tasks
        tasks = expand_steps_to_tasks(
            wf.steps, goal_id=goal_id,
            initial_context={"workflow_id": wf.plan_id},
        )

        # Insert tasks and build step_id → task_id mapping
        step_to_task: dict[str, int] = {}
        for task_dict in tasks:
            step_id = task_dict["context"]["workflow_step_id"]

            # Resolve depends_on from step IDs to task IDs
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

        logger.info(
            f"Workflow '{wf.plan_id}' started: goal_id={goal_id}, "
            f"tasks_created={len(step_to_task)}"
        )

        return goal_id
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/runner.py tests/test_workflow_runner.py
git commit -m "feat: add workflow runner for goal/task creation from definitions"
```

---

## Task 5: Workflow-Aware Task Execution Hook in Orchestrator

**Files:**
- Modify: `src/core/orchestrator.py` (add workflow step pre/post hooks)
- Modify: `src/agents/base.py` (inject artifact context for workflow steps)
- Test: `tests/test_workflow_orchestrator.py`

When the orchestrator executes a task that has `is_workflow_step: true` in its context, it needs to:
- **Pre-execution:** Fetch input artifacts from the artifact store and inject them into the task description
- **Post-execution:** Extract the agent's output and store it as the step's output artifacts

**Step 1: Write the failing test**

```python
# tests/test_workflow_orchestrator.py
import unittest

class TestWorkflowOrchestration(unittest.TestCase):

    def test_is_workflow_step(self):
        from src.workflows.engine.hooks import is_workflow_step
        ctx = {"is_workflow_step": True, "workflow_step_id": "0.1"}
        self.assertTrue(is_workflow_step(ctx))
        self.assertFalse(is_workflow_step({}))
        self.assertFalse(is_workflow_step({"is_workflow_step": False}))

    def test_enrich_task_with_artifacts(self):
        from src.workflows.engine.hooks import enrich_task_description
        task = {
            "description": "Do the analysis.",
            "context": {
                "is_workflow_step": True,
                "input_artifacts": ["doc_a"],
                "output_artifacts": ["analysis_result"],
            },
        }
        artifact_contents = {"doc_a": "Some document content"}
        enriched = enrich_task_description(task, artifact_contents)
        self.assertIn("Do the analysis.", enriched)
        self.assertIn("Some document content", enriched)

    def test_extract_output_artifacts(self):
        from src.workflows.engine.hooks import extract_output_artifact_names
        ctx = {"output_artifacts": ["raw_idea_document", "summary"]}
        names = extract_output_artifact_names(ctx)
        self.assertEqual(names, ["raw_idea_document", "summary"])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_orchestrator.py -v`
Expected: FAIL with ImportError

**Step 3: Write the hooks module**

```python
# src/workflows/engine/hooks.py
"""Pre/post execution hooks for workflow steps in the orchestrator."""
import json
import logging
from typing import Any, Optional

from .artifacts import ArtifactStore, format_artifacts_for_prompt

logger = logging.getLogger(__name__)

# Module-level artifact store (reused across calls)
_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(use_db=True)
    return _artifact_store


def is_workflow_step(context: dict) -> bool:
    """Check if a task is part of a workflow execution."""
    return bool(context.get("is_workflow_step"))


def extract_output_artifact_names(context: dict) -> list[str]:
    """Get the output artifact names from workflow context."""
    return context.get("output_artifacts", [])


def enrich_task_description(
    task: dict,
    artifact_contents: dict[str, Optional[str]],
) -> str:
    """Enrich task description with artifact context."""
    instruction = task.get("description", "")
    ctx = task.get("context", {})
    input_names = ctx.get("input_artifacts", [])

    if not input_names:
        return instruction

    available = {k: v for k, v in artifact_contents.items() if v is not None}
    if not available:
        return instruction

    artifact_section = format_artifacts_for_prompt(available)
    return f"{instruction}\n\n## Input Artifacts\n\n{artifact_section}"


async def pre_execute_workflow_step(task: dict) -> dict:
    """Pre-execution hook: fetch artifacts and enrich task.

    Called by the orchestrator before agent.execute().
    Returns the modified task dict.
    """
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        ctx = json.loads(ctx)

    if not is_workflow_step(ctx):
        return task

    goal_id = task.get("goal_id")
    input_names = ctx.get("input_artifacts", [])

    if not goal_id or not input_names:
        return task

    store = get_artifact_store()
    contents = await store.collect(goal_id, input_names)

    task["description"] = enrich_task_description(task, contents)
    return task


async def post_execute_workflow_step(
    task: dict, result: dict
) -> None:
    """Post-execution hook: store output artifacts.

    Called by the orchestrator after agent returns successfully.
    """
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

    # For single output artifact, store the whole result
    # For multiple, the agent should have structured its output with headers
    if len(output_names) == 1:
        await store.store(goal_id, output_names[0], result_text)
    else:
        # Try to split by artifact names as headers
        for name in output_names:
            await store.store(goal_id, name, result_text)

    logger.info(
        f"Stored artifacts {output_names} for step "
        f"{ctx.get('workflow_step_id', '?')}"
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_orchestrator.py -v`
Expected: PASS

**Step 5: Integrate hooks into orchestrator.py**

Add two lines to `src/core/orchestrator.py` in the `_execute_task` method:

1. Before the agent execution (around line 637), add pre-hook:

```python
# After context injection, before agent execution:
from ..workflows.engine.hooks import (
    pre_execute_workflow_step,
    post_execute_workflow_step,
    is_workflow_step,
)
task = await pre_execute_workflow_step(task)
```

2. After successful execution (around line 674), add post-hook:

```python
if status == "completed":
    # Store workflow artifacts if this is a workflow step
    task_ctx = task.get("context", {})
    if isinstance(task_ctx, str):
        import json as _json
        task_ctx = _json.loads(task_ctx)
    if is_workflow_step(task_ctx):
        await post_execute_workflow_step(task, result)
    await self._handle_complete(task, result)
```

**Step 6: Commit**

```bash
git add src/workflows/engine/hooks.py tests/test_workflow_orchestrator.py src/core/orchestrator.py
git commit -m "feat: add workflow step pre/post hooks to orchestrator"
```

---

## Task 6: Workflow Agent Type Registration

**Files:**
- Modify: `src/core/orchestrator.py` (add `workflow` agent_type handling)
- Modify: `src/core/task_classifier.py` (add workflow classification keywords)
- Test: `tests/test_workflow_dispatch.py`

Add a new `workflow` agent_type so users can trigger workflows via Telegram or API with messages like "Build me a product: [idea]".

**Step 1: Write the failing test**

```python
# tests/test_workflow_dispatch.py
import unittest

class TestWorkflowDispatch(unittest.TestCase):

    def test_workflow_keywords_classify(self):
        """Messages about building products should classify as workflow."""
        from src.workflows.engine.dispatch import should_start_workflow
        self.assertTrue(should_start_workflow(
            "Build me a product that does X"
        ))
        self.assertTrue(should_start_workflow(
            "I have an idea for a SaaS app"
        ))
        self.assertFalse(should_start_workflow(
            "Fix the bug in auth module"
        ))
        self.assertFalse(should_start_workflow(
            "Add a logout button"
        ))
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

logger = logging.getLogger(__name__)

# Patterns that suggest a full product/workflow request
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
    """Check if a user message should trigger a workflow instead of a single task."""
    return any(p.search(message) for p in _compiled)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_dispatch.py -v`
Expected: PASS

**Step 5: Wire into orchestrator**

In `src/core/orchestrator.py`, in the goal creation path (where Telegram messages create goals), add workflow detection:

```python
# In the goal/task creation path:
from ..workflows.engine.dispatch import should_start_workflow
from ..workflows.engine.runner import WorkflowRunner

if should_start_workflow(message_text):
    runner = WorkflowRunner()
    goal_id = await runner.start(
        "idea_to_product_v1",
        initial_input={"raw_idea": message_text},
        title=f"Product: {message_text[:80]}",
    )
    # Notify user that workflow started
    await telegram.send(f"Started idea-to-product workflow (goal #{goal_id})")
```

**Step 6: Add timeout for workflow agent type**

In orchestrator's `AGENT_TIMEOUTS` dict, add:
```python
"workflow": 900,  # 15 min — workflow steps can be lengthy
```

**Step 7: Commit**

```bash
git add src/workflows/engine/dispatch.py tests/test_workflow_dispatch.py src/core/orchestrator.py
git commit -m "feat: add workflow dispatch and agent type registration"
```

---

## Task 7: Template Expansion at Runtime

**Files:**
- Modify: `src/workflows/engine/runner.py` (add template expansion during Phase 8)
- Modify: `src/workflows/engine/hooks.py` (handle template trigger steps)
- Test: `tests/test_template_expansion.py`

When step `8.0` (feature_backlog_initialization) completes, its output is a feature list. The post-hook should detect this and expand the `feature_implementation_template` for each feature, creating new tasks in the DB.

**Step 1: Write the failing test**

```python
# tests/test_template_expansion.py
import unittest

class TestTemplateExpansion(unittest.TestCase):

    def test_expand_feature_backlog_to_tasks(self):
        from src.workflows.engine.runner import expand_feature_backlog
        backlog = [
            {"feature_id": "F-001", "feature_name": "User Auth"},
            {"feature_id": "F-002", "feature_name": "Dashboard"},
        ]
        template = {
            "template_id": "feature_implementation_template",
            "parameters": {"feature_id": "", "feature_name": ""},
            "context_artifacts": [],
            "steps": [
                {
                    "template_step_id": "feat.1",
                    "name": "spec_review",
                    "agent": "planner",
                    "instruction": "Review {feature_name}.",
                    "output_artifacts": ["spec_summary"],
                },
            ],
        }
        all_steps = expand_feature_backlog(backlog, template)
        self.assertEqual(len(all_steps), 2)  # 1 step × 2 features
        self.assertEqual(all_steps[0]["id"], "8.F-001.feat.1")
        self.assertIn("User Auth", all_steps[0]["instruction"])
        self.assertEqual(all_steps[1]["id"], "8.F-002.feat.1")
        self.assertIn("Dashboard", all_steps[1]["instruction"])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_template_expansion.py -v`
Expected: FAIL

**Step 3: Add expand_feature_backlog to runner.py**

```python
# Add to src/workflows/engine/runner.py:

def expand_feature_backlog(
    backlog: list[dict],
    template: dict,
) -> list[dict]:
    """Expand the feature backlog using the implementation template.

    Each feature in the backlog gets its own set of template steps.
    """
    all_steps = []
    for feature in backlog:
        fid = feature.get("feature_id", "unknown")
        params = {
            "feature_id": fid,
            "feature_name": feature.get("feature_name", fid),
        }
        # Add any extra params from the feature
        for k, v in feature.items():
            if k not in params:
                params[k] = str(v)

        prefix = f"8.{fid}"
        steps = expand_template(template, params, prefix=prefix)
        all_steps.extend(steps)

    return all_steps
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_template_expansion.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/runner.py tests/test_template_expansion.py
git commit -m "feat: add feature backlog template expansion for Phase 8"
```

---

## Task 8: CodingPipeline Integration Point

**Files:**
- Modify: `src/workflows/engine/hooks.py` (detect coding implementation steps, delegate to pipeline)
- Modify: `src/core/orchestrator.py` (route feature implementation steps to pipeline)

For Phase 8 template steps that involve actual coding (feat.3 through feat.9 = backend implementation), we delegate to CodingPipeline instead of running them as individual agent tasks. This reuses the existing pipeline's review loop, incremental progress, and model diversity.

**Step 1: Add pipeline delegation detection**

```python
# Add to src/workflows/engine/hooks.py:

# Template steps that should be delegated to CodingPipeline
PIPELINE_DELEGATE_STEPS = {
    "feat.3",   # database_migration
    "feat.4",   # backend_model
    "feat.5",   # backend_service
    "feat.6",   # backend_endpoints
    "feat.7",   # backend_middleware
    "feat.10",  # frontend_types
    "feat.11",  # frontend_state
    "feat.12",  # frontend_api_client
    "feat.13",  # frontend_components
    "feat.14",  # frontend_page_assembly
    "feat.15",  # frontend_form_handling
}


def should_delegate_to_pipeline(context: dict) -> bool:
    """Check if this workflow step should be delegated to CodingPipeline."""
    if not is_workflow_step(context):
        return False
    step_id = context.get("workflow_step_id", "")
    # Check if the step suffix matches a pipeline-delegatable step
    parts = step_id.rsplit(".", 1)
    if len(parts) == 2:
        return parts[-1] in PIPELINE_DELEGATE_STEPS
    return False
```

**Step 2: Wire into orchestrator**

In the orchestrator's execution path, before agent selection:

```python
# In _execute_task, after pre_execute_workflow_step:
from ..workflows.engine.hooks import should_delegate_to_pipeline

task_ctx = task.get("context", {})
if isinstance(task_ctx, str):
    task_ctx = json.loads(task_ctx)

if should_delegate_to_pipeline(task_ctx):
    # Route to CodingPipeline with the step's instruction as the task
    from ..workflows.pipeline import CodingPipeline
    pipeline = CodingPipeline()
    coro = pipeline.run(task)
elif agent_type == "pipeline":
    # Existing pipeline path
    ...
```

**Step 3: Commit**

```bash
git add src/workflows/engine/hooks.py src/core/orchestrator.py
git commit -m "feat: delegate coding workflow steps to CodingPipeline"
```

---

## Task 9: Recurring Steps Support (Phase 15)

**Files:**
- Modify: `src/workflows/engine/runner.py` (register recurring steps as scheduled_tasks)
- Test: `tests/test_workflow_recurring.py`

Phase 15 has steps with `"type": "recurring"` and `"trigger"` fields. These need to be registered as scheduled tasks in the existing `scheduled_tasks` table.

**Step 1: Write the failing test**

```python
# tests/test_workflow_recurring.py
import unittest

class TestRecurringSteps(unittest.TestCase):

    def test_identify_recurring_steps(self):
        from src.workflows.engine.runner import get_recurring_steps
        steps = [
            {"id": "15.1", "type": "recurring", "trigger": "Weekly"},
            {"id": "15.2", "name": "one_time_step"},
        ]
        recurring = get_recurring_steps(steps)
        self.assertEqual(len(recurring), 1)
        self.assertEqual(recurring[0]["id"], "15.1")

    def test_trigger_to_cron(self):
        from src.workflows.engine.runner import trigger_to_cron
        self.assertEqual(trigger_to_cron("Weekly"), "0 9 * * 1")
        self.assertEqual(trigger_to_cron("Daily"), "0 9 * * *")
        self.assertEqual(trigger_to_cron("Monthly"), "0 9 1 * *")
        self.assertEqual(trigger_to_cron("Quarterly"), "0 9 1 1,4,7,10 *")
        # Custom cron passthrough
        self.assertEqual(trigger_to_cron("0 */4 * * *"), "0 */4 * * *")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_recurring.py -v`

**Step 3: Implement**

```python
# Add to src/workflows/engine/runner.py:

def get_recurring_steps(steps: list[dict]) -> list[dict]:
    """Filter steps that are marked as recurring."""
    return [s for s in steps if s.get("type") == "recurring"]


def trigger_to_cron(trigger: str) -> str:
    """Convert trigger description to cron expression."""
    trigger_lower = trigger.lower().strip()

    # Common mappings
    mappings = {
        "daily": "0 9 * * *",
        "weekly": "0 9 * * 1",
        "monthly": "0 9 1 * *",
        "quarterly": "0 9 1 1,4,7,10 *",
        "hourly": "0 * * * *",
    }

    for key, cron in mappings.items():
        if key in trigger_lower:
            return cron

    # If it looks like a cron expression already, pass through
    if len(trigger.split()) == 5:
        return trigger

    # Default: weekly
    return "0 9 * * 1"


async def register_recurring_steps(
    steps: list[dict],
    goal_id: int,
    workflow_id: str,
) -> list[int]:
    """Register recurring workflow steps as scheduled tasks."""
    from src.infra.db import add_scheduled_task

    recurring = get_recurring_steps(steps)
    sched_ids = []

    for step in recurring:
        cron = trigger_to_cron(step.get("trigger", "Weekly"))
        agent = map_agent_type(step.get("agent", "executor"))

        sched_id = await add_scheduled_task(
            title=f"[{workflow_id}] {step.get('name', step['id'])}",
            description=step.get("instruction", ""),
            cron_expression=cron,
            agent_type=agent,
            context={
                "workflow_id": workflow_id,
                "workflow_step_id": step["id"],
                "goal_id": goal_id,
                "is_workflow_step": True,
                "input_artifacts": step.get("input_artifacts", []),
                "output_artifacts": step.get("output_artifacts", []),
            },
        )
        sched_ids.append(sched_id)

    return sched_ids
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflow_recurring.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/workflows/engine/runner.py tests/test_workflow_recurring.py
git commit -m "feat: add recurring step registration as scheduled tasks"
```

---

## Task 10: Workflow Status Tracking and Telegram Integration

**Files:**
- Create: `src/workflows/engine/status.py`
- Test: `tests/test_workflow_status.py`

Track workflow progress (which phases are done, current active steps, % complete) and report via Telegram at phase boundaries.

**Step 1: Write the failing test**

```python
# tests/test_workflow_status.py
import unittest

class TestWorkflowStatus(unittest.TestCase):

    def test_compute_progress(self):
        from src.workflows.engine.status import compute_workflow_progress
        tasks = [
            {"status": "completed", "context": '{"workflow_phase": "phase_0"}'},
            {"status": "completed", "context": '{"workflow_phase": "phase_0"}'},
            {"status": "processing", "context": '{"workflow_phase": "phase_1"}'},
            {"status": "pending", "context": '{"workflow_phase": "phase_1"}'},
            {"status": "pending", "context": '{"workflow_phase": "phase_2"}'},
        ]
        progress = compute_workflow_progress(tasks)
        self.assertEqual(progress["total"], 5)
        self.assertEqual(progress["completed"], 2)
        self.assertEqual(progress["in_progress"], 1)
        self.assertAlmostEqual(progress["percent"], 40.0)
        self.assertIn("phase_0", progress["completed_phases"])

    def test_format_progress_message(self):
        from src.workflows.engine.status import format_progress_message
        progress = {
            "total": 10, "completed": 3, "in_progress": 1,
            "percent": 30.0,
            "completed_phases": ["phase_0"],
            "active_phases": ["phase_1"],
            "current_step": "market_research",
        }
        msg = format_progress_message(progress, "My Product")
        self.assertIn("30.0%", msg)
        self.assertIn("My Product", msg)
```

**Step 2: Run test, verify fail, implement, verify pass**

```python
# src/workflows/engine/status.py
"""Track and report workflow progress."""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_workflow_progress(tasks: list[dict]) -> dict:
    """Compute workflow progress from task statuses."""
    total = len(tasks)
    completed = sum(1 for t in tasks if t["status"] == "completed")
    in_progress = sum(1 for t in tasks if t["status"] == "processing")
    failed = sum(1 for t in tasks if t["status"] == "failed")

    # Group by phase
    phase_tasks: dict[str, list[dict]] = {}
    for t in tasks:
        ctx = t.get("context", "{}")
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                ctx = {}
        phase = ctx.get("workflow_phase", "unknown")
        phase_tasks.setdefault(phase, []).append(t)

    completed_phases = []
    active_phases = []
    for phase, ptasks in sorted(phase_tasks.items()):
        if all(t["status"] == "completed" for t in ptasks):
            completed_phases.append(phase)
        elif any(t["status"] in ("processing", "completed") for t in ptasks):
            active_phases.append(phase)

    # Current step
    current = next(
        (t for t in tasks if t["status"] == "processing"),
        None
    )
    current_step = current["title"] if current else None

    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "failed": failed,
        "percent": round((completed / total * 100) if total > 0 else 0, 1),
        "completed_phases": completed_phases,
        "active_phases": active_phases,
        "current_step": current_step,
    }


def format_progress_message(progress: dict, workflow_title: str) -> str:
    """Format progress as a Telegram-friendly message."""
    lines = [
        f"Workflow: {workflow_title}",
        f"Progress: {progress['percent']}% ({progress['completed']}/{progress['total']})",
    ]
    if progress["completed_phases"]:
        lines.append(f"Completed phases: {', '.join(progress['completed_phases'])}")
    if progress["active_phases"]:
        lines.append(f"Active phases: {', '.join(progress['active_phases'])}")
    if progress["current_step"]:
        lines.append(f"Current: {progress['current_step']}")
    if progress["failed"]:
        lines.append(f"Failed: {progress['failed']} tasks")
    return "\n".join(lines)
```

**Step 3: Commit**

```bash
git add src/workflows/engine/status.py tests/test_workflow_status.py
git commit -m "feat: add workflow progress tracking and Telegram reporting"
```

---

## Summary: How It All Fits Together

### Architecture Diagram

```
User: "Build me a SaaS for X"
          │
          ▼
┌──────────────────────┐
│   Telegram / API     │
│  should_start_workflow│──── true ───┐
│  detects "build app" │             │
└──────────────────────┘             ▼
                              ┌──────────────┐
                              │WorkflowRunner │
                              │ .start()      │
                              │  - load JSON  │
                              │  - create goal│
                              │  - expand     │
                              │    steps →    │
                              │    tasks      │
                              └───────┬───────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │   Database    │
                              │ goals + tasks │
                              │ (depends_on)  │
                              └───────┬───────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
            ┌──────────┐     ┌──────────┐       ┌──────────┐
            │Orchestrator│   │Orchestrator│     │Orchestrator│
            │ poll loop  │   │ poll loop  │     │ poll loop  │
            │ (24/7)     │   │ (24/7)     │     │ (24/7)     │
            └──────┬─────┘   └──────┬─────┘     └──────┬─────┘
                   │                │                   │
                   ▼                ▼                   ▼
            ┌──────────┐     ┌──────────┐       ┌──────────┐
            │Pre-hook:  │    │Pre-hook:  │      │Pre-hook:  │
            │fetch      │    │fetch      │      │fetch      │
            │artifacts  │    │artifacts  │      │artifacts  │
            └──────┬────┘    └──────┬────┘      └──────┬────┘
                   │                │                   │
                   ▼                ▼                   ▼
            ┌──────────┐    ┌──────────────┐    ┌──────────┐
            │  Agent    │   │CodingPipeline│    │  Agent    │
            │(researcher│   │(Phase 8 impl)│    │ (writer,  │
            │ planner,  │   │  architect → │    │  reviewer)│
            │ analyst)  │   │  implement → │    │           │
            │           │   │  test → fix  │    │           │
            └──────┬────┘   └──────┬───────┘    └──────┬────┘
                   │               │                    │
                   ▼               ▼                    ▼
            ┌──────────┐    ┌──────────┐         ┌──────────┐
            │Post-hook: │   │Post-hook: │        │Post-hook: │
            │store      │   │store      │        │store      │
            │artifacts  │   │artifacts  │        │artifacts  │
            └───────────┘   └───────────┘        └───────────┘
```

### File Structure After Implementation

```
src/workflows/
├── engine/
│   ├── __init__.py         # Package init
│   ├── loader.py           # Load + validate JSON workflow definitions
│   ├── expander.py         # Convert steps → task dicts, expand templates
│   ├── runner.py           # Create goal + tasks in DB, template expansion
│   ├── hooks.py            # Pre/post execution hooks for orchestrator
│   ├── artifacts.py        # Artifact store (blackboard-backed)
│   ├── dispatch.py         # Detect workflow-triggering messages
│   └── status.py           # Progress tracking + Telegram reporting
├── pipeline/
│   ├── pipeline.py         # Existing CodingPipeline (unchanged)
│   ├── pipeline_context.py # Existing (unchanged)
│   └── pipeline_utils.py   # Existing (unchanged)
└── idea_to_product/
    └── idea_to_product_v1.json  # Workflow definition (unchanged)
```

### Key Design Decisions

1. **No new event loop** — The existing orchestrator's `run_loop()` already polls `get_ready_tasks()` every 2-3 seconds. Workflow tasks are just regular DB tasks with `depends_on` chains.

2. **CodingPipeline is nested, not replaced** — For Phase 8 coding steps, we delegate to `CodingPipeline.run()` which already handles architect → implement → test → review → fix cycles. The workflow engine handles everything else.

3. **Artifacts via blackboard** — Step outputs are stored on the goal's blackboard under an `"artifacts"` key. Pre-hooks inject them into downstream task descriptions. No new storage system needed.

4. **Templates expand lazily** — The `feature_implementation_template` isn't expanded at workflow start (we don't know the features yet). It expands after step `8.0` completes with the feature backlog.

5. **Recurring steps → scheduled_tasks** — Phase 15's recurring operations (monitoring, patching, retrospectives) register as scheduled tasks using the existing cron system.

6. **24/7 operation requires no changes** — The orchestrator already runs 24/7. Workflow tasks enter the DB and get picked up on the next poll cycle. Long-running workflows simply have tasks that depend on tasks that depend on tasks — all resolved by `get_ready_tasks()`.
