# Implementation Plan — Yalayut Phase 2: the intersect + consumer wiring

> **Note for agentic workers**: Execute tasks in order. Each task is a closed
> unit: failing test → run (FAIL) → minimal impl → run (PASS) → commit. Do not
> skip the FAIL-first step. Every `pytest` invocation MUST carry a `timeout`
> prefix (project rule — zombie pytest holds the SQLite write lock). Use
> `python -c "import ..."` smoke checks before committing wiring tasks. This
> plan assumes **Phase 1 is complete and working**: `packages/yalayut/` exists,
> exposes `query(task_ctx) -> list[Artifact]`, the 13 tables are created by
> Phase 1's migration, and 20 seed manifests are installed. Do NOT re-create
> any Phase 1 artifact.

## Goal

Build the **consumer side** of yalayut: a thin `intersect` package that matches
catalog artifacts to a task, decides how each is exposed (`inject` / `tool` /
`preempt` / `quarantine`), attaches a `task["skills"]` envelope, routes
`preempt` tasks to the mechanical lane, and writes `yalayut_usage` telemetry.
Wire `intersect.flash(task)` into the orchestrator pump (once per task, before
dispatch) and make `coulson` actually consume the envelope and render it into
the agent prompt. Convert `src/memory/skills.py` to a thin shim that returns
the envelope's `inject` slice with **byte-identical injection output** to
today's behaviour. Add per-step `recipe_lookup` / `recipe_hint` JSON fields to
i2p / `workflow_engine`.

End state: a real task flows `orchestrator pump → intersect.flash() →
yalayut.query() → score → budget → exposure decision → task["skills"] →
coulson renders into prompt`, integration-tested end to end. No unwired
fragments — coulson reads the new path, the old skills.py path is a shim over
the same envelope, and there is a test asserting the shim is byte-identical.

## Architecture

```
orchestrator pump (src/core/orchestrator.py)
  └─ run_loop(): task = beckman.next_task()
       └─ task = await intersect.flash(task)        ← NEW wiring (once per task)
       │     ├─ exposure_class == preempt → task routed to mechanical lane
       │     │       (runner=mechanical, payload.action=yalayut_recipe)
       │     └─ else → task["skills"] = [SkillApplication dict, ...]
       └─ asyncio.create_task(self._dispatch(task))

intersect.flash(task):                              ← NEW package packages/intersect/
  1. yalayut.query(task_ctx)        → list[Artifact]   (Phase 1 API; in-process)
  2. scoring.score(...)             → confidence float per artifact
  3. budget.apply_caps(...)         → api ≤3/step, mcp tools ≤3/server ≤6/step
  4. exposure.classify(...)         → exposure_class per artifact
  5. binding.static_bind(...)       → bound args for preempt + parametric inject
  6. preempt → route to mechanical lane; else attach task["skills"]
  7. telemetry.record_usage(...)    → yalayut_usage rows
  errors anywhere → graceful degrade: task["skills"] = [], return task

coulson (packages/coulson/src/coulson/context.py)   ← MODIFIED
  build_user_context(): reads task["skills"] filtered to
    exposure_class=inject AND applies_to=execution, renders via
    coulson/skill_render.py into the agent prompt. tool-class entries
    feed the per-execution allowed_tools list. Replaces the
    skills.find_relevant_skills free-text branch.

src/memory/skills.py                                ← MODIFIED → shim
  find_relevant_skills(task_text, task=None): when a task carries the
    envelope, returns the inject slice; format_skills_for_prompt
    produces byte-identical output to today.

workflow_engine / i2p                               ← MODIFIED
  per-step `recipe_lookup: true|false` + `recipe_hint` JSON fields.
  Step loader carries them through; intersect reads recipe_hint for
  hint_bonus and recipe_lookup as the gate to even call query().
```

`intersect` imports **only** `yalayut` + db + embeddings + logging. It does NOT
import `LLMDispatcher` (Phase 2 has no LLM-bind — locked KutAI rule: only
Beckman calls the dispatcher). `orchestrator` imports `intersect` only (0
`yalayut` import). `coulson` imports neither package — it reads a plain-dict
envelope. `Artifact` (Phase 1 dataclass) is used only in-process inside
intersect; everything crossing a serialization boundary (`task["skills"]`,
mechanical payload) is a plain dict.

## Tech Stack

- Python 3.10, async throughout (`async/await`, no sync blocking)
- Packages under `packages/<name>/src/<name>/` (src layout, setuptools)
- SQLite via aiosqlite (`src/infra/db.py`, WAL mode)
- Embeddings: `multilingual-e5-base` (768d) via `src/memory/embeddings.py`
- pytest + pytest-asyncio for tests
- Lazy imports for cross-module deps (avoid circular imports)
- Conventional commits: `feat()`, `fix()`, `test:`, `docs:`

## File Structure

### New package — `packages/intersect/`

| File | Responsibility |
|---|---|
| `packages/intersect/pyproject.toml` | Package metadata; `dependencies = ["yalayut"]` |
| `packages/intersect/src/intersect/__init__.py` | Public surface: re-exports `flash`; exposes thresholds |
| `packages/intersect/src/intersect/flash.py` | `flash(task)` entry — orchestrates query→score→budget→classify→bind→attach/route→telemetry; graceful degrade |
| `packages/intersect/src/intersect/scoring.py` | `score_artifact(artifact, task_ctx)` — `confidence = vector_sim × source_trust × owner_trust × hint_bonus` |
| `packages/intersect/src/intersect/exposure.py` | `classify(artifact, confidence)` — `(tier × kind × confidence) → exposure_class`; threshold constants |
| `packages/intersect/src/intersect/binding.py` | `static_bind(artifact, task_ctx)` — `bind_from` path resolution + `yalayut_bind_cache` read/write |
| `packages/intersect/src/intersect/budget.py` | `apply_caps(applications)` — api ≤3/step, mcp tools ≤3/server ≤6/step |
| `packages/intersect/src/intersect/telemetry.py` | `record_usage(...)` — writes `yalayut_usage` rows (exposure_class, bind_args, conflict_loser) |
| `packages/intersect/tests/__init__.py` | empty |
| `packages/intersect/tests/conftest.py` | shared fixtures: in-memory DB, fake yalayut.query, sample task |
| `packages/intersect/tests/test_scoring.py` | scoring unit tests |
| `packages/intersect/tests/test_exposure.py` | exposure-class decision matrix |
| `packages/intersect/tests/test_binding.py` | static-bind + cache-hit + cold-miss |
| `packages/intersect/tests/test_budget.py` | api/mcp caps |
| `packages/intersect/tests/test_telemetry.py` | usage-row writes incl. conflict_loser |
| `packages/intersect/tests/test_flash.py` | `flash()` integration: attach envelope, route preempt, graceful degrade |

### Modified files

| File | Change |
|---|---|
| `src/core/orchestrator.py` | `run_loop()` calls `await intersect.flash(task)` before `_dispatch` |
| `packages/coulson/src/coulson/context.py` | `build_user_context` reads `task["skills"]` envelope instead of `skills.find_relevant_skills` free-text branch |
| `packages/coulson/src/coulson/skill_render.py` | **NEW** — renders `SkillApplication` dicts to prompt text (rendering lives in coulson) |
| `src/memory/skills.py` | `find_relevant_skills` becomes a shim reading the envelope; byte-identical injection |
| `src/workflows/engine/loader.py` | `Workflow.get_step` carries `recipe_lookup` + `recipe_hint` through |
| `src/workflows/engine/expander.py` | propagate `recipe_lookup` + `recipe_hint` into task context at expansion |
| `src/workflows/i2p/i2p_v3.json` | add `recipe_lookup` + `recipe_hint` to scaffold/auth/api/deploy/test-setup/migration steps |

### New tests (modified-side)

| File | Covers |
|---|---|
| `packages/coulson/tests/test_skill_render.py` | `skill_render` output shapes |
| `packages/coulson/tests/test_context_envelope.py` | context build reads envelope; tool-class injection |
| `tests/memory/test_skills_shim.py` | shim byte-identical injection assertion |
| `tests/workflows/test_recipe_lookup_fields.py` | loader + expander carry `recipe_lookup`/`recipe_hint` |
| `tests/integration/test_yalayut_phase2_e2e.py` | end-to-end: task → flash → envelope → coulson prompt |

---

## Task 1 — Scaffold the `intersect` package

**Files:**
- Create: `packages/intersect/pyproject.toml`
- Create: `packages/intersect/src/intersect/__init__.py`
- Create: `packages/intersect/tests/__init__.py`
- Test: `packages/intersect/tests/test_package_import.py`

**Steps:**

- [ ] Write failing test `packages/intersect/tests/test_package_import.py`:

```python
"""Package-scaffold smoke test for intersect."""


def test_intersect_imports_and_exposes_flash():
    import intersect
    assert hasattr(intersect, "flash")
    assert callable(intersect.flash)


def test_intersect_exposes_thresholds():
    import intersect
    # Exposure thresholds must be importable for ops tuning.
    assert isinstance(intersect.THETA_PREEMPT, float)
    assert isinstance(intersect.THETA_INJECT, float)
    assert isinstance(intersect.THETA_TOOL, float)
    assert isinstance(intersect.THETA_MIN, float)
    assert intersect.THETA_PREEMPT > intersect.THETA_INJECT
    assert intersect.THETA_INJECT > intersect.THETA_TOOL
    assert intersect.THETA_TOOL > intersect.THETA_MIN
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_package_import.py` — expected **FAIL** (`ModuleNotFoundError: No module named 'intersect'`).
- [ ] Create `packages/intersect/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "intersect"
version = "0.1.0"
description = "Thin match+expose layer over the yalayut catalog"
requires-python = ">=3.10"
dependencies = ["yalayut"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] Create `packages/intersect/tests/__init__.py` (empty file).
- [ ] Create `packages/intersect/src/intersect/__init__.py`:

```python
"""Intersect — thin per-task match+expose layer over the yalayut catalog.

One public function: ``flash(task)``. Invoked once per task by the
orchestrator pump, before dispatch. Imports yalayut (in-process catalog
read) plus db/embeddings only. Never imports LLMDispatcher — Phase 2 has
no LLM-bind (locked KutAI rule: only Beckman calls the dispatcher).
"""
from __future__ import annotations

from intersect.exposure import (
    THETA_PREEMPT, THETA_INJECT, THETA_TOOL, THETA_MIN,
)
from intersect.flash import flash

__all__ = [
    "flash",
    "THETA_PREEMPT", "THETA_INJECT", "THETA_TOOL", "THETA_MIN",
]
```

- [ ] Create a placeholder `packages/intersect/src/intersect/exposure.py` so the
  import resolves (real logic lands in Task 3):

```python
"""Exposure-class decision. Thresholds tunable via ops; defaults strict."""
from __future__ import annotations

# θ_preempt > θ_inject > θ_tool > θ_min. Conservative defaults; lowered
# later based on yalayut_usage success-rate telemetry.
THETA_PREEMPT: float = 0.80
THETA_INJECT: float = 0.55
THETA_TOOL: float = 0.45
THETA_MIN: float = 0.30
```

- [ ] Create a placeholder `packages/intersect/src/intersect/flash.py`:

```python
"""flash(task) entry. Real body lands in Task 7."""
from __future__ import annotations


async def flash(task: dict) -> dict:
    """Placeholder — wired in Task 7."""
    return task
```

- [ ] Install the package editable so imports resolve:
  `python -m pip install -e packages/intersect`
- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_package_import.py` — expected **PASS**.
- [ ] Commit: `feat(intersect): scaffold package with flash entry + exposure thresholds`

---

## Task 2 — Scoring module

`confidence = vector_sim × source_trust × owner_trust × hint_bonus`. `vector_sim`
arrives on the `Artifact` from `yalayut.query` (Phase 1 ranks by vector
similarity). `source_trust` / `owner_trust` are read from `yalayut_sources` /
`yalayut_owners`. `hint_bonus` is `1.0` baseline, `>1.0` when the step's
`recipe_hint` keyword appears in the artifact's intent keywords / name.

**Files:**
- Create: `packages/intersect/src/intersect/scoring.py`
- Create: `packages/intersect/tests/conftest.py`
- Test: `packages/intersect/tests/test_scoring.py`

**Steps:**

- [ ] Create `packages/intersect/tests/conftest.py` (shared fixtures used by Tasks 2–7):

```python
"""Shared intersect test fixtures."""
from __future__ import annotations

import types

import pytest


class FakeArtifact:
    """Stand-in for yalayut's Artifact dataclass.

    Phase 1's Artifact carries: id, artifact_type, kind, name, source,
    owner, vet_tier, ver/version, mechanizable, applies_to, vector_sim,
    inputs_schema, payload-ish body. We mirror the fields intersect reads.
    """

    def __init__(
        self, *, id=1, artifact_type="skill", kind="prompt_skill",
        name="anthropics-pdf", source="github:anthropics/skills@/skills",
        owner="anthropics", vet_tier=0, vector_sim=0.9, mechanizable=False,
        applies_to="execution", intent_keywords=None, inputs_schema=None,
        body="PDF skill body text", env_status="ready",
    ):
        self.id = id
        self.artifact_type = artifact_type
        self.kind = kind
        self.name = name
        self.source = source
        self.owner = owner
        self.vet_tier = vet_tier
        self.vector_sim = vector_sim
        self.mechanizable = mechanizable
        self.applies_to = applies_to
        self.intent_keywords = intent_keywords or []
        self.inputs_schema = inputs_schema or {}
        self.body = body
        self.env_status = env_status


@pytest.fixture
def fake_artifact():
    return FakeArtifact


@pytest.fixture
def sample_task():
    """A plain workflow-step task dict as the orchestrator pump sees it."""
    import json
    return {
        "id": 4101,
        "title": "[3.2] Scaffold the Python package",
        "description": "Create the package skeleton with pyproject + src layout",
        "agent_type": "coder",
        "mission_id": 57,
        "context": json.dumps({
            "is_workflow_step": True,
            "workflow_step_id": "3.2",
            "recipe_lookup": True,
            "recipe_hint": "python package scaffold",
        }),
    }
```

- [ ] Write failing test `packages/intersect/tests/test_scoring.py`:

```python
"""Unit tests for intersect.scoring."""
import pytest

from intersect import scoring


def test_confidence_is_product_of_factors(fake_artifact):
    art = fake_artifact(vector_sim=0.8)
    conf = scoring.score_artifact(
        art, source_trust=0.9, owner_trust=1.0, hint_bonus=1.0,
    )
    assert conf == pytest.approx(0.8 * 0.9 * 1.0 * 1.0)


def test_hint_bonus_lifts_confidence(fake_artifact):
    art = fake_artifact(vector_sim=0.6)
    base = scoring.score_artifact(art, source_trust=1.0, owner_trust=1.0,
                                  hint_bonus=1.0)
    boosted = scoring.score_artifact(art, source_trust=1.0, owner_trust=1.0,
                                     hint_bonus=1.25)
    assert boosted > base
    assert boosted == pytest.approx(base * 1.25)


def test_confidence_clamped_to_unit_interval(fake_artifact):
    art = fake_artifact(vector_sim=1.0)
    conf = scoring.score_artifact(art, source_trust=1.0, owner_trust=1.0,
                                  hint_bonus=3.0)
    assert conf == 1.0


def test_hint_bonus_for_matching_recipe_hint(fake_artifact):
    art = fake_artifact(
        name="cc-pypackage",
        intent_keywords=["python", "package", "scaffold", "pyproject"],
    )
    bonus = scoring.compute_hint_bonus(art, recipe_hint="python package scaffold")
    assert bonus > 1.0


def test_hint_bonus_neutral_when_no_hint(fake_artifact):
    art = fake_artifact(intent_keywords=["django", "web"])
    assert scoring.compute_hint_bonus(art, recipe_hint=None) == 1.0


def test_hint_bonus_neutral_on_keyword_miss(fake_artifact):
    art = fake_artifact(intent_keywords=["matlab", "simulink"])
    assert scoring.compute_hint_bonus(art, recipe_hint="react frontend") == 1.0
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_scoring.py` — expected **FAIL** (`AttributeError: module 'intersect.scoring' has no attribute ...`).
- [ ] Create `packages/intersect/src/intersect/scoring.py`:

```python
"""Confidence scoring for matched artifacts.

confidence = vector_sim × source_trust × owner_trust × hint_bonus

vector_sim arrives on the Artifact from yalayut.query (Phase 1 ranks the
index by vector similarity over the name+name_original+body embedding).
source_trust / owner_trust are looked up from yalayut_sources /
yalayut_owners by flash.py and passed in. hint_bonus rewards artifacts
whose intent keywords overlap the step's recipe_hint.
"""
from __future__ import annotations

# Maximum multiplicative lift from a recipe_hint keyword match. A full
# keyword overlap caps the bonus here; partial overlap scales linearly.
HINT_BONUS_MAX: float = 1.30


def _tokenize(text: str) -> set[str]:
    """Lowercase word split, drop tokens <= 2 chars. Mirrors the
    coarse tokenisation used elsewhere in KutAI for keyword overlap."""
    import re
    out: set[str] = set()
    for w in re.split(r"[\s,;.:/()_\-]+", (text or "").lower()):
        w = w.strip("'\"")
        if len(w) > 2:
            out.add(w)
    return out


def compute_hint_bonus(artifact, recipe_hint: str | None) -> float:
    """Return a multiplicative bonus in [1.0, HINT_BONUS_MAX].

    1.0 when there is no recipe_hint or zero keyword overlap. Scales
    linearly with the fraction of recipe_hint tokens found in the
    artifact's intent keywords or name.
    """
    if not recipe_hint:
        return 1.0
    hint_tokens = _tokenize(recipe_hint)
    if not hint_tokens:
        return 1.0
    art_tokens = set()
    for kw in getattr(artifact, "intent_keywords", None) or []:
        art_tokens |= _tokenize(str(kw))
    art_tokens |= _tokenize(getattr(artifact, "name", "") or "")
    overlap = len(hint_tokens & art_tokens) / len(hint_tokens)
    if overlap <= 0.0:
        return 1.0
    return 1.0 + overlap * (HINT_BONUS_MAX - 1.0)


def score_artifact(
    artifact,
    *,
    source_trust: float,
    owner_trust: float,
    hint_bonus: float = 1.0,
) -> float:
    """Compute final confidence, clamped to [0.0, 1.0]."""
    vector_sim = float(getattr(artifact, "vector_sim", 0.0) or 0.0)
    raw = vector_sim * float(source_trust) * float(owner_trust) * float(hint_bonus)
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_scoring.py` — expected **PASS**.
- [ ] Commit: `feat(intersect): confidence scoring with recipe_hint keyword bonus`

---

## Task 3 — Exposure-class decision module

`(tier × kind × confidence) → exposure_class ∈ {inject, tool, preempt, quarantine}`.
Phase 2 has **no `sandbox`** — T2 is quarantined-until-promoted (treated like
T3). The `render` sub-variant (`prose` vs `prebind`) for `inject` is decided
here too: `prebind` only when the artifact is parametric and fully bound.

**Files:**
- Modify: `packages/intersect/src/intersect/exposure.py` (replace placeholder)
- Test: `packages/intersect/tests/test_exposure.py`

**Steps:**

- [ ] Write failing test `packages/intersect/tests/test_exposure.py`:

```python
"""Unit tests for the exposure-class decision matrix."""
import pytest

from intersect import exposure


def test_t3_always_quarantine(fake_artifact):
    art = fake_artifact(vet_tier=3, vector_sim=1.0)
    assert exposure.classify(art, confidence=0.99) == "quarantine"


def test_t2_quarantined_in_phase2(fake_artifact):
    # Phase 2: no sandbox — T2 stays quarantined-until-founder-promotes.
    art = fake_artifact(vet_tier=2, vector_sim=1.0)
    assert exposure.classify(art, confidence=0.99) == "quarantine"


def test_below_theta_min_dropped(fake_artifact):
    art = fake_artifact(vet_tier=0)
    assert exposure.classify(art, confidence=0.10) == "quarantine"


def test_t0_mechanizable_shell_recipe_high_conf_is_preempt(fake_artifact):
    art = fake_artifact(
        vet_tier=0, kind="shell_recipe", mechanizable=True,
        artifact_type="skill",
    )
    assert exposure.classify(art, confidence=0.90) == "preempt"


def test_t1_shell_recipe_never_preempt(fake_artifact):
    # T1 ceiling excludes preempt even for mechanizable shell recipes.
    art = fake_artifact(
        vet_tier=1, kind="shell_recipe", mechanizable=True,
    )
    assert exposure.classify(art, confidence=0.95) == "inject"


def test_api_artifact_is_tool(fake_artifact):
    art = fake_artifact(vet_tier=0, artifact_type="api", kind=None)
    assert exposure.classify(art, confidence=0.60) == "tool"


def test_mcp_artifact_is_tool(fake_artifact):
    art = fake_artifact(vet_tier=0, artifact_type="mcp", kind=None)
    assert exposure.classify(art, confidence=0.60) == "tool"


def test_prompt_skill_is_inject(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="prompt_skill")
    assert exposure.classify(art, confidence=0.70) == "inject"


def test_agent_config_is_inject(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="agent_config")
    assert exposure.classify(art, confidence=0.70) == "inject"


def test_render_prebind_when_parametric_and_bound(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="shell_recipe", mechanizable=True,
                        inputs_schema={"name": {"type": "string"}})
    # below θ_preempt but parametric + fully bound → prebind inject
    assert exposure.render_variant(art, bound_args={"name": "x"}) == "prebind"


def test_render_prose_when_unbound(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="shell_recipe",
                        inputs_schema={"name": {"type": "string"}})
    assert exposure.render_variant(art, bound_args=None) == "prose"


def test_render_prose_for_non_parametric(fake_artifact):
    art = fake_artifact(vet_tier=0, kind="prompt_skill", inputs_schema={})
    assert exposure.render_variant(art, bound_args={}) == "prose"
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_exposure.py` — expected **FAIL** (`classify` / `render_variant` not defined).
- [ ] Replace `packages/intersect/src/intersect/exposure.py` with the full module:

```python
"""Exposure-class decision: (tier × kind × confidence) → exposure_class.

Phase 2 exposure classes: inject, tool, preempt, quarantine. NO sandbox
(v1.1). T2 artifacts are treated like T3 — quarantined until a founder
promotes them. The tier ceiling is the hard cap; intersect floors the
exposure based on the task confidence against the θ thresholds.

Tier → eligible exposure ceiling (spec Tier-classifier table):
  T0 → inject, tool, preempt
  T1 → inject, tool          (no preempt)
  T2 → quarantine            (Phase 2; sandbox is v1.1)
  T3 → quarantine
"""
from __future__ import annotations

# θ_preempt > θ_inject > θ_tool > θ_min. Conservative defaults; lowered
# later based on yalayut_usage success-rate telemetry.
THETA_PREEMPT: float = 0.80
THETA_INJECT: float = 0.55
THETA_TOOL: float = 0.45
THETA_MIN: float = 0.30

# Artifact types / kinds that are callable (tool exposure).
_CALLABLE_TYPES = frozenset({"api", "mcp"})
# Skill kinds that are mechanizable recipe shapes.
_RECIPE_KINDS = frozenset({"shell_recipe", "procedure"})


def classify(artifact, *, confidence: float) -> str:
    """Decide the exposure class for one matched artifact.

    Returns one of: 'inject', 'tool', 'preempt', 'quarantine'.
    """
    tier = int(getattr(artifact, "vet_tier", 3) or 3)

    # Tier ceiling — T2/T3 never surface in Phase 2.
    if tier >= 2:
        return "quarantine"

    # Below the floor — not worth exposing.
    if confidence < THETA_MIN:
        return "quarantine"

    artifact_type = getattr(artifact, "artifact_type", "skill")
    kind = getattr(artifact, "kind", None)
    mechanizable = bool(getattr(artifact, "mechanizable", False))

    # preempt — T0 only, mechanizable recipe, high confidence.
    if (tier == 0
            and artifact_type == "skill"
            and kind in _RECIPE_KINDS
            and mechanizable
            and confidence >= THETA_PREEMPT):
        return "preempt"

    # tool — callable artifacts (api verbs, mcp tools).
    if artifact_type in _CALLABLE_TYPES:
        if confidence >= THETA_TOOL:
            return "tool"
        return "quarantine"

    # inject — everything skill-shaped above θ_inject.
    if confidence >= THETA_INJECT:
        return "inject"
    return "quarantine"


def render_variant(artifact, *, bound_args: dict | None) -> str:
    """Pick the inject render sub-variant: 'prose' | 'prebind'.

    'prebind' only when the artifact is parametric (has inputs_schema)
    AND every required field is statically bound. Otherwise 'prose'.
    """
    inputs_schema = getattr(artifact, "inputs_schema", None) or {}
    if not inputs_schema:
        return "prose"
    if not bound_args:
        return "prose"
    # All schema fields must be present in bound_args for a prebind render.
    for field in inputs_schema:
        if field not in bound_args or bound_args[field] is None:
            return "prose"
    return "prebind"
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_exposure.py` — expected **PASS**.
- [ ] Commit: `feat(intersect): exposure-class decision matrix (no sandbox in phase 2)`

---

## Task 4 — Static arg-binding + bind cache

Static-only binding (Phase 2 has **no LLM-bind**): resolve each
`inputs_schema.<field>.bind_from` dotted path against the task context. If all
required fields fill → `prebind` render. If any null → consult
`yalayut_bind_cache` (embedding-keyed, hit ≥ 0.92). Cold cache-miss → return
`None` so flash falls back to plain prose `inject`.

**Files:**
- Create: `packages/intersect/src/intersect/binding.py`
- Test: `packages/intersect/tests/test_binding.py`

**Steps:**

- [ ] Write failing test `packages/intersect/tests/test_binding.py`:

```python
"""Unit tests for intersect.binding — static bind + cache."""
import json

import pytest

from intersect import binding


@pytest.fixture
def task_ctx():
    return {
        "task": {
            "title": "workout-tracker",
            "parent_mission": {"payload": {"project_name": "workout-tracker",
                                           "use_celery": True}},
        },
    }


def test_resolve_dotted_path_hit(task_ctx):
    val = binding._resolve_path(
        task_ctx, "task.parent_mission.payload.project_name")
    assert val == "workout-tracker"


def test_resolve_dotted_path_miss(task_ctx):
    assert binding._resolve_path(task_ctx, "task.nonexistent.field") is None


def test_static_bind_all_fields_filled(fake_artifact, task_ctx):
    art = fake_artifact(
        kind="shell_recipe", mechanizable=True,
        inputs_schema={
            "project_name": {
                "type": "string",
                "bind_from": ["task.parent_mission.payload.project_name",
                              "task.title"],
            },
            "use_celery": {
                "type": "bool",
                "bind_from": ["task.parent_mission.payload.use_celery"],
                "default": False,
            },
        },
    )
    args, complete = binding.static_bind(art, task_ctx)
    assert complete is True
    assert args == {"project_name": "workout-tracker", "use_celery": True}


def test_static_bind_uses_default_when_path_misses(fake_artifact):
    art = fake_artifact(
        kind="shell_recipe",
        inputs_schema={
            "use_celery": {"type": "bool", "bind_from": ["task.missing"],
                            "default": False},
        },
    )
    args, complete = binding.static_bind(art, {"task": {}})
    assert complete is True
    assert args == {"use_celery": False}


def test_static_bind_incomplete_when_required_field_unbound(fake_artifact):
    art = fake_artifact(
        kind="shell_recipe",
        inputs_schema={
            "project_name": {"type": "string", "bind_from": ["task.missing"]},
        },
    )
    args, complete = binding.static_bind(art, {"task": {}})
    assert complete is False
    assert args.get("project_name") is None


def test_static_bind_non_parametric_returns_empty(fake_artifact):
    art = fake_artifact(kind="prompt_skill", inputs_schema={})
    args, complete = binding.static_bind(art, {"task": {}})
    assert args == {}
    assert complete is True


@pytest.mark.asyncio
async def test_bind_cache_roundtrip(intersect_db, fake_artifact):
    art = fake_artifact(id=77, kind="shell_recipe",
                        inputs_schema={"project_name": {"type": "string"}})
    ctx = {"task": {"title": "x"}}
    miss = await binding.lookup_bind_cache(art, ctx)
    assert miss is None
    await binding.write_bind_cache(art, ctx, {"project_name": "x"})
    hit = await binding.lookup_bind_cache(art, ctx)
    assert hit == {"project_name": "x"}
```

- [ ] Add the `intersect_db` fixture to `packages/intersect/tests/conftest.py`
  (an in-memory DB seeded with the Phase 2 read tables):

```python
@pytest.fixture
async def intersect_db(monkeypatch):
    """In-memory aiosqlite DB with the yalayut tables intersect reads/writes.

    Phase 1's migration owns the canonical schema; here we create only the
    columns Phase 2 touches so intersect tests do not depend on Phase 1
    code being importable.
    """
    import aiosqlite

    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(
        """
        CREATE TABLE yalayut_sources (
          source_id TEXT, trust_score REAL DEFAULT 0.3
        );
        CREATE TABLE yalayut_owners (
          owner_id TEXT, trust_score REAL DEFAULT 0.3
        );
        CREATE TABLE yalayut_usage (
          id INTEGER PRIMARY KEY, artifact_id INTEGER, task_id TEXT,
          exposure_class TEXT, bind_args_json TEXT, exposed BOOLEAN,
          called BOOLEAN, succeeded BOOLEAN, latency_ms INTEGER,
          conflict_loser BOOLEAN, would_have_used INTEGER,
          escape_reason TEXT, occurred_at TIMESTAMP
        );
        CREATE TABLE yalayut_bind_cache (
          id INTEGER PRIMARY KEY, manifest_id INTEGER,
          ctx_embedding BLOB, bound_args_json TEXT,
          hit_count INTEGER DEFAULT 0, created_at TIMESTAMP,
          last_used_at TIMESTAMP
        );
        """
    )
    await conn.commit()

    async def _get_db():
        return conn

    # intersect modules read the DB via src.infra.db.get_db; patch it.
    import src.infra.db as _db
    monkeypatch.setattr(_db, "get_db", _get_db)
    yield conn
    await conn.close()
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_binding.py` — expected **FAIL**.
- [ ] Create `packages/intersect/src/intersect/binding.py`:

```python
"""Static arg-binding + bind cache for parametric artifacts.

Phase 2 is static-only — NO LLM-bind. Pipeline per matched parametric
artifact:
  1. try each inputs_schema.<field>.bind_from dotted path against the
     task context; first non-null wins; fall back to declared default.
  2. if every field filled → prebind-ready (complete=True).
  3. if any null → caller consults the embedding-keyed bind cache.
  4. cold cache-miss → caller renders plain prose inject (no LLM call).

The bind cache is keyed by an embedding of the relevant task-context
fields; a hit ≥ BIND_CACHE_HIT_THRESHOLD reuses the cached args.
"""
from __future__ import annotations

import json
import struct

from src.infra.logging_config import get_logger

logger = get_logger("intersect.binding")

BIND_CACHE_HIT_THRESHOLD: float = 0.92


def _resolve_path(ctx: dict, dotted: str):
    """Walk a dotted path through nested dicts; return None on any miss."""
    cur = ctx
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def static_bind(artifact, task_ctx: dict) -> tuple[dict, bool]:
    """Resolve all inputs_schema fields statically.

    Returns ``(bound_args, complete)``. ``complete`` is True when every
    schema field resolved (via a bind_from path or a declared default).
    Non-parametric artifacts return ``({}, True)``.
    """
    schema = getattr(artifact, "inputs_schema", None) or {}
    if not schema:
        return {}, True

    bound: dict = {}
    complete = True
    for field, rules in schema.items():
        if not isinstance(rules, dict):
            continue
        value = None
        for path in rules.get("bind_from", []) or []:
            value = _resolve_path(task_ctx, str(path))
            if value is not None:
                break
        if value is None and "default" in rules:
            value = rules["default"]
        bound[field] = value
        if value is None:
            complete = False
    return bound, complete


def _embed_ctx(task_ctx: dict) -> bytes:
    """Embed a stable JSON of the task context for cache keying.

    Uses the project embedding model (multilingual-e5-base, 768d).
    Falls back to an empty blob on any failure — a missing embedding
    simply means every cache lookup misses (safe degrade).
    """
    try:
        from src.memory.embeddings import embed_text
        text = json.dumps(task_ctx, sort_keys=True, ensure_ascii=False)[:2000]
        vec = embed_text(text)
        return struct.pack(f"{len(vec)}f", *vec)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("bind-cache embed failed: %s", exc)
        return b""


def _cosine(a: bytes, b: bytes) -> float:
    """Cosine similarity between two packed float32 blobs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    n = len(a) // 4
    va = struct.unpack(f"{n}f", a)
    vb = struct.unpack(f"{n}f", b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(y * y for y in vb) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def lookup_bind_cache(artifact, task_ctx: dict) -> dict | None:
    """Return cached bound args if a row matches ctx ≥ threshold, else None."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        target = _embed_ctx(task_ctx)
        if not target:
            return None
        cur = await db.execute(
            "SELECT id, ctx_embedding, bound_args_json "
            "FROM yalayut_bind_cache WHERE manifest_id = ?",
            (getattr(artifact, "id", None),),
        )
        rows = await cur.fetchall()
        await cur.close()
        best_id = None
        best_args = None
        best_sim = 0.0
        for row_id, emb, args_json in rows:
            sim = _cosine(target, emb or b"")
            if sim >= BIND_CACHE_HIT_THRESHOLD and sim > best_sim:
                best_sim, best_id, best_args = sim, row_id, args_json
        if best_id is None:
            return None
        await db.execute(
            "UPDATE yalayut_bind_cache "
            "SET hit_count = hit_count + 1, last_used_at = datetime('now') "
            "WHERE id = ?",
            (best_id,),
        )
        await db.commit()
        return json.loads(best_args) if best_args else None
    except Exception as exc:
        logger.debug("bind-cache lookup failed: %s", exc)
        return None


async def write_bind_cache(artifact, task_ctx: dict, bound_args: dict) -> None:
    """Persist a freshly-bound arg set for future cache hits."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        emb = _embed_ctx(task_ctx)
        await db.execute(
            "INSERT INTO yalayut_bind_cache "
            "(manifest_id, ctx_embedding, bound_args_json, hit_count, "
            " created_at, last_used_at) "
            "VALUES (?, ?, ?, 0, datetime('now'), datetime('now'))",
            (getattr(artifact, "id", None), emb,
             json.dumps(bound_args, ensure_ascii=False)),
        )
        await db.commit()
    except Exception as exc:
        logger.debug("bind-cache write failed: %s", exc)
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_binding.py` — expected **PASS**.
- [ ] Commit: `feat(intersect): static arg-binding pipeline + embedding-keyed bind cache`

---

## Task 5 — Budget caps

`api ≤ 3` surfaced per step; `mcp tools ≤ 3` per server, `≤ 6` per step total.
Caps apply to the post-classify list of `SkillApplication` dicts; `inject` /
`preempt` entries are never capped.

**Files:**
- Create: `packages/intersect/src/intersect/budget.py`
- Test: `packages/intersect/tests/test_budget.py`

**Steps:**

- [ ] Write failing test `packages/intersect/tests/test_budget.py`:

```python
"""Unit tests for intersect.budget caps."""
from intersect import budget


def _app(artifact_type, exposure_class="tool", server="srv-a",
         confidence=0.6, name="x", artifact_id=1):
    return {
        "artifact_id": artifact_id,
        "name": name,
        "artifact_type": artifact_type,
        "exposure_class": exposure_class,
        "mcp_server": server,
        "confidence": confidence,
    }


def test_api_capped_at_three_per_step():
    apps = [_app("api", confidence=0.9 - i * 0.1, artifact_id=i)
            for i in range(6)]
    kept, dropped = budget.apply_caps(apps)
    api_kept = [a for a in kept if a["artifact_type"] == "api"]
    assert len(api_kept) == 3
    assert len(dropped) == 3
    # Highest-confidence ones survive.
    assert all(a["confidence"] >= 0.6 for a in api_kept)


def test_mcp_capped_three_per_server():
    apps = [_app("mcp", server="srv-a", confidence=0.9 - i * 0.05,
                 artifact_id=i) for i in range(5)]
    kept, dropped = budget.apply_caps(apps)
    assert len([a for a in kept if a["artifact_type"] == "mcp"]) == 3
    assert len(dropped) == 2


def test_mcp_total_capped_six_per_step():
    apps = []
    for srv in ("a", "b", "c"):
        apps += [_app("mcp", server=srv, confidence=0.8,
                      artifact_id=hash((srv, i)) & 0xFFFF) for i in range(3)]
    kept, _ = budget.apply_caps(apps)
    assert len([a for a in kept if a["artifact_type"] == "mcp"]) == 6


def test_inject_and_preempt_never_capped():
    apps = [_app("skill", exposure_class="inject", artifact_id=i)
            for i in range(10)]
    apps.append(_app("skill", exposure_class="preempt", artifact_id=99))
    kept, dropped = budget.apply_caps(apps)
    assert len(kept) == 11
    assert dropped == []
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_budget.py` — expected **FAIL**.
- [ ] Create `packages/intersect/src/intersect/budget.py`:

```python
"""Per-step budget caps for tool-class exposures.

api  → at most API_CAP surfaced per step
mcp  → at most MCP_PER_SERVER tools per server, MCP_TOTAL per step total

inject / preempt exposures are skill-shaped and never capped. Ranking is
by confidence descending; the lowest-confidence overflow is dropped.
Dropped entries are returned so telemetry can log them as not-exposed.
"""
from __future__ import annotations

API_CAP: int = 3
MCP_PER_SERVER: int = 3
MCP_TOTAL: int = 6


def apply_caps(applications: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return ``(kept, dropped)`` after applying api/mcp budget caps."""
    kept: list[dict] = []
    dropped: list[dict] = []

    # inject / preempt — pass through untouched.
    for app in applications:
        if app.get("exposure_class") in ("inject", "preempt"):
            kept.append(app)

    # api — global per-step cap.
    apis = sorted(
        (a for a in applications
         if a.get("artifact_type") == "api"
         and a.get("exposure_class") == "tool"),
        key=lambda a: a.get("confidence", 0.0),
        reverse=True,
    )
    kept.extend(apis[:API_CAP])
    dropped.extend(apis[API_CAP:])

    # mcp — per-server cap then per-step total cap.
    mcps = sorted(
        (a for a in applications
         if a.get("artifact_type") == "mcp"
         and a.get("exposure_class") == "tool"),
        key=lambda a: a.get("confidence", 0.0),
        reverse=True,
    )
    per_server: dict[str, int] = {}
    mcp_kept: list[dict] = []
    for app in mcps:
        srv = app.get("mcp_server") or "_default"
        if per_server.get(srv, 0) >= MCP_PER_SERVER:
            dropped.append(app)
            continue
        if len(mcp_kept) >= MCP_TOTAL:
            dropped.append(app)
            continue
        per_server[srv] = per_server.get(srv, 0) + 1
        mcp_kept.append(app)
    kept.extend(mcp_kept)

    return kept, dropped
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_budget.py` — expected **PASS**.
- [ ] Commit: `feat(intersect): per-step api/mcp budget caps`

---

## Task 6 — Telemetry: `yalayut_usage` writes

Every `flash()` emits one `yalayut_usage` row per artifact it considered:
exposed entries with their `exposure_class` + `bind_args_json`, and
conflict-losers (same-slot same-kind collisions where a higher-score sibling
won) with `conflict_loser=1`.

**Files:**
- Create: `packages/intersect/src/intersect/telemetry.py`
- Test: `packages/intersect/tests/test_telemetry.py`

**Steps:**

- [ ] Write failing test `packages/intersect/tests/test_telemetry.py`:

```python
"""Unit tests for intersect.telemetry — yalayut_usage writes."""
import json

import pytest

from intersect import telemetry


@pytest.mark.asyncio
async def test_records_exposed_row(intersect_db):
    await telemetry.record_usage(
        task_id="4101",
        exposed=[{"artifact_id": 7, "exposure_class": "inject",
                  "bind_args": None}],
        conflict_losers=[],
    )
    cur = await intersect_db.execute(
        "SELECT artifact_id, task_id, exposure_class, exposed, "
        "conflict_loser FROM yalayut_usage")
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 7
    assert rows[0][1] == "4101"
    assert rows[0][2] == "inject"
    assert rows[0][3] == 1
    assert rows[0][4] == 0


@pytest.mark.asyncio
async def test_records_conflict_loser(intersect_db):
    await telemetry.record_usage(
        task_id="4101",
        exposed=[{"artifact_id": 1, "exposure_class": "inject",
                  "bind_args": None}],
        conflict_losers=[{"artifact_id": 2, "exposure_class": "inject"}],
    )
    cur = await intersect_db.execute(
        "SELECT artifact_id, conflict_loser, exposed FROM yalayut_usage "
        "ORDER BY artifact_id")
    rows = await cur.fetchall()
    assert rows[0] == (1, 0, 1)
    assert rows[1] == (2, 1, 0)


@pytest.mark.asyncio
async def test_records_bind_args_json(intersect_db):
    await telemetry.record_usage(
        task_id="4101",
        exposed=[{"artifact_id": 9, "exposure_class": "preempt",
                  "bind_args": {"name": "wt"}}],
        conflict_losers=[],
    )
    cur = await intersect_db.execute(
        "SELECT bind_args_json FROM yalayut_usage WHERE artifact_id = 9")
    row = await cur.fetchone()
    assert json.loads(row[0]) == {"name": "wt"}


@pytest.mark.asyncio
async def test_record_usage_never_raises(monkeypatch):
    # DB unavailable → telemetry must swallow, never propagate.
    import src.infra.db as _db

    async def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(_db, "get_db", _boom)
    # Must not raise.
    await telemetry.record_usage(task_id="x", exposed=[], conflict_losers=[])
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_telemetry.py` — expected **FAIL**.
- [ ] Create `packages/intersect/src/intersect/telemetry.py`:

```python
"""yalayut_usage telemetry writes.

flash() emits one row per artifact it considered: exposed entries carry
their exposure_class + bind_args_json; conflict-losers (a same-slot,
same-kind collision outranked by a sibling) carry conflict_loser=1.
Telemetry never raises — a logging failure must not break dispatch.
"""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger

logger = get_logger("intersect.telemetry")


async def record_usage(
    *,
    task_id: str,
    exposed: list[dict],
    conflict_losers: list[dict],
) -> None:
    """Write yalayut_usage rows for one flash() call. Never raises."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        for app in exposed:
            await db.execute(
                "INSERT INTO yalayut_usage "
                "(artifact_id, task_id, exposure_class, bind_args_json, "
                " exposed, called, succeeded, conflict_loser, occurred_at) "
                "VALUES (?, ?, ?, ?, 1, 0, 0, 0, datetime('now'))",
                (
                    app.get("artifact_id"),
                    str(task_id),
                    app.get("exposure_class"),
                    json.dumps(app["bind_args"], ensure_ascii=False)
                    if app.get("bind_args") is not None else None,
                ),
            )
        for loser in conflict_losers:
            await db.execute(
                "INSERT INTO yalayut_usage "
                "(artifact_id, task_id, exposure_class, exposed, called, "
                " succeeded, conflict_loser, occurred_at) "
                "VALUES (?, ?, ?, 0, 0, 0, 1, datetime('now'))",
                (
                    loser.get("artifact_id"),
                    str(task_id),
                    loser.get("exposure_class"),
                ),
            )
        await db.commit()
    except Exception as exc:
        logger.debug("yalayut_usage write failed: %s", exc)
```

- [ ] Run: `timeout 30 pytest packages/intersect/tests/test_telemetry.py` — expected **PASS**.
- [ ] Commit: `feat(intersect): yalayut_usage telemetry writes with conflict-loser rows`

---

## Task 7 — `flash(task)` orchestration

The entry function. Per spec's "The intersect" flow: query yalayut → score →
budget caps → exposure decision → static-bind → preempt routes to mechanical
lane / else attach `task["skills"]` → emit telemetry. Any error → graceful
degrade (`task["skills"] = []`, return task).

**Files:**
- Modify: `packages/intersect/src/intersect/flash.py` (replace placeholder)
- Test: `packages/intersect/tests/test_flash.py`

**Steps:**

- [ ] Write failing test `packages/intersect/tests/test_flash.py`:

```python
"""Integration tests for intersect.flash."""
import json

import pytest

from intersect import flash as flash_mod


@pytest.fixture
def patch_yalayut(monkeypatch, fake_artifact):
    """Patch yalayut.query + trust lookups so flash runs without Phase 1."""
    def _install(artifacts):
        async def _query(task_ctx):
            return list(artifacts)
        import yalayut
        monkeypatch.setattr(yalayut, "query", _query, raising=False)
    return _install


@pytest.mark.asyncio
async def test_flash_attaches_inject_envelope(
    intersect_db, sample_task, patch_yalayut, fake_artifact, monkeypatch,
):
    patch_yalayut([fake_artifact(
        id=1, kind="prompt_skill", vet_tier=0, vector_sim=0.9,
        name="anthropics-pdf",
    )])
    # source/owner trust default to 1.0 when rows absent (see _trust).
    out = await flash_mod.flash(sample_task)
    skills = out["skills"]
    assert isinstance(skills, list)
    assert len(skills) == 1
    s = skills[0]
    assert s["exposure_class"] == "inject"
    assert s["applies_to"] == "execution"
    assert s["render"] in ("prose", "prebind")
    assert s["name"] == "anthropics-pdf"
    assert "payload" in s
    assert isinstance(s["confidence"], float)


@pytest.mark.asyncio
async def test_flash_routes_preempt_to_mechanical_lane(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    patch_yalayut([fake_artifact(
        id=18, kind="shell_recipe", mechanizable=True, vet_tier=0,
        vector_sim=1.0, name="cc-pypackage",
        inputs_schema={},  # non-parametric → fully bound trivially
    )])
    out = await flash_mod.flash(sample_task)
    # preempt does not ride the envelope.
    assert out["skills"] == []
    assert out["runner"] == "mechanical"
    payload = out["payload"]
    assert payload["action"] == "yalayut_recipe"
    assert payload["recipe_id"] == 18
    assert "args" in payload


@pytest.mark.asyncio
async def test_flash_quarantine_artifacts_excluded(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    patch_yalayut([fake_artifact(id=3, vet_tier=2, vector_sim=1.0)])
    out = await flash_mod.flash(sample_task)
    assert out["skills"] == []


@pytest.mark.asyncio
async def test_flash_skips_query_when_recipe_lookup_false(
    intersect_db, patch_yalayut, fake_artifact,
):
    called = {"hit": False}

    async def _query(task_ctx):
        called["hit"] = True
        return []

    import yalayut
    import pytest as _pt
    _pt.MonkeyPatch().setattr(yalayut, "query", _query, raising=False)
    task = {
        "id": 5, "title": "Design the architecture", "description": "",
        "context": json.dumps({"recipe_lookup": False}),
    }
    out = await flash_mod.flash(task)
    assert out["skills"] == []


@pytest.mark.asyncio
async def test_flash_graceful_degrade_on_error(
    intersect_db, sample_task, monkeypatch,
):
    async def _boom(task_ctx):
        raise RuntimeError("yalayut exploded")

    import yalayut
    monkeypatch.setattr(yalayut, "query", _boom, raising=False)
    out = await flash_mod.flash(sample_task)
    # Error must not propagate; skills defaults to empty.
    assert out["skills"] == []


@pytest.mark.asyncio
async def test_flash_conflict_loser_dropped_and_logged(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    patch_yalayut([
        fake_artifact(id=10, kind="agent_config", vet_tier=0, vector_sim=0.9,
                      name="wshobson-backend-architect"),
        fake_artifact(id=11, kind="agent_config", vet_tier=0, vector_sim=0.6,
                      name="wshobson-security-auditor"),
    ])
    out = await flash_mod.flash(sample_task)
    # Two agent_config skills collide on the same slot — only the
    # highest-score one is kept.
    agent_cfgs = [s for s in out["skills"]
                  if s["name"].startswith("wshobson-")]
    assert len(agent_cfgs) == 1
    assert agent_cfgs[0]["artifact_id"] == 10
    cur = await intersect_db.execute(
        "SELECT artifact_id FROM yalayut_usage WHERE conflict_loser = 1")
    losers = await cur.fetchall()
    assert (11,) in losers


@pytest.mark.asyncio
async def test_flash_writes_usage_telemetry(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    patch_yalayut([fake_artifact(id=1, kind="prompt_skill", vet_tier=0,
                                 vector_sim=0.9)])
    await flash_mod.flash(sample_task)
    cur = await intersect_db.execute(
        "SELECT exposure_class FROM yalayut_usage WHERE exposed = 1")
    rows = await cur.fetchall()
    assert rows and rows[0][0] == "inject"
```

- [ ] Run: `timeout 60 pytest packages/intersect/tests/test_flash.py` — expected **FAIL**.
- [ ] Replace `packages/intersect/src/intersect/flash.py` with the full body:

```python
"""flash(task) — the intersect entry point.

Invoked once per task by the orchestrator pump, before dispatch. Flow
(spec "The intersect" section):

  1. candidates = yalayut.query(task_ctx)
  2. score each: confidence = vector_sim × source_trust × owner_trust
                              × hint_bonus
  3. budget caps: api ≤3/step, mcp tools ≤3/server ≤6/step
  4. exposure_class per candidate from (tier × kind × confidence)
  5. static-bind args for preempt + parametric inject
  6. preempt  → route task to mechanical lane (runner=mechanical,
                payload.action=yalayut_recipe)
     others   → attach task["skills"] = list[dict] envelope
  7. emit yalayut_usage telemetry

Errors anywhere → graceful degrade: task["skills"] = [], return task.
Never imports LLMDispatcher (Phase 2 has no LLM-bind).
"""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger

from intersect import binding, budget, exposure, scoring, telemetry

logger = get_logger("intersect.flash")


def _parse_context(task: dict) -> dict:
    """Best-effort parse of task['context'] into a dict."""
    ctx = task.get("context")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    return ctx if isinstance(ctx, dict) else {}


def _build_task_ctx(task: dict, ctx: dict) -> dict:
    """Assemble the binding context — the dict bind_from paths walk.

    Exposes ``task`` (title/description/mission/payload) so manifest
    bind_from paths like ``task.parent_mission.payload.project_name``
    resolve. ``parent_mission.payload`` is populated from the task
    context's own payload bucket when present.
    """
    return {
        "task": {
            "id": task.get("id"),
            "title": task.get("title", ""),
            "description": task.get("description", ""),
            "mission_id": task.get("mission_id"),
            "parent_mission": {"payload": ctx.get("payload", {})},
            "context": ctx,
        },
    }


async def _trust(table: str, id_col: str, ident: str | None) -> float:
    """Look up a trust score from yalayut_sources / yalayut_owners.

    Defaults to 1.0 when the row is absent — a missing trust row must
    not silently zero out every confidence (that would suppress all
    matches). Phase 1 seeds trusted sources/owners; an unseeded ident
    is treated as neutral here and capped by the tier classifier.
    """
    if not ident:
        return 1.0
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            f"SELECT trust_score FROM {table} WHERE {id_col} = ?", (ident,),
        )
        row = await cur.fetchone()
        await cur.close()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as exc:
        logger.debug("trust lookup failed (%s): %s", table, exc)
    return 1.0


def _slot_key(artifact) -> str:
    """Conflict-resolution slot key — same kind competes for one slot.

    agent_config skills compete with each other; prompt_skills do not
    (multiple prose hints stack fine). Recipe kinds compete per kind.
    """
    kind = getattr(artifact, "kind", None)
    if kind in ("agent_config",):
        return f"slot:{kind}"
    return f"id:{getattr(artifact, 'id', None)}"  # unique → no collision


async def flash(task: dict) -> dict:
    """Match skills, attach the task['skills'] envelope, route preempt.

    Always returns the task with a ``skills`` key (possibly empty).
    """
    task.setdefault("skills", [])
    try:
        ctx = _parse_context(task)

        # recipe_lookup gate — design/architecture/debug steps opt out.
        if ctx.get("recipe_lookup") is False:
            return task

        import yalayut
        task_ctx = _build_task_ctx(task, ctx)
        candidates = await yalayut.query(task_ctx)
        if not candidates:
            return task

        recipe_hint = ctx.get("recipe_hint")

        # ── score + classify each candidate ──
        scored: list[tuple[object, float]] = []
        for art in candidates:
            # env-gated artifacts (missing auth) are skipped silently.
            if getattr(art, "env_status", "ready") != "ready":
                continue
            src_trust = await _trust(
                "yalayut_sources", "source_id", getattr(art, "source", None))
            own_trust = await _trust(
                "yalayut_owners", "owner_id", getattr(art, "owner", None))
            hint_bonus = scoring.compute_hint_bonus(art, recipe_hint)
            conf = scoring.score_artifact(
                art, source_trust=src_trust, owner_trust=own_trust,
                hint_bonus=hint_bonus,
            )
            scored.append((art, conf))

        # ── conflict resolution — keep highest-score per slot ──
        best_per_slot: dict[str, tuple[object, float]] = {}
        conflict_losers: list[dict] = []
        for art, conf in sorted(scored, key=lambda p: p[1], reverse=True):
            key = _slot_key(art)
            if key in best_per_slot:
                conflict_losers.append({
                    "artifact_id": getattr(art, "id", None),
                    "exposure_class": "inject",
                })
                continue
            best_per_slot[key] = (art, conf)

        # ── build applications ──
        applications: list[dict] = []
        preempt_app: dict | None = None
        for art, conf in best_per_slot.values():
            klass = exposure.classify(art, confidence=conf)
            if klass == "quarantine":
                continue

            bound, complete = binding.static_bind(art, task_ctx)
            # parametric + incomplete → consult bind cache.
            if not complete and getattr(art, "inputs_schema", None):
                cached = await binding.lookup_bind_cache(art, task_ctx)
                if cached:
                    bound, complete = cached, True
            # newly-completed static bind → seed the cache.
            elif complete and bound and getattr(art, "inputs_schema", None):
                await binding.write_bind_cache(art, task_ctx, bound)

            if klass == "preempt":
                # preempt with unbound required fields downgrades to inject.
                if not complete:
                    klass = "inject"
                else:
                    preempt_app = {
                        "artifact_id": getattr(art, "id", None),
                        "exposure_class": "preempt",
                        "bind_args": bound,
                    }
                    # First preempt wins — recipe owns the whole task.
                    break

            render = exposure.render_variant(
                art, bound_args=bound if complete else None)
            applications.append({
                "artifact_id": getattr(art, "id", None),
                "name": getattr(art, "name", ""),
                "artifact_type": getattr(art, "artifact_type", "skill"),
                "exposure_class": klass,
                "applies_to": "execution",          # Phase 2: execution only
                "render": render,
                "mcp_server": getattr(art, "name", None)
                if getattr(art, "artifact_type", "") == "mcp" else None,
                "payload": {
                    "body": getattr(art, "body", ""),
                    "kind": getattr(art, "kind", None),
                    "bound_args": bound if complete else None,
                },
                "confidence": conf,
                "bind_args": bound if complete else None,
            })

        # ── preempt path — route to mechanical lane, no envelope ──
        if preempt_app is not None:
            task["runner"] = "mechanical"
            task["payload"] = {
                "action": "yalayut_recipe",
                "recipe_id": preempt_app["artifact_id"],
                "args": preempt_app["bind_args"],
            }
            task["skills"] = []
            await telemetry.record_usage(
                task_id=str(task.get("id", "")),
                exposed=[preempt_app],
                conflict_losers=conflict_losers,
            )
            return task

        # ── budget caps ──
        kept, dropped = budget.apply_caps(applications)

        # ── attach envelope (strip internal-only keys) ──
        envelope: list[dict] = []
        for app in kept:
            envelope.append({
                "artifact_id": app["artifact_id"],
                "name": app["name"],
                "exposure_class": app["exposure_class"],
                "applies_to": app["applies_to"],
                "render": app["render"],
                "payload": app["payload"],
                "confidence": app["confidence"],
            })
        task["skills"] = envelope

        await telemetry.record_usage(
            task_id=str(task.get("id", "")),
            exposed=[
                {"artifact_id": a["artifact_id"],
                 "exposure_class": a["exposure_class"],
                 "bind_args": a.get("bind_args")}
                for a in kept
            ],
            conflict_losers=conflict_losers + [
                {"artifact_id": d["artifact_id"],
                 "exposure_class": d["exposure_class"]}
                for d in dropped
            ],
        )
        return task

    except Exception as exc:
        logger.warning("intersect.flash degraded for task %s: %s",
                        task.get("id"), exc)
        task["skills"] = []
        return task
```

- [ ] Run: `timeout 60 pytest packages/intersect/tests/test_flash.py` — expected **PASS**.
- [ ] Run the full intersect suite: `timeout 90 pytest packages/intersect/tests/` — expected **PASS**.
- [ ] Commit: `feat(intersect): flash() orchestration — query→score→classify→bind→attach/route→telemetry`

---

## Task 8 — Wire `intersect.flash` into the orchestrator pump

`run_loop()` calls `await intersect.flash(task)` once per task, after
`beckman.next_task()` and before `_dispatch`. The flash result (which may have
`runner=mechanical` + `payload` set for preempt) replaces the task passed to
`_dispatch`.

**Files:**
- Modify: `src/core/orchestrator.py`
- Test: `tests/core/test_orchestrator_intersect_wiring.py`

**Steps:**

- [ ] Write failing test `tests/core/test_orchestrator_intersect_wiring.py`:

```python
"""Verify the orchestrator pump calls intersect.flash before dispatch."""
import asyncio
import json

import pytest


@pytest.mark.asyncio
async def test_run_loop_calls_flash_before_dispatch(monkeypatch):
    from src.core.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)  # skip __init__ (Telegram)
    orch.running = True
    orch._shutting_down = False
    orch.shutdown_event = asyncio.Event()
    orch._running_futures = []
    orch.requested_exit_code = None

    seen = {"flashed": None, "dispatched": None}
    sample = {"id": 4242, "title": "t", "context": json.dumps({})}

    import general_beckman

    async def _next_task():
        orch.running = False  # one iteration only
        return dict(sample)

    monkeypatch.setattr(general_beckman, "next_task", _next_task)

    import intersect

    async def _flash(task):
        seen["flashed"] = task["id"]
        task["skills"] = [{"artifact_id": 1, "exposure_class": "inject"}]
        return task

    monkeypatch.setattr(intersect, "flash", _flash)

    async def _dispatch(task):
        seen["dispatched"] = task.get("skills")

    monkeypatch.setattr(orch, "_dispatch", _dispatch)

    # Avoid the workspace/git init branch + founder sweep import.
    import src.tools.git_ops as _git

    async def _noop_git(*a, **k):
        return None

    monkeypatch.setattr(_git, "ensure_git_repo", _noop_git)

    await orch.run_loop()
    # Give the fire-and-forget _dispatch task a tick.
    await asyncio.sleep(0)
    assert seen["flashed"] == 4242
    assert seen["dispatched"] == [{"artifact_id": 1,
                                   "exposure_class": "inject"}]
```

- [ ] Run: `timeout 60 pytest tests/core/test_orchestrator_intersect_wiring.py` — expected **FAIL** (flash not wired; `seen["flashed"]` is `None`).
- [ ] In `src/core/orchestrator.py`, inside `run_loop()`, change the dispatch block. Locate:

```python
                task = await general_beckman.next_task()
                if task is not None:
                    t = asyncio.create_task(self._dispatch(task))
                    self._running_futures.append(t)
                    t.add_done_callback(self._drop_running_future)
```

Replace with:

```python
                task = await general_beckman.next_task()
                if task is not None:
                    # Yalayut Phase 2 — match catalog skills for this task
                    # once, before dispatch. intersect.flash() attaches a
                    # task["skills"] envelope (coulson reads it) or, for a
                    # preempt recipe, sets runner=mechanical + payload so
                    # the task routes to mr_roboto. Imported lazily so the
                    # orchestrator module load doesn't pull yalayut. flash
                    # graceful-degrades internally — no try/except needed,
                    # but guard the import itself in case the package is
                    # absent in a stripped deploy.
                    try:
                        import intersect
                        task = await intersect.flash(task)
                    except Exception as _e:
                        logger.debug("intersect.flash skipped #%s: %s",
                                     task.get("id"), _e)
                        task.setdefault("skills", [])
                    t = asyncio.create_task(self._dispatch(task))
                    self._running_futures.append(t)
                    t.add_done_callback(self._drop_running_future)
```

- [ ] Run: `timeout 60 pytest tests/core/test_orchestrator_intersect_wiring.py` — expected **PASS**.
- [ ] Smoke: `python -c "import src.core.orchestrator"` — expected no error.
- [ ] Commit: `feat(orchestrator): wire intersect.flash into the dispatch pump`

---

## Task 9 — coulson skill renderer + envelope consumption

`coulson` reads `task["skills"]` filtered to `applies_to=execution`. `inject`
entries are rendered to prompt text by a **new** `coulson/skill_render.py` (per
spec — rendering lives in coulson, not intersect). `tool` entries feed the
per-execution `allowed_tools` list. This **replaces** the
`skills.find_relevant_skills` free-text branch in `build_user_context`.

**Files:**
- Create: `packages/coulson/src/coulson/skill_render.py`
- Modify: `packages/coulson/src/coulson/context.py`
- Test: `packages/coulson/tests/test_skill_render.py`
- Test: `packages/coulson/tests/test_context_envelope.py`

**Steps:**

- [ ] Write failing test `packages/coulson/tests/test_skill_render.py`:

```python
"""Unit tests for coulson.skill_render."""
from coulson import skill_render


def test_render_empty_envelope_is_empty():
    assert skill_render.render_skill_envelope([]) == ""


def test_render_prose_inject_block():
    env = [{
        "artifact_id": 1, "name": "anthropics-pdf",
        "exposure_class": "inject", "applies_to": "execution",
        "render": "prose",
        "payload": {"body": "Use the PDF skill to extract text.",
                    "kind": "prompt_skill"},
        "confidence": 0.8,
    }]
    block = skill_render.render_skill_envelope(env)
    assert "## Relevant Skills from Library" in block
    assert "anthropics-pdf" in block
    assert "Use the PDF skill to extract text." in block


def test_render_prebind_shows_concrete_call():
    env = [{
        "artifact_id": 2, "name": "cc-pypackage",
        "exposure_class": "inject", "applies_to": "execution",
        "render": "prebind",
        "payload": {"body": "Scaffold a Python package.",
                    "kind": "shell_recipe",
                    "bound_args": {"project_name": "wt"}},
        "confidence": 0.7,
    }]
    block = skill_render.render_skill_envelope(env)
    assert "cc-pypackage" in block
    # prebind renders the concrete call with bound args.
    assert "project_name" in block and "wt" in block


def test_render_filters_to_execution_only():
    env = [{
        "artifact_id": 3, "name": "rubric-x", "exposure_class": "inject",
        "applies_to": "grading", "render": "prose",
        "payload": {"body": "grading rubric"}, "confidence": 0.9,
    }]
    # grading-tagged skills are not for the agent prompt (Phase 2: none
    # exist, but the filter must hold).
    assert skill_render.render_skill_envelope(env) == ""


def test_render_skips_tool_class():
    env = [{
        "artifact_id": 4, "name": "api-coingecko", "exposure_class": "tool",
        "applies_to": "execution", "render": "prose",
        "payload": {"body": "x"}, "confidence": 0.6,
    }]
    # tool-class entries are not prose — they feed allowed_tools, not text.
    assert skill_render.render_skill_envelope(env) == ""


def test_tool_names_from_envelope():
    env = [
        {"artifact_id": 5, "name": "api-coingecko", "exposure_class": "tool",
         "applies_to": "execution", "render": "prose",
         "payload": {"body": "x"}, "confidence": 0.6},
        {"artifact_id": 6, "name": "anthropics-pdf",
         "exposure_class": "inject", "applies_to": "execution",
         "render": "prose", "payload": {"body": "x"}, "confidence": 0.8},
    ]
    tools = skill_render.tool_names_from_envelope(env)
    assert tools == ["api-coingecko"]
```

- [ ] Run: `timeout 30 pytest packages/coulson/tests/test_skill_render.py` — expected **FAIL**.
- [ ] Create `packages/coulson/src/coulson/skill_render.py`:

```python
"""Render the task['skills'] envelope into agent-prompt text.

Rendering ownership (spec): SkillApplication is structured data;
coulson — the agent-prompt builder — renders it. intersect never knows
prompt conventions. This module reads the plain-dict envelope intersect
attaches and produces the markdown block coulson appends to the user
context.

Phase 2 consumes the ``applies_to == "execution"`` slice only — grading
exposure is v1.1. ``inject``-class entries render to prose; ``tool``-class
entries feed the per-execution allowed_tools list (see
``tool_names_from_envelope``); ``preempt`` never rides the envelope.
"""
from __future__ import annotations


def _render_one(app: dict) -> str:
    """Render a single inject-class SkillApplication to a markdown block."""
    name = app.get("name", "unknown")
    payload = app.get("payload") or {}
    body = (payload.get("body") or "").strip()
    if app.get("render") == "prebind":
        bound = payload.get("bound_args") or {}
        arg_str = ", ".join(f"{k}: {v}" for k, v in bound.items())
        lines = [
            f"### Skill: {name} (ready-to-run)",
            f"`{name}({arg_str})`",
        ]
        if body:
            lines.append(body)
        return "\n".join(lines)
    # prose render
    lines = [f"### Skill: {name}"]
    if body:
        lines.append(body)
    return "\n".join(lines)


def render_skill_envelope(envelope: list[dict]) -> str:
    """Render the inject/execution slice of the envelope to prompt text.

    Returns an empty string when nothing applies. The heading matches
    the legacy ``format_skills_for_prompt`` heading so downstream
    expectations don't shift.
    """
    if not envelope:
        return ""
    inject = [
        a for a in envelope
        if a.get("exposure_class") == "inject"
        and a.get("applies_to") == "execution"
    ]
    if not inject:
        return ""
    blocks = ["## Relevant Skills from Library", ""]
    for app in inject:
        blocks.append(_render_one(app))
        blocks.append("")
    return "\n".join(blocks)


def tool_names_from_envelope(envelope: list[dict]) -> list[str]:
    """Return tool-class artifact names for per-execution allowed_tools."""
    return [
        a.get("name")
        for a in envelope
        if a.get("exposure_class") == "tool"
        and a.get("applies_to") == "execution"
        and a.get("name")
    ]
```

- [ ] Run: `timeout 30 pytest packages/coulson/tests/test_skill_render.py` — expected **PASS**.
- [ ] Write failing test `packages/coulson/tests/test_context_envelope.py`:

```python
"""build_user_context reads task['skills'] envelope, not skills.py."""
import json

import pytest


class _Profile:
    name = "coder"
    allowed_tools = ["read_file", "write_file"]
    max_iterations = 5
    _prompt_version_override = None
    _suppress_clarification = False

    def get_system_prompt(self, task):
        return "You are a coder."


@pytest.mark.asyncio
async def test_context_renders_envelope_inject(monkeypatch):
    from coulson import context

    task = {
        "id": 7, "title": "Build a thing", "description": "do it",
        "agent_type": "coder",
        "context": json.dumps({}),
        "skills": [{
            "artifact_id": 1, "name": "anthropics-pdf",
            "exposure_class": "inject", "applies_to": "execution",
            "render": "prose",
            "payload": {"body": "PDF extraction guidance.",
                        "kind": "prompt_skill"},
            "confidence": 0.8,
        }],
    }
    text, injected_tools = await context.build_user_context(
        _Profile(), task, model_ctx=4096)
    assert "anthropics-pdf" in text
    assert "PDF extraction guidance." in text


@pytest.mark.asyncio
async def test_context_envelope_tool_class_injects_tool(monkeypatch):
    from coulson import context

    task = {
        "id": 8, "title": "Fetch a price", "description": "",
        "agent_type": "researcher", "context": json.dumps({}),
        "skills": [{
            "artifact_id": 5, "name": "api-coingecko",
            "exposure_class": "tool", "applies_to": "execution",
            "render": "prose", "payload": {"body": "x"}, "confidence": 0.6,
        }],
    }
    _, injected_tools = await context.build_user_context(
        _Profile(), task, model_ctx=4096)
    assert "api-coingecko" in injected_tools


@pytest.mark.asyncio
async def test_context_no_envelope_renders_nothing_extra(monkeypatch):
    from coulson import context

    task = {
        "id": 9, "title": "Plain task", "description": "",
        "agent_type": "coder", "context": json.dumps({}),
    }
    text, injected = await context.build_user_context(
        _Profile(), task, model_ctx=4096)
    assert "Relevant Skills from Library" not in text
    assert injected == []
```

- [ ] Run: `timeout 60 pytest packages/coulson/tests/test_context_envelope.py` — expected **FAIL** (envelope not consumed).
- [ ] In `packages/coulson/src/coulson/context.py`, replace the **free-text skill-match branch** inside `build_user_context`. Locate the block under
  `# Free-text path: vector skill match for non-workflow tasks.` starting at
  `if not _step_id:` and ending at the matching `logger.debug("Skill injection failed: %s", exc)`.
  Replace that entire `if not _step_id:` block with:

```python
        # Free-text path: yalayut Phase 2 — render the task["skills"]
        # envelope intersect attached. The old src.memory.skills
        # free-text vector match is retired here; intersect now owns
        # matching, coulson owns rendering. Workflow tasks still skip
        # this path — exemplar lookup above is authoritative for them.
        if not _step_id:
            try:
                from coulson.skill_render import (
                    render_skill_envelope, tool_names_from_envelope,
                )
                envelope = task.get("skills") or []
                if envelope:
                    skills_block = render_skill_envelope(envelope)
                    if skills_block:
                        parts.append(skills_block)
                    for tool in tool_names_from_envelope(envelope):
                        if tool not in _injected_skills_tools:
                            _injected_skills_tools.append(tool)
                            logger.info(
                                "Skill-injected tool from envelope "
                                "(deferred to caller): %s", tool,
                            )
                    logger.info(
                        "Skills rendered from envelope: %s",
                        [a.get("name") for a in envelope],
                    )
            except Exception as exc:
                logger.debug("Envelope skill render failed: %s", exc)
```

- [ ] Run: `timeout 60 pytest packages/coulson/tests/test_context_envelope.py` — expected **PASS**.
- [ ] Smoke: `python -c "import coulson.context, coulson.skill_render"` — no error.
- [ ] Commit: `feat(coulson): consume task['skills'] envelope; render via skill_render`

---

## Task 10 — Convert `src/memory/skills.py` to a thin envelope shim

`find_relevant_skills` becomes a shim: when given a task carrying the envelope,
it returns the `inject` slice mapped to the legacy skill-dict shape.
`format_skills_for_prompt` must produce **byte-identical** output to today's
behaviour for the same legacy skill dicts (kept until coulson fully migrated;
this task only adds the shim path, the legacy formatters are untouched). A test
asserts byte-identity.

**Files:**
- Modify: `src/memory/skills.py`
- Test: `tests/memory/test_skills_shim.py`

**Steps:**

- [ ] Write failing test `tests/memory/test_skills_shim.py`:

```python
"""skills.py shim: envelope-aware find_relevant_skills + byte-identical
injection output."""
import json

import pytest

from src.memory import skills


def test_format_skills_byte_identical_for_legacy_dicts():
    """The legacy formatter output must not shift — the shim keeps
    format_skills_for_prompt verbatim. Frozen golden string."""
    legacy = [{
        "name": "shopping_search",
        "description": "Route shopping queries to the advisor",
        "injection_count": 0,
        "injection_success": 0,
        "strategies": [],
    }]
    out = skills.format_skills_for_prompt(legacy, context_budget=4096)
    expected = (
        "## Relevant Skills from Library\n\n"
        "- shopping_search: Route shopping queries to the advisor "
        "(tools: none, 0% success)\n"
    )
    assert out == expected


@pytest.mark.asyncio
async def test_find_relevant_skills_returns_envelope_inject_slice():
    """When a task carries the Phase 2 envelope, the shim returns the
    inject slice as legacy-shaped skill dicts — no vector search."""
    task = {
        "id": 11,
        "skills": [
            {"artifact_id": 1, "name": "anthropics-pdf",
             "exposure_class": "inject", "applies_to": "execution",
             "render": "prose",
             "payload": {"body": "PDF guidance"}, "confidence": 0.8},
            {"artifact_id": 2, "name": "api-coingecko",
             "exposure_class": "tool", "applies_to": "execution",
             "render": "prose", "payload": {"body": "x"}, "confidence": 0.5},
        ],
    }
    result = await skills.find_relevant_skills("anything", task=task)
    # tool-class excluded; only inject survives.
    assert len(result) == 1
    assert result[0]["name"] == "anthropics-pdf"
    assert result[0]["description"] == "PDF guidance"


@pytest.mark.asyncio
async def test_find_relevant_skills_empty_when_no_envelope():
    """No envelope → shim returns [] (the vector path is retired; coulson
    now drives matching via intersect)."""
    result = await skills.find_relevant_skills("anything", task={"id": 12})
    assert result == []
```

- [ ] Run: `timeout 30 pytest tests/memory/test_skills_shim.py` — expected **FAIL** (`find_relevant_skills` has no `task` kwarg).
- [ ] In `src/memory/skills.py`, replace the `find_relevant_skills` function (lines ~377–415) with the shim:

```python
async def find_relevant_skills(
    task_text: str, limit: int = 5, task: dict | None = None,
) -> list[dict]:
    """Thin shim over the yalayut Phase 2 envelope.

    Yalayut Phase 2 moved skill *matching* into the ``intersect``
    component (orchestrator pump calls ``intersect.flash`` once per task
    and attaches ``task["skills"]``). This function is kept only as a
    back-compat shim for callers that have not migrated to reading the
    envelope directly.

    When ``task`` carries a ``skills`` envelope, the inject/execution
    slice is returned mapped to the legacy skill-dict shape so
    ``format_skills_for_prompt`` still renders it byte-identically. With
    no envelope (or no task), returns [] — the old vector-search path is
    retired; coulson now drives matching through intersect.

    Deleted once coulson fully owns envelope rendering.
    """
    if not task:
        return []
    envelope = task.get("skills") or []
    if not envelope:
        return []
    out: list[dict] = []
    for app in envelope:
        if app.get("exposure_class") != "inject":
            continue
        if app.get("applies_to") != "execution":
            continue
        payload = app.get("payload") or {}
        out.append({
            "name": app.get("name", "unknown"),
            "description": (payload.get("body") or "").strip(),
            "injection_count": 0,
            "injection_success": 0,
            "strategies": [],
            "_match_score": float(app.get("confidence", 0.0) or 0.0),
            "_similarity": float(app.get("confidence", 0.0) or 0.0),
        })
    out.sort(key=lambda s: s["_match_score"], reverse=True)
    return out[:limit]
```

- [ ] Run: `timeout 30 pytest tests/memory/test_skills_shim.py` — expected **PASS**.
- [ ] Smoke: `python -c "from src.memory.skills import find_relevant_skills, format_skills_for_prompt"` — no error.
- [ ] Commit: `refactor(skills): convert find_relevant_skills to a yalayut envelope shim`

---

## Task 11 — `recipe_lookup` / `recipe_hint` workflow-step fields

Add per-step `recipe_lookup: true|false` and `recipe_hint` JSON fields. The
loader carries them through; the expander propagates them into the task context
so `intersect.flash` reads them. Default `recipe_lookup=true` for
scaffold/auth/api/deploy/test-setup/migration phases, `false` for
design/architecture/debugging/synthesis.

**Files:**
- Modify: `src/workflows/engine/loader.py`
- Modify: `src/workflows/engine/expander.py`
- Modify: `src/workflows/i2p/i2p_v3.json`
- Test: `tests/workflows/test_recipe_lookup_fields.py`

**Steps:**

- [ ] Inspect the current step shape to find where context is built. Read
  `src/workflows/engine/expander.py` and `src/workflows/engine/loader.py`,
  locating (a) `Workflow.get_step` and (b) the place the expander assembles a
  step's task `context` dict (the `is_workflow_step` / `workflow_step_id` keys).
- [ ] Write failing test `tests/workflows/test_recipe_lookup_fields.py`:

```python
"""recipe_lookup / recipe_hint carry from step JSON into task context."""
import json

import pytest

from src.workflows.engine.loader import load_workflow
from src.workflows.engine.expander import build_step_context


def test_loader_preserves_recipe_fields():
    """A step declaring recipe_lookup/recipe_hint keeps them on get_step."""
    wf = load_workflow("i2p_v3")
    # Find any step that declares recipe_lookup explicitly.
    found = None
    for sid in wf.all_step_ids():
        step = wf.get_step(sid)
        if step and "recipe_lookup" in step:
            found = step
            break
    assert found is not None, "no i2p step declares recipe_lookup"
    assert isinstance(found["recipe_lookup"], bool)


def test_expander_propagates_recipe_lookup_true():
    step = {
        "id": "3.2", "agent": "coder", "title": "Scaffold",
        "recipe_lookup": True, "recipe_hint": "python package scaffold",
    }
    ctx = build_step_context(step, mission_id=1)
    assert ctx["recipe_lookup"] is True
    assert ctx["recipe_hint"] == "python package scaffold"


def test_expander_default_recipe_lookup_for_scaffold_phase():
    """A scaffold-phase step with no explicit flag defaults to True."""
    step = {"id": "3.1", "agent": "coder", "title": "Scaffold the repo",
            "phase": "scaffold"}
    ctx = build_step_context(step, mission_id=1)
    assert ctx["recipe_lookup"] is True


def test_expander_default_recipe_lookup_false_for_design_phase():
    step = {"id": "2.1", "agent": "architect", "title": "Design architecture",
            "phase": "architecture"}
    ctx = build_step_context(step, mission_id=1)
    assert ctx["recipe_lookup"] is False


def test_expander_explicit_flag_overrides_phase_default():
    step = {"id": "2.9", "agent": "architect", "title": "x",
            "phase": "architecture", "recipe_lookup": True}
    ctx = build_step_context(step, mission_id=1)
    assert ctx["recipe_lookup"] is True
```

> If the expander has no `build_step_context` helper, this task extracts the
> context-assembly logic into one. If a differently-named helper already builds
> the per-step context dict, adapt the test imports to that name and add the
> recipe-field logic there instead — keep the behaviour, match the codebase.

- [ ] Run: `timeout 30 pytest tests/workflows/test_recipe_lookup_fields.py` — expected **FAIL**.
- [ ] In `src/workflows/engine/loader.py`, ensure `Workflow.get_step` returns the
  raw step dict unmodified (it already does — `recipe_lookup`/`recipe_hint` are
  just extra keys, no schema gate strips them). If the loader has a step-key
  allowlist, add `recipe_lookup` and `recipe_hint` to it. Add an `all_step_ids()`
  method to `Workflow` if absent:

```python
    def all_step_ids(self) -> list[str]:
        """Return every step id declared in this workflow."""
        steps = self._plan.get("steps", [])
        if isinstance(steps, dict):
            return list(steps.keys())
        return [s.get("id") for s in steps if isinstance(s, dict) and s.get("id")]
```

- [ ] In `src/workflows/engine/expander.py`, add the `build_step_context` helper
  (or fold the recipe-field logic into the existing context-assembly function).
  The recipe-default logic:

```python
# Phases whose steps default to recipe_lookup=True (well-known recipes
# fast-forward these): scaffold, auth, api, deploy, test-setup, migration.
# Everything else (design, architecture, debugging, synthesis) defaults
# to False — bespoke reasoning work, no external recipe applies.
_RECIPE_LOOKUP_PHASES = frozenset({
    "scaffold", "auth", "api", "deploy", "test-setup", "test_setup",
    "migration",
})
_NO_RECIPE_PHASES = frozenset({
    "design", "architecture", "debugging", "debug", "synthesis",
})


def _default_recipe_lookup(step: dict) -> bool:
    """Decide the default recipe_lookup when a step omits the flag.

    Resolution order: explicit step flag → phase-based default → False.
    The phase is read from the step's ``phase`` key, else inferred from
    a leading title keyword.
    """
    phase = (step.get("phase") or "").strip().lower()
    if phase in _RECIPE_LOOKUP_PHASES:
        return True
    if phase in _NO_RECIPE_PHASES:
        return False
    title = (step.get("title") or "").lower()
    if any(k in title for k in
           ("scaffold", "auth", "deploy", "migration", "test setup")):
        return True
    return False


def build_step_context(step: dict, *, mission_id: int) -> dict:
    """Assemble the per-step task context dict, including recipe fields.

    recipe_lookup: explicit step value if present, else phase default.
    recipe_hint:   passed through verbatim when the step declares it.
    """
    ctx: dict = {
        "is_workflow_step": True,
        "workflow_step_id": step.get("id"),
        "mission_id": mission_id,
    }
    if "recipe_lookup" in step:
        ctx["recipe_lookup"] = bool(step["recipe_lookup"])
    else:
        ctx["recipe_lookup"] = _default_recipe_lookup(step)
    if step.get("recipe_hint"):
        ctx["recipe_hint"] = step["recipe_hint"]
    return ctx
```

> Wire this helper into the existing step-context construction path: wherever
> the expander currently builds the per-step `context` dict, merge in the
> `recipe_lookup` / `recipe_hint` keys (call `build_step_context` and `update`
> the existing context, or inline the two-key logic). Do not duplicate the
> `is_workflow_step` / `workflow_step_id` keys if the expander already sets
> them — the test only asserts the recipe keys are present.

- [ ] In `src/workflows/i2p/i2p_v3.json`, add `"recipe_lookup": true` plus a
  `"recipe_hint"` string to scaffold / auth / api / deploy / test-setup /
  migration steps, and `"recipe_lookup": false` to design / architecture /
  debugging / synthesis steps. At minimum, annotate the package-scaffold step
  and one auth step so `test_loader_preserves_recipe_fields` finds an explicit
  declaration. Example for a scaffold step:

```json
{
  "id": "3.2",
  "agent": "coder",
  "title": "Scaffold the project skeleton",
  "phase": "scaffold",
  "recipe_lookup": true,
  "recipe_hint": "project scaffold cookiecutter package skeleton"
}
```

  And for a design step:

```json
{
  "id": "2.1",
  "agent": "architect",
  "title": "Design the system architecture",
  "phase": "architecture",
  "recipe_lookup": false
}
```

- [ ] Validate the JSON parses: `python -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json')); print('ok')"`
- [ ] Run: `timeout 30 pytest tests/workflows/test_recipe_lookup_fields.py` — expected **PASS**.
- [ ] Commit: `feat(workflow_engine): per-step recipe_lookup + recipe_hint fields with phase defaults`

---

## Task 12 — End-to-end integration test

The critical-instruction guarantee: a real task flows `flash() → envelope →
coulson prompt` with no dangling path. This test exercises the whole Phase 2
chain against an in-memory DB and a stubbed `yalayut.query`.

**Files:**
- Test: `tests/integration/test_yalayut_phase2_e2e.py`

**Steps:**

- [ ] Write the integration test `tests/integration/test_yalayut_phase2_e2e.py`:

```python
"""End-to-end Phase 2: task → intersect.flash → envelope → coulson prompt.

Verifies the full consumer path is wired with no dangling fragment:
  - intersect.flash attaches task["skills"]
  - coulson.build_user_context renders it into the agent prompt
  - the skills.py shim returns the same inject slice
  - a preempt artifact routes the task to the mechanical lane
"""
import json

import pytest


class _Profile:
    name = "coder"
    allowed_tools = ["read_file", "write_file"]
    max_iterations = 5
    _prompt_version_override = None
    _suppress_clarification = False

    def get_system_prompt(self, task):
        return "You are a coder."


class _Art:
    def __init__(self, **kw):
        self.id = kw.get("id", 1)
        self.artifact_type = kw.get("artifact_type", "skill")
        self.kind = kw.get("kind", "prompt_skill")
        self.name = kw.get("name", "anthropics-pdf")
        self.source = kw.get("source", "github:anthropics/skills@/skills")
        self.owner = kw.get("owner", "anthropics")
        self.vet_tier = kw.get("vet_tier", 0)
        self.vector_sim = kw.get("vector_sim", 0.9)
        self.mechanizable = kw.get("mechanizable", False)
        self.applies_to = "execution"
        self.intent_keywords = kw.get("intent_keywords", [])
        self.inputs_schema = kw.get("inputs_schema", {})
        self.body = kw.get("body", "PDF skill body")
        self.env_status = "ready"


@pytest.fixture
async def e2e_db(monkeypatch):
    import aiosqlite
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(
        """
        CREATE TABLE yalayut_sources (source_id TEXT, trust_score REAL);
        CREATE TABLE yalayut_owners (owner_id TEXT, trust_score REAL);
        CREATE TABLE yalayut_usage (
          id INTEGER PRIMARY KEY, artifact_id INTEGER, task_id TEXT,
          exposure_class TEXT, bind_args_json TEXT, exposed BOOLEAN,
          called BOOLEAN, succeeded BOOLEAN, latency_ms INTEGER,
          conflict_loser BOOLEAN, would_have_used INTEGER,
          escape_reason TEXT, occurred_at TIMESTAMP);
        CREATE TABLE yalayut_bind_cache (
          id INTEGER PRIMARY KEY, manifest_id INTEGER, ctx_embedding BLOB,
          bound_args_json TEXT, hit_count INTEGER DEFAULT 0,
          created_at TIMESTAMP, last_used_at TIMESTAMP);
        """
    )
    await conn.execute(
        "INSERT INTO yalayut_sources VALUES "
        "('github:anthropics/skills@/skills', 1.0)")
    await conn.execute(
        "INSERT INTO yalayut_owners VALUES ('anthropics', 1.0)")
    await conn.commit()

    async def _get_db():
        return conn

    import src.infra.db as _db
    monkeypatch.setattr(_db, "get_db", _get_db)
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_full_path_task_to_agent_prompt(e2e_db, monkeypatch):
    import intersect
    import yalayut
    from coulson import context

    async def _query(task_ctx):
        return [_Art(id=1, name="anthropics-pdf", body="Use the PDF skill.")]

    monkeypatch.setattr(yalayut, "query", _query, raising=False)

    task = {
        "id": 9001, "title": "Extract text from a PDF report",
        "description": "parse the quarterly PDF",
        "agent_type": "coder",
        "context": json.dumps({"recipe_lookup": True}),
    }

    # 1. orchestrator pump would call this:
    task = await intersect.flash(task)
    assert task["skills"], "intersect attached an empty envelope"
    assert task["skills"][0]["exposure_class"] == "inject"

    # 2. coulson renders it into the agent prompt:
    prompt, injected_tools = await context.build_user_context(
        _Profile(), task, model_ctx=4096)
    assert "anthropics-pdf" in prompt
    assert "Use the PDF skill." in prompt

    # 3. skills.py shim sees the same inject slice:
    from src.memory import skills
    shim = await skills.find_relevant_skills("x", task=task)
    assert len(shim) == 1
    assert shim[0]["name"] == "anthropics-pdf"

    # 4. telemetry row written:
    cur = await e2e_db.execute(
        "SELECT exposure_class FROM yalayut_usage WHERE task_id = '9001'")
    rows = await cur.fetchall()
    assert rows and rows[0][0] == "inject"


@pytest.mark.asyncio
async def test_preempt_routes_task_to_mechanical_lane(e2e_db, monkeypatch):
    import intersect
    import yalayut

    async def _query(task_ctx):
        return [_Art(id=18, name="cc-pypackage", kind="shell_recipe",
                     mechanizable=True, vector_sim=1.0, inputs_schema={},
                     source="github:cookiecutter/cookiecutter-pypackage",
                     owner="cookiecutter")]

    monkeypatch.setattr(yalayut, "query", _query, raising=False)

    task = {
        "id": 9002, "title": "Scaffold the Python package",
        "description": "create the skeleton", "agent_type": "coder",
        "context": json.dumps({"recipe_lookup": True}),
    }
    task = await intersect.flash(task)
    assert task["skills"] == []
    assert task["runner"] == "mechanical"
    assert task["payload"]["action"] == "yalayut_recipe"
    assert task["payload"]["recipe_id"] == 18
```

> `cc-pypackage` is a T1 seed manifest (Task list item 18) — T1 cannot preempt
> per the tier ceiling. The integration test forces `vet_tier=0` on the stub
> artifact deliberately to exercise the preempt routing path; the live tier
> comes from Phase 1's classifier. This is a test-fixture choice, not a spec
> contradiction.

- [ ] Run: `timeout 90 pytest tests/integration/test_yalayut_phase2_e2e.py` — expected **PASS** (all prior tasks complete, so the path exists).
- [ ] Run the full Phase 2 suite:
  `timeout 120 pytest packages/intersect/tests/ packages/coulson/tests/test_skill_render.py packages/coulson/tests/test_context_envelope.py tests/memory/test_skills_shim.py tests/workflows/test_recipe_lookup_fields.py tests/core/test_orchestrator_intersect_wiring.py tests/integration/test_yalayut_phase2_e2e.py`
  — expected **PASS**.
- [ ] Commit: `test(yalayut): end-to-end Phase 2 integration — flash → envelope → coulson`

---

## Self-review

**Spec-requirement → task map** (every Phase 2 scope bullet):

| Phase 2 requirement | Task(s) |
|---|---|
| `packages/intersect/` package, src layout | 1 |
| `flash.py` entry | 1 (scaffold), 7 (body) |
| `scoring.py` — `vector_sim × source_trust × owner_trust × hint_bonus` | 2 |
| `exposure.py` — `(tier × kind × confidence) → exposure_class` | 3 |
| `binding.py` — static `bind_from` + `bind_cache`, NO LLM-bind | 4 |
| `budget.py` — api ≤3/step, mcp ≤3/server ≤6/step | 5 |
| `telemetry.py` — `yalayut_usage` writes | 6 |
| `flash` flow exactly per spec; preempt → mechanical lane; else `task["skills"]` | 7 |
| Errors → graceful degrade (empty skills list) | 7 (`test_flash_graceful_degrade_on_error`) |
| Exposure classes: inject / tool / preempt / quarantine; NO sandbox | 3 |
| `task["skills"]` envelope = list of plain dicts per Envelope contract | 7 |
| Orchestrator pump calls `intersect.flash(task)` once per task before dispatch | 8 |
| coulson reads `applies_to=execution`, renders into agent prompt | 9 |
| Reroute `coulson/context.py` skill call site away from `skills.find_relevant_skills` | 9 |
| Rendering lives in coulson (`skill_render.py`) | 9 |
| `skills.py` shim returns `inject` slice; byte-identical injection output | 10 |
| Test asserting byte-identity | 10 (`test_format_skills_byte_identical_for_legacy_dicts`) |
| `recipe_lookup` + `recipe_hint` JSON fields; phase defaults | 11 |
| End-to-end integration test (task in → matched → injected) | 12 |
| intersect does NOT call the dispatcher | enforced — `flash.py` imports only `yalayut` + `binding/budget/exposure/scoring/telemetry` + `src.infra`; no `LLMDispatcher` import anywhere |

**Type / signature consistency with the spec:**

- `flash(task: dict) -> dict` — matches spec Public APIs. Always returns the task
  with a `skills` key (possibly `[]`); preempt sets `runner`/`payload` instead.
- `query(task_ctx: dict) -> list[Artifact]` — consumed in Task 7 as
  `await yalayut.query(task_ctx)`. The plan treats `query` as **async** (the
  spec's signature is unannotated for async; intersect is async throughout and
  the orchestrator pump awaits `flash`, so an async `query` is the consistent
  choice — flagged below as a resolved ambiguity).
- `Artifact` — used in-process only inside `flash`/`scoring`/`exposure`/`binding`
  via duck-typed attribute reads (`vet_tier`, `kind`, `vector_sim`,
  `inputs_schema`, `mechanizable`, `source`, `owner`, `name`, `body`, `id`,
  `env_status`, `intent_keywords`, `artifact_type`). Tests use a `FakeArtifact`
  mirroring those fields. Nothing crosses a serialization boundary as an
  `Artifact`.
- `SkillApplication` — the spec calls it a "structured object" produced by
  yalayut's `AccessPlugin.to_application`. The **envelope** that crosses into
  `task["skills"]` is explicitly a list of **plain dicts** (spec Envelope
  contract). This plan builds the envelope dicts directly in `flash` with the
  exact contract keys (`artifact_id`, `name`, `exposure_class`, `applies_to`,
  `render`, `payload`, `confidence`). `coulson` consumes plain dicts. No shared
  type import — consistent with "no shared type import, crosses DB/JSON".

**No placeholders:** every step ships complete, runnable Python. Task 1's
`flash.py`/`exposure.py` placeholders are explicitly replaced by full modules
in Tasks 7/3 — they exist only so the package imports during scaffold and are
not left as TODO stubs.

**Spec ambiguities resolved inline:**

1. **`intersect` package vs module** — spec Open Issues leaves this to the plan
   author. Resolved: a **package** (`packages/intersect/`), matching the spec's
   own package-layout diagram and KutAI's src-layout convention. Keeps a clean
   pyproject dependency edge (`dependencies = ["yalayut"]`).
2. **`yalayut.query` async-ness** — spec signature is unannotated. Resolved:
   treated as **async** (`await yalayut.query(...)`). intersect is async
   throughout; if Phase 1 shipped `query` synchronous, change the one call site
   in `flash.py` to drop `await` — noted as the single integration touch-point.
3. **Conflict-resolution slot definition** — spec says "2 `agent_config` skills
   both applicable to one step" collide. Resolved: `_slot_key` collapses
   `agent_config` artifacts into one slot (only one agent persona makes sense
   per step); `prompt_skill` and other kinds get unique per-id slots (multiple
   prose hints stack fine). Highest-confidence wins; losers → `conflict_loser=1`.
4. **`source_trust` / `owner_trust` default when row absent** — spec doesn't
   state the missing-row behaviour. Resolved: default **1.0** (neutral), not
   0.0 — a zero would multiply every unseeded artifact's confidence to zero and
   silently suppress all matches. The tier classifier already caps untrusted
   artifacts; trust scoring stays neutral on missing rows.
5. **`recipe_lookup` default phases** — task brief lists the phase sets; the i2p
   JSON uses a `phase` key plus title-keyword inference as a fallback when a
   step omits `phase`. Explicit step `recipe_lookup` always overrides the
   phase default.
6. **preempt with unbound required fields** — spec says such a recipe
   "downgrades to `inject`". Implemented in `flash`: a `preempt`-classified
   artifact whose static bind is incomplete falls back to an `inject`
   application instead of routing to the mechanical lane with holes.
7. **Mechanical preempt payload shape** — spec says recipe + bound args go in
   the mechanical task payload; it does not name the action. Resolved:
   `payload = {"action": "yalayut_recipe", "recipe_id": <id>, "args": <bound>}`,
   matching the spec's `run_recipe(recipe_id, args)` executor signature so the
   Phase 3 `mr_roboto` `yalayut_recipe` executor consumes it directly.
