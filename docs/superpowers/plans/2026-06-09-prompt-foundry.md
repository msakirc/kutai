# Prompt Foundry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the 29 pure-config `src/agents/*.py` classes into per-profile YAML data behind a new foundational **leaf** package that owns all prompt/profile content + one uniform build API, with storage behind an injected `PromptStore` port (DB stays, packages stop importing `src`).

**Architecture:** A leaf package (placeholder `prompt_foundry`, final name TBD by founder) ships the 27 static profiles as `profiles/<name>.yaml`, exposes a duck-typed `Profile` the runtime already consumes, and defines a tiny `PromptStore` Protocol. The app (`src`) wires a concrete DB-backed adapter into the Foundry at startup. `oncall_agent` + `writer` stay thin `Profile` subclasses (dynamic prompts). Overhead/husam-caller prompts (grading, code_review, reflection, brand_voice, copy_compliance, yalayut, vision, classifier) relocate into the Foundry and build via one `build_messages` API. Canonical-first: scaffold + 1 agent lands to main before the bulk migration.

**Tech Stack:** Python 3.10, PyYAML, dataclasses, `pytest` (timeout-wrapped), aiosqlite (behind the port only).

**Spec:** `docs/superpowers/specs/2026-06-09-prompt-foundry-leaf-design.md`

**Invariants (carry into every task):**
- `get_agent(x)` returns a **stable per-type singleton** (same object across calls).
- DB override (`PromptStore.get_active`) still beats the in-package seed at runtime.
- Keep `AGENT_REGISTRY` name + `get_agent(type)` signature for back-compat.
- Leaf package imports **nothing** from `src` or feature packages.
- Run all package tests with `.venv/Scripts/python -m pytest packages/<pkg>/tests -q` under `timeout`; run app tests with `tests/`. **Never mix `tests/` and `packages/` in one pytest call** (conftest collision).
- Do this in a **git worktree** (live-bot `git add -A` storm on `main`); merge in a quiet window.

---

## File Structure

**New (leaf package — placeholder dir `packages/prompt_foundry/`):**
- `src/prompt_foundry/__init__.py` — public API: `Profile`, `get_profile`, `PROFILE_REGISTRY`, `build_messages`, `set_store`, `PromptStore`.
- `src/prompt_foundry/profile.py` — `Profile` dataclass (duck-typed runtime surface) + carve-out subclasses.
- `src/prompt_foundry/loader.py` — YAML → `Profile` singletons (parse once at import).
- `src/prompt_foundry/store.py` — `PromptStore` Protocol + module-level injected store + `get_active`/`record_quality` helpers (no-op when unset).
- `src/prompt_foundry/build.py` — `build_messages(...)` uniform assembly for overhead/husam prompts.
- `src/prompt_foundry/profiles/<name>.yaml` — 27 static profile data files.
- `src/prompt_foundry/rubrics/<key>.yaml` — overhead/rubric content (Phase 3).
- `pyproject.toml` — package metadata (mirror an existing leaf package, e.g. `packages/husam/pyproject.toml`).
- `tests/` — package-local tests.

**Modified (app + consumers):**
- `src/agents/__init__.py` — `AGENT_REGISTRY`/`get_agent` rebuilt from Foundry + 2 subclasses.
- `src/app/run.py` (startup) — wire the DB adapter via `prompt_foundry.set_store(...)`.
- New `src/infra/prompt_store_adapter.py` — concrete `PromptStore` over `src/memory/prompt_versions.py`.
- `packages/coulson/src/coulson/__init__.py:203-213` — `_load_db_prompt_override` stops importing `src.memory`; uses the injected store (Phase 3).
- `packages/general_beckman/.../posthook_handlers/{brand_voice_lint,copy_compliance_review}.py`, `packages/yalayut/.../discovery/synthesize.py`, `src/tools/vision.py`, `packages/coulson/.../posthooks/*`, `src/agents/signal_classifier.py` — build via `prompt_foundry.build_messages` (Phase 3).
- `tests/agents/test_prompt_quality.py`, `tests/test_root_stays_thin.py` — retargeted (Phase 4-5).

---

## Phase 1 — Scaffold leaf + migrate `summarizer` (canonical-first, lands to main)

### Task 1: Create the leaf package skeleton

**Files:**
- Create: `packages/prompt_foundry/pyproject.toml`
- Create: `packages/prompt_foundry/src/prompt_foundry/__init__.py`

- [ ] **Step 1: Copy package metadata from an existing leaf**

Read `packages/husam/pyproject.toml`; create `packages/prompt_foundry/pyproject.toml` with the same structure, changing name to `prompt-foundry`, package path to `src/prompt_foundry`, and adding `pyyaml` to dependencies. Keep the same Python version pin.

- [ ] **Step 2: Empty public API module**

```python
# packages/prompt_foundry/src/prompt_foundry/__init__.py
"""Prompt Foundry — leaf package owning prompt/profile content + build API.

Depends on NOTHING in src/ or feature packages. Storage is injected via
set_store(); with no store, profiles fall back to in-package YAML seeds.
"""
from .profile import Profile
from .loader import PROFILE_REGISTRY, get_profile
from .store import PromptStore, set_store, get_active, record_quality

__all__ = [
    "Profile", "PROFILE_REGISTRY", "get_profile",
    "PromptStore", "set_store", "get_active", "record_quality",
]
```

- [ ] **Step 3: Install editable + verify import fails cleanly (modules not yet created)**

Run: `.venv/Scripts/python -m pip install -e packages/prompt_foundry`
Expected: install succeeds; `import prompt_foundry` will fail until Tasks 2-4 create the modules. That is expected — proceed.

- [ ] **Step 4: Commit**

```bash
git add packages/prompt_foundry/pyproject.toml packages/prompt_foundry/src/prompt_foundry/__init__.py
git commit -m "feat(prompt_foundry): scaffold leaf package skeleton"
```

---

### Task 2: `Profile` dataclass (duck-typed runtime surface)

**Files:**
- Create: `packages/prompt_foundry/src/prompt_foundry/profile.py`
- Test: `packages/prompt_foundry/tests/test_profile.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/prompt_foundry/tests/test_profile.py
from prompt_foundry.profile import Profile

def test_profile_exposes_runtime_surface():
    p = Profile(
        name="summarizer",
        description="x",
        system_prompt="You are a summarization specialist...",
        allowed_tools=["read_file"],
        max_iterations=3,
    )
    # duck-typed surface coulson/runtime consume:
    assert p.name == "summarizer"
    assert p.allowed_tools == ["read_file"]
    assert p.max_iterations == 3
    assert p.get_system_prompt({"id": 1}) == "You are a summarization specialist..."
    # runtime-mutable attrs default correctly:
    assert p._prompt_version_override is None
    assert p._suppress_clarification is False
    assert p.progress_callback is None
    # defaults for unspecified profile fields:
    assert p.default_tier == "cheap"
    assert p.execution_pattern == "react_loop"
    assert p.enable_self_reflection is False
    assert p.confidence_gate == "fail_closed"

def test_profile_get_system_prompt_static_returns_seed():
    p = Profile(name="x", description="d", system_prompt="SEED")
    assert p.get_system_prompt({"anything": True}) == "SEED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_profile.py -q` (with `timeout 60`)
Expected: FAIL — `No module named 'prompt_foundry.profile'`.

- [ ] **Step 3: Implement `Profile`**

```python
# packages/prompt_foundry/src/prompt_foundry/profile.py
"""Profile — duck-typed runtime surface, ex-agent config as data.

Mirrors the attribute surface the runtime consumes (was src/agents/base.py
BaseAgent). Static profiles return a fixed seed string from get_system_prompt;
carve-outs (oncall_agent, writer) subclass and override it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

# Mirrors src/app/config.MAX_AGENT_ITERATIONS default; profiles override.
DEFAULT_MAX_ITERATIONS = 10


@dataclass
class Profile:
    name: str
    description: str = ""
    system_prompt: str = ""          # the seed (frozen reference; DB override wins)
    default_tier: str = "cheap"
    min_tier: str = "cheap"
    allowed_tools: Optional[list[str]] = None   # None = all; [] = none
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    can_create_subtasks: bool = False
    execution_pattern: str = "react_loop"        # or "single_shot"
    enable_self_reflection: bool = False
    min_confidence: int = 0
    confidence_gate: str = "fail_closed"

    # ── runtime-mutable per-execution attrs (NOT profile data) ──
    # Set/restored by coulson.execute(); declared here so the duck-type holds.
    _prompt_version_override: Optional[str] = field(default=None, repr=False)
    _suppress_clarification: bool = field(default=False, repr=False)
    progress_callback: Optional[Callable] = field(default=None, repr=False)
    _original_allowed_tools: object = field(default=None, repr=False)

    def get_system_prompt(self, task: dict) -> str:
        return self.system_prompt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_profile.py -q` (`timeout 60`)
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/prompt_foundry/src/prompt_foundry/profile.py packages/prompt_foundry/tests/test_profile.py
git commit -m "feat(prompt_foundry): Profile dataclass with duck-typed runtime surface"
```

---

### Task 3: YAML loader → per-type singletons

**Files:**
- Create: `packages/prompt_foundry/src/prompt_foundry/loader.py`
- Create: `packages/prompt_foundry/src/prompt_foundry/profiles/summarizer.yaml`
- Test: `packages/prompt_foundry/tests/test_loader.py`

- [ ] **Step 1: Write `summarizer.yaml` (verbatim from `src/agents/summarizer.py`)**

```yaml
# packages/prompt_foundry/src/prompt_foundry/profiles/summarizer.yaml
name: summarizer
description: Condenses long content into structured summaries
default_tier: cheap
min_tier: cheap
max_iterations: 3
allowed_tools:
  - read_file
  - file_tree
  - web_search
system_prompt: |
  You are a summarization specialist. You distill long content into clear, structured, and actionable summaries.

  ## Your Workflow
  1. **Read** — Use `read_file` or `file_tree` to access the content that needs summarization.
  2. **Identify** — Find the key themes, decisions, action items, and critical details.
  3. **Structure** — Organize findings into a clear hierarchy.
  4. **Condense** — Remove redundancy while preserving meaning.

  ## Rules
  - Lead with the most important information.
  - Use bullet points and clear section headings.
  - Preserve numbers, dates, names, and specific commitments.
  - Distinguish between facts, decisions, and open questions.
  - If content is technical, keep key technical details.
  - Target 20-30% of original length unless told otherwise.
  - NEVER invent information not present in the source.
  - Do NOT omit action items or hard commitments — these must always appear.

  ## final_answer format
  ```json
  {
    "action": "final_answer",
    "result": "## Summary\n\n### Key Points\n- ...\n\n### Decisions\n- ...\n\n### Action Items\n- ...",
    "memories": {}
  }
  ```
```

> **Migration note:** the `\n` inside the JSON `result` example must remain literal `\n` characters (two chars) in the rendered string, matching the original Python `"\\n"`. In a YAML block scalar, write them as the two characters `\n` (backslash-n) — block scalars do NOT process escapes, so `\n` stays literal. Verify in Step 4's test.

- [ ] **Step 2: Write the failing test**

```python
# packages/prompt_foundry/tests/test_loader.py
from prompt_foundry.loader import get_profile, PROFILE_REGISTRY
from prompt_foundry.profile import Profile

def test_summarizer_loaded_from_yaml():
    p = get_profile("summarizer")
    assert isinstance(p, Profile)
    assert p.name == "summarizer"
    assert p.max_iterations == 3
    assert p.allowed_tools == ["read_file", "file_tree", "web_search"]
    assert p.get_system_prompt({}).startswith("You are a summarization specialist.")
    # literal backslash-n preserved (not a real newline) inside the JSON example:
    assert "\\n\\n### Key Points" in p.get_system_prompt({})

def test_singleton_identity():
    assert get_profile("summarizer") is get_profile("summarizer")

def test_registry_contains_summarizer():
    assert "summarizer" in PROFILE_REGISTRY
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_loader.py -q` (`timeout 60`)
Expected: FAIL — `No module named 'prompt_foundry.loader'`.

- [ ] **Step 4: Implement the loader (parse once at import)**

```python
# packages/prompt_foundry/src/prompt_foundry/loader.py
"""Load profiles/*.yaml into Profile singletons. Parsed ONCE at import.

Hot-path get_profile() must never parse YAML per call — it reads the
pre-built registry dict.
"""
from __future__ import annotations
from pathlib import Path
import yaml

from .profile import Profile

_PROFILES_DIR = Path(__file__).parent / "profiles"


def _load_all() -> dict[str, Profile]:
    registry: dict[str, Profile] = {}
    for yml in sorted(_PROFILES_DIR.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        name = data["name"]
        registry[name] = Profile(**data)
    return registry


PROFILE_REGISTRY: dict[str, Profile] = _load_all()


def get_profile(name: str) -> Profile | None:
    """Return the cached singleton Profile for `name`, or None if absent."""
    return PROFILE_REGISTRY.get(name)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_loader.py -q` (`timeout 60`)
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add packages/prompt_foundry/src/prompt_foundry/loader.py packages/prompt_foundry/src/prompt_foundry/profiles/summarizer.yaml packages/prompt_foundry/tests/test_loader.py
git commit -m "feat(prompt_foundry): YAML loader + summarizer profile, singleton identity"
```

---

### Task 4: `PromptStore` Protocol + injected store (no-op default)

**Files:**
- Create: `packages/prompt_foundry/src/prompt_foundry/store.py`
- Test: `packages/prompt_foundry/tests/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/prompt_foundry/tests/test_store.py
import asyncio
import prompt_foundry.store as store

def test_get_active_returns_none_without_store():
    store.set_store(None)
    assert asyncio.get_event_loop().run_until_complete(store.get_active("summarizer")) is None

def test_injected_store_is_used():
    class FakeStore:
        async def get_active(self, key):
            return "DB PROMPT" if key == "summarizer" else None
        async def save_version(self, key, text, notes="", activate=False):
            return 1
        async def record_quality(self, key, score):
            return None
        async def list_versions(self, key):
            return []
    store.set_store(FakeStore())
    loop = asyncio.get_event_loop()
    assert loop.run_until_complete(store.get_active("summarizer")) == "DB PROMPT"
    assert loop.run_until_complete(store.get_active("other")) is None
    store.set_store(None)  # reset
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_store.py -q` (`timeout 60`)
Expected: FAIL — `No module named 'prompt_foundry.store'`.

- [ ] **Step 3: Implement the port + injection**

```python
# packages/prompt_foundry/src/prompt_foundry/store.py
"""PromptStore port — versioned prompt overrides live behind this Protocol.

The app (src) injects a concrete DB-backed adapter via set_store(). With no
store set, get_active() returns None and the in-package YAML seed is used.
The leaf imports no DB and no src.
"""
from __future__ import annotations
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PromptStore(Protocol):
    async def get_active(self, key: str) -> Optional[str]: ...
    async def save_version(self, key: str, text: str, notes: str = "", activate: bool = False) -> int: ...
    async def record_quality(self, key: str, score: float) -> None: ...
    async def list_versions(self, key: str) -> list[dict]: ...


_store: Optional[PromptStore] = None


def set_store(store: Optional[PromptStore]) -> None:
    global _store
    _store = store


async def get_active(key: str) -> Optional[str]:
    if _store is None:
        return None
    return await _store.get_active(key)


async def record_quality(key: str, score: float) -> None:
    if _store is None:
        return
    await _store.record_quality(key, score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_store.py -q` (`timeout 60`)
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/prompt_foundry/src/prompt_foundry/store.py packages/prompt_foundry/tests/test_store.py
git commit -m "feat(prompt_foundry): PromptStore Protocol + injected store (no-op default)"
```

---

### Task 5: Concrete DB adapter in `src` + startup wiring

**Files:**
- Create: `src/infra/prompt_store_adapter.py`
- Modify: `src/app/run.py` (startup — add `set_store(...)` call near other init)
- Test: `tests/infra/test_prompt_store_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/infra/test_prompt_store_adapter.py
import pytest
from src.infra.prompt_store_adapter import DbPromptStore

@pytest.mark.asyncio
async def test_adapter_satisfies_protocol():
    from prompt_foundry.store import PromptStore
    assert isinstance(DbPromptStore(), PromptStore)

@pytest.mark.asyncio
async def test_get_active_delegates_to_prompt_versions(monkeypatch):
    called = {}
    async def fake_get_active_prompt(agent_type):
        called["arg"] = agent_type
        return "DBVAL"
    monkeypatch.setattr("src.memory.prompt_versions.get_active_prompt", fake_get_active_prompt)
    out = await DbPromptStore().get_active("coder")
    assert out == "DBVAL"
    assert called["arg"] == "coder"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/infra/test_prompt_store_adapter.py -q`
Expected: FAIL — `No module named 'src.infra.prompt_store_adapter'`.

- [ ] **Step 3: Implement the adapter (wraps existing prompt_versions)**

```python
# src/infra/prompt_store_adapter.py
"""Concrete PromptStore over the existing prompt_versions DB table.

This is the app-side adapter wired into prompt_foundry at startup. It is the
ONLY thing that bridges the leaf to src DB — the reference 'port' pattern for
the broader src-DB-dep kill (separate track).
"""
from __future__ import annotations
from typing import Optional

from src.memory import prompt_versions


class DbPromptStore:
    async def get_active(self, key: str) -> Optional[str]:
        return await prompt_versions.get_active_prompt(key)

    async def save_version(self, key: str, text: str, notes: str = "", activate: bool = False) -> int:
        return await prompt_versions.save_prompt_version(key, text, notes=notes, activate=activate)

    async def record_quality(self, key: str, score: float) -> None:
        await prompt_versions.record_prompt_quality(key, score)

    async def list_versions(self, key: str) -> list[dict]:
        return await prompt_versions.list_prompt_versions(key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/infra/test_prompt_store_adapter.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire at startup**

In `src/app/run.py`, find the orchestrator/startup init block (search for existing one-time setup such as `seed_skills` or DB init). Add near it:

```python
# Wire prompt_foundry's storage port to the DB adapter (one-time, at startup).
from prompt_foundry import set_store
from src.infra.prompt_store_adapter import DbPromptStore
set_store(DbPromptStore())
```

- [ ] **Step 6: Verify import + wiring smoke**

Run: `timeout 60 .venv/Scripts/python -c "import src.app.run; from prompt_foundry import store; print(store._store.__class__.__name__ if store._store else 'NONE')"`
Expected: prints `DbPromptStore` if `run.py` executes the wiring at import; if wiring is inside a function, instead assert the function calls `set_store` (grep) and print `OK`.

- [ ] **Step 7: Commit**

```bash
git add src/infra/prompt_store_adapter.py src/app/run.py tests/infra/test_prompt_store_adapter.py
git commit -m "feat(prompt_foundry): DB-backed PromptStore adapter + startup wiring"
```

---

### Task 6: Serve `summarizer` from Foundry via `get_agent` (singleton-preserving)

**Files:**
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_foundry_get_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_foundry_get_agent.py
from src.agents import get_agent

def test_summarizer_served_from_foundry():
    p = get_agent("summarizer")
    assert p.name == "summarizer"
    assert p.max_iterations == 3
    # data-backed Profile, NOT the old class:
    assert type(p).__module__.startswith("prompt_foundry")

def test_summarizer_singleton_identity():
    assert get_agent("summarizer") is get_agent("summarizer")

def test_other_agents_still_work():
    # coder still class-backed in Phase 1; identity still holds
    assert get_agent("coder") is get_agent("coder")
    assert get_agent("coder").name == "coder"

def test_unknown_falls_back_to_executor():
    assert get_agent("nonexistent").name == "executor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/test_foundry_get_agent.py -q`
Expected: FAIL — `test_summarizer_served_from_foundry` (still the old `SummarizerAgent` class).

- [ ] **Step 3: Rewire `get_agent` to prefer Foundry, fall back to classes**

Edit `src/agents/__init__.py`. Remove the `from .summarizer import SummarizerAgent` import and its registry entry, then layer Foundry on top:

```python
# agents/__init__.py  (top unchanged imports minus summarizer)
from prompt_foundry import PROFILE_REGISTRY as _FOUNDRY_PROFILES, get_profile as _get_profile
# ... remaining class imports unchanged, summarizer REMOVED ...

AGENT_REGISTRY = {
    # ... all remaining class instances, summarizer entry REMOVED ...
}


def get_agent(agent_type: str):
    """Get agent/profile by type. Foundry data profiles take precedence;
    legacy class instances are the fallback; executor is the final default."""
    p = _get_profile(agent_type)
    if p is not None:
        return p
    return AGENT_REGISTRY.get(agent_type, AGENT_REGISTRY["executor"])
```

> Singleton note: `get_profile` returns the registry singleton (built once at Foundry import), so identity holds. Do NOT construct a new Profile per call.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/test_foundry_get_agent.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Delete the migrated class**

```bash
git rm src/agents/summarizer.py
```

- [ ] **Step 6: Full agents-suite regression + import smoke**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/ -q`
Run: `timeout 60 .venv/Scripts/python -c "from src.agents import get_agent, AGENT_REGISTRY; print(get_agent('summarizer').name, get_agent('summarizer') is get_agent('summarizer'))"`
Expected: tests pass; prints `summarizer True`.

- [ ] **Step 7: Commit**

```bash
git add src/agents/__init__.py tests/agents/test_foundry_get_agent.py
git commit -m "feat(prompt_foundry): serve summarizer from Foundry, get_agent precedence + fallback"
```

---

### Task 7: Live execute-path smoke for the migrated profile

**Files:**
- Test: `tests/agents/test_foundry_execute_path.py`

- [ ] **Step 1: Write the test (coulson consumes the data Profile unchanged)**

```python
# tests/agents/test_foundry_execute_path.py
from src.agents import get_agent
from coulson.context import build_system_prompt

def test_build_system_prompt_consumes_data_profile():
    p = get_agent("summarizer")
    # coulson sets these per-execution; data Profile must tolerate them:
    p._prompt_version_override = None
    msg = build_system_prompt(p, {"id": 1, "title": "t", "description": "d"})
    assert "summarization specialist" in msg

def test_db_override_wins_over_seed():
    p = get_agent("summarizer")
    p._prompt_version_override = "OVERRIDDEN PROMPT TEXT"
    msg = build_system_prompt(p, {"id": 1, "title": "t", "description": "d"})
    assert msg.startswith("OVERRIDDEN PROMPT TEXT")
    p._prompt_version_override = None  # reset
```

- [ ] **Step 2: Run test**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/test_foundry_execute_path.py -q`
Expected: PASS (2 tests). If `coulson` import errors (not installed), run `.venv/Scripts/python -m pip install -e packages/coulson` first.

- [ ] **Step 3: Commit + land Phase 1 to main**

```bash
git add tests/agents/test_foundry_execute_path.py
git commit -m "test(prompt_foundry): execute-path smoke — coulson consumes data Profile, DB override wins"
```

**STOP — merge Phase 1 to main (canonical-first) before Phase 2.** Use the worktree merge pattern (`project_z1_merge_pattern_20260510`); merge in a quiet window.

---

## Phase 2 — Bulk-migrate the 27 static profiles

> **Pattern (apply per agent):** for each static agent file in `src/agents/`, create `profiles/<name>.yaml` whose fields equal the class attributes and whose `system_prompt` is the verbatim string returned by `get_system_prompt` (collapse the Python string-concatenation into one block scalar; preserve literal `\n`). Then delete the `.py` and remove its `__init__.py` import + registry entry. `get_agent` already prefers Foundry, so each migration is live the moment its YAML exists and the class is removed.

### Task 8: Migrate the 25 remaining static agents (one commit per agent)

**Files (repeat for each):**
- Create: `packages/prompt_foundry/src/prompt_foundry/profiles/<name>.yaml`
- Delete: `src/agents/<name>.py`
- Modify: `src/agents/__init__.py` (drop import + registry entry)
- Test: extend `tests/agents/test_foundry_get_agent.py`

Static agents to migrate (25 — all except the 2 carve-outs `oncall_agent`, `writer`, and already-done `summarizer`):
`planner, architect, coder, implementer, fixer, test_generator, reviewer, visual_reviewer, researcher, analyst, assistant, executor, shopping_advisor, product_researcher, deal_analyst, shopping_clarifier, shopping_grouper, shopping_labeler, shopping_synthesizer, integration_reviewer, adr_drift_judge, support_tier1, growth_digest_synthesizer, signal_classifier, query_planner, prior_art_synthesizer`.

> Count check: 29 agents − 2 carve-outs − 1 (summarizer, Phase 1) = **26 here.** The list above is 26 entries.

For EACH agent `<name>`:

- [ ] **Step 1: Read `src/agents/<name>.py`**; capture every class attribute that differs from `Profile` defaults and the full `get_system_prompt` return string.

- [ ] **Step 2: Write `profiles/<name>.yaml`** with those fields + the verbatim prompt as a `|` block scalar. (Follow the `summarizer.yaml` shape from Task 3. Preserve literal `\n`.)

- [ ] **Step 3: Add a registry assertion** to `tests/agents/test_foundry_get_agent.py`:

```python
def test_<name>_served_from_foundry():
    p = get_agent("<name>")
    assert p.name == "<name>"
    assert type(p).__module__.startswith("prompt_foundry")
    # spot-check one non-default attr (e.g. max_iterations / allowed_tools)
```

- [ ] **Step 4: Remove the class** — delete the import line + registry entry in `src/agents/__init__.py`, then `git rm src/agents/<name>.py`.

- [ ] **Step 5: Verify** — `timeout 60 .venv/Scripts/python -m pytest tests/agents/test_foundry_get_agent.py -q` PASS; `… -c "from src.agents import get_agent; print(get_agent('<name>').name)"` prints `<name>`.

- [ ] **Step 6: Commit** — `git commit -m "refactor(prompt_foundry): migrate <name> agent to YAML profile"`.

> **Special case — `signal_classifier`:** its prompt is also referenced by routing. Keep `execution_pattern: single_shot` in YAML. Do NOT migrate its *classification logic* — only the prompt string. The class `src/agents/signal_classifier.py` holds only `get_system_prompt` (verified pure-config); if it holds parsing helpers, leave those in a non-agent module and migrate only the prompt.

---

### Task 9: Carve-outs `oncall_agent` + `writer` as thin `Profile` subclasses

**Files:**
- Modify: `packages/prompt_foundry/src/prompt_foundry/profile.py` (add subclasses)
- Modify: `packages/prompt_foundry/src/prompt_foundry/loader.py` (register subclasses)
- Create: `profiles/oncall_agent.yaml`, `profiles/writer.yaml` (static fields + base prompt)
- Delete: `src/agents/oncall_agent.py`, `src/agents/writer.py`
- Test: `packages/prompt_foundry/tests/test_carveouts.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/prompt_foundry/tests/test_carveouts.py
from prompt_foundry.loader import get_profile

def test_oncall_interpolates_domain():
    p = get_profile("oncall_agent")
    base = p.get_system_prompt({})
    withdom = p.get_system_prompt({"context": {"domain": "payments"}})
    assert "payments" in withdom
    assert "payments" not in base  # default domain 'ops'

def test_writer_branches_on_markdown_schema(tmp_path):
    p = get_profile("writer")
    plain = p.get_system_prompt({"title": "blog post"})
    assert isinstance(plain, str) and len(plain) > 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_carveouts.py -q` (`timeout 60`)
Expected: FAIL — profiles not present / no subclass behavior.

- [ ] **Step 3: Read the originals**, then add subclasses to `profile.py`:

```python
# append to packages/prompt_foundry/src/prompt_foundry/profile.py

class OncallProfile(Profile):
    def get_system_prompt(self, task: dict) -> str:
        domain = (task.get("context") or {}).get("domain", "ops")
        return self.system_prompt.replace("{{domain}}", domain)


def _detect_markdown_schema(task: dict) -> bool:
    # Verbatim port of src/agents/writer.py module helper — copy its body here.
    ...


class WriterProfile(Profile):
    # `markdown_prompt` carries the alternate branch text (loaded from YAML).
    markdown_prompt: str = ""

    def get_system_prompt(self, task: dict) -> str:
        if _detect_markdown_schema(task):
            return self.markdown_prompt or self.system_prompt
        return self.system_prompt
```

> Port `_detect_markdown_schema` verbatim from `src/agents/writer.py`. In `oncall_agent.yaml`, put `{{domain}}` where the original interpolated `domain`. In `writer.yaml`, add a `markdown_prompt: |` field with the markdown-branch string and make `WriterProfile` a dataclass field (add `markdown_prompt: str = ""` to the dataclass if subclass fields don't load — simplest: add it to base `Profile` as an optional field).

- [ ] **Step 4: Register subclasses in the loader** — map names to classes:

```python
# loader.py — replace the generic Profile(**data) construction:
from .profile import Profile, OncallProfile, WriterProfile

_PROFILE_CLASSES = {"oncall_agent": OncallProfile, "writer": WriterProfile}

def _load_all() -> dict[str, Profile]:
    registry = {}
    for yml in sorted(_PROFILES_DIR.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        name = data["name"]
        cls = _PROFILE_CLASSES.get(name, Profile)
        registry[name] = cls(**data)
    return registry
```

- [ ] **Step 5: Delete the classes + de-register**

```bash
git rm src/agents/oncall_agent.py src/agents/writer.py
```
Remove their imports + registry entries from `src/agents/__init__.py`.

- [ ] **Step 6: Run tests**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/ -q` (`timeout 60`)
Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/ -q`
Expected: PASS.

- [ ] **Step 7: Commit** — `git commit -m "refactor(prompt_foundry): oncall_agent + writer as thin Profile subclasses"`.

---

### Task 10: Trim `src/agents/__init__.py` to the back-compat shim

**Files:**
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_foundry_get_agent.py`

- [ ] **Step 1: Reduce `__init__.py`** — only `base.py` (still hosts A.12/A.13 residual until the follow-on track) need remain class-backed. `AGENT_REGISTRY` becomes a thin view over Foundry:

```python
# agents/__init__.py
from prompt_foundry import PROFILE_REGISTRY, get_profile

# Back-compat: AGENT_REGISTRY name preserved; now a view of Foundry profiles.
AGENT_REGISTRY = PROFILE_REGISTRY


def get_agent(agent_type: str):
    """Get profile by type; fallback to executor (Foundry profile)."""
    return get_profile(agent_type) or PROFILE_REGISTRY["executor"]
```

- [ ] **Step 2: Run full agents suite**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/ -q`
Expected: PASS. Fix any test still importing a deleted `*Agent` class → switch to `get_agent("x")`.

- [ ] **Step 3: Commit** — `git commit -m "refactor(prompt_foundry): AGENT_REGISTRY is now a Foundry view; classes gone"`.

---

## Phase 3 — Migrate overhead/husam-caller prompts + unanimous build API

### Task 11: `build_messages` uniform assembly API

**Files:**
- Create: `packages/prompt_foundry/src/prompt_foundry/build.py`
- Create: `packages/prompt_foundry/src/prompt_foundry/rubrics/` (dir)
- Test: `packages/prompt_foundry/tests/test_build.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/prompt_foundry/tests/test_build.py
from prompt_foundry.build import build_messages, register_rubric

def test_build_messages_system_plus_user():
    register_rubric("grading", system="You are a strict SEMANTIC evaluator.",
                    user_template="Task: {title}\nResult: {response}")
    msgs = build_messages("grading", {"title": "T", "response": "R"})
    assert msgs[0] == {"role": "system", "content": "You are a strict SEMANTIC evaluator."}
    assert msgs[1]["role"] == "user"
    assert "Task: T" in msgs[1]["content"]
    assert "Result: R" in msgs[1]["content"]

def test_build_messages_appends_dynamic_blocks():
    register_rubric("r2", system="S", user_template="U")
    msgs = build_messages("r2", {}, extra_blocks=["BLOCK1", "BLOCK2"])
    assert msgs[0]["content"] == "S\n\nBLOCK1\n\nBLOCK2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_build.py -q` (`timeout 60`)
Expected: FAIL — no module `prompt_foundry.build`.

- [ ] **Step 3: Implement `build.py`**

```python
# packages/prompt_foundry/src/prompt_foundry/build.py
"""Uniform prompt assembly for overhead/husam-caller prompts.

Rubric content (system + user template) lives in rubrics/*.yaml, loaded once.
Callers pass task fields + optional dynamic blocks; this is the ONE place
overhead messages get built (unanimity). Dynamic context (mission lessons,
calibration, tools) is passed IN as extra_blocks — the leaf can't fetch it.
"""
from __future__ import annotations
from pathlib import Path
import yaml

_RUBRICS_DIR = Path(__file__).parent / "rubrics"
_RUBRICS: dict[str, dict] = {}


def _load_rubrics() -> None:
    for yml in sorted(_RUBRICS_DIR.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        _RUBRICS[data["key"]] = data


def register_rubric(key: str, system: str, user_template: str) -> None:
    """Programmatic registration (tests / dynamic rubrics)."""
    _RUBRICS[key] = {"key": key, "system": system, "user_template": user_template}


def build_messages(key: str, fields: dict, extra_blocks: list[str] | None = None) -> list[dict]:
    r = _RUBRICS[key]
    system = r["system"]
    if extra_blocks:
        system = "\n\n".join([system, *extra_blocks])
    user = r["user_template"].format(**{k: fields.get(k, "") for k in _format_keys(r["user_template"])})
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _format_keys(template: str) -> set[str]:
    import string
    return {fn for _, fn, _, _ in string.Formatter().parse(template) if fn}


if _RUBRICS_DIR.exists():
    _load_rubrics()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_build.py -q` (`timeout 60`)
Expected: PASS (2 tests).

- [ ] **Step 5: Commit** — `git commit -m "feat(prompt_foundry): build_messages uniform assembly + rubric registry"`.

---

### Task 12: Migrate overhead rubrics into the Foundry (one commit per source)

**Files (repeat per source):**
- Create: `packages/prompt_foundry/src/prompt_foundry/rubrics/<key>.yaml`
- Modify: the spawner to call `build_messages`
- Test: alongside each spawner

Sources to migrate (each = a `rubrics/<key>.yaml` + a call-site swap):

| key | current location | call site to swap |
|-----|------------------|-------------------|
| `grading` | `packages/coulson/.../posthooks/grading.py` (`GRADING_SYSTEM`/`GRADING_PROMPT`) | grading builder → `build_messages("grading", fields)` |
| `code_review` | `packages/coulson/.../posthooks/code_review.py` | code-review builder |
| `reflection` | `packages/coulson/.../posthooks/reflection_posthook.py` (`REFLECT_SYSTEM_BASE`, REFLECTION_BLOCKS/STACK_BLOCKS/LAYER_BLOCKS) | `build_reflect_messages` / `build_reflection_prompt` |
| `self_critique` | `packages/coulson/.../self_critique.py` | `build_self_critique_message` |
| `constrained_emit` | `packages/coulson/.../posthooks/reflection_posthook.py` | `build_emit_messages` |
| `brand_voice` | `packages/general_beckman/.../posthook_handlers/brand_voice_lint.py:_run_llm_tone_pass` | husam spec messages |
| `copy_compliance` | `packages/general_beckman/.../posthook_handlers/copy_compliance_review.py:_check_privacy_mismatch_llm` | husam spec messages |
| `yalayut_synth` | `packages/yalayut/.../discovery/synthesize.py:llm_synthesize` | husam spec messages |
| `vision` | `src/tools/vision.py:analyze_image` | husam spec messages (note: includes image blocks — keep image assembly in caller, only the text system/user via build_messages) |

> **REFLECTION_BLOCKS / STACK_BLOCKS / LAYER_BLOCKS:** these are *content keyed by agent/stack/layer*, not single rubrics. Move the dicts into `rubrics/reflection_blocks.yaml` (and stack/layer) as data; have `build_reflection_prompt` read them via a Foundry helper. Keep the *flag* `enable_self_reflection` (Profile field) and the P5 bridge wiring (`src/core/dispatch_prep.py`) untouched.

For EACH source:

- [ ] **Step 1: Write the rubric YAML** — `key`, `system`, `user_template` (with `{field}` placeholders matching what the spawner passes).

- [ ] **Step 2: Write a failing test** asserting the spawner now produces the same messages via `build_messages` (compare system + that user contains the key fields). Example for grading:

```python
def test_grading_uses_foundry_build():
    from prompt_foundry.build import build_messages
    msgs = build_messages("grading", {"title": "T", "description": "D", "response": "R"})
    assert "SEMANTIC" in msgs[0]["content"]
    assert "R" in msgs[1]["content"]
```

- [ ] **Step 3: Swap the call site** to `build_messages(...)`; delete the now-duplicated frozen strings from the spawner.

- [ ] **Step 4: Run that package's tests** — `timeout 60 .venv/Scripts/python -m pytest packages/<pkg>/tests -q` (or `tests/...` for vision). Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "refactor(prompt_foundry): migrate <key> rubric to Foundry build_messages"`.

---

### Task 13: De-couple coulson's DB-prompt load from `src` (close the violation)

**Files:**
- Modify: `packages/coulson/src/coulson/__init__.py:203-213`
- Test: `packages/coulson/tests/test_prompt_override_via_store.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/coulson/tests/test_prompt_override_via_store.py
import asyncio
import prompt_foundry.store as store
import coulson

class _P:
    name = "summarizer"
    _prompt_version_override = None

def test_override_loaded_from_injected_store():
    class S:
        async def get_active(self, key): return "FROM STORE" if key == "summarizer" else None
        async def save_version(self, *a, **k): return 1
        async def record_quality(self, *a, **k): return None
        async def list_versions(self, k): return []
    store.set_store(S())
    p = _P()
    asyncio.get_event_loop().run_until_complete(coulson._load_db_prompt_override(p))
    assert p._prompt_version_override == "FROM STORE"
    store.set_store(None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/coulson/tests/test_prompt_override_via_store.py -q`
Expected: FAIL — coulson still imports `src.memory.prompt_versions`.

- [ ] **Step 3: Rewrite `_load_db_prompt_override` to use the injected store**

```python
# packages/coulson/src/coulson/__init__.py
async def _load_db_prompt_override(profile) -> None:
    """Load active prompt override from the injected prompt store (no src dep)."""
    profile._prompt_version_override = None
    try:
        from prompt_foundry.store import get_active
        db_prompt = await get_active(profile.name)
        if db_prompt:
            profile._prompt_version_override = db_prompt
    except Exception:
        pass
```

- [ ] **Step 4: Run test + coulson suite**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/coulson/tests/ -q`
Expected: PASS.

- [ ] **Step 5: Grep-confirm the violation is gone**

Run: `git grep -n "src.memory.prompt_versions" packages/coulson` → expected: empty.

- [ ] **Step 6: Commit** — `git commit -m "refactor(coulson): load prompt override via prompt_foundry store, drop src import"`.

---

## Phase 4 — Retarget tests + seed path

### Task 14: Retarget `test_prompt_quality.py` to the Foundry registry

**Files:**
- Modify: `tests/agents/test_prompt_quality.py`

- [ ] **Step 1: Read the current test** — it iterates agents and asserts 3 invariants (first line `You are …`; must/always + don't/never; `final_answer` + fenced ` ```json `).

- [ ] **Step 2: Repoint the source** to Foundry profiles:

```python
from prompt_foundry import PROFILE_REGISTRY

def _prompts():
    for name, profile in PROFILE_REGISTRY.items():
        yield name, profile.get_system_prompt({"id": 0, "title": "x", "description": "x"})
```
Keep the 3 invariant assertions unchanged; iterate `_prompts()`.

- [ ] **Step 3: Run**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/test_prompt_quality.py -q`
Expected: PASS for all 29 (27 static + 2 carve-outs). Fix any YAML prompt that lost a structural line in migration (re-diff against the deleted `.py`).

- [ ] **Step 4: Commit** — `git commit -m "test(prompt_foundry): prompt-quality invariants read from Foundry registry"`.

---

### Task 15: Re-point the seed path at the Foundry seed

**Files:**
- Modify: `src/memory/prompt_versions.py:180-210` (`seed_from_agents`)
- Test: `tests/memory/test_seed_from_foundry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_seed_from_foundry.py
import pytest
from src.memory.prompt_versions import seed_from_agents

@pytest.mark.asyncio
async def test_seed_reads_foundry(monkeypatch):
    saved = {}
    async def fake_get_active(at): return None
    async def fake_save(agent_type, prompt_text, notes="", activate=False):
        saved[agent_type] = prompt_text; return 1
    monkeypatch.setattr("src.memory.prompt_versions.get_active_prompt", fake_get_active)
    monkeypatch.setattr("src.memory.prompt_versions.save_prompt_version", fake_save)
    n = await seed_from_agents()
    assert n >= 27
    assert "summarizer" in saved and saved["summarizer"].startswith("You are a summarization")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/memory/test_seed_from_foundry.py -q`
Expected: FAIL — `seed_from_agents` still iterates `AGENT_REGISTRY` via `get_agent().get_system_prompt`.

- [ ] **Step 3: Repoint `seed_from_agents`** to the Foundry registry:

```python
async def seed_from_agents() -> int:
    seeded = 0
    try:
        from prompt_foundry import PROFILE_REGISTRY
        dummy = {"id": 0, "title": "seed", "description": "seed"}
        for name, profile in PROFILE_REGISTRY.items():
            try:
                if await get_active_prompt(name):
                    continue
                prompt = profile.get_system_prompt(dummy)
                if prompt and len(prompt) > 20:
                    await save_prompt_version(name, prompt, notes="Auto-seeded from Foundry", activate=True)
                    seeded += 1
            except Exception:
                continue
    except Exception as exc:
        logger.debug(f"seed_from_agents failed: {exc}")
    return seeded
```

- [ ] **Step 4: Run test**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/memory/test_seed_from_foundry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "refactor(prompt_foundry): seed_from_agents reads Foundry registry"`.

---

## Phase 5 — Guardrails

### Task 16: Dep-purity test — Foundry imports nothing from `src`/feature packages

**Files:**
- Test: `packages/prompt_foundry/tests/test_purity.py`

- [ ] **Step 1: Write the test (mirror `test_husam_does_not_import_coulson`)**

```python
# packages/prompt_foundry/tests/test_purity.py
import subprocess, sys, pathlib

SRC = pathlib.Path(__file__).parent.parent / "src" / "prompt_foundry"

def test_no_src_or_feature_imports():
    bad = []
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for marker in ("import src", "from src", "import coulson", "from coulson",
                       "import husam", "from husam", "general_beckman", "fatih_hoca",
                       "import yalayut", "from yalayut"):
            if marker in text:
                bad.append((py.name, marker))
    assert not bad, f"Foundry leaf imports forbidden deps: {bad}"

def test_import_in_clean_subprocess():
    # imports without src on path → proves leaf-ness
    r = subprocess.run([sys.executable, "-c", "import prompt_foundry; print('ok')"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
```

- [ ] **Step 2: Run**

Run: `timeout 60 .venv/Scripts/python -m pytest packages/prompt_foundry/tests/test_purity.py -q`
Expected: PASS. If FAIL, the offending import must be replaced by injection (store) or a passed-in param.

- [ ] **Step 3: Commit** — `git commit -m "test(prompt_foundry): dep-purity guardrail (leaf imports nothing from src)"`.

---

### Task 17: Classifier/workflow agent_types ⊆ registry keys

**Files:**
- Test: `tests/agents/test_agent_types_subset.py`

- [ ] **Step 1: Write the test**

```python
# tests/agents/test_agent_types_subset.py
import json, pathlib
from prompt_foundry import PROFILE_REGISTRY

def _i2p_agent_types():
    p = pathlib.Path("src/workflows/i2p/i2p_v3.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    types = set()
    def walk(o):
        if isinstance(o, dict):
            if "agent" in o and isinstance(o["agent"], str): types.add(o["agent"])
            if "agent_type" in o and isinstance(o["agent_type"], str): types.add(o["agent_type"])
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(data)
    return types

def test_i2p_agent_types_in_registry():
    keys = set(PROFILE_REGISTRY) | {"mechanical"}  # mechanical = mr_roboto, not a profile
    missing = {t for t in _i2p_agent_types() if t not in keys}
    assert not missing, f"i2p agent_types route to executor fallback: {missing}"
```

- [ ] **Step 2: Run**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/agents/test_agent_types_subset.py -q`
Expected: PASS. If FAIL, either a profile is misnamed or i2p references a dropped agent — reconcile (a real routing bug surfaced).

- [ ] **Step 3: Commit** — `git commit -m "test(prompt_foundry): assert i2p/classifier agent_types subset of registry"`.

---

### Task 18: Extend `test_root_stays_thin.py` — `src/agents/*.py` stays small

**Files:**
- Modify: `tests/test_root_stays_thin.py`

- [ ] **Step 1: Add the assertion**

```python
def test_src_agents_is_thin():
    import pathlib
    pys = list(pathlib.Path("src/agents").glob("*.py"))
    names = {p.name for p in pys}
    # Only the shim + base residual remain; profiles live in prompt_foundry as data.
    allowed = {"__init__.py", "base.py"}
    extra = names - allowed
    assert not extra, f"new agent classes must be Foundry YAML, not src/agents/*.py: {extra}"
```

> If `base.py` is fully retired by the A.12/A.13 follow-on track, tighten `allowed` to `{"__init__.py"}` then.

- [ ] **Step 2: Run**

Run: `timeout 60 .venv/Scripts/python -m pytest tests/test_root_stays_thin.py -q`
Expected: PASS.

- [ ] **Step 3: Commit** — `git commit -m "test(prompt_foundry): guard src/agents stays thin (new agents = data)"`.

---

## Final verification (before merge)

- [ ] `timeout 60 .venv/Scripts/python -c "from src.agents import get_agent; print(get_agent('coder').name, get_agent('coder') is get_agent('coder'))"` → `coder True`.
- [ ] `timeout 60 .venv/Scripts/python -m pytest tests/agents/ -q` → all pass.
- [ ] `.venv/Scripts/python -m pytest packages/prompt_foundry/tests/ -q` (`timeout 60`) → all pass.
- [ ] `.venv/Scripts/python -m pytest packages/coulson/tests/ -q` (`timeout 60`) → all pass.
- [ ] `git grep -n "src.memory.prompt_versions" packages/` → only the adapter's allowed bridge is in `src/`, none in `packages/`.
- [ ] `git grep -nE "import .*Agent" src/ packages/` (excluding tests) → no concrete `*Agent` class imports remain.
- [ ] **Founder names the package** → rename `prompt_foundry` dir + `pyproject` + imports in one mechanical commit.
- [ ] **Live:** 1 multi-step mission touching coder/reviewer/researcher; confirm prompts resolve, self-reflection fires for coder, and a grading posthook runs via the new build path. (USER restart-gated.)
- [ ] Merge worktree → main in a quiet window (`project_z1_merge_pattern_20260510`).

## Out of scope (separate tracks)
- Broader src-DB-dep kill for other packages (this ships only the reference port).
- base.py A.12/A.13 residual relocation (`_build_model_requirements` → fatih_hoca, `_maybe_constrained_emit` → coulson).
