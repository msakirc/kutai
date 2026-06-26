# Canonical Product-Name Pin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin one canonical product name for an i2p mission, inject it into every step's prompt, and mechanically enforce that `reverse_pitch` and `product_charter` contain it — so the HabitTrack-vs-FlowState drift that halted mission 89 becomes structurally near-impossible.

**Architecture:** A new phase_0 LLM step `0.0y product_name_pick` emits a `product_name` artifact into the artifact store via `output_artifacts`. The async `build_user_context` reads it fresh from the store each dispatch and injects a "Product Name (canonical)" line. A new mechanical check `verify_contains_product_name` (declared on `0.0z` and `0.1`) reads the same store artifact and whole-word-verifies the produced doc contains the name; absence blocks → producer re-pends with the name in its prompt → converges. The LLM reviewer (1.13 check 10) stays as backstop. No `missions.context` writes (no inject_lessons race).

**Tech Stack:** Python 3.10 async, pytest, SQLite/aiosqlite, the KutAI i2p workflow engine (mr_roboto mechanical executors, coulson prompt builders, general_beckman post-hook registry).

**Spec:** `docs/superpowers/specs/2026-06-23-canonical-product-name-pin-design.md`

---

## Task 1: `verify_contains_product_name` pure helper

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/verify_contains_product_name.py`
- Test: `packages/mr_roboto/tests/test_verify_contains_product_name.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_verify_contains_product_name.py
from mr_roboto.verify_contains_product_name import verify_contains_product_name


def test_present_whole_word_passes():
    res = verify_contains_product_name(
        product_name="FlowState",
        artifact_texts=["# Launch\nIntroducing FlowState, the app."],
    )
    assert res["ok"] is True
    assert res["found"] is True


def test_absent_fails():
    res = verify_contains_product_name(
        product_name="FlowState",
        artifact_texts=["# Launch\nIntroducing HabitTrack, the app."],
    )
    assert res["ok"] is False
    assert res["found"] is False


def test_case_insensitive():
    res = verify_contains_product_name(
        product_name="FlowState", artifact_texts=["we love flowstate here"],
    )
    assert res["ok"] is True


def test_substring_in_larger_word_does_not_match():
    # "Flow" must not match inside "Flowers"; whole-word only.
    res = verify_contains_product_name(
        product_name="Flow", artifact_texts=["Flowers everywhere, no product"],
    )
    assert res["found"] is False
    assert res["ok"] is False


def test_empty_name_is_defensive_skip():
    res = verify_contains_product_name(product_name="  ", artifact_texts=["anything"])
    assert res["ok"] is True
    assert res["skipped"] == "no product_name pinned"


def test_none_name_is_defensive_skip():
    res = verify_contains_product_name(product_name=None, artifact_texts=[])
    assert res["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_verify_contains_product_name.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mr_roboto.verify_contains_product_name'`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/mr_roboto/src/mr_roboto/verify_contains_product_name.py
"""Whole-word presence check: does an artifact contain the canonical product name?

Pure / deterministic — no I/O. The mr_roboto dispatch branch supplies the
`product_name` (read from the artifact store) and the artifact texts (read from
disk); this module only decides present/absent. When `product_name` is empty or
None the check is a defensive SKIP (ok=True) — we never hard-block on our own
missing precondition; the reviewer backstop (1.13 check 10) covers that case.
"""
from __future__ import annotations

import re
from typing import Any


def _whole_word_present(text: str, name: str) -> bool:
    if not text or not name:
        return False
    return re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE) is not None


def verify_contains_product_name(
    *, product_name: str | None, artifact_texts: list[str]
) -> dict[str, Any]:
    name = (product_name or "").strip()
    if not name:
        return {
            "ok": True, "skipped": "no product_name pinned",
            "product_name": None, "checked": len(artifact_texts), "found": False,
        }
    found = any(_whole_word_present(t or "", name) for t in artifact_texts)
    return {
        "ok": bool(found), "product_name": name,
        "checked": len(artifact_texts), "found": found,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_verify_contains_product_name.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/verify_contains_product_name.py packages/mr_roboto/tests/test_verify_contains_product_name.py
git commit -m "feat(mr_roboto): add verify_contains_product_name pure check"
```

---

## Task 2: wire `verify_contains_product_name` into the mr_roboto dispatcher

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (add a branch in `_run_dispatch`, near the `verify_charter_shape` branch ~line 1100)
- Modify: `packages/mr_roboto/src/mr_roboto/reversibility.py` (add the verb to `VERB_REVERSIBILITY`)
- Test: `packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py`

> **Note on the store mock:** the real `ArtifactStore.retrieve` returns a **JSON
> string** (the engine stores the model's raw result string at `hooks.py:1641`), not a
> Python dict. The tests below cover BOTH the string case (production truth) and the
> dict case (defensive), because the dispatch branch handles both.

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py
import json
import pytest


@pytest.mark.asyncio
async def test_dispatch_passes_when_artifact_contains_name(tmp_path, monkeypatch):
    import mr_roboto

    # Fake artifact store returning the pinned name as the JSON STRING the
    # engine actually stores (production truth).
    class _Store:
        async def retrieve(self, mid, name):
            assert name == "product_name"
            return '{"product_name": "FlowState"}'

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    # Make _resolve_path_list return our temp file verbatim (absolute path).
    art = tmp_path / "reverse_pitch.md"
    art.write_text("# Launch\nIntroducing FlowState.", encoding="utf-8")
    monkeypatch.setattr(mr_roboto, "_resolve_path_list", lambda paths: [str(art)])

    task = {
        "id": 1, "mission_id": 42,
        "payload": {
            "action": "verify_contains_product_name",
            "artifact_paths": [str(art)],
        },
    }
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "completed"
    assert res.result["found"] is True


@pytest.mark.asyncio
async def test_dispatch_fails_when_artifact_missing_name(tmp_path, monkeypatch):
    import mr_roboto

    class _Store:
        async def retrieve(self, mid, name):
            return {"product_name": "FlowState"}

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    art = tmp_path / "reverse_pitch.md"
    art.write_text("# Launch\nIntroducing HabitTrack.", encoding="utf-8")
    monkeypatch.setattr(mr_roboto, "_resolve_path_list", lambda paths: [str(art)])

    task = {
        "id": 1, "mission_id": 42,
        "payload": {
            "action": "verify_contains_product_name",
            "artifact_paths": [str(art)],
        },
    }
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "failed"
    assert "FlowState" in (res.error or "")


@pytest.mark.asyncio
async def test_dispatch_skips_when_no_name_pinned(tmp_path, monkeypatch):
    import mr_roboto

    class _Store:
        async def retrieve(self, mid, name):
            return None  # nothing pinned yet

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    art = tmp_path / "reverse_pitch.md"
    art.write_text("anything", encoding="utf-8")
    monkeypatch.setattr(mr_roboto, "_resolve_path_list", lambda paths: [str(art)])

    task = {
        "id": 1, "mission_id": 42,
        "payload": {
            "action": "verify_contains_product_name",
            "artifact_paths": [str(art)],
        },
    }
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "completed"  # defensive skip, never hard-block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py -q`
Expected: FAIL — `_run_dispatch` returns a failed/unknown-action Action (no branch yet), so `test_dispatch_passes...` fails its `status == "completed"` assertion.

- [ ] **Step 3: Add the dispatch branch**

In `packages/mr_roboto/src/mr_roboto/__init__.py`, immediately AFTER the existing `if action == "verify_charter_shape":` block (ends ~line 1127 with `return Action(status="failed", error=str(e))`), insert:

```python
    if action == "verify_contains_product_name":
        # Z1 — canonical product-name enforcement. Reads the pinned name from
        # the artifact store (produced by step 0.0y) and whole-word-checks that
        # the produced doc (reverse_pitch / product_charter) contains it.
        from mr_roboto.verify_contains_product_name import (
            verify_contains_product_name as _verify_pn,
        )
        try:
            name = None
            mid = task.get("mission_id")
            if mid is not None:
                from src.workflows.engine.hooks import get_artifact_store
                raw = await get_artifact_store().retrieve(int(mid), "product_name")
                if isinstance(raw, dict):
                    name = raw.get("product_name")
                elif isinstance(raw, str) and raw.strip():
                    import json as _json
                    try:
                        _d = _json.loads(raw)
                        name = _d.get("product_name") if isinstance(_d, dict) else raw
                    except Exception:
                        name = raw
            texts: list[str] = []
            for _p in (_resolve_path_list(payload.get("artifact_paths")) or []):
                try:
                    with open(_p, "r", encoding="utf-8") as _fh:
                        texts.append(_fh.read())
                except Exception:
                    continue
            res = _verify_pn(product_name=name, artifact_texts=texts)
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=(
                        "verify_contains_product_name: produced artifact does not "
                        f"contain canonical product name {res.get('product_name')!r}"
                    ),
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/__init__.py packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py
git commit -m "feat(mr_roboto): dispatch verify_contains_product_name (store read + whole-word check)"
```

- [ ] **Step 6: Write the failing reversibility test**

Every sibling `verify_*` verb maps to `"full"` in `VERB_REVERSIBILITY`. Without an entry, the verb defaults to `"partial"` — which under an opt-in `KUTAI_CONFIRM_POLICY=partial_or_worse` would arm a confirm gate and park this read-only check as `waiting_human` (silently disabling enforcement). Add the mapping.

```python
# append to packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py
def test_verify_contains_product_name_is_full_reversibility():
    from mr_roboto.reversibility import get_reversibility
    assert get_reversibility("verify_contains_product_name") == "full"
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py::test_verify_contains_product_name_is_full_reversibility -q`
Expected: FAIL — resolves to `"partial"` (the default), not `"full"`.

- [ ] **Step 8: Add the reversibility mapping**

In `packages/mr_roboto/src/mr_roboto/reversibility.py`, in the `VERB_REVERSIBILITY` dict, next to the other `verify_*` entries, add:

```python
    "verify_contains_product_name": "full",  # read-only presence check
```

- [ ] **Step 9: Run test to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py -q`
Expected: PASS (4 passed).

- [ ] **Step 10: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/reversibility.py packages/mr_roboto/tests/test_verify_contains_product_name_dispatch.py
git commit -m "fix(mr_roboto): map verify_contains_product_name reversibility=full"
```

---

## Task 3: register `verify_contains_product_name` in the post-hook registry

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/posthooks.py` (add one line to `POST_HOOK_REGISTRY`, next to `verify_charter_shape` ~line 505)
- Test: `packages/general_beckman/tests/test_posthook_registry_product_name.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/general_beckman/tests/test_posthook_registry_product_name.py
def test_verify_contains_product_name_registered_as_blocker_check():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["verify_contains_product_name"]
    assert spec.kind == "verify_contains_product_name"
    assert spec.verb == "verify_contains_product_name"
    assert spec.default_severity == "blocker"
    assert spec.auto_wire_triggers == []


def test_check_kind_is_derived_for_apply():
    # apply._CHECK_KINDS derives verify_* kinds from the registry by name.
    from general_beckman.apply import _CHECK_KINDS
    assert "verify_contains_product_name" in _CHECK_KINDS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/general_beckman/tests/test_posthook_registry_product_name.py -q`
Expected: FAIL — `KeyError: 'verify_contains_product_name'`.

- [ ] **Step 3: Add the registry entry**

In `packages/general_beckman/src/general_beckman/posthooks.py`, inside `POST_HOOK_REGISTRY`, immediately after the `"verify_charter_shape": _shape_check_spec(...)` entry (~line 505), add:

```python
    "verify_contains_product_name": _shape_check_spec(
        "verify_contains_product_name",
        "Canonical product-name presence in reverse_pitch / product_charter."),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/general_beckman/tests/test_posthook_registry_product_name.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/posthooks.py packages/general_beckman/tests/test_posthook_registry_product_name.py
git commit -m "feat(beckman): register verify_contains_product_name check kind"
```

---

## Task 4: product-name prompt injection helpers in coulson

**Files:**
- Modify: `packages/coulson/src/coulson/context.py` (add `_load_product_name` + `_product_name_block`; call them in `build_user_context`)
- Test: `packages/coulson/tests/test_product_name_injection.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/coulson/tests/test_product_name_injection.py
import pytest


def test_product_name_block_present():
    from coulson.context import _product_name_block
    block = _product_name_block("FlowState")
    assert block is not None
    assert "FlowState" in block
    assert "EXACTLY" in block


def test_product_name_block_empty_returns_none():
    from coulson.context import _product_name_block
    assert _product_name_block("   ") is None
    assert _product_name_block(None) is None


@pytest.mark.asyncio
async def test_load_product_name_from_store(monkeypatch):
    from coulson import context as ctx

    # The engine stores the model result as a JSON STRING (hooks.py:1641).
    class _Store:
        async def retrieve(self, mid, name):
            assert name == "product_name"
            return '{"product_name": "FlowState"}'

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    assert await ctx._load_product_name(42) == "FlowState"


@pytest.mark.asyncio
async def test_load_product_name_missing_returns_none(monkeypatch):
    from coulson import context as ctx

    class _Store:
        async def retrieve(self, mid, name):
            return None

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    assert await ctx._load_product_name(42) is None
    assert await ctx._load_product_name(None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/coulson/tests/test_product_name_injection.py -q`
Expected: FAIL — `ImportError: cannot import name '_product_name_block'`.

- [ ] **Step 3: Add the helpers and the injection call**

In `packages/coulson/src/coulson/context.py`, add these two module-level functions (place them near `_get_mission_lessons_cached`):

```python
def _product_name_block(name: str | None) -> str | None:
    """The canonical-product-name prompt block, or None when no name is pinned."""
    name = (name or "").strip()
    if not name:
        return None
    return (
        "## Product Name (canonical)\n"
        f"The product is named **{name}**. Use this name EXACTLY in every "
        "artifact. Do NOT invent or vary the name."
    )


async def _load_product_name(mission_id) -> str | None:
    """Best-effort read of the pinned product name from the artifact store
    (produced by i2p step 0.0y). Returns the stripped name or None. Never raises.
    Mirrors inject_north_star._load_success_metrics."""
    if mission_id is None:
        return None
    try:
        from src.workflows.engine.hooks import get_artifact_store
        raw = await get_artifact_store().retrieve(int(mission_id), "product_name")
    except Exception:
        return None
    name = None
    if isinstance(raw, dict):
        name = raw.get("product_name")
    elif isinstance(raw, str) and raw.strip():
        import json as _json
        try:
            _d = _json.loads(raw)
            name = _d.get("product_name") if isinstance(_d, dict) else raw
        except Exception:
            name = raw
    name = (name or "").strip()
    return name or None
```

Then in `build_user_context` (async), immediately AFTER the `task_context` parse block (the lines that normalize `task_context` to a dict, ~line 790), add:

```python
    # ── Canonical product name (i2p) — injected fresh from the store ──
    try:
        _pn = await _load_product_name(
            task.get("mission_id") or task_context.get("mission_id")
        )
        _pn_block = _product_name_block(_pn)
        if _pn_block:
            parts.append(_pn_block)
    except Exception:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/coulson/tests/test_product_name_injection.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/coulson/src/coulson/context.py packages/coulson/tests/test_product_name_injection.py
git commit -m "feat(coulson): inject canonical product name into build_user_context"
```

---

## Task 5: insert step `0.0y` and wire checks into i2p_v3.json

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` (insert `0.0y` immediately before `0.0z`; flip `0.0z.depends_on`; add a `checks[]` entry to `0.0z` and `0.1`)
- Test: `tests/workflows/test_product_name_pin_workflow.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workflows/test_product_name_pin_workflow.py
import json
from pathlib import Path

WF = Path("src/workflows/i2p/i2p_v3.json")


def _steps():
    data = json.loads(WF.read_text(encoding="utf-8"))
    return data["steps"], {s["id"]: s for s in data["steps"]}


def test_naming_step_exists_and_is_object_no_produces():
    steps, by_id = _steps()
    s = by_id["0.0y"]
    assert s["agent"] == "analyst"
    assert s["depends_on"] == []
    assert s["input_artifacts"] == ["raw_idea", "strategic_context"]
    assert s["output_artifacts"] == ["product_name"]
    assert "produces" not in s
    assert s["artifact_schema"]["product_name"]["type"] == "object"
    assert "product_name" in s["artifact_schema"]["product_name"]["required_fields"]
    assert s.get("requires_grading") is not False
    assert "tools_hint" in s and "difficulty" in s


def test_0_0y_is_before_0_0z_in_array():
    steps, _ = _steps()
    ids = [s["id"] for s in steps]
    assert ids.index("0.0y") < ids.index("0.0z")


def test_0_0z_depends_on_naming_step():
    _, by_id = _steps()
    assert by_id["0.0z"]["depends_on"] == ["0.0y"]


def test_checks_declared_on_pitch_and_charter():
    _, by_id = _steps()
    for sid, art in (("0.0z", "reverse_pitch.md"), ("0.1", "product_charter.md")):
        kinds = [c["kind"] for c in by_id[sid].get("checks", [])]
        assert "verify_contains_product_name" in kinds
        chk = next(c for c in by_id[sid]["checks"]
                   if c["kind"] == "verify_contains_product_name")
        assert chk["payload"]["action"] == "verify_contains_product_name"
        assert any(art in p for p in chk["payload"]["artifact_paths"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflows/test_product_name_pin_workflow.py -q`
Expected: FAIL — `KeyError: '0.0y'` (step not present).

- [ ] **Step 3: Edit the workflow JSON**

3a. Insert this new step object in the `steps` array **immediately before** the `0.0z` object (so it is physically earlier — required by `test_no_forward_reference_dependencies`):

```json
    {
      "id": "0.0y",
      "phase": "phase_0",
      "name": "product_name_pick",
      "agent": "analyst",
      "difficulty": "easy",
      "tools_hint": [],
      "depends_on": [],
      "may_need_clarification": false,
      "input_artifacts": [ "raw_idea", "strategic_context" ],
      "output_artifacts": [ "product_name" ],
      "instruction": "Decide the ONE canonical product name for this idea, using raw_idea and strategic_context. Pick a single, real, memorable brand name (not a description, not a placeholder, not a code). Every later artifact will be forced to use this exact name, so choose deliberately. Do NOT search the web. Do NOT write any file. IMPORTANT: produce your answer as a JSON object in your final response; your response text IS the artifact. Emit exactly: {\"product_name\": \"<TheName>\"} with a non-empty name.",
      "done_when": "A single non-empty product_name has been chosen.",
      "artifact_schema": {
        "product_name": {
          "type": "object",
          "required_fields": [ "product_name" ]
        }
      },
      "context": { "estimated_output_tokens": 200 },
      "reversibility": "full"
    },
```

3b. In the `0.0z` step, change `"depends_on": []` to `"depends_on": [ "0.0y" ]`.

3c. In the `0.0z` step, add a second entry to its existing `checks` array (alongside `verify_reverse_pitch_shape`):

```json
        {
          "kind": "verify_contains_product_name",
          "payload": {
            "action": "verify_contains_product_name",
            "artifact_paths": [ "mission_{mission_id}/.charter/reverse_pitch.md" ]
          }
        }
```

3d. In the `0.1` step, add a second entry to its existing `checks` array (alongside `verify_charter_shape`):

```json
        {
          "kind": "verify_contains_product_name",
          "payload": {
            "action": "verify_contains_product_name",
            "artifact_paths": [ "mission_{mission_id}/.charter/product_charter.md" ]
          }
        }
```

- [ ] **Step 4: Run tests to verify they pass (and nothing structural broke)**

Run: `python -m pytest tests/workflows/test_product_name_pin_workflow.py tests/test_i2p_v3.py tests/workflows/test_i2p_v3_dep_integrity.py -q`
Expected: new tests green; `test_i2p_v3_dep_integrity.py` green. **Baseline caveat:** `tests/test_i2p_v3.py::test_v3_all_steps_have_artifact_schema` is **already RED on `main`** (`Step 13.demo_storyboard_draft missing artifact_schema` — unrelated, pre-existing). Confirm your change adds **no NEW** failures: the `test_i2p_v3.py` group baseline is `1 failed, 50 passed`. Run it once before Task 5 to capture the baseline, and confirm the same single failure after — `0.0y` must NOT appear in any failure.

- [ ] **Step 5: Verify the JSON is well-formed**

Run: `python -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json', encoding='utf-8')); print('OK')"`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json tests/workflows/test_product_name_pin_workflow.py
git commit -m "feat(i2p): pin canonical product name (0.0y) + enforce on reverse_pitch/charter"
```

---

## Task 6: full regression + import smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the affected package test suites**

Run: `python -m pytest packages/mr_roboto/tests packages/general_beckman/tests/test_posthook_registry_product_name.py packages/coulson/tests/test_product_name_injection.py tests/workflows/test_product_name_pin_workflow.py -q`
Expected: PASS, no errors/warnings beyond the known pynvml FutureWarning.

- [ ] **Step 2: Import smoke (no circular-import regressions)**

Run: `python -c "import mr_roboto; from coulson import context; from general_beckman import posthooks; print('imports OK')"`
Expected: `imports OK`.

- [ ] **Step 3: Phase-code-leak guard still green**

Run: `python -m pytest tests/workflows/test_i2p_v3_phase_code_leak.py -q`
Expected: PASS (the 0.0y instruction has no "product (CODE)" adjacency).

- [ ] **Step 4: Commit (if any test fixtures were adjusted)**

```bash
git add -A
git commit -m "test: regression pass for canonical product-name pin" || echo "nothing to commit"
```

---

## Self-review notes

- **Spec coverage:** ORIGIN (Task 5 `0.0y`) · PROPAGATION (Task 4 injection) · ENFORCEMENT (Tasks 1-3 check + registry + Task 5 `checks[]`). Reviewer-backstop unchanged (no task — intentional). All spec "Files touched" map to a task.
- **Type consistency:** `verify_contains_product_name(product_name=, artifact_texts=)` and the dispatch branch's `result` keys (`ok`, `found`, `product_name`, `skipped`) are identical across Tasks 1-2. `_load_product_name` / `_product_name_block` names match across Task 4 and tests.
- **No `missions.context` write** anywhere → no inject_lessons race (the prior draft's risk is gone by construction).
- **Restart-gated deploy:** after Task 6 green → `/restart` → live-verify a fresh mission (name pinned; pitch + charter agree) → only then push.
