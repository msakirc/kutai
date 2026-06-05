# Deterministic Materializer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one deterministic engine pass the sole writer of declared `produces` paths, producing a single canonical artifact used for both the on-disk file and the in-memory `output_value` the schema gate validates.

**Architecture:** Generalize the engine fill-missing hook (`src/workflows/engine/hooks.py:1500-1557`) into `materialize_produces()`. It gathers candidates (on-disk write + `output_value`, each optionally fence-unwrapped), selects the best by the declared `artifact_schema`, stamps `mission_id` front-matter idempotently, and writes the canonical path last — then returns that content so the schema gate validates exactly what is on disk. The scattered coulson auto-persist/canonicalize blocks and the legacy root-`.md` write are removed; `constrained_emit` (LLM JSON reshape) stays as the after-the-fact rescue.

**Tech Stack:** Python 3.10, pytest (+pytest-timeout, 120s default), aiosqlite, sentence-transformers (embeddings, not touched here). Pure helpers live in `packages/coulson/src/coulson/grounding.py`; the impure orchestrator lives in `src/workflows/engine/hooks.py`.

**Spec:** `docs/superpowers/specs/2026-06-05-deterministic-materializer-design.md`

---

## File Structure

- `packages/coulson/src/coulson/grounding.py` — add two pure functions: `stamp_front_matter()`, `select_canonical()`. Reuse existing `unwrap_fenced_artifact()`.
- `packages/coulson/tests/test_materializer_helpers.py` — **create**; unit tests for the two pure functions.
- `src/workflows/engine/hooks.py` — add `materialize_produces()` (impure orchestrator); replace the legacy root-`.md` write (`1500-1513`) + fill-missing block (`1515-1557`) with a single call; retire `_produces_file_is_stale()` (`272-300`).
- `tests/workflows/test_materialize_produces.py` — **create**; integration tests against a temp workspace.
- `packages/coulson/src/coulson/react.py` — **remove** the AUTO-PERSIST block (`820-864`) and CANONICALIZE OVERRIDE block (`866-938`); the engine now owns materialization.

---

## Task 1: `stamp_front_matter()` — idempotent mission_id stamp

**Files:**
- Modify: `packages/coulson/src/coulson/grounding.py`
- Test: `packages/coulson/tests/test_materializer_helpers.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# packages/coulson/tests/test_materializer_helpers.py
"""Materializer pure helpers (deterministic-materializer spec, 2026-06-05)."""
from __future__ import annotations

import json

from coulson.grounding import stamp_front_matter, select_canonical


# ── stamp_front_matter — markdown ──────────────────────────────────────────

def test_md_no_frontmatter_prepends_block():
    out = stamp_front_matter("# Charter\n\nbody", 81, "md")
    assert out.startswith("---\n")
    assert "mission_id: 81" in out.split("---")[1]
    assert "# Charter" in out


def test_md_frontmatter_missing_mission_id_injects():
    src = '---\ntitle: "x"\n---\n\n# Charter\nbody'
    out = stamp_front_matter(src, 81, "md")
    fm = out.split("---")[1]
    assert "mission_id: 81" in fm
    assert 'title: "x"' in fm
    # still exactly one front-matter block (two `---` fences)
    assert out.count("---") == 2


def test_md_frontmatter_with_mission_id_is_unchanged():
    src = '---\nmission_id: 81\ntitle: "x"\n---\n\n# Charter'
    assert stamp_front_matter(src, 81, "md") == src  # idempotent, no double-stamp


# ── stamp_front_matter — json ──────────────────────────────────────────────

def test_json_injects_mission_id_when_absent():
    out = stamp_front_matter('{"items": [1, 2]}', 81, "json")
    assert json.loads(out)["mission_id"] == 81
    assert json.loads(out)["items"] == [1, 2]


def test_json_with_mission_id_is_unchanged():
    src = '{"mission_id": 81, "items": []}'
    assert json.loads(stamp_front_matter(src, 81, "json"))["mission_id"] == 81


def test_json_unparseable_returned_as_is():
    assert stamp_front_matter("{ broken", 81, "json") == "{ broken"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest packages/coulson/tests/test_materializer_helpers.py -q`
Expected: FAIL with `ImportError: cannot import name 'stamp_front_matter'`

- [ ] **Step 3: Implement `stamp_front_matter`**

Add to `packages/coulson/src/coulson/grounding.py` (after `unwrap_fenced_artifact`):

```python
import json as _json

# Leading YAML front-matter: ``---\n...\n---`` at the very start of the file.
_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def stamp_front_matter(content: str, mission_id: int, kind: str) -> str:
    """Idempotently stamp ``mission_id`` into an artifact's metadata.

    ``md``  : ensure a leading ``---`` front-matter block carrying
              ``mission_id``. Inject the key if the block exists without it;
              prepend a minimal block if absent; no-op if already present.
              Never produces a second ``---`` block (handoff Q(c)).
    ``json``: ensure a top-level ``mission_id`` key. No-op if present or the
              content does not parse (best-effort — never corrupt a file).
    """
    if not isinstance(content, str):
        return content
    if kind == "json":
        try:
            obj = _json.loads(content)
        except (ValueError, TypeError):
            return content
        if isinstance(obj, dict) and "mission_id" not in obj:
            obj = {"mission_id": mission_id, **obj}
            return _json.dumps(obj, ensure_ascii=False, indent=2)
        return content

    # markdown
    m = _FRONT_MATTER_RE.match(content)
    if m:
        body = m.group(1)
        if re.search(r"^\s*mission_id\s*:", body, re.MULTILINE):
            return content  # already stamped — idempotent
        new_block = f"---\n{body}\nmission_id: {mission_id}\n---\n"
        return new_block + content[m.end():]
    return f"---\nmission_id: {mission_id}\n---\n\n{content}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest packages/coulson/tests/test_materializer_helpers.py -q`
Expected: PASS (6 passed) — `select_canonical` import will still error; see Task 2. If so, run only the `stamp_front_matter` tests: append `-k stamp` to the command; expect PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/coulson/src/coulson/grounding.py packages/coulson/tests/test_materializer_helpers.py
git commit -m "feat(coulson): stamp_front_matter — idempotent mission_id stamp"
```

---

## Task 2: `select_canonical()` — pick the best artifact candidate

**Files:**
- Modify: `packages/coulson/src/coulson/grounding.py`
- Test: `packages/coulson/tests/test_materializer_helpers.py`

- [ ] **Step 1: Write the failing tests** (append to the test file)

```python
# ── select_canonical ───────────────────────────────────────────────────────

_NARRATION_WRAP = (
    "## Analysis\n\n### Corrected Artifact Content\n\n"
    "```yaml\n---\nmission_id: 81\n---\n\n## Landscape\nx\n\n## Notes\ny\n```\n"
)
_DISK_NARRATION = "## Analysis\n### Findings\n- listed\n### Recommendations\nready."


def _needs(*sections):
    def _ok(c: str) -> bool:
        return all(f"## {s}" in c for s in sections)
    return _ok


def test_prefers_unwrapped_artifact_over_narration_wrapper():
    schema_ok = _needs("Landscape", "Notes")
    got = select_canonical([_NARRATION_WRAP, _DISK_NARRATION], schema_ok)
    assert got is not None
    assert got.strip().startswith("---")          # cleaned, front-matter first
    assert "## Landscape" in got and "```" not in got
    assert "### Corrected Artifact Content" not in got


def test_keeps_raw_doc_when_no_fence():
    schema_ok = _needs("Vision")
    doc = "# Charter\n\n## Vision\nbody"
    assert select_canonical([doc, None], schema_ok) == doc


def test_returns_most_substantial_when_none_conform():
    schema_ok = _needs("DoesNotExist")
    short, long = "## A\nx", "## B\n" + ("word " * 50)
    assert select_canonical([short, long], schema_ok) == long


def test_none_when_no_usable_candidate():
    assert select_canonical([None, "", "   "], _needs("X")) is None
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/Scripts/python.exe -m pytest packages/coulson/tests/test_materializer_helpers.py -k select -q`
Expected: FAIL with `ImportError: cannot import name 'select_canonical'`

- [ ] **Step 3: Implement `select_canonical`**

Add to `packages/coulson/src/coulson/grounding.py`:

```python
def select_canonical(candidates, schema_ok: Callable[[str], bool]):
    """Pick the best artifact form from competing candidates.

    ``candidates`` is an ordered list of raw strings (e.g. ``[output_value,
    disk_content]``). For each, the fence-unwrapped form is also considered.
    Among forms that PASS ``schema_ok`` the cleanest is chosen (front-matter
    / header at file start, then shortest — least surrounding narration). If
    none pass, the most-substantial form is returned so a file always exists.
    Returns ``None`` only when there is no usable (non-blank) candidate.
    """
    forms: list[str] = []
    for c in candidates:
        if not isinstance(c, str) or not c.strip():
            continue
        forms.append(c)
        u = unwrap_fenced_artifact(c)
        if isinstance(u, str) and u.strip() and u.strip() != c.strip():
            forms.append(u)
    if not forms:
        return None

    passing = [f for f in forms if schema_ok(f)]
    if passing:
        def _rank(f: str):
            s = f.strip()
            starts_clean = s.startswith(("---", "#", "{", "["))
            return (0 if starts_clean else 1, len(s))
        return min(passing, key=_rank)
    return max(forms, key=lambda f: len(f.strip()))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest packages/coulson/tests/test_materializer_helpers.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add packages/coulson/src/coulson/grounding.py packages/coulson/tests/test_materializer_helpers.py
git commit -m "feat(coulson): select_canonical — schema-aware artifact candidate selection"
```

---

## Task 3: `materialize_produces()` — the sole-writer orchestrator

**Files:**
- Modify: `src/workflows/engine/hooks.py` (add function near the other helpers, e.g. after `_produces_file_is_stale` around line 300)
- Test: `tests/workflows/test_materialize_produces.py` (create)

- [ ] **Step 1: Write the failing integration tests**

```python
# tests/workflows/test_materialize_produces.py
"""materialize_produces — sole writer of produces paths (spec 2026-06-05)."""
from __future__ import annotations

import json
import os

import pytest

from src.workflows.engine.hooks import materialize_produces


_SCHEMA_MD = {"type": "markdown", "required_sections": ["Landscape", "Notes"]}

_NARRATION_WRAP = (
    "## Analysis\n\n### Corrected Artifact Content\n\n"
    "```yaml\n---\nmission_id: 81\n---\n\n## Landscape\nx\n\n## Notes\ny\n```\n"
)


def _ctx(produces, schema=None):
    c = {"produces": produces}
    if schema is not None:
        c["artifact_schema"] = schema
    return c


@pytest.mark.asyncio
async def test_writes_unwrapped_canonical_and_returns_it(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr("src.workflows.engine.hooks.WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_81/.prd/competitive_positioning.md"
    ctx = _ctx([rel], _SCHEMA_MD)
    task = {"mission_id": 81, "agent_type": "analyst"}

    out = await materialize_produces(ctx, task, {"result": _NARRATION_WRAP}, _NARRATION_WRAP)

    disk = (tmp_path / rel).read_text(encoding="utf-8")
    assert disk.strip().startswith("---")       # front-matter at file start
    assert "## Landscape" in disk and "```" not in disk
    assert out == disk                            # returned == on-disk (gate parity)


@pytest.mark.asyncio
async def test_mechanical_executor_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("src.workflows.engine.hooks.WORKSPACE_DIR", str(tmp_path), raising=False)
    ctx = _ctx(["mission_81/x.md"], _SCHEMA_MD)
    task = {"mission_id": 81, "executor": "mechanical"}
    out = await materialize_produces(ctx, task, {}, "## Landscape\nx")
    assert out == "## Landscape\nx"               # unchanged
    assert not (tmp_path / "mission_81/x.md").exists()


@pytest.mark.asyncio
async def test_no_schema_writes_passthrough(tmp_path, monkeypatch):
    monkeypatch.setattr("src.workflows.engine.hooks.WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_81/notes.md"
    ctx = _ctx([rel])                              # no artifact_schema
    task = {"mission_id": 81, "agent_type": "analyst"}
    body = "# Notes\n\nplain body"
    out = await materialize_produces(ctx, task, {}, body)
    assert (tmp_path / rel).read_text(encoding="utf-8").endswith(body)  # stamped + body
    assert "mission_id: 81" in (tmp_path / rel).read_text(encoding="utf-8")
    assert out.endswith(body)


@pytest.mark.asyncio
async def test_json_unwrapped_and_stamped(tmp_path, monkeypatch):
    monkeypatch.setattr("src.workflows.engine.hooks.WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_81/.intake/draft.json"
    schema = {"type": "object", "required": ["items"]}
    ctx = _ctx([rel], schema)
    task = {"mission_id": 81, "agent_type": "analyst"}
    wrapped = '## Summary\n```json\n{"items": [1, 2]}\n```\n'
    out = await materialize_produces(ctx, task, {}, wrapped)
    disk = json.loads((tmp_path / rel).read_text(encoding="utf-8"))
    assert disk["items"] == [1, 2]
    assert disk["mission_id"] == 81
    assert json.loads(out)["items"] == [1, 2]
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/workflows/test_materialize_produces.py -q`
Expected: FAIL with `ImportError: cannot import name 'materialize_produces'`

- [ ] **Step 3: Implement `materialize_produces`**

Add to `src/workflows/engine/hooks.py` (import the helpers at top of the function to avoid load-order issues):

```python
async def materialize_produces(ctx: dict, task: dict, result, output_value):
    """Sole writer of declared ``produces`` paths.

    For each single declared ``.md`` / ``.json`` produces path, gather the
    on-disk content (whatever the agent's write_file left) and ``output_value``
    as competing candidates, pick the schema-best via ``select_canonical``,
    stamp ``mission_id`` front-matter idempotently, and write the canonical
    path last. Returns the canonical content as the new ``output_value`` (when
    a single path is declared) so the in-memory schema gate validates exactly
    what landed on disk. Fully fail-soft — never raises, always leaves a file.
    """
    mission_id = task.get("mission_id") or ctx.get("mission_id")
    produces = ctx.get("produces") or []
    if not (output_value and mission_id) or not isinstance(produces, list):
        return output_value

    # Mechanical siblings (workflow_advance, git_commit, ...) inherit ctx but
    # do not emit artifacts — mirror the schema gate's _is_producer guard.
    executor = (task.get("executor") or ctx.get("executor") or "")
    agent_type = (task.get("agent_type") or ctx.get("agent_type") or "")
    if executor == "mechanical" or agent_type == "mechanical":
        return output_value

    from coulson.grounding import select_canonical, stamp_front_matter

    schema = ctx.get("artifact_schema") or {}

    def _schema_ok(c: str) -> bool:
        try:
            return bool(validate_artifact_schema(c, schema)[0])
        except Exception:
            return False

    import os as _os
    single = len([e for e in produces if isinstance(e, str)
                  and e.endswith((".md", ".json"))]) == 1
    canonical_out = output_value
    for entry in produces:
        if not (isinstance(entry, str) and entry.endswith((".md", ".json"))):
            continue
        abs_path = entry if _os.path.isabs(entry) else _os.path.join(WORKSPACE_DIR, entry)
        disk = None
        try:
            with open(abs_path, encoding="utf-8") as fh:
                disk = fh.read()
        except OSError:
            disk = None
        chosen = select_canonical([output_value, disk], _schema_ok)
        if not isinstance(chosen, str):
            continue
        kind = "json" if entry.endswith(".json") else "md"
        chosen = stamp_front_matter(chosen, int(mission_id), kind)
        try:
            _os.makedirs(_os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(chosen)
            logger.info(
                f"[Workflow Hook] materialize_produces -> {abs_path} "
                f"({len(chosen)} chars)"
            )
        except OSError as e:
            logger.debug(f"[Workflow Hook] materialize write failed {abs_path}: {e}")
            continue
        if single:
            canonical_out = chosen
    return canonical_out
```

Note: `WORKSPACE_DIR` must be importable at module level in `hooks.py`. If it is currently imported inside functions (it is, e.g. `from ...tools.workspace import WORKSPACE_DIR`), add a module-level import at the top of `hooks.py`: `from src.tools.workspace import WORKSPACE_DIR`. Verify the existing in-function imports still work (they shadow harmlessly) or remove the now-redundant local imports in the blocks you replace in Task 4.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/workflows/test_materialize_produces.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/workflows/engine/hooks.py tests/workflows/test_materialize_produces.py
git commit -m "feat(engine): materialize_produces — deterministic sole writer of produces paths"
```

---

## Task 4: Wire materializer into the hook; remove legacy writes

**Files:**
- Modify: `src/workflows/engine/hooks.py:1500-1557` (replace both write blocks with one call)

- [ ] **Step 1: Replace the two write blocks**

Delete the legacy root-`.md` write block (`# ── Write artifacts to disk in mission directory ──`, ~`1500-1513`) AND the `# ── Persist to the DECLARED produces paths ──` fill-missing block (`~1515-1557`). Replace both with:

```python
    # ── Materialize the declared produces paths (sole writer) ──
    # One deterministic pass: pick the schema-best of {on-disk write,
    # output_value} (fence-unwrapped), stamp mission_id, write the canonical
    # path, and return that content so the schema gate below validates exactly
    # what is on disk. Replaces the legacy root-.md write + the fill-missing
    # block + the coulson auto-persist/canonicalize blocks.
    if output_value and (task.get("mission_id") or ctx.get("mission_id")):
        try:
            output_value = await materialize_produces(ctx, task, result, output_value)
        except Exception as _e:
            logger.debug(f"[Workflow Hook] materialize_produces skipped: {_e}")
```

- [ ] **Step 2: Run the existing engine/hook tests to verify nothing regressed**

Run: `.venv/Scripts/python.exe -m pytest tests/workflows/ -q -p no:cacheprovider`
Expected: PASS (no failures attributable to this change; pre-existing skips/marks fine)

- [ ] **Step 3: Run the materializer tests again (integration through the hook is unchanged)**

Run: `.venv/Scripts/python.exe -m pytest tests/workflows/test_materialize_produces.py -q`
Expected: PASS (4 passed)

- [ ] **Step 4: Commit**

```bash
git add src/workflows/engine/hooks.py
git commit -m "refactor(engine): route produces persistence through materialize_produces"
```

---

## Task 5: Retire `_produces_file_is_stale`

**Files:**
- Modify: `src/workflows/engine/hooks.py:272-300` (delete the function) and any references.

- [ ] **Step 1: Confirm no remaining callers**

Run: `.venv/Scripts/python.exe -m pytest -q --collect-only >/dev/null; grep -rn "_produces_file_is_stale" src tests packages`
Expected: only the definition line (no callers, since Task 4 removed the fill-missing block that used it). If a test references it, delete that test or port it to `select_canonical` (the staleness heuristic is now subsumed by schema-based selection).

- [ ] **Step 2: Delete the function**

Remove the `def _produces_file_is_stale(...)` block (`~272-300`) from `hooks.py`.

- [ ] **Step 3: Verify imports + suite**

Run: `.venv/Scripts/python.exe -c "import src.workflows.engine.hooks" && .venv/Scripts/python.exe -m pytest tests/workflows/ -q -p no:cacheprovider`
Expected: import OK; tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/workflows/engine/hooks.py
git commit -m "refactor(engine): retire _produces_file_is_stale (subsumed by select_canonical)"
```

---

## Task 6: Remove coulson auto-persist + canonicalize blocks

**Files:**
- Modify: `packages/coulson/src/coulson/react.py:820-938` (delete both blocks)

- [ ] **Step 1: Delete the AUTO-PERSIST and CANONICALIZE OVERRIDE blocks**

Remove the `# ── AUTO-PERSIST RECOVERY ──` block and the `# ── CANONICALIZE OVERRIDE ──` block (the full `try/except` spans, ~`820-938`), including the now-unused local imports inside them. Keep the SUB-ITERATION GUARD CHECK block that follows. Do NOT remove the imports of `autopersist_candidate` / `recanonicalize_candidate` at the top of the file yet if other code references them — grep first (next step). `unwrap_fenced_artifact`, `select_canonical`, `stamp_front_matter` remain in `grounding.py` for the engine.

- [ ] **Step 2: Drop now-dead imports if unreferenced**

Run: `grep -n "autopersist_candidate\|recanonicalize_candidate" packages/coulson/src/coulson/react.py`
If the only matches are the import lines, remove those imports. (`grounding.py` keeps the functions; their tests in `test_autopersist_candidate.py` / `test_recanonicalize.py` still pass — they test the pure helpers, not the react wiring.)

- [ ] **Step 3: Verify import + coulson suite**

Run: `.venv/Scripts/python.exe -c "import coulson.react" && .venv/Scripts/python.exe -m pytest packages/coulson/tests/ -q -p no:cacheprovider`
Expected: import OK; PASS (the e2e detect-and-bail test now bounded by the 120s timeout). If a coulson test asserted the synthetic `auto_persist`/`recanonicalized` tool_calls entry, update it to assert the engine materializer behavior instead (or delete if redundant with the engine integration test).

- [ ] **Step 4: Commit**

```bash
git add packages/coulson/src/coulson/react.py
git commit -m "refactor(coulson): remove in-loop auto-persist/canonicalize — engine materializer owns it"
```

---

## Task 7: Regression — mission-81 end-to-end + full suites

**Files:**
- Test: `tests/workflows/test_materialize_produces.py` (add the mission-81 fixture)

- [ ] **Step 1: Add the mission-81 #289715 regression test**

```python
@pytest.mark.asyncio
async def test_mission81_289715_regression(tmp_path, monkeypatch):
    """The real failure: agent wrote a narration report to the produces path
    while the correct doc sat in a ```yaml fence in result. Materializer must
    overwrite disk with the unwrapped artifact AND pass both the loose schema
    and a strict front-matter check."""
    monkeypatch.setattr("src.workflows.engine.hooks.WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_81/.prd/competitive_positioning.md"
    abs_p = tmp_path / rel
    abs_p.parent.mkdir(parents=True, exist_ok=True)
    abs_p.write_text("## Findings\n- listed\n## Recommendations\nready.", encoding="utf-8")  # narration on disk
    ctx = _ctx([rel], {"type": "markdown",
                       "required_sections": ["Landscape", "Value Thesis", "Strengths",
                                             "Our Differentiators", "Switching Costs", "Notes"]})
    task = {"mission_id": 81, "agent_type": "analyst"}
    result_text = (
        "## Analysis\n```yaml\n---\nmission_id: 81\n---\n\n"
        "## Landscape\na\n## Value Thesis\nb\n## Strengths\nc\n"
        "## Our Differentiators\nd\n## Switching Costs\ne\n## Notes\nf\n```\n"
    )
    out = await materialize_produces(ctx, task, {"result": result_text}, result_text)
    disk = abs_p.read_text(encoding="utf-8")
    assert disk.lstrip().startswith("---")            # strict front-matter gate
    assert "## Landscape" in disk and "## Notes" in disk
    assert "## Findings" not in disk                  # narration replaced
    assert out == disk
```

- [ ] **Step 2: Run the regression**

Run: `.venv/Scripts/python.exe -m pytest tests/workflows/test_materialize_produces.py -q`
Expected: PASS (5 passed)

- [ ] **Step 3: Run the full affected suites (single run, bounded by the 120s timeout)**

Run: `.venv/Scripts/python.exe -m pytest packages/coulson/tests/ tests/workflows/ -q -p no:cacheprovider`
Expected: PASS. Record the pass count. Do NOT launch concurrent pytest processes.

- [ ] **Step 4: Run general_beckman (post-hook chain unaffected, confirm)**

Run: `.venv/Scripts/python.exe -m pytest packages/general_beckman/tests/ -q -p no:cacheprovider`
Expected: PASS (pre-existing unrelated failures, if any, noted but not introduced by this work).

- [ ] **Step 5: Commit**

```bash
git add tests/workflows/test_materialize_produces.py
git commit -m "test(engine): mission-81 #289715 narration-to-file regression through materializer"
```

---

## Follow-ups (out of scope — see spec §7)

1. `_schema_version` stamping — needs a version threaded into `artifact_schema` via the expander first.
2. Multifile produces — v1 stamps/validates each declared path in place; a result→multi-artifact splitter is later.
3. Audit non-workflow coulson paths (ad-hoc `/task`, `single_shot`) that relied on the removed auto-persist — if any exists, route it through the engine hook or keep a thin fallback (spec §8 risk b).

## Rollout

Not live until KutAI `/restart` via Telegram. After cutover, re-run i2p phase 1 (or retry #289715); confirm `materialize_produces -> <path>` log lines and the absence of `auto_persist`/`recanonicalized` synthetic writes.
