# Legacy Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse i2p_v3 to the single canonical paraflow path — delete the 16 legacy-only steps, strip every `legacy_pre_*` gate, drop the legacy DB columns, and remove the engine's pre-migration rescue branches.

**Architecture:** The workflow carries a legacy shadow (steps + columns + engine back-compat) that only ever served pre-existing missions (all `legacy_pre_*` columns are `0` in the live DB; no real legacy missions exist). Remove it in 5 independently-mergeable phases, each gated by `tests/workflows/test_i2p_v3_dep_integrity.py` + `tests/test_workflow_loader.py`. Phase 2 also closes a latent gap: downstream frontend (phase 7) was never rewired when paraflow replaced the design-spec suite.

**Tech Stack:** Python 3.10 (`.venv/Scripts/python.exe`), pytest, SQLite (aiosqlite), JSON workflow at `src/workflows/i2p/i2p_v3.json`.

**Spec:** `docs/superpowers/specs/2026-05-25-legacy-removal-design.md`

**Conventions for every JSON edit task:**
- Edits to `i2p_v3.json` are targeted (preserve 2-space indentation + key order). Do NOT round-trip the file through `json.dump` (it reformats all ~12k lines).
- After every JSON edit, validate parse: `Run: .venv/Scripts/python.exe -c "import json,io; json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('OK')"`
- The orchestrator may be live. Run ONLY DB-isolated tests with a timeout. NEVER run a suite that opens the live DB while the orchestrator runs. NEVER kill llama-server / wrapper / orchestrator.

---

## Phase 0 — Prep & harness

### Task 0.1: Branch + baseline + revert the uncommitted debug guard

**Files:**
- Revert: `src/workflows/engine/hooks.py` (the `_skipped` early-return guard added 2026-05-24)
- Delete: `tests/test_skip_when_schema_skip.py`

- [ ] **Step 1: Create the work branch**

```bash
git checkout -b chore/legacy-removal
```

- [ ] **Step 2: Revert the uncommitted post-hook skip guard**

In `src/workflows/engine/hooks.py`, inside `_post_execute_workflow_step_impl`, remove the block added during the 2026-05-24 debug session (the `# ── skip_when short-circuit ──` comment through the `return` that fires on `result.get("_skipped") is True`). Restore the function so it goes straight from `if not is_workflow_step(ctx): return` to `mission_id = ctx.get("mission_id") or task.get("mission_id")`.

- [ ] **Step 3: Delete the guard's test**

```bash
git rm tests/test_skip_when_schema_skip.py
```

- [ ] **Step 4: Confirm the dep-integrity + loader baseline is green BEFORE any change**

Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_i2p_v3_dep_integrity.py tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: PASS (this is the regression gate for every later phase).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(legacy): revert 2026-05-24 skip guard; branch for legacy removal"
```

### Task 0.2: Add the end-state "no-legacy" guard test (initially skipped)

**Files:**
- Create: `tests/workflows/test_no_legacy_residue.py`

- [ ] **Step 1: Write the end-state test, marked skip until the cleanup lands**

```python
"""End-state lock: after legacy removal, the workflow JSON and the missions
schema must carry zero `legacy_pre_` residue. Skipped until the final phase
flips it on (Task 5.3)."""
from __future__ import annotations
import io, sqlite3, re
import pytest

WF = "src/workflows/i2p/i2p_v3.json"

@pytest.mark.skip(reason="enabled in Task 5.3 once removal is complete")
def test_workflow_has_no_legacy_pre_gates():
    text = io.open(WF, encoding="utf-8").read()
    hits = re.findall(r"legacy_pre_\w+", text)
    assert not hits, f"workflow still references: {sorted(set(hits))}"

@pytest.mark.skip(reason="enabled in Task 5.3 once removal is complete")
def test_missions_table_has_no_legacy_columns(tmp_path):
    # Build a fresh schema from db.py and assert no legacy_pre_ columns.
    import asyncio
    from src.infra.db import init_db, get_db
    async def _check():
        await init_db()
        db = await get_db()
        cur = await db.execute("PRAGMA table_info(missions)")
        cols = [r[1] for r in await cur.fetchall()]
        await cur.close()
        return [c for c in cols if c.startswith("legacy_pre_")]
    leftover = asyncio.get_event_loop().run_until_complete(_check())
    assert not leftover, f"missions still has columns: {leftover}"
```

- [ ] **Step 2: Verify it collects + skips cleanly**

Run: `timeout 60 .venv/Scripts/python.exe -m pytest tests/workflows/test_no_legacy_residue.py -q -p no:cacheprovider`
Expected: `2 skipped`.

- [ ] **Step 3: Commit**

```bash
git add tests/workflows/test_no_legacy_residue.py
git commit -m "test(legacy): add skipped end-state no-legacy guard"
```

---

## Phase 1 — Charter chain (delete 0.2/0.4/0.5)

Smallest deletion; proves the dep-rewire harness. Deleted artifacts: `problem_statement`, `open_questions_list`, `clarification_request`. Surviving consumers: `0.6` (input_artifacts) and `0.4a` (depends_on 0.4).

### Task 1.1: Rewire the two consumers BEFORE deleting (keeps the graph valid at every commit)

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` — step `0.6` (idea_brief) and step `0.4a` (compliance_fingerprint_collection)

- [ ] **Step 1: Trim step `0.6` input_artifacts**

In `0.6` (`idea_brief_compilation_and_review`), remove `"problem_statement"` and `"clarification_request"` from `input_artifacts`. Keep `clarification_answers` ONLY if it has a surviving producer — verify:
Run: `.venv/Scripts/python.exe -c "import json,io; s=json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print([x['id'] for x in s if 'clarification_answers' in (x.get('output_artifacts') or [])])"`
If the result is empty, also remove `"clarification_answers"` from `0.6` input_artifacts.

- [ ] **Step 2: Simplify `0.6` instruction to charter-only**

In `0.6`'s `instruction`, delete the sentence: *"When the charter is absent (legacy missions), fall back to `problem_statement` + `clarification_answers` (the old phase-0 micro-artifacts)."* The charter (`product_charter`) is now always present; the brief distills it unconditionally.

- [ ] **Step 3: Rewire `0.4a` depends_on**

`0.4a` (`compliance_fingerprint_collection`, mechanical, inputs `[]`) currently `depends_on: ["0.4"]`. Its real prerequisite is the charter existing. Change `depends_on` to `["0.1"]` (product_charter). (It reads a preflight file + writes the fingerprint; it does not consume `0.4`'s `open_questions_list`.)

- [ ] **Step 4: Validate JSON parses**

Run: `.venv/Scripts/python.exe -c "import json,io; json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor(i2p): rewire 0.6 + 0.4a off the legacy charter chain"
```

### Task 1.2: Delete steps 0.2, 0.4, 0.5

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`

- [ ] **Step 1: Remove the three step objects**

Delete the entire step objects with `"id": "0.2"`, `"id": "0.4"`, and `"id": "0.5"` from the `steps` array (each runs from its opening `{` through its closing `},`). Use the `id` + `name` (`problem_statement_extraction`, `scope_ambiguity_detection`, `human_clarification_request`) to locate them.

- [ ] **Step 2: Validate JSON parses**

Run: `.venv/Scripts/python.exe -c "import json,io; json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run the dep-integrity + loader gate**

Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_i2p_v3_dep_integrity.py tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: PASS. If it reports a dangling `depends_on`/`input_artifacts` referencing 0.2/0.4/0.5 or their artifacts, fix that reference (return to Task 1.1) before committing.

- [ ] **Step 4: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor(i2p): delete legacy charter-chain steps 0.2/0.4/0.5"
```

### Task 1.3: Strip `legacy_pre_charter` skip_when from canonical steps

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`

- [ ] **Step 1: List the remaining legacy_pre_charter sites**

Run: `.venv/Scripts/python.exe -c "import json,io; s=json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print([(x['id'],x.get('skip_when')) for x in s if isinstance(x.get('skip_when'),str) and 'legacy_pre_charter' in x['skip_when']])"`
Expected: a list of `== '1'` canonical steps (e.g. `0.0z`, `0.0a.draft`, `0.0a`, `0.1`, `0.1.verify`, `0.0z.verify`, `0.0z.confirm`, `0.0c`, `0.0c.verify`, `0.0c.request`).

- [ ] **Step 2: Remove the `skip_when` key from each listed step**

For every step in that list, delete its `"skip_when": "mission.legacy_pre_charter == '1'",` line. (These are `== '1'` = canonical → unconditional. There must be no `!= '1'` left — those were the deleted steps.)

- [ ] **Step 3: Validate parse + confirm zero legacy_pre_charter remains**

Run: `.venv/Scripts/python.exe -c "import json,io; t=io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8').read(); import json as j; j.loads(t); print('charter refs:', t.count('legacy_pre_charter'))"`
Expected: `charter refs: 0`

- [ ] **Step 4: Run the gate**

Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_i2p_v3_dep_integrity.py tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor(i2p): make charter steps unconditional (drop legacy_pre_charter gate)"
```

---

## Phase 2 — per_screen_plans design suite (delete 5.1–5.11b) + frontend reconnect

**Latent gap warning:** steps `7.9, 7.10, 7.11, 7.12, 7.17, 13.10, 11.2, 11.3` and `6.1` consume the design-spec artifacts produced by 5.1–5.11b. Those steps already SKIP for new missions, so these consumers already run starved of those inputs — this phase reconnects them to the paraflow outputs (`per_screen_plans_chunk_a/b`, `html_prototypes_chunk_a/b`, `user_flow` from `5.0c`, `design_tokens` from `5.0a`). The dep-integrity test catches dangling references; SEMANTIC correctness must be confirmed by a real mission run after merge (see Task 2.5).

Surviving design producers to rewire toward: `5.0a` → `design_tokens`; `5.0c` → `user_flow`; `5.20a/5.20b` → `per_screen_plans_chunk_a/b`; `5.30a/5.30b` → `html_prototypes_chunk_a/b`.

### Task 2.1: Reconnect phase-7 frontend implementers to paraflow artifacts

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` — steps `7.9`, `7.10`, `7.11`, `7.12`

- [ ] **Step 1: Rewire `7.9` (design_system_implementation)**

Current inputs `["design_tokens","layout_system_spec","navigation_design"]` → set to `["design_tokens","per_screen_plans_chunk_a","per_screen_plans_chunk_b"]` (`design_tokens` survives via 5.0a; per-screen plans carry the layout/navigation/shell intent). In its `instruction`, replace references to "layout_system_spec" / "navigation_design" with "the layout, app-shell and navigation described across the per-screen plans". Add `depends_on` edges `"5.20a"`,`"5.20b"` (keep existing 7.5/7.5m).

- [ ] **Step 2: Rewire `7.10` (primitive_components)**

Inputs `["component_spec_primitives","design_tokens_code"]` → `["html_prototypes_chunk_a","design_tokens_code"]`. In `instruction`, replace "Implement all primitive components from component_spec_primitives: …" with "Extract and implement the primitive components used across the HTML prototypes (html_prototypes_chunk_a) — Button, Input, Textarea, Select, Checkbox, Radio, Toggle, Label, Badge, Avatar, Icon, Divider, Spinner, Skeleton, Link, Tag — each with all variants, sizes, states, and accessibility." Add `depends_on` `"5.30a"`.

- [ ] **Step 3: Rewire `7.11` (composite_components)**

Inputs `["component_spec_composite","design_tokens_code","interaction_spec"]` → `["html_prototypes_chunk_a","html_prototypes_chunk_b","design_tokens_code"]`. In `instruction`, replace "Implement all composite components from component_spec_composite: …" with "Implement the composite components observed in the HTML prototypes (html_prototypes_chunk_a/b): Card, Modal, Toast, Alert, Dropdown, Tooltip, Popover, Tabs, Accordion, Table, Pagination, Breadcrumb, EmptyState, Sidebar, Navbar, Footer — composed from primitive components, with interaction behavior as shown in the prototypes." Add `depends_on` `"5.30b"`.

- [ ] **Step 4: Rewire `7.12` (component_playground)**

Inputs `["component_spec_primitives","component_spec_composite"]` → `["primitive_components_code","composite_components_code"]` (it already `depends_on` 7.10/7.11 which produce those). In `instruction`, replace "with stories for every primitive and composite component" — keep, but source from the implemented code, not the dead specs.

- [ ] **Step 5: Validate parse + gate**

Run: `.venv/Scripts/python.exe -c "import json,io; json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('OK')"`
Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_i2p_v3_dep_integrity.py tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: `OK` then PASS.

- [ ] **Step 6: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor(i2p): reconnect phase-7 frontend to paraflow prototype artifacts"
```

### Task 2.2: Rewire the remaining design-spec consumers (6.1, 7.17, 11.2, 11.3, 13.10)

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` — steps `6.1`, `7.17`, `11.2`, `11.3`, `13.10`

- [ ] **Step 1: Rewire `6.1` (epic_definition)**

Remove `"design_handoff_summary"` from `input_artifacts`; change `depends_on` `"5.11b"` → `"5.30b"` (the paraflow design phase end). In `instruction`, drop any mention of `design_handoff`.

- [ ] **Step 2: Rewire `7.17` (implementation_context_package)**

Replace `"design_handoff"` in `input_artifacts` with `"html_prototypes_chunk_a"`. In `instruction`, replace "design_handoff" with "the HTML prototypes".

- [ ] **Step 3: Rewire `11.2` + `11.3` (docs / in-app copy)**

In both, replace `"user_flow_onboarding"` in `input_artifacts` with `"user_flow"` (produced by `5.0c`). Verify `user_flow` has a producer:
Run: `.venv/Scripts/python.exe -c "import json,io; s=json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print([x['id'] for x in s if 'user_flow' in (x.get('output_artifacts') or [])])"`
Expected: includes `5.0c`. In each instruction, replace "user_flow_onboarding" with "the onboarding track of user_flow".

- [ ] **Step 4: Rewire `13.10` (seo_implementation)**

Replace `"information_architecture"` in `input_artifacts` with `"per_screen_plans_chunk_a"` (sitemap/IA intent now lives in the per-screen plans). In `instruction`, replace "information_architecture" with "the screen inventory and navigation in the per-screen plans".

- [ ] **Step 5: Validate parse + gate**

Run: `.venv/Scripts/python.exe -c "import json,io; json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('OK')"`
Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_i2p_v3_dep_integrity.py tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: `OK` then PASS.

- [ ] **Step 6: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor(i2p): rewire 6.1/7.17/11.2/11.3/13.10 to paraflow design outputs"
```

### Task 2.3: Delete steps 5.1–5.11b

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`

- [ ] **Step 1: Confirm no surviving consumer references the design-spec artifacts**

Run: `.venv/Scripts/python.exe -c "
import json,io
s=json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8'))
DEL={'5.1','5.2','5.3','5.4a','5.4b','5.5','5.6','5.7','5.8','5.9','5.10','5.11a','5.11b'}
arts=set()
for d in DEL:
  for x in s:
    if x.get('id')==d:
      arts|=set(x.get('output_artifacts') or [])
bad=[]
for x in s:
  if x.get('id') in DEL: continue
  for a in (x.get('input_artifacts') or []):
    if a in arts and a!='design_tokens': bad.append((x['id'],a))
  for dep in (x.get('depends_on') or []):
    if dep in DEL: bad.append((x['id'],'dep:'+dep))
print('REMAINING REFS (must be empty):', bad)
"`
Expected: `REMAINING REFS (must be empty): []`. If not empty, finish Tasks 2.1/2.2 for the listed step before deleting.

- [ ] **Step 2: Delete the 13 step objects**

Delete the step objects with ids `5.1, 5.2, 5.3, 5.4a, 5.4b, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11a, 5.11b` from the `steps` array.

- [ ] **Step 3: Validate parse + gate**

Run: `.venv/Scripts/python.exe -c "import json,io; json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('OK')"`
Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_i2p_v3_dep_integrity.py tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: `OK` then PASS.

- [ ] **Step 4: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor(i2p): delete legacy design-spec suite 5.1-5.11b"
```

### Task 2.4: Strip `legacy_pre_per_screen_plans` skip_when from the paraflow steps

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` — `5.20a, 5.20a.verify_shape, 5.20b, 5.20b.verify_shape, 5.20.verify_consistency, 5.30a, 5.30a.verify_shape, 5.30b, 5.30b.verify_shape`

- [ ] **Step 1: Remove the `skip_when` key (all `== '1'`) from each of the 9 paraflow steps**

For each, delete its `"skip_when": "mission.legacy_pre_per_screen_plans == '1'",` line. (No `== '0'` remain — those were the deleted 5.1-5.11b.)

- [ ] **Step 2: Confirm zero per_screen_plans refs + parse**

Run: `.venv/Scripts/python.exe -c "import json,io; t=io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8').read(); import json as j; j.loads(t); print('per_screen refs:', t.count('legacy_pre_per_screen_plans'))"`
Expected: `per_screen refs: 0`

- [ ] **Step 3: Run the gate**

Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_i2p_v3_dep_integrity.py tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor(i2p): make paraflow per-screen/HTML steps unconditional"
```

### Task 2.5: Note for post-merge validation (no code — record only)

- [ ] **Step 1:** Append a line to `docs/handoff/2026-05-25-legacy-removal-progress.md` (create if absent): *"Phase 2 rewired phase-7/11/13 frontend+docs consumers from the deleted design-spec artifacts to paraflow outputs. Structural gate (dep-integrity) passes; SEMANTIC correctness (do the implementers produce good components from html_prototypes?) is UNVALIDATED — must be confirmed by a real i2p mission reaching phase 7. Flag to founder."*

- [ ] **Step 2: Commit**

```bash
git add docs/handoff/2026-05-25-legacy-removal-progress.md
git commit -m "docs(legacy): record phase-2 semantic-validation debt"
```

---

## Phase 3 — Bulk strip remaining `legacy_pre_* == '1'` gates

All remaining `legacy_pre_*` gates are `== '1'` canonical steps (adr, falsification, compliance, non_goals, prior_art, competitive_positioning, premortem, spec_alive, github_init, inheritance, idea_dedup, html_oids, preview_url, design_tokens, user_flow). Each becomes unconditional. `skip_real_vendor_checks` is NOT legacy — preserve it.

### Task 3.1: Remove every remaining `legacy_pre_*` skip_when line

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`

- [ ] **Step 1: Inventory the remaining sites (sanity check before editing)**

Run: `.venv/Scripts/python.exe -c "
import json,io
s=json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8'))
sites=[(x['id'],x['skip_when']) for x in s if isinstance(x.get('skip_when'),str) and 'legacy_pre_' in x['skip_when']]
print('count:',len(sites))
bad=[t for t in sites if \"!= '1'\" in t[1] or \"== '0'\" in t[1]]
print('NON-canonical (must be empty):',bad)
for t in sites: print(' ',t[0],t[1])
"`
Expected: `NON-canonical (must be empty): []` (all surviving legacy gates are `== '1'`). If any non-canonical remain, a deletion was missed — STOP and resolve.

- [ ] **Step 2: Delete each `"skip_when": "mission.legacy_pre_<...> == '1'",` line**

Remove every line whose `skip_when` value contains `legacy_pre_`. Do NOT touch the line containing `skip_real_vendor_checks`.

- [ ] **Step 3: Confirm only the genuine flag remains + parse**

Run: `.venv/Scripts/python.exe -c "import json,io; t=io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8').read(); import json as j; j.loads(t); print('legacy_pre_ refs:', t.count('legacy_pre_'), '| skip_real_vendor_checks refs:', t.count('skip_real_vendor_checks'))"`
Expected: `legacy_pre_ refs: 0 | skip_real_vendor_checks refs: 2` (the two vendor-check gates survive).

- [ ] **Step 4: Run the gate**

Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/test_i2p_v3_dep_integrity.py tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "refactor(i2p): strip all remaining legacy_pre_* gates (keep skip_real_vendor_checks)"
```

### Task 3.2: Remove the `_z1_tier3_patched` diagnostic marker if present

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`

- [ ] **Step 1: Check + remove**

Run: `.venv/Scripts/python.exe -c "import io; t=io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8').read(); print('marker present:', '_z1_tier3_patched' in t)"`
If present, remove the top-level `"_z1_tier3_patched": true,` key.

- [ ] **Step 2: Parse + commit (skip if marker absent)**

Run: `.venv/Scripts/python.exe -c "import json,io; json.load(io.open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); print('OK')"`
```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "chore(i2p): drop stale _z1_tier3_patched marker"
```

---

## Phase 4 — DB columns + migrations

**Requires the orchestrator stopped + a DB backup.** This is the only non-trivial-rollback phase.

### Task 4.1: Remove the legacy column CREATE + migration blocks in db.py

**Files:**
- Modify: `src/infra/db.py`
- Delete: `scripts/z1_migrate_legacy_charter_flag.py`

- [ ] **Step 1: Locate every legacy column reference in db.py**

Run: `.venv/Scripts/python.exe -c "import io; [print(i+1, l.rstrip()) for i,l in enumerate(io.open('src/infra/db.py',encoding='utf-8')) if 'legacy_pre_' in l or 'interview_skip_reason' in l]"`
This prints every CREATE-column line, every `ALTER TABLE missions ADD COLUMN legacy_pre_*` migration, and every `UPDATE missions SET legacy_pre_* = 1` backfill, with line numbers.

- [ ] **Step 2: Remove the CREATE-table legacy columns**

In the `CREATE TABLE missions (...)` block, delete every `legacy_pre_* INTEGER DEFAULT 0,` line and the `legacy_pre_p7`, `legacy_pre_critic_gate`, and `interview_skip_reason` lines (confirm `interview_skip_reason` truly has no reader first: `Run: grep -rn interview_skip_reason src packages` → expect only db.py).

- [ ] **Step 3: Remove the ALTER/backfill migration blocks**

Delete each idempotent migration block of the form `try: ALTER TABLE missions ADD COLUMN legacy_pre_* … ; UPDATE missions SET legacy_pre_* = 1 …`. These are the Z1 T1–T6 migrations identified in Step 1.

- [ ] **Step 4: Delete the standalone migration script**

```bash
git rm scripts/z1_migrate_legacy_charter_flag.py
```
Also grep for tier-patch scripts that set these flags: `grep -rln "legacy_pre_" scripts` — for each hit, remove the legacy-flag lines (or delete the script if it is wholly a legacy-flag patch).

- [ ] **Step 5: Confirm db.py no longer references legacy columns**

Run: `.venv/Scripts/python.exe -c "import io; t=io.open('src/infra/db.py',encoding='utf-8').read(); print('legacy_pre_ in db.py:', t.count('legacy_pre_'))"`
Expected: `legacy_pre_ in db.py: 0`

- [ ] **Step 6: Verify db.py imports + a fresh schema builds clean**

Run: `timeout 60 .venv/Scripts/python.exe -c "import asyncio, os, tempfile; os.environ['DB_PATH']=tempfile.mktemp(suffix='.db'); from src.infra.db import init_db, get_db; asyncio.run(init_db()); print('fresh schema OK')"`
Expected: `fresh schema OK` (builds the new missions table with no legacy columns; does NOT touch the live DB).

- [ ] **Step 7: Commit**

```bash
git add src/infra/db.py scripts/
git commit -m "refactor(db): remove legacy_pre_* columns + Z1 backfill migrations from schema"
```

### Task 4.2: Add a forward migration that drops the columns from existing DBs

**Files:**
- Modify: `src/infra/db.py` (the migration section that runs on `init_db`)

- [ ] **Step 1: Add the drop-columns migration**

In the migrations section of `init_db` (after the `CREATE TABLE` calls, alongside the other idempotent `ALTER` migrations), add:

```python
# Legacy removal (2026-05-25): drop the legacy_pre_* gate columns + dead
# diagnostics. SQLite >= 3.35 supports DROP COLUMN; guard per-column on the
# live PRAGMA so this is idempotent and safe to re-run.
_LEGACY_DROP_COLS = [
    "legacy_pre_charter", "legacy_pre_adr", "legacy_pre_falsification",
    "legacy_pre_non_goals", "legacy_pre_competitive_positioning",
    "legacy_pre_per_screen_plans", "legacy_pre_html_oids",
    "legacy_pre_preview_url", "legacy_pre_premortem", "legacy_pre_spec_alive",
    "legacy_pre_compliance", "legacy_pre_critic_gate", "legacy_pre_github_init",
    "legacy_pre_idea_dedup", "legacy_pre_inheritance", "legacy_pre_prior_art",
    "legacy_pre_design_tokens", "legacy_pre_user_flow", "legacy_pre_p7",
    "interview_skip_reason",
]
try:
    _cur = await db.execute("PRAGMA table_info(missions)")
    _existing = {r[1] for r in await _cur.fetchall()}
    await _cur.close()
    for _c in _LEGACY_DROP_COLS:
        if _c in _existing:
            await db.execute(f"ALTER TABLE missions DROP COLUMN {_c}")
    await db.commit()
    logger.info("Legacy-removal migration: dropped legacy_pre_* columns")
except Exception as _e:
    logger.warning(f"Legacy-removal column drop skipped: {_e}")
```

(If the runtime SQLite is < 3.35, `DROP COLUMN` raises; the `except` logs and skips — the columns then sit inert and harmless, dropped on the next SQLite upgrade. Verify version: `Run: .venv/Scripts/python.exe -c "import sqlite3; print(sqlite3.sqlite_version)"` — expect >= 3.35.)

- [ ] **Step 2: Test the migration on a COPY of the live DB (never the live file)**

Run:
```bash
.venv/Scripts/python.exe -c "
import shutil, sqlite3, asyncio, os, tempfile
src='C:/Users/sakir/ai/kutai/kutai.db'; dst=tempfile.mktemp(suffix='.db')
shutil.copy(src, dst); os.environ['DB_PATH']=dst
from src.infra.db import init_db, get_db
async def go():
    await init_db()
    db=await get_db()
    cur=await db.execute('PRAGMA table_info(missions)')
    cols=[r[1] for r in await cur.fetchall()]; await cur.close()
    print('legacy cols after migration:', [c for c in cols if c.startswith('legacy_pre_')])
asyncio.run(go())
"
```
Expected: `legacy cols after migration: []`

- [ ] **Step 3: Commit**

```bash
git add src/infra/db.py
git commit -m "feat(db): forward migration dropping legacy_pre_* columns (idempotent, guarded)"
```

---

## Phase 5 — Engine rescue removal (scope C)

Remove pure pre-migration back-compat. KEEP the general `skip_when_expr` artifact eval + the beckman sweep retro-skip backstop (shopping_v2 uses them).

### Task 5.1: Remove the legacy branches in should_skip + expander

**Files:**
- Modify: `src/workflows/engine/hooks.py` (`should_skip_workflow_step`)
- Modify: `src/workflows/engine/expander.py` (executor legacy-shape rescue)

- [ ] **Step 1: Remove the `mission.<column>` skip branch in should_skip**

In `should_skip_workflow_step`, delete the block that handles `artifact_name == "mission"` (the `SELECT {col} FROM missions` branch). With no `mission.*` skip_when left in any workflow, it is dead. The artifact-based path (`store.retrieve(mission_id, artifact_name)`) stays — shopping_v2 needs it.

- [ ] **Step 2: Remove the legacy JSON-lookup fallback in should_skip**

Delete the block that, when `skip_when_expr` is absent from ctx, looks the step up in the live workflow JSON by `workflow_step_id` (the "Legacy rescue: tasks expanded before the skip_when_expr context field was added" block). Current expansions always populate `skip_when_expr`; only pre-fix DB rows relied on this, and no legacy missions exist.

- [ ] **Step 3: Remove the expander executor legacy-shape rescue**

In `expander.py`, delete the block translating old `{"executor": "<action>", ...}` context into the canonical `_mechanical_context`/payload shape (the "legacy shape rescue" comment block).

- [ ] **Step 4: Verify imports + targeted should_skip tests (DB-isolated)**

Run: `timeout 90 .venv/Scripts/python.exe -m pytest tests/test_clarify_action_schema_skip.py -q -p no:cacheprovider`
Expected: PASS (this suite mocks the DB; if a removed branch was load-bearing for it, fix before commit).
Run: `timeout 30 .venv/Scripts/python.exe -c "import src.workflows.engine.hooks, src.workflows.engine.expander; print('import OK')"`
Expected: `import OK`

- [ ] **Step 5: Commit**

```bash
git add src/workflows/engine/hooks.py src/workflows/engine/expander.py
git commit -m "refactor(engine): remove mission.<col> skip + pre-migration rescue branches"
```

### Task 5.2: Remove the orchestrator executor legacy-shape rescue

**Files:**
- Modify: `src/core/orchestrator.py`

- [ ] **Step 1: Delete the two legacy-shape rescue tiers**

In the dispatch path, delete the block(s) that (a) promote `ctx.executor` to `payload.action` for old `{"executor": "<action>"}` rows and (b) look the step up in the workflow JSON to recover a clobbered action name (the "Legacy-shape rescue" / "Deeper rescue" comment blocks). Keep the canonical `is_mech` dispatch (`runner == "mechanical"` / `payload`) intact.

- [ ] **Step 2: Verify import + orchestrator constructs**

Run: `timeout 30 .venv/Scripts/python.exe -c "import src.core.orchestrator; print('import OK')"`
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "refactor(orchestrator): remove pre-2026-04-24 executor legacy-shape rescue"
```

### Task 5.3: Enable the end-state no-legacy guard test

**Files:**
- Modify: `tests/workflows/test_no_legacy_residue.py`

- [ ] **Step 1: Remove both `@pytest.mark.skip` decorators**

- [ ] **Step 2: Run it (orchestrator must be stopped — `test_missions_table_has_no_legacy_columns` opens the configured DB)**

Run: `timeout 90 .venv/Scripts/python.exe -m pytest tests/workflows/test_no_legacy_residue.py -q -p no:cacheprovider`
Expected: `2 passed`. (Run after the founder `/stop`s the orchestrator + the Phase-4 migration has applied to the live DB. If only the schema test fails because the live DB hasn't migrated yet, restart loads init_db which applies the drop.)

- [ ] **Step 3: Commit**

```bash
git add tests/workflows/test_no_legacy_residue.py
git commit -m "test(legacy): enable end-state no-legacy guard (workflow + schema clean)"
```

---

## Phase 6 — Final verification & docs

### Task 6.1: Full targeted regression + update test files asserting legacy behavior

**Files:**
- Modify/Delete: the legacy-asserting tests — `tests/i2p/test_t5a_steps.py`, `test_t5b_steps.py`, `test_t6a_steps.py`, `test_t6b_steps.py`, `test_t6c_steps.py`, `tests/i2p/test_adr_shape.py`, `test_falsification.py`, `test_non_goals.py`, `tests/i2p/reviewer_regression/test_reviewer_regression.py`, `tests/test_clarify_action_schema_skip.py`, `tests/integration/test_workflow_pipeline.py`

- [ ] **Step 1: Find every test asserting legacy_pre_ or the deleted steps**

Run: `grep -rln "legacy_pre_\|'0.2'\|\"0.2\"\|'0.5'\|'5.11b'" tests`
For each hit: if the test asserts a `legacy_pre_*` gate exists or that step 0.2/0.4/0.5/5.1-5.11b exists, update it to the new reality (gate gone / step deleted) or delete the test if it only covered legacy behavior.

- [ ] **Step 2: Run each touched i2p test file (DB-isolated ones) with a timeout**

Run: `timeout 150 .venv/Scripts/python.exe -m pytest tests/i2p -q -p no:cacheprovider`
Expected: PASS (after updates). Investigate any failure individually; do not bulk-skip.

- [ ] **Step 3: Run the workflow gate once more**

Run: `timeout 120 .venv/Scripts/python.exe -m pytest tests/workflows/ tests/test_workflow_loader.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test(legacy): update/remove tests asserting removed legacy steps + gates"
```

### Task 6.2: Final progress doc + founder handoff

**Files:**
- Modify: `docs/handoff/2026-05-25-legacy-removal-progress.md`

- [ ] **Step 1: Record the end state**

Document: steps deleted (16), gates stripped (count from Task 3.1 Step 1), columns dropped (20), engine branches removed, and the OPEN item from Task 2.5 (phase-7 paraflow-reconnect semantic validation needs a real mission run). List the founder actions: `/stop` before the live DB migration applies; `/restart` to apply it; run a fresh i2p mission to validate phases 0 + 5 + 7 end-to-end.

- [ ] **Step 2: Commit**

```bash
git add docs/handoff/2026-05-25-legacy-removal-progress.md
git commit -m "docs(legacy): final removal progress + founder validation steps"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** §5 deletions → Phases 1.2/2.3; §6 skip-strip → 1.3/2.4/3.1; §7 rewiring → 1.1/2.1/2.2; §8 DB → Phase 4; §9 engine rescues → Phase 5; §10 guard revert → Task 0.1; §12 tests → Task 0.2 + 6.1. All covered.
- **Latent gap:** the phase-7 frontend starvation (consumers of deleted design-spec artifacts) is surfaced in Phase 2 intro + Task 2.5 as semantic-validation debt — structural gate is automated, semantic gate is a real mission run.
- **No placeholders:** every consumer rewire names exact step + exact input swap; bulk strips are line-deletes validated by parse + ref-count + dep-integrity.
