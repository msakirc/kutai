# SP4b Plan 2 — Workflow-split demo_storyboard + press_kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pull the LLM call out of two mr_roboto verbs (`demo/storyboard`, `press_kit/assemble`) so each becomes a workflow with an `agent:`-typed producer step (prompt in step JSON) feeding a mechanical sink that does NO LLM work — honoring [[no-direct-dispatcher-from-mechanical]].

**Architecture:** Producer step (`agent:reviewer`/`agent:planner`) writes its LLM output to a `produces` file in the mission workspace via the materializer (`src/workflows/engine/hooks.py:272` — note it SKIPS mechanical steps, `:290`). The mechanical sink step reads that file **by path** from the shared mission workspace (no blackboard, no `prior_steps` — `prior_steps` truncates at 1500 chars, `src/core/context_injection.py:124`), then does its deterministic work (normalize/write, zip/version/founder_action). demo splits inside the existing `i2p_v3.json`; press_kit becomes a new `press_kit.json`.

**Tech Stack:** Python 3.10 async, pytest, JSON workflow definitions, `general_beckman` (untouched), `mr_roboto` verbs, `src/workflows/engine`.

**Scope guard:** ZERO changes to `packages/general_beckman/`. incident + crisis are DEFERRED to Plan 3 (need `degrade_on_exhaustion`, which touches Beckman). See spec §1, §3: `docs/superpowers/specs/2026-06-07-cps-sp4b-plan2-design.md`.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `packages/mr_roboto/src/mr_roboto/demo_storyboard.py` | Mechanical sink: read raw storyboard file, normalize scenes, write `demo/storyboard.json`. NO LLM. | Rewrite (gut LLM) |
| `packages/mr_roboto/src/mr_roboto/__init__.py` | Router: repoint `demo/storyboard` branch to the gutted sink; add `/press_kit` launcher branch | Modify (~4149, ~4272) |
| `src/workflows/i2p/i2p_v3.json` | Add producer step `13.demo_storyboard_draft`; repoint sink `13.demo_storyboard` to read the draft | Modify (~9700) |
| `packages/mr_roboto/src/mr_roboto/press_kit_assemble.py` | Mechanical sink: read 4 one-pager files, zip/version/founder_action. NO LLM (`_draft_one_pager_llm` deleted) | Rewrite (gut LLM) |
| `src/workflows/press_kit.json` | NEW workflow: 4 `agent:planner` producers → 1 mechanical `assemble` | Create |
| `src/app/telegram_bot.py` | `/press_kit` command → `WorkflowRunner.start("press_kit", …)` | Modify |
| `packages/mr_roboto/tests/test_demo_storyboard_sink.py` | Pin the gutted demo sink | Create |
| `packages/mr_roboto/tests/test_press_kit_sink.py` | Pin the gutted press_kit sink | Create |

---

## Pre-flight (worktree + substrate)

- [ ] **P1: Create the worktree from Plan 1's tip.** Plan 1 (`worktree-cps-sp4b`) is NOT merged; Plan 2 branches from it.

Run (via EnterWorktree tool): name `cps-sp4b-plan2`, baseRef `worktree-cps-sp4b`.
Then from the worktree use the MAIN venv python: `C:\Users\sakir\Dropbox\Workspaces\kutay\.venv\Scripts\python.exe`.

- [ ] **P2: Confirm import baseline (no LLM, no DB needed).**

Run: `.venv\Scripts\python.exe -c "import mr_roboto; from mr_roboto.demo_storyboard import run; from mr_roboto.press_kit_assemble import run as r2; print('OK')"`
Expected: `OK`

---

## Task 1: Gut the LLM out of demo_storyboard.py → mechanical sink

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/demo_storyboard.py`
- Test: `packages/mr_roboto/tests/test_demo_storyboard_sink.py`

The sink reads the producer's raw output file (`demo/storyboard_raw.json`) from the workspace, normalizes scenes, writes `demo/storyboard.json`. The existing pure helper `_parse_storyboard_response` (lines 93-121) is reused verbatim — it stays. Everything LLM (`_STORYBOARD_SYSTEM`, `_STORYBOARD_USER_TMPL`, `_enqueue_storyboard_llm`, the `spec`/`messages` build, the enqueue call) is deleted.

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_demo_storyboard_sink.py
import json
import os
import pytest
from mr_roboto.demo_storyboard import run


@pytest.mark.asyncio
async def test_sink_reads_raw_file_normalizes_and_writes(tmp_path):
    # Producer wrote this raw file (simulated materializer output).
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    raw = {
        "title": "Demo",
        "scenes": [
            {"title": "Intro", "target_seconds": 5, "viewport_state": "home",
             "narrator_text": "Welcome"},
            {"title": "Silent pan", "target_seconds": 4, "viewport_state": "dash",
             "narrator_text": ""},
        ],
    }
    (demo_dir / "storyboard_raw.json").write_text(json.dumps(raw), encoding="utf-8")

    res = await run(
        mission_id=1,
        workspace_path=str(tmp_path),
        raw_filename="demo/storyboard_raw.json",
    )

    assert res["ok"] is True
    out_path = tmp_path / "demo" / "storyboard.json"
    assert out_path.is_file()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    # Normalization: ids backfilled, visual_only derived from empty narrator_text.
    assert written["scenes"][0]["id"] == "scene_1"
    assert written["scenes"][0]["visual_only"] is False
    assert written["scenes"][1]["visual_only"] is True
    assert res["scene_count"] == 2


@pytest.mark.asyncio
async def test_sink_missing_raw_file_returns_error(tmp_path):
    res = await run(
        mission_id=1,
        workspace_path=str(tmp_path),
        raw_filename="demo/storyboard_raw.json",
    )
    assert res["ok"] is False
    assert "raw" in res["error"].lower()


@pytest.mark.asyncio
async def test_sink_makes_no_llm_call(tmp_path, monkeypatch):
    # Hard guard: importing/calling the sink must not reach beckman.enqueue.
    import mr_roboto.demo_storyboard as mod
    assert not hasattr(mod, "_enqueue_storyboard_llm"), "LLM enqueue must be deleted"
    assert not hasattr(mod, "_STORYBOARD_SYSTEM"), "LLM prompt must be deleted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv\Scripts\python.exe -m pytest packages/mr_roboto/tests/test_demo_storyboard_sink.py -v`
Expected: FAIL — `run()` still has the old signature (`spec_text=…`) and the LLM symbols still exist.

- [ ] **Step 3: Rewrite demo_storyboard.py as the mechanical sink**

Replace lines 51-215 (everything from `# LLM storyboard prompt` through the end of `run`) with the sink below. KEEP the module docstring (lines 1-15, update the "calling an LLM" sentence), the imports (16-26), `_run_subprocess` (33-48), and `_parse_storyboard_response` (93-121).

```python
def _normalize_scenes(storyboard: dict) -> int:
    """Backfill scene ids + visual_only in place. Returns scene count."""
    scenes = storyboard.get("scenes") or []
    for i, scene in enumerate(scenes):
        scene.setdefault("id", f"scene_{i + 1}")
        scene.setdefault(
            "visual_only", not bool(scene.get("narrator_text", "").strip())
        )
    return len(scenes)


async def run(
    *,
    mission_id: int,
    workspace_path: str,
    raw_filename: str = "demo/storyboard_raw.json",
) -> dict[str, Any]:
    """Mechanical sink: read the producer's raw storyboard, normalize, write.

    The LLM draft is produced by the `13.demo_storyboard_draft` workflow step
    (agent:reviewer) and materialized to ``<workspace>/<raw_filename>``. This
    verb makes NO LLM call.

    Returns::
        {"ok": True, "storyboard": {...}, "storyboard_path": str, "scene_count": int}
        {"ok": False, "error": str}
    """
    raw_path = os.path.join(workspace_path, raw_filename)
    try:
        with open(raw_path, encoding="utf-8") as fh:
            content = fh.read()
    except OSError as exc:
        return {"ok": False, "error": f"raw storyboard file missing at {raw_path}: {exc}"}

    storyboard = _parse_storyboard_response(content)
    if storyboard is None:
        return {"ok": False, "error": f"raw storyboard unparseable: {content[:200]!r}"}

    scene_count = _normalize_scenes(storyboard)

    demo_dir = os.path.join(workspace_path, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    storyboard_path = os.path.join(demo_dir, "storyboard.json")
    with open(storyboard_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)

    logger.info(
        "demo_storyboard sink: written",
        mission_id=mission_id,
        scene_count=scene_count,
        storyboard_path=storyboard_path,
    )
    return {
        "ok": True,
        "storyboard": storyboard,
        "storyboard_path": storyboard_path,
        "scene_count": scene_count,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 .venv\Scripts\python.exe -m pytest packages/mr_roboto/tests/test_demo_storyboard_sink.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/demo_storyboard.py packages/mr_roboto/tests/test_demo_storyboard_sink.py
git commit -m "refactor(sp4b): gut LLM from demo_storyboard -> mechanical sink reads raw file"
```

---

## Task 2: Repoint the demo/storyboard router branch to the gutted sink

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py:4149-4163`

The router branch currently passes `spec_text` and `parent_task_id` to the old LLM verb. The new sink takes `workspace_path` + `raw_filename`.

- [ ] **Step 1: Replace the `demo/storyboard` router branch**

Replace lines 4149-4163 with:

```python
    if action == "demo/storyboard":
        # Mechanical sink: normalize the producer's raw storyboard + write
        # demo/storyboard.json. The LLM draft is the 13.demo_storyboard_draft
        # workflow step (agent:reviewer). This branch makes NO LLM call.
        from mr_roboto.demo_storyboard import run as _demo_storyboard
        try:
            res = await _demo_storyboard(
                mission_id=payload.get("mission_id"),
                workspace_path=payload.get("workspace_path") or "",
                raw_filename=payload.get("raw_filename") or "demo/storyboard_raw.json",
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error") or "demo/storyboard failed", result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))
```

- [ ] **Step 2: Verify import + dispatch shape**

Run: `timeout 60 .venv\Scripts\python.exe -m pytest packages/mr_roboto/tests/test_demo_storyboard_sink.py -v`
Expected: PASS (unchanged — sink behavior is covered; this step only re-wires the caller).

- [ ] **Step 3: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/__init__.py
git commit -m "refactor(sp4b): repoint demo/storyboard router branch at mechanical sink"
```

---

## Task 3: Add the producer step + repoint the sink in i2p_v3.json

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` (the `13.demo_storyboard` step at ~9700)

Insert a NEW producer step BEFORE the existing `13.demo_storyboard`. Keep the sink's id `13.demo_storyboard` so the 3 dependents (`13.demo_record`:9737, `13.demo_caption`:9757, `13.demo_accessibility`:9809) need ZERO repointing — they still `depends_on "13.demo_storyboard"`, which is now the mechanical sink that still produces `demo/storyboard.json`.

- [ ] **Step 1: Insert the producer step**

Immediately before the `13.demo_storyboard` object (line 9700), insert:

```json
    {
      "id": "13.demo_storyboard_draft",
      "phase": "phase_13",
      "name": "demo_storyboard_draft",
      "agent": "reviewer",
      "difficulty": "medium",
      "tools_hint": ["read_file", "file_tree"],
      "skip_when": "goal does not include public_demo",
      "depends_on": [
        "13.14.git_commit_green"
      ],
      "instruction": "You are a demo video scriptwriter. Read the product spec/idea in the mission workspace, then generate a JSON storyboard for a product demo video. The storyboard MUST be a single JSON object: {\"title\": string, \"total_target_seconds\": int, \"scenes\": [{\"id\": \"scene_N\", \"title\": string, \"target_seconds\": int (3-120), \"viewport_state\": string (snake_case), \"narrator_text\": string (empty if visual-only), \"visual_only\": boolean}]}. Use 3-6 scenes. Reply ONLY with the JSON object — no preamble, no code fences.",
      "done_when": "A JSON object with a non-empty scenes array is produced.",
      "produces": [
        "demo/storyboard_raw.json"
      ],
      "context": {}
    },
```

- [ ] **Step 2: Repoint the sink step's depends_on + payload**

In the `13.demo_storyboard` object (now following the producer), change:
- `depends_on` from `["13.14.git_commit_green"]` to `["13.demo_storyboard_draft"]`
- `payload` from `{"action": "demo/storyboard"}` to:

```json
      "payload": {
        "action": "demo/storyboard",
        "raw_filename": "demo/storyboard_raw.json"
      },
```

Leave `produces: ["demo/storyboard.json"]`, `post_hooks`, `name`, `id` unchanged. Update its `instruction` to: `"Mechanical sink: normalize the storyboard drafted by 13.demo_storyboard_draft and write demo/storyboard.json with ordered scenes. No LLM."`

- [ ] **Step 3: Validate the JSON parses + the workflow loads**

Run:
```
timeout 60 .venv\Scripts\python.exe -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json', encoding='utf-8')); print('JSON OK')"
timeout 60 .venv\Scripts\python.exe -c "import asyncio; from src.workflows.engine.loader import load_workflow; wf=load_workflow('i2p_v3'); ids=[s.get('id') for s in wf.steps]; assert '13.demo_storyboard_draft' in ids; assert '13.demo_storyboard' in ids; print('LOAD OK', len(ids), 'steps')"
```
Expected: `JSON OK` then `LOAD OK <n> steps`.

- [ ] **Step 4: Assert producer is `agent:`-typed, sink is mechanical (drift guard test)**

```python
# append to packages/mr_roboto/tests/test_demo_storyboard_sink.py
def test_i2p_demo_split_shapes():
    import json
    wf = json.load(open("src/workflows/i2p/i2p_v3.json", encoding="utf-8"))
    steps = {s["id"]: s for s in wf["steps"]}
    draft = steps["13.demo_storyboard_draft"]
    sink = steps["13.demo_storyboard"]
    assert draft["agent"] == "reviewer"
    assert "executor" not in draft  # producer is NOT mechanical
    assert draft["produces"] == ["demo/storyboard_raw.json"]
    assert sink["executor"] == "mechanical"
    assert sink["depends_on"] == ["13.demo_storyboard_draft"]
    # dependents still point at the sink id (no repoint needed)
    assert "13.demo_storyboard" in steps["13.demo_record"]["depends_on"]
```

- [ ] **Step 5: Run + commit**

Run: `timeout 60 .venv\Scripts\python.exe -m pytest packages/mr_roboto/tests/test_demo_storyboard_sink.py -v`
Expected: PASS (4 tests).

```bash
git add src/workflows/i2p/i2p_v3.json packages/mr_roboto/tests/test_demo_storyboard_sink.py
git commit -m "feat(sp4b): split i2p demo_storyboard into agent:reviewer producer + mechanical sink"
```

---

## Task 4: Gut the LLM out of press_kit_assemble.py → mechanical fan-in sink

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/press_kit_assemble.py`
- Test: `packages/mr_roboto/tests/test_press_kit_sink.py`

Delete `_draft_one_pager_llm` (lines 84-118) and `_AUDIENCE_PROMPTS` (lines 53-77 — these prompts move into `press_kit.json` step instructions, Task 5). The sink reads each audience one-pager from a workspace file the producer wrote, instead of calling the LLM. Keep `_get_latest_version`, `_emit_founder_action`, `_audience_extra_section`, `_build_zip`, `AUDIENCE_VARIANTS` verbatim.

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_press_kit_sink.py
import os
import pytest
import mr_roboto.press_kit_assemble as mod
from mr_roboto.press_kit_assemble import run, AUDIENCE_VARIANTS


@pytest.mark.asyncio
async def test_sink_reads_four_onepagers_and_zips(tmp_path, monkeypatch):
    # Producers wrote one_pager_{aud}.md into the workspace.
    src_dir = tmp_path / "press_kit" / "src"
    src_dir.mkdir(parents=True)
    for aud in AUDIENCE_VARIANTS:
        (src_dir / f"one_pager_{aud}.md").write_text(f"# {aud} one-pager\n\nbody", encoding="utf-8")

    monkeypatch.setattr(mod, "_get_latest_version", lambda product_id: _async(0))
    monkeypatch.setattr(mod, "_emit_founder_action", lambda **kw: _async(None))

    res = await run(
        mission_id=1,
        product_id="prod_x",
        workspace_path=str(tmp_path),
        onepager_dir="press_kit/src",
    )
    assert res["ok"] is True
    assert res["version"] == 1
    for aud in AUDIENCE_VARIANTS:
        assert os.path.isfile(res["manifest"]["variants"][aud]["zip_path"])


@pytest.mark.asyncio
async def test_sink_missing_onepager_fails_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_get_latest_version", lambda product_id: _async(0))
    monkeypatch.setattr(mod, "_emit_founder_action", lambda **kw: _async(None))
    res = await run(
        mission_id=1, product_id="prod_x",
        workspace_path=str(tmp_path), onepager_dir="press_kit/src",
    )
    assert res["ok"] is False
    assert "one_pager" in res["error"]


def test_no_llm_symbols():
    assert not hasattr(mod, "_draft_one_pager_llm"), "LLM draft fn must be deleted"
    assert not hasattr(mod, "_AUDIENCE_PROMPTS"), "prompts moved to press_kit.json"


def _async(val):
    async def _c():
        return val
    return _c()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 .venv\Scripts\python.exe -m pytest packages/mr_roboto/tests/test_press_kit_sink.py -v`
Expected: FAIL — `_draft_one_pager_llm`/`_AUDIENCE_PROMPTS` still present; `run` still has old signature.

- [ ] **Step 3: Rewrite the run() signature + the one-pager read**

In `run()`: delete the `spec_text` param, add `onepager_dir: str = "press_kit/src"`. Replace the LLM draft block (lines 220-225, the `one_pager_text = await _draft_one_pager_llm(...)` and its write) with a read from the producer's file. Replace `spec_hash` (line 212, depends on `spec_text`) with a hash over the concatenated one-pagers. Concretely, inside the `for audience in AUDIENCE_VARIANTS:` loop replace the draft+write with:

```python
            # Read the producer's one-pager for this audience (written by the
            # 1.draft_onepager_{aud} workflow step, agent:planner). NO LLM here.
            src_op = os.path.join(workspace_path, onepager_dir, f"one_pager_{audience}.md")
            try:
                with open(src_op, encoding="utf-8") as fh:
                    one_pager_text = fh.read()
            except OSError as exc:
                return {"ok": False, "error": f"one_pager_{audience}.md missing at {src_op}: {exc}"}

            with open(os.path.join(aud_dir, "one_pager.md"), "w", encoding="utf-8") as fh:
                fh.write(one_pager_text)
```

And replace line 212 `spec_hash = hashlib.sha256(spec_text.encode())...` with:

```python
        spec_hash = ""  # set after reading one-pagers
```

then after the audience loop (before building `manifest`), compute:

```python
        spec_hash = hashlib.sha256(
            "".join(
                open(os.path.join(workspace_path, onepager_dir, f"one_pager_{a}.md"),
                     encoding="utf-8").read()
                for a in AUDIENCE_VARIANTS
            ).encode()
        ).hexdigest()[:16]
```

Delete `_draft_one_pager_llm` (84-118) and `_AUDIENCE_PROMPTS` (53-77). Update the module docstring's "one_pager.md — LLM-drafted" line to "one_pager.md — read from producer step (agent:planner)".

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 .venv\Scripts\python.exe -m pytest packages/mr_roboto/tests/test_press_kit_sink.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/press_kit_assemble.py packages/mr_roboto/tests/test_press_kit_sink.py
git commit -m "refactor(sp4b): gut LLM from press_kit_assemble -> mechanical fan-in sink reads 4 one-pagers"
```

---

## Task 5: Create press_kit.json workflow (4 producers → 1 mechanical assemble)

**Files:**
- Create: `src/workflows/press_kit.json`

The 4 producers each `produces` `press_kit/src/one_pager_{aud}.md`; the assemble sink `depends_on` all 4 and reads them via `onepager_dir`. The 4 audience prompts come from the now-deleted `_AUDIENCE_PROMPTS` (Task 4), inlined into each step's `instruction`.

- [ ] **Step 1: Write the workflow file**

```json
{
  "plan_id": "press_kit",
  "version": "1.0",
  "metadata": {
    "description": "Z7 T3C — assemble a versioned 4-audience press kit. 4 agent:planner producers draft one-pagers (investor/journalist/partner/candidate); 1 mechanical assemble step zips/versions and emits a founder_action sign-off.",
    "agents_required": ["planner", "mechanical"],
    "trigger": "/press_kit Telegram command",
    "product_scoping": "product_id required"
  },
  "steps": [
    {
      "id": "1.draft_onepager_investor",
      "name": "draft_onepager_investor",
      "agent": "planner",
      "difficulty": "medium",
      "tools_hint": ["read_file"],
      "depends_on": [],
      "instruction": "Write a concise investor-facing one-pager for the product (spec in the mission workspace). Emphasise: traction metrics, market size, unit economics, team credentials, fundraising context. Omit culture fluff. Output: Markdown prose, 200-400 words, no JSON wrapper, no code fences.",
      "done_when": "A 200-400 word investor one-pager in Markdown is produced.",
      "produces": ["press_kit/src/one_pager_investor.md"],
      "context": {}
    },
    {
      "id": "1.draft_onepager_journalist",
      "name": "draft_onepager_journalist",
      "agent": "planner",
      "difficulty": "medium",
      "tools_hint": ["read_file"],
      "depends_on": [],
      "instruction": "Write a journalist-facing one-pager for the product (spec in the mission workspace). Lead with the news hook (what changed, why now), include 3-5 concrete stats, a founder quote, and a clear narrative arc. Avoid marketing jargon. Output: Markdown prose, 200-400 words, no JSON wrapper, no code fences.",
      "done_when": "A 200-400 word journalist one-pager in Markdown is produced.",
      "produces": ["press_kit/src/one_pager_journalist.md"],
      "context": {}
    },
    {
      "id": "1.draft_onepager_partner",
      "name": "draft_onepager_partner",
      "agent": "planner",
      "difficulty": "medium",
      "tools_hint": ["read_file"],
      "depends_on": [],
      "instruction": "Write a partner/integration-focused one-pager for the product (spec in the mission workspace). Highlight: tech stack, API surface, customer overlap, joint integration opportunity, go-to-market potential. Concrete and actionable. Output: Markdown prose, 200-400 words, no JSON wrapper, no code fences.",
      "done_when": "A 200-400 word partner one-pager in Markdown is produced.",
      "produces": ["press_kit/src/one_pager_partner.md"],
      "context": {}
    },
    {
      "id": "1.draft_onepager_candidate",
      "name": "draft_onepager_candidate",
      "agent": "planner",
      "difficulty": "medium",
      "tools_hint": ["read_file"],
      "depends_on": [],
      "instruction": "Write a candidate-facing (recruiting) one-pager for the product (spec in the mission workspace). Highlight: mission and why it matters, team culture, growth trajectory, open roles, why this is a compelling place to work. Warm but not hyperbolic. Output: Markdown prose, 200-400 words, no JSON wrapper, no code fences.",
      "done_when": "A 200-400 word candidate one-pager in Markdown is produced.",
      "produces": ["press_kit/src/one_pager_candidate.md"],
      "context": {}
    },
    {
      "id": "2.assemble",
      "name": "assemble_press_kit",
      "agent": "mechanical",
      "executor": "mechanical",
      "reversibility": "full",
      "difficulty": "easy",
      "tools_hint": [],
      "depends_on": [
        "1.draft_onepager_investor",
        "1.draft_onepager_journalist",
        "1.draft_onepager_partner",
        "1.draft_onepager_candidate"
      ],
      "input_artifacts": [
        "press_kit/src/one_pager_investor.md",
        "press_kit/src/one_pager_journalist.md",
        "press_kit/src/one_pager_partner.md",
        "press_kit/src/one_pager_candidate.md"
      ],
      "payload": {
        "action": "press_kit/assemble",
        "onepager_dir": "press_kit/src"
      },
      "post_hooks": ["press_kit_freshness"],
      "instruction": "Mechanical sink: read the 4 one-pagers drafted by step 1, copy assets, build per-audience zips, version, and emit the sign-off founder_action. No LLM.",
      "done_when": "4 per-audience zips written under press_kit/v{N}/ and a founder_action emitted.",
      "produces": [],
      "context": {}
    }
  ],
  "trigger_mapping": {
    "manual": {
      "telegram_command": "/press_kit",
      "description": "Founder requests a press kit.",
      "prompts_for": ["product_id"]
    }
  }
}
```

- [ ] **Step 2: Validate JSON + workflow load + fan-in resolves**

Run:
```
timeout 60 .venv\Scripts\python.exe -c "import json; json.load(open('src/workflows/press_kit.json', encoding='utf-8')); print('JSON OK')"
timeout 60 .venv\Scripts\python.exe -c "from src.workflows.engine.loader import load_workflow; wf=load_workflow('press_kit'); ids=[s['id'] for s in wf.steps]; asm=[s for s in wf.steps if s['id']=='2.assemble'][0]; assert len(asm['depends_on'])==4; assert asm['executor']=='mechanical'; print('LOAD OK', ids)"
```
Expected: `JSON OK` then `LOAD OK [...5 ids...]`.

- [ ] **Step 3: Drift-guard test (producers agent-typed, sink mechanical, 4-way fan-in)**

```python
# append to packages/mr_roboto/tests/test_press_kit_sink.py
def test_press_kit_workflow_shapes():
    import json
    wf = json.load(open("src/workflows/press_kit.json", encoding="utf-8"))
    steps = {s["id"]: s for s in wf["steps"]}
    for aud in ("investor", "journalist", "partner", "candidate"):
        p = steps[f"1.draft_onepager_{aud}"]
        assert p["agent"] == "planner"
        assert "executor" not in p
        assert p["produces"] == [f"press_kit/src/one_pager_{aud}.md"]
    asm = steps["2.assemble"]
    assert asm["executor"] == "mechanical"
    assert len(asm["depends_on"]) == 4
```

- [ ] **Step 4: Run + commit**

Run: `timeout 60 .venv\Scripts\python.exe -m pytest packages/mr_roboto/tests/test_press_kit_sink.py -v`
Expected: PASS (4 tests).

```bash
git add src/workflows/press_kit.json packages/mr_roboto/tests/test_press_kit_sink.py
git commit -m "feat(sp4b): press_kit.json — 4 agent:planner producers fan into mechanical assemble"
```

---

## Task 6: /press_kit launches the workflow (thin router)

**Files:**
- Modify: `src/app/telegram_bot.py` (add `cmd_press_kit` + register handler)
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py:4272` (the `press_kit/assemble` branch keeps working as the mechanical sink — verify only)

The launcher follows `_create_mission_with_workflow` (`telegram_bot.py:1619`): call `WorkflowRunner.start("press_kit", …)`. The old router branch at ~4272 still dispatches `press_kit/assemble` to the (now mechanical) verb — leave it; the `2.assemble` step routes through it.

- [ ] **Step 1: Add the command handler**

In `telegram_bot.py`, near other workflow-launching commands, add:

```python
    async def cmd_press_kit(self, update, context):
        """/press_kit <product_id> — launch the press_kit workflow."""
        chat_id = update.effective_chat.id
        args = (context.args or [])
        if not args:
            await self._reply(update, "Usage: /press_kit <product_id>")
            return
        product_id = args[0]
        from src.workflows.engine.runner import WorkflowRunner
        runner = WorkflowRunner()
        mission_id = await runner.start(
            workflow_name="press_kit",
            initial_input={"product_id": product_id},
            title=f"Press kit: {product_id}",
            chat_id=chat_id,
        )
        await self._reply(update, f"Press kit mission #{mission_id} launched for {product_id}.")
```

- [ ] **Step 2: Register the handler**

In `_setup_handlers()`, add alongside the other `CommandHandler` registrations:

```python
        self.app.add_handler(CommandHandler("press_kit", self.cmd_press_kit))
```

- [ ] **Step 3: Verify import (no DB/telegram connect)**

Run: `timeout 60 .venv\Scripts\python.exe -c "import ast; ast.parse(open('src/app/telegram_bot.py', encoding='utf-8').read()); print('PARSE OK')"`
Expected: `PARSE OK`

- [ ] **Step 4: Commit**

```bash
git add src/app/telegram_bot.py
git commit -m "feat(sp4b): /press_kit launches press_kit workflow (thin launcher)"
```

---

## Task 7: Schema-gate acceptance for the new produces paths

**Files:**
- Investigate: `src/workflows/engine` schema gate (240 schemas hard-checked — [[project_schema_gate_shipped_20260605]]) + `_live_artifact_schema`.

New `produces` paths (`demo/storyboard_raw.json`, `press_kit/src/one_pager_*.md`) pass through the deterministic artifact-schema gate. If the gate hard-fails any path lacking a registered schema, the producers will DLQ.

- [ ] **Step 1: Determine whether unregistered produces paths pass the gate**

Run: `timeout 60 .venv\Scripts\python.exe -c "from src.workflows.engine.hooks import validate_artifact_schema; print(validate_artifact_schema('{\"scenes\":[]}', {}))"`
Expected: observe whether empty-schema `{}` returns pass=True. If True → unregistered paths pass, no action needed (skip Step 2).

- [ ] **Step 2: Register lenient schemas only if Step 1 shows hard-fail**

If the gate requires a registered schema, add minimal entries (a non-empty-string check for the `.md` one-pagers; a `scenes`-array check for `storyboard_raw.json`) wherever `_live_artifact_schema` maps produces→schema. Mirror an existing lenient `.md` schema entry. Add a test asserting each new path validates a representative sample.

- [ ] **Step 3: Commit (only if Step 2 ran)**

```bash
git add -A
git commit -m "feat(sp4b): register lenient schemas for demo/press_kit producer artifacts"
```

---

## Task 8: Full sequential suite + finish

- [ ] **Step 1: Run the mr_roboto suite (one invocation, no concurrency)**

Run: `timeout 120 .venv\Scripts\python.exe -m pytest packages/mr_roboto/tests/ -p no:randomly -q`
Expected: all green (no other pytest running — shared `kutai.db` WAL deadlock risk).

- [ ] **Step 2: Run the workflow-engine suite separately**

Run: `timeout 120 .venv\Scripts\python.exe -m pytest tests/workflows/ -q`
Expected: green (or pre-existing failures unrelated to this change — diff against baseline).

- [ ] **Step 3: Confirm no remaining await_inline in the two split verbs**

Run: `rg -n "await_inline|enqueue|_draft_one_pager_llm|_enqueue_storyboard_llm" packages/mr_roboto/src/mr_roboto/demo_storyboard.py packages/mr_roboto/src/mr_roboto/press_kit_assemble.py`
Expected: NO matches (both verbs are LLM-free).

- [ ] **Step 4: Merge to main (`--no-ff`) after review**

Use the `superpowers:finishing-a-development-branch` skill to decide merge vs PR.

---

## Self-Review

**Spec coverage (against `2026-06-07-cps-sp4b-plan2-design.md`):**
- §1 demo split (producer + sink, dependents NOT repointed because sink keeps its id) → Tasks 1-3. ✅
- §1 press_kit new workflow + fan-in → Tasks 4-5. ✅
- §2.1 prompt-in-step-JSON → Task 3 Step 1, Task 5 Step 1 (instructions carry prompts). ✅
- §2.2 fan-in (list depends_on + input_artifacts) → Task 5. ✅
- §2.3 thin launcher → Task 6. ✅
- §6 "no LLM call, no dispatcher/husam" → Tasks 1/4 delete LLM; Task 8 Step 3 verifies. ✅
- §scope "zero Beckman change" → no task touches `packages/general_beckman/`. ✅
- incident/crisis deferred → out of plan by construction. ✅

**Placeholder scan:** Task 7 Step 2 is conditional (gated on Step 1's observed output) — this is a genuine branch, not a placeholder; both arms have concrete actions. No TBD/TODO elsewhere.

**Type consistency:** sink signatures — `demo_storyboard.run(*, mission_id, workspace_path, raw_filename)` used identically in Task 1 test, Task 2 router, Task 3 payload. `press_kit_assemble.run(*, mission_id, product_id, workspace_path, onepager_dir, …)` used identically in Task 4 test + Task 5 payload. `produces` path strings match between producer steps and sink read paths (`demo/storyboard_raw.json`; `press_kit/src/one_pager_{aud}.md`). ✅

**Known residual risk (flagged, not silently dropped):** the producer's spec/idea delivery in i2p — the `agent:reviewer` producer relies on `workspace_snapshot` + `read_file` to find the mission spec (context_injection.py:133 attaches the snapshot for reviewer agents). If a mission's spec isn't in the workspace at phase_13, the producer has no input. The existing `13.demo_storyboard` had the same latent gap (its payload carried no `spec_text`), so this is not a regression — but a first live i2p `public_demo` run should confirm the producer sees the spec. Task 8 cannot cover this offline.
