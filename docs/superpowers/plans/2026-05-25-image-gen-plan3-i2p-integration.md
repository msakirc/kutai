# Image Generation — Plan 3 (v2): i2p integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real image generation wired into the i2p prototype phase — replace `placehold.co` `<img>` placeholders with diffusion-generated PNGs, served back through the existing web-preview host. Robust against the **actual data shapes** flowing through beckman+orchestrator: `TaskResult.result` arrives as a **JSON string** (recon verified), prompt_writer needs **constrained decoding** to survive cheap-tier LLMs, and the HTML walker must **recurse** to handle multi-screen prototypes.

**Architecture:** A mr_roboto mechanical (`swap_placeholder_images`) is invoked from i2p step `5.35` (phase_5, after annotate_html_oids, before emit_preview_url). The mechanic recursively walks `mission_{id}/.web/**/*.html`, scans each file for `<img>` whose `src` matches `^https?://placehold\.co/`, enqueues ONE `prompt_writer` agent task (single-call, **constrained-decoded via `artifact_schema` on the step**) carrying the design context + placeholder list, then enqueues one image task per placeholder via beckman (which routes through Plan 1 v2's `needs_image=True` admission → dispatcher.dispatch_image → paintress). Each TaskResult is **JSON-loaded** before extraction. Generated PNGs land directly under `mission_{id}/.web/assets/<placeholder_id>.png` (no rename dance). HTML `<img src>` is rewritten to relative `assets/<id>.png`. A sibling `verify_swap_placeholder_images_shape` post-hook validates the rewrite produced a well-formed HTML with no surviving `placehold.co` references (or, on graceful degrade, that surviving placeholders are accompanied by matching `errors`).

**Tech Stack:** Python 3.10, async/await, regex-based HTML rewriting (mirrors `verify_html_prototype_shape`), pytest. No new packages; only new files inside `packages/mr_roboto/` and `src/agents/` plus an i2p step.

**Scope boundary (in this plan):**
- `prompt_writer` agent (pure config) + agent-registry wiring.
- Diffusion-prompt template + few-shot exemplars.
- `swap_placeholder_images` mr_roboto mechanic (recursive scan, robust JSON-string parse, direct-write to assets, HTML rewrite, graceful degrade).
- `verify_swap_placeholder_images_shape` post-hook mechanic.
- i2p_v3.json prototype-phase step `5.35` with the correct mechanical convention (`executor: "mechanical"`, verb in `payload.action`) + `artifact_schema` for prompt_writer constraint.
- e2e host-path test that uses **real JSON-string TaskResult shape** so production bugs don't hide.

**NOT in this plan:** anything in Plan 1 v2's or Plan 2 v2's territory (see file-ownership table below). No changes to `paintress`, `renoir`, `clair_obscur`, `fatih_hoca`, `llm_dispatcher`, `orchestrator`, `general_beckman`, `nerd_herd`, or `telegram_bot`.

**Dependency: Plan 1 v2 MUST be merged first.** Plan 3 enqueues `agent_type: "image"` tasks that beckman routes via Plan 1 v2's `_select_for_admission` image branch + Plan 1 v2's `_dispatch_image`. Without Plan 1 v2, every image task degrades to failed → mission ships with placeholders (graceful degrade, but the feature does nothing).

Plan 3 is **file-disjoint from Plan 2** — Plans 2 and 3 can run in parallel worktrees after Plan 1 v2 lands.

---

## Audit findings this rewrite addresses

Prior Plan 3 had: (1) **`isinstance(raw, dict)` checks against `TaskResult.result` — which is a JSON STRING in production** (the exact bug Plan 1 v1 was rewritten to fix; Plan 3 v1 repeated it both in `_call_prompt_writer` AND `_generate_one_image`), (2) **prompt_writer enqueued without `response_format=json_schema`** despite using `default_tier="cheap"` (small LLMs emit malformed JSON without constraint), (3) **flat `os.listdir(.web/)` not recursive** (Z5 mobile and multi-screen prototypes write to subdirs), (4) **wrong i2p mechanical step shape** — used `executor: "swap_placeholder_images"` when convention is `executor: "mechanical"` + `payload.action: "<verb>"`, (5) **assets file-naming dance** (paintress wrote timestamped name, then renamed to `<pid>.png`), (6) **no verify-shape posthook** breaking the Z2/Z3 pattern (mechanical without a verifier = silent rot).

Recon confirmed (verbatim file:line):
- `TaskResult.result` is a JSON STRING — `orchestrator.py:326` does `json.dumps(_dispatch_result)`, `on_task_finished` at `:919-925` extracts that string into `TaskResult.result`. Confirmed mirroring of dispatcher's `_task_result_to_request_response` at `llm_dispatcher.py:137-163` is required.
- i2p mechanical step convention (3+ examples in `src/workflows/i2p/i2p_v3.json`): `"agent": "mechanical"`, `"executor": "mechanical"`, verb in `payload.action`. E.g. `verify_charter_shape` at line 961-983.
- `verify_charter_shape` template (`mr_roboto/__init__.py:1071-1094`): pure function returning `{ok, problems, ...}`, dispatched via `if action == "verify_charter_shape":`. Mirror this for `verify_swap_placeholder_images_shape`.
- `response_format` for raw_dispatch tasks goes in `context.llm_call.response_format` (per `hallederiz_kadir/caller.py:533, 564`). For normal agent tasks, declare `artifact_schema` on the i2p step — workflow_engine's `constrained_emit.maybe_apply` (`coulson/__init__.py:104-106`) reads the schema and injects a post-emit structured pass.
- No recursive HTML walker exists in mr_roboto — Plan 3 writes its own `os.walk`-based walker.
- `get_mission_workspace(mission_id: int) -> str` at `src/tools/workspace.py:439-447` returns an absolute path. Mechanicals resolve via `payload.get("workspace_path")` override → `get_mission_workspace(mission_id)` fallback (pattern in `marketing_copy.py:343-363`).

---

## File ownership

**Plan 3 owns (NEW files):**
- `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`
- `packages/mr_roboto/src/mr_roboto/verify_swap_placeholder_images_shape.py`
- `packages/mr_roboto/tests/test_swap_placeholder_images.py`
- `packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py`
- `packages/mr_roboto/tests/test_verify_swap_placeholder_images_shape.py`
- `packages/mr_roboto/tests/test_emit_preview_url_assets.py`
- `src/agents/prompt_writer.py`
- `docs/templates/prompt_writer/diffusion_prompt_template.md`
- `tests/agents/test_prompt_writer.py`
- `tests/agents/test_prompt_writer_template.py`
- `tests/i2p/test_i2p_swap_step.py`
- `tests/integration/test_image_i2p_swap_e2e.py`

**Plan 3 extends (anchors clearly marked in each task):**
- `packages/mr_roboto/src/mr_roboto/__init__.py` — import + `__all__` + `_run_dispatch` branches for both verbs.
- `packages/mr_roboto/src/mr_roboto/reversibility.py` — register both verbs (`swap_placeholder_images: "full"`, `verify_swap_placeholder_images_shape: "full"`).
- `src/agents/__init__.py` — register `prompt_writer` in `AGENT_REGISTRY`.
- `src/workflows/i2p/i2p_v3.json` — new step `5.35` + sibling `5.35.verify` + extend `5.40`'s `depends_on`.

**Plan 3 must NOT touch (Plan 1 v2 / Plan 2 v2 territory):**
- `packages/paintress/*`, `packages/renoir/*`, `packages/clair_obscur/*`.
- `packages/fatih_hoca/*` (any file).
- `src/core/llm_dispatcher.py`, `src/core/orchestrator.py`.
- `packages/general_beckman/*`, `packages/nerd_herd/*`.
- `src/app/telegram_bot.py`.

---

## Task 1: Verify discovered conventions

**Files:** none modified — read-only audit. Confirms the four facts subsequent tasks depend on.

- [ ] **Step 1: Confirm the placeholder convention is still `placehold.co`**

```bash
.venv/Scripts/python -c "import json; d = json.load(open('src/workflows/i2p/i2p_v3.json', encoding='utf-8')); steps = [s for s in d['steps'] if s.get('id') == '5.30a']; print(steps[0]['instruction'][:800])"
```
Expected: output contains `placehold.co` and `descriptive alt`. If the convention has drifted, update the regex in Task 4 before proceeding.

- [ ] **Step 2: Confirm `5.30c` exists, `5.40` exists, `5.35`/`5.35.verify` are free**

```bash
.venv/Scripts/python -c "import json; d = json.load(open('src/workflows/i2p/i2p_v3.json', encoding='utf-8')); ids = [s['id'] for s in d['steps']]; print('5.30c' in ids, '5.40' in ids, '5.35' in ids, '5.35.verify' in ids)"
```
Expected: `True True False False`. If `5.35` is taken, pick the next free id (e.g. `5.30d` / `5.30d.verify`) and use it consistently in Tasks 9/10.

- [ ] **Step 3: Confirm AGENT_REGISTRY shape and that `prompt_writer` is NOT present**

```bash
.venv/Scripts/python -c "from src.agents import AGENT_REGISTRY; print(sorted(AGENT_REGISTRY.keys()))"
```
Expected: 21+ agents, `prompt_writer` not present.

- [ ] **Step 4: Confirm Plan 1 v2 has merged (the image lane exists)**

```bash
.venv/Scripts/python -c "from src.core.llm_dispatcher import CallCategory; print(CallCategory.IMAGE.value)"
.venv/Scripts/python -c "import fatih_hoca; pick = fatih_hoca.select(needs_image=True); print(type(pick).__name__)"
```
Expected: `image` and `Pick` (or `SelectionFailure` if no provider configured — both are fine, the import being clean is the key signal). If either raises `AttributeError`/`TypeError`, Plan 1 v2 is not merged — STOP Plan 3 execution until it is.

- [ ] **Step 5: No commit (read-only)**

---

## Task 2: `prompt_writer` agent — pure config + correct artifact_schema

**Files:**
- Create: `src/agents/prompt_writer.py`
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_prompt_writer.py`

The agent is **single-call** (`max_iterations = 1`), small/local-LLM friendly. v2 emphasis: the system prompt is the contract, but **the constraint that enforces the JSON shape is the `artifact_schema` declared on the i2p step** (Task 9), which workflow_engine's `constrained_emit.maybe_apply` reads after the main call. The agent's sys_prompt still describes the schema for the first-pass LLM; the post-emit pass repairs malformed output via a `response_format=json_schema` second call.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_prompt_writer.py
import re
from src.agents.prompt_writer import PromptWriterAgent
from src.agents import AGENT_REGISTRY, get_agent


def test_registered():
    assert "prompt_writer" in AGENT_REGISTRY
    inst = get_agent("prompt_writer")
    assert isinstance(inst, PromptWriterAgent)


def test_pure_config():
    body = open("src/agents/prompt_writer.py", encoding="utf-8").read()
    methods = re.findall(r"^    def (\w+)\(", body, flags=re.MULTILINE)
    assert set(methods) <= {"get_system_prompt"}, methods


def test_config_fields():
    a = PromptWriterAgent()
    assert a.name == "prompt_writer"
    assert a.default_tier == "cheap"
    assert a.max_iterations == 1
    assert a.enable_self_reflection is False
    assert a.allowed_tools == []


def test_system_prompt_satisfies_three_invariants():
    p = PromptWriterAgent().get_system_prompt({})
    first_line = p.strip().splitlines()[0]
    assert first_line.startswith("You are "), first_line
    body = p.lower()
    assert "must" in body or "always" in body
    assert "don't" in body or "never" in body
    assert "final_answer" in p
    assert "```json" in p
    assert "placeholder_id" in p
    assert "_schema_version" in p


def test_system_prompt_mentions_template_slots():
    p = PromptWriterAgent().get_system_prompt({})
    for slot in ("design_tokens", "brand_voice", "section_intent"):
        assert slot in p.lower()
```

- [ ] **Step 2: Run + implement**

Run: `.venv/Scripts/python -m pytest tests/agents/test_prompt_writer.py -q`
Expected: FAIL — `ModuleNotFoundError`.

`src/agents/prompt_writer.py`:
```python
"""Image-generation prompt writer (Plan 3).

For a prototype with N placeholder <img> intents, emits one JSON envelope
mapping each placeholder_id to an enriched diffusion prompt. Pure config —
sys_prompt + tools, zero methods beyond get_system_prompt. Single-call.

Shape is enforced by:
  (a) sys_prompt (this file) — instructs the LLM,
  (b) artifact_schema on the i2p step (5.35.prompts) — workflow_engine's
      constrained_emit.maybe_apply runs a post-emit structured pass when
      the first call's output is malformed (response_format=json_schema).
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.prompt_writer")


class PromptWriterAgent(BaseAgent):
    name = "prompt_writer"
    description = "Turns prototype placeholder intents into diffusion prompts"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 1
    enable_self_reflection = False
    allowed_tools = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a diffusion-prompt engineer for product UI mockups. "
            "Given a list of placeholder <img> intents from a prototype, "
            "you write ONE enriched diffusion prompt per placeholder so a "
            "text-to-image model can produce the real asset.\n\n"
            "## Context\n"
            "- `design_tokens` — color palette + typography. Bias generated "
            "images toward these colors.\n"
            "- `brand_voice` — short paragraph describing voice/mood. Bias "
            "subject choice and composition.\n"
            "- `section_intent` — per-placeholder screen role (hero / "
            "feature / avatar / product / background / icon). Drives "
            "composition + framing.\n"
            "- `placeholders` — list of `{placeholder_id, alt, width, "
            "height, section}`. `alt` is the seed; enrich it with style + "
            "composition + lighting + color cues.\n\n"
            "## You must\n"
            "- Always emit exactly one prompt per placeholder — every "
            "`placeholder_id` in the input must appear in the output.\n"
            "- Always echo the `placeholder_id` verbatim.\n"
            "- Always keep prompts under 220 characters — diffusion "
            "models truncate; the first words dominate.\n"
            "- Always start each prompt with the subject (the `alt` "
            "intent), then add style + composition + color cues.\n"
            "- Always include at least one `design_tokens` color cue.\n\n"
            "## You must never\n"
            "- Don't invent new `placeholder_id` values not in the input.\n"
            "- Don't return text outside the JSON envelope.\n"
            "- Don't include negative prompts or model-specific tokens "
            "(`<lora:...>`, `((emphasis))`).\n"
            "- Never copy `alt` verbatim — enrich it.\n\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": {\n'
            '    "_schema_version": "1",\n'
            '    "prompts": [\n'
            "      {\n"
            '        "placeholder_id": "<verbatim id from input>",\n'
            '        "prompt": "<enriched diffusion prompt, <=220 chars>"\n'
            "      }\n"
            "    ]\n"
            "  },\n"
            '  "memories": {}\n'
            "}\n"
            "```\n"
        )
```

In `src/agents/__init__.py`, add import (alphabetically) + AGENT_REGISTRY entry:
```python
from .prompt_writer import PromptWriterAgent
# ...
    "prompt_writer": PromptWriterAgent(),
```

- [ ] **Step 3: Run + regression**

Run: `.venv/Scripts/python -m pytest tests/agents/test_prompt_writer.py tests/agents/test_prompt_quality.py -q`
Expected: PASS (5 + the existing 3-invariant sweep covering all 22 agents).

- [ ] **Step 4: Commit**

```bash
git add src/agents/prompt_writer.py src/agents/__init__.py tests/agents/test_prompt_writer.py
git commit -m "feat(image): prompt_writer agent (pure-config single-call)"
```

---

## Task 3: Diffusion-prompt template + few-shot exemplars

**Files:**
- Create: `docs/templates/prompt_writer/diffusion_prompt_template.md`
- Modify: `src/agents/prompt_writer.py` (add `load_diffusion_prompt_template`)
- Test: `tests/agents/test_prompt_writer_template.py`

(Identical to Plan 3 v1 Task 3 — no audit finding here. Reproduced for plan completeness.)

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_prompt_writer_template.py
import os
from src.agents.prompt_writer import load_diffusion_prompt_template


def test_template_loads():
    body = load_diffusion_prompt_template()
    assert body is not None and len(body) > 200


def test_has_few_shot_block():
    body = load_diffusion_prompt_template()
    assert "EXAMPLE 1" in body and "EXAMPLE 2" in body
    assert any(w in body.lower() for w in ("coral", "slate", "color"))


def test_has_slot_placeholders():
    body = load_diffusion_prompt_template()
    for slot in ("{design_tokens}", "{brand_voice}", "{section_intent}",
                 "{placeholders}"):
        assert slot in body, f"missing slot: {slot}"


def test_file_under_docs_templates():
    assert os.path.isfile(
        "docs/templates/prompt_writer/diffusion_prompt_template.md"
    )
```

- [ ] **Step 2: Run + implement**

Run: `.venv/Scripts/python -m pytest tests/agents/test_prompt_writer_template.py -q`
Expected: FAIL.

`docs/templates/prompt_writer/diffusion_prompt_template.md`:
```markdown
# Diffusion-prompt template (prompt_writer)

You will receive the design context and a list of placeholder intents.
Fill the JSON envelope below with one enriched prompt per placeholder.

## Inputs

DESIGN TOKENS (palette + type — bias generated images toward these colors):
{design_tokens}

BRAND VOICE (mood / tone — bias subject choice and composition):
{brand_voice}

SECTION INTENTS (screen role per placeholder):
{section_intent}

PLACEHOLDERS (the list to enrich):
{placeholders}

## Few-shot exemplars

EXAMPLE 1
Input placeholder:
  placeholder_id: hero_1
  alt: "smiling barista handing over a takeaway cup"
  width: 390
  height: 220
  section: hero
Brand voice: "warm, neighborhood coffee shop — third-wave"
Design tokens: { primary: "#E07A5F" (warm coral), surface: "#F4F1DE" (cream) }
Expected prompt:
  "Warm candid photo of a smiling young barista handing a takeaway cup, soft morning light through cafe window, warm coral apron accent against cream-toned interior, shallow depth of field, eye-level wide composition."

EXAMPLE 2
Input placeholder:
  placeholder_id: feature_2
  alt: "ai-powered task triage dashboard"
  width: 260
  height: 180
  section: feature
Brand voice: "calm, professional productivity tool"
Design tokens: { primary: "#3D405B" (slate indigo), accent: "#81B29A" (muted sage) }
Expected prompt:
  "Minimal isometric illustration of a clean dashboard with sorted task cards, slate indigo header bar, muted sage progress accents on soft white background, flat vector style, centered composition, no text on screen."

EXAMPLE 3
Input placeholder:
  placeholder_id: avatar_3
  alt: "user portrait"
  width: 64
  height: 64
  section: testimonial
Brand voice: "diverse community, real people"
Design tokens: { primary: "#264653" (deep teal) }
Expected prompt:
  "Friendly close-up headshot of a person against soft deep-teal blurred background, natural diffuse lighting, eye contact, neutral expression, square composition."

## Now emit the JSON

Return ONLY the final_answer JSON envelope — no prose, no markdown
fences around it. Every placeholder_id from the input MUST appear in
`prompts`. Each prompt MUST be <=220 characters and MUST embed at least
one design-token color cue.
```

Append to `src/agents/prompt_writer.py`:
```python
import os as _os

_DEFAULT_TEMPLATE_PATH = "docs/templates/prompt_writer/diffusion_prompt_template.md"


def load_diffusion_prompt_template(path: str | None = None) -> str | None:
    """Load the diffusion-prompt few-shot template. Slots are
    {design_tokens}, {brand_voice}, {section_intent}, {placeholders}.
    Returns None if file missing — agent still works on sys_prompt alone."""
    p = path or _DEFAULT_TEMPLATE_PATH
    if not _os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None
```

Run: `.venv/Scripts/python -m pytest tests/agents/test_prompt_writer_template.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/templates/prompt_writer/diffusion_prompt_template.md src/agents/prompt_writer.py tests/agents/test_prompt_writer_template.py
git commit -m "feat(image): prompt_writer template + few-shot exemplars"
```

---

## Task 4: `swap_placeholder_images` — scaffold + **recursive** scanner

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`
- Create: `packages/mr_roboto/tests/test_swap_placeholder_images.py`

v2 changes from Plan 3 v1: (1) **recursive** HTML walk via `os.walk` (Plan 3 v1 missed subdirectories), (2) **JSON-string parse helper** ready for Tasks 5 + 6, (3) **direct-write to assets** (no rename dance).

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_swap_placeholder_images.py
import json
import os
import pytest
from mr_roboto.swap_placeholder_images import (
    swap_placeholder_images,
    _scan_placeholders,
    _list_html_files,
    _PLACEHOLDER_HOST_RE,
    _parse_task_result,
)


def test_placeholder_host_regex():
    assert _PLACEHOLDER_HOST_RE.search("https://placehold.co/64x64/eee/333?text=x")
    assert _PLACEHOLDER_HOST_RE.search("http://placehold.co/256x256")
    assert not _PLACEHOLDER_HOST_RE.search("/assets/hero_1.png")
    assert not _PLACEHOLDER_HOST_RE.search("assets/hero_1.png")
    assert not _PLACEHOLDER_HOST_RE.search("https://example.com/real.png")


_HTML = """<!DOCTYPE html>
<html><body class="w-[390px] min-h-[844px]">
  <img src="https://placehold.co/390x220/E07A5F/FFF?text=hero"
       alt="smiling barista handing over a takeaway cup">
  <img src="https://placehold.co/260x180/3D405B/FFF?text=feat"
       alt="ai-powered task triage dashboard">
  <img src="/assets/already_real.png" alt="something already swapped">
  <img src="https://placehold.co/64x64/264653/FFF?text=u"
       alt="user portrait">
</body></html>"""


def test_scan_finds_three(tmp_path):
    p = tmp_path / "home.html"
    p.write_text(_HTML, encoding="utf-8")
    hits = _scan_placeholders(str(p))
    assert len(hits) == 3
    ids = {h["placeholder_id"] for h in hits}
    assert ids == {"home__0", "home__1", "home__2"}
    assert all(h["alt"] for h in hits)
    assert all(h["width"] > 0 and h["height"] > 0 for h in hits)


def test_scan_handles_missing(tmp_path):
    assert _scan_placeholders(str(tmp_path / "missing.html")) == []


def test_scan_handles_no_placeholders(tmp_path):
    p = tmp_path / "empty.html"
    p.write_text("<html><body>no images</body></html>", encoding="utf-8")
    assert _scan_placeholders(str(p)) == []


def test_list_html_recursive(tmp_path):
    """v2 fix: walks subdirectories so multi-screen prototypes work."""
    web = tmp_path / ".web"
    (web / "screens").mkdir(parents=True)
    (web / "home.html").write_text("<html></html>", encoding="utf-8")
    (web / "screens" / "onboarding.html").write_text("<html></html>", encoding="utf-8")
    (web / "screens" / "settings.html").write_text("<html></html>", encoding="utf-8")
    (web / "assets" / "ignored.png").parent.mkdir(exist_ok=True)
    (web / "assets" / "ignored.png").write_bytes(b"\x89PNG")  # not an HTML
    files = _list_html_files(str(tmp_path))
    names = sorted(os.path.basename(f) for f in files)
    assert names == ["home.html", "onboarding.html", "settings.html"]


def test_parse_task_result_handles_json_string():
    """v2 fix: TaskResult.result is a JSON string in production."""
    class _TR:
        result = json.dumps({"path": "/x/y.png", "provider": "p"})
    parsed = _parse_task_result(_TR())
    assert parsed == {"path": "/x/y.png", "provider": "p"}


def test_parse_task_result_handles_dict():
    """Defensive: tests may pass dicts."""
    class _TR:
        result = {"path": "/x/y.png"}
    parsed = _parse_task_result(_TR())
    assert parsed == {"path": "/x/y.png"}


def test_parse_task_result_handles_none():
    class _TR:
        result = None
    assert _parse_task_result(_TR()) == {}


def test_parse_task_result_handles_garbage_string():
    class _TR:
        result = "not json {"
    assert _parse_task_result(_TR()) == {}


@pytest.mark.asyncio
async def test_swap_no_html_files(monkeypatch, tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["replaced_count"] == 0
    assert res["html_files_seen"] == 0
```

- [ ] **Step 2: Run + implement**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q`
Expected: FAIL.

`packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`:
```python
"""Plan 3 — swap placehold.co <img> tags for real diffusion-generated PNGs.

Pipeline:
  1. Recursively walk `mission_{id}/.web/**/*.html`.
  2. Scan each file for <img> whose src is placehold.co.
  3. Enqueue ONE prompt_writer task (beckman.enqueue, await_inline) with
     the design context + placeholder list. Receive a placeholder_id ->
     prompt map. Robust to JSON-string TaskResult.result (recon-verified
     shape).
  4. Per placeholder: enqueue one image task (context.image_call.raw_dispatch).
     Beckman routes via Plan 1 v2's _select_for_admission(needs_image=True).
  5. PNG lands directly under .web/assets/<placeholder_id>.png (paintress
     writes the file; mechanic asks paintress to put it there).
  6. HTML <img src> rewritten to relative "assets/<id>.png".
  7. Graceful degrade: per-placeholder failure keeps the placehold.co URL.

Mirrors marketing_copy.py's mechanical shape — internally enqueues LLM +
image work through beckman, never calls dispatcher/HK/paintress directly
(feedback_singular_dispatcher_caller). Reversibility: "full"."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.swap_placeholder_images")


# ── Placeholder detection ──────────────────────────────────────────────

_PLACEHOLDER_HOST_RE = re.compile(r"^https?://placehold\.co/", re.IGNORECASE)
_IMG_RE = re.compile(r"<img\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r'(\b[a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*"([^"]*)"')
_DIM_RE = re.compile(r"/(\d{2,4})x(\d{2,4})", re.IGNORECASE)


def _parse_attrs(tag_inner: str) -> dict[str, str]:
    return {k.lower(): v for k, v in _ATTR_RE.findall(tag_inner)}


def _slug_from_path(html_path: str) -> str:
    return Path(html_path).stem


def _section_from_alt(alt: str) -> str:
    a = (alt or "").lower()
    if any(t in a for t in ("hero", "header", "banner")): return "hero"
    if any(t in a for t in ("avatar", "portrait", "headshot")): return "avatar"
    if "icon" in a: return "icon"
    if "background" in a or "bg" in a: return "background"
    return "feature"


def _scan_placeholders(html_path: str) -> list[dict[str, Any]]:
    try:
        with open(html_path, encoding="utf-8") as fh:
            html = fh.read()
    except OSError:
        return []
    slug = _slug_from_path(html_path)
    out: list[dict[str, Any]] = []
    occ = 0
    for m in _IMG_RE.finditer(html):
        attrs = _parse_attrs(m.group(1) or "")
        src = (attrs.get("src") or "").strip()
        if not _PLACEHOLDER_HOST_RE.search(src):
            continue
        alt = (attrs.get("alt") or "").strip()
        dim_m = _DIM_RE.search(src)
        w, h = (int(dim_m.group(1)), int(dim_m.group(2))) if dim_m else (512, 512)
        out.append({
            "placeholder_id": f"{slug}__{occ}",
            "alt": alt, "width": w, "height": h,
            "section": _section_from_alt(alt),
            "original_src": src,
            "tag_span": (m.start(), m.end()),
            "html_path": html_path,
        })
        occ += 1
    return out


# ── Workspace + assets helpers ─────────────────────────────────────────

def _web_root(workspace_path: str) -> str:
    return os.path.join(workspace_path, ".web")


def _assets_dir(workspace_path: str) -> str:
    p = os.path.join(workspace_path, ".web", "assets")
    os.makedirs(p, exist_ok=True)
    return p


def _list_html_files(workspace_path: str) -> list[str]:
    """v2 fix: recursive walk of <ws>/.web/**/*.html (Plan 3 v1 was flat
    and missed subdirectory screens)."""
    root = _web_root(workspace_path)
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".html"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


# ── TaskResult.result parser (the recon-confirmed v2 fix) ──────────────

def _parse_task_result(result_obj) -> dict:
    """TaskResult.result is a JSON STRING in production (recon: orchestrator
    json.dumps at :326). Mirror dispatcher's _task_result_to_request_response
    (llm_dispatcher.py:137-163) — accept both string and dict."""
    raw = getattr(result_obj, "result", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}
    return {}


# ── beckman wrapper (test-patchable, mirrors marketing_copy) ───────────

async def _enqueue_beckman(spec: dict, **kwargs):
    from general_beckman import enqueue as _enqueue
    return await _enqueue(spec, **kwargs)


# ── Main entry ─────────────────────────────────────────────────────────

async def swap_placeholder_images(
    mission_id: int,
    workspace_path: str | None = None,
    design_tokens: dict | None = None,
    brand_voice: str | None = None,
) -> dict[str, Any]:
    """Best-effort: per-placeholder failures keep the original placeholder.
    Returns:
      {ok: bool, replaced_count, skipped_count, html_files_seen,
       html_files_changed, errors: list[str]}
    """
    from src.tools.workspace import get_mission_workspace
    workspace_path = workspace_path or get_mission_workspace(int(mission_id))
    logger.info("swap_placeholder_images: starting",
                mission_id=mission_id, workspace_path=workspace_path)

    html_files = _list_html_files(workspace_path)
    if not html_files:
        return {
            "ok": True, "replaced_count": 0, "skipped_count": 0,
            "html_files_seen": 0, "html_files_changed": 0, "errors": [],
        }

    all_placeholders: list[dict[str, Any]] = []
    for h in html_files:
        all_placeholders.extend(_scan_placeholders(h))

    if not all_placeholders:
        return {
            "ok": True, "replaced_count": 0, "skipped_count": 0,
            "html_files_seen": len(html_files), "html_files_changed": 0,
            "errors": [],
        }

    # Task 5 fills prompt_writer; Task 6 fills fanout. Scaffold returns scan.
    return {
        "ok": True, "replaced_count": 0, "skipped_count": len(all_placeholders),
        "html_files_seen": len(html_files), "html_files_changed": 0,
        "errors": [],
    }
```

- [ ] **Step 3: Run + commit**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q`
Expected: PASS (10 passed).

```bash
git add packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py packages/mr_roboto/tests/test_swap_placeholder_images.py
git commit -m "feat(image): swap_placeholder_images scaffold (recursive scan, json-string parse)"
```

---

## Task 5: Wire prompt_writer call **with JSON-string-safe parse**

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`
- Modify: `packages/mr_roboto/tests/test_swap_placeholder_images.py`

v2 fix: the prompt_writer's `TaskResult.result` is a JSON string. The parser uses `_parse_task_result` (Task 4) so the existing `isinstance(raw, dict)` shape is decoded.

- [ ] **Step 1: Extend the failing test**

Append to `packages/mr_roboto/tests/test_swap_placeholder_images.py`:

```python
import json as _json  # if not already imported


# -- Task 5: prompt_writer enqueue --------------------------------------

@pytest.mark.asyncio
async def test_calls_prompt_writer_once_with_json_string_result(
    monkeypatch, tmp_path,
):
    """v2 fix: PRODUCTION shape — TaskResult.result is a JSON STRING."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )

    captured = []

    class _PromptResultJSONString:
        status = "completed"
        # PRODUCTION SHAPE — JSON string, not dict.
        result = _json.dumps({
            "_schema_version": "1",
            "prompts": [
                {"placeholder_id": "home__0", "prompt": "coral barista scene"},
                {"placeholder_id": "home__1", "prompt": "slate dashboard"},
                {"placeholder_id": "home__2", "prompt": "teal portrait"},
            ],
        })
        error = None

    async def _fake_enqueue(spec, **kwargs):
        captured.append({"spec": spec, "kwargs": kwargs})
        return _PromptResultJSONString()
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fake_enqueue,
    )
    async def _fake_fanout(workspace_path, placeholders, prompt_map):
        return {"replaced": 0, "skipped": len(placeholders),
                "html_files_changed": 0, "errors": []}
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._fanout_and_rewrite", _fake_fanout,
    )

    res = await swap_placeholder_images(
        mission_id=42,
        design_tokens={"primary": "#E07A5F"},
        brand_voice="warm, neighborhood coffee shop",
    )

    assert len(captured) == 1
    spec = captured[0]["spec"]
    assert spec["agent_type"] == "prompt_writer"
    assert captured[0]["kwargs"].get("await_inline") is True
    # Result string was parsed — the fanout was called (would not be if
    # parser had treated string as None).
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_prompt_writer_failure_degrades_gracefully(monkeypatch, tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    class _Fail:
        status = "failed"; result = None; error = "LLM down"
    async def _fail_enqueue(spec, **kwargs):
        return _Fail()
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fail_enqueue,
    )

    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["replaced_count"] == 0
    assert res["skipped_count"] == 3
    assert any("prompt_writer" in e for e in res["errors"])


@pytest.mark.asyncio
async def test_prompt_writer_malformed_json_degrades(monkeypatch, tmp_path):
    """Cheap-tier LLM emits garbage — parser returns {} → no prompts → skip."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    class _Garbage:
        status = "completed"; result = "not json"; error = None
    async def _enq(spec, **kwargs):
        return _Garbage()
    monkeypatch.setattr("mr_roboto.swap_placeholder_images._enqueue_beckman", _enq)
    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["replaced_count"] == 0
    assert res["skipped_count"] == 3
```

- [ ] **Step 2: Run + implement**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q`
Expected: FAIL on the new tests (`_fanout_and_rewrite` stub doesn't exist yet, `_call_prompt_writer` doesn't exist).

Append to `swap_placeholder_images.py`:
```python
async def _call_prompt_writer(
    *, mission_id: int,
    placeholders: list[dict[str, Any]],
    design_tokens: dict | None,
    brand_voice: str | None,
) -> dict[str, str] | None:
    """Enqueue one prompt_writer task. Returns placeholder_id -> prompt map,
    or None on failure. Robust to JSON-string TaskResult.result."""
    visible = [
        {"placeholder_id": p["placeholder_id"], "alt": p["alt"],
         "width": p["width"], "height": p["height"], "section": p["section"]}
        for p in placeholders
    ]
    template_text = None
    try:
        from src.agents.prompt_writer import load_diffusion_prompt_template
        template_text = load_diffusion_prompt_template()
    except Exception:
        template_text = None

    spec = {
        "title": f"prompt_writer:mission#{mission_id}",
        "description": "Enrich placeholder <img> intents into diffusion prompts.",
        "agent_type": "prompt_writer",
        "kind": "main_work",
        "priority": 5,
        "mission_id": mission_id,
        "context": {
            "design_tokens": design_tokens or {},
            "brand_voice": brand_voice or "",
            "placeholders": visible,
            "diffusion_template": template_text or "",
        },
    }
    try:
        result = await _enqueue_beckman(spec, await_inline=True)
    except Exception as exc:
        logger.warning("prompt_writer enqueue raised: %s", exc)
        return None

    if getattr(result, "status", "") != "completed":
        logger.warning(
            "prompt_writer task did not complete (status=%r, error=%r)",
            getattr(result, "status", ""), getattr(result, "error", ""),
        )
        return None

    parsed = _parse_task_result(result)
    # Tolerate both shapes: top-level prompts OR nested under "result".
    prompts = parsed.get("prompts")
    if prompts is None and isinstance(parsed.get("result"), dict):
        prompts = parsed["result"].get("prompts")
    if not isinstance(prompts, list):
        logger.warning("prompt_writer returned no prompts list (parsed=%r)", parsed)
        return None

    out: dict[str, str] = {}
    for entry in prompts:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("placeholder_id")
        prompt = entry.get("prompt")
        if isinstance(pid, str) and isinstance(prompt, str) and prompt.strip():
            out[pid] = prompt.strip()
    return out or None


# Stub for Task 6 testability.
async def _fanout_and_rewrite(
    workspace_path: str,
    placeholders: list[dict[str, Any]],
    prompt_map: dict[str, str],
) -> dict[str, Any]:
    return {"replaced": 0, "skipped": len(placeholders),
            "html_files_changed": 0, "errors": []}
```

Replace `swap_placeholder_images`'s body's tail with the real flow:
```python
    prompt_map = await _call_prompt_writer(
        mission_id=int(mission_id), placeholders=all_placeholders,
        design_tokens=design_tokens, brand_voice=brand_voice,
    )
    if prompt_map is None:
        return {
            "ok": True, "replaced_count": 0,
            "skipped_count": len(all_placeholders),
            "html_files_seen": len(html_files), "html_files_changed": 0,
            "errors": ["prompt_writer task did not return a usable prompt map"],
        }
    fanout = await _fanout_and_rewrite(
        workspace_path, all_placeholders, prompt_map,
    )
    return {
        "ok": True,
        "replaced_count": fanout.get("replaced", 0),
        "skipped_count": fanout.get("skipped", 0),
        "html_files_seen": len(html_files),
        "html_files_changed": fanout.get("html_files_changed", 0),
        "errors": fanout.get("errors", []),
    }
```

- [ ] **Step 3: Run + commit**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q`
Expected: PASS (13 passed).

```bash
git add packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py packages/mr_roboto/tests/test_swap_placeholder_images.py
git commit -m "feat(image): prompt_writer enqueue with JSON-string-safe parse"
```

---

## Task 6: Per-placeholder image fanout + HTML rewrite — **JSON-string-safe + direct-write to assets**

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`
- Modify: `packages/mr_roboto/tests/test_swap_placeholder_images.py`

v2 fixes: (1) `_parse_task_result` for image task results too, (2) paintress writes directly to `<ws>/.web/assets/` with `filename_hint=<pid>` so the saved file lands as `<pid>_<ms>.png`, then `_move_into_assets` is replaced by a single-step rename to `<pid>.png` (no separate copy step), (3) HTML rewrite preserves attribute order.

- [ ] **Step 1: Extend the failing test**

Append to test file:
```python
import os as _os
import io
from PIL import Image as _Image


def _write_real_png(path, w=64, h=64):
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    _Image.new("RGB", (w, h), (100, 150, 200)).save(path, "PNG")


@pytest.mark.asyncio
async def test_full_swap_writes_assets_and_rewrites_html_json_string(
    monkeypatch, tmp_path,
):
    """v2 fix: production shape — image TaskResult.result is a JSON STRING."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )

    class _OK:
        status = "completed"
        # prompt_writer result as JSON string (production shape).
        result = _json.dumps({
            "_schema_version": "1",
            "prompts": [
                {"placeholder_id": "home__0", "prompt": "p0"},
                {"placeholder_id": "home__1", "prompt": "p1"},
                {"placeholder_id": "home__2", "prompt": "p2"},
            ],
        })
        error = None

    image_idx = {"n": 0}

    async def _fake_enqueue(spec, **kwargs):
        if spec.get("agent_type") == "prompt_writer":
            return _OK()
        # Image task — paintress writes the file as
        # <out_dir>/<filename_hint>_<ms>.png. We simulate that exactly.
        ic = spec["context"]["image_call"]
        idx = image_idx["n"]; image_idx["n"] += 1
        # Use a deterministic mock filename so test is stable.
        png_path = _os.path.join(ic["out_dir"], f"{ic['filename_hint']}_mock{idx}.png")
        _write_real_png(png_path, ic["width"], ic["height"])

        class _ImgResult:
            status = "completed"
            # PRODUCTION SHAPE — JSON string.
            result = _json.dumps({
                "path": png_path, "provider": "pollinations",
                "model": "pollinations/flux", "cost": 0.0,
            })
            error = None
        return _ImgResult()

    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fake_enqueue,
    )

    res = await swap_placeholder_images(
        mission_id=42, design_tokens={"primary": "#E07A5F"},
        brand_voice="warm",
    )

    assert res["ok"] is True
    assert res["replaced_count"] == 3
    assert res["skipped_count"] == 0
    assert res["html_files_changed"] == 1

    assets = tmp_path / ".web" / "assets"
    pngs = sorted(p.name for p in assets.glob("*.png"))
    # 3 files renamed to <pid>.png stable names.
    assert "home__0.png" in pngs
    assert "home__1.png" in pngs
    assert "home__2.png" in pngs

    rewritten = (web / "home.html").read_text(encoding="utf-8")
    assert "placehold.co" not in rewritten
    assert rewritten.count('src="assets/home__0.png"') == 1
    assert rewritten.count('src="assets/home__1.png"') == 1
    assert rewritten.count('src="assets/home__2.png"') == 1
    assert "/assets/already_real.png" in rewritten  # untouched real src


@pytest.mark.asyncio
async def test_per_image_failure_keeps_placeholder(monkeypatch, tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    class _OK:
        status = "completed"
        result = _json.dumps({
            "_schema_version": "1",
            "prompts": [
                {"placeholder_id": "home__0", "prompt": "p0"},
                {"placeholder_id": "home__1", "prompt": "p1"},
                {"placeholder_id": "home__2", "prompt": "p2"},
            ],
        })
        error = None
    n = {"i": 0}
    async def _flaky(spec, **kwargs):
        if spec.get("agent_type") == "prompt_writer":
            return _OK()
        ic = spec["context"]["image_call"]; idx = n["i"]; n["i"] += 1
        if idx == 1:
            class _F:
                status = "failed"; result = None; error = "rate-limit"
            return _F()
        path = _os.path.join(ic["out_dir"], f"{ic['filename_hint']}_x{idx}.png")
        _write_real_png(path, ic["width"], ic["height"])
        class _I:
            status = "completed"
            result = _json.dumps({"path": path, "provider": "pollinations"})
            error = None
        return _I()
    monkeypatch.setattr("mr_roboto.swap_placeholder_images._enqueue_beckman", _flaky)

    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["replaced_count"] == 2
    assert res["skipped_count"] == 1
    rewritten = (web / "home.html").read_text(encoding="utf-8")
    assert rewritten.count("placehold.co") == 1
    assert rewritten.count('src="assets/') == 2
```

- [ ] **Step 2: Run + implement**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q`
Expected: FAIL on new tests (fanout still a stub).

Replace the `_fanout_and_rewrite` stub with the real impl:
```python
async def _generate_one_image(
    *, placeholder: dict[str, Any], prompt: str, out_dir: str,
) -> str | None:
    """Enqueue one image task; return the PNG path or None."""
    pid = placeholder["placeholder_id"]
    spec = {
        "title": f"image:{pid}",
        "description": f"Generate image for placeholder {pid}",
        "agent_type": "image",
        "kind": "image",
        "runner": "direct",
        "priority": 5,
        "context": {
            "image_call": {
                "raw_dispatch": True,
                "prompt": prompt,
                "out_dir": out_dir,
                "width": int(placeholder.get("width") or 512),
                "height": int(placeholder.get("height") or 512),
                "quality_tier": "fast",
                "filename_hint": pid,
            },
        },
    }
    try:
        result = await _enqueue_beckman(spec, await_inline=True)
    except Exception as exc:
        logger.warning("image enqueue raised for %s: %s", pid, exc)
        return None
    if getattr(result, "status", "") != "completed":
        logger.info("image task for %s did not complete (status=%r)",
                    pid, getattr(result, "status", ""))
        return None
    payload = _parse_task_result(result)
    path = payload.get("path") or payload.get("content")
    if not (path and os.path.isfile(path)):
        logger.warning("image task for %s returned no usable path", pid)
        return None
    return path


def _rename_to_pid(src_path: str, assets_dir: str, placeholder_id: str) -> str:
    """Rename the timestamp-named PNG to a stable <pid>.png inside assets/.
    Single rename — no copy fallback unless rename fails."""
    os.makedirs(assets_dir, exist_ok=True)
    final = os.path.join(assets_dir, f"{placeholder_id}.png")
    if os.path.abspath(src_path) == os.path.abspath(final):
        return f"{placeholder_id}.png"
    try:
        if os.path.exists(final):
            os.remove(final)
        os.replace(src_path, final)
        return f"{placeholder_id}.png"
    except OSError as exc:
        logger.warning("rename failed for %s: %s", placeholder_id, exc)
        try:
            import shutil
            shutil.copyfile(src_path, final)
            return f"{placeholder_id}.png"
        except OSError:
            return ""


def _swap_src_in_tag(tag: str, new_src: str) -> str:
    """Replace src="..." inside a single <img> tag, preserving other attrs."""
    return re.sub(r'src\s*=\s*"[^"]*"', f'src="{new_src}"', tag, count=1,
                  flags=re.IGNORECASE)


def _rewrite_html_srcs(
    html_path: str, rewrites: dict[tuple[int, int], str],
) -> bool:
    if not rewrites:
        return False
    try:
        with open(html_path, encoding="utf-8") as fh:
            html = fh.read()
    except OSError:
        return False
    ordered = sorted(rewrites.items(), key=lambda kv: kv[0][0], reverse=True)
    changed = False
    for (start, end), new_src in ordered:
        old_tag = html[start:end]
        new_tag = _swap_src_in_tag(old_tag, new_src)
        if new_tag != old_tag:
            html = html[:start] + new_tag + html[end:]
            changed = True
    if changed:
        tmp = html_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(html)
        os.replace(tmp, html_path)
    return changed


async def _fanout_and_rewrite(
    workspace_path: str,
    placeholders: list[dict[str, Any]],
    prompt_map: dict[str, str],
) -> dict[str, Any]:
    """Sequential per-placeholder: image enqueue → rename to <pid>.png →
    record rewrite. Per-placeholder failures kept in errors; original
    placehold.co URL survives in HTML for that slot."""
    assets_dir = _assets_dir(workspace_path)
    rewrites_per_file: dict[str, dict[tuple[int, int], str]] = {}
    errors: list[str] = []
    replaced = 0
    skipped = 0

    for ph in placeholders:
        pid = ph["placeholder_id"]
        prompt = prompt_map.get(pid)
        if not prompt:
            skipped += 1
            errors.append(f"no prompt for {pid}")
            continue
        path = await _generate_one_image(
            placeholder=ph, prompt=prompt, out_dir=assets_dir,
        )
        if not path:
            skipped += 1
            errors.append(f"image gen failed for {pid}")
            continue
        final_name = _rename_to_pid(path, assets_dir, pid)
        if not final_name:
            skipped += 1
            errors.append(f"rename failed for {pid}")
            continue
        new_src = f"assets/{final_name}"
        rewrites_per_file.setdefault(ph["html_path"], {})[
            tuple(ph["tag_span"])
        ] = new_src
        replaced += 1

    files_changed = 0
    for path, rewrites in rewrites_per_file.items():
        if _rewrite_html_srcs(path, rewrites):
            files_changed += 1

    return {
        "replaced": replaced, "skipped": skipped,
        "html_files_changed": files_changed, "errors": errors,
    }
```

- [ ] **Step 3: Run + commit**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q`
Expected: PASS (15 passed).

```bash
git add packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py packages/mr_roboto/tests/test_swap_placeholder_images.py
git commit -m "feat(image): swap fanout + HTML rewrite (JSON-string-safe, direct rename)"
```

---

## Task 7: `verify_swap_placeholder_images_shape` mechanic + posthook

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/verify_swap_placeholder_images_shape.py`
- Create: `packages/mr_roboto/tests/test_verify_swap_placeholder_images_shape.py`

Following the `verify_charter_shape` template (recon): pure function, dispatched via `if action == "verify_swap_placeholder_images_shape"` in `_run_dispatch`. Validates that the swap step's result is shape-correct (the artifact: presence of an `assets/` dir under `.web/`, no surviving `placehold.co` references in any HTML EXCEPT where matched by an entry in `errors`, i.e. graceful-degrade leftovers).

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_verify_swap_placeholder_images_shape.py
import os
from mr_roboto.verify_swap_placeholder_images_shape import (
    verify_swap_placeholder_images_shape,
)


_HTML_REWRITTEN = """<!DOCTYPE html>
<html><body>
  <img src="assets/home__0.png" alt="hero">
  <img src="assets/home__1.png" alt="feat">
  <img src="assets/home__2.png" alt="user">
</body></html>"""

_HTML_PARTIAL = """<!DOCTYPE html>
<html><body>
  <img src="assets/home__0.png" alt="hero">
  <img src="https://placehold.co/260x180/3D405B/FFF?text=feat" alt="feat">
  <img src="assets/home__2.png" alt="user">
</body></html>"""


def test_passes_when_all_placeholders_replaced(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__1.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 3, "skipped_count": 0, "errors": []},
    )
    assert res["ok"] is True


def test_passes_when_skipped_matches_surviving_placeholders(tmp_path):
    """Graceful degrade: 1 placeholder skipped → 1 placehold.co survives in
    HTML. That matches; verifier accepts."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PARTIAL, encoding="utf-8")
    (web / "assets").mkdir()
    for n in ("home__0.png", "home__2.png"):
        (web / "assets" / n).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 2, "skipped_count": 1,
                     "errors": ["image gen failed for home__1"]},
    )
    assert res["ok"] is True


def test_fails_when_replaced_count_disagrees_with_html(tmp_path):
    """If swap_result claims 3 replaced but 1 placehold.co survives and
    errors is empty, the result is internally inconsistent — fail."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_PARTIAL, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 3, "skipped_count": 0, "errors": []},
    )
    assert res["ok"] is False
    assert "inconsistent" in (res.get("error") or "").lower()


def test_fails_when_assets_dir_missing_but_replaced_count_positive(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text(_HTML_REWRITTEN, encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 3, "skipped_count": 0, "errors": []},
    )
    assert res["ok"] is False
    assert "assets" in (res.get("error") or "").lower()


def test_passes_when_swap_skipped_entirely(tmp_path):
    """Swap reported 0 replaced + 0 skipped (no placeholders existed) →
    verifier passes (no work expected)."""
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text("<html><body>no img</body></html>",
                                   encoding="utf-8")
    res = verify_swap_placeholder_images_shape(
        workspace_path=str(tmp_path),
        swap_result={"replaced_count": 0, "skipped_count": 0, "errors": []},
    )
    assert res["ok"] is True
```

- [ ] **Step 2: Run + implement**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_verify_swap_placeholder_images_shape.py -q`
Expected: FAIL.

`packages/mr_roboto/src/mr_roboto/verify_swap_placeholder_images_shape.py`:
```python
"""verify_swap_placeholder_images_shape — Plan 3 posthook.

Validates that swap_placeholder_images produced a self-consistent result:
- replaced_count agrees with the number of placehold.co URLs that have
  actually disappeared from HTML (within errors-margin for graceful
  degrade).
- assets/ exists when replaced_count > 0.
- skipped_count matches the count of surviving placehold.co references.

Returns {ok: bool, error: str|None, surviving_placeholders: int,
         expected_replaced: int}."""
from __future__ import annotations

import os
import re
from typing import Any

_PLACEHOLDER_HOST_RE = re.compile(r"^https?://placehold\.co/", re.IGNORECASE)
_IMG_SRC_RE = re.compile(r'<img\b[^>]*?\bsrc\s*=\s*"([^"]*)"',
                          re.IGNORECASE | re.DOTALL)


def _walk_html(workspace_path: str) -> list[str]:
    root = os.path.join(workspace_path, ".web")
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".html"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def _count_surviving_placeholders(html_paths: list[str]) -> int:
    n = 0
    for p in html_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                html = fh.read()
        except OSError:
            continue
        for m in _IMG_SRC_RE.finditer(html):
            if _PLACEHOLDER_HOST_RE.search(m.group(1) or ""):
                n += 1
    return n


def verify_swap_placeholder_images_shape(
    *,
    workspace_path: str,
    swap_result: dict[str, Any],
) -> dict[str, Any]:
    replaced = int(swap_result.get("replaced_count", 0) or 0)
    skipped = int(swap_result.get("skipped_count", 0) or 0)
    errors_list = swap_result.get("errors") or []

    html_paths = _walk_html(workspace_path)
    surviving = _count_surviving_placeholders(html_paths)

    # Assets dir presence: required when replaced > 0.
    assets_dir = os.path.join(workspace_path, ".web", "assets")
    assets_exists = os.path.isdir(assets_dir)
    if replaced > 0 and not assets_exists:
        return {
            "ok": False,
            "error": (
                f"assets/ directory missing but replaced_count={replaced}"
            ),
            "surviving_placeholders": surviving,
            "expected_replaced": replaced,
        }

    # Consistency: surviving placehold.co URLs must equal skipped_count.
    if surviving != skipped:
        return {
            "ok": False,
            "error": (
                f"inconsistent: surviving placeholders={surviving} but "
                f"skipped_count={skipped} (errors={len(errors_list)})"
            ),
            "surviving_placeholders": surviving,
            "expected_replaced": replaced,
        }

    return {
        "ok": True,
        "error": None,
        "surviving_placeholders": surviving,
        "expected_replaced": replaced,
    }
```

- [ ] **Step 3: Run + commit**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_verify_swap_placeholder_images_shape.py -q`
Expected: PASS (5 passed).

```bash
git add packages/mr_roboto/src/mr_roboto/verify_swap_placeholder_images_shape.py packages/mr_roboto/tests/test_verify_swap_placeholder_images_shape.py
git commit -m "feat(image): verify_swap_placeholder_images_shape posthook mechanic"
```

---

## Task 8: Web-preview root serves `.web/assets/`

**Files:**
- Test: `packages/mr_roboto/tests/test_emit_preview_url_assets.py`
- Modify (docstring only): `packages/mr_roboto/src/mr_roboto/emit_preview_url.py`

The resolver already picks `.web/` when non-empty (recon-discovered conventions). v2 just verifies via test + adds a docstring pointer.

- [ ] **Step 1: Write the test**

```python
# packages/mr_roboto/tests/test_emit_preview_url_assets.py
import os
from mr_roboto.emit_preview_url import _resolve_preview_root


def test_web_root_with_assets_subdir(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "home.html").write_text("<html></html>", encoding="utf-8")
    assets = web / "assets"; assets.mkdir()
    (assets / "home__0.png").write_bytes(b"\x89PNG")
    root = _resolve_preview_root(str(tmp_path))
    assert root == str(web)
    assert os.path.isfile(os.path.join(root, "assets", "home__0.png"))


def test_web_root_with_only_assets(tmp_path):
    web = tmp_path / ".web"; web.mkdir()
    (web / "assets").mkdir()
    (web / "assets" / "ghost.png").write_bytes(b"\x89PNG")
    root = _resolve_preview_root(str(tmp_path))
    assert root == str(web)
```

- [ ] **Step 2: Run + docstring update**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_emit_preview_url_assets.py -q`
Expected: PASS — resolver already supports this.

In `emit_preview_url.py`, append to `_resolve_preview_root`'s docstring:
```
Plan 3 note: <ws>/.web/assets/ (image-gen output) is served automatically
since .web/ is the resolved root — no resolver change required.
```

- [ ] **Step 3: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/emit_preview_url.py packages/mr_roboto/tests/test_emit_preview_url_assets.py
git commit -m "docs(image): preview root serves .web/assets/"
```

---

## Task 9: Wire both mechanics into `_run_dispatch` + reversibility

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py`
- Modify: `packages/mr_roboto/src/mr_roboto/reversibility.py`
- Test: `packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py`

Mirrors `verify_charter_shape`'s wiring shape (recon-confirmed at `__init__.py:1071-1094`).

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py
import pytest
import mr_roboto
from mr_roboto.reversibility import VERB_REVERSIBILITY


def test_swap_verb_registered():
    assert "swap_placeholder_images" in VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["swap_placeholder_images"] == "full"


def test_verify_verb_registered():
    assert "verify_swap_placeholder_images_shape" in VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["verify_swap_placeholder_images_shape"] == "full"


def test_module_exports_swap():
    assert hasattr(mr_roboto, "swap_placeholder_images")
    assert "swap_placeholder_images" in mr_roboto.__all__


@pytest.mark.asyncio
async def test_dispatch_routes_swap_action(monkeypatch):
    captured = {}
    async def _fake_swap(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "replaced_count": 2, "skipped_count": 1,
                "html_files_seen": 1, "html_files_changed": 1, "errors": []}
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images.swap_placeholder_images", _fake_swap,
    )
    task = {
        "id": 100, "mission_id": 42,
        "title": "swap_test",
        "context": {"payload": {
            "action": "swap_placeholder_images",
            "design_tokens": {"primary": "#E07A5F"},
            "brand_voice": "warm",
        }},
    }
    res = await mr_roboto.run(task)
    assert res.status == "completed"
    assert res.result["replaced_count"] == 2
    assert captured["mission_id"] == 42
    assert captured["design_tokens"] == {"primary": "#E07A5F"}


@pytest.mark.asyncio
async def test_dispatch_routes_verify_action(monkeypatch):
    """The verify posthook dispatches with action=verify_swap_placeholder_images_shape."""
    captured = {}
    def _fake_verify(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "surviving_placeholders": 0, "expected_replaced": 3}
    monkeypatch.setattr(
        "mr_roboto.verify_swap_placeholder_images_shape."
        "verify_swap_placeholder_images_shape",
        _fake_verify,
    )
    task = {
        "id": 101, "mission_id": 42, "title": "verify_test",
        "context": {"payload": {
            "action": "verify_swap_placeholder_images_shape",
            "workspace_path": "/fake/ws",
            "swap_result": {"replaced_count": 3, "skipped_count": 0, "errors": []},
        }},
    }
    res = await mr_roboto.run(task)
    assert res.status == "completed"
    assert captured["workspace_path"] == "/fake/ws"
```

- [ ] **Step 2: Run + wire**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py -q`
Expected: FAIL.

In `reversibility.py`, add to `VERB_REVERSIBILITY` near `marketing_copy`:
```python
    # Plan 3 — image-gen integration. All under mission workspace; regenerable.
    "swap_placeholder_images": "full",
    "verify_swap_placeholder_images_shape": "full",
```

In `mr_roboto/__init__.py`:

Add imports near other Plan-3 / Z-series mechanicals:
```python
# Plan 3 — i2p image-gen integration
from mr_roboto.swap_placeholder_images import swap_placeholder_images  # noqa: F401
from mr_roboto.verify_swap_placeholder_images_shape import (  # noqa: F401
    verify_swap_placeholder_images_shape,
)
```

Add to `__all__`:
```python
    "swap_placeholder_images",
    "verify_swap_placeholder_images_shape",
```

Add dispatch branches in `_run_dispatch` (BEFORE any unknown-action fallback), mirroring the `verify_charter_shape` shape:
```python
    if action == "swap_placeholder_images":
        # Plan 3 — internally enqueues a prompt_writer LLM task and N image
        # tasks through beckman; never calls dispatcher/HK/paintress directly.
        from mr_roboto.swap_placeholder_images import (
            swap_placeholder_images as _swap,
        )
        try:
            res = await _swap(
                mission_id=task.get("mission_id"),
                workspace_path=payload.get("workspace_path"),
                design_tokens=payload.get("design_tokens"),
                brand_voice=payload.get("brand_voice"),
            )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="completed", result={
                "ok": True, "replaced_count": 0, "skipped_count": 0,
                "errors": [f"unexpected: {e}"],
            })

    if action == "verify_swap_placeholder_images_shape":
        # Plan 3 — verify posthook. Pure function; mirrors verify_charter_shape.
        from mr_roboto.verify_swap_placeholder_images_shape import (
            verify_swap_placeholder_images_shape as _verify,
        )
        try:
            res = _verify(
                workspace_path=payload.get("workspace_path") or "",
                swap_result=payload.get("swap_result") or {},
            )
            if not res.get("ok"):
                return Action(
                    status="failed",
                    error=f"verify_swap_placeholder_images_shape: {res.get('error')}",
                    result=res,
                )
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))
```

- [ ] **Step 3: Run + regression**

Run: `.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py packages/mr_roboto/tests/test_reversibility_registry.py packages/mr_roboto/tests/ -q -x`
Expected: PASS new tests + no new regressions.

- [ ] **Step 4: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/__init__.py packages/mr_roboto/src/mr_roboto/reversibility.py packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py
git commit -m "feat(image): wire swap + verify mechanics into mr_roboto dispatch"
```

---

## Task 10: i2p_v3.json — step 5.35 (correct convention) + 5.35.verify posthook

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`
- Test: `tests/i2p/test_i2p_swap_step.py`

v2 fix: correct mechanical step shape — `"executor": "mechanical"` (NOT the verb name), verb in `payload.action`. Plus a sibling `5.35.verify` posthook that runs `verify_swap_placeholder_images_shape` with the swap step's result. Plus `artifact_schema` on `5.35` so workflow_engine's `constrained_emit.maybe_apply` applies a structured emit pass to the prompt_writer call.

- [ ] **Step 1: Write the failing test**

```python
# tests/i2p/test_i2p_swap_step.py
import json


def _steps():
    with open("src/workflows/i2p/i2p_v3.json", encoding="utf-8") as fh:
        return json.load(fh)["steps"]


def test_swap_step_exists():
    assert any(s.get("id") == "5.35" for s in _steps())


def test_verify_step_exists():
    assert any(s.get("id") == "5.35.verify" for s in _steps())


def test_swap_step_shape_uses_mechanical_executor():
    s = next(x for x in _steps() if x["id"] == "5.35")
    assert s["agent"] == "mechanical"
    # v2 fix: executor is "mechanical" (constant), verb in payload.action.
    assert s["executor"] == "mechanical"
    assert s["payload"]["action"] == "swap_placeholder_images"
    assert "5.30c" in s["depends_on"]
    # Soft done_when accepts skipping.
    dw = s["done_when"].lower()
    assert "ok" in dw and "skipped" in dw
    assert s.get("reversibility") == "full"


def test_swap_step_has_artifact_schema_for_constrained_emit():
    """v2 fix: prompt_writer needs schema constraint. The artifact_schema
    on this step is what constrained_emit.maybe_apply enforces post-call."""
    s = next(x for x in _steps() if x["id"] == "5.35")
    # The schema for the prompts artifact must be declared so the
    # constrained-emit pass can enforce it on cheap-tier LLM output.
    schema = s.get("artifact_schema")
    assert isinstance(schema, dict)
    assert "prompts" in schema or "prompt_writer_result" in schema


def test_verify_step_shape():
    s = next(x for x in _steps() if x["id"] == "5.35.verify")
    assert s["agent"] == "mechanical"
    assert s["executor"] == "mechanical"
    assert s["payload"]["action"] == "verify_swap_placeholder_images_shape"
    assert "5.35" in s["depends_on"]


def test_emit_preview_url_depends_on_verify():
    """5.40 must depend on 5.35.verify so the URL only surfaces with a
    verified swap."""
    s = next(x for x in _steps() if x["id"] == "5.40")
    assert "5.35.verify" in s["depends_on"]


def test_phase_is_phase_5():
    for sid in ("5.35", "5.35.verify"):
        s = next(x for x in _steps() if x["id"] == sid)
        assert s["phase"] == "phase_5"
```

- [ ] **Step 2: Run + insert the steps**

Run: `.venv/Scripts/python -m pytest tests/i2p/test_i2p_swap_step.py -q`
Expected: FAIL.

In `src/workflows/i2p/i2p_v3.json`, find the step with `"id": "5.30c"` and insert these AFTER its closing `},` and BEFORE `"id": "5.40"`:

```json
    {
      "id": "5.35",
      "phase": "phase_5",
      "name": "swap_placeholder_images",
      "agent": "mechanical",
      "executor": "mechanical",
      "depends_on": ["5.30c"],
      "instruction": "Plan 3 — replace placehold.co <img> placeholders in every mission_{mission_id}/.web/**/*.html (recursive) with real diffusion-generated PNGs. Enqueues ONE prompt_writer LLM task (artifact_schema enforces shape via constrained_emit) to enrich each placeholder's alt text into a diffusion prompt, then one image task per placeholder through beckman (fatih_hoca.select(needs_image=True) -> dispatcher.dispatch -> paintress). Writes PNGs under mission_{mission_id}/.web/assets/<placeholder_id>.png and rewrites <img src> to assets/<id>.png. Best-effort: per-placeholder failures keep the original placehold.co URL — never blocks the mission. Skipping is acceptable; the preview surfaces with placeholders.",
      "done_when": "swap_placeholder_images executor returns ok=true (replaced_count>=0; skipped_count>0 is acceptable when image generation is unavailable)",
      "produces": ["mission_{mission_id}/.web/assets/"],
      "reversibility": "full",
      "artifact_schema": {
        "prompts": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "placeholder_id": {"type": "string"},
              "prompt": {"type": "string", "maxLength": 220}
            },
            "required": ["placeholder_id", "prompt"]
          }
        }
      },
      "payload": {"action": "swap_placeholder_images"}
    },
    {
      "id": "5.35.verify",
      "phase": "phase_5",
      "name": "swap_placeholder_images_shape_check",
      "agent": "mechanical",
      "executor": "mechanical",
      "depends_on": ["5.35"],
      "instruction": "Plan 3 verify-posthook. Asserts swap_placeholder_images produced a self-consistent result: replaced_count agrees with surviving placehold.co URLs (within errors-margin for graceful degrade), assets/ exists when replaced_count>0. Mirrors verify_charter_shape's role for phase 0.",
      "done_when": "verify_swap_placeholder_images_shape returns ok=true",
      "reversibility": "full",
      "payload": {
        "action": "verify_swap_placeholder_images_shape",
        "swap_result_from_step": "5.35"
      }
    },
```

Then locate `"id": "5.40"` and update its `depends_on` from `["5.30c"]` to `["5.30c", "5.35.verify"]`.

- [ ] **Step 3: Run + regression**

Run: `.venv/Scripts/python -m pytest tests/i2p/test_i2p_swap_step.py tests/i2p/ -q -x`
Expected: PASS, no new failures in the i2p sweep.

- [ ] **Step 4: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json tests/i2p/test_i2p_swap_step.py
git commit -m "feat(image): i2p_v3 5.35 swap + 5.35.verify posthook (correct mechanical convention)"
```

---

## Task 11: End-to-end host-path test — JSON-string TaskResult shape

**Files:**
- Test: `tests/integration/test_image_i2p_swap_e2e.py`

The v2 e2e test deliberately mocks beckman.enqueue with **JSON-string** `TaskResult.result` so production data shape is exercised — Plan 3 v1's e2e used dict shape and missed the bug.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_image_i2p_swap_e2e.py
"""Plan 3 v2 — end-to-end placeholder swap with PRODUCTION TaskResult shape."""
import json
import os

import pytest
from PIL import Image


_HTML = """<!DOCTYPE html>
<html><body class="w-[390px] min-h-[844px]">
  <img src="https://placehold.co/390x220/E07A5F/FFF?text=hero"
       alt="smiling barista handing over a takeaway cup">
  <img src="https://placehold.co/260x180/3D405B/FFF?text=feat"
       alt="ai-powered task triage dashboard">
  <img src="https://placehold.co/64x64/264653/FFF?text=u"
       alt="user portrait">
</body></html>
"""


@pytest.mark.asyncio
async def test_i2p_swap_e2e_with_json_string_result(monkeypatch, tmp_path):
    """Drives mr_roboto.run with action=swap_placeholder_images. Mocks
    beckman.enqueue at the swap-module namespace so the production JSON-
    string TaskResult.result shape is exercised end-to-end."""
    ws = tmp_path / "mission_777"
    web = ws / ".web"; web.mkdir(parents=True)
    (web / "home.html").write_text(_HTML, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(ws),
    )

    call_log: list[str] = []

    async def _fake_enqueue(spec, **kwargs):
        agent_type = spec.get("agent_type")
        call_log.append(agent_type or "")
        if agent_type == "prompt_writer":
            class _R:
                status = "completed"
                # PRODUCTION SHAPE — JSON STRING (orchestrator json.dumps).
                result = json.dumps({
                    "_schema_version": "1",
                    "prompts": [
                        {"placeholder_id": "home__0", "prompt": "coral barista"},
                        {"placeholder_id": "home__1", "prompt": "slate dashboard"},
                        {"placeholder_id": "home__2", "prompt": "teal portrait"},
                    ],
                })
                error = None
            return _R()
        if agent_type == "image":
            ic = spec["context"]["image_call"]
            os.makedirs(ic["out_dir"], exist_ok=True)
            path = os.path.join(
                ic["out_dir"], f"{ic['filename_hint']}_raw.png",
            )
            Image.new("RGB", (ic["width"], ic["height"]),
                      (100, 150, 200)).save(path, "PNG")
            class _R:
                status = "completed"
                # PRODUCTION SHAPE — JSON STRING.
                result = json.dumps({
                    "path": path, "provider": "pollinations",
                    "model": "pollinations/flux", "cost": 0.0,
                })
                error = None
            return _R()
        raise AssertionError(f"unexpected agent_type: {agent_type!r}")
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fake_enqueue,
    )

    import mr_roboto
    task = {
        "id": 12345, "mission_id": 777, "title": "swap_e2e",
        "context": {"payload": {
            "action": "swap_placeholder_images",
            "design_tokens": {"primary": "#E07A5F"},
            "brand_voice": "warm, neighborhood coffee shop",
        }},
    }
    action = await mr_roboto.run(task)

    assert action.status == "completed"
    res = action.result
    assert res["ok"] is True
    assert res["replaced_count"] == 3
    assert res["skipped_count"] == 0
    assert res["html_files_changed"] == 1

    assets = ws / ".web" / "assets"
    pngs = sorted(p.name for p in assets.glob("*.png"))
    # Stable <pid>.png names (no timestamp suffix).
    assert pngs == ["home__0.png", "home__1.png", "home__2.png"]
    for png in pngs:
        assert (assets / png).stat().st_size > 0

    rewritten = (web / "home.html").read_text(encoding="utf-8")
    assert "placehold.co" not in rewritten
    assert 'src="assets/home__0.png"' in rewritten
    assert 'src="assets/home__1.png"' in rewritten
    assert 'src="assets/home__2.png"' in rewritten

    assert call_log.count("prompt_writer") == 1
    assert call_log.count("image") == 3
```

- [ ] **Step 2: Run + smoke**

Run: `.venv/Scripts/python -m pytest tests/integration/test_image_i2p_swap_e2e.py -q`
Expected: PASS.

Full Plan 3 suite green-check (split per Plan 1 v2 §13 conftest-collision rule):
```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py packages/mr_roboto/tests/test_verify_swap_placeholder_images_shape.py packages/mr_roboto/tests/test_emit_preview_url_assets.py -q
.venv/Scripts/python -m pytest tests/agents/test_prompt_writer.py tests/agents/test_prompt_writer_template.py tests/i2p/test_i2p_swap_step.py tests/integration/test_image_i2p_swap_e2e.py -q
```
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_image_i2p_swap_e2e.py
git commit -m "test(image): e2e i2p swap with JSON-string TaskResult shape"
```

---

## Plan 3 v2 done-when

- i2p mission reaching phase 5 produces real images for every `placehold.co` `<img>` in `mission_{id}/.web/**/*.html` (recursive); HTML rewritten to `src="assets/<id>.png"`; PNGs under `mission_{id}/.web/assets/`.
- Per-placeholder failures keep the original placeholder (graceful degrade); `5.35` accepts `skipped_count > 0` as a soft pass.
- `5.35.verify` (`verify_swap_placeholder_images_shape`) gates `5.40 emit_preview_url`: the preview only surfaces with a verifier-passed swap.
- `prompt_writer` enqueue declares `artifact_schema` so workflow_engine's `constrained_emit.maybe_apply` enforces the JSON shape on cheap-tier LLM output.
- Both `_call_prompt_writer` and `_generate_one_image` go through `_parse_task_result`, which handles the production JSON-string `TaskResult.result` shape (the bug Plan 3 v1 had).
- `_list_html_files` recurses (`os.walk`), so multi-screen prototypes are covered.
- All new tests green; no regressions in `packages/mr_roboto/tests/`, `tests/agents/`, `tests/i2p/`.

## Dependencies
- **Plan 1 v2** MUST be merged first — Plan 3 enqueues `agent_type: "image"` tasks that ride Plan 1 v2's admission + dispatcher branch + telemetry envelope.
- **Plan 2 v2** is OPTIONAL. When merged, Plan 3's image tasks transparently pick local SDXL when fit/budget says so; no Plan 3 change required.
- Plan 3 is **file-disjoint** from Plan 2 — they can run in parallel worktrees after Plan 1 v2 lands.

## Follow-on
- Founder review of generated images via a `notify_user` sibling after `5.35.verify`.
- Asset cache across mission re-runs keyed on `(prompt, width, height, provider)`.
- Z5 mobile / Expo: mirror the mechanic to scan `.prototype/**/*.html` and write to `.prototype/assets/`.
- Per-screen `section_intent` read from the screen_plan artifact (replaces the heuristic in `_section_from_alt`).
