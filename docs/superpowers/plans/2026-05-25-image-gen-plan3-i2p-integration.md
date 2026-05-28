# Image Generation — Plan 3: i2p integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real image generation wired into the i2p prototype phase — replace `placehold.co` `<img>` placeholders with diffusion-generated PNGs, served back through the existing web-preview host. End-to-end: a mission reaches phase 5, prototype HTML is emitted with `placehold.co` placeholders (current Z1 behaviour), Plan 3's `swap_placeholder_images` mechanic enqueues ONE `prompt_writer` LLM task to enrich each placeholder's `alt` text into a diffusion prompt, enqueues one image task per placeholder through beckman (which selects via Plan 1's image scorer and dispatches via Plan 1's paintress branch), and rewrites the `<img src="...">` to point at the generated asset under `mission_{id}/assets/`.

**Architecture:** Plan 3 is the consumer-side wiring for the design spec §8 "i2p prototype swap" bullet. It depends on Plan 1 (cloud spine: `paintress`, image scorer, dispatcher image branch, beckman image-aware admission, `needs_image=True`). It is INDEPENDENT of Plan 2 (`clair_obscur` + GPU handover) — every image task in Plan 3 flows through Plan 1's cloud path; if Plan 2 is also merged the same beckman→hoca→paintress chain transparently picks local providers when fit/budget says so. Plan 3 is file-disjoint from Plan 2 so the two can run in parallel worktrees.

**Tech Stack:** Python 3.10, async/await, BeautifulSoup4 (already used by `annotate_html_oids` for HTML rewriting), pytest. The new `prompt_writer` agent is a pure-config `BaseAgent` subclass (per `feedback_no_agent_modes` + `project_agents_polish_20260508` — sys_prompt + tools, zero methods). The `swap_placeholder_images` mechanic is a normal mr_roboto executor following the `marketing_copy` precedent (beckman-enqueue from inside a mechanical, await_inline=True).

**Scope boundary (in this plan):**
- `prompt_writer` agent (config-only) + agent-registry wiring.
- Prompt-writing templates for small LLMs (system prompt + few-shot exemplars).
- `swap_placeholder_images` mr_roboto mechanic — scan HTML, enqueue prompt_writer, enqueue N image tasks, rewrite `src`, write assets under `mission_{id}/assets/`.
- i2p_v3.json prototype-phase step that invokes the mechanic (soft `done_when` — skipping is acceptable).
- Web-preview host extension so `/assets/<file>.png` resolves under the mission's assets directory.
- e2e host-path test.

**NOT in this plan:** anything in Plan 1's or Plan 2's territory (see file-ownership table below). No changes to `paintress`, `renoir`, `clair_obscur`, `fatih_hoca`, `llm_dispatcher`, `orchestrator`, `general_beckman`, `nerd_herd`, or `telegram_bot`.

---

## File ownership

**Plan 3 owns (NEW files):**
- `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`
- `packages/mr_roboto/tests/test_swap_placeholder_images.py`
- `src/agents/prompt_writer.py`
- `docs/templates/prompt_writer/diffusion_prompt_template.md` (template + few-shots)
- `tests/integration/test_image_i2p_swap_e2e.py`
- (Possibly a small fixture HTML for tests, kept inline in the test file — no separate fixture file.)

**Plan 3 extends (anchors clearly marked in each task):**
- `packages/mr_roboto/src/mr_roboto/__init__.py` — import + `__all__` + `_run_dispatch` branch.
- `packages/mr_roboto/src/mr_roboto/reversibility.py` — `swap_placeholder_images: "full"` in `VERB_REVERSIBILITY`.
- `packages/mr_roboto/src/mr_roboto/emit_preview_url.py` — extend `_resolve_preview_root` to also expose `mission_{id}/assets/` under the served tree (the static server's `--directory` already serves whatever is at the root; we ensure `assets/` ends up there by copying / linking — see Task 7 for the exact mechanic).
- `src/agents/__init__.py` — register `prompt_writer` in `AGENT_REGISTRY`.
- `src/workflows/i2p/i2p_v3.json` — new step `5.35` (swap_placeholder_images) in phase_5, between `5.30c` (annotate_html_oids) and `5.40` (emit_preview_url).

**Plan 3 must NOT touch (Plan 1 / Plan 2 territory):**
- `packages/paintress/*`, `packages/renoir/*`, `packages/clair_obscur/*`.
- `packages/fatih_hoca/*` (any file).
- `src/core/llm_dispatcher.py`, `src/core/orchestrator.py`.
- `packages/general_beckman/*`, `packages/nerd_herd/*`.
- `src/app/telegram_bot.py`.

If Plan 3 needs to know a value that lives in Plan-1-territory (e.g. the `image_call` context shape, the `needs_image` kwarg), Plan 3 calls beckman.enqueue with the spec's `context.image_call` filled in exactly as Plan 1's `/image` command does — that contract is already public.

---

## Discovered conventions (recorded inline so subsequent tasks don't re-grep)

These were confirmed before writing the plan (so Task 1's "research" step below is verification, not discovery):

1. **Placeholder convention** (from `src/workflows/i2p/i2p_v3.json:5076`, step `5.30a` instruction):
   ```
   Images: PLACEHOLDER ONLY — every <img> src must use
   https://placehold.co/<W>x<H>/<bg>/<fg>?text=<intent>.
   Z2 gorsel_ustasi swaps to real images later. Each <img> MUST carry a
   descriptive alt="..." attribute that describes what the real image
   should show — this becomes the prompt for image gen.
   ```
   So: **placeholder = `<img>` whose `src` matches `^https?://placehold\.co/`**, and **the `alt` attribute is the seed for the diffusion prompt**. We use both: detect by src host, prompt from alt.

2. **Mission workspace helper** (`src/tools/workspace.py:439`):
   ```python
   get_mission_workspace(mission_id: int) -> str
   ```
   Returns absolute `workspace/mission_{id}/`. We write PNGs under `<that>/assets/<name>.png` and rewrite `src` to a path the served preview root sees (see Task 7 for the resolver tweak).

3. **HTML prototype output location**: `mission_{mission_id}/.web/<slug>.html` (from step 5.30a's `produces:`).

4. **Web-preview root resolver** (`packages/mr_roboto/src/mr_roboto/emit_preview_url.py:_resolve_preview_root`): currently picks `<ws>/.prototype/` if `index.html` exists, else `<ws>/.web/` if non-empty. The static server serves that directory as the document root. To make rewritten `<img src="../assets/foo.png">` (or `src="assets/foo.png"`) resolve, we need `assets/` to be reachable from the served root — we put it inside `.web/assets/` (Task 4 writes there). The resolver doesn't need to change for `.web/` paths since `.web/assets/` is already under the served root. For `.prototype/` (Expo bundle, used by Z5), we mirror to `.prototype/assets/`. Task 7 codifies this.

5. **Mr Roboto dispatch shape**: a verb is wired by (a) module import at top of `__init__.py`, (b) appearance in `__all__`, (c) an `if action == "<verb>":` branch in `_run_dispatch` returning an `Action`, (d) an entry in `VERB_REVERSIBILITY`. The `marketing_copy` verb at `__init__.py:102` + `reversibility.py:302` is the closest precedent — same shape: mechanical body that calls `beckman.enqueue` internally for the LLM step, writes artifact to workspace, surfaces founder action. We mirror it.

6. **Agent registry** (`src/agents/__init__.py`): import the class, instantiate in `AGENT_REGISTRY` dict. `signal_classifier.py` is the canonical small/pure-config precedent — sys_prompt + `default_tier="cheap"` + `max_iterations=2` + `allowed_tools=[]` + `enable_self_reflection=False`. We mirror it (single-call, no tools).

7. **Beckman enqueue + await_inline**: from `marketing_copy.py:117-120`:
   ```python
   from general_beckman import enqueue as _enqueue
   return await _enqueue(spec, **kwargs)
   ```
   For a one-shot LLM task we pass `await_inline=True` and read `.result` off the returned `TaskResult` (see `/image` cmd in Plan 1 Task 12 for the result-shape handling).

---

## Task 1: Verify discovered conventions still hold

**Files:** none modified — read-only audit.

This is a SHORT verification task (per the project's "Audit call sites not docstrings" rule). We confirm the four facts the subsequent tasks depend on, recording the exact line numbers we'll anchor edits to.

- [ ] **Step 1: Confirm the placeholder convention**

```bash
.venv/Scripts/python -c "import json, re; d = json.load(open('src/workflows/i2p/i2p_v3.json', encoding='utf-8')); steps = [s for s in d['steps'] if s.get('id') == '5.30a']; print(steps[0]['instruction'][:800])"
```
Expected: output contains `placehold.co` and `descriptive alt`. If the convention has changed, STOP and update the rest of this plan's regex/detection logic in Tasks 4 / 5 before proceeding.

- [ ] **Step 2: Confirm step 5.30c → 5.40 ordering and that step id `5.35` is free**

```bash
.venv/Scripts/python -c "import json; d = json.load(open('src/workflows/i2p/i2p_v3.json', encoding='utf-8')); ids = [s['id'] for s in d['steps']]; print('5.30c' in ids, '5.40' in ids, '5.35' in ids)"
```
Expected: `True True False`. If `5.35` is already taken, pick the next free id between 5.30c and 5.40 (e.g. `5.30d`) and use it consistently in Task 6.

- [ ] **Step 3: Confirm `_resolve_preview_root` shape**

```bash
.venv/Scripts/python -c "from mr_roboto.emit_preview_url import _resolve_preview_root; import inspect; print(inspect.getsource(_resolve_preview_root))"
```
Expected: function body matches the version recorded in "Discovered conventions" §4 — picks `.prototype/index.html` first, then non-empty `.web/`. If the signature/body has drifted, adjust Task 7's edit anchor.

- [ ] **Step 4: Confirm `AGENT_REGISTRY` shape**

```bash
.venv/Scripts/python -c "from src.agents import AGENT_REGISTRY; print(sorted(AGENT_REGISTRY.keys()))"
```
Expected: the listed 21 agents from `project_agents_polish_20260508`. `prompt_writer` must NOT be present yet. If it is, STOP and resolve the collision (someone else added it).

- [ ] **Step 5: No commit**

This is a read-only verification task. Nothing to stage.

---

## Task 2: `prompt_writer` agent (pure config)

**Files:**
- Create: `src/agents/prompt_writer.py`
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_prompt_writer.py`

The agent is single-call (no ReAct iteration), small/local-LLM-friendly, and emits one JSON object mapping placeholder ids to enriched diffusion prompts. Follows the `signal_classifier.py` template — pure config, zero methods.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_prompt_writer.py
"""prompt_writer agent — pure-config invariants + registry wiring."""
import re
from src.agents.prompt_writer import PromptWriterAgent
from src.agents import AGENT_REGISTRY, get_agent


def test_prompt_writer_registered():
    assert "prompt_writer" in AGENT_REGISTRY
    inst = get_agent("prompt_writer")
    assert isinstance(inst, PromptWriterAgent)


def test_prompt_writer_is_pure_config():
    """Per feedback_no_agent_modes + project_agents_polish_20260508."""
    body = open("src/agents/prompt_writer.py", encoding="utf-8").read()
    # No method definitions beyond get_system_prompt (the one allowed shape).
    method_defs = re.findall(r"^    def (\w+)\(", body, flags=re.MULTILINE)
    assert set(method_defs) <= {"get_system_prompt"}, method_defs


def test_prompt_writer_config_fields():
    a = PromptWriterAgent()
    assert a.name == "prompt_writer"
    assert a.default_tier == "cheap"
    assert a.max_iterations == 1  # single-call; no ReAct loop
    assert a.enable_self_reflection is False
    assert a.allowed_tools == []


def test_system_prompt_satisfies_three_invariants():
    """CLAUDE.md: first line `You are ...`, must/always + don't/never,
    final_answer + fenced ```json``` schema."""
    p = PromptWriterAgent().get_system_prompt({})
    first_line = p.strip().splitlines()[0]
    assert first_line.startswith("You are "), first_line
    body = p.lower()
    assert ("must" in body or "always" in body), "missing must/always"
    assert ("don't" in body or "never" in body), "missing don't/never"
    assert "final_answer" in p
    assert "```json" in p
    # The schema mentions placeholder_id + prompt + _schema_version
    assert "placeholder_id" in p
    assert "_schema_version" in p


def test_system_prompt_mentions_template_inputs():
    """Should reference design_tokens / brand_voice / section_intent — the
    three slots the diffusion-prompt template fills."""
    p = PromptWriterAgent().get_system_prompt({})
    for slot in ("design_tokens", "brand_voice", "section_intent"):
        assert slot in p.lower(), f"missing slot reference: {slot}"
```

- [ ] **Step 2: Run test to verify it fails**

```
.venv/Scripts/python -m pytest tests/agents/test_prompt_writer.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agents.prompt_writer'`.

- [ ] **Step 3: Implement the agent**

`src/agents/prompt_writer.py`:

```python
# agents/prompt_writer.py
"""Image-generation prompt writer agent (Plan 3).

For a prototype with N placeholder <img> tags, this agent reads the
design context (design_tokens, brand_voice, section_intent) plus each
placeholder's intent (its `alt` text + size + section role) and emits
ONE structured JSON artifact mapping each placeholder_id to an enriched
diffusion prompt.

Pure config (sys_prompt + tools, zero methods beyond get_system_prompt) —
see feedback_no_agent_modes + project_agents_polish_20260508. Single-call:
no ReAct iteration needed; the response_format=json_schema constraint on
beckman.enqueue keeps small/local LLMs honest about the shape.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.prompt_writer")


class PromptWriterAgent(BaseAgent):
    name = "prompt_writer"
    description = "Turns prototype placeholder intents into diffusion prompts"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 1  # single-call — no ReAct loop
    enable_self_reflection = False
    allowed_tools = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a diffusion-prompt engineer for product UI mockups. "
            "Given a list of placeholder <img> intents from a mobile "
            "prototype, you write ONE enriched diffusion prompt per "
            "placeholder so a text-to-image model can produce the real "
            "asset.\n"
            "\n"
            "## Context you receive\n"
            "- `design_tokens` — color palette + typography from the "
            "mission's design system. Use these to keep generated images "
            "visually consistent with the prototype's chrome.\n"
            "- `brand_voice` — short paragraph describing the product's "
            "voice / mood (founder-set, may be terse). Use to bias subject "
            "matter and tone (calm vs. energetic, premium vs. casual).\n"
            "- `section_intent` — the screen role each placeholder lives "
            "in (hero / feature-illustration / avatar / product-shot / "
            "background / icon). Use to pick composition, framing, "
            "subject distance.\n"
            "- `placeholders` — list of `{placeholder_id, alt, width, "
            "height, section}`. The `alt` text is the seed: it already "
            "tells you what the real image should show. Enrich it with "
            "style descriptors, composition, lighting, and design-token "
            "color cues.\n"
            "\n"
            "## You must\n"
            "- Always emit exactly one prompt per placeholder — every "
            "`placeholder_id` in the input must appear in the output.\n"
            "- Always echo the `placeholder_id` verbatim.\n"
            "- Always keep prompts under 220 characters — diffusion "
            "models truncate; the first words dominate.\n"
            "- Always start each prompt with the subject (the `alt` "
            "intent), then add style + composition + color cues.\n"
            "- Always include at least one `design_tokens` color cue "
            "(e.g. \"warm coral accent\", \"muted slate background\") so "
            "the generated image harmonizes with the surrounding UI.\n"
            "\n"
            "## You must never\n"
            "- Don't invent new `placeholder_id` values that weren't in "
            "the input — the swap mechanic joins on id.\n"
            "- Don't return text outside the JSON envelope. No prose, no "
            "markdown fences around the result, no commentary.\n"
            "- Don't include negative prompts or model-specific tokens "
            "(e.g. `<lora:...>`, `((emphasis))`) — these break some "
            "providers. Plain natural-language only.\n"
            "- Never copy the `alt` text verbatim — enrich it. A "
            "verbatim copy is a failure.\n"
            "\n"
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

- [ ] **Step 4: Register in `AGENT_REGISTRY`**

In `src/agents/__init__.py`, add the import + registry entry. The import line goes alphabetically near the other `from .` imports (after `from .signal_classifier import SignalClassifierAgent`). The `AGENT_REGISTRY` entry goes at the end of the dict (the trailing comma after `"signal_classifier": SignalClassifierAgent(),` keeps the diff small).

```python
# Add to the imports block:
from .prompt_writer import PromptWriterAgent

# Add to AGENT_REGISTRY (last entry, before the closing brace):
    "prompt_writer": PromptWriterAgent(),
```

- [ ] **Step 5: Run test to verify it passes**

```
.venv/Scripts/python -m pytest tests/agents/test_prompt_writer.py -q
```
Expected: PASS (5 passed).

- [ ] **Step 6: Regression — prompt-quality invariants still green**

```
.venv/Scripts/python -m pytest tests/agents/test_prompt_quality.py -q
```
Expected: PASS — the existing 3-invariant test sweep now covers prompt_writer too (it iterates AGENT_REGISTRY). If it fails on prompt_writer, fix the system prompt to satisfy whatever invariant tripped.

- [ ] **Step 7: Commit**

```bash
git add src/agents/prompt_writer.py src/agents/__init__.py tests/agents/test_prompt_writer.py
git commit -m "feat(image): prompt_writer agent (pure-config, single-call)"
```

---

## Task 3: Prompt-writing template + few-shot exemplars

**Files:**
- Create: `docs/templates/prompt_writer/diffusion_prompt_template.md`
- Test: `tests/agents/test_prompt_writer_template.py`

The agent's sys_prompt is the contract. This template is the **scaffolding** the spec §8 calls for — a few-shot block the `swap_placeholder_images` mechanic prepends to the user message when calling small/local LLMs. Co-located with the founder-voice templates (`docs/templates/brand_voices/`) which `marketing_copy.py:_load_brand_voice_doc` already reads from — same shape, same loader pattern.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_prompt_writer_template.py
import os
from src.agents.prompt_writer import load_diffusion_prompt_template


def test_template_loads():
    body = load_diffusion_prompt_template()
    assert body is not None
    assert len(body) > 200


def test_template_has_few_shot_block():
    body = load_diffusion_prompt_template()
    # Each exemplar shows an input → output pair the agent learns from.
    assert "EXAMPLE 1" in body
    assert "EXAMPLE 2" in body
    # At least one example references a design_tokens color cue.
    assert ("coral" in body.lower()
            or "slate" in body.lower()
            or "color" in body.lower())


def test_template_has_slot_placeholders():
    body = load_diffusion_prompt_template()
    for slot in ("{design_tokens}", "{brand_voice}", "{section_intent}",
                 "{placeholders}"):
        assert slot in body, f"missing slot: {slot}"


def test_template_file_lives_under_docs_templates():
    """Co-located with brand_voices/ so the same loader convention applies."""
    assert os.path.isfile(
        "docs/templates/prompt_writer/diffusion_prompt_template.md"
    )
```

- [ ] **Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest tests/agents/test_prompt_writer_template.py -q
```
Expected: FAIL — `ImportError: cannot import name 'load_diffusion_prompt_template'`.

- [ ] **Step 3: Write the template**

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
Brand voice: "warm, neighborhood coffee shop — third-wave, not pretentious"
Design tokens: { primary: "#E07A5F" (warm coral), surface: "#F4F1DE" (cream) }
Expected prompt:
  "Warm candid photo of a smiling young barista handing a takeaway cup to a customer, soft morning light through cafe window, warm coral apron accent against cream-toned interior, shallow depth of field, third-wave coffee shop atmosphere, eye-level wide composition."

EXAMPLE 2
Input placeholder:
  placeholder_id: feature_2_illustration
  alt: "ai-powered task triage dashboard"
  width: 260
  height: 180
  section: feature
Brand voice: "calm, professional productivity tool for solo founders"
Design tokens: { primary: "#3D405B" (slate indigo), accent: "#81B29A" (muted sage) }
Expected prompt:
  "Minimal isometric illustration of a clean dashboard with sorted task cards, slate indigo header bar, muted sage progress accents on a soft white background, flat vector style, calm professional mood, centered composition, no text on screen."

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
  "Friendly close-up portrait headshot of a person against soft deep-teal blurred background, natural diffuse lighting, eye contact, neutral expression, square composition, photographic style, candid not staged."

## Now emit the JSON

Return ONLY the final_answer JSON envelope — no prose, no markdown
fences. Every placeholder_id from the input MUST appear in `prompts`.
Each prompt MUST be <=220 characters and MUST embed at least one
design-token color cue.
```

- [ ] **Step 4: Add the loader to `prompt_writer.py`**

Append to `src/agents/prompt_writer.py` (after the class, module-level):

```python
import os as _os


_DEFAULT_TEMPLATE_PATH = "docs/templates/prompt_writer/diffusion_prompt_template.md"


def load_diffusion_prompt_template(path: str | None = None) -> str | None:
    """Load the diffusion-prompt few-shot template.

    Returns the raw template text with {design_tokens}, {brand_voice},
    {section_intent}, {placeholders} slots un-substituted. Returns None
    if the file is missing (graceful degrade — agent still works on its
    sys_prompt alone).
    """
    p = path or _DEFAULT_TEMPLATE_PATH
    if not _os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None
```

- [ ] **Step 5: Run test to verify it passes**

```
.venv/Scripts/python -m pytest tests/agents/test_prompt_writer_template.py -q
```
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add docs/templates/prompt_writer/diffusion_prompt_template.md src/agents/prompt_writer.py tests/agents/test_prompt_writer_template.py
git commit -m "feat(image): prompt_writer template + few-shot exemplars"
```

---

## Task 4: `swap_placeholder_images` mechanic — scaffold + placeholder scan

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`
- Create: `packages/mr_roboto/tests/test_swap_placeholder_images.py`

This task builds the mechanic's skeleton + the placeholder-scanning pass. Subsequent tasks wire the prompt_writer call (Task 5) and the image-task fanout + HTML rewrite (Task 6).

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_swap_placeholder_images.py
import os
import pytest
from mr_roboto.swap_placeholder_images import (
    swap_placeholder_images,
    _scan_placeholders,
    _PLACEHOLDER_HOST_RE,
)


# -- Pure-function tests (no I/O) -------------------------------------------

def test_placeholder_host_regex():
    assert _PLACEHOLDER_HOST_RE.search("https://placehold.co/64x64/eee/333?text=x")
    assert _PLACEHOLDER_HOST_RE.search("http://placehold.co/256x256")
    # Real asset (already swapped) — must NOT match.
    assert not _PLACEHOLDER_HOST_RE.search("/assets/hero_1.png")
    assert not _PLACEHOLDER_HOST_RE.search("assets/hero_1.png")
    assert not _PLACEHOLDER_HOST_RE.search("https://example.com/real.png")


_HTML_THREE_PLACEHOLDERS = """<!DOCTYPE html>
<html><body class="w-[390px] min-h-[844px]">
  <img src="https://placehold.co/390x220/E07A5F/FFF?text=hero"
       alt="smiling barista handing over a takeaway cup">
  <img src="https://placehold.co/260x180/3D405B/FFF?text=feat"
       alt="ai-powered task triage dashboard">
  <img src="/assets/already_real.png" alt="something already swapped">
  <img src="https://placehold.co/64x64/264653/FFF?text=u"
       alt="user portrait">
</body></html>"""


def test_scan_placeholders_finds_three_not_four(tmp_path):
    p = tmp_path / "home.html"
    p.write_text(_HTML_THREE_PLACEHOLDERS, encoding="utf-8")
    hits = _scan_placeholders(str(p))
    assert len(hits) == 3
    # Each hit carries placeholder_id, alt, width, height, original src.
    ids = {h["placeholder_id"] for h in hits}
    assert all(h["alt"] for h in hits)
    assert all(h["width"] > 0 and h["height"] > 0 for h in hits)
    # placeholder_id is deterministic from file slug + occurrence index.
    assert ids == {"home__0", "home__1", "home__2"}


def test_scan_placeholders_handles_no_html(tmp_path):
    assert _scan_placeholders(str(tmp_path / "missing.html")) == []


def test_scan_placeholders_handles_zero_placeholders(tmp_path):
    p = tmp_path / "empty.html"
    p.write_text("<!DOCTYPE html><html><body>no images</body></html>",
                 encoding="utf-8")
    assert _scan_placeholders(str(p)) == []


# -- Integration test (with mocked enqueue) ---------------------------------

@pytest.mark.asyncio
async def test_swap_placeholder_images_no_html_files(monkeypatch, tmp_path):
    """Mission workspace exists but .web/ is empty → ok=True with 0 replaced."""
    web_dir = tmp_path / ".web"
    web_dir.mkdir()
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )
    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["replaced_count"] == 0
    assert res["skipped_count"] == 0
    assert res["html_files_seen"] == 0
```

- [ ] **Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'mr_roboto.swap_placeholder_images'`.

- [ ] **Step 3: Implement scaffold + scanner**

`packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`:

```python
"""Plan 3 — swap placehold.co <img> tags for real diffusion-generated PNGs.

Pipeline (Z1 phase 5 mechanic):
  1. Read every HTML file under the mission's `.web/` directory.
  2. Scan each file for <img> tags whose src is a placehold.co URL — these
     are the Z1 placeholders (per i2p_v3.json step 5.30a contract).
  3. Enqueue ONE prompt_writer LLM task (beckman.enqueue, await_inline)
     passing the design context + the full placeholder list. Receive a
     placeholder_id → enriched diffusion prompt map.
  4. For each placeholder: enqueue one image task via beckman.enqueue
     (context.image_call.raw_dispatch=True; beckman calls
     fatih_hoca.select(needs_image=True); dispatcher routes to paintress;
     paintress writes the PNG and returns its path).
  5. Move each generated PNG under `mission_{id}/.web/assets/` (relative
     to the served preview root) and rewrite the original <img src> to
     `assets/<filename>.png`.
  6. Graceful degrade: if a single placeholder's image task fails, keep
     the original placehold.co URL for that slot — never fail the whole
     step.

Mirrors marketing_copy.py's mechanical shape: a non-LLM verb body that
internally enqueues LLM + image work through beckman, never calls the
dispatcher or HK directly (feedback_singular_dispatcher_caller).

Reversibility: `full` — all writes are under the mission workspace and
git-reversible. No external publish. Generated PNGs are regenerable.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.swap_placeholder_images")


# ── Placeholder detection ──────────────────────────────────────────────

_PLACEHOLDER_HOST_RE = re.compile(
    r"^https?://placehold\.co/", re.IGNORECASE
)

# <img ...> matcher (single-line tag bodies). Mirrors verify_html_prototype_shape.
_IMG_RE = re.compile(r"<img\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(
    r'(\b[a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*"([^"]*)"',
)
# Pull width/height out of the placehold.co URL: e.g. ".../390x220/..."
_DIM_RE = re.compile(r"/(\d{2,4})x(\d{2,4})", re.IGNORECASE)


def _parse_attrs(tag_inner: str) -> dict[str, str]:
    return {k.lower(): v for k, v in _ATTR_RE.findall(tag_inner)}


def _slug_from_path(html_path: str) -> str:
    return Path(html_path).stem  # "home.html" -> "home"


def _section_from_alt(alt: str) -> str:
    """Cheap heuristic — section role inferred from alt text wording.

    The agent uses this as a hint; the real section is part of the alt
    intent already. We default to 'feature' (the most common case).
    """
    a = (alt or "").lower()
    if "hero" in a or "header" in a or "banner" in a:
        return "hero"
    if "avatar" in a or "portrait" in a or "headshot" in a:
        return "avatar"
    if "icon" in a:
        return "icon"
    if "background" in a or "bg" in a:
        return "background"
    return "feature"


def _scan_placeholders(html_path: str) -> list[dict[str, Any]]:
    """Return per-placeholder records for one HTML file. [] if file missing."""
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
        if dim_m:
            w, h = int(dim_m.group(1)), int(dim_m.group(2))
        else:
            w, h = 512, 512
        out.append({
            "placeholder_id": f"{slug}__{occ}",
            "alt": alt,
            "width": w,
            "height": h,
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
    """Where generated PNGs live (inside the served preview root)."""
    p = os.path.join(workspace_path, ".web", "assets")
    os.makedirs(p, exist_ok=True)
    return p


def _list_html_files(workspace_path: str) -> list[str]:
    root = _web_root(workspace_path)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if name.lower().endswith(".html"):
            out.append(os.path.join(root, name))
    return out


# ── Main entry ─────────────────────────────────────────────────────────

async def swap_placeholder_images(
    mission_id: int,
    workspace_path: str | None = None,
    design_tokens: dict | None = None,
    brand_voice: str | None = None,
) -> dict[str, Any]:
    """Scan mission HTML, generate real images for placehold.co <img>s,
    rewrite src. Best-effort: per-placeholder failures keep the original.

    Returns:
        {
          "ok": bool,
          "replaced_count": int,
          "skipped_count": int,
          "html_files_seen": int,
          "html_files_changed": int,
          "errors": list[str],
        }
    """
    # Lazy import — keeps scanner-only tests free of workspace deps.
    from src.tools.workspace import get_mission_workspace

    workspace_path = workspace_path or get_mission_workspace(int(mission_id))
    logger.info(
        "swap_placeholder_images: starting",
        mission_id=mission_id,
        workspace_path=workspace_path,
    )

    html_files = _list_html_files(workspace_path)
    if not html_files:
        return {
            "ok": True,
            "replaced_count": 0,
            "skipped_count": 0,
            "html_files_seen": 0,
            "html_files_changed": 0,
            "errors": [],
        }

    # Scan every HTML; collect placeholders.
    all_placeholders: list[dict[str, Any]] = []
    for h in html_files:
        all_placeholders.extend(_scan_placeholders(h))

    if not all_placeholders:
        logger.info(
            "swap_placeholder_images: no placeholders found in %d html files",
            len(html_files),
        )
        return {
            "ok": True,
            "replaced_count": 0,
            "skipped_count": 0,
            "html_files_seen": len(html_files),
            "html_files_changed": 0,
            "errors": [],
        }

    # Task 5 fills in the prompt_writer enqueue + result handling.
    # Task 6 fills in the per-placeholder image fanout + HTML rewrite.
    # For now (Task 4 scaffold), we return the scan as-is.
    return {
        "ok": True,
        "replaced_count": 0,
        "skipped_count": len(all_placeholders),
        "html_files_seen": len(html_files),
        "html_files_changed": 0,
        "placeholders": all_placeholders,  # debug breadcrumb; Task 6 removes
        "errors": [],
    }
```

- [ ] **Step 4: Run test to verify it passes**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q
```
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py packages/mr_roboto/tests/test_swap_placeholder_images.py
git commit -m "feat(image): swap_placeholder_images scaffold + scanner"
```

---

## Task 5: Wire the `prompt_writer` call into the mechanic

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`
- Modify: `packages/mr_roboto/tests/test_swap_placeholder_images.py`

The mechanic enqueues ONE prompt_writer task through beckman (await_inline) carrying the design context + the placeholder list. Receives a `placeholder_id → prompt` map. Mirrors `marketing_copy.py`'s `enqueue` wrapper for test-patchability.

- [ ] **Step 1: Extend the failing test**

Append to `packages/mr_roboto/tests/test_swap_placeholder_images.py`:

```python
# -- Task 5: prompt_writer enqueue --------------------------------------

@pytest.mark.asyncio
async def test_swap_calls_prompt_writer_once(monkeypatch, tmp_path):
    """One prompt_writer task per swap call, regardless of placeholder count."""
    web = tmp_path / ".web"
    web.mkdir()
    (web / "home.html").write_text(_HTML_THREE_PLACEHOLDERS, encoding="utf-8")

    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )

    captured: list[dict] = []

    class _FakeResult:
        status = "completed"
        result = {
            "_schema_version": "1",
            "prompts": [
                {"placeholder_id": "home__0", "prompt": "coral barista scene"},
                {"placeholder_id": "home__1", "prompt": "slate dashboard"},
                {"placeholder_id": "home__2", "prompt": "teal portrait"},
            ],
        }

    async def _fake_enqueue(spec, **kwargs):
        captured.append({"spec": spec, "kwargs": kwargs})
        return _FakeResult()

    # Patch in the swap module's namespace (mirrors marketing_copy's
    # local `enqueue` wrapper).
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman",
        _fake_enqueue,
    )
    # Stub the image fanout so Task 5 can be tested in isolation. Task 6
    # replaces the stub with the real fanout.
    async def _fake_fanout(workspace_path, placeholders, prompt_map):
        return {"replaced": 0, "skipped": len(placeholders), "errors": []}
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._fanout_and_rewrite",
        _fake_fanout,
    )

    res = await swap_placeholder_images(
        mission_id=42,
        design_tokens={"primary": "#E07A5F"},
        brand_voice="warm, neighborhood coffee shop",
    )

    assert len(captured) == 1, "prompt_writer must be enqueued exactly once"
    spec = captured[0]["spec"]
    assert spec["agent_type"] == "prompt_writer"
    # Context carries the design inputs + the placeholder list.
    ctx_payload = spec.get("context", {})
    assert "design_tokens" in str(ctx_payload)
    assert "warm, neighborhood" in str(ctx_payload)
    # await_inline so the mechanic can read the result synchronously.
    assert captured[0]["kwargs"].get("await_inline") is True


@pytest.mark.asyncio
async def test_swap_handles_prompt_writer_failure_gracefully(
    monkeypatch, tmp_path,
):
    """If prompt_writer fails, the whole step degrades to 'skipped' but ok=True."""
    web = tmp_path / ".web"
    web.mkdir()
    (web / "home.html").write_text(_HTML_THREE_PLACEHOLDERS, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )

    class _FailResult:
        status = "failed"
        result = None
        error = "LLM down"

    async def _fail_enqueue(spec, **kwargs):
        return _FailResult()

    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fail_enqueue,
    )

    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True  # never hard-fail the step
    assert res["replaced_count"] == 0
    assert res["skipped_count"] == 3
    assert any("prompt_writer" in e for e in res["errors"])
```

- [ ] **Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q
```
Expected: FAIL on the two new tests (`_enqueue_beckman`/`_fanout_and_rewrite` attrs don't exist).

- [ ] **Step 3: Add the prompt_writer call**

Append to `swap_placeholder_images.py` (above `swap_placeholder_images` async def):

```python
# ── beckman wrappers (test-patchable) ─────────────────────────────────

async def _enqueue_beckman(spec: dict, **kwargs):
    """Thin wrapper; mirrors marketing_copy's `enqueue` for test patching."""
    from general_beckman import enqueue as _enqueue
    return await _enqueue(spec, **kwargs)


async def _call_prompt_writer(
    *,
    mission_id: int,
    placeholders: list[dict[str, Any]],
    design_tokens: dict | None,
    brand_voice: str | None,
) -> dict[str, str] | None:
    """Enqueue one prompt_writer task and return placeholder_id -> prompt map.

    Returns None on any failure (caller degrades gracefully).
    """
    # Strip internal-only fields before showing the agent.
    visible = [
        {
            "placeholder_id": p["placeholder_id"],
            "alt": p["alt"],
            "width": p["width"],
            "height": p["height"],
            "section": p["section"],
        }
        for p in placeholders
    ]

    # Optional few-shot scaffold (graceful degrade if missing).
    template_text = None
    try:
        from src.agents.prompt_writer import load_diffusion_prompt_template
        template_text = load_diffusion_prompt_template()
    except Exception:
        template_text = None

    spec = {
        "title": f"prompt_writer:mission#{mission_id}",
        "description": (
            "Enrich placeholder <img> intents into diffusion prompts for "
            "real image generation."
        ),
        "agent_type": "prompt_writer",
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
            getattr(result, "status", ""),
            getattr(result, "error", ""),
        )
        return None

    raw = getattr(result, "result", None)
    # Mirror marketing_copy._parse_llm_result loosely: accept dict-with-
    # prompts, or a wrapped {result: {prompts: ...}}.
    if isinstance(raw, dict):
        prompts = raw.get("prompts")
        if prompts is None and isinstance(raw.get("result"), dict):
            prompts = raw["result"].get("prompts")
    else:
        prompts = None

    if not isinstance(prompts, list):
        logger.warning("prompt_writer returned no prompts list (raw=%r)", raw)
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


# ── Image fanout + HTML rewrite (Task 6 fills this in) ────────────────

async def _fanout_and_rewrite(
    workspace_path: str,
    placeholders: list[dict[str, Any]],
    prompt_map: dict[str, str],
) -> dict[str, Any]:
    """Task 6 implements this. Stub for Task 5 testability."""
    return {"replaced": 0, "skipped": len(placeholders), "errors": []}
```

Then replace the body of `swap_placeholder_images` (the "Task 5 fills in" stub) with the real flow:

```python
    # ── Step 1: enqueue prompt_writer for the whole batch ──────────────
    prompt_map = await _call_prompt_writer(
        mission_id=int(mission_id),
        placeholders=all_placeholders,
        design_tokens=design_tokens,
        brand_voice=brand_voice,
    )
    if prompt_map is None:
        return {
            "ok": True,  # graceful degrade — never hard-fail
            "replaced_count": 0,
            "skipped_count": len(all_placeholders),
            "html_files_seen": len(html_files),
            "html_files_changed": 0,
            "errors": ["prompt_writer task did not return a usable prompt map"],
        }

    # ── Step 2: per-placeholder image fanout + HTML rewrite ────────────
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

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q
```
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py packages/mr_roboto/tests/test_swap_placeholder_images.py
git commit -m "feat(image): swap mechanic enqueues prompt_writer"
```

---

## Task 6: Per-placeholder image fanout + HTML rewrite

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py`
- Modify: `packages/mr_roboto/tests/test_swap_placeholder_images.py`

The mechanic enqueues one image task per placeholder (each carrying `context.image_call.raw_dispatch=True` so beckman routes via Plan 1's `needs_image=True` admission path). On success, the PNG path is moved under `<ws>/.web/assets/` (inside the served preview root), and the HTML `src` attribute is rewritten to the relative path `assets/<filename>.png`. On per-placeholder failure, the original `placehold.co` URL stays.

- [ ] **Step 1: Extend the failing test**

Append to `packages/mr_roboto/tests/test_swap_placeholder_images.py`:

```python
# -- Task 6: image fanout + rewrite -------------------------------------

@pytest.mark.asyncio
async def test_full_swap_writes_assets_and_rewrites_html(
    monkeypatch, tmp_path,
):
    """End-to-end (with mocked enqueue): 3 placeholders → 3 PNGs under
    .web/assets/, HTML has 3 rewritten src attrs, 1 untouched real src."""
    web = tmp_path / ".web"
    web.mkdir()
    (web / "home.html").write_text(_HTML_THREE_PLACEHOLDERS, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )

    class _OK:
        status = "completed"
        result = {
            "_schema_version": "1",
            "prompts": [
                {"placeholder_id": "home__0", "prompt": "coral barista scene"},
                {"placeholder_id": "home__1", "prompt": "slate dashboard"},
                {"placeholder_id": "home__2", "prompt": "teal portrait"},
            ],
        }

    image_call_count = {"n": 0}

    async def _fake_enqueue(spec, **kwargs):
        # First call: prompt_writer (no image_call in context)
        if spec.get("agent_type") == "prompt_writer":
            return _OK()
        # Subsequent calls: image tasks. paintress would have written a PNG;
        # we simulate by writing a real PNG file to the requested out_dir
        # and returning a TaskResult whose .result has the path.
        ic = spec["context"]["image_call"]
        from PIL import Image
        os.makedirs(ic["out_dir"], exist_ok=True)
        idx = image_call_count["n"]
        image_call_count["n"] += 1
        png_path = os.path.join(ic["out_dir"], f"gen_{idx}.png")
        Image.new("RGB", (ic["width"], ic["height"]), (100, 150, 200)).save(
            png_path, "PNG"
        )

        class _ImgResult:
            status = "completed"
            result = {
                "path": png_path,
                "provider": "pollinations",
                "model": "pollinations/flux",
            }
        return _ImgResult()

    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fake_enqueue,
    )

    res = await swap_placeholder_images(
        mission_id=42,
        design_tokens={"primary": "#E07A5F"},
        brand_voice="warm, neighborhood coffee shop",
    )

    assert res["ok"] is True
    assert res["replaced_count"] == 3
    assert res["skipped_count"] == 0
    assert res["html_files_changed"] == 1

    # 3 PNGs landed under .web/assets/ (the served preview root + assets/).
    assets_dir = tmp_path / ".web" / "assets"
    pngs = sorted(p.name for p in assets_dir.glob("*.png"))
    assert len(pngs) == 3

    # The HTML now references the 3 PNGs and still has the 1 already-real src.
    rewritten = (web / "home.html").read_text(encoding="utf-8")
    assert "placehold.co" not in rewritten
    assert rewritten.count('src="assets/') == 3
    assert "/assets/already_real.png" in rewritten


@pytest.mark.asyncio
async def test_swap_per_image_failure_keeps_placeholder(monkeypatch, tmp_path):
    """If image generation fails for placeholder N, that <img> keeps its
    placehold.co URL; the others still get swapped."""
    web = tmp_path / ".web"
    web.mkdir()
    (web / "home.html").write_text(_HTML_THREE_PLACEHOLDERS, encoding="utf-8")
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace", lambda mid: str(tmp_path)
    )

    class _OK:
        status = "completed"
        result = {
            "_schema_version": "1",
            "prompts": [
                {"placeholder_id": "home__0", "prompt": "p0"},
                {"placeholder_id": "home__1", "prompt": "p1"},
                {"placeholder_id": "home__2", "prompt": "p2"},
            ],
        }

    n = {"i": 0}

    async def _flaky_enqueue(spec, **kwargs):
        if spec.get("agent_type") == "prompt_writer":
            return _OK()
        ic = spec["context"]["image_call"]
        idx = n["i"]
        n["i"] += 1
        if idx == 1:
            class _Fail:
                status = "failed"
                result = None
                error = "provider rate-limit"
            return _Fail()
        from PIL import Image
        os.makedirs(ic["out_dir"], exist_ok=True)
        path = os.path.join(ic["out_dir"], f"gen_{idx}.png")
        Image.new("RGB", (ic["width"], ic["height"]), (100, 150, 200)).save(
            path, "PNG"
        )

        class _ImgResult:
            status = "completed"
            result = {"path": path, "provider": "pollinations"}
        return _ImgResult()

    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _flaky_enqueue,
    )

    res = await swap_placeholder_images(mission_id=42)
    assert res["ok"] is True
    assert res["replaced_count"] == 2
    assert res["skipped_count"] == 1
    # 2 placehold.co srcs replaced, 1 still present.
    rewritten = (web / "home.html").read_text(encoding="utf-8")
    assert rewritten.count("placehold.co") == 1
    assert rewritten.count('src="assets/') == 2
```

- [ ] **Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q
```
Expected: FAIL — fanout still returns the stub `{replaced: 0, ...}`.

- [ ] **Step 3: Implement `_fanout_and_rewrite`**

Replace the stub `_fanout_and_rewrite` in `swap_placeholder_images.py`:

```python
async def _generate_one_image(
    *,
    mission_id_for_log: str,
    placeholder: dict[str, Any],
    prompt: str,
    out_dir: str,
) -> str | None:
    """Enqueue one image task; return the generated PNG path or None on fail."""
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
        logger.warning(
            "image enqueue raised for %s: %s", pid, exc,
        )
        return None

    if getattr(result, "status", "") != "completed":
        logger.info(
            "image task for %s did not complete (status=%r)",
            pid, getattr(result, "status", ""),
        )
        return None

    raw = getattr(result, "result", None)
    path = None
    if isinstance(raw, dict):
        path = raw.get("path") or raw.get("content")
    if not (path and os.path.isfile(path)):
        logger.warning("image task for %s returned no usable path", pid)
        return None
    return path


def _move_into_assets(src_path: str, assets_dir: str, placeholder_id: str) -> str:
    """Move (or copy) the generated PNG into the served-root assets dir.

    Returns the final filename (basename) inside assets_dir.
    """
    os.makedirs(assets_dir, exist_ok=True)
    # Stable filename per placeholder so re-runs overwrite rather than pile up.
    final_name = f"{placeholder_id}.png"
    final_path = os.path.join(assets_dir, final_name)
    try:
        # If src is already inside the assets dir, no-op.
        if os.path.abspath(src_path) == os.path.abspath(final_path):
            return final_name
        # Atomic-ish replace.
        if os.path.exists(final_path):
            os.remove(final_path)
        try:
            os.replace(src_path, final_path)
        except OSError:
            # Cross-device or permission — fall back to copy.
            shutil.copyfile(src_path, final_path)
    except OSError as exc:
        logger.warning("move-to-assets failed for %s: %s", placeholder_id, exc)
        # Best-effort: copy in place if move failed.
        try:
            shutil.copyfile(src_path, final_path)
        except OSError:
            return ""
    return final_name


def _rewrite_html_srcs(
    html_path: str, rewrites: dict[tuple[int, int], str],
) -> bool:
    """Replace tag bodies at (start, end) spans with new <img> tags whose src
    has been swapped. Returns True if the file was modified."""
    if not rewrites:
        return False
    try:
        with open(html_path, encoding="utf-8") as fh:
            html = fh.read()
    except OSError:
        return False

    # Apply replacements RIGHT-TO-LEFT so earlier spans stay valid.
    ordered = sorted(rewrites.items(), key=lambda kv: kv[0][0], reverse=True)
    changed = False
    for (start, end), new_src in ordered:
        old_tag = html[start:end]
        new_tag = _IMG_RE.sub(
            lambda m: _swap_src_in_tag(m.group(0), new_src),
            old_tag,
            count=1,
        )
        if new_tag != old_tag:
            html = html[:start] + new_tag + html[end:]
            changed = True

    if changed:
        tmp = html_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(html)
        os.replace(tmp, html_path)
    return changed


def _swap_src_in_tag(tag: str, new_src: str) -> str:
    """Replace the src="..." attribute value inside a single <img> tag."""
    return re.sub(
        r'src\s*=\s*"[^"]*"',
        f'src="{new_src}"',
        tag,
        count=1,
        flags=re.IGNORECASE,
    )


async def _fanout_and_rewrite(
    workspace_path: str,
    placeholders: list[dict[str, Any]],
    prompt_map: dict[str, str],
) -> dict[str, Any]:
    """For each placeholder, enqueue an image task, move the PNG into
    .web/assets/, and rewrite the <img src> in the source HTML.

    Sequential (not parallel) — beckman + fatih_hoca already pace the
    queue; firing N tasks in parallel would defeat the swap-budget +
    cloud-capacity guards. A 10-image batch takes longer but is correct.
    """
    assets_dir = _assets_dir(workspace_path)
    out_dir_for_gen = assets_dir  # paintress writes here directly

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
            mission_id_for_log=str(workspace_path),
            placeholder=ph,
            prompt=prompt,
            out_dir=out_dir_for_gen,
        )
        if not path:
            skipped += 1
            errors.append(f"image gen failed for {pid}")
            continue

        final_name = _move_into_assets(path, assets_dir, pid)
        if not final_name:
            skipped += 1
            errors.append(f"move-to-assets failed for {pid}")
            continue

        # Path the rewritten <img src> points at — relative to .web/
        # (the served preview root). The HTML lives at .web/<slug>.html
        # so a sibling-relative `assets/<file>.png` resolves correctly.
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
        "replaced": replaced,
        "skipped": skipped,
        "html_files_changed": files_changed,
        "errors": errors,
    }
```

Also remove the temporary `"placeholders": all_placeholders` debug breadcrumb from `swap_placeholder_images`'s no-placeholder return (Task 4 left it for testability — Task 6 trims it; the two earlier tests don't assert on it).

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py -q
```
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py packages/mr_roboto/tests/test_swap_placeholder_images.py
git commit -m "feat(image): swap mechanic fanout + HTML src rewrite"
```

---

## Task 7: Web-preview root resolver accepts the assets subdir

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/emit_preview_url.py`
- Test: `packages/mr_roboto/tests/test_emit_preview_url_assets.py`

The current `_resolve_preview_root` picks `.prototype/` if `index.html` exists, else `.web/` if non-empty. Since Plan 3's fanout writes assets to `<ws>/.web/assets/`, those files are already inside the served root — `assets/<file>.png` resolves automatically when the resolver picks `.web/`. The only fix needed: ensure `.web/` is still picked when it contains ONLY the auto-generated `assets/` subdir (an edge case if the html files vanish but assets remain — paranoia), and add a corresponding mirror for the `.prototype/` path used by Z5 (Expo). For `.prototype/` the html prototype already comes from a different generator, so swap_placeholder_images only ever touches `.web/`. Adding the Z5 mirror is OUT of scope for Plan 3 — leave Z5 to its own integration when image gen reaches Expo.

So the only real change here is documentation + a test confirming that `<ws>/.web/assets/foo.png` is accessible under the resolved root. We don't change resolver logic — we test the existing behaviour holds.

- [ ] **Step 1: Write the verification test**

```python
# packages/mr_roboto/tests/test_emit_preview_url_assets.py
"""Plan 3 — confirm rewritten <img src="assets/..."> resolves under the
preview root resolver. We don't change resolver logic, only verify the
existing `.web/` pick exposes `.web/assets/` automatically."""
import os
from mr_roboto.emit_preview_url import _resolve_preview_root


def test_web_root_with_assets_subdir(tmp_path):
    web = tmp_path / ".web"
    web.mkdir()
    (web / "home.html").write_text("<html></html>", encoding="utf-8")
    assets = web / "assets"
    assets.mkdir()
    (assets / "home__0.png").write_bytes(b"\x89PNG\r\n\x1a\nFAKE")

    root = _resolve_preview_root(str(tmp_path))
    assert root == str(web)
    # The generated PNG is reachable as <root>/assets/home__0.png — that's
    # what the rewritten <img src="assets/home__0.png"> will hit when the
    # static HTTP server's --directory is the resolved root.
    assert os.path.isfile(os.path.join(root, "assets", "home__0.png"))


def test_web_root_with_only_assets(tmp_path):
    """Paranoia: if html files vanish but assets remain, we still resolve."""
    web = tmp_path / ".web"
    web.mkdir()
    (web / "assets").mkdir()
    (web / "assets" / "ghost.png").write_bytes(b"\x89PNG")

    root = _resolve_preview_root(str(tmp_path))
    # Current resolver picks `.web/` when it's a non-empty dir — assets/
    # alone qualifies.
    assert root == str(web)
```

- [ ] **Step 2: Run the test**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_emit_preview_url_assets.py -q
```
Expected: PASS (2 passed) — the existing resolver already supports this. If it fails, the resolver has changed since Plan 3 was written; fix `_resolve_preview_root` to accept `.web/` with only an `assets/` subdir (current logic does — `os.listdir(web)` returns `["assets"]` which is truthy).

- [ ] **Step 3: Add inline note in `emit_preview_url.py`**

In `packages/mr_roboto/src/mr_roboto/emit_preview_url.py`, find `_resolve_preview_root` (line ~44) and append to its docstring:

```python
def _resolve_preview_root(workspace_path: str) -> str | None:
    """Return the directory to serve, or None if nothing is ready.

    Priority:
    1. ``<ws>/.prototype/index.html`` exists → return ``<ws>/.prototype``
    2. ``<ws>/.web`` is a non-empty directory → return ``<ws>/.web``
    3. None

    Plan 3 note: when the i2p ``swap_placeholder_images`` mechanic has
    run, ``<ws>/.web/assets/`` holds the generated PNGs the rewritten
    HTML references as ``src="assets/<file>.png"``. These are served
    automatically by the static HTTP server since ``.web/`` is the
    resolved root — no resolver change required.
    """
```

(This is documentation-only; the body stays identical.)

- [ ] **Step 4: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/emit_preview_url.py packages/mr_roboto/tests/test_emit_preview_url_assets.py
git commit -m "docs(image): preview resolver serves .web/assets/ for swapped images"
```

---

## Task 8: Wire `swap_placeholder_images` into mr_roboto's dispatcher

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py`
- Modify: `packages/mr_roboto/src/mr_roboto/reversibility.py`
- Test: `packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py`

Standard mr_roboto wiring: top-of-file import, add to `__all__`, register reversibility, add `if action == "swap_placeholder_images":` branch in `_run_dispatch`.

- [ ] **Step 1: Write the failing test**

```python
# packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py
"""mr_roboto dispatch wiring for swap_placeholder_images."""
import pytest
import mr_roboto
from mr_roboto.reversibility import VERB_REVERSIBILITY


def test_verb_registered_in_reversibility():
    assert "swap_placeholder_images" in VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["swap_placeholder_images"] == "full"


def test_module_exports_executor():
    assert hasattr(mr_roboto, "swap_placeholder_images")
    assert "swap_placeholder_images" in mr_roboto.__all__


@pytest.mark.asyncio
async def test_dispatch_routes_swap_action(monkeypatch):
    captured = {}

    async def _fake_swap(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "replaced_count": 2,
            "skipped_count": 1,
            "html_files_seen": 1,
            "html_files_changed": 1,
            "errors": [],
        }

    # Patch the executor at the dispatch-site lookup, NOT on the module
    # (the dispatch branch imports from mr_roboto.swap_placeholder_images
    # at call time).
    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images.swap_placeholder_images",
        _fake_swap,
    )

    task = {
        "id": 100,
        "mission_id": 42,
        "title": "swap_placeholder_images_test",
        "context": {
            "payload": {
                "action": "swap_placeholder_images",
                "design_tokens": {"primary": "#E07A5F"},
                "brand_voice": "warm",
            },
        },
    }
    res = await mr_roboto.run(task)
    assert res.status == "completed"
    assert res.result["replaced_count"] == 2
    assert captured["mission_id"] == 42
    assert captured["design_tokens"] == {"primary": "#E07A5F"}
    assert captured["brand_voice"] == "warm"
```

- [ ] **Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py -q
```
Expected: FAIL — `swap_placeholder_images` not in `VERB_REVERSIBILITY`, not in `__all__`, dispatch returns "unknown action".

- [ ] **Step 3: Wire reversibility**

In `packages/mr_roboto/src/mr_roboto/reversibility.py`, add an entry near the `marketing_copy` entry (line ~302). Keep alphabetical groupings reasonable — under the "Local artifact emitters" or after `marketing_copy`:

```python
    # ---- Plan 3 — i2p image-gen integration ---------------------------------
    # swap_placeholder_images: writes PNGs under <ws>/.web/assets/ and rewrites
    # <img src> in <ws>/.web/*.html — all under the mission workspace,
    # git-reversible, no external publish. Generated PNGs are regenerable
    # (re-run swaps them again).
    "swap_placeholder_images": "full",
```

- [ ] **Step 4: Wire the import + `__all__` + dispatch branch**

In `packages/mr_roboto/src/mr_roboto/__init__.py`:

- Add the import after the other Z-series mechanical imports (near the `marketing_copy` line ~102):
  ```python
  # Plan 3 — i2p image-gen integration
  from mr_roboto.swap_placeholder_images import swap_placeholder_images  # noqa: F401
  ```

- Add to `__all__` (after `"fastlane",` near line 175):
  ```python
      "swap_placeholder_images",
  ```

- Add a dispatch branch in `_run_dispatch`. The natural place is alongside the other Z-series local-file emitters — but mr_roboto's `_run_dispatch` is one long `if action == ...:` chain, so just add it near the END (before the unknown-action fallback). Find the last `if action == ...` block in the function and append:

  ```python
      if action == "swap_placeholder_images":
          # Plan 3 — i2p image-gen integration. Mechanical body internally
          # enqueues a prompt_writer LLM task and N image tasks through
          # beckman; never calls dispatcher/HK/paintress directly.
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
              # Soft-success: even when nothing is replaced (graceful
              # degrade), ok=True. The workflow's done_when accepts
              # ok=true with skipped_count > 0.
              return Action(status="completed", result=res)
          except Exception as e:
              # Belt-and-braces — the executor is already best-effort.
              return Action(
                  status="completed",
                  result={
                      "ok": True,
                      "replaced_count": 0,
                      "skipped_count": 0,
                      "errors": [f"unexpected: {e}"],
                  },
              )
  ```

  IMPORTANT: locate the **last** `if action == ...` block before any "unknown action" handler. If `_run_dispatch` ends with an explicit "unknown action" return, insert the new branch BEFORE that return. Use the existing surrounding indentation (4 spaces inside the function).

- [ ] **Step 5: Run tests**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py packages/mr_roboto/tests/test_reversibility_registry.py -q
```
Expected: PASS — the dispatch test passes (3 passed), and the reversibility registry consistency test (`test_reversibility_registry.py`) stays green because every verb in `_run_dispatch` has a `VERB_REVERSIBILITY` entry.

- [ ] **Step 6: Regression on the full mr_roboto suite**

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/ -q
```
Expected: no new failures (note: per the project's known-reds list in `project_residual_reanalysis_20260521`, there may be pre-existing flakes — Plan 3 must not introduce new ones).

- [ ] **Step 7: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/__init__.py packages/mr_roboto/src/mr_roboto/reversibility.py packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py
git commit -m "feat(image): wire swap_placeholder_images into mr_roboto dispatch"
```

---

## Task 9: i2p_v3.json — prototype-phase swap step

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`
- Test: `tests/i2p/test_i2p_swap_step.py`

Add step `5.35` (between `5.30c annotate_html_oids` and `5.40 emit_preview_url`). Soft `done_when` so a no-op return (`ok=true` with `skipped_count > 0`) still passes — image generation must NEVER block the mission. Step depends on `5.30c` (HTML is annotated by then).

- [ ] **Step 1: Write the failing test**

```python
# tests/i2p/test_i2p_swap_step.py
"""Plan 3 — i2p_v3.json swap_placeholder_images step shape + position."""
import json


def _load_steps():
    with open("src/workflows/i2p/i2p_v3.json", encoding="utf-8") as fh:
        return json.load(fh)["steps"]


def test_step_exists():
    steps = _load_steps()
    swap = [s for s in steps if s.get("id") == "5.35"]
    assert len(swap) == 1, "expected exactly one step with id 5.35"


def test_step_shape():
    steps = _load_steps()
    swap = next(s for s in steps if s["id"] == "5.35")
    assert swap["agent"] == "mechanical"
    assert swap["executor"] == "swap_placeholder_images"
    assert swap["payload"]["action"] == "swap_placeholder_images"
    # depends_on must include 5.30c (HTMLs are annotated by then).
    assert "5.30c" in swap["depends_on"]
    # Soft done_when — accepts skipped_count > 0.
    dw = swap["done_when"].lower()
    assert "swap_placeholder_images" in dw or "ok" in dw
    assert "skipped" in dw, "done_when must explicitly permit skipping"
    assert swap.get("reversibility") == "full"


def test_step_runs_before_emit_preview_url():
    steps = _load_steps()
    # 5.40 emit_preview_url must depend transitively on (or after) 5.35
    # so the preview URL is emitted with real images, not placeholders.
    by_id = {s["id"]: s for s in steps}
    emit = by_id["5.40"]
    # emit_preview_url depends on 5.30c; we strengthen the chain by making
    # 5.35 also depend on 5.30c (parallel sibling) — the natural ordering
    # gives 5.30c → 5.35 → 5.40 once 5.40's depends_on is extended to
    # include 5.35.
    assert "5.35" in emit["depends_on"], (
        "5.40 emit_preview_url must depend on 5.35 so the URL surfaces "
        "real images, not placeholders"
    )


def test_phase_5_in_phase():
    steps = _load_steps()
    swap = next(s for s in steps if s["id"] == "5.35")
    assert swap["phase"] == "phase_5"
```

- [ ] **Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest tests/i2p/test_i2p_swap_step.py -q
```
Expected: FAIL — no step `5.35`.

- [ ] **Step 3: Insert the step**

In `src/workflows/i2p/i2p_v3.json`, find the step with `"id": "5.30c"` (around line 5176, the `annotate_html_oids` step). Insert the new step IMMEDIATELY after `5.30c`'s closing `},` and BEFORE the `"id": "5.40"` step. Then **also** extend `5.40`'s `depends_on` to include `"5.35"`.

New step body:

```json
    {
      "id": "5.35",
      "phase": "phase_5",
      "name": "swap_placeholder_images",
      "agent": "mechanical",
      "depends_on": [
        "5.30c"
      ],
      "executor": "swap_placeholder_images",
      "payload": {
        "action": "swap_placeholder_images"
      },
      "context": {
        "estimated_output_tokens": 0
      },
      "instruction": "Plan 3 — replace placehold.co <img> placeholders in every mission_{mission_id}/.web/*.html with real diffusion-generated PNGs. Enqueues one prompt_writer LLM task (single-call, response_format=json_schema) to enrich each placeholder's alt text into a diffusion prompt, then one image task per placeholder through beckman (which calls fatih_hoca.select(needs_image=True) → dispatcher.dispatch → paintress). Writes PNGs under mission_{mission_id}/.web/assets/ (inside the served preview root) and rewrites <img src> to relative `assets/<file>.png`. Best-effort: per-placeholder failures keep the original placehold.co URL — never blocks the mission. Skipping the whole step (e.g. no image provider available) is acceptable; the preview surfaces with placeholders.",
      "done_when": "swap_placeholder_images executor returns ok=true (replaced_count >= 0; skipped_count > 0 is acceptable when image generation is unavailable)",
      "produces": [
        "mission_{mission_id}/.web/assets/"
      ],
      "reversibility": "full"
    },
```

Then locate the step with `"id": "5.40"` and update its `depends_on` from:
```json
      "depends_on": [
        "5.30c"
      ],
```
to:
```json
      "depends_on": [
        "5.30c",
        "5.35"
      ],
```

- [ ] **Step 4: Run test to verify it passes**

```
.venv/Scripts/python -m pytest tests/i2p/test_i2p_swap_step.py -q
```
Expected: PASS (4 passed).

- [ ] **Step 5: Regression — i2p JSON-shape tests stay green**

```
.venv/Scripts/python -m pytest tests/i2p/ -q -x
```
Expected: no NEW failures. If the project has an "every step has a dispatchable executor / no orphan dependency" sweep test (`tests/i2p/test_no_dead_ends.py` or similar), it will catch a typo here — fix the step ID before proceeding.

- [ ] **Step 6: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json tests/i2p/test_i2p_swap_step.py
git commit -m "feat(image): i2p_v3 step 5.35 — swap placeholder images"
```

---

## Task 10: End-to-end host-path integration test

**Files:**
- Test: `tests/integration/test_image_i2p_swap_e2e.py`

Proves the whole lane wires together: a mission workspace with three placeholder `<img>` tags → mr_roboto.run with `action: swap_placeholder_images` → mocked beckman.enqueue serves prompt_writer + 3 image tasks → 3 PNGs land under `mission_{id}/.web/assets/` → HTML is rewritten. Host-path (per the recurring lesson + Plan 1 Task 13 precedent).

Does NOT depend on Plan 1 being merged — the test mocks `beckman.enqueue` at the import-path level. When Plan 1 is merged, the same test still passes (the mock supersedes the real enqueue).

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_image_i2p_swap_e2e.py
"""Plan 3 — end-to-end i2p placeholder swap.

Mocks beckman.enqueue (since the real chain requires a configured image
provider). Asserts:
  - mr_roboto.run dispatches `swap_placeholder_images`
  - the prototype HTML has its 3 placehold.co <img>s rewritten to
    `assets/<id>.png`
  - 3 real PNG files exist on disk under mission_{id}/.web/assets/
  - the result.replaced_count == 3
"""
import io
import os

import pytest


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
async def test_i2p_swap_end_to_end(monkeypatch, tmp_path):
    # 1) Stand up a fake mission workspace with one HTML file.
    ws = tmp_path / "mission_777"
    web = ws / ".web"
    web.mkdir(parents=True)
    (web / "home.html").write_text(_HTML, encoding="utf-8")

    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace",
        lambda mid: str(ws),
    )

    # 2) Mock the prompt_writer + per-image enqueues. The prompt_writer
    # returns the 3 prompts; each image task writes a real 64x64 PNG.
    call_log: list[str] = []

    class _PromptResult:
        status = "completed"
        result = {
            "_schema_version": "1",
            "prompts": [
                {"placeholder_id": "home__0", "prompt": "coral barista scene with warm cream interior"},
                {"placeholder_id": "home__1", "prompt": "slate indigo task dashboard, muted sage progress"},
                {"placeholder_id": "home__2", "prompt": "friendly headshot, deep teal background"},
            ],
        }

    async def _fake_enqueue(spec, **kwargs):
        agent_type = spec.get("agent_type")
        call_log.append(agent_type or "")
        if agent_type == "prompt_writer":
            return _PromptResult()
        if agent_type == "image":
            ic = spec["context"]["image_call"]
            from PIL import Image
            os.makedirs(ic["out_dir"], exist_ok=True)
            path = os.path.join(
                ic["out_dir"], f"{ic['filename_hint']}_raw.png",
            )
            Image.new("RGB", (ic["width"], ic["height"]), (100, 150, 200)).save(
                path, "PNG"
            )

            class _ImgResult:
                status = "completed"
                result = {
                    "path": path,
                    "provider": "pollinations",
                    "model": "pollinations/flux",
                }
            return _ImgResult()
        raise AssertionError(f"unexpected agent_type: {agent_type!r}")

    monkeypatch.setattr(
        "mr_roboto.swap_placeholder_images._enqueue_beckman", _fake_enqueue,
    )

    # 3) Dispatch via mr_roboto.run (the real dispatch path).
    import mr_roboto
    task = {
        "id": 12345,
        "mission_id": 777,
        "title": "swap_e2e",
        "context": {
            "payload": {
                "action": "swap_placeholder_images",
                "design_tokens": {"primary": "#E07A5F"},
                "brand_voice": "warm, neighborhood coffee shop",
            },
        },
    }
    action = await mr_roboto.run(task)

    # 4) Assert the result shape.
    assert action.status == "completed"
    res = action.result
    assert res["ok"] is True
    assert res["replaced_count"] == 3
    assert res["skipped_count"] == 0
    assert res["html_files_changed"] == 1

    # 5) 3 PNGs on disk under .web/assets/.
    assets = ws / ".web" / "assets"
    pngs = sorted(p.name for p in assets.glob("*.png"))
    assert len(pngs) == 3, pngs
    # Each PNG is a real image, not a 0-byte file.
    for png in pngs:
        assert (assets / png).stat().st_size > 0

    # 6) HTML rewritten.
    rewritten = (web / "home.html").read_text(encoding="utf-8")
    assert "placehold.co" not in rewritten
    assert rewritten.count('src="assets/') == 3

    # 7) Call ordering: prompt_writer once, image three times.
    assert call_log.count("prompt_writer") == 1
    assert call_log.count("image") == 3
```

- [ ] **Step 2: Run the test**

```
.venv/Scripts/python -m pytest tests/integration/test_image_i2p_swap_e2e.py -q
```
Expected: PASS (1 passed).

- [ ] **Step 3: Full green-check across Plan 3's new tests**

Per Plan 1 Task 13's split-invocation rule (root `tests/` + package `packages/*/tests/` must not share a single pytest invocation):

```
.venv/Scripts/python -m pytest packages/mr_roboto/tests/test_swap_placeholder_images.py packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py packages/mr_roboto/tests/test_emit_preview_url_assets.py -q
.venv/Scripts/python -m pytest tests/agents/test_prompt_writer.py tests/agents/test_prompt_writer_template.py tests/i2p/test_i2p_swap_step.py tests/integration/test_image_i2p_swap_e2e.py -q
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_image_i2p_swap_e2e.py
git commit -m "test(image): end-to-end i2p swap_placeholder_images host-path"
```

---

## Plan 3 done-when

- i2p mission reaching phase 5 produces real images for every `placehold.co` `<img>` in `mission_{id}/.web/*.html`, with the HTML rewritten to `src="assets/<id>.png"` and PNGs on disk under `mission_{id}/.web/assets/`.
- Per-placeholder failures keep the original placeholder (graceful degrade); step `5.35` accepts `skipped_count > 0` as a soft pass — image generation never blocks the mission.
- The cloudflared preview URL (step `5.40`) now serves real images via the same `.web/` static root (no resolver change required).
- All new tests green:
  - `tests/agents/test_prompt_writer.py` (5)
  - `tests/agents/test_prompt_writer_template.py` (4)
  - `packages/mr_roboto/tests/test_swap_placeholder_images.py` (9)
  - `packages/mr_roboto/tests/test_swap_placeholder_images_dispatch.py` (3)
  - `packages/mr_roboto/tests/test_emit_preview_url_assets.py` (2)
  - `tests/i2p/test_i2p_swap_step.py` (4)
  - `tests/integration/test_image_i2p_swap_e2e.py` (1)
- No new failures in:
  - `packages/mr_roboto/tests/` (baseline reds per `project_residual_reanalysis_20260521` excepted)
  - `tests/agents/test_prompt_quality.py` (prompt_writer satisfies the 3 invariants)
  - `tests/i2p/` (workflow JSON shape sweep)
- mr_roboto dispatcher knows `swap_placeholder_images` (entry in `_run_dispatch`, `__all__`, `VERB_REVERSIBILITY`); a real i2p mission run would invoke the mechanic at step `5.35`.

## Dependencies on other plans

- **Plan 1 (cloud spine)** MUST be merged before Plan 3's mechanic produces real images in production — Plan 3's image enqueues (`agent_type: "image"`, `context.image_call.raw_dispatch=True`) rely on Plan 1's beckman admission branch (`needs_image=True`), Plan 1's dispatcher `_dispatch_image` path, and Plan 1's `paintress` providers. Plan 3's tests mock `beckman.enqueue` so they pass without Plan 1 — but a live mission with Plan 3 alone would degrade every image to "skipped" (no provider) and leave placeholders, which is the spec's intended graceful-degrade behaviour.
- **Plan 2 (local clair_obscur + GPU handover)** is OPTIONAL. When Plan 2 is also merged, Plan 3's `agent_type: "image"` tasks transparently pick local SDXL via Plan 1's already-extended scorer; no Plan 3 change required.
- Plan 3 is **file-disjoint** from Plan 2 (see "File ownership" above) — they can run in parallel worktrees and merge in either order.

## Follow-on (after Plan 3 executes)

- **Founder review of generated images.** Spec §8 mentions surfacing the swap result; a follow-up could enqueue a `notify_user` mechanical sibling after `5.35` so the founder sees a preview gallery (similar to the `propose_spec_patch` Apply/Reject pattern in mr_roboto's `_surface_spec_patch_proposal`).
- **Asset cache across mission re-runs.** Right now a re-run regenerates every image. A cheap content-addressed cache keyed on `(prompt, width, height, provider)` would skip already-generated assets.
- **Z5 mobile / Expo path.** `.prototype/index.html` (Expo Web) is out of Plan 3 scope. When image gen reaches Z5, mirror the mechanic to scan `.prototype/**.html` and write to `.prototype/assets/`.
- **Per-screen design context.** Plan 3 passes a single `design_tokens` / `brand_voice` to the prompt_writer. A future refinement could pass per-screen `section_intent` (currently inferred heuristically from `alt` text in `_section_from_alt`) — read directly from the screen_plan artifact for that file.
