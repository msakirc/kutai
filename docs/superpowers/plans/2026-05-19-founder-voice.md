# Founder Voice for Z7 Public-Content Generators — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the founder one place to define their writing voice, and feed it into the LLM prompts of the three Z7 public-content generators so launch posts, cold outreach, and marketing copy are written in the founder's voice instead of generic corporate-speak.

**Architecture:** A single founder-authored markdown file `docs/templates/brand_voices/founder.md` (same directory as the existing brand-voice lint profiles). A new loader `load_founder_voice()` in `src/ops/brand_voice.py` returns the file's prose body, or `""` when the file is absent or still carries the unfilled sentinel. Three generators default their voice input to `load_founder_voice()`. Empty voice → current behavior (no regression).

**Tech Stack:** Python 3.10, pytest, existing `src/ops/brand_voice.py` front-matter parser.

**Scope boundary:** This is NOT the de-scoped Z0 `founder_profiles` table. No DB schema, no wizard, no multi-tenancy. One markdown file + one loader + three 1-3 line wire points. `changelog_draft` (mechanical KAC bullets, not prose) and `lifecycle_email` (DB-stored pre-authored templates) are intentionally excluded.

---

### Task 1: Founder voice template + loader

**Files:**
- Create: `docs/templates/brand_voices/founder.md`
- Modify: `src/ops/brand_voice.py` (add `load_founder_voice` after `load_brand_voice`, ~line 285)
- Test: `tests/ops/test_founder_voice.py`

- [ ] **Step 1: Create the founder voice template**

Create `docs/templates/brand_voices/founder.md` with this exact content:

```markdown
---
slug: founder
display_name: Founder Voice
version: "1.0"
---
<!-- FOUNDER_VOICE_UNFILLED: delete this comment line once you have filled in the
     sections below. While this line is present, generators ignore this file and
     fall back to generic voice. -->

## Who I am

<!-- Your name, your product, what it does, who it is for. 2-4 sentences. -->

## How I write

<!-- Describe your voice: formal or casual? short punchy sentences or longer?
     dry humor or earnest? first person ("I built this") or company ("we")?
     3-6 bullet points. -->

## Words and phrasing I use

<!-- Phrases, terms, or framings that sound like you. And words to avoid. -->

## Sample sentences in my voice

<!-- 3-5 sentences you actually wrote, or would write, about your product.
     The LLM mimics these. -->
```

- [ ] **Step 2: Write the failing test**

Create `tests/ops/test_founder_voice.py`:

```python
import pytest
from src.ops.brand_voice import load_founder_voice


def test_unfilled_template_returns_empty(tmp_path):
    vf = tmp_path / "founder.md"
    vf.write_text(
        "---\nslug: founder\n---\n"
        "<!-- FOUNDER_VOICE_UNFILLED: delete this -->\n\n## Who I am\n",
        encoding="utf-8",
    )
    assert load_founder_voice(voices_dir=str(tmp_path)) == ""


def test_missing_file_returns_empty(tmp_path):
    assert load_founder_voice(voices_dir=str(tmp_path)) == ""


def test_filled_template_returns_body(tmp_path):
    vf = tmp_path / "founder.md"
    vf.write_text(
        "---\nslug: founder\n---\n"
        "## Who I am\n\nI build KutAI, a personal AI agent.\n",
        encoding="utf-8",
    )
    out = load_founder_voice(voices_dir=str(tmp_path))
    assert "I build KutAI" in out
    assert out.strip() != ""


def test_real_repo_template_is_unfilled():
    # The shipped template must read as unfilled until the founder edits it.
    assert load_founder_voice() == ""
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/ops/test_founder_voice.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'load_founder_voice'`

- [ ] **Step 4: Implement `load_founder_voice`**

In `src/ops/brand_voice.py`, append after `load_brand_voice` (end of file):

```python
# Sentinel left in the shipped founder.md template. While present, the
# founder has not personalized their voice yet → loader returns "".
_FOUNDER_UNFILLED_MARKER = "FOUNDER_VOICE_UNFILLED"


def load_founder_voice(voices_dir: str | None = None) -> str:
    """Return the founder's voice description as a prose block for LLM prompts.

    Reads ``docs/templates/brand_voices/founder.md``. Returns ``""`` when the
    file is absent or still carries the unfilled-template sentinel — callers
    treat an empty string as "no founder voice", their existing default.
    """
    _voices_dir = voices_dir or _default_voices_dir()
    path = os.path.join(_voices_dir, "founder.md")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        logger.debug("founder_voice: unreadable: %s (%s)", path, exc)
        return ""
    if _FOUNDER_UNFILLED_MARKER in text:
        return ""
    voice = _parse_brand_voice(text)
    body = (voice.raw_body_md or "").strip()
    # Strip residual HTML-comment scaffolding so the LLM sees only real guidance.
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
    return body
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/ops/test_founder_voice.py -q -p no:cacheprovider`
Expected: PASS — 4 passed

- [ ] **Step 6: Commit**

```bash
git add docs/templates/brand_voices/founder.md src/ops/brand_voice.py tests/ops/test_founder_voice.py
git commit -m "feat(voice): founder voice template + load_founder_voice loader"
```

---

### Task 2: Wire founder voice into launch_drafts

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/launch_drafts.py:148`
- Test: `packages/mr_roboto/tests/test_launch_drafts.py` (add one test; if file absent, create it)

- [ ] **Step 1: Write the failing test**

Add to `packages/mr_roboto/tests/test_launch_drafts.py` (create the file with this content if it does not exist):

```python
import pytest
from unittest.mock import patch, AsyncMock
import mr_roboto.launch_drafts as ld


@pytest.mark.asyncio
async def test_launch_draft_falls_back_to_founder_voice():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["desc"] = spec["description"]
        return {"task_id": 1}

    with patch.object(ld, "_enqueue", fake_enqueue), \
         patch.object(ld, "fetch_launch_lessons", AsyncMock(return_value=[])), \
         patch.object(ld, "load_founder_voice", return_value="I write plainly."):
        # payload carries NO brand_voice → loader fallback must fill it
        res = await ld.run("twitter", {"product_id": "p1", "launch_id": 7, "spec": "x"})

    assert res["status"] == "enqueued"
    assert "I write plainly." in captured["desc"]


@pytest.mark.asyncio
async def test_explicit_brand_voice_wins_over_founder_voice():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["desc"] = spec["description"]
        return {"task_id": 1}

    with patch.object(ld, "_enqueue", fake_enqueue), \
         patch.object(ld, "fetch_launch_lessons", AsyncMock(return_value=[])), \
         patch.object(ld, "load_founder_voice", return_value="FALLBACK"):
        res = await ld.run("twitter", {"product_id": "p1", "launch_id": 7,
                                       "spec": "x", "brand_voice": "EXPLICIT"})

    assert "EXPLICIT" in captured["desc"]
    assert "FALLBACK" not in captured["desc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_launch_drafts.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: <module 'mr_roboto.launch_drafts'> does not have the attribute 'load_founder_voice'`

- [ ] **Step 3: Implement the wire**

In `packages/mr_roboto/src/mr_roboto/launch_drafts.py`, add to the imports near the top of the file:

```python
from src.ops.brand_voice import load_founder_voice
```

Then change line 148 from:

```python
    brand_voice = payload.get("brand_voice") or ""
```

to:

```python
    # Explicit caller brand_voice wins; otherwise fall back to the founder's
    # own voice profile (docs/templates/brand_voices/founder.md). Empty when
    # the founder has not filled the template — same as the prior default.
    brand_voice = payload.get("brand_voice") or load_founder_voice()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_launch_drafts.py -q -p no:cacheprovider`
Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/launch_drafts.py packages/mr_roboto/tests/test_launch_drafts.py
git commit -m "feat(voice): launch_drafts falls back to founder voice"
```

---

### Task 3: Wire founder voice into outreach_draft

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/outreach_draft.py:64-69`
- Test: `packages/mr_roboto/tests/test_outreach_draft.py` (add one test; if file absent, create it)

- [ ] **Step 1: Write the failing test**

Add to `packages/mr_roboto/tests/test_outreach_draft.py` (create the file with this content if it does not exist):

```python
import pytest
from unittest.mock import patch
import mr_roboto.outreach_draft as od


@pytest.mark.asyncio
async def test_outreach_draft_includes_founder_voice():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["desc"] = spec["description"]
        return {"task_id": 1}

    with patch.object(od, "enqueue", fake_enqueue), \
         patch.object(od, "load_founder_voice", return_value="Dry, direct, no fluff."):
        res = await od.run_outreach_draft(
            product_id="p1", mission_id=3,
            prospect_data={"name": "Sam"}, template_id="cold", list_id="L1",
        )

    assert res["status"] == "enqueued"
    assert "Dry, direct, no fluff." in captured["desc"]


@pytest.mark.asyncio
async def test_outreach_draft_no_voice_when_unfilled():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["desc"] = spec["description"]
        return {"task_id": 1}

    with patch.object(od, "enqueue", fake_enqueue), \
         patch.object(od, "load_founder_voice", return_value=""):
        await od.run_outreach_draft(
            product_id="p1", mission_id=3,
            prospect_data={"name": "Sam"}, template_id="cold", list_id="L1",
        )

    # Empty voice → no dangling "Brand voice:" header
    assert "Brand voice:" not in captured["desc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_outreach_draft.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: ... does not have the attribute 'load_founder_voice'`

- [ ] **Step 3: Implement the wire**

In `packages/mr_roboto/src/mr_roboto/outreach_draft.py`, add to the imports near the top of the file:

```python
from src.ops.brand_voice import load_founder_voice
```

Then change the body of `run_outreach_draft` — the lines that build `instruction` and `spec` (currently lines 64-69):

```python
    instruction = _TEMPLATE_INSTRUCTIONS.get(
        template_id, _DEFAULT_TEMPLATE_INSTRUCTION)
    _kind = "follow-up" if template_id == "follow_up" else "outreach"
    spec = {
        "title": f"Draft {_kind} for {prospect_data.get('name', 'prospect')} ({product_id})",
        "description": instruction,
```

to:

```python
    instruction = _TEMPLATE_INSTRUCTIONS.get(
        template_id, _DEFAULT_TEMPLATE_INSTRUCTION)
    # Write the draft in the founder's voice when they have defined one.
    _voice = load_founder_voice()
    if _voice:
        instruction = f"{instruction}\n\nBrand voice:\n{_voice[:800]}"
    _kind = "follow-up" if template_id == "follow_up" else "outreach"
    spec = {
        "title": f"Draft {_kind} for {prospect_data.get('name', 'prospect')} ({product_id})",
        "description": instruction,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_outreach_draft.py -q -p no:cacheprovider`
Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/outreach_draft.py packages/mr_roboto/tests/test_outreach_draft.py
git commit -m "feat(voice): outreach_draft writes in founder voice"
```

---

### Task 4: Wire founder voice into marketing_copy

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/marketing_copy.py` (`run_marketing_copy`, ~line 427)
- Test: `packages/mr_roboto/tests/test_marketing_copy.py` (add one test; if file absent, create it)

- [ ] **Step 1: Write the failing test**

Add to `packages/mr_roboto/tests/test_marketing_copy.py` (create the file with this content if it does not exist):

```python
import pytest
from unittest.mock import patch
import mr_roboto.marketing_copy as mc


@pytest.mark.asyncio
async def test_marketing_copy_prompt_includes_founder_voice():
    captured = {}

    async def fake_enqueue(spec, **kw):
        captured["spec"] = spec
        return {"result": {"hero": ["H"]}}

    with patch.object(mc, "enqueue", fake_enqueue), \
         patch.object(mc, "load_founder_voice", return_value="Plainspoken, concrete."):
        await mc.run_marketing_copy(
            product_id="p1", mission_id=5,
            product_spec={"name": "Thing"},
        )

    desc = captured["spec"]["description"]
    user_msg = captured["spec"]["context"]["llm_call"]["messages"][1]["content"]
    assert "Plainspoken, concrete." in desc
    assert "Plainspoken, concrete." in user_msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_marketing_copy.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: ... does not have the attribute 'load_founder_voice'`

- [ ] **Step 3: Implement the wire**

In `packages/mr_roboto/src/mr_roboto/marketing_copy.py`, add to the imports near the top of the file:

```python
from src.ops.brand_voice import load_founder_voice
```

Then in `run_marketing_copy`, change the prompt-build line (currently line 427) from:

```python
    prompt = _build_prompt(product_spec, faq_seed)
```

to:

```python
    prompt = _build_prompt(product_spec, faq_seed)
    # Prepend the founder's voice so generated hero/feature/pricing copy
    # reads in their voice, not generic corporate-speak. No-op when unfilled.
    _voice = load_founder_voice()
    if _voice:
        prompt = f"Brand voice — write all copy in this voice:\n{_voice[:800]}\n\n{prompt}"
```

This single change flows into both `spec["description"]` (line 430) and the
`messages` user content (line 445), because both already reference `prompt`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_marketing_copy.py -q -p no:cacheprovider`
Expected: PASS — 1 passed

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/marketing_copy.py packages/mr_roboto/tests/test_marketing_copy.py
git commit -m "feat(voice): marketing_copy generates in founder voice"
```

---

### Task 5: Regression sweep + docs note

**Files:**
- Modify: `docs/templates/brand_voices/README.md` (append a section)

- [ ] **Step 1: Run the full affected-package test suites**

Run: `python -m pytest packages/mr_roboto/tests/ -q -p no:cacheprovider`
Expected: PASS — no regressions (pre-existing failures, if any, unchanged).

Run: `python -m pytest tests/ops/test_founder_voice.py -q -p no:cacheprovider`
Expected: PASS — 4 passed.

- [ ] **Step 2: Add a README section**

Append to `docs/templates/brand_voices/README.md`:

```markdown
## founder.md — the founder's personal voice

`founder.md` is a special profile. Unlike the per-audience lint profiles
above, its prose body is fed directly into the LLM prompt of the public-content
generators (launch posts, cold outreach, marketing copy) so they write in the
founder's voice. It ships as a fill-in template carrying a
`FOUNDER_VOICE_UNFILLED` sentinel — while that sentinel is present,
`load_founder_voice()` returns `""` and generators use their generic default.
Delete the sentinel line once the template is filled in.
```

- [ ] **Step 3: Commit**

```bash
git add docs/templates/brand_voices/README.md
git commit -m "docs(voice): document founder.md voice profile"
```

---

## Self-Review

- **Spec coverage:** The gap was "Z7 public-content generators are voiceless." Tasks 2/3/4 cover the three LLM-prose generators (launch posts, cold outreach, marketing copy). Task 1 provides the source. `changelog`/`lifecycle_email` excluded with rationale (mechanical / DB-template). Covered.
- **Type consistency:** `load_founder_voice(voices_dir: str | None = None) -> str` — same signature shape as the sibling `load_brand_voice`; called with no args by all three generators; called with `voices_dir=` only in tests. `_parse_brand_voice`, `_default_voices_dir`, `re`, `os`, `logger` all already exist in `brand_voice.py`. Consistent.
- **No placeholders:** every code step shows complete code; every run step shows the command + expected output.
```
