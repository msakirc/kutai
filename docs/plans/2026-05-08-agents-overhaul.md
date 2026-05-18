# KutAI Agents Overhaul Plan (revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift output quality of the existing 21-agent roster via prompt polish (MetaGPT-style phrasing), tightened self-reflection on workhorse agents, and a sharper classifier — without merging, dropping, or adding mode flags.

**Architecture:** All 21 agents stay as-is (post-2026-05-04 kill-agents pure config). No file deletes, no alias maps, no payload mode/phase flags. Changes land in `src/agents/*.py::get_system_prompt`, `prompt_versions` DB rows for live swap, `src/core/task_classifier.py`, and `packages/coulson/react.py` self-reflection block.

**Tech Stack:** Python 3.10, SQLite (`prompt_versions`), pytest, existing `/prompt save` flow.

**Audit reference:** `docs/research/2026-05-08-agent-usage-audit.md`

---

## Out of Scope (revised — explicitly forbidden)

- **No agent merges.** No mode/phase/artifact_type flag-based consolidation. Memory `feedback_no_agent_modes.md`.
- **No drops.** Agents with 0 traffic (code_reviewer, visual_reviewer, product_researcher) stay — i2p hasn't reached late phases. Memory `feedback_zero_traffic_not_dead.md`.
- **No alias maps.** Workflow JSONs unchanged.
- No coulson runtime changes beyond reflection prompt block.
- No fatih_hoca scoring changes.
- No telegram_bot.py split.
- `watch_produces` event-react flag — not in this plan, write spec only when use case appears.

---

## Phase 1: Prompt polish (MetaGPT diff)

### Task 1: Fetch + diff MetaGPT role prompts

**Files:**
- Create: `docs/research/2026-05-08-metagpt-prompt-diff.md`

- [ ] **Step 1: Fetch source prompts**

```bash
curl -s https://raw.githubusercontent.com/geekan/MetaGPT/main/metagpt/actions/write_prd.py > /tmp/mg_pm.py
curl -s https://raw.githubusercontent.com/geekan/MetaGPT/main/metagpt/actions/design_api.py > /tmp/mg_arch.py
curl -s https://raw.githubusercontent.com/geekan/MetaGPT/main/metagpt/actions/write_code.py > /tmp/mg_eng.py
curl -s https://raw.githubusercontent.com/geekan/MetaGPT/main/metagpt/actions/write_test.py > /tmp/mg_qa.py
curl -s https://raw.githubusercontent.com/geekan/MetaGPT/main/metagpt/actions/run_code.py > /tmp/mg_qa_run.py
curl -s https://raw.githubusercontent.com/geekan/MetaGPT/main/metagpt/actions/research.py > /tmp/mg_research.py
curl -s https://raw.githubusercontent.com/geekan/MetaGPT/main/metagpt/actions/debug_error.py > /tmp/mg_fix.py
```

If any 404s, log + skip — repo layout may have shifted.

- [ ] **Step 2: Extract template strings**

For each file: pull the prompt-template literals (look for triple-quoted strings, `PROMPT_TEMPLATE = """..."""` or `OUTPUT_MAPPING`).

- [ ] **Step 3: Write diff doc**

For each pair below, list 3-5 concrete phrasing wins to lift:

| KutAI agent | MetaGPT counterpart | source file |
|---|---|---|
| planner | ProductManager (write_prd) | `mg_pm.py` |
| architect | Architect (design_api) | `mg_arch.py` |
| coder | Engineer (write_code) | `mg_eng.py` |
| implementer | Engineer (write_code) | `mg_eng.py` |
| fixer | DebugError | `mg_fix.py` |
| test_generator | QA (write_test) | `mg_qa.py` |
| reviewer | RunCode review section | `mg_qa_run.py` |
| researcher | Research (CollectLinks/WebBrowseAndSummarize) | `mg_research.py` |

Things to look for: explicit DO/DON'T sections, output schema templates, role-priming opener ("You are X with Y years of experience"), end-of-task self-check checklists, structured-output JSON examples.

- [ ] **Step 4: Commit diff doc**

```bash
git add docs/research/2026-05-08-metagpt-prompt-diff.md
git commit -m "docs(agents): MetaGPT prompt diff for polish targets"
```

---

### Task 2: Polish workhorse prompts (implementer + test_generator + executor + planner)

**Files:**
- Modify: `src/agents/implementer.py`, `src/agents/test_generator.py`, `src/agents/executor.py`, `src/agents/planner.py`
- Test: `tests/agents/test_prompt_quality.py`

These four are the highest-traffic agents (13509 / 11529 / 10138 / 5114 picks per 60d). Biggest leverage.

- [ ] **Step 1: Failing test for prompt invariants**

```python
import pytest
from src.agents import get_agent

@pytest.mark.parametrize("name", ["implementer", "test_generator", "executor", "planner"])
def test_prompt_has_role_primer(name):
    """Prompt should start with explicit role identity."""
    p = get_agent(name).get_system_prompt({"description": "x"})
    first_line = p.strip().split("\n")[0].lower()
    assert "you are" in first_line, f"{name} prompt missing role primer"

@pytest.mark.parametrize("name", ["implementer", "test_generator", "executor", "planner"])
def test_prompt_has_dos_and_donts(name):
    p = get_agent(name).get_system_prompt({"description": "x"}).lower()
    assert ("don't" in p or "never" in p or "do not" in p), f"{name} missing negative guardrails"
    assert ("must" in p or "always" in p), f"{name} missing positive directives"

@pytest.mark.parametrize("name", ["implementer", "test_generator", "executor", "planner"])
def test_prompt_has_final_answer_schema(name):
    p = get_agent(name).get_system_prompt({"description": "x"})
    assert "final_answer" in p
    assert "```json" in p, f"{name} missing JSON schema example"
```

- [ ] **Step 2: Run, expect FAIL where invariants don't hold.**

```bash
timeout 30 pytest tests/agents/test_prompt_quality.py -v
```

- [ ] **Step 3: Apply diff wins per agent**

For each of the 4: open `src/agents/<name>.py::get_system_prompt`, edit body. Use phrasing from `2026-05-08-metagpt-prompt-diff.md`. Keep current tool list + workflow steps; tighten language.

- [ ] **Step 4: Re-run quality test + smoke registry import**

```bash
timeout 30 pytest tests/agents/test_prompt_quality.py -v
python -c "from src.agents import AGENT_REGISTRY; [a.get_system_prompt({'description':'x'}) for a in AGENT_REGISTRY.values()]"
```
Expected: all PASS, no exception on full registry walk.

- [ ] **Step 5: Commit**

```bash
git add src/agents/implementer.py src/agents/test_generator.py src/agents/executor.py src/agents/planner.py tests/agents/test_prompt_quality.py
git commit -m "feat(agents): polish workhorse prompts (implementer/test_gen/executor/planner)"
```

---

### Task 3: Polish remaining active prompts (reviewer, coder, fixer, researcher, analyst, writer, summarizer, architect)

**Files:**
- Modify: `src/agents/reviewer.py`, `src/agents/coder.py`, `src/agents/fixer.py`, `src/agents/researcher.py`, `src/agents/analyst.py`, `src/agents/writer.py`, `src/agents/summarizer.py`, `src/agents/architect.py`

- [ ] **Step 1: Extend prompt_quality test to all 8**

```python
ALL_ACTIVE = [
    "reviewer", "coder", "fixer", "researcher",
    "analyst", "writer", "summarizer", "architect",
]
@pytest.mark.parametrize("name", ALL_ACTIVE)
def test_prompt_invariants_active(name):
    p = get_agent(name).get_system_prompt({"description": "x"})
    first = p.strip().split("\n")[0].lower()
    assert "you are" in first
    assert "final_answer" in p
```

- [ ] **Step 2: Apply MetaGPT diff wins per agent.**

Use `2026-05-08-metagpt-prompt-diff.md`. Each agent gets 3-5 surgical edits, not a rewrite.

- [ ] **Step 3: Run + smoke**

```bash
timeout 30 pytest tests/agents/test_prompt_quality.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/agents/ tests/agents/test_prompt_quality.py
git commit -m "feat(agents): polish reviewer/coder/fixer/researcher/analyst/writer/summarizer/architect prompts"
```

---

### Task 4: Polish low-traffic + 0-traffic prompts (assistant, grader, code_reviewer, visual_reviewer, shopping_*, deal_analyst, product_researcher, artifact_summarizer)

**Files:** all remaining 9 agents in `src/agents/*.py`.

These have 0 or near-zero traffic but stay in the roster. Bring them up to invariant baseline so when i2p reaches them, they don't lag.

- [ ] **Step 1: Add to prompt_quality parametrize, run, expect failures.**

- [ ] **Step 2: Edit each prompt to satisfy invariants** (role primer + final_answer schema + DO/DON'T section). Match style of polished workhorses.

- [ ] **Step 3: Commit**

```bash
git add src/agents/
git commit -m "feat(agents): bring low-traffic prompts to invariant baseline"
```

---

## Phase 2: Self-reflection tightening

### Task 5: Audit current self-reflection wiring

**Files:** read-only.

- [ ] **Step 1: Trace flag → prompt injection**

```bash
rg "enable_self_reflection" src/agents/ packages/coulson/ packages/hallederiz_kadir/
```

Document: which agents have flag True, where coulson reads it, what prompt it injects, when in iteration loop it fires.

- [ ] **Step 2: Save findings to plan**

Add subsection to this doc OR create `docs/research/2026-05-08-self-reflection-audit.md` with: agents with flag True, coulson injection point (file:line), current reflection prompt text.

- [ ] **Step 3: Commit**

```bash
git add docs/research/2026-05-08-self-reflection-audit.md
git commit -m "docs(agents): self-reflection wiring audit"
```

---

### Task 6: Enable + tighten self-reflection on workhorses

**Files:**
- Modify: `src/agents/implementer.py`, `src/agents/fixer.py`, `src/agents/test_generator.py` (set `enable_self_reflection = True`)
- Modify: `packages/coulson/react.py` (per-agent reflection prompt) — only if Task 5 reveals reflection prompt is generic; otherwise inject role-specific block
- Test: `tests/agents/test_self_reflection.py`

- [ ] **Step 1: Failing test**

```python
@pytest.mark.parametrize("name", ["coder", "implementer", "fixer", "test_generator"])
def test_reflection_enabled(name):
    assert get_agent(name).enable_self_reflection is True

def test_coder_reflection_checklist():
    """Reflection should ask: ran code? tests pass? TODOs left? imports verified?"""
    from packages.coulson.react import build_reflection_prompt
    p = build_reflection_prompt(agent_name="coder", iteration=3).lower()
    for keyword in ["run", "test", "todo", "import"]:
        assert keyword in p, f"coder reflection missing '{keyword}'"

def test_implementer_reflection_checklist():
    from packages.coulson.react import build_reflection_prompt
    p = build_reflection_prompt(agent_name="implementer", iteration=3).lower()
    for keyword in ["lint", "syntax", "spec", "interface"]:
        assert keyword in p
```

- [ ] **Step 2: Run, expect FAIL.**

```bash
timeout 30 pytest tests/agents/test_self_reflection.py -v
```

- [ ] **Step 3: Set flag True on implementer/fixer/test_generator** (coder already True per audit).

- [ ] **Step 4: Add per-agent reflection prompt in coulson**

If `build_reflection_prompt` doesn't exist or is generic, add per-agent dispatch:

```python
# packages/coulson/react.py (sketch)
REFLECTION_BLOCKS = {
    "coder": "Self-check: (1) ran the code? (2) tests pass? (3) TODOs left? (4) imports verified?",
    "implementer": "Self-check: (1) lint clean? (2) py_compile passes? (3) matches ARCHITECTURE.md? (4) only your assigned file touched?",
    "fixer": "Self-check: (1) every feedback bullet addressed? (2) tests run after edit? (3) no unintended deletions?",
    "test_generator": "Self-check: (1) tests run? (2) cover the spec? (3) no flaky waits/sleeps? (4) assert messages helpful?",
}

def build_reflection_prompt(agent_name: str, iteration: int) -> str:
    block = REFLECTION_BLOCKS.get(agent_name, "Review your output before final_answer.")
    return f"[iteration {iteration}] {block}"
```

- [ ] **Step 5: Run tests + smoke a coder ReAct via test fixture (no real LLM call required).**

```bash
timeout 60 pytest tests/agents/test_self_reflection.py packages/coulson/tests/ -v
```

- [ ] **Step 6: Commit**

```bash
git add src/agents/ packages/coulson/ tests/agents/test_self_reflection.py
git commit -m "feat(agents): per-agent self-reflection checklists for code workhorses"
```

---

## Phase 3: Classifier hardening

### Task 7: Document agent-pick rules in classifier prompt

**Files:**
- Modify: `src/core/task_classifier.py`
- Test: `tests/core/test_task_classifier_picks.py`

- [ ] **Step 1: Failing parametrized test**

```python
@pytest.mark.parametrize("desc,expected", [
    ("find me a coffee machine under 5000 TL", "shopping_advisor"),
    ("write a parser for JSON logs", "coder"),
    ("implement the User model from ARCHITECTURE.md", "implementer"),
    ("fix the auth bug from review feedback", "fixer"),
    ("review this PR for security issues", "reviewer"),
    ("score this answer 0-10", "grader"),
    ("what's the capital of Turkey", "assistant"),
    ("research climate impact of EVs", "researcher"),
    ("design the auth module", "architect"),
    ("decompose this mission into steps", "planner"),
    ("write tests for login.py", "test_generator"),
    ("summarize this 5k-word article", "summarizer"),
    ("analyze the fee structure of this contract", "analyst"),
    ("write a blog post about Bayer Munich win", "writer"),
])
def test_classifier_picks(desc, expected):
    from src.core.task_classifier import classify
    result = classify(desc)
    assert result.agent_type == expected, f"got {result.agent_type} for: {desc}"
```

- [ ] **Step 2: Run, count failures.**

- [ ] **Step 3: Edit `CLASSIFIER_PROMPT`**

Per agent, add 1-line rule: "pick when X / NOT when Y". For overlap pairs (coder vs implementer, reviewer vs code_reviewer vs grader, summarizer vs artifact_summarizer, shopping_advisor vs product_researcher vs deal_analyst vs shopping_clarifier), add explicit disambiguator.

Example:
```
- coder: ad-hoc multi-file build / standalone project. Has git_commit + run_code.
- implementer: ONE file from existing ARCHITECTURE.md. No git, no run.
- fixer: edits driven by review/test feedback artifact. No new files.
```

- [ ] **Step 4: Iterate prompt until ≥12/14 pass.**

- [ ] **Step 5: Commit**

```bash
git add src/core/task_classifier.py tests/core/test_task_classifier_picks.py
git commit -m "feat(classifier): explicit pick/reject rules per agent"
```

---

## Phase 4: Verification

### Task 8: End-to-end smoke

**Files:** none.

- [ ] **Step 1: Full agent + classifier suite**

```bash
timeout 120 pytest tests/agents/ tests/core/ -m "not llm" -v
```
Expected: green.

- [ ] **Step 2: Registry walk**

```bash
python -c "
from src.agents import AGENT_REGISTRY
for name, agent in AGENT_REGISTRY.items():
    p = agent.get_system_prompt({'description': 'smoke'})
    assert 'final_answer' in p, name
    assert 'you are' in p.split('\n')[0].lower(), name
print(f'{len(AGENT_REGISTRY)} agents, all prompts valid')
"
```

- [ ] **Step 3: Run a small i2p mission via Telegram**

`/task "build a hello-world flask app with one /ping route"`

Verify in `model_pick_log`:
- planner picked, then architect, then implementer (or coder), then test_generator, then reviewer.
- No KeyError on agent lookup.
- Reflection blocks visible in prompt content (sampled from yazbunu logs).

- [ ] **Step 4: Run a small shopping query**

`/shop "coffee machine under 5000 TL"`

Verify shopping_advisor handles it. (No regression — shopping pipeline state still in flux per memory `project_shopping_state_20260505`, just confirm no breakage from prompt edits.)

- [ ] **Step 5: Tag**

```bash
git tag agents-polish-2026-05-08
```

---

## Phase 5: Documentation

### Task 9: Update CLAUDE.md + memory

**Files:**
- Modify: `CLAUDE.md`
- Create: `~/.claude/projects/.../memory/project_agents_polish_20260508.md`

- [ ] **Step 1: CLAUDE.md** — confirm "21 agents pure config" still correct (no changes from this plan). If any prompt structure became canonical (e.g. "all agents must have role primer + final_answer schema"), document under Code Style.

- [ ] **Step 2: Memory entry** summarizing what shipped: prompt invariant test, MetaGPT polish per agent, classifier rules, per-agent reflection blocks. Note any agents that were deferred.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: agents polish summary"
```

---

## Self-Review Checklist

- ✅ No agent merges, drops, or mode flags (per user feedback 2026-05-08).
- ✅ All 21 agents stay registered.
- ✅ Workflow JSONs untouched.
- ✅ Polish gives objective gain (invariant tests + classifier regression suite).
- ✅ Workhorses get per-agent reflection (highest leverage).
- ✅ Smoke covers i2p + shopping (no regression).
- ⚠️ Risk: prompt edits regress an agent. Mitigation: invariant tests catch shape drift; revert via DB `prompt_versions` if any live agent regresses.
- ⚠️ Risk: classifier prompt grows unwieldy with 21 agents × 2 lines each. Mitigation: keep ≤2 lines/agent, drop verbose preamble.
- ⚠️ Risk: 0-traffic agents (code_reviewer, visual_reviewer, etc.) get polish that's never validated. Accepted — invariant tests cover shape; semantic quality validated when i2p reaches those phases.

## Deferred (write spec only when use case appears)

- `watch_produces` event-react flag.
- `model_pick_log`-driven prompt A/B tuning loop.
- Adding `classifier_picked_agent` column to `tasks` for classifier divergence tracking.
- telegram_bot.py split.
- Shopping cluster reorganization (defer until pipeline_v2 stabilizes per `project_shopping_state_20260505`).
