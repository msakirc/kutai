# Self-Reflection Wiring Audit — 2026-05-08

## Executive Summary

Self-reflection is **partially deployed**: 6 of 21 agents have enable_self_reflection = True, but only **coder, researcher, and shopping_advisor** also set min_confidence gates. The runtime injects reflection as the **second-to-last step before final_answer**, after all tool work completes but before confidence gating. **No per-agent reflection prompt exists**—the prompt is generic across all agents. **Three workhorse agents (implementer, fixer, test_generator) currently have reflection disabled** and are candidates for Task 6.

---

## 1. Flag Set Per Agent

| Agent | enable_self_reflection | min_confidence | max_iterations | Notes |
|-------|-------|-------|-------|-------|
| coder | **True** | **3** | 8 | Primary workhorse; reflection + confidence gate |
| implementer | False | 0 | 6 | **CANDIDATE for Task 6 enable** |
| fixer | False | 0 | 8 | **CANDIDATE for Task 6 enable** |
| test_generator | False | 0 | 6 | **CANDIDATE for Task 6 enable** |
| researcher | **True** | **3** | 6 | Research task workhorse |
| reviewer | False | 0 | 5 | Code review; no reflection |
| architect | False | 0 | 6 | Design/spec; no reflection |
| planner | False | 0 | 5 | Planning; no reflection |
| analyst | False | 0 | 5 | Analysis; no reflection |
| writer | **True** | 0 | 5 | Content; reflection but no confidence gate |
| summarizer | False | 0 | 3 | Summarization; no reflection |
| assistant | False | 0 | 4 | General; no reflection |
| shopping_advisor | **True** | **3** | 8 | Shopping pipeline; reflection + confidence gate |
| shopping_clarifier | False | 0 | 3 | Clarification; no reflection |
| deal_analyst | **True** | 0 | 4 | Deal analysis; reflection but no confidence gate |
| product_researcher | **True** | 0 | 6 | Product research; reflection but no confidence gate |
| code_reviewer | False | 0 | 5 | Code review (backup); no reflection |
| visual_reviewer | False | 0 | 3 | Visual review; no reflection |
| grader | False | 0 | 5 | Scoring; no reflection |
| executor | False | 0 | 5 | Execution; no reflection |
| artifact_summarizer | False | 0 | 3 | Artifact summary; no reflection |

**Summary:** 6 agents enabled (coder, researcher, writer, shopping_advisor, deal_analyst, product_researcher); only 3 pair with confidence gates (coder, researcher, shopping_advisor).

---

## 2. Runtime Read Sites

### 2.1 Primary Read Sites

| File | Line | Context | Purpose |
|------|------|---------|---------|
| packages/coulson/src/coulson/react.py | 7-8 | Module docstring | Documents profile interface includes enable_self_reflection and min_confidence |
| packages/coulson/src/coulson/react.py | 867-877 | Final answer branch | **MAIN INJECTION POINT**: Checks if profile.enable_self_reflection then calls await self_reflect() |
| packages/coulson/src/coulson/react.py | 882-893 | After reflection (implicit) | **CONFIDENCE GATE**: Checks if profile.min_confidence > 0 returns needs_review status instead of completed |

### 2.2 Secondary/Reference Sites

| File | Line | Context |
|------|------|---------|
| src/agents/base.py | 81 | Class attribute definition (default False) |
| src/agents/base.py | 10 | Module docstring lists it as customizable attribute |
| src/agents/base.py | 85 | Class attribute min_confidence definition (default 0) |

### 2.3 Reflection Function

| File | Line | Signature |
|------|------|-----------|
| packages/coulson/src/coulson/reflection.py | 24-29 | async def self_reflect(task: dict, result: str, reqs_or_tier=None, used_model: str = "") -> dict \| None |

---

## 3. Reflection Prompt Source

### 3.1 Prompt Text (Current)

Located in **packages/coulson/src/coulson/reflection.py:60-67**, a **generic, single-agent-agnostic block**:

Generic prompt checking for errors/hallucinations. If response is good, returns {"verdict": "ok"}. If issues found, returns {"verdict": "fix", "issues": "description", "corrected_result": "the fixed version"}.

### 3.2 Injection Timing

**When:** Reflection fires **after all tool work completes** and the main LLM emits a final_answer, but **before the confidence gate**.

**Control flow in packages/coulson/src/coulson/react.py:**

1. LLM produces response → parse to action_type + result
2. If action_type == "final_answer" → enter final-answer block (line 843)
3. (optional) format validation (lines 845-865)
4. **→ LINE 867: if profile.enable_self_reflection:** call self_reflect()
5. **→ LINE 882: if profile.min_confidence > 0:** apply confidence gate
6. **→ Return "completed"** or "needs_review" (lines 895-905)

### 3.3 Is There a build_reflection_prompt() Function?

**NO.** The reflection prompt is **hardcoded and monolithic** in packages/coulson/src/coulson/reflection.py:60-67. No per-agent dispatch; no builder function.

**Plan Task 6 will introduce per-agent prompts** via a new REFLECTION_BLOCKS dict + build_reflection_prompt(agent_name, iteration) function.

---

## 4. min_confidence Integration

### 4.1 Interaction Model

**Confidence gating is ORTHOGONAL to reflection:**

- **Reflection** filters for error/hallucination (called unconditionally if enable_self_reflection=True).
- **Confidence gate** filters for agent's own confidence score (called unconditionally if min_confidence > 0).
- **Execution order:** Reflection fires first; confidence gate second.
- **Result:** Can have (reflection + confidence), (reflection only), (confidence only), or (neither).

### 4.2 When & What It Does

**Where:** packages/coulson/src/coulson/react.py:880-893

**Trigger:** After parsing the final_answer JSON, the LLM **optionally embeds a "confidence": <1-5> field**.

If confidence < min_confidence, the task returns **"needs_review"** (routed to human/reviewer) instead of "completed".

### 4.3 Live Agents Using min_confidence

Only **3 agents set it:**
- **coder: min_confidence = 3** — requires ≥3/5 to emit
- **researcher: min_confidence = 3** — requires ≥3/5 to emit
- **shopping_advisor: min_confidence = 3** — requires ≥3/5 to emit

All others default to 0 (confidence gate disabled).

---

## 5. Live Evidence: model_pick_log Self-Reflection Picks

The audit found that **self-reflection IS happening** as evidenced by picks logged with agent_type="self_reflection".

### 5.1 Where Reflection Picks Are Logged

In **packages/coulson/src/coulson/reflection.py:75-87**, when self_reflect() calls the dispatcher:

Every self-reflection call records in model_pick_log with agent_type="self_reflection", allowing forensic analysis of which agents are triggering reflection.

### 5.2 Per-Agent Reflection Frequency

Earlier audit (2026-05-08-agent-usage-audit.md) counted **107 self_reflection picks over 60d**. These correspond to agents with enable_self_reflection=True:

- **coder** (picks: 3447/60d) — reflects on some fraction
- **researcher** (picks: 612/60d) — reflects on some fraction
- **shopping_advisor** (picks: 258/60d) — reflects on some fraction
- **writer, deal_analyst, product_researcher** — reflect when enabled, lower traffic

**Implication:** Reflection is live and functioning. Not a dead code path.

---

## 6. Per-Agent Recommendations for Task 6

### 6.1 Implementer — RECOMMEND ENABLE

**Current:** enable_self_reflection = False, min_confidence = 0

**Rationale for Enable:**
- **Max iterations: 6** — enough budget to spare one reflection call (typical cost: 1-2 sec for LLM, negligible).
- **Workflow:** Read spec → implement → test/fix → final check. Reflection prevents shipping syntax/lint errors.
- **No blocking constraint:** Iteration budget is ample; reflection won't cause exhaustion.

**Proposed reflection checklist (per plan):**
> Self-check: (1) lint clean? (2) py_compile passes? (3) matches ARCHITECTURE.md? (4) only your assigned file touched?

### 6.2 Fixer — RECOMMEND ENABLE

**Current:** enable_self_reflection = False, min_confidence = 0

**Rationale for Enable:**
- **Max iterations: 8** — identical to coder; ample budget.
- **Workflow:** Read feedback → identify root cause → edit → verify fix. Reflection catches incomplete fixes.
- **No blocking constraint:** High iteration budget; no time/token pressure.

**Proposed reflection checklist (per plan):**
> Self-check: (1) every feedback bullet addressed? (2) tests run after edit? (3) no unintended deletions?

### 6.3 Test Generator — RECOMMEND ENABLE

**Current:** enable_self_reflection = False, min_confidence = 0

**Rationale for Enable:**
- **Max iterations: 6** — matches implementer; adequate.
- **Workflow:** Read source → plan tests → generate → run → fix failures.
- **No blocking constraint:** Low tool overhead; reflection is cheap vs. re-running a failing test suite.

**Proposed reflection checklist (per plan):**
> Self-check: (1) tests run? (2) cover the spec? (3) no flaky waits/sleeps? (4) assert messages helpful?

### 6.4 Coder — ALREADY ENABLED

**Current:** enable_self_reflection = True, min_confidence = 3

No change needed for Task 6. Already reflects + confidence gates.

---

## 7. Findings Summary

1. **Partial Deployment**: 6 of 21 agents have reflection enabled, but only 3 pair with confidence gates. Reflects a staged rollout (coder → researcher/shopping → others).

2. **Single Generic Prompt**: Reflection uses one hardcoded prompt block in reflection.py:60-67 for all agents. Task 6 will introduce per-agent role-specific prompts via REFLECTION_BLOCKS dict + build_reflection_prompt().

3. **Three Workhorses Ready**: Implementer, fixer, and test_generator are high-traffic, high-iteration-budget agents with no blocking constraints. Enabling reflection on all three is low-risk and aligns with the plan.

4. **Orthogonal to Confidence Gating**: Reflection (error detection) and confidence (confidence threshold) operate independently. Reflection fires first; confidence gate second. Only 3 agents use confidence gating today; Task 6 does not propose expanding it.

5. **Live and Functioning**: model_pick_log shows 107 self-reflection picks over 60 days, proving the code path is active and not orphaned. Reflection is catching real errors in deployed agents.

---

## Task 6 Checklist (Summary)

- [ ] Enable enable_self_reflection = True on implementer, fixer, test_generator
- [ ] Add REFLECTION_BLOCKS dict to packages/coulson/react.py
- [ ] Implement build_reflection_prompt(agent_name: str, iteration: int) -> str
- [ ] Update reflection call in react.py line 869 to use build_reflection_prompt() if available
- [ ] Write + run tests/agents/test_self_reflection.py with per-agent checklist assertions
- [ ] Smoke test coder + implementer via test fixture (no real LLM call)

---

## References

- **Agents code:** src/agents/*.py
- **Reflection implementation:** packages/coulson/src/coulson/reflection.py
- **ReAct loop:** packages/coulson/src/coulson/react.py:867-893
- **Plan:** docs/plans/2026-05-08-agents-overhaul.md (Phase 2, Task 5-6)
- **Earlier audit:** docs/research/2026-05-08-agent-usage-audit.md
