# Agents Overhaul — Decisions (revised 2026-05-08)

**Source:** `docs/research/2026-05-08-agent-usage-audit.md`
**Plan:** `docs/plans/2026-05-08-agents-overhaul.md`

## TL;DR

**Keep all 21. Polish prompts. Tighten classifier. Per-agent reflection on workhorses.**

User constraints (locked):
- **No merges via mode/phase/artifact_type flags** — meaningless complication.
- **No drops on 0-traffic alone** — i2p hasn't reached late-pipeline agents yet.
- **No alias maps** — workflow JSONs unchanged.

## Roster (no change)

All 21 stay:
analyst, architect, artifact_summarizer, assistant, code_reviewer, coder, deal_analyst, executor, fixer, grader, implementer, planner, product_researcher, researcher, reviewer, shopping_advisor, shopping_clarifier, summarizer, test_generator, visual_reviewer, writer.

## Why distinct (key triplet)

Real role differences confirmed by tools + traffic:

| | coder | implementer | fixer |
|---|---|---|---|
| Position | front (ad-hoc /task) | middle (i2p fan-out from architect) | back (after reviewer feedback) |
| Tools | 17 (incl. git, run_code, scaffold, web_search) | 11 (no git, no run, no web) | 10 (no git, no run, no web, no project_info) |
| Picks/60d | 216 | 13509 ⭐ | 126 |
| Iter budget | 8 | 6 | 8 |
| Self-reflection | True | False (TODO: enable) | False (TODO: enable) |

Tool whitelist is a **safety guardrail** specific to pipeline position. Merging would either expand implementer's blast radius (gets git_commit) or shrink coder's (loses build capability). Both worse than current.

## What ships in this plan

1. **Prompt polish** — MetaGPT-style phrasing (role primer, DO/DON'T, JSON schema) lifted into all 21. Workhorses first, low-traffic last.
2. **Per-agent reflection** — `coder/implementer/fixer/test_generator` get role-specific self-check blocks in coulson.
3. **Classifier rules** — explicit pick/reject per agent so coder/implementer/fixer (etc.) don't get confused.
4. **Invariant test suite** — every prompt has role primer + final_answer JSON schema.

## What doesn't ship

- No file deletes.
- No prompt mode branches.
- No alias resolver.
- No workflow JSON edits.
- No fatih_hoca / coulson runtime changes beyond reflection prompt block.
- Shopping cluster polish: gets baseline invariant + classifier rules, but deeper rework deferred until pipeline_v2 stabilizes (`project_shopping_state_20260505`).

## Risks

| risk | mitigation |
|---|---|
| Prompt edit regresses an agent | invariant tests; DB `prompt_versions` rollback |
| Classifier prompt bloats with 21 agents | cap at 2 lines/agent, drop verbose preamble |
| 0-traffic agents polished blind | invariant tests cover shape; semantic eval deferred to first real call |
| Reflection block doesn't fire because flag plumbing broken | Task 5 audits wiring before flipping flags |
