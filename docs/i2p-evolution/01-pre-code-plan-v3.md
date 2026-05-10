# Z1 — Pre-code execution plan (v3: prototype-depth)

**Date:** 2026-05-09
**Source doc:** [01-pre-code.md](01-pre-code.md)
**Predecessors:** [01-pre-code-plan.md](01-pre-code-plan.md) (v1), [01-pre-code-plan-v2.md](01-pre-code-plan-v2.md) (v2)
**Reads:** v1, v2, [00-README.md](00-README.md), [z0-mission-preflight.md](z0-mission-preflight.md), `src/workflows/i2p/i2p_v3.json`, `src/workflows/engine/expander.py`, `packages/mr_roboto/src/mr_roboto/__init__.py`, `packages/vecihi/src/vecihi/__init__.py`, `src/agents/reviewer.py`, `src/tools/workspace.py`, `data/kutai.db`, `docs/plans/2026-05-08-agents-overhaul.md`.

v2 was structurally honest but had four load-bearing factual errors and missed three architectural realities that change the shape of multiple proposals. v3 rewalks every proposal against the actual codebase, locks one shape per proposal (no "depends"), inlines the JSON/Python/JSON-step drafts, drafts the reviewer-prompt criteria as bullets, attaches a telemetry plan and locked migration, and adds two new gaps v2 missed.

---

## Diff vs v2

Each line is either a correction (cite v2 line if disagreeing) or a new addition v2 didn't make.

1. **v2 used the wrong workspace path prefix.** v2 wrote `missions/<id>/.evidence/`, `missions/<id>/.brand/`, `missions/<id>/.compliance/`. The actual convention is `workspace/mission_<id>/...` — `src/tools/workspace.py:18` defines `WORKSPACE_DIR` (default `workspace/`) and `:439-445` defines `get_mission_workspace(mission_id)` returning `os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")`. Every `produces` path in v2 is wrong. Fixed throughout v3 to `workspace/mission_<id>/.evidence/index.json` etc. Without this fix, `expander.py:178-181` auto-grounding fires against a path that never resolves; `mr_roboto.check_grounding` (`mr_roboto/__init__.py:69-92`) and `verify_artifacts` (`:94-115`) fail every step.
2. **v2 cited the wrong expander path.** v2 said `src/workflows/i2p/expander.py:178-181`. The file is at `src/workflows/engine/expander.py:178-181` — verified. The Z1 source doc (01-pre-code.md) inherits this misnomer indirectly. Note: v2's *line numbers* (178-181) are correct in the actual file; only the directory was off. Cite as `src/workflows/engine/expander.py:178-181` going forward.
3. **v2 underspecified the reviewer-prompt surface.** v2 said "Modify reviewer prompts at 3.11 + 4.16" without naming where the prompts live. They are NOT in `src/agents/reviewer.py` (that file at `:31-91` is a generic prompt for ad-hoc reviews). Step-specific reviewer prompts are inline in `i2p_v3.json[<step>].instruction` — see `i2p_v3.json:2531` for `3.11.instruction`. So "update reviewer prompt" means "edit the `instruction` string of the reviewer step in i2p_v3.json", not "edit `src/agents/reviewer.py`". v3 inlines the exact instruction-string deltas.
4. **`request_review` action does not exist.** Z1 source doc proposed `request_review` as a "first-class action distinct from needs_clarification." Verified: only `needs_clarification` exists in `packages/coulson/src/coulson/parsing.py:319`, `react.py:857-860`, `react.py:1432`, and `general_beckman/result_router.py:148`. v2 doesn't call this out; v3 explicitly defers `request_review` to a separate proposal (P10) with engine-extension scope, and refuses to assume it for the six core proposals.
5. **No real i2p missions exist in the DB.** v2's failure walkthroughs cite "mission 57" from CLAUDE.md memory but the DB query `SELECT id, title FROM missions` returns one row: `(1, 'Shopping: gaming mice', ...)`. Blackboards exist for missions 42/100/101 but contain stub data only (`{'spec': 'Spec content here'}`, `{'analysis': 'Analysis result text'}`). Real failure-mode walkthroughs against the proposals are necessarily synthetic — labeled `[synthetic]` throughout v3. Mission 57 in memory is a *historical* shopping mission, not i2p; the "narration pattern impossible" entry refers to the G grounding fix, not a Z1 phase failure. v2 implies real missions are available; v3 makes the synthetic source explicit.
6. **v2 said reviewers will "silently regress" but didn't show the prompt deltas.** v3 inlines the instruction-string deltas as concrete additions to existing prompts (e.g., "Append to existing reviewer instruction at `i2p_v3.json:2531`: 'Reject when ...' "). Without this concreteness, reviewer prompt regression remains a hand-wave even with proposal 7's fixture suite.
7. **v2 missed: `general_beckman/result_router.py:148-155`** routes `needs_clarification` back to `waiting_human` only when keyboard sent. New mechanical actions that surface compliance/evidence gaps via Telegram must follow the same shape — return `Action(status="needs_clarification", result={..., "keyboard_sent": True})` per the precedent at `mr_roboto/__init__.py:227-244`. v2 doesn't say this; v3 makes it part of every wizard-style action's signature.
8. **v2 missed: `prompt_versions` table exists.** From `data/kutai.db` schema: `prompt_versions` is a real table. Per `docs/plans/2026-05-08-agents-overhaul.md` "Architecture" section, agents-overhaul uses it for live prompt swap. Reviewer-prompt iteration in v3 lands as `prompt_versions` rows, not direct i2p_v3.json edits — this preserves rollback per existing `/prompt save` flow.
9. **v2 risk R6 (agents-overhaul churn) is partially closed.** Reading `docs/plans/2026-05-08-agents-overhaul.md` line 13 explicitly: "All 21 agents stay as-is (post-2026-05-04 kill-agents pure config). No file deletes, no alias maps, no payload mode/phase flags." So agent names referenced from i2p_v3.json (`analyst`, `researcher`, `architect`, `reviewer`, `writer`, `mechanical`) are stable. R6 reduces from blocking to advisory. v3 retains R6 only for the prompt-polish landing window (Task 2 of agents-overhaul edits the same workhorse prompts).
10. **v2 missed: 4.14 `instruction` says "Each ADR: title, status, context, decision, consequences"** — only Nygard's five fields. Schema fields list at `i2p_v3.json:3165-3171` matches. So adding `options_considered`, `falsification_signal`, `chosen_option_id`, `reversal_cost` requires both schema AND instruction edits in lockstep. v2 mentions instruction edit but doesn't show the diff; v3 inlines it.
11. **v2 missed: tasks rows store `agent_type`, not "agent name from JSON"** — `tasks.agent_type` is a TEXT column (verified via `PRAGMA table_info(tasks)`) and gets `"mechanical"` for mr_roboto-routed steps. The mechanical-flag detection in `expander.py:212` checks `step.get("executor") == "mechanical" or agent_name == "mechanical"`. New mechanical step IDs (e.g., compliance wizard) must declare `agent: "mechanical"` AND set `executor: "mechanical"` AND emit `context.executor: "<action_name>"` for the mr_roboto routing to fire. v2's wiring of `0.4a` and `5.5b1` lacks this; v3 corrects.
12. **v2 missed: cost numbers should account for retry envelope.** Beckman's worker_attempts max defaults to 3-5 (DB column `max_worker_attempts`). A 30k-token step that exhausts retries is 90-150k tokens. v2's per-mission token cost estimates assume single-pass. v3 multiplies high-uncertainty steps (intake extraction, prior-art verdict) by ×2 expected retries.
13. **v2 missed: there is no `request_evidence_uploads` Telegram precedent.** `notify_user` (`mr_roboto/__init__.py:246-252`) sends one-way. `clarify` (`:227-244`) sends a question + waits. There is no "open upload window" infrastructure. v2 invents this without citing it. v3 reframes intake as a `clarify`-style multi-question collection, with file uploads handled out-of-band by the founder dropping files into `workspace/mission_<id>/.evidence/` (or via a new `/evidence` slash command, deferred as P10b).
14. **v2 missed: `produces` validation in expander.** `expander.py:160-165` filters via `_is_valid_produces_entry` — must verify what shapes are accepted. Path strings with `<id>` placeholders may be filtered. v3 specifies that `produces` MUST be the resolved path string (mission_id substituted at expansion time, not literal `<id>`).

---

## Mission-grounded reality check

Real-mission evidence is sparse. The only non-stub mission in `data/kutai.db` is mission 1 (shopping), and none of the i2p_v3 phases 0-6 have observable execution traces. Synthetic missions, labeled, are the basis for failure walkthroughs.

### Mission 1 (`shopping: gaming mice`) — real, but tangential
- Status: `active` since 2026-04-23.
- Tasks: 9 rows, all `pending` with `worker_attempts=0`. The mission appears to be a stale/test row from the shopping pipeline overhaul; no useful Z1 signal.
- Lesson: **the mission DB does not retain phase 0-6 artifact traces.** Artifacts go to `workspace/mission_<id>/...` files; the blackboard table holds a small pinned digest only (mission 42/100/101 blackboards are 168-195 bytes each; real ADR text would be >1KB).
- Implication for v3: every "audit" gate must resolve the artifact via filesystem (`workspace/mission_<id>/...`), not a DB join. The `verify_artifacts` action precedent at `mr_roboto/__init__.py:94-115` uses `paths` parameter with workspace_path — this is the correct contract.

### Mission 57 [CLAUDE.md memory reference, NOT in current DB]
Per memory `project_g_grounding_shipped_20260506.md`: mission 57 was where the "narration pattern" surfaced — agent claimed to write files but the workspace was empty. This is the canonical "produces declared, nothing written" failure. v3's evidence-index check (P1) and ADR-completeness check (P3) follow this pattern: declare paths as `produces`, let auto-grounding (`expander.py:178-181`) fire, and back the path resolution with a content-shape check via a new `*_check` mechanical action.

### [synthetic] Mission F1 — "Turkish reseller arbitrage tool"
Founder posts to Kutay: "I want a tool that scans Trendyol/Hepsiburada for under-listed items and tells me to buy + relist on N11." z0 collects: founder=solo, ambition=`private_beta`, jurisdictions=`[TR]`, user_classes=`[b2b, individual_seller]`, budget=`100 USD/mo cloud`. Phase 0 of i2p starts:
- 0.1 emits `idea_brief` from founder pitch only — no interview transcripts. Persona "Mehmet, weekend reseller" is hallucinated from "common reseller archetype" training echo.
- 0.2 problem_statement: severity="high", frequency="daily" — no source.
- 0.3 assumption_identification correctly fires `risk_if_wrong=high` on "resellers will pay 49TL/mo for this." `validation_method` says "user interviews" — but no intake mechanism.
- 0.5 clarification asks 5 questions — none of them flag the assumption-evidence gap.
- 1.3 competitive_landscape misses Hepsiburada's own seller tools (recently added) and 3 dead competitors (`SellerSpy.tr`, `ArbitrajPro`, `n11boost`) — none of which exist in LLM training data.
- 4.1 picks NextJS+Supabase; alternatives lost.
- 4.4 says "PostgreSQL 3NF"; no ADR asks why not BigQuery for the price-history table.
- Phase 6 lock; founder builds for two weeks; phase 7+ fails because compliance phase reveals KVKK (Turkish GDPR) requires data residency in TR — Supabase eu-west-1 violates. Mission rolls back to phase 4. Multi-week loss.

This synthetic walkthrough is the canonical failure-mode pattern Z1 must prevent. Each proposal below cites where it would have intervened.

### [synthetic] Mission F2 — "B2B health-data analytics for under-13 clinics"
Founder posts: "Pediatric obesity tracking for clinics in EU+US." z0 ambition=`public_launch`, jurisdictions=`[EU, US]`, user_classes=`[health, children]`. Today's i2p:
- 0.4 scope_ambiguity_detection has no compliance category → never asks "are you HIPAA-covered or BAA-required?"
- 1.11 regulatory_research is a "fish for issues" prompt — produces a generic GDPR/HIPAA paragraph, no scoping by `<jurisdiction × user_class × data_category>`.
- Phase 4 picks Vercel+Postgres+Auth0. Auth0 is fine; Vercel ToS forbids HIPAA without enterprise contract. Postgres unencrypted-at-rest is fine for free tier; HIPAA requires it.
- Phase 13 (compliance) discovers all three. Mission stalls.

Compliance-fingerprint (P6) prevents this by: (a) `0.4a` collects `data_categories_coarse=[health, minors_data]`, (b) `1.11a` emits `compliance_overlay.required_documents` including a `data_processor_compatibility` doc that flags Vercel free tier as non-HIPAA before phase 4 picks it.

---

## Per-proposal prototype-depth

### P1. Structured user-research intake (closes G1, G7)

#### Re-audit of v2
v2's audit of step IDs is correct. v2's `0.0a/0.0b` insertion ordering is correct. v2's wiring to mechanical action `evidence_index_check` is correct in concept. **Errors v3 fixes:**
- Path prefix wrong (Diff #1): `workspace/mission_<id>/.evidence/index.json`, not `missions/<id>/...`.
- v2 invents `request_evidence_uploads` Telegram action without precedent (Diff #13). v3 reframes as `clarify`-shape question: "Drop interview/screenshot files into your mission workspace and reply DONE; or reply SKIP."
- v2's `0.0b evidence_extraction` agent type is `analyst` with `tools_hint: [read_file, vision]`. Verified those tools exist (`src/tools/workspace.py:183 read_file`, vision tool referenced in v2). Correct.

#### Chosen shape (locked)
Two new steps: `0.0a evidence_intake_request` (mechanical) + `0.0b evidence_extraction` (analyst LLM). Universal `evidence_refs: list[str]` field added to persona/FR/BR/feature/value-prop-pain artifacts. Sentinel `["agent_inference"]` allowed at `prototype` ambition; rejected at `private_beta+` (warning only, not blocker — see Severity below).

#### Inline JSON artifact examples

**evidence_index — POPULATED state (mission F1, founder dropped one interview transcript + screenshot of competitor):**
```json
{
  "_schema_version": "1",
  "mission_id": 57,
  "evidence_items": [
    {
      "evidence_id": "EV-001",
      "source_type": "interview",
      "source_uri": "workspace/mission_57/.evidence/EV-001.txt",
      "captured_at": "2026-05-09T10:14:00Z",
      "founder_note": "30min call with Hakan, runs 4 Trendyol stores",
      "redaction_status": "founder_consented",
      "sha256": "9a3c8b7e..."
    },
    {
      "evidence_id": "EV-002",
      "source_type": "screenshot",
      "source_uri": "workspace/mission_57/.evidence/EV-002.png",
      "captured_at": "2026-05-09T10:18:00Z",
      "founder_note": "SellerSpy.tr landing page, Sept 2023 wayback",
      "redaction_status": "raw",
      "sha256": "f1d4..."
    }
  ],
  "founder_skipped": false,
  "skip_reason": null
}
```

**evidence_index — EMPTY (founder skipped):**
```json
{
  "_schema_version": "1",
  "mission_id": 57,
  "evidence_items": [],
  "founder_skipped": true,
  "skip_reason": "First mission, building from gut. Will validate with users post-prototype."
}
```

**evidence_index — PARTIAL (founder uploaded 1 file, said DONE before intended):**
```json
{
  "_schema_version": "1",
  "mission_id": 57,
  "evidence_items": [
    {
      "evidence_id": "EV-001",
      "source_type": "voice_memo",
      "source_uri": "workspace/mission_57/.evidence/EV-001.m4a",
      "captured_at": "2026-05-09T10:14:00Z",
      "founder_note": null,
      "redaction_status": "raw",
      "sha256": "..."
    }
  ],
  "founder_skipped": false,
  "skip_reason": null
}
```
The "partial" state is intentionally indistinguishable from "complete with one item" — the system honors founder intent. The `0.0b` extractor's confidence on a single voice memo is low; downstream `evidence_refs_audit` flags missions where ≥80% of FRs cite the same EV-NNN as low diversity.

**extracted_evidence (per evidence item):**
```json
{
  "_schema_version": "1",
  "evidence_id": "EV-001",
  "extracted_claims": [
    {
      "claim_id": "EV-001.C1",
      "claim": "Hakan checks stock 3x daily by hand",
      "claim_category": "current_tool",
      "quote": "I literally have a notebook. Every morning, after lunch, before bed.",
      "confidence": "high",
      "extracted_at": "2026-05-09T10:30:00Z"
    },
    {
      "claim_id": "EV-001.C2",
      "claim": "Pays 200TL/mo for a generic price tracker that doesn't filter for relisting margin",
      "claim_category": "willingness_to_pay",
      "quote": "200 lira a month for [tool name redacted]. Useless for what I do.",
      "confidence": "high",
      "extracted_at": "2026-05-09T10:30:00Z"
    }
  ]
}
```

**evidence_refs field examples (added to existing persona/FR/feature items):**
```json
// On a persona item in 2.4 user_personas:
{
  "name": "Mehmet, weekend reseller",
  "...other persona fields...": "...",
  "evidence_refs": ["EV-001.C1", "EV-001.C2"],
  "is_inference": false
}

// On a hallucinated persona at prototype tier:
{
  "name": "Casual side-hustler",
  "evidence_refs": ["agent_inference"],
  "is_inference": true
}
```

#### Inline Python pseudo-code for hooks/tools

**`evidence_index_check` mechanical action** — lands as new file `packages/mr_roboto/src/mr_roboto/evidence_index_check.py`; case branch added in `mr_roboto/__init__.py` between current `:346` and `:348` (before unknown-action return). Pseudo-code:

```python
# packages/mr_roboto/src/mr_roboto/evidence_index_check.py
import json, hashlib, os
from mr_roboto.actions import Action
from src.tools.workspace import get_mission_workspace

async def evidence_index_check(task: dict) -> Action:
    payload = task.get("payload") or {}
    mission_id = task["mission_id"]
    ws = get_mission_workspace(mission_id)
    index_path = os.path.join(ws, ".evidence", "index.json")
    if not os.path.isfile(index_path):
        return Action(
            status="failed",
            error=f"evidence_index_check: missing {index_path}",
        )
    try:
        idx = json.loads(open(index_path, encoding="utf-8").read())
    except Exception as e:
        return Action(status="failed", error=f"evidence_index_check: parse {e}")
    if idx.get("_schema_version") != "1":
        return Action(status="failed",
                      error=f"evidence_index_check: schema_version={idx.get('_schema_version')}")
    items = idx.get("evidence_items") or []
    if not items and not idx.get("founder_skipped"):
        return Action(status="failed",
                      error="evidence_index_check: empty items and not founder_skipped")
    for it in items:
        src = os.path.join(ws, os.path.relpath(it["source_uri"], ws)) \
              if it["source_uri"].startswith("workspace/") \
              else os.path.join(ws, it["source_uri"])
        if not os.path.isfile(src):
            return Action(status="failed",
                          error=f"evidence_index_check: source_uri {it['source_uri']} not found")
        actual = hashlib.sha256(open(src, "rb").read()).hexdigest()
        if actual != it.get("sha256"):
            return Action(status="failed",
                          error=f"evidence_index_check: sha256 mismatch on {it['evidence_id']}")
    return Action(status="completed", result={"items": len(items),
                                              "skipped": idx.get("founder_skipped")})
```

**Dispatch entry in `mr_roboto/__init__.py`** (added before line 348):
```python
    if action == "evidence_index_check":
        from mr_roboto.evidence_index_check import evidence_index_check
        return await evidence_index_check(task)
```

**`evidence_refs_audit` mechanical action** — sibling file. Pseudo-code:
```python
# packages/mr_roboto/src/mr_roboto/evidence_refs_audit.py
async def evidence_refs_audit(task: dict) -> Action:
    payload = task.get("payload") or {}
    mission_id = task["mission_id"]
    artifact_name = payload["artifact_name"]   # e.g. "user_personas"
    threshold = float(payload.get("threshold_inference_pct", 0.5))
    tier = payload.get("ambition_tier", "prototype")
    art = await load_artifact(mission_id, artifact_name)   # via engine/artifacts.py
    items = art if isinstance(art, list) else art.get("items") or []
    if not items:
        return Action(status="completed", result={"reason": "no items"})
    n_inf = sum(1 for it in items
                if it.get("is_inference") is True
                or it.get("evidence_refs") == ["agent_inference"])
    pct = n_inf / len(items)
    severity = "warning"  # never blocker — see "blockers/levels" pattern shipped 2026-05-05
    triggered = (pct > threshold) and tier in ("private_beta", "public_launch", "revenue_product")
    return Action(
        status="completed",
        result={
            "ok": True,
            "inference_pct": pct,
            "severity": severity if triggered else "ok",
            "blockers": [{"field": f"{artifact_name}.evidence_refs",
                          "severity": "warning",
                          "reason": f"{int(pct*100)}% items lack evidence"}] if triggered else [],
            "levels": ["critical", "high"]   # warning is below — surface only
        },
    )
```

**Founder intake (clarify-shape)** — reused, no new mechanical action. Step `0.0a`'s context:
```json
{
  "executor": "clarify",
  "kind": "evidence_intake",
  "questions": [
    "Reply with the kinds of evidence you have, or DONE / SKIP.",
    "Drop files into workspace/mission_{mission_id}/.evidence/ then reply DONE."
  ],
  "ack_keywords": ["DONE", "SKIP"]
}
```
This relies on `mr_roboto.clarify` (existing, `mr_roboto/__init__.py:227-244`) returning `Action(status="needs_clarification", result={..., "keyboard_sent": True})` until the founder replies. After reply, `0.0a` re-runs and the post-hook `evidence_index_check` validates whatever the founder dropped (or that `founder_skipped: true`).

#### i2p_v3.json step JSON drafts

**New step `0.0a` — mechanical evidence intake:**
```json
{
  "id": "0.0a",
  "phase": "phase_0",
  "name": "evidence_intake_request",
  "agent": "mechanical",
  "executor": "mechanical",
  "difficulty": "trivial",
  "tools_hint": [],
  "depends_on": [],
  "may_need_clarification": true,
  "input_artifacts": [],
  "output_artifacts": ["evidence_index"],
  "produces": ["workspace/mission_{mission_id}/.evidence/index.json"],
  "post_hooks": ["evidence_index_check"],
  "instruction": "(mechanical — see context.executor)",
  "done_when": "evidence_index.json exists with _schema_version=1 and items written or founder_skipped=true.",
  "context": {
    "executor": "clarify",
    "kind": "evidence_intake",
    "questions": [
      "Want to upload founder evidence (interview notes, surveys, screenshots, voice memos)? Drop files into workspace/mission_{mission_id}/.evidence/ then reply DONE. Reply SKIP to proceed without."
    ]
  },
  "skip_when": "ambition_tier == 'prototype' and founder_evidence_skipped_default == true"
}
```

**New step `0.0b` — analyst extraction:**
```json
{
  "id": "0.0b",
  "phase": "phase_0",
  "name": "evidence_extraction",
  "agent": "analyst",
  "difficulty": "medium",
  "tools_hint": ["read_file", "vision"],
  "depends_on": ["0.0a"],
  "may_need_clarification": false,
  "input_artifacts": ["evidence_index"],
  "output_artifacts": ["extracted_evidence", "pitch_evidence_conflicts"],
  "produces": [
    "workspace/mission_{mission_id}/.evidence/extracted.json",
    "workspace/mission_{mission_id}/.evidence/pitch_conflicts.json"
  ],
  "instruction": "For each item in evidence_index.evidence_items: read source_uri (use vision tool for images, read_file for text). Extract claims into extracted_evidence with claim_id=<EV-NNN>.C<M>, claim_category from enum, verbatim quote when possible, confidence from {high,medium,low}. Then compare extracted claims against the founder pitch in idea_brief and produce pitch_evidence_conflicts for any contradiction (severity: high|medium|low).",
  "done_when": "extracted_evidence and pitch_evidence_conflicts both exist and validate against schema.",
  "artifact_schema": {
    "extracted_evidence": {
      "type": "array",
      "min_items": 0,
      "item_fields": ["evidence_id", "extracted_claims"]
    },
    "pitch_evidence_conflicts": {
      "type": "array",
      "min_items": 0,
      "item_fields": ["conflict_id", "pitch_claim_quote", "evidence_claim_quote",
                      "evidence_id", "severity", "agent_resolution_proposal"]
    }
  },
  "skip_when": "evidence_index.founder_skipped == true"
}
```

**Modified step `0.5` — append to existing instruction (`i2p_v3.json:824-851`):** append clause: `"When pitch_evidence_conflicts contains items with severity='high', surface up to 3 of those as the FIRST priority items in your top-5 question budget; only then fill from open_questions_list and assumption_list."` Add `pitch_evidence_conflicts` to `input_artifacts`.

**Modified steps `2.4`, `2.5`, `2.2`, `1.7`, `3.1`, `3.7`** — add `evidence_refs` (and `is_inference` boolean) to the per-item `item_fields` in `artifact_schema`. Append to `instruction`: `"Each item MUST cite evidence_refs as a list of EV-NNN.C<M> ids from extracted_evidence; if no evidence supports the claim, set evidence_refs=['agent_inference'] AND is_inference=true."` Add `extracted_evidence` to `input_artifacts`.

**Add `post_hooks: ["evidence_refs_audit"]`** to steps `2.4`, `2.5`, `3.1`, `3.7`. The post-hook runs via the same engine path that wires `grounding` (auto-prepended) and `verify_artifacts` (declared). Each hook needs a payload that names which artifact to audit; v3 specifies that `expander.py` already propagates `step.context` through to dispatch, so context like:
```json
"context": {
  "evidence_refs_audit": {"artifact_name": "user_personas", "threshold_inference_pct": 0.5}
}
```
…is the right shape. The mechanical handler reads `task["payload"]["evidence_refs_audit"]` (mr_roboto routes the post-hook payload). Verify against `expander.py:204-207` step_ctx propagation — confirmed.

#### Reviewer prompt criteria additions

**`3.11 requirements_review` instruction** — `i2p_v3.json:2531`. Append after the existing schema clause:
```
Additional checks (added v3):
- Reject (status=fail) if any functional_requirement has evidence_refs == [] (empty list — neither agent_inference nor founder evidence cited).
- Issue (status=needs_revision) when >40% of functional_requirements have is_inference=true at private_beta+ tier.
- For each functional_requirement with is_inference=true, the description MUST include the phrase "agent inference" or "founder did not supply evidence" — flag if missing.
```

**`1.13 research_quality_review` instruction** — append:
```
Additional checks (added v3):
- Reject (verdict=fail) if value_proposition_canvas pains/gains lack evidence_refs.
- Reject if user_personas inferred (is_inference=true) outnumber evidence-backed personas at private_beta+ tier.
```

#### Failure walkthroughs

**[synthetic] Mission F1 — would P1 have caught it?** Yes:
- `0.0a` would have prompted founder for evidence; founder's "Hakan call" recording becomes `EV-001.m4a`.
- `0.0b` extracts: `EV-001.C2 = "Pays 200TL/mo for [tool name redacted]. Useless for what I do."` — claim_category=`willingness_to_pay`, confidence=`high`.
- `2.4 user_personas` Mehmet persona now cites `evidence_refs: ["EV-001.C1"]` with `is_inference: false`.
- `evidence_refs_audit` on the personas array reports `inference_pct: 0.5` (only Mehmet has evidence, hallucinated "Casual side-hustler" doesn't). At `private_beta` tier: severity=warning (not blocker). The reviewer at `1.13` would mark `needs_revision` per the new prompt clause.
- Founder iterates or accepts. Either way: spec is now distinguishable between empirical and inferred; downstream phases can prioritize.

**[synthetic] Mission F2 — would P1 have caught the compliance failure?** No — that's P6 territory. P1 would catch the missing user-research evidence (no clinical interviews), but the Vercel-HIPAA mismatch needs compliance fingerprint.

#### Engine extension required
- `mr_roboto/__init__.py`: add 2 dispatch branches (`evidence_index_check`, `evidence_refs_audit`) before `:348`. **No engine code changes** — mechanical actions extend by file addition.
- `expander.py`: NO change needed; existing step_ctx propagation (`:204-207`) handles the post-hook payload.
- `src/workflows/engine/artifacts.py`: must support reading the new `evidence_index` and `extracted_evidence` artifacts. Verify `ArtifactStore.get(name)` resolves them — they're declared in `output_artifacts`, should auto-register.

#### Telemetry plan
- New yazbunu event class: `z1_evidence` with fields `{mission_id, evidence_count, founder_skipped, inference_pct, ambition_tier, step_id}`. Emit from `evidence_index_check` and `evidence_refs_audit`.
- New DB column on `tasks`: optional `extra_metrics_json TEXT NULL` if not already present (verify via `PRAGMA table_info(tasks)` — column not in current schema, requires migration). Alternative: pipe to `pending_llm_summaries` (existing table) keyed on `(mission_id, step_id)`.
- Dashboard query: `SELECT mission_id, AVG(inference_pct) FROM ... GROUP BY mission_id` — track which missions are evidence-poor.
- Locked: emit to yazbunu only (no DB column). Yazbunu file `logs/yazbunu/z1_evidence.jsonl` rotates per existing pattern. Dashboard reads that.

#### Migration concrete
- New tables: NONE. Evidence lives on filesystem, indexed via `evidence_index.json`.
- Schema-version marker: every artifact this proposal touches gains `_schema_version: "1"`. Mission-level marker: add column `missions.legacy_pre_evidence INTEGER DEFAULT 0`. Migration script: `scripts/migrations/2026-05-XX-evidence-refs-backfill.py` sets `legacy_pre_evidence=1` for all existing missions; reviewers gated on that column accept `evidence_refs == []` for legacy.
- Set site: migration script runs once at deploy.
- Read site: `1.13` and `3.11` reviewer instructions append: `"If the mission has legacy_pre_evidence=1 (passed via context.mission_flags), do not enforce evidence_refs presence."` The mission flag is loaded by `general_beckman.apply` and passed through task context.

#### Locked alternatives
- **Chose A (dedicated steps + universal field) over B (inline in 0.1 pitch).** Scenario where B beats A: founder writes a long, structured pitch with `## Evidence` section already containing 5 quotes — A's intake step is friction. Decision: ship A; provide an opt-out via `0.0a skip_when: founder_pitch_has_evidence_section` (a context check that scans `idea_brief` for an `## Evidence` heading). B becomes A's degenerate path, not a fork.
- **Chose mechanical-clarify shape over invented `request_evidence_uploads`.** Scenario where invented tool would beat clarify: founder needs a guided form with field names + dropdowns. Decision: defer the form to a P10b follow-up; the clarify shape is sufficient for v3.

---

### P2. Taste delegation (mood boards + tone-of-voice)

#### Re-audit of v2
v2 is correct that `5.5` is the existing `wireframe_review` step (verified `i2p_v3.json:3540-3572`). v2's renumber to `5.5b1` is fine. **Errors v3 fixes:**
- Path prefix wrong (Diff #1).
- v2 declares `vecihi.scrape_image(url, max_dim_px=1024)` as a new module. Vecihi's `scrape_url` returns HTML/text — image saving requires a separate path. v3 names this as a new helper function in `packages/vecihi/src/vecihi/scrape_image.py` that wraps `scrape_url` for the GET, parses the response bytes, validates content-type, saves under workspace.
- v2 doesn't note that `5.5b1`'s `produces` path with placeholders must resolve at expansion time. v3 specifies `produces: ["workspace/mission_{mission_id}/.brand/index.json"]` and confirms `expander.py:160-165` accepts the resolved string after substitution.

#### Chosen shape (locked)
Insert `5.5b1 brand_direction_proposals` between `5.5` and `5.6`. Generates 3-5 mood boards (scraped images via `vecihi.scrape_image`). `triggers_clarification: true` so founder picks via Telegram. Modify `5.6` to consume `selected_brand_direction`. Modify `5.10 design_review` reviewer prompt to audit attribution.

For ambition tier `prototype`: skip 5.5b1, use `data/brand_templates/<style>.json` library (10 pre-built directions). For `private_beta+`: full 5.5b1.

#### Inline JSON artifact examples

**brand_direction_options — POPULATED (3 of 5 boards rendered):**
```json
{
  "_schema_version": "1",
  "step_id": "5.5b1",
  "options": [
    {
      "option_id": "BD-001",
      "name": "Calm Professional",
      "rationale": "Anchors to Hakan-style persona ‘weekend reseller’ — values reliability over flash. Echoes Stripe Atlas / Linear minimalism.",
      "mood_board": [
        {
          "image_url": "workspace/mission_57/.brand/BD-001/img-1.png",
          "source_url": "https://dribbble.com/shots/12345-calm-dashboard",
          "attribution_required": true,
          "fetched_at": "2026-05-09T11:00:00Z",
          "sha256": "ab12...",
          "license_signal": "unknown"
        }
      ],
      "tone_samples": [
        {
          "voice_attribute": "concise",
          "sample_copy_blocks": [
            "Your stock dipped below threshold on 4 SKUs.",
            "Re-list price set to TRY 245 (margin 23%)."
          ]
        }
      ],
      "risks": ["May read as cold to first-time founders"]
    },
    {
      "option_id": "BD-002",
      "name": "Bazaar Energy",
      "rationale": "Turkish reseller cultural fit; warm reds, market-stall typography.",
      "mood_board": [
        {
          "image_url": "workspace/mission_57/.brand/BD-002/img-1.png",
          "source_url": "https://dribbble.com/shots/67890",
          "attribution_required": true,
          "fetched_at": "2026-05-09T11:01:00Z",
          "sha256": "cd34...",
          "license_signal": "unknown"
        }
      ],
      "tone_samples": [
        {"voice_attribute": "warm",
         "sample_copy_blocks": ["Hoş geldiniz! 4 ürününüz tükenmek üzere."]}
      ],
      "risks": ["Visual style may date faster"]
    }
  ],
  "min_options": 3,
  "max_options": 5,
  "regeneration_count": 0
}
```

**brand_direction_options — PARTIAL (2 of 5 boards failed scrape):**
Same shape but one option has `mood_board: [{"image_url": null, "fetch_failed": true, "source_url": "...", "fetched_at": "...", "tier_attempted": "browser"}]`. The mechanical post-hook `image_attribution_present` allows up to 1 failed image per option as long as ≥3 options have ≥1 image each.

**selected_brand_direction:**
```json
{
  "_schema_version": "1",
  "option_id": "BD-002",
  "founder_note": "Bazaar energy fits the audience. Tone down the red 20%.",
  "selected_at": "2026-05-09T11:30:00Z",
  "rejected_options": ["BD-001", "BD-003", "BD-004", "BD-005"]
}
```

#### Inline Python pseudo-code

**`vecihi.scrape_image`** — new file `packages/vecihi/src/vecihi/scrape_image.py`:
```python
# packages/vecihi/src/vecihi/scrape_image.py
import os, hashlib
from vecihi import scrape_url, ScrapeTier
from src.tools.workspace import get_mission_workspace

ALLOWED_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_BYTES = 5 * 1024 * 1024

async def scrape_image(url: str, mission_id: int, option_id: str,
                       index: int, max_tier: ScrapeTier = ScrapeTier.STEALTH) -> dict:
    res = await scrape_url(url, max_tier=max_tier, return_bytes=True)
    if not res.ok or not res.bytes_payload:
        return {"image_url": None, "fetch_failed": True, "source_url": url,
                "tier_attempted": res.tier_used.name.lower(),
                "fetched_at": res.fetched_at_iso}
    ct = (res.content_type or "").split(";", 1)[0].strip().lower()
    if ct not in ALLOWED_MIMES:
        return {"image_url": None, "fetch_failed": True, "source_url": url,
                "reason": f"bad_content_type:{ct}", "fetched_at": res.fetched_at_iso}
    if len(res.bytes_payload) > MAX_BYTES:
        return {"image_url": None, "fetch_failed": True, "source_url": url,
                "reason": "too_large", "fetched_at": res.fetched_at_iso}
    ext = {"image/png": ".png", "image/jpeg": ".jpg",
           "image/webp": ".webp", "image/gif": ".gif"}[ct]
    ws = get_mission_workspace(mission_id)
    out_dir = os.path.join(ws, ".brand", option_id)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"img-{index}{ext}")
    with open(out, "wb") as f:
        f.write(res.bytes_payload)
    sha = hashlib.sha256(res.bytes_payload).hexdigest()
    return {
        "image_url": os.path.relpath(out, start="."),
        "source_url": url,
        "attribution_required": True,
        "fetched_at": res.fetched_at_iso,
        "sha256": sha,
        "license_signal": "unknown"
    }
```
**Note:** Vecihi's existing `scrape_url` (`packages/vecihi/src/vecihi/fetchers.py`) currently returns parsed text. The `return_bytes=True` parameter is a small extension required to return raw bytes + content-type. v3 names this as a Vecihi extension: 1 new optional kwarg, ~20 LOC change in `fetchers.py`.

**`image_attribution_present` mechanical action** — new file:
```python
# packages/mr_roboto/src/mr_roboto/image_attribution_present.py
async def image_attribution_present(task: dict) -> Action:
    payload = task.get("payload") or {}
    mission_id = task["mission_id"]
    art = await load_artifact(mission_id, "brand_direction_options")
    options = art.get("options") or []
    if len(options) < art.get("min_options", 3):
        return Action(status="failed",
                      error=f"image_attribution_present: only {len(options)} options")
    options_with_image = 0
    for opt in options:
        boards = opt.get("mood_board") or []
        ok_images = [b for b in boards
                     if b.get("image_url") and b.get("source_url")
                     and b.get("sha256")]
        if ok_images:
            options_with_image += 1
            for b in ok_images:
                ws = get_mission_workspace(mission_id)
                p = os.path.join(ws, os.path.relpath(b["image_url"], start="workspace/"))
                # accept either relative-to-cwd or relative-to-workspace
                p = b["image_url"] if os.path.isfile(b["image_url"]) else p
                if not os.path.isfile(p):
                    return Action(status="failed",
                                  error=f"image_attribution_present: file missing {b['image_url']}")
                actual = hashlib.sha256(open(p, "rb").read()).hexdigest()
                if actual != b["sha256"]:
                    return Action(status="failed",
                                  error=f"image_attribution_present: sha256 mismatch")
    if options_with_image < 3:
        return Action(status="failed",
                      error=f"image_attribution_present: {options_with_image}/3 options have images")
    return Action(status="completed", result={"options_ok": options_with_image})
```

#### i2p_v3.json step JSON drafts

**New step `5.5b1`:**
```json
{
  "id": "5.5b1",
  "phase": "phase_5",
  "name": "brand_direction_proposals",
  "agent": "analyst",
  "difficulty": "medium",
  "tools_hint": ["web_search", "scrape_url", "scrape_image", "vision"],
  "depends_on": ["5.5", "2.1", "2.4", "1.8"],
  "may_need_clarification": true,
  "input_artifacts": ["product_vision", "user_personas",
                      "competitor_ux_evaluation_summary"],
  "output_artifacts": ["brand_direction_options", "selected_brand_direction"],
  "produces": ["workspace/mission_{mission_id}/.brand/index.json"],
  "post_hooks": ["image_attribution_present"],
  "instruction": "Generate 3-5 brand directions, each anchored to a specific persona and tone. For each: fetch 2-4 reference images via scrape_image (NOT generated). Save under workspace/mission_{mission_id}/.brand/<option_id>/. Tone samples: 2 short copy blocks per voice attribute. Trigger needs_clarification with the option summaries; founder replies with option_id. On rejection of all (regeneration_count < 3): regenerate with founder hint.",
  "done_when": "selected_brand_direction.option_id resolves to one of brand_direction_options.options[].option_id.",
  "artifact_schema": {
    "brand_direction_options": {"type": "object", "required_fields": ["options", "min_options", "max_options"]},
    "selected_brand_direction": {"type": "object", "required_fields": ["option_id", "selected_at", "rejected_options"]}
  },
  "skip_when": "ambition_tier == 'prototype'"
}
```

**New step `5.5b1_template` (prototype-tier alternative):**
```json
{
  "id": "5.5b1_template",
  "phase": "phase_5",
  "name": "brand_direction_template_pick",
  "agent": "mechanical",
  "executor": "mechanical",
  "depends_on": ["5.5"],
  "may_need_clarification": true,
  "output_artifacts": ["selected_brand_direction"],
  "context": {
    "executor": "clarify",
    "kind": "variant_choice",
    "options_source": "data/brand_templates/index.json"
  },
  "skip_when": "ambition_tier != 'prototype'"
}
```

**Modified `5.6 brand_and_design_tokens`** — `i2p_v3.json:3573-3618`. `input_artifacts` += `["brand_direction_options", "selected_brand_direction"]`. `instruction` prepend: `"Look up selected_brand_direction.option_id in brand_direction_options.options[]. Use that option's mood_board for color extraction (vision tool on each image_url) and its tone_samples for tone_of_voice. Do NOT re-invent — materialize from the chosen option."` Remove the inline `if multiple valid directions, trigger needs_clarification` clause (delegated upstream).

#### Reviewer prompt criteria additions

**`5.10 design_review` instruction** — append:
```
Additional checks (added v3):
- Verify selected_brand_direction.option_id exists in brand_direction_options.options[].option_id (else status=fail).
- Verify design_tokens.colors derive from mood_board images of the chosen option (heuristic: at least one primary color matches a dominant color in img-1).
- Flag (status=needs_revision) when any mood_board image has license_signal='unknown' AND the brand asset will be used in marketing surfaces (downstream 07-humanish-layers consumes).
```

#### Failure walkthroughs

**[synthetic] Mission F1 — would P2 help?** Yes, marginally. Today `5.6` picks "modern minimal" by default for every Turkish reseller mission — wrong cultural fit. P2's `BD-002 Bazaar Energy` option exists *because* persona `Mehmet` has evidence_refs from P1. Without P1, the LLM would still pick generic. So **P2 is downstream of P1 in value, not just sequence.**

**[synthetic] Mission F3 — Solo founder spends 2 weeks on logo.** A founder using Kutay for a personal-finance side project iterates on 5 logo options for 14 days. P2's mood-board step takes 5 minutes; the chosen direction propagates to `5.6 design_tokens`; `5.7 component_specs` materializes consistent components. Founder's time saved: ~13 days. (Note: P2 doesn't replace logo design — it sets direction. Logo is still founder territory.)

#### Engine extension required
- `packages/vecihi/src/vecihi/fetchers.py`: add `return_bytes` kwarg to `scrape_url` (~20 LOC).
- `mr_roboto/__init__.py`: 1 new dispatch branch (`image_attribution_present`).
- No changes to expander or runtime.

#### Telemetry plan
- Yazbunu event: `z1_brand` with `{mission_id, options_generated, options_with_image, regeneration_count, founder_choice_id, scrape_tier_used}`.
- HTTP-tier vs browser-tier ratio: surfaced via existing vecihi tier-counter.

#### Migration concrete
- In-flight missions in phases ≤5: tag `missions.legacy_pre_taste_delegation INTEGER DEFAULT 0`; set 1 for existing missions; reviewer at 5.10 skips brand-direction audit when set.
- Set site: migration script.
- Read site: 5.10 reviewer prompt clause.

#### Locked alternatives
- **Chose A (scrape) over B (template library) for private_beta+.** Scenario where B wins: founder building in a saturated niche (todo apps) where template "modern SaaS clean" matches 100% of expected aesthetic. Decision: B is the prototype-tier fallback (`5.5b1_template` step), A is private_beta+ default.
- **Chose scrape over generated images.** Scenario where generated wins: founder explicitly requests "AI-generated" art direction. Decision: not in v3 scope; defer to P10c.

---

### P3. ADRs as first-class

#### Re-audit of v2
v2's audit is correct. ADR scope counted right (~13 ADRs across 4.1-4.10). v2's reviewer-prompt update for 4.16 is on the right track but doesn't show the prompt diff. **Errors v3 fixes:**
- v2 doesn't show the per-step `instruction` deltas (Diff #10).
- v2 doesn't address `4.14`'s existing `instruction` at `i2p_v3.json:3159`: `"Write Architecture Decision Records (ADRs) for every major decision: ... Each ADR: title, status, context, decision, consequences."` — this string explicitly teaches the LLM the 5-field shape. Reframing 4.14 to a register requires REWRITING this instruction.
- v2's `adr_alternatives_min` reads `ambition_tier` from "the mission row" but the mission row's `context` JSON column is the actual store. v3 specifies extraction.

#### Chosen shape (locked)
- Steps `4.1, 4.2, 4.4, 4.6, 4.8, 4.9, 4.10` each emit one or more ADR-shaped artifacts (per `decision_domain`).
- `4.14` becomes register-only: reads upstream ADRs, emits `adr_register` summary.
- New mechanical actions: `adr_alternatives_min` (per-step), `adr_completeness_check` (at 4.14).
- Reviewer at `4.16 architecture_review` gets prompt update.
- Single atomic ship; no in-flight migration except legacy-tag for in-progress phase-4 missions.

#### Inline JSON artifact examples

**ADR — POPULATED (mission F1, database choice):**
```json
{
  "_schema_version": "1",
  "adr_id": "ADR-2026-05-09-004",
  "title": "Use PostgreSQL with TimescaleDB extension for price history",
  "status": "accepted",
  "decision_domain": "database",
  "context": "Need to store hourly price snapshots for ~50k SKUs across 4 marketplaces. 1.2M rows/day, 36-month retention target = ~1.3B rows. Query patterns: (a) latest price per SKU, (b) 30-day price chart per SKU, (c) cross-SKU margin scan.",
  "options_considered": [
    {
      "option_id": "OPT-A",
      "name": "PostgreSQL plain (BTree on (sku_id, captured_at))",
      "pros": ["Familiar", "Free hosting via Supabase TR-region"],
      "cons": ["Index bloat at 1B+ rows", "30-day chart query slow without partitioning"],
      "evaluation_score": 5.5,
      "rejected_because": "Without time-series partitioning, monthly maintenance becomes manual; founder is solo.",
      "evidence_refs": ["agent_inference"]
    },
    {
      "option_id": "OPT-B",
      "name": "PostgreSQL + TimescaleDB",
      "pros": ["Auto-partitioning", "Continuous aggregates for chart query", "Same Postgres semantics"],
      "cons": ["Supabase doesn't support TimescaleDB on free tier — need self-host or paid"],
      "evaluation_score": 8.0,
      "rejected_because": null,
      "evidence_refs": ["EV-001.C1"]
    },
    {
      "option_id": "OPT-C",
      "name": "ClickHouse",
      "pros": ["Best-in-class for time-series scans"],
      "cons": ["New stack for solo founder", "Hosted ClickHouse Cloud = $50/mo minimum, exceeds budget"],
      "evaluation_score": 6.0,
      "rejected_because": "Cost ceiling 100 USD/mo (z0 budget); ClickHouse Cloud + app hosting overshoots.",
      "evidence_refs": ["agent_inference"]
    }
  ],
  "decision": "OPT-B PostgreSQL + TimescaleDB, self-hosted on TR-region VPS (Hetzner Helsinki has TR egress acceptable for KVKK). $20/mo VPS fits budget.",
  "chosen_option_id": "OPT-B",
  "consequences": {
    "positive": ["Continuous aggregates make 30-day chart sub-100ms", "Same SQL surface — no new tooling"],
    "negative": ["Self-host = solo-founder ops burden", "Backup strategy required (daily pg_dump → S3-compat)"],
    "neutral": ["TimescaleDB extension upgrades require Postgres restart"]
  },
  "falsification_signal": {
    "hypothesis": "Hourly snapshots × 50k SKUs is the right granularity",
    "signal_if_wrong": "If founder activates >100 SKUs in week 1, hourly = 168k rows/SKU/year × 100 = 16.8M rows just for actively-tracked, not budgeted",
    "kill_threshold": "Active SKU count > 200 by week 4 → reopen to consider per-tier sampling"
  },
  "revisit_trigger": "Active SKU > 200 OR 30-day chart query p95 > 800ms",
  "reversal_cost": "medium",
  "supersedes_adr_id": null,
  "decided_at": "2026-05-09T12:00:00Z",
  "decided_by": "agent_with_clarification"
}
```

**ADR — EMPTY/SKELETAL (prototype tier, 2 options only):**
```json
{
  "_schema_version": "1",
  "adr_id": "ADR-2026-05-09-001",
  "title": "Use NextJS",
  "status": "accepted",
  "decision_domain": "frontend_framework",
  "context": "Solo founder, 2-week prototype window.",
  "options_considered": [
    {"option_id": "OPT-A", "name": "NextJS", "pros": ["Familiar"], "cons": [],
     "evaluation_score": 7, "rejected_because": null, "evidence_refs": ["agent_inference"]},
    {"option_id": "OPT-B", "name": "Vite + React", "pros": ["Faster dev"],
     "cons": ["No SSR built-in"], "evaluation_score": 5,
     "rejected_because": "SEO matters for marketplace landing pages.",
     "evidence_refs": ["agent_inference"]}
  ],
  "decision": "OPT-A",
  "chosen_option_id": "OPT-A",
  "consequences": {"positive": ["Mature ecosystem"], "negative": [], "neutral": []},
  "falsification_signal": {
    "hypothesis": "Need SSR for SEO",
    "signal_if_wrong": "If founder pivots to API-only product, SSR is dead weight",
    "kill_threshold": "If marketing decides on API-first GTM"
  },
  "revisit_trigger": "Pivot to API-first",
  "reversal_cost": "high",
  "supersedes_adr_id": null,
  "decided_at": "2026-05-09T12:01:00Z",
  "decided_by": "agent"
}
```

**adr_register — emitted by 4.14:**
```json
{
  "_schema_version": "1",
  "adrs": [
    {"adr_id": "ADR-2026-05-09-001", "title": "Use NextJS", "status": "accepted",
     "decision_domain": "frontend_framework", "summary": "NextJS over Vite for SSR.",
     "links_to": [], "supersedes": []},
    {"adr_id": "ADR-2026-05-09-004", "title": "PostgreSQL+TimescaleDB",
     "status": "accepted", "decision_domain": "database",
     "summary": "TimescaleDB for time-series.",
     "links_to": ["ADR-2026-05-09-002"], "supersedes": []}
  ],
  "completeness_check": {
    "required_domains": ["architecture_pattern", "frontend_framework",
                         "backend_framework", "database", "infrastructure", "auth"],
    "missing": [],
    "vendor_adrs_count": 5
  }
}
```

#### Inline Python pseudo-code

**`adr_alternatives_min` mechanical action:**
```python
# packages/mr_roboto/src/mr_roboto/adr_alternatives_min.py
import sqlite3, json, os
async def adr_alternatives_min(task: dict) -> Action:
    payload = task.get("payload") or {}
    mission_id = task["mission_id"]
    artifact_name = payload.get("artifact_name")  # e.g. "architecture_pattern_decision"
    # Read ambition_tier from missions.context JSON
    db = os.environ.get("DB_PATH", "data/kutai.db")
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT context FROM missions WHERE id=?",
                       (mission_id,)).fetchone()
    ctx = json.loads(row[0]) if row and row[0] else {}
    tier = ctx.get("ambition_tier") or "private_beta"
    min_alts = 2 if tier == "prototype" else 3
    art = await load_artifact(mission_id, artifact_name)
    # Artifact may be a single ADR or {adrs: [...]}
    adrs = art if isinstance(art, list) else \
           ([art] if "adr_id" in art else art.get("adrs") or [])
    bad = []
    for adr in adrs:
        if adr.get("is_legacy"):
            continue
        alts = adr.get("options_considered") or []
        if len(alts) < min_alts:
            bad.append({"adr_id": adr.get("adr_id"),
                        "found": len(alts), "required": min_alts})
    if bad:
        return Action(status="failed",
                      error=f"adr_alternatives_min: {len(bad)} ADRs short. {bad}",
                      result={"bad": bad})
    return Action(status="completed", result={"checked": len(adrs)})
```

**`adr_completeness_check` mechanical action:**
```python
# packages/mr_roboto/src/mr_roboto/adr_completeness_check.py
REQUIRED_DOMAINS_PRIVATE = {
    "architecture_pattern", "frontend_framework", "backend_framework",
    "database", "infrastructure", "auth"
}
REQUIRED_DOMAINS_PROTO = {
    "architecture_pattern", "frontend_framework", "database"
}
async def adr_completeness_check(task: dict) -> Action:
    payload = task.get("payload") or {}
    mission_id = task["mission_id"]
    register = await load_artifact(mission_id, "adr_register")
    tier = await get_ambition_tier(mission_id)
    required = REQUIRED_DOMAINS_PROTO if tier == "prototype" else REQUIRED_DOMAINS_PRIVATE
    present = {a["decision_domain"] for a in register.get("adrs") or []}
    missing = required - present
    if missing:
        return Action(status="failed",
                      error=f"adr_completeness_check: missing domains {sorted(missing)}",
                      result={"missing": sorted(missing)})
    # Validate links_to is acyclic
    graph = {a["adr_id"]: a.get("links_to") or [] for a in register.get("adrs") or []}
    if has_cycle(graph):
        return Action(status="failed",
                      error="adr_completeness_check: cycle in links_to graph")
    return Action(status="completed",
                  result={"adrs": len(register.get("adrs") or [])})
```

#### i2p_v3.json step JSON drafts

**Modified `4.1 architecture_pattern_selection`** (`i2p_v3.json:2544-2581`) — change `output_artifacts` from `["architecture_pattern_decision"]` to a single ADR. Update `artifact_schema`:
```json
"artifact_schema": {
  "architecture_pattern_decision": {
    "type": "object",
    "required_fields": ["adr_id", "title", "status", "decision_domain",
                        "context", "options_considered", "decision",
                        "chosen_option_id", "consequences", "falsification_signal",
                        "revisit_trigger", "reversal_cost", "decided_at", "decided_by"]
  }
},
"post_hooks": ["adr_alternatives_min"],
"context": {
  "adr_alternatives_min": {"artifact_name": "architecture_pattern_decision"}
}
```
Update `instruction`: append `"Output as ADR with options_considered (min 2 for prototype, 3 for private_beta+), each option has pros/cons/evaluation_score/rejected_because. Include falsification_signal and reversal_cost. decision_domain MUST be 'architecture_pattern'."`

**Reframed `4.14 architecture_decisions` → `adr_consolidation`** — change `instruction` to:
```
Consolidate the upstream ADRs (architecture_pattern_decision, tech_stack_decision, system_architecture, database_schema, auth_design, third_party_selections, infrastructure_designs, communication_designs) into adr_register. For each ADR, emit a summary entry with adr_id, title, status, decision_domain, summary (≤30 words), links_to (other adr_ids this depends on), supersedes (deprecated adr_ids). Compute completeness_check: required_domains (load from context.required_domains based on ambition_tier), missing (required - present), vendor_adrs_count (count of decision_domain=third_party_service).
```
Update `output_artifacts` to `["adr_register"]`. Add `depends_on` += `["4.4", "4.9", "4.10"]`. Add `post_hooks: ["adr_completeness_check"]`.

**Modified `4.16 architecture_review`** — append to instruction (currently at `i2p_v3.json:3306+`):
```
Additional checks (added v3):
- For each ADR in adr_register: verify falsification_signal.hypothesis is non-empty AND signal_if_wrong is observable (rejects vague phrases like "if it's wrong").
- For each ADR: verify chosen_option_id resolves to one of options_considered[].option_id.
- For each rejected option (option_id != chosen_option_id): verify rejected_because is non-empty AND non-trivial (>15 chars).
- Verify links_to graph is acyclic (no ADR cycles).
- Reject (status=fail) when any of the above fails on a non-legacy ADR.
```

#### Reviewer prompt criteria additions
Inlined above for `4.16`. Same pattern at `5.10` if downstream ADRs land in phase 5 — none planned for v3.

#### Failure walkthroughs

**[synthetic] Mission F1 — would P3 catch the database choice?** Yes:
- Without P3: 4.4 emits `database_schema = {tables: [...], normalization: "3NF"}`. PostgreSQL choice baked in implicitly. No alternatives logged.
- With P3: 4.4 emits ADR-shaped database_schema with options_considered including TimescaleDB and ClickHouse. Reviewer at 4.16 verifies the cost-ceiling reasoning (rejected_because cites z0 budget). Founder reviews and either accepts or replies "actually try Supabase paid tier" → mission revises with new evidence.
- Actual win: in 6 months when phase 5 (build) hits the "TimescaleDB needs upgrade" moment, the ADR's `revisit_trigger` activates a notification.

**[synthetic] Mission F4 — solo founder picks "always Supabase".** Founder explicitly requests "always use Supabase, I have credits." P3 still emits ADR with `options_considered: [{Supabase, ...}, {alternative_X, rejected_because: "founder constraint"}]`. The honest "founder constraint" reason is auditable later if Supabase pricing changes.

#### Engine extension required
- `mr_roboto/__init__.py`: 2 new dispatch branches (`adr_alternatives_min`, `adr_completeness_check`).
- `src/workflows/engine/artifacts.py`: must support reading `architecture_pattern_decision` etc. as either single-ADR objects OR `{adrs: [...]}`. `load_artifact` already returns whatever was emitted; the mechanical hook code handles both shapes (see pseudo-code above).
- No expander/runtime changes.

#### Telemetry plan
- Yazbunu event `z1_adr` per ADR: `{mission_id, adr_id, decision_domain, options_count, chosen_option_id, has_falsification_signal: bool, reversal_cost}`.
- DB enrichment: optionally extend `model_pick_log.snapshot_summary` to include `adr_count` for the mission row (cross-reference to selection-quality study).
- Locked: yazbunu only; no DB column changes.

#### Migration concrete
- New table column: `missions.legacy_pre_adr INTEGER DEFAULT 0`.
- Migration script: `scripts/migrations/2026-05-XX-adr-legacy-tag.py` — sets `legacy_pre_adr=1` for missions where `tasks` show phase 4 already past `4.14` completion.
- Read site: `4.16` reviewer instruction appends `"If mission has legacy_pre_adr=1 (passed via context.mission_flags), skip ADR-shape audit; accept old shape (5 fields)."`
- Read site: `adr_alternatives_min` checks `is_legacy: true` per-ADR (set by a one-shot reshape on existing artifacts) and skips.
- Reshape script: optional, only for missions still in phase 4 at landing. Wraps existing `architecture_pattern_decision` into ADR shape with `options_considered: []`, `is_legacy: true`.

#### Locked alternatives
- **Chose A (per-step ADR + 4.14 register) over B (4.14 expanded only).** Scenario where B wins: token budget so tight (sub-prototype tier) that 13 ADRs blow up. Decision: at sub-prototype, skip ADRs entirely (no `4.1` ADR; legacy 5-field shape stays). v3 doesn't introduce a sub-prototype tier; if z0 ever does, this is the gating field.
- **Chose A over C (RFC-threaded).** Scenario where C wins: enterprise mission with multi-stakeholder review. Defer to a P10d follow-up.

---

### P4. Failure-mode column (falsifiability)

#### Re-audit of v2
v2's audit is correct. v2's hybrid-shape recommendation (per-item field for high-stakes + register summary) is sound. **Errors v3 fixes:**
- v2 doesn't show the per-step `instruction` deltas.
- v2's `falsification_completeness` reads `ambition_tier from mission row` — same lookup pattern as P3; v3 specifies the exact JSON path `missions.context["ambition_tier"]`.
- v2 mentions cross-zone consumer contracts (08-operations reads `revisit_trigger`, 09-growth reads `kill_threshold`) but doesn't say how those zones are notified at landing. v3 adds: emit a `## Inbound from Z1` section in 08 + 09 docs as part of P4 ship.

#### Chosen shape (locked)
Hybrid (v2 alt C): per-item `failure_mode` field on commitment-shaped artifacts (FRs, NFRs, MVP features, success metrics, ADRs already covered by P3) + a thin `falsification_register` artifact emitted by a new step `3.11a` (between `3.11` review and `4.1` arch). Register is a join over upstream items; cheap to compute, valuable for downstream zones.

#### Inline JSON artifact examples

**failure_mode field (on a functional_requirement):**
```json
{
  "req_id": "FR-014",
  "title": "Daily price scan triggers cron",
  "description": "Cron job at 02:00 TR-time pulls latest prices from Trendyol/Hepsiburada/N11/GG.",
  "source_story_ids": ["US-007"],
  "priority": "must",
  "category": "functional",
  "evidence_refs": ["EV-001.C1"],
  "is_inference": false,
  "failure_mode": {
    "hypothesis": "Daily granularity is sufficient to catch arbitrage opportunities",
    "signal_if_wrong": "Price changes intraday > daily; founder reports missing windows",
    "kill_threshold": "If >5 missed-window complaints in 30 days, switch to hourly",
    "we_dont_know_yet": false
  }
}
```

**failure_mode — PARTIAL (`we_dont_know_yet: true`):**
```json
{
  "req_id": "FR-022",
  "title": "Multilingual support TR/EN",
  "...other fields...": "...",
  "failure_mode": {
    "hypothesis": "EN matters for diaspora users",
    "signal_if_wrong": "",
    "kill_threshold": "",
    "we_dont_know_yet": true
  }
}
```

**falsification_register — emitted by new 3.11a:**
```json
{
  "_schema_version": "1",
  "items": [
    {
      "ref": "FR-014",
      "ref_type": "functional_requirement",
      "kill_threshold": "If >5 missed-window complaints in 30 days, switch to hourly",
      "owner_zone": "08-operations",
      "monitoring_recipe_id": null
    },
    {
      "ref": "AARRR-activation",
      "ref_type": "success_metric",
      "kill_threshold": "MAU < 100 by week 8",
      "owner_zone": "09-growth",
      "monitoring_recipe_id": null
    },
    {
      "ref": "NFR-perf-001",
      "ref_type": "nfr",
      "revisit_trigger": "p95 > 800ms over 3 days",
      "owner_zone": "08-operations",
      "monitoring_recipe_id": null
    }
  ],
  "we_dont_know_yet_pct": 0.18
}
```

#### Inline Python pseudo-code

**`falsification_completeness` mechanical action:**
```python
# packages/mr_roboto/src/mr_roboto/falsification_completeness.py
async def falsification_completeness(task: dict) -> Action:
    payload = task.get("payload") or {}
    mission_id = task["mission_id"]
    artifact_name = payload["artifact_name"]
    item_path = payload.get("item_path", "items")  # default "items" else top-level array
    art = await load_artifact(mission_id, artifact_name)
    items = art.get(item_path) if isinstance(art, dict) else art
    items = items or []
    if not items:
        return Action(status="completed", result={"reason": "no items"})
    tier = await get_ambition_tier(mission_id)
    bad = []
    unknown = 0
    for it in items:
        fm = it.get("failure_mode") or {}
        if not fm.get("hypothesis"):
            bad.append({"id": it.get("req_id") or it.get("feature_id"),
                        "missing": "hypothesis"})
            continue
        if fm.get("we_dont_know_yet"):
            unknown += 1
            continue
        if not fm.get("signal_if_wrong"):
            bad.append({"id": it.get("req_id"), "missing": "signal_if_wrong"})
        if not fm.get("kill_threshold"):
            bad.append({"id": it.get("req_id"), "missing": "kill_threshold"})
    pct_unknown = unknown / len(items) if items else 0
    severity = "warning"
    if tier in ("private_beta", "public_launch", "revenue_product") and pct_unknown > 0.5:
        severity = "high"
        bad.append({"reason": f"{int(pct_unknown*100)}% items have we_dont_know_yet"})
    if bad and severity == "high":
        return Action(status="failed",
                      error=f"falsification_completeness: {len(bad)} issues. {bad[:5]}",
                      result={"bad": bad, "pct_unknown": pct_unknown})
    return Action(status="completed",
                  result={"checked": len(items), "pct_unknown": pct_unknown,
                          "warnings": bad if bad else None})
```

#### i2p_v3.json step JSON drafts

**Modified `2.5 feature_brainstorm`** (`i2p_v3.json:1683-1723`) — add `failure_mode` to `item_fields`. Append to instruction: `"For each MVP-priority feature, populate failure_mode with hypothesis (what must hold for this to deliver value), signal_if_wrong (observable), kill_threshold (quantitative if possible). If unknown, set we_dont_know_yet=true (capped at 50% of items at private_beta+)."` Add `post_hooks: ["falsification_completeness"]`, `context: {"falsification_completeness": {"artifact_name": "features"}}`.

**Modified `2.9 success_metrics_definition`** — add `kill_threshold` to per-AARRR-metric schema. Instruction append: `"Each AARRR metric MUST have target_value AND kill_threshold (quantitative). Example: target=D7 retention 30%; kill_threshold=D7 retention <15% by week 4."`

**Modified `3.1 functional_requirements_extraction`** — add `failure_mode` and `evidence_refs` (P1 overlap) to per-FR. Same post-hook.

**Modified `3.2 nfr_performance_and_scalability`** — add `revisit_trigger` to per-NFR.

**Modified `0.5 human_clarification_request`** — append to instruction: `"Prioritize assumptions where risk_if_wrong=high in your top-5 question budget. Drop questions about low-risk assumptions if needed."`

**New step `3.11a falsification_register_emit`:**
```json
{
  "id": "3.11a",
  "phase": "phase_3",
  "name": "falsification_register_emit",
  "agent": "writer",
  "difficulty": "easy",
  "tools_hint": [],
  "depends_on": ["3.11"],
  "may_need_clarification": false,
  "input_artifacts": ["functional_requirements", "nfr_performance",
                      "nfr_availability", "success_metrics", "features"],
  "output_artifacts": ["falsification_register"],
  "produces": ["workspace/mission_{mission_id}/.falsification/register.json"],
  "instruction": "Walk all upstream commitment-shaped items (FRs with kill_threshold, NFRs with revisit_trigger, AARRR metrics with kill_threshold, MVP features with kill_criteria). Emit one row per item: ref (id), ref_type, kill_threshold or revisit_trigger, owner_zone (08-operations | 09-growth | 02-build), monitoring_recipe_id (null at Z1 — populated when 08 wires).",
  "done_when": "falsification_register exists with items array.",
  "skip_when": "ambition_tier == 'prototype'"
}
```

#### Reviewer prompt criteria additions

**`3.11 requirements_review`** — append to instruction (already modified by P1 above; both clauses stack):
```
Additional checks (added v3, P4):
- Reject (status=fail) if any FR with priority=must has failure_mode.hypothesis empty.
- Issue (status=needs_revision) when >50% of FRs have we_dont_know_yet=true at private_beta+ tier.
- For each must-priority FR, signal_if_wrong MUST be observable (reject vague phrases like "users complain").
```

#### Failure walkthroughs

**[synthetic] Mission F1 — would P4 catch over-confident FRs?** Yes:
- Without P4: FR-014 "Daily price scan triggers cron" ships. 6 weeks later founder discovers intraday volatility — 3 days of opportunity loss before realizing.
- With P4: FR-014's `failure_mode.signal_if_wrong = "founder reports missing windows"`; `kill_threshold = ">5 complaints/30 days → switch to hourly"`. The `falsification_register` row makes this monitorable in zone 08; 09-growth can A/B daily-vs-hourly explicitly.

**[synthetic] Mission F5 — `we_dont_know_yet` foot-gun.** Founder rushes; agent sets `we_dont_know_yet: true` on 80% of FRs. P4's `falsification_completeness` blocks at private_beta+ when >50% unknown. Reviewer surfaces. Founder must decide: reduce scope OR fill in failure modes.

#### Engine extension required
- `mr_roboto/__init__.py`: 1 new dispatch branch (`falsification_completeness`).
- No artifact schema changes outside i2p_v3.json.

#### Telemetry plan
- Yazbunu `z1_falsification`: `{mission_id, step_id, items_total, items_unknown, pct_unknown, blocked: bool}`.
- Cross-zone consumer audit: a follow-up GitHub issue tracks "08-operations.md and 09-growth.md must add `## Inbound from Z1` sections referencing falsification_register schema."

#### Migration concrete
- New column: `missions.legacy_pre_falsification INTEGER DEFAULT 0`. Migration sets =1 for existing missions in phase ≥3.
- Read site: 3.11 reviewer + `falsification_completeness` skip enforcement when set.

#### Locked alternatives
- **Chose hybrid (alt C from v2) over A (per-item only).** Scenario where pure-A wins: register adds maintenance overhead. Decision: register is one new step, low cost; cross-zone value is the win.
- **`we_dont_know_yet` escape hatch retained.** Scenario where it should be removed: enterprise tier where every commitment must be falsifiable. Decision: enterprise is not in scope; cap (50%) is the safety valve.

---

### P5. Web-grounded prior art (closes G5)

#### Re-audit of v2
v2's audit and concrete schema are sound. The vecihi tier escalation costs are right. **Errors v3 fixes:**
- Path prefix wrong (Diff #1): `workspace/mission_<id>/.research/prior_art_report.json`.
- v2's `prior_art.py` lands in `packages/vecihi/src/vecihi/prior_art.py`. Acceptable, but v2 acknowledges "search aggregator" is a different abstraction. v3 keeps it under vecihi for v3 (sibling to `fetchers.py`) but flags as candidate for extraction to `packages/prior_art/` later.
- v2's "always falls back to B (cache)" is hand-wavy. v3 specifies the trigger: any source returns 429 OR ≥3 sources return 0 results in 30s → fall to cache. Cache hit is annotated in `search_summary.cache_hit_for_sources`.
- v2's `prior_art_url_resolves` post-hook is misnamed: it modifies the artifact mid-flight. v3 keeps the validation but renames to `prior_art_min_coverage` (which v2 also has) — the URL-resolution sweep happens *inside* `find_prior_art`, not as a separate hook.

#### Chosen shape (locked)
- New module `packages/vecihi/src/vecihi/prior_art.py` exposing `find_prior_art(idea_summary, domain_keywords, k, ambition_tier) -> dict`.
- New tool registered in `src/tools/` registry making `find_prior_art` LLM-callable.
- New step `1.0 prior_art_search` (parallel with 1.1-1.4).
- New mechanical action `prior_art_min_coverage` (post-hook).
- Modified `1.14 go_no_go_assessment` and `2.1 product_vision_and_positioning` consume prior art.
- Cache table `prior_art_cache(domain_keywords_hash TEXT, results_json TEXT, fetched_at TEXT, ttl_hours INTEGER DEFAULT 168)`.

#### Inline JSON artifact examples

**prior_art_report — POPULATED (mission F1):**
```json
{
  "_schema_version": "1",
  "search_summary": {
    "queries_run": ["turkish reseller arbitrage tool",
                    "trendyol price tracker reseller",
                    "n11 price monitor saas"],
    "sources_used": ["hn_algolia", "wikipedia", "wayback", "product_hunt"],
    "rate_limit_hits": [],
    "total_results_inspected": 47,
    "cache_hit_for_sources": []
  },
  "attempted_solutions": [
    {
      "name": "SellerSpy.tr",
      "founded_year": 2019,
      "status": "dead",
      "url": "https://sellerspy.tr",
      "wayback_first_capture": "2019-08-12",
      "wayback_last_capture": "2023-09-01",
      "traction_signal": "200 paying users at peak per founder LinkedIn",
      "failure_mode": "Founder ran out of runway; couldn't get past Trendyol API cease-and-desist",
      "sources": ["https://news.ycombinator.com/item?id=...",
                  "https://web.archive.org/.../sellerspy.tr"],
      "thesis_summary": "Same arbitrage thesis: scan TR marketplaces, surface margin opportunities, charge B2B.",
      "evidence_refs": ["agent_inference"]
    },
    {
      "name": "ArbitrajPro",
      "founded_year": 2021,
      "status": "dormant",
      "url": "https://arbitrajpro.io",
      "wayback_first_capture": "2021-03-04",
      "wayback_last_capture": "2024-11-15",
      "traction_signal": "Unknown — no public metrics",
      "failure_mode": null,
      "sources": ["https://web.archive.org/.../arbitrajpro.io"],
      "thesis_summary": "International arbitrage focus, TR was secondary.",
      "evidence_refs": ["agent_inference"]
    }
  ],
  "adjacent_failures": [
    {
      "name": "Honey for sellers (US)",
      "founded_year": 2018,
      "status": "acquired",
      "url": "https://...",
      "wayback_first_capture": "2018-01-01",
      "wayback_last_capture": "2023-06-12",
      "traction_signal": "Acquired by PayPal $4B (consumer side)",
      "failure_mode": null,
      "sources": ["https://en.wikipedia.org/wiki/Honey_(browser_extension)"],
      "thesis_summary": "Consumer not seller; price tracking surface similar.",
      "evidence_refs": ["agent_inference"]
    }
  ],
  "key_lessons": [
    {
      "lesson_id": "L-001",
      "lesson": "Marketplace ToS is the kill vector; SellerSpy died because Trendyol blocked their scrapers, not because the product was bad.",
      "evidence_refs": ["https://news.ycombinator.com/item?id=..."],
      "applies_to_us": "Use official APIs where available (Trendyol seller API exists for verified resellers); user-tier scraping fallback only."
    },
    {
      "lesson_id": "L-002",
      "lesson": "Solo TR-market plays cap at 200 users; B2B in TR has price-elastic small businesses.",
      "evidence_refs": ["agent_inference"],
      "applies_to_us": "Plan for international expansion in year 2; TR alone won't sustain."
    }
  ],
  "graveyard_count": 2,
  "verdict": "graveyard_well_populated"
}
```

**prior_art_report — `blue_ocean_validated` (suspicious):**
```json
{
  "_schema_version": "1",
  "search_summary": {
    "queries_run": ["super-niche idea here"],
    "sources_used": ["hn_algolia", "wikipedia"],
    "rate_limit_hits": [],
    "total_results_inspected": 3,
    "cache_hit_for_sources": []
  },
  "attempted_solutions": [],
  "adjacent_failures": [],
  "key_lessons": [],
  "graveyard_count": 0,
  "verdict": "blue_ocean_validated"
}
```
This trips `prior_art_min_coverage` because `len(queries_run) < 3` AND `total_results_inspected < 20`. The mechanical hook rejects with `severity: warning`, forcing the agent to retry with broader queries before claiming blue ocean.

#### Inline Python pseudo-code

**`vecihi.prior_art.find_prior_art`:**
```python
# packages/vecihi/src/vecihi/prior_art.py
import asyncio, hashlib, json, sqlite3, time
from vecihi import scrape_url, ScrapeTier
import aiohttp

HN_ALGOLIA = "https://hn.algolia.com/api/v1/search?query={q}&tags=story&hitsPerPage=20"
WIKI = "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch={q}&srlimit=10"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx?url={u}&output=json&limit=5"

async def find_prior_art(idea_summary: str, domain_keywords: list[str],
                         k: int = 10, ambition_tier: str = "private_beta") -> dict:
    queries = _build_queries(idea_summary, domain_keywords)
    cache_key = hashlib.sha256(("|".join(sorted(domain_keywords))).encode()).hexdigest()
    cached = _read_cache(cache_key)
    if cached and not _stale(cached, hours=168):
        return _annotate_cache_hit(cached, sources=cached["search_summary"]["sources_used"])
    candidates = []
    sources_used = []
    rate_limit_hits = []
    total_inspected = 0
    async with aiohttp.ClientSession() as sess:
        # Run HN + Wiki in parallel
        hn_task = asyncio.create_task(_query_hn(sess, queries[:3]))
        wiki_task = asyncio.create_task(_query_wiki(sess, queries[:3]))
        try:
            hn_results = await asyncio.wait_for(hn_task, timeout=10)
            sources_used.append("hn_algolia")
            candidates.extend(_normalize_hn(hn_results))
            total_inspected += len(hn_results)
        except (asyncio.TimeoutError, aiohttp.ClientResponseError) as e:
            rate_limit_hits.append({"source": "hn_algolia", "err": str(e)[:60]})
        try:
            wiki_results = await asyncio.wait_for(wiki_task, timeout=10)
            sources_used.append("wikipedia")
            candidates.extend(_normalize_wiki(wiki_results))
            total_inspected += len(wiki_results)
        except Exception as e:
            rate_limit_hits.append({"source": "wikipedia", "err": str(e)[:60]})
    # Wayback validation per top-k candidate (private_beta+ only)
    if ambition_tier in ("private_beta", "public_launch", "revenue_product"):
        wayback_validated = await _wayback_sweep(candidates[:k])
        candidates = wayback_validated
        sources_used.append("wayback")
    # Product Hunt via existing scrape_url (Vecihi tier escalation)
    if ambition_tier in ("public_launch", "revenue_product"):
        ph = await _scrape_product_hunt(queries[0])
        if ph:
            sources_used.append("product_hunt")
            candidates.extend(ph)
            total_inspected += len(ph)
    # Dedup, classify, build report
    candidates = _dedup_by_name(candidates)
    attempted, adjacent = _split_by_relevance(candidates, idea_summary)
    lessons = _extract_lessons(attempted)
    graveyard_count = sum(1 for s in attempted if s["status"] == "dead")
    verdict = _classify_verdict(attempted, total_inspected, queries)
    report = {
        "_schema_version": "1",
        "search_summary": {
            "queries_run": queries[:5],
            "sources_used": sources_used,
            "rate_limit_hits": rate_limit_hits,
            "total_results_inspected": total_inspected,
            "cache_hit_for_sources": []
        },
        "attempted_solutions": attempted[:k],
        "adjacent_failures": adjacent[:5],
        "key_lessons": lessons,
        "graveyard_count": graveyard_count,
        "verdict": verdict
    }
    _write_cache(cache_key, report)
    return report
```

**`prior_art_min_coverage` mechanical action:**
```python
# packages/mr_roboto/src/mr_roboto/prior_art_min_coverage.py
async def prior_art_min_coverage(task: dict) -> Action:
    mission_id = task["mission_id"]
    report = await load_artifact(mission_id, "prior_art_report")
    summary = report.get("search_summary") or {}
    queries = summary.get("queries_run") or []
    inspected = summary.get("total_results_inspected", 0)
    attempted = report.get("attempted_solutions") or []
    verdict = report.get("verdict")
    if verdict == "blue_ocean_validated":
        if len(queries) < 3 or inspected < 20:
            return Action(
                status="failed",
                error="prior_art_min_coverage: blue_ocean claim requires "
                      f">=3 queries (got {len(queries)}) and >=20 inspected (got {inspected})"
            )
    elif len(attempted) < 3:
        return Action(
            status="failed",
            error=f"prior_art_min_coverage: only {len(attempted)} solutions found; "
                  "broaden queries or set verdict=no_data"
        )
    # Hallucination check: every dead solution needs Wayback OR HN reference
    suspicious = []
    for sol in attempted:
        if sol["status"] in ("dead", "dormant"):
            sources = sol.get("sources") or []
            wb = sol.get("wayback_first_capture")
            if not wb and not any("ycombinator" in s for s in sources):
                suspicious.append(sol["name"])
    if suspicious:
        return Action(
            status="failed",
            error=f"prior_art_min_coverage: unverifiable dead solutions: {suspicious}",
            result={"suspicious": suspicious, "severity": "warning"}
        )
    return Action(status="completed", result={"attempted": len(attempted), "verdict": verdict})
```

#### i2p_v3.json step JSON drafts

**New step `1.0 prior_art_search`:**
```json
{
  "id": "1.0",
  "phase": "phase_1",
  "name": "prior_art_search",
  "agent": "researcher",
  "difficulty": "medium",
  "tools_hint": ["find_prior_art", "web_search", "scrape_url"],
  "depends_on": ["0.6"],
  "may_need_clarification": false,
  "input_artifacts": ["idea_brief_final"],
  "output_artifacts": ["prior_art_report"],
  "produces": ["workspace/mission_{mission_id}/.research/prior_art_report.json"],
  "post_hooks": ["prior_art_min_coverage"],
  "instruction": "Call find_prior_art(idea_summary=idea_brief_final.summary, domain_keywords=idea_brief_final.domain_tags, k=10, ambition_tier=context.ambition_tier). Inspect the returned report. If verdict='blue_ocean_validated' but you suspect the agent missed adjacent solutions, broaden domain_keywords and re-call once. Save the final report to produces path.",
  "done_when": "prior_art_report exists and prior_art_min_coverage passes."
}
```

**Modified `1.14 go_no_go_assessment`** (`i2p_v3.json:1475-1509`) — `input_artifacts` += `["prior_art_report"]`; instruction append:
```
Factor the prior_art_report into competitive_feasibility:
- competitive_feasibility -= 0.5 * min(prior_art_report.graveyard_count, 5), capped at -2.5.
- Cite at least one entry from prior_art_report.key_lessons in your reasoning.
- If verdict='blue_ocean_validated' AND prior_art_report.search_summary.total_results_inspected >= 20, add +0.5 to competitive_feasibility (rare bonus for genuine novelty).
```

**Modified `2.1 product_vision_and_positioning`** (`i2p_v3.json:1511-1547`) — `input_artifacts` += `["prior_art_report"]`; instruction append: `"Append section 'Avoiding the graveyard': for each prior_art_report.key_lessons[i], state how this product's design avoids that failure mode. Be concrete (cite the FR/ADR that addresses each lesson)."`

**Modified `1.11 regulatory_research`** — `input_artifacts` += `["prior_art_report"]`; instruction append: `"Cross-check graveyard entries: any regulatory failure modes (e.g., compliance shutdowns) cited in key_lessons MUST be addressed in this research output."`

#### Reviewer prompt criteria additions

**`1.13 research_quality_review`** — append:
```
Additional checks (added v3, P5):
- Verify prior_art_report exists.
- Reject (verdict=fail) if prior_art_report.attempted_solutions has any entry with status='dead' and no Wayback OR HN source.
- Reject if 1.14 go_no_go did not cite at least one key_lesson.
- Reject if 2.1 product_vision lacks 'Avoiding the graveyard' section when prior_art_report.graveyard_count > 0.
```

#### Failure walkthroughs

**[synthetic] Mission F1 — would P5 have caught the graveyard?** Yes. `find_prior_art` surfaces SellerSpy.tr + ArbitrajPro. `1.14` reduces competitive_feasibility from default 7 to 6 (graveyard_count=2). `2.1` adds "Avoiding the graveyard: SellerSpy died from Trendyol scraping bans → use official seller API only." Founder reads, decides whether the official-API path is workable; mission may pivot to "API-tier resellers only."

**[synthetic] Mission F6 — hallucinated competitor.** Agent returns `attempted_solutions: [{"name": "PriceGuru.io", "status": "dead", "url": "...", "sources": []}]` — no Wayback. `prior_art_min_coverage` rejects. Agent retries with Wayback validation; PriceGuru.io is removed.

#### Engine extension required
- New module: `packages/vecihi/src/vecihi/prior_art.py` (~200 LOC).
- New tool registration: register `find_prior_art` in `src/tools/__init__.py` LLM tool registry.
- New DB table: `prior_art_cache(key TEXT PRIMARY KEY, report_json TEXT, fetched_at TEXT)` — migration script.
- `mr_roboto/__init__.py`: 1 new dispatch branch (`prior_art_min_coverage`).
- Vecihi `scrape_url`: `return_bytes` (already added in P2; reused here for image fetching is N/A — only text needed for prior art).

#### Telemetry plan
- Yazbunu `z1_prior_art`: `{mission_id, queries_run, total_inspected, graveyard_count, verdict, cache_hit, sources_used, rate_limit_hits_count}`.
- DB metric: weekly aggregate of `cache_hit_pct` for monitoring API rate-limit pressure.

#### Migration concrete
- New table: `prior_art_cache` (created by migration).
- New column: `missions.legacy_pre_prior_art INTEGER DEFAULT 0`.
- Read site: 1.13/1.14/2.1 reviewers + downstream skip when set.

#### Locked alternatives
- **Chose A (live multi-source) with B (cache) as automatic fallback.** Scenario where pure-cache wins: rate-limit storm. Decision: cache is automatic fallback at 429 or empty results, not user-selected.
- **Chose vecihi sub-module over new package.** Scenario where new package wins: prior_art grows to its own surface (search aggregator across many domains). Decision: defer; ship as `vecihi.prior_art` for v3.

---

### P6. Compliance fingerprint (closes G6)

#### Re-audit of v2
v2's audit + cross-jurisdiction conflict surface + bilingual templates are sound. **Errors v3 fixes:**
- Path prefix wrong (Diff #1).
- v2's `compliance_fingerprint_wizard` is a new mechanical action with Telegram form. v3 reframes (per Diff #13): use `clarify`-shape with structured questions per missing field; no new "wizard" infra.
- v2 says z0 is authoritative for coarse, Z1 for refined — correct. v3 makes z0 ↔ Z1 hand-off explicit: `0.4a` reads `mission_preflight.compliance_fingerprint` (z0 output, lives at `workspace/mission_<id>/.preflight/`) and only asks deltas for Z1-refined fields.
- v2's `compliance_blocker_check` runs at every phase boundary ≥7. v3 is more explicit: it's a mechanical post-hook on a designated "phase boundary" step. No such step exists in i2p_v3 today; v3 adds `0.4a_blocker_check` as a no-op step appended to phase 6's terminal step.

#### Chosen shape (locked)
- New step `0.4a compliance_fingerprint_collection` (mechanical clarify) — depends_on `0.4`.
- New step `1.11a compliance_overlay` (analyst LLM) — depends_on `1.11`.
- New mechanical action `compliance_template_present` (post-hook on 1.11a).
- New mechanical action `compliance_blocker_check` (post-hook on phase-6 terminal step `6.6`).
- New tool `compliance_template_render` (LLM-callable) backing 1.11a.
- Modified `0.6`, `1.11`, `3.3`, `3.4` consume `compliance_fingerprint`.

#### Inline JSON artifact examples

**compliance_fingerprint — POPULATED (mission F2):**
```json
{
  "_schema_version": "1",
  "source": "z0_then_z1_delta",
  "collected_at": "2026-05-09T13:00:00Z",
  "jurisdictions": ["EU", "US"],
  "user_classes": ["health", "children"],
  "data_categories_coarse": ["pii", "health", "minors_data"],
  "data_residency_required": true,
  "age_gate_required": true,
  "third_party_processors_expected": ["auth0", "sendgrid"],
  "data_export_requirements": ["gdpr_dsar", "hipaa_access"],
  "retention_max_days": 2555,
  "founder_attestations": {
    "founder_will_sign_dpa_with_processors": true,
    "founder_acknowledges_not_legal_advice": true
  }
}
```

**compliance_overlay — POPULATED:**
```json
{
  "_schema_version": "1",
  "required_documents": [
    {
      "doc_type": "privacy_policy",
      "applicable_jurisdictions": ["EU", "US"],
      "template_id": "gdpr_hipaa_privacy_v3",
      "template_version": "3.1",
      "last_reviewed": "2026-02-15",
      "generated_template_path": "workspace/mission_57/.compliance/legal/privacy_policy.md",
      "founder_review_required": true,
      "blocker_for_phase": 13,
      "requires_legal_review": true
    },
    {
      "doc_type": "data_processor_compatibility",
      "applicable_jurisdictions": ["US"],
      "template_id": "hipaa_processor_audit_v1",
      "template_version": "1.0",
      "last_reviewed": "2026-04-01",
      "generated_template_path": "workspace/mission_57/.compliance/legal/processor_audit.md",
      "founder_review_required": true,
      "blocker_for_phase": 4,
      "requires_legal_review": false
    }
  ],
  "monitoring_obligations": [
    {"obligation": "HIPAA audit log retention 6y", "owner": "agent",
     "automation_status": "automated"}
  ],
  "data_subject_rights_implementation": {
    "access": "automated_via_user_dashboard",
    "deletion": "automated_via_admin_panel",
    "portability": "automated_export_json",
    "rectification": "user_dashboard"
  },
  "cross_jurisdiction_conflicts": [
    {"conflict": "GDPR right-to-erasure vs HIPAA 6-year retention",
     "resolution": "Pseudonymize on user erasure request; retain audit log per HIPAA with PII removed."}
  ]
}
```

**compliance_fingerprint — EMPTY (founder skipped at prototype):**
```json
{
  "_schema_version": "1",
  "source": "z1_intake",
  "collected_at": "2026-05-09T13:00:00Z",
  "jurisdictions": ["UNKNOWN"],
  "user_classes": ["UNKNOWN"],
  "data_categories_coarse": [],
  "data_residency_required": false,
  "age_gate_required": false,
  "third_party_processors_expected": [],
  "data_export_requirements": [],
  "retention_max_days": null,
  "founder_attestations": {
    "founder_will_sign_dpa_with_processors": false,
    "founder_acknowledges_not_legal_advice": true
  }
}
```
At prototype tier this is acceptable; reviewers warn rather than block.

#### Inline Python pseudo-code

**`compliance_template_render` — new tool function:**
```python
# packages/mr_roboto/src/mr_roboto/compliance_template_render.py
import os, datetime
from jinja2 import Template

TEMPLATE_ROOT = "data/compliance_templates"
STALE_DAYS = 180

def compliance_template_render(fingerprint: dict, doc_type: str,
                               lang: str = "en") -> dict:
    juris = fingerprint["jurisdictions"][0] if fingerprint["jurisdictions"] else "default"
    tpl_path = os.path.join(TEMPLATE_ROOT, juris, lang, f"{doc_type}.md.j2")
    if not os.path.isfile(tpl_path):
        # fall back to en-default
        tpl_path = os.path.join(TEMPLATE_ROOT, "default", "en", f"{doc_type}.md.j2")
    if not os.path.isfile(tpl_path):
        return {"ok": False, "error": f"no template for {doc_type} in {juris}/{lang}"}
    meta_path = tpl_path.replace(".md.j2", ".meta.json")
    if os.path.isfile(meta_path):
        meta = json.loads(open(meta_path).read())
        last_reviewed = datetime.date.fromisoformat(meta["last_reviewed"])
        age = (datetime.date.today() - last_reviewed).days
        if age > STALE_DAYS:
            return {"ok": False, "error": f"template_stale: last_reviewed={last_reviewed}, age={age}d"}
    tpl = Template(open(tpl_path, encoding="utf-8").read())
    rendered = tpl.render(**fingerprint, generated_at=datetime.datetime.utcnow().isoformat())
    return {"ok": True, "rendered": rendered, "template_id": doc_type, "template_version": meta.get("version", "0")}
```

**`compliance_blocker_check` mechanical action:**
```python
# packages/mr_roboto/src/mr_roboto/compliance_blocker_check.py
async def compliance_blocker_check(task: dict) -> Action:
    mission_id = task["mission_id"]
    payload = task.get("payload") or {}
    current_phase = int(payload.get("current_phase", 6))
    overlay = await load_artifact(mission_id, "compliance_overlay")
    if not overlay:
        return Action(status="completed", result={"reason": "no overlay"})
    pending = []
    signoffs = await get_founder_signoffs(mission_id)
    for doc in overlay.get("required_documents") or []:
        if doc.get("blocker_for_phase", 999) <= current_phase:
            if doc.get("founder_review_required") and doc["doc_type"] not in signoffs:
                pending.append({"doc_type": doc["doc_type"],
                                "blocker_for_phase": doc["blocker_for_phase"]})
    if pending:
        return Action(status="failed",
                      error=f"compliance_blocker_check: {len(pending)} docs await founder signoff",
                      result={"pending": pending})
    return Action(status="completed", result={"checked": current_phase})
```

#### i2p_v3.json step JSON drafts

**New step `0.4a compliance_fingerprint_collection`:**
```json
{
  "id": "0.4a",
  "phase": "phase_0",
  "name": "compliance_fingerprint_collection",
  "agent": "mechanical",
  "executor": "mechanical",
  "depends_on": ["0.4"],
  "may_need_clarification": true,
  "input_artifacts": [],
  "output_artifacts": ["compliance_fingerprint"],
  "produces": ["workspace/mission_{mission_id}/.compliance/fingerprint.json"],
  "context": {
    "executor": "clarify",
    "kind": "compliance_fingerprint",
    "questions": [
      "Refined compliance fields. Reply per field or 'I dont know' (you'll be flagged if at private_beta+).",
      "data_residency_required (true/false):",
      "age_gate_required (true/false):",
      "expected third-party processors (comma-separated):",
      "retention_max_days (integer or null):"
    ]
  },
  "skip_when": "ambition_tier == 'prototype' and compliance_fingerprint_skipped_default == true"
}
```

**New step `1.11a compliance_overlay`:**
```json
{
  "id": "1.11a",
  "phase": "phase_1",
  "name": "compliance_overlay",
  "agent": "analyst",
  "difficulty": "medium",
  "tools_hint": ["compliance_template_render"],
  "depends_on": ["1.11"],
  "may_need_clarification": false,
  "input_artifacts": ["compliance_fingerprint", "regulatory_requirements"],
  "output_artifacts": ["compliance_overlay"],
  "produces": ["workspace/mission_{mission_id}/.compliance/overlay.json"],
  "post_hooks": ["compliance_template_present"],
  "instruction": "For each <jurisdiction × data_category> combination in compliance_fingerprint, identify required_documents (privacy_policy, cookie_banner, dpa, tos, retention_policy, age_gate, accessibility_statement, data_processing_record). Call compliance_template_render(fingerprint, doc_type, lang) for each. Save rendered output to generated_template_path. Surface cross_jurisdiction_conflicts where present. Set requires_legal_review=true on every doc; founder_review_required=true on privacy_policy, dpa, tos.",
  "done_when": "compliance_overlay.required_documents non-empty (or fingerprint says jurisdictions=[]) and all generated_template_path files exist."
}
```

**Modified `1.11 regulatory_research`** — `input_artifacts` += `["compliance_fingerprint"]`. Instruction prepend: `"Use compliance_fingerprint to scope research. Only research regulations applicable per jurisdictions × user_classes × data_categories_coarse. Skip generic 'GDPR overview' if EU not in jurisdictions."`

**Modified `3.3 nfr_availability_and_security`** — `input_artifacts` += `["compliance_fingerprint", "compliance_overlay"]`. Instruction append: `"Map every data_categories_coarse[i] to specific authentication+authorization+encryption controls. If compliance_overlay.required_documents includes a doc with requires_legal_review=true, list the corresponding NFR (e.g., 'audit-log retention NFR for HIPAA')."`

**Modified `3.4 data_requirements`** — Instruction append: `"Each entity's sensitivity_level and retention_policy MUST be consistent with compliance_fingerprint.data_categories_coarse and retention_max_days. Reject any entity proposing retention > retention_max_days."`

**Modified `6.6 project_plan_review`** (or whichever is the terminal step of phase 6 — verify against `i2p_v3.json`) — add `post_hooks: ["compliance_blocker_check"]` with context `{"compliance_blocker_check": {"current_phase": 6}}`.

#### Reviewer prompt criteria additions

**`1.11a` would-be reviewer** — Z1 doesn't add a new reviewer step; instead, audit lands at `4.16` via:

**`4.16 architecture_review`** — append:
```
Additional checks (added v3, P6):
- For each ADR with decision_domain in {auth, third_party_service, infrastructure, database}:
  verify the chosen option is compatible with compliance_fingerprint and compliance_overlay.
  Examples:
  - if compliance_fingerprint.data_residency_required=true AND chosen hosting region violates → status=fail.
  - if compliance_fingerprint.user_classes contains 'health' AND chosen processor lacks BAA in compliance_overlay.required_documents.processor_audit → status=fail.
- If compliance_overlay.cross_jurisdiction_conflicts non-empty AND no ADR addresses each conflict → status=needs_revision.
```

**`6.6 project_plan_review`** — append: `"Verify compliance_blocker_check did not surface pending docs for phase ≤6."`

#### Failure walkthroughs

**[synthetic] Mission F2 — would P6 have caught Vercel-HIPAA?** Yes:
- `0.4a` collects: `data_categories_coarse=[health, minors_data]`, `jurisdictions=[EU, US]`, `data_residency_required=true`.
- `1.11` (now scoped) returns: HIPAA + COPPA + GDPR. No "general internet regs" noise.
- `1.11a` emits `compliance_overlay.required_documents` with `data_processor_compatibility` doc; `blocker_for_phase: 4`.
- Phase 4 starts: `4.6 third_party_selections` (where Vercel would be picked) reads compliance_overlay; agent picks AWS HIPAA-eligible services instead of Vercel free tier.
- If agent still picks Vercel: `4.16` reviewer rejects with explicit citation.

**[synthetic] Mission F1 — TR/KVKK case.** Founder's `0.4a` says `data_residency_required=true` (KVKK). Agent at `4.6` would pick Hetzner Helsinki (acceptable for TR data residency under KVKK with adequacy decision) instead of US-only providers.

#### Engine extension required
- `mr_roboto/__init__.py`: 2 new dispatch branches (`compliance_template_present`, `compliance_blocker_check`).
- New tool registration: `compliance_template_render` in tool registry.
- New data: `data/compliance_templates/<jurisdiction>/<lang>/<doc_type>.md.j2` files + `.meta.json` siblings. Initial template set: GDPR-EU/en, HIPAA-US/en, KVKK-TR/tr, COPPA-US/en, default/en. All hand-curated (NOT agent-generated) — labeled `last_reviewed` per OQ4.
- New table: `founder_signoffs(mission_id INTEGER, doc_type TEXT, signed_at TEXT, signature_hash TEXT, PRIMARY KEY (mission_id, doc_type))`.

#### Telemetry plan
- Yazbunu `z1_compliance`: `{mission_id, jurisdictions, user_classes, data_categories_coarse, docs_required_count, conflicts_count, signoffs_pending}`.

#### Migration concrete
- Migration script creates `data/compliance_templates/` skeleton + populates 5 initial templates.
- New column `missions.legacy_pre_compliance INTEGER DEFAULT 0`.
- Read site: 4.16 / 6.6 reviewers + `compliance_blocker_check` skip when set.

#### Locked alternatives
- **Chose A (full step) over B (inline in 0.4).** Scenario where B wins: compliance is trivially "no PII, public US only" — full step is overkill. Decision: B is the prototype-tier degenerate path (`0.4a skip_when: ambition_tier == 'prototype' and ...`); A is the default.
- **Chose hand-curated templates over agent-generated.** Scenario where agent-generated wins: novel jurisdiction (e.g., new-country regulation lands tomorrow). Decision: defer to a P10e follow-up; the "180-day staleness" rule + manual quarterly refresh is the v3 baseline.

---

## New gaps surfaced (beyond v2)

These v2 still missed even after correcting v1.

### N1. Telegram interface flooding at private_beta+
Across P1+P2+P6, a private_beta+ mission could trigger: evidence intake clarify, brand pick clarify, compliance fingerprint clarify, all in phases 0-1. v2's R7 (founder fatigue) is about cumulative count across phases; v3 names the specific anti-pattern: **three clarify steps before phase 2**. Mitigation: add a Z1-level coordination, "consolidated phase-0 clarify" — `0.7 phase_0_consolidated_clarify` that batches outstanding clarifies into one Telegram session if more than 2 fire in phase_0. Defer the wiring; raise as known issue.

### N2. Reviewer LLM cost compounds across 5 reviewers + new prompts
Reviewer steps `1.13`, `3.11`, `4.16`, `5.10`, `6.6` get prompt additions from P1-P6. Each addition is +200-500 tokens. 5 reviewers × 500 tokens × 1.5 retry envelope = +3.75k tokens. Compared to per-mission ADRs (40k from P3) this is small, but cumulative reviewer cost rises ~25%. Telemetry must track per-reviewer token count explicitly.

### N3. Workspace scoping — `_safe_resolve` rejects paths starting with `workspace/mission_<id>/`
`src/tools/workspace.py:66-77` `_safe_resolve` joins relative paths under `WORKSPACE_DIR`. The `produces` paths in v3 use `workspace/mission_<id>/.evidence/index.json` (relative to repo root). When the LLM agent's `write_file` tool is called with this path, `_safe_resolve` will join it AGAIN under WORKSPACE_DIR → `workspace/workspace/mission_<id>/...` → fail. v3 specifies: `produces` paths in i2p_v3.json must use the **relative-to-WORKSPACE_DIR** form (`mission_{mission_id}/.evidence/index.json`); `verify_artifacts` and `check_grounding` already accept this via the `workspace_path` payload field (`mr_roboto/__init__.py:101, 127`).

**Correction to all P1-P6 `produces` strings:** drop the `workspace/` prefix. Final form: `mission_{mission_id}/.evidence/index.json`. The mechanical-action pseudo-code above resolves via `get_mission_workspace(mission_id)` which already returns the absolute path — those are correct.

### N4. `_schema_version` collision risk during parallel landings
v2 introduces `_schema_version: "1"` in proposal 7. If P1 and P3 both land bumping different artifacts to "1" but later need different schema "2", the cross-artifact join (e.g., P3 ADR consuming P1 evidence_refs) becomes ambiguous. v3 names: `_schema_version` is per-artifact-name, scoped via `(artifact_name, _schema_version)`. The reviewer prompt clauses must cite `_schema_version` explicitly (e.g., "applies to user_personas v1+").

### N5. No graveyard for ADR `supersedes`
P3's ADR has `supersedes_adr_id`. When ADR-2026-05-09-005 supersedes ADR-001, where does the old ADR live? v3 specifies: superseded ADRs stay in their original `4.X_decision` artifact; `adr_register` lists them with `status: "superseded"`. The reviewer at 4.16 verifies the `supersedes_adr_id` resolves.

---

## Sequencing — locked, with explicit dependency table

| Order | Proposal | Hard prereq | Soft prereq | Why this order |
|---|---|---|---|---|
| 1 | P7 — spec versioning + reviewer regression fixtures | none | none | Without `_schema_version` + fixture suite, every later proposal silently regresses reviewers. |
| 2 | P4 — failure-mode column | P7 | none | Smallest schema change; `0.3 assumption_identification` precedent reduces risk; trains the "schema bump → reviewer prompt → fixture" loop. |
| 3 | P3 — ADRs as first-class | P7 | P4 (falsification fields shape ADR's falsification_signal subset) | Schema-heaviest; reuses P4's falsification language. |
| 4 | P6 — compliance fingerprint | P7 | z0 Phase B (preflight wizard ships) | Independent schema; new mechanical actions + tool registration; templates are static data. |
| 5 | P5 — web-grounded prior art | P7 | none | Heaviest tool/module work but cleanly contained; can run parallel to P3 if engineering capacity. |
| 6 | P1 — structured intake + P8 — pitch-evidence conflict surface | P7, z0 Phase B | P4 (evidence_refs sentinel `agent_inference` predates is_inference field) | Highest founder-UX impact; rip up phase 0 ordering. P1 + P8 land atomically. |
| 7 | P2 — taste delegation | P7 | P1 (mood_board rationale references evidence-backed personas) | Downstream value of P1; also gated on Vecihi `return_bytes` extension. |
| 8 | P9 — cross-mission inheritance | P7 | All of P1, P3, P6 | Requires prior missions to have all the new shapes to inherit. |

**Parallel safe:** P5 with P3 (different surfaces). **Hard blocking on z0:** P1, P6, P9.

**DAG:**
```
P7 ─┬─ P4 ─ P3 ─┬─ P5
    │           │
    ├─ P6 ───┐  │
    │        │  │
    └─ P1+P8 ┴──┴─ P2 ─ P9
```

---

## Risks register (extended)

| # | Risk | Source | Affects | Mitigation |
|---|---|---|---|---|
| R1 | Token-cost compounding | v2 | All | Ambition-tier gating; prototype tier strips P1 intake / P3 alternatives / P5 deep / P6 full overlay. |
| R2 | Reviewer prompt regressions | v2 | All | P7 ships first; block subsequent merges on regression-fixture pass. v3 inlines prompt deltas to make CI-checkable. |
| R3 | In-flight mission breakage | v2 | All | Per-proposal migration; legacy-tag column on `missions`; reviewers gated on tag. |
| R4 | Engine surface mismatch | v2 | P1, P2, P3, P5, P6 | v2/v3 specify new mechanical actions; lands as new files in `packages/mr_roboto/src/mr_roboto/`. |
| R5 | Vecihi tier escalation cost on prior_art | v2 | P5 | HTTP tier only for Wayback/HN/Wiki; browser only for Crunchbase fallback. |
| R6 | Concurrent agents-overhaul churn | v2 | P3, P4, P6 | `docs/plans/2026-05-08-agents-overhaul.md` line 13 confirms no agent renames; R6 reduces from blocker to advisory. Keep coordination on landing windows for prompt-polish steps (Task 2 of overhaul touches workhorse prompts; reviewer prompt edits in v3 are separate prompts but check for collision). |
| R7 | Founder fatigue | v2 | P1, P2, P6 | Strict ambition-tier gating; prototype skips P1/P2-full/P6-full; private_beta+ pays. v3 N1 raises consolidated-clarify follow-up. |
| R8 | Hallucinated prior-art | v2 | P5 | `prior_art_min_coverage` Wayback/HN check; explicit `dead_or_hallucinated` status. |
| R9 | Compliance template staleness as legal risk | v2 | P6 | 180-day refusal in render tool; explicit non-legal-advice watermark; `requires_legal_review: true` blocker for `public_launch`. |
| R10 | z0 contract drift | v2 | P1, P2, P5, P6, P9 | Block landing of P1/P6/P9 until z0 Phases A+B merged AND z0 surface contract locked. P2 gated on z0 ambition_tier signal. |
| R11 | Schema-version forking during parallel landing | v2 | All | Single-merge-train per Sequencing; P7 ships first; per-artifact scoping per N4. |
| R12 (NEW v3) | Workspace path resolution mismatch | v3 N3 | All | Use `mission_{mission_id}/...` relative-to-WORKSPACE_DIR in `produces`; mechanical actions resolve via `get_mission_workspace`. CI lint on i2p_v3.json `produces` strings. |
| R13 (NEW v3) | Phase 0 clarify storm | v3 N1 | P1+P2+P6 | `0.7_phase_0_consolidated_clarify` follow-up; for v3 launch, only land 1 of P1/P2/P6 at a time per mission cohort to avoid cluster firing. |
| R14 (NEW v3) | Reviewer token cost rise | v3 N2 | All reviewers | Telemetry per-reviewer token count; if cumulative cost > 10% of mission tokens, split reviewer steps. |
| R15 (NEW v3) | `request_review` action assumed but not present | v3 Diff #4 | None of P1-P6 (they don't depend on it) | Z1 doc proposes `request_review` — confirm it's deferred to a separate proposal (P10) outside v3 scope; do not block. |
| R16 (NEW v3) | Founder file-drop UX has no precedent | v3 Diff #13 | P1 | Use `clarify`-shape ack; defer `/evidence` slash command to P10b. Founder must understand "drop file then reply DONE." |
| R17 (NEW v3) | `prompt_versions` not used by Z1 | v3 Diff #8 | All reviewer prompt edits | If reviewer prompt diffs don't go through `prompt_versions`, they bypass the rollback path. v3 mandates: every reviewer prompt edit lands as a new `prompt_versions` row, then `i2p_v3.json` references the version_id. Requires verifying current reviewer-step instruction loading honors `prompt_versions`; if not, file as P10f. |

---

## Updates

(none — this is v3's initial cut)
