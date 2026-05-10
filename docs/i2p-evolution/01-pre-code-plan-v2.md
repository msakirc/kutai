# Z1 — Pre-code execution plan (v2: deep-dive)

**Date:** 2026-05-08
**Source doc:** [01-pre-code.md](01-pre-code.md)
**Predecessor:** [01-pre-code-plan.md](01-pre-code-plan.md) (v1)
**Reads:** v1, [00-README.md](00-README.md), [z0-mission-preflight.md](z0-mission-preflight.md), `src/workflows/i2p/i2p_v3.json`, `src/workflows/engine/expander.py`, `src/workflows/engine/hooks.py`, `packages/mr_roboto/src/mr_roboto/__init__.py`, `packages/mr_roboto/src/mr_roboto/verify_artifacts.py`, `packages/vecihi/src/vecihi/`, `docs/plans/2026-05-07-i2p-capability-expansion.md`.

v1 mapped 7 gaps to step IDs and proposed six phased plans. v1 was **structurally
correct but mechanically thin**: most "new post-hooks" are unimplementable as
written against the actual engine surface, ADR migration ignores in-flight
missions, prior-art tool spec ignores Vecihi's actual fetcher contract, and
several proposed schemas would silently bypass the auto-grounding rule that
fires whenever `produces` is non-empty (`expander.py:178-181`). v2 fixes all
six and adds three proposals v1 missed.

## Diff vs v1

Concrete corrections + additions, ordered by impact.

1. **Engine extension model is narrower than v1 implied.** v1 names new
   "post-hooks" as if they were free-form. The actual surface
   (`expander.py:169-181` + `mr_roboto/__init__.py:run()` dispatch) is: a
   post-hook is a string in `step.post_hooks`; the runner maps that string to
   a mechanical action via `mr_roboto.run({"payload": {"action": "<kind>"}})`.
   Every new post-hook v1 proposes (`evidence_refs_audit`,
   `image_attribution_present`, `adr_completeness_check`,
   `falsification_completeness`, `prior_art_min_coverage`,
   `compliance_template_present`, `compliance_blocker_check`) requires a new
   action branch in `mr_roboto/__init__.py` between line 348 (`return Action(status="failed", error=f"unknown mechanical action: {action!r}")`) and any executor module. v1 names none of these wiring points. v2 specifies them.
2. **`produces` auto-wires `grounding`.** v1 doesn't note this:
   `expander.py:178-181` *automatically prepends* a `grounding` post-hook to
   any step with `produces`. v1's Phase A (intake) Phase A grounding gate
   ("verify the index file exists OR contains literal `{"status": "founder_skipped_evidence"}`") would be misread by `mr_roboto.check_grounding`
   — that hook checks tool_calls audit log against produces paths, not
   artifact body content. v2 splits this into a *separate* `evidence_index_grounding` action (or uses `verify_artifacts` with the produced file path).
3. **`4.14 architecture_decisions` already calls itself ADR.** v1's "top
   surprise #1" caught this but v2 adds: it has `min_items: 3` only on the
   **count of ADRs**, not alternatives-per-ADR. So v1's claim "phase 4 emits
   the chosen pattern; the rejected three are lost" is precisely true; the
   shape change is converting `item_fields: [title, status, context, decision, consequences]` (line 3164-3171) to include `options_considered: [{...}]` and forcing this shape upstream into 4.1, 4.2, 4.6, 4.8. v1 says "convergence to ADR shape"; v2 names exact JSON field paths, validation rules, and shows that **`4.14 register synthesis` becomes register-only — but only after backfill of 4.1–4.8 emits ADR-shaped output for the in-flight mission**.
4. **v1's `find_prior_art` ignores Vecihi's actual interface.** Vecihi
   (`packages/vecihi/src/vecihi/fetchers.py`) exposes `scrape_url` /
   `scrape_urls` over four tiers (HTTP→TLS→Stealth→Browser); it has **no
   search capability** and no API client. HN Algolia, Wayback CDX, MediaWiki
   are JSON HTTP endpoints — those should land in a NEW `vecihi.search`
   submodule (or even better, a new package since "search aggregator" is a
   different abstraction from "scrape one URL"). v1 says "lands in
   `packages/vecihi/`" without naming the file; v2 specifies
   `packages/vecihi/src/vecihi/prior_art.py` as a sibling module
   that *uses* `scrape_url` for verification but adds its own search-API
   client.
5. **v1 misses `failure_mode` already-shipped reviewers' verdict gates.**
   Reviewer steps (`1.13`, `3.11`, `4.16`, `5.10`) all have
   `status: {equals: ["pass"]}` (1.13 uses `verdict.equals: ["pass"]`,
   3.11 uses `status.equals: ["pass"]`, 4.16 uses `status.equals: ["pass", "approved"]`). Adding falsification fields means **reviewer prompts must
   be updated** to reject when `failure_mode` is empty/TBD; otherwise reviewers
   pass missions that lack the new fields. v1 names "reviewer at 3.11 rejects
   requirements where if_wrong_consequence == 'Needs validation'" but doesn't
   actually specify the prompt change or schema change to make that enforceable. v2 specifies both, and adds a regression risk: bumping reviewer
   strictness mid-flight will fail every in-flight mission's review pass.
6. **Compliance fingerprint position is contradictory with z0.** v1 says
   "z0 already collects compliance fingerprint; Z1's `0.4a` reads z0 output
   and *only asks the deltas*" but z0-mission-preflight.md §G says z0 collects
   "high-level" only and "Refined fully in `01-pre-code.md`". So z0 and Z1
   *both* collect, and Z1 should be authoritative for refined data
   categories / age-gate / processors. v1's "0.4a skips with passthrough if z0
   supplied" is wrong — z0 cannot supply the refined shape. v2 names exactly
   which fields z0 owns (`jurisdictions`, `user_classes`, `data_categories`
   coarse) and which fields Z1 must always collect (`age_gate_required`,
   `data_residency_required`, `third_party_processors`).
7. **Mood-board step (`5.5a`) collides with existing `5.5 wireframe_review`.**
   v1's new step ID `5.5a` collides with existing `5.5 wireframe_review`
   (`i2p_v3.json:3539-3572`). v2 renumbers to `5.5b1` (after `5.5`, before
   `5.6`) and notes the dependency: brand selection happens **after**
   wireframe review completes (`5.5`) so the founder isn't asked to pick
   visuals before flow shape is locked.
8. **v1 doesn't address artifact migration for in-flight missions.** ADR
   shape change + falsification field addition + compliance fingerprint
   require backfill on missions paused at any phase. v2 adds a migration plan
   per proposal.
9. **v1 doesn't propose reviewer prompt regression suite.** Adding fields to
   schemas means reviewer prompts need updating; the prompts are LLM-driven
   and silently regress. v2 proposes a small canonical-fixture suite per
   reviewer step.
10. **v1 missing: spec versioning.** When schemas evolve, `evidence_refs`
    on persona v1 and `evidence_refs` on persona v2 must be distinguishable.
    v2 adds a `_schema_version` field convention to every artifact this plan touches.
11. **v1 missing: cost ceiling interaction.** Each proposal costs LLM/HTTP/founder-time. v1 estimates effort but not per-mission token/HTTP cost.
    v2 attaches a per-proposal "cost added per mission" estimate.
12. **v1 missing: shopping/agents-overhaul interaction.** Latest commits
    (M src/agents/{implementer,planner,test_generator}.py from gitStatus) +
    docs/plans/2026-05-08-agents-overhaul.md hint at agent-shape churn
    landing concurrently. v2 adds a Risks register entry.

---

## Per-proposal deep-dive

### 1. Structured user-research intake (closes G1, G7)

#### Re-audit
v1 cites `0.1`, `0.2`, `1.2`, `2.4`, `2.9`, `3.7` — verified at `i2p_v3.json:683-718` (0.1 instruction "The user's raw idea is provided in the Context Artifacts section below. Take it exactly as stated"), `:720-751` (0.2 fields include `severity`/`frequency` with "Needs validation" sentinel never validated), `:937-971` (1.2 input_artifacts is just `idea_brief_final`, no founder evidence channel), `:1632-1681` (2.4 input is `audience_research_data_summary`+`value_proposition_canvas` — no evidence_refs slot exists). All correct.

**Missed by v1:**
- `0.3 assumption_identification` (`:753-785`) is the **closest existing precedent** for evidence-aware fields. Its `risk_if_wrong` + `validation_method` schema is exactly the falsifiability shape — and it has no evidence channel either. v1 calls this out under "top surprises" but doesn't extend evidence intake into 0.3.
- `2.2 value_proposition_canvas` (`:1597-1597`+) extracts `customer_profile.{jobs,pains,gains}` from `audience_research_data_summary` only. Pains and jobs are exactly where founder interview evidence should land. v1 misses this step.
- `1.7 competitor_review_and_sentiment` (`:1144-1195`) is a natural inbound for founder-collected reviews / quotes — v1 misses this.

#### Steel-man counter
"Don't add intake; force founders to put evidence in the original pitch" is the strongest counter. Reasons it loses anyway: voice memos and PDFs can't paste into Telegram; the pitch is written before research; the founder doesn't know which evidence the agent needs. The real second-order effect v1 missed: **founders without evidence will skip the gate, and the agent will then mark every persona `is_inference: true`** — the system is honest but every later phase is built on inference, defeating the point. Mitigation: ambition-tier-gated requirement (z0 already supplies tier).

#### Alternatives (3 shapes)
- **A. v1's shape: dedicated 0.0a/0.0b intake steps + universal `evidence_refs` field.** Strongest. Cost: high (6 steps modified, new artifact, new tool, new mechanical action). Migration: in-flight missions backfill `evidence_refs: ["agent_inference"]`.
- **B. Inline evidence in 0.1.** Founder pastes interview text directly in the pitch under a `## Evidence` section; agent parses; no new step. Cost: low. Loses: voice memos, screenshots, PDF surveys (multi-modal evidence the v1 shape supports).
- **C. Side-channel attachment-only intake (no Telegram form).** Founder drops files into `missions/<id>/evidence/`; agent picks up at 0.2 with a directory scan. Cost: medium. Loses: structured per-claim metadata; agent has to re-extract every time.

**Recommendation:** A for `private_beta+`, B for `prototype` ambition tier. C is a fallback for power users with prior evidence corpora.

#### Concrete schema
```json
// evidence_index (artifact)
{
  "_schema_version": "1",
  "mission_id": 57,
  "evidence_items": [
    {
      "evidence_id": "EV-001",
      "source_type": "interview" | "survey" | "screenshot" | "voice_memo" | "prior_research" | "analytics_export",
      "source_uri": "missions/57/evidence/EV-001.txt",
      "captured_at": "2026-05-08T12:00:00Z",
      "founder_note": "string, optional",
      "redaction_status": "raw" | "redacted" | "founder_consented",
      "sha256": "<hash of source>"
    }
  ],
  "founder_skipped": false,
  "skip_reason": "string|null"
}
```
- All fields required except `founder_note`, `skip_reason`.
- `evidence_items` must be non-empty unless `founder_skipped: true`.
- Validation: `mr_roboto.evidence_index_check` (new action) — asserts `len(evidence_items) >= 1 OR founder_skipped is True`; asserts each `source_uri` resolves under workspace; asserts `sha256` matches file.

```json
// extracted_evidence (artifact, per-evidence)
{
  "_schema_version": "1",
  "evidence_id": "EV-001",
  "extracted_claims": [
    {
      "claim_id": "EV-001.C1",
      "claim": "string",
      "claim_category": "pain" | "current_tool" | "willingness_to_pay" | "demographic" | "jtbd" | "ux_complaint" | "feature_request" | "objection",
      "quote": "string|null",
      "confidence": "high" | "medium" | "low",
      "extracted_at": "iso8601"
    }
  ]
}
```

```json
// evidence_refs (field added to existing artifacts)
"evidence_refs": ["EV-001.C1", "EV-002.C3"]
// OR
"evidence_refs": ["agent_inference"]   // explicit sentinel
"is_inference": true
```

Field is required on every: persona item (in `2.4 user_personas`), problem_statement field-level (in `0.2`), feature item (in `2.5`), functional requirement (in `3.1`), business rule (in `3.7`), value-prop canvas pain/gain item (in `2.2`).

#### Wiring
- **New mechanical action:** `evidence_index_check` — adds case at `mr_roboto/__init__.py:348` (before unknown-action branch). Reads `payload.path = "missions/<id>/.evidence/index.json"`; loads JSON; asserts schema. Exists as a new file `packages/mr_roboto/src/mr_roboto/evidence_index_check.py`.
- **New mechanical action:** `evidence_refs_audit` — case in `mr_roboto/__init__.py`; new file `packages/mr_roboto/src/mr_roboto/evidence_refs_audit.py`. Signature: `audit(mission_id, artifact_name, threshold_inference_pct, ambition_tier) -> {ok, inference_pct, severity}`. Reads the artifact via `src/workflows/engine/artifacts.ArtifactStore`. Returns `severity: "warning"` (not blocker) if `inference_pct > threshold` AND `ambition_tier in {private_beta, public_launch, revenue_product}`. Hooked at end of 2.4, 2.9, 3.1, 3.7. **Severity routing** uses the existing `blockers: {field: severity, levels:[critical, high]}` pattern shipped 2026-05-05 — emit `severity: "warning"`, levels `["critical","high"]` → does not block, just surfaces.
- **New tool (LLM-callable):** `mr_roboto.request_evidence_uploads(mission_id, evidence_types: list[str], deadline: datetime|None)` — new file `packages/mr_roboto/src/mr_roboto/request_evidence_uploads.py`. Sends Telegram message via existing `notify_user`-style infra; opens upload window; on each upload writes to `missions/<id>/.evidence/EV-NNN.<ext>` and updates `index.json`. Wired as case in `mr_roboto/__init__.py`.
- **New extractor (analyst-driven, not mechanical):** Step `0.0b evidence_extraction` is an LLM step not a mechanical one — agent type `analyst`, tools_hint `read_file`, `vision` (already exists at `src/tools/vision.py`), produces `extracted_evidence`. Audio is out-of-scope per OQ5.
- **Step insertion order in i2p_v3.json:** Insert `0.0a` and `0.0b` **between mission start and existing `0.1`**. This requires `0.1` and downstream to gain `evidence_index` + `extracted_evidence` in `input_artifacts`. The auto-grounding rule (`expander.py:178-181`) fires on `0.0a` because we declare `produces: ["missions/<id>/.evidence/index.json"]` — but `<id>` resolution at expansion time is fine because expander has mission_id in scope (verify with `expander.py` codepath; otherwise use a sentinel and resolve at hook-time).
- **Failure mode triggers:** If `0.0a` fails grounding → `dlq` (consistent with existing convention; recoverable via `/dlq retry`). If `evidence_refs_audit` returns `severity: warning` → emit a Telegram notification but continue.

#### Cost / latency
- Per mission: +1 mechanical Telegram round-trip (intake), +1 founder wait (variable, default 24h per OQ8 idle threshold), +N extraction LLM calls (1 per evidence item; bounded ≤ 20 typical). Token cost: ~2k input + 1k output per item ≈ 3k × 10 items = 30k tokens added.
- Latency: bounded by founder; mission can wait via `waiting_human` task status.

#### Migration
- In-flight missions (any phase) at landing: add nullable `evidence_index` + `extracted_evidence` artifact slots; backfill `evidence_refs: ["agent_inference"]` on every existing persona / FR / BR / feature item via a **one-shot DB migration script** (`scripts/migrations/2026-05-XX-evidence-refs-backfill.py`). Already-completed missions: read-only, no backfill. Reviewers: prompts updated to accept `evidence_refs == ["agent_inference"]` as valid for legacy missions tagged `legacy_pre_evidence: true` on the mission row.
- Schema evolution: every artifact gains `_schema_version: "1"`; future bumps update both the artifact-emitter prompt and the validator.

#### Prior-art outside KutAI
**Dovetail / EnjoyHQ pattern**: structured research repository where each "highlight" (= `extracted_claim`) links back to source recording with timestamp; tags are taxonomy-driven; cross-tagging surfaces themes. Our `claim_category` enum is a lite version of that; the `quote` field is the highlight reference. **Lean Canvas's "riskiest assumption test"** (Ash Maurya): every claim must be testable. Our `confidence` field encodes that lite. **`design-research-process` from IDEO**: differentiates "data" (what was said) from "insight" (what it means). Our `claim` vs `claim_category` separation echoes this; the `is_inference` flag is the analytical-vs-empirical line.

#### Prior-agent missed
- Reviewer prompt regression: `3.11 requirements_review` and `1.13 research_quality_review` need prompt updates to flag `evidence_refs: []` as an issue.
- Vision-tool dispatch on screenshots: existing `src/tools/vision.py` is dispatched as part of `extracted_evidence`; if vision tool fails or isn't available, `0.0b` should not block — fall back to "unable to extract" with `is_inference: true`.
- `2.2 value_proposition_canvas` and `1.7 competitor_review_and_sentiment` should also gain `evidence_refs` — v1 missed both.
- The `0.5 human_clarification_request` (line 822-851) cap of "TOP 5 most impactful questions" needs to honor evidence-vs-pitch conflicts as a priority signal. Today it picks 5 from `open_questions_list` + `assumption_list` only — needs `pitch_evidence_conflicts` added.

---

### 2. Taste delegation (mood boards, tone-of-voice samples)

#### Re-audit
v1 cites `5.6 brand_and_design_tokens` — verified at `i2p_v3.json:3573-3618`. Confirms: (a) `may_need_clarification: true` with weak inline gating ("if multiple valid directions"), (b) directly emits `brand_identity` + `design_tokens`, (c) `tools_hint: []` — no scrape capability today.

**v1's step ID `5.5a` is wrong: collides with `5.5 wireframe_review` at line 3540.** v2 renumbers to `5.5b1` (or `5.5.5`). Position in DAG matters: must depend on `5.5 wireframe_review` (so wireframes are stable) and feed `5.6`. New depends_on: `[5.5, 2.1, 2.4, 1.8]`.

**Missed by v1:** `5.7 component_specs` (`:3620-3653`) reads `design_tokens`. So the chosen brand direction propagates from `5.5b1 → 5.6 → 5.7`. The mood-board choice must be recorded as a *decision artifact* (not just an artifact) so `5.10 design_review` can audit "we picked option B because…". v1 has `selected_brand_direction_id` but no audit/justification field.

#### Steel-man counter
"3-5 mood boards × every mission is wasteful and most founders pick option 1." Counter: Wave 7 of 2026-05-07 plan defers in-house design tokens — until then, scraped boards are the cheapest provenance signal. Second-order effect v1 missed: **scraped images carry attribution debt** (license, source, expiry of the source URL). If the founder ships the product with a scraped reference image baked into pitch deck or onboarding, that's a takedown vector. Mitigation: every `mood_board` item carries `attribution_required: true` flag; downstream marketing zone (07-humanish-layers) honors as "do not use this image; reference only."

#### Alternatives (3 shapes)
- **A. v1's shape: scrape 3-5 reference boards via web search.** Provenance preserved; founder picks. Cost: medium (web fetches + vision extraction).
- **B. Curated in-house board library.** Pre-built directions ("modern SaaS / playful consumer / regulated finance / craft / tactical-cli") — founder picks from N=10 templates. Cost: low (no web fetch). Loses: market-tied novelty (template can't reflect 2026 visual zeitgeist).
- **C. Founder-supplied reference URLs.** Founder pastes 3-5 URLs they like; agent extracts tokens. Cost: lowest. Loses: pure delegation (founder still does the curation work). Best for ambition tier `prototype`.

**Recommendation:** B for `prototype` (no extra LLM cost), A for `private_beta+`. Tier from z0.

#### Concrete schema
```json
// brand_direction_options (artifact)
{
  "_schema_version": "1",
  "options": [
    {
      "option_id": "BD-001",
      "name": "Calm Professional",
      "rationale": "Anchors to user_persona ‘Senior PM, 38, regulated industry’ frustration ‘cluttered dashboards’.",
      "mood_board": [
        {
          "image_url": "<scraped local path under missions/<id>/.brand/BD-001/img-N.png>",
          "source_url": "<original URL>",
          "attribution_required": true,
          "fetched_at": "iso8601",
          "sha256": "<hash>",
          "license_signal": "unknown" | "creative_commons" | "all_rights_reserved"
        }
      ],
      "tone_samples": [
        {
          "voice_attribute": "concise" | "warm" | "authoritative" | "playful",
          "sample_copy_blocks": ["string", "string"]
        }
      ],
      "risks": ["string"]
    }
  ],
  "min_options": 3,
  "max_options": 5
}

// selected_brand_direction (artifact, populated by clarify response)
{
  "_schema_version": "1",
  "option_id": "BD-002",
  "founder_note": "string, optional",
  "selected_at": "iso8601",
  "rejected_options": ["BD-001", "BD-003", "BD-004"]
}
```

#### Wiring
- **New step `5.5b1 brand_direction_proposals`:** agent `analyst`, tools_hint `[web_search, scrape_url, vision]`, `triggers_clarification: true`, depends_on `[5.5, 2.1, 2.4, 1.8]`. `produces: [missions/<id>/.brand/index.json]`. Auto-grounded by `expander.py:178`.
- **New tool:** `vecihi.scrape_image(url, max_dim_px=1024) -> {local_path, mime, hash, source_url, fetched_at}`. Lands at `packages/vecihi/src/vecihi/scrape_image.py`. Reuses `scrape_url` for the underlying GET; falls through tier escalation. Refuses URLs over 5MB. Saves under `missions/<id>/.brand/<option_id>/img-N.<ext>`.
- **New mechanical action:** `image_attribution_present` — case in `mr_roboto/__init__.py`. New file `packages/mr_roboto/src/mr_roboto/image_attribution_present.py`. Asserts every `mood_board[].image_url` resolves to a local path under workspace AND `source_url` is non-empty AND `sha256` matches file. Returns `severity: "high"` blocker if any miss.
- **Modify `5.6 brand_and_design_tokens`:** `input_artifacts` += `[brand_direction_options, selected_brand_direction]`; instruction prepends "Use `selected_brand_direction.option_id` to look up the chosen option in `brand_direction_options.options[]`. Materialize `design_tokens` from that option's mood_board (color extraction via vision tool) and tone_samples." Remove the inline `if multiple valid directions, trigger needs_clarification` clause — delegated to `5.5b1`.
- **Failure modes:** scrape_image fails → vecihi tier escalation; if all tiers fail, agent records `mood_board` item with `image_url: null, fetch_failed: true` and continues with remaining boards. If <3 boards have at least one image, `image_attribution_present` blocks → revise.

#### Cost / latency
- Per mission: +N web_searches (≈4-6), +5×3 = ~15 image fetches, +1 founder wait. Token cost: ~5k for option drafting; image fetch is HTTP not LLM. Latency dominated by scrape (browser-tier ~30s × 15 = up to 7m worst case; HTTP-tier <2s × 15 ≈ 30s typical).

#### Migration
- In-flight missions at phase ≤5: prompt founder to fill brand selection retroactively. Phase >5: skip backfill; tag mission `pre_taste_delegation: true`; downstream marketing reads existing `brand_identity` directly.

#### Prior-art outside KutAI
**Mubu / Brandbird / Looka pattern**: 3-5 logo/palette options generated at intake; user picks. Their UX is the gold standard. Our scrape-not-generate per OQ2 *trades* novelty for provenance. **Spotify Backstage's ADR pattern for design**: every design decision logged. We'll do that via `selected_brand_direction.rejected_options` audit. **Figma's Variables release** (2024): design tokens as the SoT for cross-platform — reinforces v1's `design_tokens` continuing to be the downstream consumer.

#### Prior-agent missed
- License signal classification — not in v1's schema. Not a takedown defense but a flag for downstream.
- `tone_samples` propagation: 07-humanish-layers (marketing) reads — but does it read from `brand_direction_options[chosen]` or from `5.6 brand_identity`? v1 doesn't say. v2 says: 07 reads `selected_brand_direction.option_id` joined to `brand_direction_options` (preserves rationale + risks for marketing).
- Re-rendering on rejection: if founder rejects all 5, what triggers re-generation? v1 has no "I don't like any of these" branch. v2: clarify response includes "regenerate with hint: <text>" path; bounded to 3 cycles.

---

### 3. ADRs as first-class

#### Re-audit
v1 cites `4.1`, `4.2`, `4.6`, `4.8`, `4.14` — verified at `:2544-2581`, `:2583-2629`, `:2795-2835`, `:2873-2917`, `:3133-3175`. **v1's "top surprise #1" claim is correct**: `4.14` already emits ADRs (line 3162-3171: `adrs` array with `min_items: 3`, `item_fields: [title, status, context, decision, consequences]`) — but synthesised post-hoc. Each upstream step (4.1 etc.) emits a single decision artifact and `4.14` reads them all together to write ADRs.

**Missed by v1:**
- `4.4 database_schema_design` (`:2666-2702`) bakes a normalization decision (`3NF minimum`) without an ADR. Adding ADR shape upstream means database choice gets ADR treatment — v1 doesn't cite 4.4 but should.
- `4.9 infrastructure_designs` (`:2918-2971`) — caching tech choice, queue tech choice, file storage service. Three more ADR-able decisions. v1 misses.
- `4.10 communication_designs` — email service, search tech, realtime tech. Three more.
- Therefore: v1 names 4 ADR-emitting steps; the real count is 7+. ADR scope is much bigger than v1 framed.

#### Steel-man counter
"3+ alternatives × 7 decisions × every mission inflates token cost." Counter: alternatives capped at 80 words each (v1's mitigation, retained). Token cost per ADR: ~1.5k input + 1.5k output × 7 ADRs = 21k tokens added. That's real. Mitigation: ambition-tier-gated alternative count (prototype: 2 options; private_beta+: 3+).

Second-order effect v1 missed: **the `4.16 architecture_review` reviewer prompt** asks reviewer to validate "every FR addressable, all NFRs have design elements" (line 3324). With ADR-shaped decisions the reviewer must additionally audit "every accepted ADR's `falsification_signal` is testable; every rejected option's `rejected_because` is non-empty." The reviewer prompt **must be updated**, otherwise the new structure passes review trivially.

#### Alternatives (3 shapes)
- **A. v1's shape: ADR-per-decision in choosing step; 4.14 becomes register.** Strongest. Highest token cost. Best audit trail.
- **B. Single `architecture_decisions_register` artifact (no upstream change).** Keep 4.1/4.2/4.6/4.8 as today; expand 4.14 to capture alternatives during synthesis. Cost: lowest. Loses: alternatives are reconstructed post-hoc, prone to LLM hallucination of "rejected options I never actually evaluated."
- **C. RFC-style threaded ADR.** Each ADR is a conversation: agent drafts → reviewer comments → agent revises → final. Cost: highest (2-3 LLM passes per ADR). Best decision quality. Probably overkill for prototype/private_beta.

**Recommendation:** A for all tiers; pair with reviewer prompt update at 4.16.

#### Concrete schema
```json
// ADR (used as artifact shape for 4.1, 4.2, 4.4, 4.6, 4.8, 4.9, 4.10)
{
  "_schema_version": "1",
  "adr_id": "ADR-2026-05-08-001",
  "title": "string, ≤80 chars",
  "status": "proposed" | "accepted" | "superseded" | "rejected",
  "decision_domain": "architecture_pattern" | "frontend_framework" | "backend_framework" | "database" | "infrastructure" | "auth" | "third_party_service" | "caching" | "queue" | "storage" | "email" | "search" | "realtime",
  "context": "string, ≤300 words",
  "options_considered": [
    {
      "option_id": "OPT-A",
      "name": "string",
      "pros": ["string", ...],   // ≤5 bullets, ≤30 words each
      "cons": ["string", ...],
      "evaluation_score": 0.0,    // 0-10, justification in pros/cons
      "rejected_because": "string, ≤80 words",
      "evidence_refs": ["EV-001.C2"] | ["agent_inference"]
    }
    // min_items: 2 for prototype, 3 for private_beta+
  ],
  "decision": "string, ≤200 words — references option_id of chosen",
  "chosen_option_id": "OPT-B",
  "consequences": {
    "positive": ["string"],
    "negative": ["string"],
    "neutral": ["string"]
  },
  "falsification_signal": {
    "hypothesis": "string — what we believe must hold",
    "signal_if_wrong": "string — observable signal that hypothesis is false",
    "kill_threshold": "string — quantitative if possible"
  },
  "revisit_trigger": "string — what event or threshold triggers reopening this ADR",
  "reversal_cost": "low" | "medium" | "high" | "irreversible",
  "supersedes_adr_id": "ADR-2026-..." | null,
  "decided_at": "iso8601",
  "decided_by": "agent" | "agent_with_clarification"
}
```

```json
// adr_register (artifact emitted by reframed 4.14)
{
  "_schema_version": "1",
  "adrs": [
    {
      "adr_id": "ADR-...",
      "title": "string",
      "status": "accepted",
      "decision_domain": "string",
      "summary": "string, ≤30 words",
      "links_to": ["adr_id"],   // dependencies
      "supersedes": ["adr_id"]
    }
  ],
  "completeness_check": {
    "required_domains": ["architecture_pattern", "frontend_framework", "backend_framework", "database", "infrastructure", "auth"],
    "missing": [],
    "vendor_adrs_count": 5
  }
}
```

#### Wiring
- **Modify steps 4.1, 4.2, 4.4, 4.6, 4.8, 4.9, 4.10:** Each emits one or more ADR artifacts (per `decision_domain`). 4.2 emits 4 ADRs (frontend/backend/database/infra). 4.9 emits 3 (caching/queue/storage). 4.10 emits 3 (email/search/realtime). Total: ~13 ADRs per mission.
- **Reframe `4.14` to `adr_consolidation`:** depends_on adds 4.4, 4.9, 4.10; output `adr_register`; instruction "Verify every required domain has ≥1 accepted ADR; emit register; flag missing."
- **New mechanical action:** `adr_completeness_check` — case in `mr_roboto/__init__.py`. New file `packages/mr_roboto/src/mr_roboto/adr_completeness_check.py`. Reads `adr_register`; asserts `completeness_check.missing == []`. Hooked at `4.14`.
- **New mechanical action:** `adr_alternatives_min` — case + new file `packages/mr_roboto/src/mr_roboto/adr_alternatives_min.py`. Asserts `len(options_considered) >= 2` (prototype) or `>= 3` (private_beta+). Reads ambition_tier from mission row. Hooked at end of every ADR-emitting step.
- **Modify 4.16 architecture_review prompt:** add audit clauses for `falsification_signal.hypothesis non-empty`, `every options_considered[i].rejected_because non-empty`, `chosen_option_id resolves to one of options_considered[].option_id`. Reviewer outputs `issues[i].field = "adr.<adr_id>.falsification_signal"` for findings.

#### Cost / latency
- Per mission: ~13 ADRs × (1.5k input + 1.5k output) ≈ 40k tokens added in phase 4. Latency: serial dependency on 4.1→4.2→4.4 path is unchanged but each step is heavier. Expect +30-60% phase 4 wall-clock.

#### Migration
- In-flight missions in phases 0-3: no impact; ADRs land naturally when phase 4 starts.
- In-flight missions in phase 4 with completed 4.1/4.2: backfill ADR shape via mechanical "reshape" step that wraps existing `architecture_pattern_decision` and `tech_stack_decision` into ADR shape with `options_considered: []` and `is_legacy: true` flag. 4.14 register accepts legacy ADRs but `adr_alternatives_min` skips when `is_legacy`.
- Phase 5+: no backfill; mission tagged `legacy_pre_adr: true`; 4.16 review skipped.

#### Prior-art outside KutAI
**Michael Nygard's original ADR format (2011)** — title, status, context, decision, consequences. Our shape is a strict superset adding falsification + reversal-cost + options. **Spotify's Tech Radar + ADR mashup** — ADRs feed quarterly tech-radar review. We don't need radar v1 but the `revisit_trigger` field hints at it. **AWS DACI (Driver/Approver/Contributor/Informed)** — adds owner tracking. Our `decided_by` is a lite version. **Pat Kua's "Lightweight ADR"** — ours is on the heavy end; mitigated by the prototype-tier 2-option allowance.

#### Prior-agent missed
- 4.14 reframe ordering risk: if reviewer at 4.16 runs against new ADR shape but 4.14 still emits old shape, every review fails. v2: ship 4.14 reshape + 4.16 prompt update + ADR-shape upstream in **one atomic change**.
- ADR ID scheme: v1 says `adr_id` but doesn't propose format. v2 proposes `ADR-YYYY-MM-DD-NNN` mission-local (NNN unique per mission). Cross-mission learning later via `founder_profile.prior_missions` join (per OQ3 v1 resolution).
- Cross-ADR dependencies: an auth-design ADR depends on a backend-framework ADR. v1's register has no `links_to`; v2 includes it. 4.16 reviewer should validate the dependency DAG is acyclic.

---

### 4. Failure-mode column (falsifiability)

#### Re-audit
v1 cites `2.5`, `2.6`, `2.7`, `2.9`, `3.1`, `3.2`, `3.7`, ADRs. Verified:
- `2.5 feature_brainstorm` (`:1683-1723`) — `item_fields: [feature_id, feature_name, description, source, category]`. No falsification field. **Correct.**
- `2.7 mvp_scope_definition` (`:1762-1793`) — `required_fields: [mvp_feature_list, excluded_features_rationale, mvp_value_statement]`. No kill criteria per feature. **Correct.**
- `2.9 success_metrics_definition` (`:1832-1881`) — schema fields `name, formula, data_source, target_value, measurement_frequency`. No `kill_threshold`. **Correct.**
- `3.1 functional_requirements_extraction` (`:2002-2037`) — `item_fields: [req_id, title, description, source_story_ids, priority, category]`. No `if_wrong_consequence`. **Correct.**
- `3.2 nfr_performance_and_scalability` (`:2038-2079`) — schema lacks `revisit_trigger`. **Correct.**
- `0.3 assumption_identification` (`:753-785`) — has `risk_if_wrong` + `validation_method`. v1 correctly identifies as the precedent.

**Missed by v1:**
- `1.14 go_no_go_assessment` (`:1474-1509`) score gates — these are pre-decision falsification checkpoints. The score is a kill criterion (`< 4: No-Go`); but per-dimension thresholds (market_attractiveness < 5 etc.) are not articulated as falsification fields. v2 proposes adding `kill_threshold` to each scored dimension.
- `2.10 monetization_strategy` — `pricing_model, tiers, revenue_projections`. The revenue_projection IS a falsifiable claim. No kill criterion. v2 adds.
- `4.12 scalability_and_dr_plan` — RTO/RPO targets. These are falsifiable promises. Need `revisit_trigger`.
- `2.6 feature_prioritization` — MoSCoW priorities are based on assumed user value. No kill criterion; v1 says "adds `kill_criteria` field per top-priority feature" but doesn't specify schema. v2 specifies.

#### Steel-man counter
"Every falsification signal is going to be agent-generated and therefore noise." Counter: `0.3 assumption_identification` already produces these and they're useful at `0.5 human_clarification_request` selection. Second-order effect v1 missed: **kill_threshold drives downstream monitoring**. If `2.9 success_metrics.aarrr_metrics[].kill_threshold = "MAU < 100 by week 8"`, that's an alerting rule for the operations zone (08). The wiring must be explicit so 09-growth and 08-operations *consume* the field, not just write it. v1 mentions "feeds 09-growth monitoring rules" — v2 names the consumer field path.

#### Alternatives (3 shapes)
- **A. v1's shape: per-item field on every commitment-shaped artifact.** Comprehensive. Token cost: medium (~30 extra fields per mission).
- **B. Single `falsification_register` artifact at end of phase 3.** Lighter — one extra step, single artifact, references upstream items by ID. Cost: low. Loses: in-line audit (reviewer at 3.11 must do the join).
- **C. Hybrid: per-item field for high-stakes (ADRs, NFRs, MVP features) + register summary.** Cost: medium. Best of both. **Recommendation.**

#### Concrete schema
```json
// failure_mode (field on feature/requirement/metric)
{
  "hypothesis": "string — what we believe must hold for this to deliver value",
  "signal_if_wrong": "string — observable signal we'd see if hypothesis is false",
  "kill_threshold": "string — quantitative; e.g. 'D7 retention < 15% by week 4'",
  "we_dont_know_yet": false   // explicit TBD; soft signal not bypass
}

// kill_criteria (on top-priority MVP features only)
"kill_criteria": "string — references metric name in success_metrics.aarrr_metrics[]"
```

For metrics in `2.9 success_metrics`: add `kill_threshold` field next to `target_value` per AARRR metric.

For NFRs `3.2`/`3.3`: add `revisit_trigger` field per requirement (e.g. "if p95 > 800ms over 3 days, reopen architecture decision").

#### Wiring
- Schema-only changes propagate via `artifact_schema` JSON in i2p_v3.json; no engine code change.
- **New mechanical action:** `falsification_completeness` — case + new file `packages/mr_roboto/src/mr_roboto/falsification_completeness.py`. Signature: `check(mission_id, step_id, artifact_name, ambition_tier) -> {ok, missing_fields, severity}`. Asserts every item has non-empty `failure_mode.hypothesis` AND non-empty `failure_mode.signal_if_wrong` AND (non-empty `kill_threshold` OR `we_dont_know_yet: true`). For ambition `prototype`, `we_dont_know_yet: true` is acceptable; `private_beta+`, must be empty.
- **Modify reviewer prompts at 3.11 + 4.16:** add audit "every functional_requirement[i].failure_mode.signal_if_wrong is testable; every NFR has revisit_trigger." The reviewer step's `artifact_schema` should NOT change (status:pass remains the gate); the *prompt* changes so that issues are surfaced.
- **Cross-zone consumer contract:** `09-growth.md` reads `success_metrics.aarrr_metrics[].kill_threshold`. `08-operations.md` reads `nfr_*.revisit_trigger` to seed monitoring rules. v2 requires those zone docs gain a `## Inbound from Z1` section.

#### Cost / latency
- Per mission: ~30 extra fields × ~50 tokens = 1.5k extra output tokens; effectively free relative to other proposals.

#### Migration
- In-flight missions: backfill `failure_mode: {we_dont_know_yet: true}` on every commitment item via DB script. Reviewers tagged `legacy_pre_falsification: true` on mission row, prompt accepts `we_dont_know_yet` for all tiers. Subsequent phases get full strictness.

#### Prior-art outside KutAI
**Lean Canvas RAT (Riskiest Assumption Test)** by Ash Maurya: every business model assumption gets a test. Our `failure_mode` is the artifact-level RAT. **Atlassian's "Pre-mortem" technique**: imagine the project failed; what killed it? Each `signal_if_wrong` is a pre-mortem signal. **Eric Ries' Lean Startup "innovation accounting"**: every metric must have a target AND a kill threshold AND a pivot trigger. Our `kill_threshold` + `revisit_trigger` (in ADRs) is the lite version.

#### Prior-agent missed
- The `we_dont_know_yet` escape hatch is a foot-gun: founders will set it on everything. v2 mitigates by capping count: `falsification_completeness` rejects when `>50%` of items have `we_dont_know_yet: true` (warning at prototype, blocker at private_beta+).
- Telemetry hook: every `kill_threshold` reached in production should auto-create a revision task on the mission. v1 misses this. v2 names it as a follow-up to wire in 08-operations.md.
- 0.3-style assumption fields exist but `0.5 human_clarification_request` doesn't surface assumptions where `risk_if_wrong` is HIGH. Update 0.5 prompt to prioritize high-risk assumptions in its top-5 selection.

---

### 5. Web-grounded prior art (closes G5)

#### Re-audit
v1 cites `1.3` (`:973-1013`) — verified: `status` field allows `active/acquired/dead` but no graveyard *narrative*; no failure_mode field; no founded_year cross-checking. **Correct.** v1 cites `1.4` (`:1015-1053`) — confirmed. v1 cites `1.7` (`:1144-1195`) focuses on *living* competitors. **Correct.**

**Missed by v1:**
- `1.14 go_no_go_assessment` should consume `prior_art_report` (v1 says modify 1.14 — correct). But v1 misses that **`competitive_feasibility` already factors competitors**; v1's "if ≥3 dead competitors with same thesis → reduce competitive_feasibility" is a direction, not a formula. v2 specifies the formula: `competitive_feasibility -= 0.5 * min(graveyard_count, 5)` capped at -2.5. Concrete and testable.
- `2.1 product_vision_and_positioning` (`:1511-1547`) — v1's instruction-add is reasonable; v2 specifies the prompt addition: "Append section *Avoiding the graveyard*: for each prior_art_report.key_lessons[i], state how this product's design avoids that failure mode."
- `1.11 regulatory_research` — graveyard narratives often surface regulatory failures (e.g. health-claim apps shut down by FTC). v1 misses cross-feeding `prior_art_report` into 1.11 input. v2 wires.

#### Steel-man counter
"Vecihi's existing tier escalation already handles HN/PH/Crunchbase via `web_search` — why a new tool?" Counter (re-audit of vecihi): `vecihi/fetchers.py` exposes `scrape_url` only; no search. The `web_search` and `smart_search` tools (in `src/tools/`) handle search but generic web search returns SEO-spam, not graveyard narratives. The new tool's value is **structured search across known graveyard sources** (HN Algolia, Wikipedia, Wayback CDX, Crunchbase), each with different APIs. Mitigation of the "yet another tool" objection: the new tool composes existing infra, doesn't reinvent.

Second-order effect v1 missed: **hallucination risk on dead-competitor names**. If the LLM extracts "Soylent for Cats (2014, killed 2017)" from search results, but no such company existed, the spec is poisoned with fiction. v1's mitigation `vecihi.fetch_url(competitor.website_url)` validation is good but incomplete — Wayback CDX should *also* be validated (every claimed dead competitor must have at least one Wayback snapshot before 2024 OR a Crunchbase/HN reference).

#### Alternatives (3 shapes)
- **A. v1's shape: live multi-source `find_prior_art` tool.** Live data, freshest. Cost: 4-6 HTTP calls + parsing per mission. Reliability: depends on rate limits.
- **B. Cached corpus: weekly cron pre-fetches "startup graveyard" indices** (HN graveyard, AngelList shutdowns, IndieHackers post-mortems) into a local table; tool reads from cache with semantic search. Cost: low per-mission. Loses: novelty (cache is N days stale; 1-month-dead startups won't appear).
- **C. Founder-curated graveyard.** Founder pre-supplies "I know about X, Y, Z that died" in the pitch; agent extends. Cost: lowest. Loses: graveyards the founder hasn't heard of (the whole point).

**Recommendation:** A for `private_beta+`, B for `prototype` (cache populated by a 6h cron in `packages/general_beckman/` per existing pattern). Hybrid: A always falls back to B on rate-limit hit.

#### Concrete schema
```json
// prior_art_report (artifact)
{
  "_schema_version": "1",
  "search_summary": {
    "queries_run": ["string"],
    "sources_used": ["hn_algolia", "wikipedia", "wayback", "crunchbase_scrape", "product_hunt"],
    "rate_limit_hits": [],
    "total_results_inspected": 47
  },
  "attempted_solutions": [
    {
      "name": "string",
      "founded_year": 2014,
      "status": "alive" | "acquired" | "dead" | "dormant" | "dead_or_hallucinated",
      "url": "string|null",
      "wayback_first_capture": "iso8601|null",
      "wayback_last_capture": "iso8601|null",
      "traction_signal": "string — funding, users, press at peak",
      "failure_mode": "string|null — required if status=dead",
      "sources": ["url"],   // min_items: 1 if status != alive
      "thesis_summary": "string ≤80 words",
      "evidence_refs": ["agent_inference"] | ["EV-001.C2"]
    }
  ],
  "adjacent_failures": [ /* same shape */ ],
  "key_lessons": [
    {
      "lesson_id": "L-001",
      "lesson": "string ≤80 words",
      "evidence_refs": ["url"],
      "applies_to_us": "string — how we're different / mitigating"
    }
  ],
  "graveyard_count": 4,
  "verdict": "graveyard_well_populated" | "blue_ocean_validated" | "mixed" | "no_data"
}
```

#### Wiring
- **New module:** `packages/vecihi/src/vecihi/prior_art.py` — exposes `find_prior_art(idea_summary: str, domain_keywords: list[str], k: int = 10, ambition_tier: str = "private_beta") -> prior_art_report`. Internally calls in parallel:
  - HN Algolia: `https://hn.algolia.com/api/v1/search?query=<keywords>&tags=story` (free, no key, 1 req/s polite)
  - Wikipedia: MediaWiki API `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=<keywords>` (free)
  - Wayback CDX: `https://web.archive.org/cdx/search/cdx?url=<candidate>&output=json` (free; per candidate)
  - Product Hunt: existing scrape path via `scrape_url` (uses Vecihi's tier escalation)
  - Crunchbase: only via Brave/GCSE fallback (ToS-grey per v1's risk note)
  Each source's results aggregated; deduped by name (fuzzy match); status classified via heuristics: `last_news_mention < 2022 AND domain doesn't resolve → dead`; `acquired_by present → acquired`; etc. Adds `dead_or_hallucinated` status when no Wayback snapshot AND no HN reference exists.
- **New tool spec for LLM:** `find_prior_art` tool wrapping the above; agent at step `1.0 prior_art_search` calls it.
- **New step `1.0 prior_art_search`:** agent `researcher`, tools_hint `[find_prior_art, web_search, extract_url]`, depends_on `[0.6]` (parallel with 1.1, 1.2, 1.3). produces `prior_art_report.json` (declares for grounding). `produces: ["missions/<id>/.research/prior_art_report.json"]`.
- **New mechanical action:** `prior_art_min_coverage` — case + new file `packages/mr_roboto/src/mr_roboto/prior_art_min_coverage.py`. Asserts `len(attempted_solutions) >= 3` OR `verdict == "blue_ocean_validated"` AND `len(search_summary.queries_run) >= 3`. Empty results → fail.
- **New mechanical action:** `prior_art_url_resolves` — for each `attempted_solutions[i].url`, fetch via `vecihi.scrape_url` HEAD (or GET if HEAD fails); if status is `dead` AND no Wayback hit AND no HN reference → mark `status: dead_or_hallucinated` AND set `severity: warning`. Hooked at end of `1.0`.
- **Modify `1.14 go_no_go_assessment`:** input_artifacts += `prior_art_report`; instruction adds: "factor `graveyard_count` and `key_lessons` into `competitive_feasibility` score per formula `score -= 0.5 * min(graveyard_count, 5)` capped at -2.5. Cite at least one lesson from `key_lessons` in reasoning."
- **Modify `2.1 product_vision_and_positioning` instruction:** append "*Avoiding the graveyard*: for each `prior_art_report.key_lessons[i]`, state how this product's design avoids that failure mode."

#### Cost / latency
- Per mission: 4-6 HTTP API calls + 5-15 Wayback CDX calls + 5-10 url-resolve checks. Total HTTP: ~20-30 requests, ~30-90s wall-clock. Tokens: ~5k input + 3k output for the LLM step.
- Prototype tier: lite version (HN + Wikipedia only, no Wayback per-candidate validation). Token cost halved.

#### Migration
- In-flight missions in phases 0-1: skip; mission tagged `legacy_pre_prior_art: true`. Phase 2+: read prior_art_report as empty if absent; reviewer accepts.

#### Prior-art outside KutAI
**Failory's startup post-mortem index**, **Crunchbase's "Why startups fail"**, **HN's recurring "Show HN: my failed startup"**, **CB Insights "Top reasons startups fail"** — all are graveyard corpora. **Y Combinator's "Library" Talks** mentions "before YC, every founder pasted 5 dead competitors into their app" — that's prior-art-as-pre-flight in the wild. **"Premortem" technique** (Gary Klein, 1989) — written failure mode hypothesis. We're systematizing it.

#### Prior-agent missed
- Hallucination guard via Wayback wasn't in v1; v2 makes it a mechanical post-hook.
- Crunchbase ToS risk noted by v1 but no ToS-safe alternative path. v2: degrade to GCSE fallback only when HN/Wiki/Wayback yielded <3 results.
- The `verdict: "blue_ocean_validated"` is a hallucination magnet — agents will pick it to skip the gate. v2: `prior_art_min_coverage` requires `search_summary.queries_run >= 3` AND `total_results_inspected >= 20` to honor `blue_ocean_validated`.
- Cross-mission learning: if mission N found "Mealime acquired", mission N+1 in food-tech should inherit that fact. v1 misses; v2 says: prior-art findings cached in a `prior_art_cache` table keyed on `domain_keywords` hash; subsequent missions in same domain pre-populate from cache, then refresh.

---

### 6. Compliance fingerprint (closes G6)

#### Re-audit
v1 cites `0.4` (`:787-822`), `1.11` (`:1317-1352`). Verified.
- `0.4 scope_ambiguity_detection` categories: Audience/Scope/Business Model/Constraints/Platform/Differentiation/Priority. **No jurisdictions/data-categories/user-class.** v1 correct.
- `1.11 regulatory_research` is unscoped: "Identify legal, regulatory, and compliance requirements for this product." It's a "fish for issues" prompt. v1 correct.

**Missed by v1:**
- `2.1 product_vision_and_positioning` doesn't cite compliance constraints in vision. If an idea is compliance-fatal (e.g., "AI medical advice for under-13s in EU"), the vision should reflect or pivot. v1 misses.
- `3.3 nfr_availability_and_security` (`:2080-2120`) maps `regulatory_requirements` to "specific technical controls". This is the natural consumer of compliance fingerprint — v1 mentions but doesn't specify the wiring.
- `3.4 data_requirements` (`:2122-2170`) has `sensitivity_level` and `retention_policy` per entity. These should derive from `compliance_fingerprint.data_categories`. v1 misses.

#### Steel-man counter
"This is legal advice without lawyers." Counter: v1's `compliance_overlay` carries explicit "this is not legal advice" watermark + 180-day staleness warning. Second-order effect v1 missed: **founders will treat the auto-generated privacy policy as authoritative**. The 06-real-world-bridge.md zone is named as the executor but the founder must sign off. v2 names a hard gate: privacy policy doc carries `requires_legal_review: true` flag; mission cannot ship to public_launch tier without `founder_attests_legal_reviewed: <iso8601>` in the mission row.

#### Alternatives (3 shapes)
- **A. v1's shape: 0.4a wizard + 1.11a overlay step + cross-phase blocker.** Comprehensive. Highest founder friction.
- **B. Inline compliance question in 0.4 scope_ambiguity_detection.** Add `Compliance` category to the existing 7-category list; questions surface in 0.5 clarification. Cost: low. Loses: structured fingerprint downstream consumers can't query.
- **C. Z0-only: collect everything at preflight; Z1 just consumes.** Cost: lowest at Z1. Requires z0 to fully refine — contradicts z0's "high-level only" framing. **Reject.**

**Recommendation:** A. B is acceptable for prototype tier.

#### Concrete schema
```json
// compliance_fingerprint (artifact)
{
  "_schema_version": "1",
  "source": "z0_preflight" | "z1_intake" | "z0_then_z1_delta",
  "collected_at": "iso8601",
  // From z0 (coarse)
  "jurisdictions": ["EU", "US", "UK", "CA", "TR", "BR", "JP", ...],
  "user_classes": ["consumer", "b2b", "health", "children", "financial", "government", "education"],
  "data_categories_coarse": ["pii", "payment", "health", "biometric", "location", "minors_data", "sensitive_communications"],
  // Z1-only refined
  "data_residency_required": true,
  "age_gate_required": true,
  "third_party_processors_expected": ["stripe", "sendgrid", "vercel", ...],
  "data_export_requirements": ["gdpr_dsar", "ccpa_request"],
  "retention_max_days": 365,
  "founder_attestations": {
    "founder_will_sign_dpa_with_processors": true,
    "founder_acknowledges_not_legal_advice": true
  }
}

// compliance_overlay (artifact)
{
  "_schema_version": "1",
  "required_documents": [
    {
      "doc_type": "privacy_policy" | "cookie_banner" | "dpa" | "tos" | "retention_policy" | "age_gate" | "accessibility_statement" | "data_processing_record",
      "applicable_jurisdictions": ["EU"],
      "template_id": "gdpr_privacy_v3",
      "template_version": "3.1",
      "last_reviewed": "2026-02-15",
      "generated_template_path": "missions/<id>/legal/privacy_policy.md",
      "founder_review_required": true,
      "blocker_for_phase": 13,
      "requires_legal_review": true
    }
  ],
  "monitoring_obligations": [
    {"obligation": "Cookie consent log retention 1y", "owner": "agent" | "founder", "automation_status": "automated"|"manual"}
  ],
  "data_subject_rights_implementation": {
    "access": "automated_via_user_dashboard" | "manual_email",
    "deletion": "automated_via_admin_panel",
    "portability": "automated_export_json",
    "rectification": "user_dashboard"
  }
}
```

#### Wiring
- **New step `0.4a compliance_fingerprint_collection`:** agent `mechanical`, executor: `compliance_fingerprint_wizard`. depends_on `[0.4]`. produces `missions/<id>/.compliance/fingerprint.json`. `triggers_clarification: true`. Skipped if z0 supplied refined fields (z0 status check via context).
- **New mechanical action:** `compliance_fingerprint_wizard` — case + new file `packages/mr_roboto/src/mr_roboto/compliance_fingerprint_wizard.py`. Reads z0 founder_profile + z0 mission_preflight; computes deltas; sends Telegram form for Z1-only refined fields; "I don't know" allowed; agent attempts inference for unanswered fields from `idea_brief_final` after the fact.
- **Modify `0.6 idea_brief_compilation`:** input_artifacts += `compliance_fingerprint`; new section `Compliance Footprint`.
- **Modify `1.11 regulatory_research`:** input_artifacts += `compliance_fingerprint`; instruction prepends: "Use `compliance_fingerprint` to scope research. Only research regulations applicable per `jurisdictions × user_classes × data_categories_coarse`." Cuts 1.11 token cost ~40%.
- **New step `1.11a compliance_overlay`:** agent `analyst`, depends_on `[1.11]`. produces `missions/<id>/.compliance/overlay.json` AND each generated template doc. Calls a new tool `compliance_template_render(fingerprint, doc_type) -> str`.
- **New tool:** `compliance_template_render` — lands at `packages/mr_roboto/src/mr_roboto/compliance_template_render.py`. Reads from `data/compliance_templates/<jurisdiction>/<doc_type>.md.j2` (Jinja2-style); fills placeholders from fingerprint. Refuses if template `last_reviewed_date < today - 180d`.
- **New mechanical action:** `compliance_template_present` — asserts every `required_documents[i].generated_template_path` resolves to a file under workspace.
- **New mechanical action:** `compliance_blocker_check` — runs at every phase boundary ≥7. Reads `compliance_overlay.required_documents[]`. If any doc's `blocker_for_phase <= current_phase` AND `founder_review_required == true` AND no `founder_signoff` in mission row → block.
- **Modify `3.3 nfr_availability_and_security`:** input_artifacts += `compliance_fingerprint`; instruction adds "Map every `data_categories_coarse[i]` to specific authentication+authorization+encryption controls."
- **Modify `3.4 data_requirements`:** instruction adds "Each entity's `sensitivity_level` and `retention_policy` MUST be consistent with `compliance_fingerprint.data_categories_coarse` and `retention_max_days`."

#### Cost / latency
- Per mission: +1 founder Telegram form (variable wait), +1 LLM step at 1.11a (~5k tokens), +N template renders (mechanical, fast). 1.11 token cost reduced ~40% by scoping. Net ~+3k tokens.

#### Migration
- In-flight missions in phases 0-1: prompt founder for fingerprint at next clarify gate. Phase 2+: tag `legacy_pre_compliance: true`; downstream consumers fall back to `1.11 regulatory_requirements` only; cross-phase blocker disabled.

#### Prior-art outside KutAI
**Stripe Atlas / Clerky compliance checklists** — pre-flight legal infra at company formation. **Termly / iubenda privacy-policy-as-a-service** — fingerprint-driven template generation is exactly their model. **NIST Privacy Framework** — `data_subject_rights_implementation` mirrors NIST's "Communicate" function. **GDPR Article 30 (Records of Processing)** — our `monitoring_obligations` is a lite ROPA.

#### Prior-agent missed
- z0/Z1 contradiction noted in Diff #6 above.
- Template currency: 180-day rule named in OQ4 but v1 doesn't say what the template-render tool does when stale. v2: refuses with explicit `template_stale: true` error; founder must run `/compliance refresh` (handled by 06).
- Bilingual templates: TR market means TR-localized templates needed; v1 silent. v2: template path `data/compliance_templates/<jurisdiction>/<lang>/<doc_type>.md.j2` with `<lang>` defaulting to founder_profile.locale.
- Cross-jurisdiction conflicts: GDPR + COPPA + China PIPL can demand contradictory things. v1 silent. v2: `compliance_overlay` adds `cross_jurisdiction_conflicts: [{conflict, resolution}]` field; reviewer at 1.11a flags.

---

## New proposals not in v1

### 7. Spec versioning + reviewer prompt regression suite

**Why this matters:** Five of the six proposals above add fields to existing artifact schemas. Reviewer prompts at 1.13/3.11/4.16/5.10/6.6 are LLM-driven and silently regress when inputs change. Today there's no test that "given this input, the reviewer flags this issue."

**Proposal:**
- Every artifact this plan touches gains `_schema_version: "1"` field.
- Schema bumps require migration script + reviewer prompt update + a regression fixture.
- New test directory `tests/reviewers/fixtures/<step_id>/` with files: `input_<scenario>.json`, `expected_issues.json`. CI loads each fixture, runs the reviewer once, asserts issue list matches expected (string contains, not exact). Bounded fixture set (3-5 per reviewer).
- Lands in `tests/reviewers/test_reviewer_regressions.py`.

**Cost:** 5 reviewers × 4 fixtures = 20 LLM calls per CI run (~5k tokens each). Run on PR only, not every commit.

**Migration:** Existing missions tagged `_schema_version: "0"`; reviewers gated on version (legacy missions use legacy prompt; new missions use updated prompt). Sunset legacy prompt 90 days after launch.

### 8. Founder-evidence ↔ agent-inference conflict surface

**Why this matters:** OQ1 v1-resolution says "evidence wins; agent flags contradiction; founder confirms." But v1 doesn't specify *where* the conflict surfaces. If the agent flags a conflict in 0.1 but the founder doesn't see it until 0.5 clarification (5 steps later), context is lost.

**Proposal:**
- New artifact `pitch_evidence_conflicts` produced by 0.0b (extraction step).
- Schema: `[{conflict_id, pitch_claim_quote, evidence_claim_quote, evidence_id, severity: high|medium|low, agent_resolution_proposal: "evidence_wins"|"pitch_wins"|"both_valid_in_context"}]`.
- 0.5 human_clarification_request reads `pitch_evidence_conflicts` and surfaces the top-3 high-severity conflicts as part of its 5-question budget.
- Founder responds: `accept_evidence` / `accept_pitch` / `merge` per conflict.
- Resolution recorded in `clarification_answers` with audit trail.

**Cost:** Folded into proposal 1 (intake); ~+500 tokens.

### 9. Cross-mission compliance & ADR inheritance

**Why this matters:** OQ3 v1 resolution says "cross-mission learning compounds: mission-N can read previous-mission ADRs in same domain via `founder_profile.prior_missions` join." Same applies to compliance fingerprint. v1 doesn't specify *how*.

**Proposal:**
- DB table `mission_artifacts_index(mission_id, artifact_name, artifact_path, schema_version, domain_keywords[], created_at, founder_id)`.
- At mission start (z0), if `prior_missions > 0`, agent reads top-3 most-recent ADRs and compliance_fingerprint from same `founder_id` and same domain_keywords overlap.
- Used as **hints**, not defaults. Surfaced in 0.5 clarification: "Your last mission used PostgreSQL for primary store; reuse?"
- Also feeds proposal 5 (prior_art_cache) — mission-local cache of dead competitors becomes founder-local cache.

**Cost:** DB schema migration; ~+500 tokens at mission start.

---

## Sequencing recommendation

**Land order with rationale.** Each step's success unblocks the next.

```
0. spec versioning + reviewer regression fixture infra (proposal 7)
   ↓ (every later proposal adds fields; need versioning + tests first)

1. Failure-mode column (proposal 4)
   ↓ (smallest schema-only change; trains the team on
      "schema bump → reviewer prompt update → regression fixture" loop;
      0.3 precedent reduces risk; no new mechanical actions)

2. ADRs as first-class (proposal 3)
   ↓ (extends the falsification fields into options_considered;
      4.14 reframe is contained; reviewer prompt update is one step)

3. Compliance fingerprint (proposal 6)
   ↓ (independent of ADRs but needs versioning; new mechanical actions
      land here for first time)

4. Web-grounded prior art (proposal 5)
   ↓ (independent of compliance; new vecihi module; one new step;
      heaviest tool work but contained)

5. Structured user-research intake (proposal 1) + conflict surface (proposal 8)
   ↓ (highest founder-UX impact; rip up phase 0 ordering;
      depends on z0's evidence-bucket provisioning being live;
      land 1+8 atomically)

6. Taste delegation (proposal 2)
   ↓ (depends on intake's evidence_refs propagation through 2.4 and downstream;
      brand_direction_options[].rationale references personas which now have
      evidence_refs)

7. Cross-mission inheritance (proposal 9)
   ↓ (best last; depends on prior missions having all of the above shapes)
```

**Critical path:** 0 → 1 → 2 → 3 → 6.
**Parallel:** 4 can run alongside 1-3.
**Blocking on z0:** 5, 6, 7 require z0 Phase B (preflight wizard) shipped.

---

## Risks register

Cross-proposal risks (single-proposal risks are inside each section).

| # | Risk | Affects | Mitigation |
|---|---|---|---|
| R1 | **Token-cost compounding.** Proposals 1+3+5 each add ~30k tokens per mission. Cumulative ~100k extra; phase 4 alone could double. | All | Ambition-tier gating on every proposal; prototype tier strips alternatives + intake + prior-art-deep. |
| R2 | **Reviewer prompt regressions.** Six proposals touch reviewer audits. Without proposal 7 first, every landing breaks reviewer pass-rate silently. | All | Proposal 7 (regression suite) lands first. Block subsequent merges on regression-fixture pass. |
| R3 | **In-flight mission breakage.** Schema changes break mid-flight missions. | All | Per-proposal migration plan; legacy-tagged missions skip new gates; sunset 90 days after launch. |
| R4 | **Engine surface mismatch.** v1 names hooks the engine can't dispatch. | Proposals 1, 2, 3, 5, 6 | v2 specifies new mechanical action for every new hook; lands as new file under `packages/mr_roboto/src/mr_roboto/`. |
| R5 | **Vecihi tier escalation cost on prior_art.** Browser tier on each Wayback CDX hit could 10x latency. | Proposal 5 | HTTP tier only for Wayback/HN/Wiki APIs (they support it); browser tier reserved for Crunchbase fallback. |
| R6 | **Concurrent agents-overhaul churn.** `gitStatus` shows M src/agents/{implementer,planner,test_generator}.py + docs/plans/2026-05-08-agents-overhaul. Z1 modifies `agent: analyst/researcher/architect` step assignments; if agents-overhaul changes those agent names, every i2p_v3.json edit breaks. | All Z1 work | Read agents-overhaul plan before sequencing; coordinate landing windows; fail-fast test that loads i2p_v3.json against agents registry. |
| R7 | **Founder fatigue.** Proposals 1+2+6 add founder Telegram interactions: evidence intake (1), brand pick (1), compliance fingerprint (1). For prototype-tier missions this is too much. | Proposals 1, 2, 6 | Strict ambition-tier gating; prototype skips intake (uses pitch text only), gets template brand library (proposal 2 alt B), uses 0.4 inline compliance category (proposal 6 alt B). |
| R8 | **Hallucinated prior-art is worse than no prior-art.** | Proposal 5 | `prior_art_url_resolves` mechanical hook + Wayback validation; `dead_or_hallucinated` status routed back to founder. |
| R9 | **Compliance template staleness as a legal risk.** Auto-generated docs cited as authoritative. | Proposal 6 | Hard 180-day refusal in `compliance_template_render`; explicit "not legal advice" watermark; `requires_legal_review: true` blocker for `public_launch` tier. |
| R10 | **z0 contract drift.** v2 assumes z0 ships specific fields (founder_profile.brand_voice_samples, mission_preflight.compliance_fingerprint coarse). If z0 ships different shapes, every Z1 proposal that consumes z0 needs re-spec. | Proposals 1, 2, 5, 6, 9 | Block landing of Z1 proposals until z0 Phases A+B are merged AND z0 surface contract is locked. |
| R11 | **Schema-version forking.** Proposal 7 introduces `_schema_version: "1"`. If two proposals land in parallel both bumping to "2" with different shapes, missions corrupt. | All | Single-merge-train for Z1 proposals; serialized landing per Sequencing. |

---

## Top 3 cross-zone surprises (carried from v1, refined)

1. **`4.14 architecture_decisions` already emits ADRs post-hoc.** v1's surprise is correct. v2 adds: the `min_items: 3` is on count of ADRs, not alternatives-per-ADR; the schema fields are exactly Nygard format; the upstream-shape change is the actual refactor. *4.14 stays — becomes register-only.*
2. **`0.3 assumption_identification` already has falsifiability fields.** v1's surprise is correct. v2 adds: 0.5 clarification step doesn't currently prioritize high-`risk_if_wrong` assumptions; that's a free win.
3. **z0 supplies *coarse* compliance fingerprint, not refined.** v1's surprise framed z0 as "redundant" — v2 reframes: z0 is fast preflight, Z1 is deep collection. The integration is delta wizard, not skip-if-supplied.

**v2-original cross-zone surprise:** **`expander.py:178-181` auto-prepends `grounding` to every step with `produces`.** This means any proposal declaring `produces` gets free L2 grounding for free — but **also** means a proposal declaring an artifact path under `missions/<id>/.evidence/` will trigger `mr_roboto.check_grounding`, which inspects tool_calls audit log not file contents. Several v1 hook proposals were misaligned with this; v2 corrected by routing artifact-content checks through new `*_check` actions and reserving `grounding` for tool-call alignment.

---

## Updates

(none — this is v2's initial cut)
