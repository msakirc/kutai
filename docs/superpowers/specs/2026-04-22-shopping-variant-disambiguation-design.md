# Shopping Variant Disambiguation — Design Spec

**Date**: 2026-04-22
**Owner**: Şakir / KutAI
**Status**: Proposed
**Predecessor commits**: `3f08e85` (review scraping), today's `<hash>` (wiring fix), today's `<hash>` (JSON unwrap + query-match penalty)
**Companion spec (deferred)**: quantitative review intelligence — scheduled for a follow-up design.

## Problem

Today, when a user types an ambiguous product query (`Samsung s25`, `iphone 15`, `dyson v15`), pipeline_v2 returns one product card — frequently the wrong variant. The live example that motivates this spec: query `Samsung s25` returned `Samsung Galaxy S25 FE`, surpassing the actual S25 model. Cause: `select_groups` ranks purely by cross-site prominence, so the variant with the broadest distribution wins regardless of query fit.

Static title-matching penalties (today's shipped fix) are brittle — they help phones marginally but do nothing for:
- replacement parts pretending to be products (`iPhone 15 screen`)
- accessories flooding the result set (cases, chargers)
- knock-offs that share every query token with the authentic listing
- refurbished / grade-B listings mixed in
- any non-phone category (appliances, tools, apparel)

The system needs to *understand* what each scraped listing is, not just measure its title against the query.

## Goals

- **G1** Distinct SKUs never merge. Samsung S25 and S25 FE are two groups, never one.
- **G2** When a query matches >1 authentic variant after filtering, the bot asks the user to pick, with a *Compare all* escape hatch.
- **G3** Accessories, replacement parts, knock-offs, and refurbished listings are dropped before variant counting.
- **G4** When only one variant survives, no clarify step — fast path preserved.
- **G5** Works category-agnostically. No phone-specific logic.

## Non-goals

- Quantitative review intelligence (deferred to a follow-up spec).
- Cross-variant price comparison beyond the flat *Compare all* table.
- Recommendation logic ("Ultra is better for photos").
- Stock notification / price-drop monitoring.
- Teaching the bot what's a "variant" vs a "distinct product" beyond what the grouping LLM + category signal can infer.

## Architecture

```
user query
  └─ _fetch_products  (unchanged)
      └─ Product now carries: sku, category_path  ← schema extension
          └─ step_resolve  (unchanged flow, passes sku/category through)
              └─ step_group
                   │   (1) deterministic bucket by sku across sites
                   │   (2) LLM grouping on sku-less residuals
                   └─ step_label  ← new; one LLM call labels every group
                        └─ step_filter  ← new; pure code, drops
                             │  non-authentic-product, intent-mismatched,
                             │  low-authenticity groups
                             └─ step_variant_gate  ← new; pure code
                                  │  one survivor  → synthesize + format
                                  │  >1 survivors → emit NeedsClarification
                                  └─ clarify (salako mechanical executor)
                                       │  Telegram inline keyboard:
                                       │     [variant₁] … [variantₙ] [Compare all]
                                       │  user tap → _pending_action resume
                                       │     ├─ variant chosen: resume synthesis
                                       │     └─ Compare all: step_compare_all
                                       │        emits compact table card
                                       └─ deliver response
```

## Data schema

### `Product` (`src/shopping/models.py`)

Add:
- `sku: str | None = None` — site-native product identifier (e.g. Hepsiburada `HBCV00004X9ZCH`, Trendyol content-id).
- `category_path` — already exists; must be populated consistently.

### `Candidate` (`src/workflows/shopping/pipeline_v2.py`)

Add:
- `sku: str | None = None`
- `category_path: str | None = None`

### `ProductGroup`

Add labelled fields, all populated by `step_label`:
- `product_type: str` — one of `authentic_product | accessory | replacement_part | knockoff | refurbished | unknown`.
- `base_model: str` — free-form, extracted by LLM (e.g. `Samsung Galaxy S25`).
- `variant: str | None` — free-form variant suffix (e.g. `FE`, `Plus`, `Ultra`, or `None` for base models).
- `authenticity_confidence: float` — 0.0–1.0.
- `matches_user_intent: bool` — LLM's judgment: does this group answer the user's query?

`is_accessory_or_part` kept for backward compat but superseded by `product_type`.

## Pipeline steps

### `step_group` (rewrite)

1. Deterministic bucket: products sharing a non-null `sku` collapse into one group regardless of site. This is the authoritative signal — same SKU across Trendyol and Hepsiburada is definitively the same product.
2. Remaining products (sku `None` or unique): fall through to existing LLM grouping prompt, unchanged. Prompt is extended to receive `sku` and `category_path` alongside title as hints.

### `step_label` (new)

One batched LLM call. Input: query, list of groups with their representative title, site, category_path, member count. Output schema:

```json
{
  "groups": [
    {
      "group_id": 0,
      "product_type": "authentic_product",
      "base_model": "Samsung Galaxy S25",
      "variant": null,
      "authenticity_confidence": 0.95,
      "matches_user_intent": true
    }, ...
  ]
}
```

Fallback on LLM/parse error: treat every non-accessory-flagged group as `authentic_product`, `matches_user_intent=true`, `authenticity_confidence=0.5`, `variant=null`. Degraded but functional.

### `step_filter` (new, pure code)

Drops groups where any of:
- `product_type != "authentic_product"`
- `matches_user_intent == false`
- `authenticity_confidence < 0.7`

Threshold `0.7` is tunable via `FILTER_AUTHENTICITY_MIN` module constant.

### `step_variant_gate` (new, pure code)

Given filtered groups:
- Zero survivors: emits `{escalation_needed: true, reason: "all_filtered"}`.
- One distinct `(base_model, variant)`: pick highest-prominence group in that bucket, proceed to synthesis.
- Multiple distinct variants: emit `NeedsClarification` sentinel with:
  - `variant_options: [{label, group_id, price_min, price_max}]` — sorted by prominence desc, cap at 5.
  - `group_payloads: {group_id: ProductGroup}` — so resume path doesn't re-scrape.

### `step_compare_all` (new)

Emits a compact markdown table:

```
*Samsung Galaxy S25 — Karşılaştırma*
────────────────────
• *Vanilla* — 32.500–34.800 TL ⭐ 4.7 (1.2k)
• *Plus* — 38.900–41.500 TL ⭐ 4.8 (890)
• *Ultra* — 48.000–54.000 TL ⭐ 4.9 (2.1k)
• *FE* — 19.500–22.000 TL ⭐ 4.5 (640)
────────────────────
Seçmek için sorunuzu daraltın (örn. "samsung s25 ultra").
```

Price range = min/max across group members. Rating = best-populated member. Review count = sum across members.

## Workflow JSON

`src/workflows/shopping/shopping_v2.json` (and `quick_search_v2`, `product_research_v2`) gains a conditional branch:

```
phase_1:
  resolve_candidates
  group_label_filter_gate    ← merged step; handler runs step_group + step_label + step_filter + step_variant_gate
phase_2 (conditional):
  synthesize_reviews         ← when variant gate selected single group
  OR clarify                 ← when variant gate emitted NeedsClarification
phase_3:
  format_response            ← after synthesize
  OR format_compare_all      ← after user picks Compare all
```

The `clarify` step is `agent: "mechanical"`, executor `clarify`, payload is the variant gate output.

## Telegram integration

- `_pending_action[chat_id] = {kind: "variant_choice", mission_id: N, options: [...], payloads: {...}}`
- Inline keyboard per variant button callbacks to a new handler `_handle_variant_choice`:
  - On specific variant: looks up group payload, writes it as a new artifact, resumes the mission at `synthesize_reviews`.
  - On "Compare all": runs `step_compare_all` on all stored payloads and replies with the markdown table.
- Idempotency: callback checks `_pending_action` still matches `mission_id`; duplicate taps are swallowed silently with a toast.
- Timeout: if no tap within the mission timeout (default 30min), mission moves to DLQ per CLAUDE.md policy.

## Scraper audit

All 15+ scrapers under `src/shopping/scrapers/` must populate `Product.sku` and `Product.category_path` where the source provides them. Scope:

- **Phase 1 — quick wins** (SKU already parsed internally, just expose): `trendyol`, `hepsiburada`, `amazon_tr`, `akakce`, `epey`, `kitapyurdu`, `dr_com_tr`, `decathlon_tr`, `home_improvement` (koçtaş + ikea), `grocery` (migros + getir).
- **Phase 2 — add extraction**: `sahibinden`, `arabam`, `direnc_net`, `google_cse`, `forums` (technopat, donanimhaber).
- **Phase 3 — leave as-is**: `eksisozluk`, `sikayetvar` (community, no real SKU).

Each scraper change carries its own unit test with fixture HTML or recorded API response.

Scrapers without SKU remain fully functional — they flow into the LLM grouping path.

## Testing strategy

**Unit**
- `step_group` SKU bucketing — mix of sku-equal, sku-partial-null, title-only.
- `step_label` parsing — well-formed JSON, malformed JSON, missing keys.
- `step_filter` — one test per taxonomy value.
- `step_variant_gate` — 0/1/N survivors.
- `step_compare_all` — table format golden.

**Scraper tests** — per-scraper fixture assertions on `sku` + `category_path`. 15+ new tests.

**Integration** — mocked `_fetch_products` returning:
1. Phone ambiguous (S25 + S25 FE + S25 Ultra authentic) — assert clarify triggered with 3 options.
2. Phone single (one variant) — assert no clarify, direct synthesis.
3. Accessory dominance (case + charger + one real phone) — assert accessories filtered, single-variant fast path.
4. Knockoff mix — assert low-confidence knockoff dropped.
5. All-filtered — assert `escalation_needed` path.
6. Cross-site SKU dedup — assert 3 products from 3 sites with same SKU become one group.

**Live smoke** — new `tests/shopping/verify_variant_flow_live.py`: real query `Samsung s25`, assert clarify payload contains `S25` and `S25 FE` as distinct options.

## Migration / rollout

- Product and Candidate schema additions are additive with defaults — no breaking changes.
- `step_group` rewrite is a new function; old function deprecated after tests pass.
- Workflow JSON changes: versioned (`shopping_v2` → `shopping_v2.1` or just in-place update; fine to update in-place since no persisted mission data relies on the old step graph at rest).
- Scraper changes: ship in batches of 5 per PR, each batch independently landable.

## Rollback

- Pipeline-side: revert `pipeline_v2.py` + workflow JSON commit. No data migration needed; Product schema extensions stay harmlessly.
- Scraper-side: per-scraper revert — each scraper commit is independent.

## Success criteria

- Query `Samsung s25` surfaces a clarify step with ≥3 variant options and a *Compare all* button (live smoke).
- Query `Samsung Galaxy S25 Ultra 256GB` skips clarify and returns a single card (live smoke).
- Accessory-heavy queries (e.g. `iphone 15 case`) do **not** clarify (single `accessory` group survives intent filter, or 0 groups if we decide cases are not the goal).
- No regression: existing 23 `test_pipeline_v2.py` tests still pass; full targeted suite green.

## Open questions (to resolve during plan writing)

- Threshold `authenticity_confidence < 0.7` picked intuitively; validate with a small labelled set after step_label lands.
- `step_label` and `step_group` LLM calls — keep as separate calls for isolation, or fuse into one? Start separate for testability; fuse later if latency dictates.
- "Compare all" — show when 2 variants, or only when ≥3? Probably ≥2 (user already faced the clarify); flag in implementation.
