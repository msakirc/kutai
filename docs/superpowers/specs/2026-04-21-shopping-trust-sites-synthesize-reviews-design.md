# Shopping Pipeline v2 — Trust Sites, Synthesize Reviews

**Date:** 2026-04-21
**Status:** Design approved, implementation pending
**Supersedes (partially):** `docs/superpowers/plans/2026-04-16-shopping-workflows-refactor.md` (matcher/filter portions)

## Context

The 2026-04-16 shopping refactor built heuristic product matching (`product_matcher.py`: EAN → MPN → fuzzy name → spec fingerprint) and a token-overlap relevance filter (`_filter_relevant`) to group cross-site search results into a single "winner + other prices" view.

On 2026-04-20 a live test ("Siemens s100") exposed a structural failure: the matcher grouped a Siemens EQ.3 **brewing-unit replacement part** (4,800 TL on Amazon) with real Siemens **EQ6 Plus S100 machines** (24-25k TL on Hepsiburada/Akakçe) under one winner. The "Diğer Fiyatlar" block misleadingly cited prices from a different product line. Without EAN/MPN populated, fuzzy name matching on "Siemens" + "S100" cannot distinguish product family, SKU class, or accessory vs main product.

**Design thesis:** E-commerce sites (Trendyol, Hepsiburada, Amazon.tr, Akakçe) have spent 15+ years solving query → product relevance. Their top-ranked result for a query is authoritative. We cannot outbuild their teams with token overlap or spec fingerprints. Our unique value is not re-ranking products — it is **aggregating and synthesizing reviews** across sites and community sources, which no single shopping site does.

A prior user message (session `467ff150`, 2026-04-15) voiced the same intuition: *"Why don't we trust the sites for sorting? I am sure trendyol would suggest the correct machine if I would have searched directly there."* The pivot was not executed at the time.

## Decisions

| # | Decision |
|---|---|
| 1 | Drop `product_matcher.py` and `_filter_relevant` from primary search ranking. Trust each site's own top-N ordering. |
| 2 | Use an LLM call to group cross-site results into product groups — replaces heuristic matcher. |
| 3 | Review aggregation + synthesis becomes the core value. Synthesis is a separate LLM call per group. |
| 4 | Scope covers all three shopping workflows: `product_research`, `quick_search`, `shopping` (category). Category differs only by breadth + retained clarifier step. |
| 5 | Output format: reviews-first Telegram card (praise / complaints / red flags), prices below. |
| 6 | Rollout: parallel `_v2.json` workflow files. `wf_map` in `telegram_bot.py` switches to v2. No feature flag — same playbook as i2p → i2p_v3. |
| 7 | Both LLM calls use `main_work` tier. Grouping is marked low difficulty; synthesis is higher difficulty. Fatih Hoca picks tier. |
| 8 | Scraper depth (extracting review text from product detail pages) is explicitly **out of scope**. Synthesis works with whatever snippet data scrapers already return. Where data is thin, output shows an "insufficient review data" note. |

## Architecture

One shared primitive — `resolve → group → synthesize → format` — used by all three v2 workflows. They differ only in breadth (per-site N) and whether a clarifier step precedes resolution.

### Step-by-step (named-product, e.g. `/shop Siemens EQ6 S100`)

```
user_query
  → step_resolve        (shopping_search tool, top-3 per site across ~5 sites)
  → step_group          (LLM call, main_work low difficulty)
  → step_synthesize     (LLM call per kept group, main_work higher difficulty)
  → format_group_card × N
  → Telegram message
```

### Step-by-step (category, e.g. `/research_product` → "🏷 Kategori" → "kahve makinesi 5000 TL altı")

```
user_query
  → step_clarify        (existing v1 clarifier, retained unchanged)
  → step_resolve        (top-8 per site, ~40 candidates)
  → step_group          (LLM → keep top 2-3 groups by position aggregation)
  → step_synthesize × 2-3
  → format_response     (2-3 cards)
```

### Ranking inside groups

No composite value score, no weighted heuristic. A group's prominence is `sum(1 / site_rank_position)` across members. A product that sits at #1 on Trendyol AND #1 on Hepsiburada outranks one at #3 on a single site. Site ordering is the ground truth.

### How many groups render

- Named-product workflows (`product_research_v2`, `quick_search_v2`): render up to **2** non-accessory groups, ordered by prominence. Second group only renders if its prominence ≥ 50% of the top group — otherwise just the headline. This handles the "Siemens S100 returns both EQ3 and EQ6" case: if both are genuinely popular we show both, if one dominates we lead with it.
- Category workflow (`shopping_v2`): render up to **3** non-accessory groups, same 50%-of-top threshold for the 3rd.
- Accessory-flagged groups are never rendered, regardless of prominence.

## Module Layout

### New files

**`src/workflows/shopping/pipeline_v2.py`** — single module, pure async functions:

```python
@dataclass
class Candidate:
    title: str
    site: str
    site_rank: int          # 1-based position in that site's search result
    price: float | None
    original_price: float | None
    url: str
    rating: float | None
    review_count: int | None
    review_snippets: list[str]   # whatever scraper returned

@dataclass
class ProductGroup:
    representative_title: str
    member_indices: list[int]    # indices into candidates list
    is_accessory_or_part: bool
    prominence: float            # sum(1 / site_rank) across members

@dataclass
class ReviewSynthesis:
    praise: list[str]
    complaints: list[str]
    red_flags: list[str]
    insufficient_data: bool

async def step_resolve(query: str, per_site_n: int) -> list[Candidate]: ...
async def step_group(candidates: list[Candidate]) -> list[ProductGroup]: ...
async def step_synthesize_reviews(group: ProductGroup, candidates: list[Candidate]) -> ReviewSynthesis: ...
def format_group_card(group: ProductGroup, synthesis: ReviewSynthesis, candidates: list[Candidate]) -> str: ...
def format_response(cards: list[str]) -> str: ...
```

**`src/workflows/shopping/prompts_v2.py`** — two templates:
- `GROUPING_PROMPT` — frozen, versioned, input: JSON list of `{index, title, site, price}`; output: JSON list of groups with `{representative_title, member_indices, is_accessory_or_part}`.
- `SYNTHESIS_PROMPT` — frozen, versioned, input: representative title + review snippets; output: JSON `{praise, complaints, red_flags, insufficient_data}`.

**Workflow JSONs:**
- `src/workflows/shopping/product_research_v2.json`
- `src/workflows/shopping/quick_search_v2.json`
- `src/workflows/shopping/shopping_v2.json` (retains clarifier phase from v1)

Each wires its steps to a new `shopping_pipeline_v2` agent type, matching the existing `shopping_pipeline` convention.

### Files touched in place

- `src/app/telegram_bot.py:4503-4511` — `wf_map` values switch to `_v2` names.
- Agent registry — add `shopping_pipeline_v2` mapping to `pipeline_v2.py` entry.

### Files deleted in follow-up PR (not this one)

After v2 proves out on live traffic (~1 week of manual observation):
- `src/workflows/shopping/pipeline.py` (v1)
- `src/workflows/shopping/product_matcher.py`
- `src/workflows/shopping/shopping.json`, `product_research.json`, `quick_search.json` (v1)
- Any helpers used only by v1 (`_filter_relevant`, `_annotate_fake_discounts`, etc.)

## Data Flow & Degradation

### Happy path artifacts

```
user_query (str)
  → candidates: list[Candidate]
  → groups: list[ProductGroup]              (accessories dropped here)
  → groups_with_synthesis: list[tuple[ProductGroup, ReviewSynthesis]]
  → response: str (Telegram markdown)
```

### Degradation matrix

| Failure | Fallback |
|---|---|
| `step_resolve` returns zero candidates | Return "Bulunamadı" message. For `quick_search_v2`, existing `escalation_target: "shopping_v2"` fires. |
| `step_group` LLM fails or times out | Fallback grouping: each site's top-1 becomes its own group. User sees parallel cards per site — worst-case acceptable, matches "trust sites" thesis. |
| `step_synthesize_reviews` fails | Card omits praise/complaints; renders prices + community counts + footer `⚠️ Yeterli inceleme verisi yok`. |
| Synthesis returns `insufficient_data: true` | Same as above — neutral footer, no fabricated text. |
| Cloud quota exhausted during synthesis | Send price-only card immediately; queue a synthesis follow-up task via General Beckman (`fire_and_forget=True`). Deferred synthesis sends a second Telegram message when quota recovers. |
| Single scraper times out | Drop from candidate pool, continue. Existing `shopping_search` tool behavior. |
| Category clarifier LLM fails | Proceed with raw query. Existing v1 behavior retained. |

### No new DB schema

- `model_pick_log` already captures both LLM calls.
- Observability: structured log entries in `step_group` and `step_synthesize_reviews` capture `group_count`, `accessory_drop_count`, `snippet_count`, `synthesis_insufficient`, `fallback_triggered`. Use `logger.info` with a fixed event name for grep-ability. No persisted workflow-state fields.

## Output Format

Reviews-first Telegram markdown per product group. Example:

```
*Siemens EQ.6 Plus S100* ⭐ 4.5/5 (312 değerlendirme)

👍 Kullanıcılar beğeniyor:
• Köpük kalitesi iyi
• Sessiz çalışıyor
• Temizlik kolay

👎 Şikayetler:
• Demleme ünitesi 2 yıl sonra arızalanıyor
• Fiyat yüksek

⚠️ Dikkat:
• Şikayetvar'da 47 şikayet kaydı

💰 *Fiyatlar:*
• Hepsiburada — 24.745 TL
• Akakçe — 25.499 TL
• Amazon.tr — stokta yok

💬 Topluluk: Technopat (12 konu), Ekşi (8 entry)
```

Sections omitted when empty:
- No praise/complaints/red_flags → omit those blocks entirely
- `insufficient_data: true` → show only title + rating + price list + community + footer
- No community data → omit the 💬 line

Bullet points in praise/complaints/red_flags are capped at 3 each by the synthesis prompt.

## Testing

All LLM-touching tests use recorded fixtures; no live LLM calls in CI or developer test runs.

### Unit tests (`tests/workflows/shopping/test_pipeline_v2.py`)

- `test_step_resolve_respects_per_site_n` — mock `shopping_search`, assert slicing.
- `test_step_resolve_no_filtering` — candidates returned verbatim in site-rank order.
- `test_step_group_parses_llm_output` — frozen LLM response fixtures × 5 scenarios (Siemens part-vs-machine, clear category tiers, single-site result, all-same-product, malformed LLM output → fallback).
- `test_step_synthesize_parses_llm_output` — fixtures for (full synthesis), (insufficient_data), (malformed output).
- `test_format_group_card` — snapshot tests: full card, insufficient-data card, accessory-flagged (asserts accessory does NOT render), community-only card.
- `test_position_aggregation` — pure Python, `1/site_rank` sum math.

### Integration tests (`tests/workflows/shopping/test_pipeline_v2_integration.py`, marker `@pytest.mark.shopping_v2`)

All LLM calls are patched with fixtures. Scraper calls use recorded HTTP cassettes (existing infra in `tests/fixtures/scrapers/`).

- `test_pipeline_siemens_s100_drops_accessory` — the exact 2026-04-20 failure. Recorded scraper output + fixture grouping LLM response that correctly flags the brewing-unit part as accessory. Assert rendered output contains the EQ6 machine, not the part.
- `test_pipeline_category_coffee_machines` — category query, assert 2-3 groups rendered.
- `test_pipeline_thin_reviews_renders_footer` — query with scarce review snippets, assert neutral footer appears.
- `test_pipeline_grouping_llm_failure_fallback` — grouping LLM raises, assert per-site fallback groups.

### Manual validation before merge

Cut a dev branch, point `wf_map` at v2 files in a local instance, replay ~10 recent shopping queries from Telegram audit log. Acceptance criteria:
- No regressions vs v1 on clean queries (query matches a single clear product).
- Clear improvement on the Siemens-style cases (ambiguous model numbers, accessory pollution).
- Reviews section is populated with real user language (not fabricated) when scraper data supports it.

## Rollout Plan (summary)

1. Land v2 pipeline + JSONs + prompts + tests.
2. Switch `wf_map` to v2 names.
3. Manual Telegram validation (≥10 real queries, ≥1 week passive observation).
4. If no regressions: follow-up PR deletes v1 pipeline, matcher, filter, and old JSONs.
5. If regressions found: revert `wf_map` to v1 names (one-line change), fix v2, retry.

## Out of Scope

- **Scraper review-text depth.** Extracting full review bodies from product detail pages is a separate project. v2 works with whatever snippet text current scrapers return.
- **Fake-discount detection rework.** The existing cross-store median heuristic becomes unreliable with LLM grouping. If a group's prices diverge wildly, the synthesis step can mention it as a complaint/red flag. No structured `is_suspicious_discount` flag in v2.
- **Scraper fleet changes.** No new sites, no changes to existing scraper output shape.
- **`price_watch.json`, `combo_research.json`, `gift_recommendation.json`, `exploration.json`.** These smaller workflows remain on v1 logic. Migration to v2 deferred until after primary three are proven.
- **Deletion of v1 code.** Kept for rollback in this PR; deleted in a follow-up.

## Open Questions

None blocking. Two items to revisit after launch:

1. **Group ranking weight.** `1 / site_rank` position aggregation might need tuning — e.g. weighting by site authority (Trendyol/Hepsiburada > smaller sites). Defer to post-launch data.
2. **Community scoring.** Şikayetvar complaint volume currently surfaces as a red flag only if synthesis picks it up. A structured "community weight" signal (complaint count / mention count) could feed into the red-flags section deterministically. Defer.
