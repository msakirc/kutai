# Shopping v2 Validation Notes

**Date:** 2026-04-21
**Spec:** `docs/superpowers/specs/2026-04-21-shopping-trust-sites-synthesize-reviews-design.md`
**Plan:** `docs/superpowers/plans/2026-04-21-shopping-v2-implementation.md`
**Cutover commit:** `057ec71`

## Automated test status

- `tests/shopping/test_pipeline_v2.py` — **20 / 20 passing**
- Broader `tests/shopping/` suite — 712 passing, 1 pre-existing timing-flaky failure (`test_record_1000_requests_total_time`) unrelated to v2 changes.

## Manual Telegram checklist

Run each of these on the live bot (KutAI) and record observations below. Successful Siemens test is the headline case — the 2026-04-20 bug was that the brewing-unit accessory was picked as winner.

### 1. Named product — ambiguous model number

- Send: `/shop Siemens EQ6 S100`
- Expected: reviews-first card(s), DL-Pro brewing unit accessory excluded, 24-25k TL machine surfaced.
- Observed: _fill in_

### 2. Deep-research fork — specific

- Tap `🔬 Detaylı Araştır` → `🎯 Belirli ürün` → send: `Philips 3200 LatteGo`
- Expected: `product_research_v2` workflow runs, reviews-first card.
- Observed: _fill in_

### 3. Deep-research fork — category

- Tap `🔬 Detaylı Araştır` → `🏷 Kategori` → send: `kahve makinesi 5000 TL altı`
- Expected: clarifier may run; `shopping_v2` workflow produces up to 3 cards.
- Observed: _fill in_

### 4. Quick search

- Send: `/shop bluetooth kulaklık`
- Expected: `quick_search_v2` workflow, fast (<60s), 1-2 cards.
- Observed: _fill in_

### 5. Thin review data

- Send: `/shop` + some niche product with little community coverage.
- Expected: card shows prices, but review section shows `⚠️ Yeterli inceleme verisi yok`.
- Observed: _fill in_

## Things to verify in logs / DB per mission

- `model_pick_log` rows for `task="shopping_grouper"` and `task="shopping_review_synthesizer"` — two entries per mission (category flow has more, plus clarify).
- `workflow_state` entries show the per-step artifact flow: `search_results` → `grouped_synth` → `shopping_response`.

## Known limitations (not blockers)

- `review_snippets` is empty for all candidates because real scrapers don't populate it. Synthesis therefore short-circuits to `insufficient_data=True` for most real queries. This is expected per the spec (scraper depth is out of scope for v2). Output still shows title + price + rating panels, just no praise/complaints/red-flags until scrapers are deepened.
- `community_counts` is always `{}` from the pipeline — the community scrape hookup is a follow-up.

## Rollback

If v2 regresses: revert commit `057ec71` (wf_map cutover). One-line revert. All v1 code still present and untouched.

## Follow-ups (not in this PR)

1. Delete v1 (`src/workflows/shopping/pipeline.py`, `product_matcher.py`, v1 JSONs) after ≥1 week running clean.
2. Deepen scrapers to fetch review text from product detail pages — unblocks synthesis quality.
3. Wire `community_counts` from scraped Technopat/Ekşi/Şikayetvar thread counts into `group_and_synthesize` output.
4. Migrate `combo_research`, `gift_recommendation`, `exploration`, `price_watch` to v2 primitives.
