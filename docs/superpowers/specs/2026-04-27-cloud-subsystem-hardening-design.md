# Cloud Subsystem Hardening — Design

**Date:** 2026-04-27
**Scope:** G2, G3, G4, G5 from cloud audit (auth fail-fast, telemetry provider column, provider model discovery, cloud benchmarking).
**Out of scope:** G1 (e2e cloud call test), G6 (KDV battle-test), auto-write discovered models back to YAML, cost-aware scheduling.

## Audit summary

End-to-end cloud path is paper-wired and code-complete: registry, selection, dispatcher branch, hallederiz execution, KDV admission, secret detection. Confirmed gaps:

- **G2** Silent auth failures — litellm reads keys itself; missing/wrong key surfaces as retried network error.
- **G3** `model_pick_log` lacks provider column; cloud-vs-local analytics impossible without name-prefix joins.
- **G4** Cloud catalog is hand-curated `cloud:` block in `src/models/models.yaml`; `litellm_name` list goes stale.
- **G5** Capability scores rely on `CLOUD_PROFILES` name-substring match; unknown models get flat `6.0`. AA benchmark enricher exists for locals but not extended to cloud.

YAML clarification: `cloud:` block was treated as catalog by `registry.py:1172-1212`, but file header (`Most properties are auto-detected. Use this for overrides only.`) declares it overrides-only. Design honors stated intent.

## Architecture

New subsystem **CloudDiscovery** under `packages/fatih_hoca/src/fatih_hoca/cloud/`:

```
cloud/
  discovery.py            # orchestrator
  family.py               # cross-provider model identity (normalization)
  cache.py                # disk cache (TTL 7d, served on failure)
  providers/
    groq.py
    openai.py
    anthropic.py
    gemini.py
    cerebras.py
    sambanova.py
    openrouter.py
```

Each adapter exports `async fetch_models(api_key) -> ProviderResult` where `ProviderResult` carries `auth_ok: bool`, `models: list[DiscoveredModel]`, `error: str | None`. `DiscoveredModel` is a dict of opportunistically-scraped fields.

### Boot data flow

```
config.AVAILABLE_KEYS
  └─> cloud_discovery.refresh_all()
       ├─> per provider (concurrent): fetch_models(key)
       │     ├─ 200       → cache write, parse models
       │     ├─ 401/403   → auth fail, provider disabled, telegram alert
       │     └─ 5xx/net   → cache served if fresh, else disabled, telegram alert
       └─> diff vs previous snapshot, log adds/removes/active-flips
  └─> for each (provider, discovered_model):
       ├─ family_key = family.normalize(provider, litellm_name)
       ├─ detected = detect_cloud_model(litellm_name, provider)
       ├─ merge scraped fields (provider-data wins for context, pricing, sampling defaults)
       └─ register ModelInfo(family=family_key)
  └─> bench_enricher.enrich_cloud(family_key) for each unique family
  └─> fatih_hoca._available_providers = {p for p in providers if result.auth_ok}
  └─> if any provider failed: telegram alert (throttled ≤1/24h per provider)
```

### Periodic refresh

Beckman scheduled_task `cloud_refresh` fires every 6h. Re-runs `cloud_discovery.refresh_all()` and bench enrichment. On any provider state flip (auth_ok ↔ auth_fail), telegram alert.

No long-running background loop. No process sleeping for days.

## §G2 — Auth fail-fast at boot

`cloud_discovery.refresh_all()` doubles as auth gate. Adapter outcomes:

| Outcome | Status code | Action |
|---------|-------------|--------|
| OK | 200 | Models registered, provider enabled |
| Auth fail | 401, 403 | Provider disabled, telegram alert |
| Server error | 5xx | Cache served if fresh; else disabled, telegram alert |
| Network error | timeout, conn-refused | Cache served if fresh; else disabled, telegram alert |
| Quota exceeded at probe | 429 | Provider enabled, log warning (probe rate-limit treated as transient) |

Selector already filters by `_available_providers` (`selector.py:241`), so disabled providers never get picked. No per-call pre-flight needed in `caller.py`.

Removes silent litellm auth-error retry case: if key bad, provider disabled before any work routes to it.

## §G3 — `model_pick_log.provider`

Migration in `src/infra/db.py` schema bootstrap (idempotent):

```sql
ALTER TABLE model_pick_log ADD COLUMN provider TEXT;
UPDATE model_pick_log SET provider='local' WHERE provider IS NULL;
CREATE INDEX IF NOT EXISTS idx_pick_log_provider ON model_pick_log(provider);
```

All existing rows are local picks (cloud was unreachable). Backfill is single statement.

Insert path in `packages/fatih_hoca/src/fatih_hoca/pick_log.py` populates `provider=model.provider` from registry record at log-write time.

Enables queries:
```sql
-- cloud share over time
SELECT date(timestamp), provider, COUNT(*) FROM model_pick_log GROUP BY 1,2;
-- cloud failure rate by provider
SELECT provider, AVG(success) FROM model_pick_log WHERE provider != 'local' GROUP BY 1;
```

## §G4 — Discovery, scheduled refresh, cache

### Per-provider scrape

Adapters read provider-specific shapes and project to `DiscoveredModel`:

| Provider | Useful fields exposed |
|----------|----------------------|
| groq | `context_window`, `max_completion_tokens`, `active`, `owned_by` |
| openrouter | `context_length`, `pricing.prompt`, `pricing.completion`, `top_provider.max_completion_tokens`, `top_provider.is_moderated` |
| gemini | `inputTokenLimit`, `outputTokenLimit`, `supportedGenerationMethods`, `temperature`, `topP`, `topK` |
| openai | `id`, `created`, `owned_by` (sparse) |
| anthropic | `id`, `display_name`, `created_at`, `type` (sparse) |
| cerebras / sambanova | `id` only |

Merge rules into `detect_cloud_model()` output:

- groq `active=false` → skip registration entirely
- openrouter `pricing.*` → override `cost_per_1k_input`, `cost_per_1k_output` (litellm db often stale for openrouter)
- gemini sampling defaults → seed `sampling_overrides` when no YAML override
- groq / openrouter `context_window` / `context_length` → override litellm db value
- groq / openrouter `max_completion_tokens` → override `max_tokens`

YAML `cloud:` block stays respected. Manual additions merged on top of discovered set so preview/private models providers don't expose can still be hand-listed.

### Cross-provider family dedup

`family.py::normalize(provider, litellm_name) -> str` extracts canonical model identity. Examples:

| Inputs | Family key |
|--------|-----------|
| `groq/llama-3.3-70b-versatile` | `llama-3.3-70b` |
| `cerebras/llama3.3-70b` | `llama-3.3-70b` |
| `sambanova/Meta-Llama-3.3-70B-Instruct` | `llama-3.3-70b` |
| `claude-sonnet-4-20250514` | `claude-sonnet-4` |
| `gpt-4o` | `gpt-4o` |
| `gemini/gemini-2.0-flash` | `gemini-2.0-flash` |

Normalization rules per family pattern (regex set, ordered most-specific first). Strip provider prefix, lowercase, drop date/build suffix, drop `instruct`/`versatile`/`it`/`turbo` modifiers.

Unmatched → falls back to bare `litellm_name`. Logged as `family_unknown` so new families surface for manual rule addition.

`ModelInfo` gets new field `family: str`. Same family across providers shares benchmark cache entry. Diverging per-provider fields: `rate_limit_rpm/tpm`, `cost_per_1k_*`, `context_length` (when provider truncates).

### Cache

`.benchmark_cache/cloud_models/<provider>.json`:

```json
{
  "fetched_at": "2026-04-27T10:00:00Z",
  "status": "ok",
  "models": [{"litellm_name": "...", "context_window": 131072, ...}, ...]
}
```

TTL 7d. On TTL expiry without successful refresh, marked stale (still served, warning logged). On 2× TTL, evicted (provider treated as unreachable on next failure).

### Telegram alert

- Channel: existing KutAI bot (cloud discovery happens after Telegram is online).
- Throttle: per-provider cooldown 24h. State held in `cloud/alert_throttle.json`.
- Triggers: `auth_fail` first time, `auth_ok → auth_fail` transition (always), `auth_fail → auth_ok` recovery (always).
- Message: `[cloud] groq disabled — 401 Invalid API key (next retry 16:00)`

### Quota / rate-limit signal

Discovery does NOT probe quota endpoints. Reasoning:
- `/models` rarely returns rate-limit headers (those come on request endpoints).
- Dedicated quota endpoints inconsistent across providers (groq has `/openai/v1/usage`, anthropic has `/v1/organizations/usage_report`, gemini does not).
- KDV header parsing on real calls is the source-of-truth; discovery probe duplicates it.

`_FREE_TIER_DEFAULTS` retained as cold-start estimate. KDV updates from response headers on first real call.

**KDV no-data warning**: 6h scheduled task additionally checks per-enabled-cloud-provider observation count. Zero observations after Nh of being enabled (configurable, default 24h) → log warning + KDV insight surfaces in `/status`. Means defaults still in use, no real quota signal.

## §G5 — Cloud benchmarking

Extend `packages/fatih_hoca` benchmark enricher to accept cloud `ModelInfo`. Match strategy hybrid + fallback chain:

1. **Hybrid family match**:
   - Open-source-on-cloud (groq/cerebras/sambanova/openrouter): use `family_key` against AA index. One AA hit fills capabilities for all providers serving that family.
   - Proprietary (claude/gpt/gemini): exact litellm_name match against AA proprietary entries.
2. **Fallback chain**: AA hit → `CLOUD_PROFILES` name-substring match → flat `6.0` across capabilities.

`CLOUD_PROFILES` retained as fallback (hand-curated, better signal than flat default for unknown models). Not removed.

### Validation milestone

Until a per-family review flag is set, the **active** capability values used by selection come from `CLOUD_PROFILES` (or flat 6.0 fallback) even when AA produced a match. AA-derived values are computed and stored on `ModelInfo.benchmark_scores` but not promoted to `ModelInfo.capabilities` until reviewed. Review state lives in `.benchmark_cache/cloud_match_approved.json` keyed by family. Once approved, AA values become active capabilities.

This applies to both proprietary and open-source-on-cloud families — humans approve the family→benchmark mapping before it influences picks.

After first end-to-end discovery run with real keys, dump artifact `.benchmark_cache/cloud_match_review.json`:

```json
[
  {
    "litellm_name": "groq/llama-3.3-70b-versatile",
    "family": "llama-3.3-70b",
    "matched_aa_entry": "Llama 3.3 70B Instruct",
    "source": "aa",
    "final_capabilities": {"reasoning": 7.4, "coding": 6.8, ...}
  },
  ...
]
```

Manual review gate before trusting bench-overrides for proprietary models. Until reviewed, proprietary models keep `CLOUD_PROFILES` capabilities even when AA hit available.

## §Test plan

**Unit:**
- Each provider adapter: 200 + scrape, 401, 403, 5xx, network timeout, malformed JSON.
- `family.normalize()`: matrix of (provider, litellm_name) → expected family across all 7 providers + 3 families each.
- Cache: write, read fresh, read stale, read evicted, concurrent writers.
- Discovery diff logic: add, remove, active-flip, unchanged.
- Alert throttling: first failure, repeat failure within 24h (suppressed), failure after 24h (re-alerted), recovery (always alerted).

**Migration smoke:** in-memory sqlite, schema add, backfill, insert with provider, query by provider.

**Integration (gated by env-var presence, opt-in):**
- One live `/models` call per provider that has key set in `.env`. Asserts adapter parsing + cache write.

**Out of scope here:** e2e cloud LLM call test (gap G1, deferred to next phase).

## Implementation order

1. Skeleton: `cloud/` package, dummy adapters, family.py, cache.py with tests.
2. G3 migration + provider column wiring.
3. Real adapters one-by-one (start groq — free + responsive `/models`).
4. Discovery orchestrator + cache integration + telegram alert.
5. Beckman scheduled_task wiring.
6. G5 bench enricher cloud extension + validation artifact.
7. Live discovery run with real keys, manual review of `cloud_match_review.json`.
8. KDV no-data warning hook.

## Risks

- **Family normalization false positives**: misclassifying Qwen-Coder-32B as Qwen-32B inflates coding scores. Mitigation: ordered regex (most-specific first), `family_unknown` log channel for new releases.
- **Provider `/models` rate-limited at boot**: 6h scheduled refresh hits same endpoint. Mitigation: cache-first fallback covers transient 429s; TTL 7d means real refresh only needs to succeed monthly.
- **Telegram alert spam during outage**: 24h per-provider throttle with state-flip override. Worst case: one alert per provider per day during sustained outage.
- **CLOUD_PROFILES drift**: hand-curated table grows stale as providers add models. Mitigation: AA is primary source; CLOUD_PROFILES used only on miss; `family_unknown` log surfaces gaps.
