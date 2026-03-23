# Dynamic Rate Limits + Quota Planner Design

**Date:** 2026-03-16
**Status:** Approved

## Problem

Rate limits are hardcoded in `_FREE_TIER_DEFAULTS` (model_registry.py) and `PROVIDER_AGGREGATE_LIMITS` (rate_limiter.py). The system cannot detect when providers change limits, upgrade tiers, or have per-model limits that differ from provider-wide limits. Expensive models (Claude, GPT-4o) are underutilized because there's no strategy to maximize their usage while reserving capacity for hard tasks.

## Solution

Two components:

### A. Dynamic Limit Discovery via Response Headers

Parse `x-ratelimit-*` / `anthropic-ratelimit-*` headers from every API response to learn actual RPM/TPM limits and remaining capacity. Each provider has different header formats:

| Provider | Header format | Per-model? | Special |
|----------|--------------|------------|---------|
| OpenAI | `x-ratelimit-{limit,remaining,reset}-{requests,tokens}` | Yes | Standard |
| Anthropic | `anthropic-ratelimit-{requests,tokens}-{limit,remaining,reset}` | Yes | Token bucket, ITPM/OTPM separate |
| Groq | `x-ratelimit-*` (OpenAI-compatible) | Org-level | |
| Gemini | `x-ratelimit-{limit,remaining,reset}` | Yes | RPD (daily) + RPM |
| Cerebras | `x-ratelimit-*-{requests-day,tokens-minute}` | Yes | Daily limits, float-second resets |
| SambaNova | RPM + RPD limit/remaining/reset | Per-model | RPD important |

A normalizer (`header_parser.py`) converts all formats into a common `RateLimitSnapshot` dataclass. The existing sliding-window tracker becomes the fallback; headers are authoritative when fresh.

### B. Quota Planner

A `QuotaPlanner` that dynamically adjusts the minimum difficulty threshold for using expensive (paid) models based on:

- Current quota utilization from headers
- Max difficulty of upcoming tasks in the queue
- Time until quota resets
- Recent 429 frequency

When quotas are healthy and no hard tasks are queued, the threshold lowers (more tasks use expensive models). When quotas are tight or hard tasks are pending, the threshold rises (reserve capacity). Never blocks actual work — expensive models stay as candidates, just with lower scores.

## Files Changed

| File | Change |
|------|--------|
| `src/models/header_parser.py` | NEW — Provider-specific header normalization |
| `src/models/quota_planner.py` | NEW — Dynamic difficulty threshold |
| `src/models/rate_limiter.py` | Enhanced with header-derived state, daily limits, limit discovery |
| `src/core/router.py` | Wire header parsing, integrate quota planner in scoring |
| `src/models/model_registry.py` | Clarify `_FREE_TIER_DEFAULTS` as initial fallbacks |

## Key Principles

- Headers are authoritative when available; sliding window is fallback
- Never block work — expensive models always remain candidates, just penalized in scoring
- Reserve capacity for hard tasks by peeking at the task queue
- When limits restore, signal backpressure queue to retry
- Daily limits (Cerebras, SambaNova) are first-class citizens
