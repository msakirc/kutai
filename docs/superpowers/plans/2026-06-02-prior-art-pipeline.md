# Prior-Art Pipeline (3-stage, fabrication-proof) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single LLM `prior_art_search` step (which fabricates competitor lists with null URLs because it transcribes a deterministic tool by hand) with a 3-stage pipeline where the synthesis LLM can only judge candidates that were really fetched.

**Architecture:** `1.0a` LLM derives search queries → `1.0b` mechanical fetches real candidates (relocated out of the `vecihi` scraper into `src/research/`) → `1.0c` LLM synthesizes a report **only** from fetched candidates. The `prior_art_min_coverage` post-hook is hardened so every `attempted_solutions[i].url` must exist in the fetched candidate set — fabrication becomes structurally impossible, not prompt-begged.

**Tech Stack:** Python 3.10, aiohttp, aiosqlite, pytest. Packages: `mr_roboto` (mechanical dispatch), `general_beckman` (post-hook payload routing), `src/workflows/i2p/i2p_v3.json` (workflow), `src/agents` (ReAct agents).

---

## Why this shape (root cause, proven)

Task #259348 (mission_80, step 1.0) was retried across Qwen-9B / Qwen-35B / gemma-26B over 4+ hours. Every model wrote `attempted_solutions` with `url: null` for real apps it knew from training (Habitica, TickTick, Streaks, Aime, Forest). The `find_prior_art` tool **never** emits `url=None` (it uses `or ""`) and **always** emits `search_summary`, so the saved artifact provably did not come from the tool. The researcher's system prompt mandates prose `final_answer` and never mentions the JSON schema or writing a file — so the model shortcuts to fabrication. The schema check + post-hook correctly reject it → DLQ.

`find_prior_art` is also architecturally misplaced: `vecihi` is a fetch engine (HTTP→TLS→Stealth→Browser); the prior-art query/classify/verdict logic is research orchestration that was dumped into it during Z1 Tier 6B.

## File Structure (decomposition)

- `src/research/__init__.py` — new package marker.
- `src/research/prior_art.py` — **relocated** fetch/candidate-assembly. Exposes `fetch_candidates(...)`. Keeps source fetchers, normalize, URL-resolve, dedup, cache. **Deletes** the weak heuristics `_build_queries`, `_classify_verdict`, `_extract_lessons` (LLMs replace them) and the full-report assembly `find_prior_art`.
- `packages/mr_roboto/src/mr_roboto/prior_art_fetch.py` — new mechanical verb `prior_art_fetch`: reads queries JSON, calls `fetch_candidates`, writes candidates JSON.
- `src/agents/query_planner.py` — new LLM agent: idea_brief_final → queries + domain_keywords (JSON).
- `src/agents/prior_art_synthesizer.py` — new LLM agent: candidates → prior_art_report (JSON), URLs constrained to candidates.
- `packages/mr_roboto/src/mr_roboto/prior_art_min_coverage.py` — **modify**: add `candidates_path` param + "url ∈ candidates" rule.
- `packages/general_beckman/src/general_beckman/apply.py` — **modify** payload builder (line ~2197) to pass `candidates_path`.
- `src/workflows/i2p/i2p_v3.json` — **modify**: replace step `1.0` with `1.0a`/`1.0b`/`1.0c`.
- Retire: `src/tools/prior_art.py`, its registration in `src/tools/__init__.py`, `find_prior_art` in `src/agents/researcher.py` allowed_tools, `vecihi/prior_art.py` + its `vecihi/__init__.py` export.

## Environment notes (carried from handoffs)

- venv python: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`.
- `tests/` and `packages/*/tests/` collide on conftest — run them in **separate** pytest invocations.
- Always run pytest with a timeout and `-p no:warnings`. DB-integration tests load the embedding model (~19s) — slow, not hung.
- Changes load on next KutAI restart (founder restarts via Telegram), not via git.

---

### Task 1: New `src/research` package + relocate fetch logic

**Files:**
- Create: `src/research/__init__.py`
- Create: `src/research/prior_art.py`
- Test: `tests/research/test_prior_art_fetch_candidates.py`

- [ ] **Step 1: Create the package marker**

`src/research/__init__.py`:
```python
"""Research-domain orchestration (prior-art, idea validation).

Uses vecihi/HTTP to *fetch*; this package owns the research *logic*
(query handling, candidate normalization, verdict inputs). Keeps domain
logic out of the vecihi scraper.
"""
```

- [ ] **Step 2: Write the failing test for `fetch_candidates`**

`tests/research/test_prior_art_fetch_candidates.py`:
```python
import asyncio
import pytest
from src.research import prior_art


class _FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self, content_type=None):
        return self._payload

    async def text(self):
        return ""


class _FakeSession:
    """Returns HN hits for the algolia URL, empty otherwise. HEAD ok."""
    def __init__(self):
        self.closed = False

    def get(self, url, params=None, timeout=None):
        if "algolia" in url:
            return _FakeResp(200, {"hits": [
                {"title": "Habitica", "url": "https://habitica.com",
                 "objectID": "1"},
            ]})
        return _FakeResp(200, {})

    def head(self, url, timeout=None, allow_redirects=True):
        return _FakeResp(200, {})

    async def close(self):
        self.closed = True


def test_fetch_candidates_returns_real_urls_only(tmp_path):
    sess = _FakeSession()
    out = asyncio.run(prior_art.fetch_candidates(
        queries=["habit tracker app"],
        domain_keywords=["habit tracker"],
        ambition_tier="private_beta",
        k=10,
        session=sess,
        db_path=str(tmp_path / "c.db"),
    ))
    assert isinstance(out, dict)
    cands = out["candidates"]
    assert cands, "expected at least one fetched candidate"
    # Every fetched candidate carries a real http(s) URL — never None.
    for c in cands:
        assert isinstance(c["url"], str) and c["url"].startswith("http")
    # search_summary metadata travels with the candidates for 1.0c.
    assert "queries_run" in out["search_summary"]
    assert out["search_summary"]["total_results_inspected"] >= 1


def test_fetch_candidates_empty_when_no_hits(tmp_path):
    class _Empty(_FakeSession):
        def get(self, url, params=None, timeout=None):
            return _FakeResp(200, {})
    out = asyncio.run(prior_art.fetch_candidates(
        queries=["nonexistent"], domain_keywords=["nope"],
        session=_Empty(), db_path=str(tmp_path / "c.db"),
    ))
    assert out["candidates"] == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/research/test_prior_art_fetch_candidates.py -p no:warnings -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.research'` / `fetch_candidates` undefined.

- [ ] **Step 4: Implement `src/research/prior_art.py`**

Copy `packages/vecihi/src/vecihi/prior_art.py` to `src/research/prior_art.py`, then:
- Keep verbatim: the cache block (`_cache_db_path`, `_ensure_cache_table`, `_hash_keywords`, `_read_cache`, `_write_cache`), all source fetchers (`_fetch_json`, `_query_hn`, `_query_wikipedia`, `_query_wayback`, `_query_product_hunt`), all normalizers (`_normalize_hn`, `_normalize_wiki`, `_normalize_ph`), `_dedup_by_name`, the URL sweep (`_head_ok`, `_resolve_urls`), `_annotate_cache_hit`, module constants.
- **Delete** `_build_queries`, `_split_by_relevance`, `_extract_lessons`, `_classify_verdict`, and the entire `find_prior_art` function (the verdict/lessons/relevance heuristics move to the LLM).
- Add the new public entry point. It accepts queries (already produced by the 1.0a LLM) instead of building them, runs the same multi-source sweep + URL resolution, and returns candidates + search metadata (no verdict, no lessons):

```python
async def fetch_candidates(
    queries: list[str],
    domain_keywords: list[str] | None = None,
    k: int = _DEFAULT_K,
    ambition_tier: str = "private_beta",
    *,
    db_path: str | None = None,
    session: "aiohttp.ClientSession | None" = None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> dict[str, Any]:
    """Fetch + normalize + URL-resolve prior-art candidates for a fixed
    query set. Returns ``{"candidates": [...], "search_summary": {...}}``.

    Every returned candidate carries a real, HEAD-resolved http(s) URL
    (unreachable ones are marked ``status="dead"`` but keep their URL).
    Verdict / lessons / relevance judgement is the synthesis LLM's job
    (step 1.0c) — this function never invents entries.
    """
    if aiohttp is None:  # pragma: no cover
        raise RuntimeError("aiohttp required for src.research.prior_art")

    domain_keywords = list(domain_keywords or [])
    queries = [q for q in (queries or []) if isinstance(q, str) and q.strip()][:5]
    cache_key = _hash_keywords(domain_keywords or [(" ".join(queries))[:64]])
    cached = _read_cache(cache_key, db_path=db_path)

    candidates: list[dict[str, Any]] = []
    sources_used: list[str] = []
    rate_limit_hits: list[dict[str, str]] = []
    total_inspected = 0
    empty_count = 0

    own_session = False
    if session is None:
        session = aiohttp.ClientSession()
        own_session = True

    started = asyncio.get_event_loop().time()
    try:
        # Tier 1: HN  (identical body to the old find_prior_art tiers)
        try:
            hn_hits, hn_rate = await asyncio.wait_for(
                _query_hn(session, queries), timeout=_SOURCE_TIMEOUT + 1)
            if hn_rate:
                rate_limit_hits.append({"source": "hn_algolia", "err": "429"})
            sources_used.append("hn_algolia")
            if hn_hits:
                candidates.extend(_normalize_hn(hn_hits))
                total_inspected += len(hn_hits)
            else:
                empty_count += 1
        except asyncio.TimeoutError:
            rate_limit_hits.append({"source": "hn_algolia", "err": "timeout"})
            empty_count += 1

        # Tier 2: Wikipedia
        try:
            wiki_hits, wiki_rate = await asyncio.wait_for(
                _query_wikipedia(session, queries), timeout=_SOURCE_TIMEOUT + 1)
            if wiki_rate:
                rate_limit_hits.append({"source": "wikipedia", "err": "429"})
            sources_used.append("wikipedia")
            if wiki_hits:
                candidates.extend(_normalize_wiki(wiki_hits))
                total_inspected += len(wiki_hits)
            else:
                empty_count += 1
        except asyncio.TimeoutError:
            rate_limit_hits.append({"source": "wikipedia", "err": "timeout"})
            empty_count += 1

        # Tier 3: Wayback (per top-k candidate). Reuse the old block verbatim.
        if ambition_tier in ("private_beta", "public_launch", "revenue_product") and candidates:
            sources_used.append("wayback")
            top = candidates[:k]
            wb_coros = [_query_wayback(session, c["url"]) for c in top if c.get("url")]
            if wb_coros:
                try:
                    wb_results = await asyncio.wait_for(
                        asyncio.gather(*wb_coros, return_exceptions=True),
                        timeout=_SOURCE_TIMEOUT + 4)
                except asyncio.TimeoutError:
                    wb_results = [None] * len(wb_coros)
                idx = 0
                for c in top:
                    if not c.get("url"):
                        continue
                    res = wb_results[idx] if idx < len(wb_results) else None
                    idx += 1
                    if isinstance(res, dict):
                        c.update({kk: vv for kk, vv in res.items() if kk.startswith("wayback_")})
                        if res.get("wayback_first_capture") and not c.get("status"):
                            c["status"] = "dormant"

        # Tier 4: Product Hunt (public_launch+)
        if ambition_tier in ("public_launch", "revenue_product") and queries:
            try:
                ph_hits, ph_rate = await asyncio.wait_for(
                    _query_product_hunt(session, queries[0]), timeout=_SOURCE_TIMEOUT + 1)
                if ph_rate:
                    rate_limit_hits.append({"source": "product_hunt", "err": "429"})
                if ph_hits:
                    sources_used.append("product_hunt")
                    candidates.extend(_normalize_ph(ph_hits))
                    total_inspected += len(ph_hits)
            except asyncio.TimeoutError:
                rate_limit_hits.append({"source": "product_hunt", "err": "timeout"})

        elapsed = asyncio.get_event_loop().time() - started
        should_fallback = bool(rate_limit_hits) or (empty_count >= 3 and elapsed <= _FALLBACK_TIMEOUT)
        if should_fallback and cached:
            cand = cached.get("candidates") or []
            summ = cached.get("search_summary") or {}
            summ = dict(summ)
            summ["cache_hit_for_sources"] = sources_used
            return {"candidates": cand, "search_summary": summ}

        candidates = _dedup_by_name(candidates)
        candidates = await _resolve_urls(session, candidates)
    finally:
        if own_session:
            try:
                await session.close()
            except Exception:
                pass

    def _clean(items):
        return [{kk: vv for kk, vv in it.items() if not kk.startswith("_")} for it in items]

    out = {
        "candidates": _clean(candidates[: max(k, 20)]),
        "search_summary": {
            "queries_run": queries,
            "sources_used": sources_used,
            "rate_limit_hits": rate_limit_hits,
            "total_results_inspected": total_inspected,
            "cache_hit_for_sources": [],
        },
    }
    try:
        _write_cache(cache_key, out, ttl_hours=ttl_hours, db_path=db_path)
    except Exception:  # pragma: no cover
        pass
    return out


__all__ = ["fetch_candidates", "SCHEMA_VERSION", "DEFAULT_TTL_HOURS"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/research/test_prior_art_fetch_candidates.py -p no:warnings -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/research/__init__.py src/research/prior_art.py tests/research/test_prior_art_fetch_candidates.py
git commit -m "feat(research): relocate prior-art fetch out of vecihi into src/research; queries-driven fetch_candidates (no heuristic verdict)"
```

---

### Task 2: Mechanical verb `prior_art_fetch` (1.0b)

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/prior_art_fetch.py`
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (import + `__all__` + dispatch branch in `_run_dispatch`)
- Modify: `packages/mr_roboto/src/mr_roboto/reversibility.py` (add `prior_art_fetch: full`)
- Test: `packages/mr_roboto/tests/test_prior_art_fetch_verb.py`

- [ ] **Step 1: Write the failing test**

`packages/mr_roboto/tests/test_prior_art_fetch_verb.py`:
```python
import asyncio
import json
import mr_roboto
from mr_roboto.prior_art_fetch import prior_art_fetch


def test_prior_art_fetch_reads_queries_writes_candidates(tmp_path, monkeypatch):
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps({
        "queries": ["habit tracker app"],
        "domain_keywords": ["habit tracker"],
        "ambition_tier": "private_beta",
    }), encoding="utf-8")
    cand_path = tmp_path / "candidates.json"

    async def fake_fetch(queries, domain_keywords=None, k=10,
                         ambition_tier="private_beta", **kw):
        assert queries == ["habit tracker app"]
        return {"candidates": [{"name": "Habitica",
                                "url": "https://habitica.com",
                                "status": "unknown", "sources": []}],
                "search_summary": {"queries_run": queries,
                                   "total_results_inspected": 1}}

    monkeypatch.setattr("src.research.prior_art.fetch_candidates", fake_fetch)

    res = asyncio.run(prior_art_fetch(
        queries_path=str(queries_path),
        candidates_path=str(cand_path)))
    assert res["ok"] is True
    assert res["candidate_count"] == 1
    written = json.loads(cand_path.read_text(encoding="utf-8"))
    assert written["candidates"][0]["url"] == "https://habitica.com"


def test_prior_art_fetch_via_dispatch(tmp_path, monkeypatch):
    qp = tmp_path / "q.json"; qp.write_text('{"queries":["x"]}', encoding="utf-8")
    cp = tmp_path / "c.json"

    async def fake_fetch(queries, **kw):
        return {"candidates": [], "search_summary": {"queries_run": queries}}
    monkeypatch.setattr("src.research.prior_art.fetch_candidates", fake_fetch)

    action = asyncio.run(mr_roboto.run({
        "id": 1, "mission_id": 80, "title": "t",
        "payload": {"action": "prior_art_fetch",
                    "queries_path": str(qp), "candidates_path": str(cp)},
    }))
    assert action.status == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest packages/mr_roboto/tests/test_prior_art_fetch_verb.py -p no:warnings -q`
Expected: FAIL — `No module named 'mr_roboto.prior_art_fetch'`.

- [ ] **Step 3: Implement the verb**

`packages/mr_roboto/src/mr_roboto/prior_art_fetch.py`:
```python
"""Mechanical verb — prior-art candidate fetch (i2p step 1.0b).

Reads the 1.0a query artifact, runs the deterministic multi-source fetch
in ``src.research.prior_art.fetch_candidates``, and writes the candidates
artifact for the 1.0c synthesis LLM. No LLM call, no fabrication: every
candidate URL was really fetched + HEAD-resolved.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.prior_art_fetch")


def _resolve(path: str | None) -> str | None:
    if not path or os.path.isabs(path) or os.path.isfile(path):
        return path
    try:
        from src.tools.workspace import WORKSPACE_DIR
        cand = os.path.join(WORKSPACE_DIR, path)
        return cand
    except Exception:
        return path


async def prior_art_fetch(
    queries_path: str,
    candidates_path: str,
    *,
    db_path: str | None = None,
) -> dict[str, Any]:
    qpath = _resolve(queries_path)
    if not qpath or not os.path.isfile(qpath):
        return {"ok": False, "error": f"queries artifact missing: {queries_path}"}
    try:
        with open(qpath, encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception as e:
        return {"ok": False, "error": f"failed to read queries: {e}"}

    queries = spec.get("queries") or []
    keywords = spec.get("domain_keywords") or []
    tier = spec.get("ambition_tier") or "private_beta"

    from src.research.prior_art import fetch_candidates
    out = await fetch_candidates(
        queries=queries, domain_keywords=keywords,
        ambition_tier=tier, db_path=db_path)

    cpath = _resolve(candidates_path) or candidates_path
    os.makedirs(os.path.dirname(cpath) or ".", exist_ok=True)
    with open(cpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, default=str)

    return {"ok": True,
            "candidate_count": len(out.get("candidates") or []),
            "candidates_path": candidates_path}
```

- [ ] **Step 4: Wire into `mr_roboto/__init__.py`**

Add near the other submodule imports (after line 64 `from mr_roboto.prior_art_min_coverage import prior_art_min_coverage`):
```python
from mr_roboto.prior_art_fetch import prior_art_fetch
```
Add `"prior_art_fetch",` to `__all__`.
Add this dispatch branch inside `_run_dispatch` (alongside the other `if action == ...` branches, e.g. just after the `prior_art_min_coverage` branch if present, else anywhere in the chain):
```python
    if action == "prior_art_fetch":
        from mr_roboto.prior_art_fetch import prior_art_fetch as _paf
        try:
            res = await _paf(
                queries_path=payload.get("queries_path"),
                candidates_path=payload.get("candidates_path"),
            )
            if not res.get("ok"):
                return Action(status="failed", error=res.get("error"), result=res)
            return Action(status="completed", result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))
```

- [ ] **Step 5: Add reversibility tag**

In `packages/mr_roboto/src/mr_roboto/reversibility.py`, add `"prior_art_fetch": "full",` to `VERB_REVERSIBILITY` (read-only network fetch + workspace write = full).

- [ ] **Step 6: Run tests to verify they pass**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest packages/mr_roboto/tests/test_prior_art_fetch_verb.py -p no:warnings -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/prior_art_fetch.py packages/mr_roboto/src/mr_roboto/__init__.py packages/mr_roboto/src/mr_roboto/reversibility.py packages/mr_roboto/tests/test_prior_art_fetch_verb.py
git commit -m "feat(mr_roboto): prior_art_fetch mechanical verb (i2p 1.0b) — deterministic candidate fetch"
```

---

### Task 3: `query_planner` agent (1.0a)

**Files:**
- Create: `src/agents/query_planner.py`
- Modify: `src/agents/__init__.py` (import + register in `AGENT_REGISTRY`)
- Test: `tests/agents/test_query_planner_prompt.py`

- [ ] **Step 1: Write the failing test (3-invariant prompt quality + registry)**

`tests/agents/test_query_planner_prompt.py`:
```python
from src.agents import get_agent, AGENT_REGISTRY


def test_query_planner_registered():
    assert "query_planner" in AGENT_REGISTRY
    agent = get_agent("query_planner")
    assert agent.name == "query_planner"


def test_query_planner_prompt_invariants():
    agent = get_agent("query_planner")
    p = agent.get_system_prompt({})
    assert p.lstrip().startswith("You are ")
    low = p.lower()
    assert ("must" in low or "always" in low)
    assert ("don't" in low or "never" in low)
    assert "final_answer" in p
    assert "```json" in p


def test_query_planner_no_fetch_tools():
    # 1.0a only PLANS queries; it must not fetch (that's the mechanical step).
    agent = get_agent("query_planner")
    assert "find_prior_art" not in agent.allowed_tools
    assert "web_search" not in agent.allowed_tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/agents/test_query_planner_prompt.py -p no:warnings -q`
Expected: FAIL — `"query_planner" not in AGENT_REGISTRY`.

- [ ] **Step 3: Implement the agent**

`src/agents/query_planner.py`:
```python
"""Query-planner agent (i2p step 1.0a).

Reads idea_brief_final and emits a small set of prior-art SEARCH QUERIES
+ domain keywords. It does NOT fetch and does NOT write a report — the
mechanical 1.0b step fetches; the 1.0c synthesizer judges. Splitting this
out is what stops models fabricating a competitor list from training.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.query_planner")


class QueryPlannerAgent(BaseAgent):
    name = "query_planner"
    description = "Derives prior-art search queries from an idea brief"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 2
    allowed_tools = ["read_file", "write_file"]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a prior-art search planner. You turn an idea brief into "
            "a small set of high-signal search queries that will surface "
            "competing and dead/dormant products.\n"
            "\n"
            "## Rules\n"
            "- You MUST output ONLY queries and domain keywords. You must "
            "NEVER name specific competitor products yourself — finding real "
            "products is the fetch step's job, not yours.\n"
            "- ALWAYS produce 3-5 queries: mix the core idea phrase with "
            "domain keywords and shutdown/graveyard angles "
            "(e.g. \"<domain> shutdown\", \"<domain> startup dead\").\n"
            "- Don't add commentary, don't fabricate URLs, don't write prose.\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": {\n'
            '    "queries": ["...", "...", "..."],\n'
            '    "domain_keywords": ["...", "..."],\n'
            '    "ambition_tier": "private_beta"\n'
            "  }\n"
            "}\n"
            "```\n"
            "Write the same JSON object to the produces path with write_file."
        )
```

- [ ] **Step 4: Register the agent**

In `src/agents/__init__.py`, add the import next to the other agent imports and add `"query_planner": QueryPlannerAgent(),` to `AGENT_REGISTRY`.

- [ ] **Step 5: Run tests + the global prompt-quality suite**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/agents/test_query_planner_prompt.py tests/agents/test_prompt_quality.py -p no:warnings -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agents/query_planner.py src/agents/__init__.py tests/agents/test_query_planner_prompt.py
git commit -m "feat(agents): query_planner (i2p 1.0a) — derives prior-art queries, never names products"
```

---

### Task 4: `prior_art_synthesizer` agent (1.0c)

**Files:**
- Create: `src/agents/prior_art_synthesizer.py`
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_prior_art_synthesizer_prompt.py`

- [ ] **Step 1: Write the failing test**

`tests/agents/test_prior_art_synthesizer_prompt.py`:
```python
from src.agents import get_agent, AGENT_REGISTRY


def test_synthesizer_registered():
    assert "prior_art_synthesizer" in AGENT_REGISTRY
    assert get_agent("prior_art_synthesizer").name == "prior_art_synthesizer"


def test_synthesizer_prompt_invariants():
    p = get_agent("prior_art_synthesizer").get_system_prompt({})
    assert p.lstrip().startswith("You are ")
    low = p.lower()
    assert ("must" in low or "always" in low)
    assert ("don't" in low or "never" in low)
    assert "final_answer" in p
    assert "```json" in p


def test_synthesizer_prompt_forbids_inventing_urls():
    p = get_agent("prior_art_synthesizer").get_system_prompt({}).lower()
    # The core anti-fabrication instruction must be present.
    assert "candidates" in p
    assert ("only" in p and "url" in p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/agents/test_prior_art_synthesizer_prompt.py -p no:warnings -q`
Expected: FAIL — not registered.

- [ ] **Step 3: Implement the agent**

`src/agents/prior_art_synthesizer.py`:
```python
"""Prior-art synthesizer agent (i2p step 1.0c).

Reads the fetched candidates artifact (prior_art_candidates.json) and the
idea brief, then judges which candidates are real attempted solutions,
extracts lessons, and sets a verdict — emitting prior_art_report.json.

Hard constraint: every attempted_solutions[i].url MUST be copied from a
fetched candidate. The synthesizer may NOT add products from its own
knowledge. The prior_art_min_coverage post-hook enforces this.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.prior_art_synthesizer")


class PriorArtSynthesizerAgent(BaseAgent):
    name = "prior_art_synthesizer"
    description = "Judges fetched prior-art candidates into a graveyard report"
    default_tier = "medium"
    min_tier = "cheap"
    max_iterations = 3
    allowed_tools = ["read_file", "write_file"]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a prior-art analyst. You judge a list of ALREADY-FETCHED "
            "candidate products into a structured graveyard report.\n"
            "\n"
            "## Rules\n"
            "- You MUST read the candidates artifact first. Every "
            "attempted_solutions entry's `url` and `name` MUST be copied "
            "verbatim from a fetched candidate. You may ONLY use products and "
            "URLs that appear in the candidates file.\n"
            "- You must NEVER invent a product, NEVER write a `url` that is "
            "not in the candidates, and NEVER set `url` to null. If a "
            "candidate has no URL, drop it.\n"
            "- ALWAYS set `verdict` to one of: graveyard_well_populated, "
            "graveyard_thin, blue_ocean_validated, blue_ocean_suspicious. "
            "Use blue_ocean_validated only with >=3 queries and >=20 "
            "results inspected (see search_summary).\n"
            "- ALWAYS extract at least one key_lesson when attempted_solutions "
            "is non-empty.\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": {\n'
            '    "_schema_version": "1",\n'
            '    "search_summary": { "queries_run": [], "sources_used": [], '
            '"total_results_inspected": 0 },\n'
            '    "attempted_solutions": [ {"name": "...", "url": "https://...", '
            '"status": "dead", "thesis_summary": "...", "sources": ["..."]} ],\n'
            '    "key_lessons": [ {"lesson": "...", "evidence_refs": ["..."]} ],\n'
            '    "verdict": "graveyard_thin"\n'
            "  }\n"
            "}\n"
            "```\n"
            "Write this JSON object to the produces path with write_file."
        )
```

- [ ] **Step 4: Register the agent**

In `src/agents/__init__.py`, import + add `"prior_art_synthesizer": PriorArtSynthesizerAgent(),` to `AGENT_REGISTRY`.

- [ ] **Step 5: Run tests + prompt-quality suite**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/agents/test_prior_art_synthesizer_prompt.py tests/agents/test_prompt_quality.py -p no:warnings -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agents/prior_art_synthesizer.py src/agents/__init__.py tests/agents/test_prior_art_synthesizer_prompt.py
git commit -m "feat(agents): prior_art_synthesizer (i2p 1.0c) — judges fetched candidates, URLs constrained to candidates"
```

---

### Task 5: Harden `prior_art_min_coverage` — URL ∈ candidates

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/prior_art_min_coverage.py`
- Test: `packages/mr_roboto/tests/test_prior_art_min_coverage_candidates.py`

- [ ] **Step 1: Write the failing test**

`packages/mr_roboto/tests/test_prior_art_min_coverage_candidates.py`:
```python
import json
from mr_roboto.prior_art_min_coverage import prior_art_min_coverage


def _report(url):
    return {
        "search_summary": {"queries_run": ["a", "b", "c"],
                           "total_results_inspected": 25},
        "attempted_solutions": [
            {"name": "Habitica", "url": url, "status": "dead",
             "sources": ["https://news.ycombinator.com/item?id=1"]}],
        "key_lessons": [{"lesson": "x"}],
        "verdict": "graveyard_thin",
    }


def test_url_not_in_candidates_is_rejected(tmp_path):
    cand = tmp_path / "candidates.json"
    cand.write_text(json.dumps({"candidates": [
        {"name": "RealApp", "url": "https://real.example"}]}), encoding="utf-8")
    out = prior_art_min_coverage(
        report=_report("https://habitica.com"),  # fabricated, not fetched
        candidates_path=str(cand))
    assert out["ok"] is False
    assert any("not in fetched candidates" in p for p in out["problems"])


def test_url_in_candidates_passes(tmp_path):
    cand = tmp_path / "candidates.json"
    cand.write_text(json.dumps({"candidates": [
        {"name": "Habitica", "url": "https://habitica.com"}]}), encoding="utf-8")
    out = prior_art_min_coverage(
        report=_report("https://habitica.com"),
        candidates_path=str(cand))
    assert out["ok"] is True, out["problems"]


def test_no_candidates_path_keeps_legacy_behavior():
    # Back-compat: without candidates_path the membership rule is skipped.
    out = prior_art_min_coverage(report=_report("https://habitica.com"))
    assert out["ok"] is True, out["problems"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest packages/mr_roboto/tests/test_prior_art_min_coverage_candidates.py -p no:warnings -q`
Expected: FAIL — `candidates_path` is an unexpected keyword arg.

- [ ] **Step 3: Implement**

In `prior_art_min_coverage.py`:

Add a loader helper near `_load_report`:
```python
def _load_candidate_urls(candidates_path: str | None) -> set[str] | None:
    """Return the set of fetched candidate URLs, or None if no path given.

    None means "membership rule disabled" (back-compat). An empty set means
    the candidates file existed but held no URLs.
    """
    if not candidates_path:
        return None
    rp = _resolve_report_path(candidates_path)
    if not rp or not os.path.isfile(rp):
        return None
    try:
        with open(rp, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    urls: set[str] = set()
    for c in (data.get("candidates") or []):
        if isinstance(c, dict) and isinstance(c.get("url"), str):
            urls.add(c["url"].strip())
    return urls
```

Change the signature:
```python
def prior_art_min_coverage(
    report: dict[str, Any] | None = None,
    report_path: str | None = None,
    candidates_path: str | None = None,
) -> dict[str, Any]:
```

After the existing Rule 2 loop (URL scheme validation), add Rule 2b:
```python
    # Rule 2b — anti-fabrication: every attempted URL must come from the
    # fetched candidates set. Disabled when candidates_path is absent.
    candidate_urls = _load_candidate_urls(candidates_path)
    if candidate_urls is not None:
        for i, sol in enumerate(attempted):
            if not isinstance(sol, dict):
                continue
            url = (sol.get("url") or "").strip()
            if url and url not in candidate_urls:
                problems.append(
                    f"attempted_solutions[{i}].url not in fetched candidates: "
                    f"{url!r} (name={sol.get('name')!r}) — synthesizer may "
                    f"only cite fetched candidates, not invented products"
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest packages/mr_roboto/tests/test_prior_art_min_coverage_candidates.py packages/mr_roboto/tests/test_prior_art_min_coverage.py -p no:warnings -q`
Expected: PASS (new 3 + existing suite green).

- [ ] **Step 5: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/prior_art_min_coverage.py packages/mr_roboto/tests/test_prior_art_min_coverage_candidates.py
git commit -m "feat(mr_roboto): prior_art_min_coverage rejects attempted URLs not in fetched candidates (anti-fabrication)"
```

---

### Task 6: Route `candidates_path` into the post-hook payload

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/apply.py` (the `prior_art_min_coverage` payload builder, ~line 2197)
- Test: `packages/general_beckman/tests/test_prior_art_posthook_candidates_path.py`

- [ ] **Step 1: Write the failing test**

`packages/general_beckman/tests/test_prior_art_posthook_candidates_path.py`:
```python
from general_beckman.apply import _posthook_agent_and_payload


class _A:
    kind = "prior_art_min_coverage"
    source_task_id = 42


def test_candidates_path_threaded_from_source_ctx():
    source = {"context": {
        "produces": ["mission_80/.research/prior_art_report.json"],
        "candidates_path": "mission_80/.research/prior_art_candidates.json",
    }}
    runner, payload = _posthook_agent_and_payload(_A(), source)
    assert runner == "mechanical"
    assert payload["payload"]["candidates_path"] == \
        "mission_80/.research/prior_art_candidates.json"
    assert payload["payload"]["report_path"] == \
        "mission_80/.research/prior_art_report.json"
```
(If `_posthook_agent_and_payload` is not the exact public name/signature, adapt the import to the real builder — confirm by grepping `def _posthook_agent_and_payload` in apply.py before writing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest packages/general_beckman/tests/test_prior_art_posthook_candidates_path.py -p no:warnings -q`
Expected: FAIL — `candidates_path` not in payload (KeyError).

- [ ] **Step 3: Implement**

In `apply.py`, the `if a.kind == "prior_art_min_coverage":` block (~2197), add `candidates_path` resolution + include it in the payload:
```python
    if a.kind == "prior_art_min_coverage":
        produces = list(source_ctx.get("produces") or [])
        report_path = (
            source_ctx.get("report_path")
            or (produces[0] if produces else None)
        )
        candidates_path = source_ctx.get("candidates_path")
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "prior_art_min_coverage",
            "executor": "mechanical",
            "payload": {
                "action": "prior_art_min_coverage",
                "report_path": report_path,
                "report": source_ctx.get("report"),
                "candidates_path": candidates_path,
            },
        })
```

Also extend the mr_roboto dispatch branch for `prior_art_min_coverage` (in `mr_roboto/__init__.py` `_run_dispatch`, near line 4001 of apply.py reference — the actual dispatch is in mr_roboto) to forward `candidates_path` to the verb. Find the existing `if action == "prior_art_min_coverage":` dispatch and add `candidates_path=payload.get("candidates_path")` to the call. (Confirm the dispatch site — grep `prior_art_min_coverage` in `mr_roboto/__init__.py`.)

- [ ] **Step 4: Run tests**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest packages/general_beckman/tests/test_prior_art_posthook_candidates_path.py -p no:warnings -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/apply.py packages/mr_roboto/src/mr_roboto/__init__.py packages/general_beckman/tests/test_prior_art_posthook_candidates_path.py
git commit -m "feat(beckman): thread candidates_path into prior_art_min_coverage post-hook payload"
```

---

### Task 7: Rewire `i2p_v3.json` — split step 1.0 into 1.0a/1.0b/1.0c

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`
- Test: `tests/i2p/test_prior_art_pipeline_steps.py`

- [ ] **Step 1: Write the failing test**

`tests/i2p/test_prior_art_pipeline_steps.py`:
```python
import json


def _steps():
    d = json.load(open("src/workflows/i2p/i2p_v3.json", encoding="utf-8"))
    out = {}
    def walk(o):
        if isinstance(o, dict):
            if o.get("id"):
                out[o["id"]] = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return out


def test_old_single_step_removed():
    s = _steps()
    assert "1.0" not in s
    assert {"1.0a", "1.0b", "1.0c"} <= set(s)


def test_pipeline_wiring():
    s = _steps()
    assert s["1.0a"]["depends_on"] == ["0.6"]
    assert s["1.0a"]["agent"] == "query_planner"
    assert s["1.0b"]["agent"] == "mechanical"
    assert s["1.0b"]["depends_on"] == ["1.0a"]
    assert s["1.0b"]["payload"]["action"] == "prior_art_fetch"
    assert s["1.0c"]["agent"] == "prior_art_synthesizer"
    assert s["1.0c"]["depends_on"] == ["1.0b"]
    # The report artifact + path + post-hook stay on the final step so
    # downstream consumers (1.13/1.14/2.1) are unchanged.
    assert "prior_art_report" in s["1.0c"]["output_artifacts"]
    assert s["1.0c"]["post_hooks"] == ["prior_art_min_coverage"]
    assert any("prior_art_report.json" in p for p in s["1.0c"]["produces"])
    # candidates_path must be on 1.0c context for the post-hook membership rule.
    assert "candidates_path" in s["1.0c"].get("context", {})


def test_json_still_valid():
    json.load(open("src/workflows/i2p/i2p_v3.json", encoding="utf-8"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/i2p/test_prior_art_pipeline_steps.py -p no:warnings -q`
Expected: FAIL — `1.0` still present, `1.0a` absent.

- [ ] **Step 3: Replace the step**

In `src/workflows/i2p/i2p_v3.json`, find the `"id": "1.0"` step object and replace it with these three step objects (preserve surrounding array commas):

```json
{
  "id": "1.0a",
  "phase": "phase_1",
  "name": "prior_art_query_plan",
  "agent": "query_planner",
  "difficulty": "easy",
  "depends_on": ["0.6"],
  "input_artifacts": ["idea_brief_final"],
  "output_artifacts": ["prior_art_queries"],
  "produces": ["mission_{mission_id}/.research/prior_art_queries.json"],
  "tools_hint": ["read_file", "write_file"],
  "instruction": "Read idea_brief_final. Emit 3-5 prior-art SEARCH QUERIES plus domain_keywords that will surface competing and dead/dormant products. Do NOT name specific products yourself. Write {\"queries\":[...],\"domain_keywords\":[...],\"ambition_tier\":context.ambition_tier or 'private_beta'} to the produces path.",
  "done_when": "prior_art_queries.json exists with >=3 queries.",
  "artifact_schema": {
    "prior_art_queries": {
      "type": "object",
      "required_fields": ["queries", "domain_keywords"]
    }
  },
  "context": {"estimated_output_tokens": 400},
  "reversibility": "full"
},
{
  "id": "1.0b",
  "phase": "phase_1",
  "name": "prior_art_fetch",
  "agent": "mechanical",
  "executor": "mechanical",
  "depends_on": ["1.0a"],
  "input_artifacts": ["prior_art_queries"],
  "output_artifacts": ["prior_art_candidates"],
  "produces": ["mission_{mission_id}/.research/prior_art_candidates.json"],
  "payload": {
    "action": "prior_art_fetch",
    "queries_path": "mission_{mission_id}/.research/prior_art_queries.json",
    "candidates_path": "mission_{mission_id}/.research/prior_art_candidates.json"
  },
  "done_when": "prior_art_candidates.json exists.",
  "reversibility": "full"
},
{
  "id": "1.0c",
  "phase": "phase_1",
  "name": "prior_art_synthesize",
  "agent": "prior_art_synthesizer",
  "difficulty": "medium",
  "depends_on": ["1.0b"],
  "input_artifacts": ["idea_brief_final", "prior_art_candidates"],
  "output_artifacts": ["prior_art_report"],
  "produces": ["mission_{mission_id}/.research/prior_art_report.json"],
  "post_hooks": ["prior_art_min_coverage"],
  "tools_hint": ["read_file", "write_file"],
  "instruction": "Read prior_art_candidates.json and idea_brief_final. Judge which fetched candidates are real attempted solutions. Every attempted_solutions url+name MUST be copied from a fetched candidate — never invent a product or URL. Extract >=1 key_lesson when attempted_solutions is non-empty. Set verdict. Write the prior_art_report JSON to the produces path verbatim.",
  "done_when": "prior_art_report exists at the produces path and prior_art_min_coverage post-hook returns ok=true.",
  "artifact_schema": {
    "prior_art_report": {
      "type": "object",
      "required_fields": ["search_summary", "attempted_solutions", "key_lessons", "verdict"],
      "_schema_version": "1"
    }
  },
  "context": {
    "estimated_output_tokens": 3000,
    "candidates_path": "mission_{mission_id}/.research/prior_art_candidates.json"
  },
  "reversibility": "full"
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/i2p/test_prior_art_pipeline_steps.py -p no:warnings -q`
Expected: PASS.

- [ ] **Step 5: Validate the whole workflow loads + no dangling `1.0` deps**

Run:
```
C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -c "import json; d=json.load(open('src/workflows/i2p/i2p_v3.json',encoding='utf-8')); ids=[]; \
walk=lambda o:[ids.append(o['id'])] if isinstance(o,dict) and o.get('id') else None; \
import collections; print('loaded ok')"
```
Then grep that nothing still `depends_on` the literal `"1.0"` (only `1.0a/b/c` are valid):
`grep -n '\"1.0\"' src/workflows/i2p/i2p_v3.json` → expect no `depends_on: ["1.0"]` matches (consumers use the `prior_art_report` artifact name, not the step id — verified: 1.13/1.14/2.1 reference the artifact).

- [ ] **Step 6: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json tests/i2p/test_prior_art_pipeline_steps.py
git commit -m "feat(i2p): split prior_art_search into 1.0a plan / 1.0b fetch / 1.0c synthesize"
```

---

### Task 8: Retire the `find_prior_art` LLM tool + vecihi module

**Files:**
- Delete: `packages/vecihi/src/vecihi/prior_art.py`
- Modify: `packages/vecihi/src/vecihi/__init__.py` (remove `find_prior_art` import + `__all__` entry)
- Delete: `src/tools/prior_art.py`
- Modify: `src/tools/__init__.py` (remove the `find_prior_art` optional-tool registration, lines ~94-108)
- Modify: `src/agents/researcher.py` (remove `"find_prior_art"` from `allowed_tools` + the prompt sentence referencing it)
- Delete/relocate: `packages/vecihi/tests/test_prior_art.py` → migrate any still-relevant fetch assertions into `tests/research/test_prior_art_fetch_candidates.py`; delete the report/verdict-specific tests (those behaviors moved to the LLM).
- Test: rely on import checks + existing suites.

- [ ] **Step 1: Remove the references**

Edit each file above. In `src/agents/researcher.py`, delete the `"find_prior_art",` line from `allowed_tools` and remove the prompt sentence "For prior-art / idea-validation tasks … use `find_prior_art` instead — …" (keep the surrounding web_search guidance).

- [ ] **Step 2: Verify imports + the old tool is gone**

Run:
```
C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -c "import src.tools; import src.agents; from src.agents import get_agent; a=get_agent('researcher'); assert 'find_prior_art' not in a.allowed_tools; import importlib; \
exec('try:\n import vecihi.prior_art\n raise SystemExit(\"vecihi.prior_art still importable\")\nexcept ModuleNotFoundError:\n print(\"vecihi.prior_art removed OK\")')"
```
Expected: prints `vecihi.prior_art removed OK`, no AssertionError.

- [ ] **Step 3: Run the vecihi + tools suites to catch dangling references**

Run (separately):
```
C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest packages/vecihi/tests -p no:warnings -q
C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/ -k "prior_art or tools" -p no:warnings -q --timeout=120
```
Expected: PASS (after migrating/deleting the obsolete `test_prior_art.py` cases).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: retire find_prior_art LLM tool + vecihi.prior_art module (replaced by 1.0a/b/c pipeline)"
```

---

### Task 9: Full regression + register-coverage

**Files:** none (verification only).

- [ ] **Step 1: Run the affected suites separately (conftest collision)**

```
C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest packages/mr_roboto/tests packages/general_beckman/tests -p no:warnings -q --timeout=180
C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/research tests/agents tests/i2p -p no:warnings -q --timeout=180
```
Expected: all green. Investigate any red before claiming done (verification-before-completion).

- [ ] **Step 2: Confirm `prior_art_fetch` is in the post-hook registry coverage test**

If `tests/i2p/test_post_hooks_registry_coverage.py` enumerates registered verbs, ensure `prior_art_fetch` (a plain dispatch verb, not a post-hook) does not need registry entry; only confirm `prior_art_min_coverage` still resolves. Run:
```
C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe -m pytest tests/i2p/test_post_hooks_registry_coverage.py -p no:warnings -q
```

- [ ] **Step 3: Final commit (if any test fixups were needed)**

```bash
git add -A
git commit -m "test: prior-art pipeline regression fixups"
```

---

## Self-Review

**Spec coverage:**
- Relocate out of vecihi → Tasks 1, 8. ✓
- 3-stage pipeline (queries / fetch / synthesize) → Tasks 3, 2, 4, 7. ✓
- Anti-fabrication (url ∈ candidates) → Tasks 5, 6. ✓
- Two focused agents → Tasks 3, 4. ✓
- Downstream consumers unchanged (artifact name `prior_art_report` + path preserved on 1.0c) → Task 7 test. ✓

**Type/name consistency:**
- New module fn `fetch_candidates(queries, domain_keywords, k, ambition_tier, *, db_path, session, ttl_hours)` — same name used in Tasks 1, 2.
- Verb `prior_art_fetch(queries_path, candidates_path, *, db_path)` — same in Tasks 2, 7.
- Agents `query_planner`, `prior_art_synthesizer` — same in Tasks 3, 4, 7.
- Post-hook param `candidates_path` — same in Tasks 5, 6, 7.

**Open verification (do at execution, not guessed):**
- Task 6: confirm the exact name/signature of the post-hook payload builder (`_posthook_agent_and_payload`) and the mr_roboto `prior_art_min_coverage` dispatch site before editing. Grep both first.
- Task 4/3: `tests/agents/test_prompt_quality.py` enumerates `AGENT_REGISTRY` — both new agents must pass the 3 invariants (the prompts above are written to satisfy them; run the suite to confirm).

## Live-system note

None of this is live until the founder restarts KutAI. mission_80 is already DLQ'd at the old step 1.0 — do NOT keep retrying #80 against the old workflow. Validate on a FRESH mission after restart (per the 2026-06-01 handoff: poisoned missions hit dead producers).
