# SP4b Plan 1 — reviews CPS (classify + draft_reply)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the LLM out of `reviews_classify` and `reviews_draft_reply` so each LLM hop is an admitted Beckman task on the pump (CPS producer → durable continuation → mechanical sink), leaving mr_roboto with only the mechanical sink.

**Architecture:** A producer module *outside* mr_roboto loads the review, builds the verb-specific prompt + `raw_dispatch` overhead spec, and `enqueue(...on_complete=, on_error=, cont_state=)` (no `await_inline`). The LLM child runs on `lane=oneshot`. On completion a registered continuation handler (the mechanical *sink*, living in `mr_roboto.executors`) parses the output, persists/validates, routes side-effects, and enforces the never-auto-post contract. The `on_error` path runs the same sink with the heuristic/canned fallback.

**Tech Stack:** Python 3.10 async, aiosqlite, general_beckman continuations (`register_resume` / `_HANDLER_MODULES`), pytest (sequential, `timeout`-prefixed).

**Scope:** This is Plan 1 of 2 for SP4b. Plan 2 (workflow splits: demo / incident / press_kit / crisis) is a separate plan — distinct subsystem (workflow engine). Spec: `docs/superpowers/specs/2026-06-05-cps-sp4b-design.md` §4 (reviews rows) + §5.3 + §7.

---

## File Structure

- **Create** `src/reviews/__init__.py` — empty package marker.
- **Create** `src/reviews/producers.py` — `enqueue_classify(...)`, `enqueue_draft_reply(...)`: review-load + prompt-build + raw_dispatch spec + `enqueue` with continuation. Holds the prompts (`_CLASSIFY_PROMPT`, `_PLATFORM_CONVENTIONS`) moved out of mr_roboto.
- **Create** `packages/mr_roboto/src/mr_roboto/executors/reviews_continuations.py` — the mechanical sinks `_classify_resume` / `_classify_resume_err` / `_draft_reply_resume` / `_draft_reply_resume_err` + `register_continuations()`.
- **Modify** `packages/general_beckman/src/general_beckman/continuations.py:175` — append `"mr_roboto.executors.reviews_continuations"` to `_HANDLER_MODULES`.
- **Modify** `packages/mr_roboto/src/mr_roboto/reviews_classify.py` — delete `_call_llm_classify`; keep `_heuristic_classify`, `_parse_llm_response`, `_emit_low_star_founder_action`, `_enqueue_bug_investigation`, `VALID_*`, `LOW_STAR_THRESHOLD` (re-exported, consumed by the sink). `run()` becomes a thin shim that delegates to the producer (kept for any direct callers) OR is removed if no callers remain — Task 6 decides.
- **Modify** `packages/mr_roboto/src/mr_roboto/reviews_draft_reply.py` — delete `_call_llm_draft_reply`; keep `_fallback_draft` (consumed by sink). `_PLATFORM_CONVENTIONS` moves to the producer.
- **Modify** `packages/mr_roboto/src/mr_roboto/__init__.py:4642` and `:4653` — router branches enqueue the producer and return `Action(completed, {enqueued: task_id})`.
- **Modify** `src/app/jobs/reviews_poll_daily.py:113-128` — Phase 2 loop enqueues the producer per review instead of `await classify_run(...)`; count `total_enqueued`.
- **Create** `tests/reviews/test_reviews_cps.py` — the full slice (producer enqueues correctly; sink persists/validates/side-effects; never-auto-post; fallback on err).

**Sink handler signature (canonical, from `posthook_continuations.py`):** `async def handler(child_task_id: int, result: dict, state: dict) -> None`. Read LLM text with the dual-shape decode helper (copy `_extract_content` — see Task 2).

**cont_state contract (classify):** `{review_id:int, product_id:str, platform:str, author:str, rating:int, body_md:str}`.
**cont_state contract (draft_reply):** `{review_id:int, product_id:str, platform:str, author:str, rating:int}`.

---

## Task 1: producer package + classify producer

**Files:**
- Create: `src/reviews/__init__.py`
- Create: `src/reviews/producers.py`
- Test: `tests/reviews/test_reviews_cps.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/reviews/test_reviews_cps.py
import json
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_enqueue_classify_builds_overhead_child_with_continuation():
    from src.reviews import producers

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return 4242

    # Producer loads the review row itself.
    fake_row = (7, "g2", "Ada", 1, "It crashes on save")

    class _Cur:
        async def fetchone(self):
            return fake_row

    class _DB:
        async def execute(self, *a, **k):
            return _Cur()

    async def fake_get_db():
        return _DB()

    with patch.object(producers, "enqueue", fake_enqueue), \
         patch("src.infra.db.get_db", fake_get_db):
        tid = await producers.enqueue_classify(review_id=7, product_id="prod-x")

    assert tid == 4242
    k = captured["kwargs"]
    assert k["on_complete"] == "reviews.classify.resume"
    assert k["on_error"] == "reviews.classify.resume_err"
    assert k["lane"] == "oneshot"
    st = k["cont_state"]
    assert st["review_id"] == 7 and st["rating"] == 1 and st["body_md"]
    llm = captured["spec"]["context"]["llm_call"]
    assert llm["raw_dispatch"] is True and llm["call_category"] == "overhead"
    assert "crashes" in llm["messages"][0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py::test_enqueue_classify_builds_overhead_child_with_continuation -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.reviews'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/reviews/__init__.py
```
(empty file)

```python
# src/reviews/producers.py
"""SP4b — reviews CPS producers (LLM extracted out of mr_roboto).

Each function loads the review, builds the verb-specific prompt + a
raw_dispatch OVERHEAD spec, and enqueues it as an admitted pump task with a
durable continuation (on_complete -> mechanical sink). NO await_inline.
The prompts live HERE (outside mr_roboto), never in the mechanical verb.
"""
from __future__ import annotations

import time
import uuid

from general_beckman import enqueue  # re-exported below for test patching
from general_beckman.lanes import LANE_ONESHOT
from src.infra.logging_config import get_logger

logger = get_logger("reviews.producers")

_CLASSIFY_PROMPT = (
    "You are classifying a product review. Return JSON only.\n\n"
    "Rating: {rating}/5\n"
    "Review: {body}\n\n"
    'Respond ONLY with: {{"sentiment": "<positive|negative|neutral>", '
    '"theme_tag": "<UX|pricing|bug|feature-request|support|generic-positive|generic-negative>"}}\n'
    "Pick the single most relevant theme_tag."
)


async def _load_review(review_id: int):
    """Return (review_id, platform, author, rating, body_md) or None."""
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT review_id, platform, author, rating, body_md "
        "FROM external_reviews WHERE review_id=?",
        (review_id,),
    )
    return await cur.fetchone()


def _suffix() -> str:
    return f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"


async def enqueue_classify(*, review_id: int, product_id: str) -> int | None:
    row = await _load_review(review_id)
    if row is None:
        logger.warning("enqueue_classify: review_id=%s not found", review_id)
        return None
    _, platform, author, rating, body_md = row
    rating = int(rating or 0)
    body_md = body_md or ""

    prompt = _CLASSIFY_PROMPT.format(rating=rating, body=body_md[:800])
    spec = {
        "title": f"reviews_classify:llm:{_suffix()}",
        "description": "Classify review sentiment + theme.",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 2,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": "reviewer",
            "difficulty": 3,
            "messages": [{"role": "user", "content": prompt}],
            "failures": [],
            "estimated_input_tokens": 250,
            "estimated_output_tokens": 50,
        }},
    }
    return await enqueue(
        spec,
        lane=LANE_ONESHOT,
        on_complete="reviews.classify.resume",
        on_error="reviews.classify.resume_err",
        cont_state={
            "review_id": review_id, "product_id": product_id,
            "platform": platform, "author": author or "Unknown",
            "rating": rating, "body_md": body_md,
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py::test_enqueue_classify_builds_overhead_child_with_continuation -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/reviews/ tests/reviews/test_reviews_cps.py
rtk git commit -m "feat(reviews): classify CPS producer — LLM child on pump w/ continuation"
```

---

## Task 2: classify sink (mechanical continuation handler) + fallback

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/executors/reviews_continuations.py`
- Test: `tests/reviews/test_reviews_cps.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/reviews/test_reviews_cps.py

@pytest.mark.asyncio
async def test_classify_resume_persists_and_routes_bug_sideeffect():
    from mr_roboto.executors import reviews_continuations as rc

    updates = []

    class _DB:
        async def execute(self, sql, params=()):
            updates.append((sql, params)); return None
        async def commit(self):
            return None

    async def fake_get_db():
        return _DB()

    emitted = {}
    async def fake_low_star(**kw):
        emitted["low_star"] = kw
    bug = {}
    async def fake_bug(spec, **kw):
        bug["spec"] = spec; return 1

    state = {"review_id": 7, "product_id": "p", "platform": "g2",
             "author": "Ada", "rating": 1, "body_md": "it crashes"}
    result = {"result": {"content": '{"sentiment":"negative","theme_tag":"bug"}'}}

    with patch("src.infra.db.get_db", fake_get_db), \
         patch.object(rc, "_emit_low_star_founder_action", fake_low_star), \
         patch.object(rc, "_enqueue_bug_investigation", fake_bug):
        await rc._classify_resume(99, result, state)

    assert any("UPDATE external_reviews" in s for s, _ in updates)
    assert emitted["low_star"]["theme_tag"] == "bug"     # 1-star -> founder action
    assert bug["spec"]["title"].startswith("[BUG]")      # bug theme -> investigation


@pytest.mark.asyncio
async def test_classify_resume_err_uses_heuristic():
    from mr_roboto.executors import reviews_continuations as rc

    updates = []
    class _DB:
        async def execute(self, sql, params=()):
            updates.append(params); return None
        async def commit(self): return None
    async def fake_get_db(): return _DB()

    state = {"review_id": 7, "product_id": "p", "platform": "g2",
             "author": "Ada", "rating": 5, "body_md": "love the UX"}
    with patch("src.infra.db.get_db", fake_get_db), \
         patch.object(rc, "_emit_low_star_founder_action", AsyncMock()), \
         patch.object(rc, "_enqueue_bug_investigation", AsyncMock()):
        await rc._classify_resume_err(99, {"error": "no candidates"}, state)

    # heuristic: rating 5 -> positive
    assert any(p[0] == "positive" for p in updates)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py -k classify_resume -v`
Expected: FAIL — `ModuleNotFoundError: mr_roboto.executors.reviews_continuations`.

- [ ] **Step 3: Write minimal implementation**

```python
# packages/mr_roboto/src/mr_roboto/executors/reviews_continuations.py
"""SP4b — reviews CPS mechanical sinks (continuation handlers).

These are MECHANICAL: no LLM call, no dispatcher import. They receive the
already-produced LLM output (or fire on_error with a heuristic fallback),
validate/persist, route side-effects, and enforce the never-auto-post
contract. Registered in general_beckman.continuations._HANDLER_MODULES.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

# Mechanical helpers reused from the (now LLM-free) verb modules.
from mr_roboto.reviews_classify import (
    VALID_SENTIMENTS, VALID_THEMES, LOW_STAR_THRESHOLD,
    _heuristic_classify, _parse_llm_response,
    _emit_low_star_founder_action, _enqueue_bug_investigation,
)
from mr_roboto.reviews_draft_reply import _fallback_draft

logger = get_logger("mr_roboto.executors.reviews_continuations")


def _extract_content(result: dict) -> str:
    """Dual-shape decode (normal terminal vs restart-reconcile)."""
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = inner.get("content", "")
    elif inner is not None:
        content = inner
    else:
        content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


async def _persist_classification(review_id, sentiment, theme_tag) -> None:
    from src.infra.db import get_db
    db = await get_db()
    await db.execute(
        "UPDATE external_reviews SET sentiment=?, theme_tag=? WHERE review_id=?",
        (sentiment, theme_tag, review_id),
    )
    await db.commit()


async def _route_classify_sideeffects(state: dict, sentiment: str, theme_tag: str) -> None:
    review_id = state["review_id"]; rating = int(state.get("rating") or 0)
    if rating <= LOW_STAR_THRESHOLD:
        await _emit_low_star_founder_action(
            review_id=review_id, platform=state.get("platform") or "",
            author=state.get("author") or "Unknown", rating=rating,
            body_md=state.get("body_md") or "", product_id=state.get("product_id") or "",
            theme_tag=theme_tag,
        )
    if theme_tag == "bug":
        await _enqueue_bug_investigation({
            "title": f"[BUG] Investigate report from {state.get('platform')} review",
            "description": (
                f"Review on {state.get('platform')} by {state.get('author')!r} "
                f"classified as bug. Body: {(state.get('body_md') or '')[:200]}..."
            ),
            "agent_type": "mechanical", "kind": "overhead",
            "context": {"review_id": review_id, "platform": state.get("platform"),
                        "product_id": state.get("product_id"),
                        "body_md": (state.get("body_md") or "")[:500]},
        })


async def _apply_classify(state: dict, sentiment: str, theme_tag: str) -> None:
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"
    if theme_tag not in VALID_THEMES:
        theme_tag = "generic-negative"
    await _persist_classification(state["review_id"], sentiment, theme_tag)
    await _route_classify_sideeffects(state, sentiment, theme_tag)


async def _classify_resume(child_task_id: int, result: dict, state: dict) -> None:
    raw = _extract_content(result).strip()
    c = _parse_llm_response(raw, state.get("body_md") or "", int(state.get("rating") or 0))
    await _apply_classify(state, c.get("sentiment", "neutral"), c.get("theme_tag", "generic-negative"))


async def _classify_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    logger.warning("reviews classify child failed (%s) — heuristic fallback",
                   (result or {}).get("error"))
    c = _heuristic_classify(state.get("body_md") or "", int(state.get("rating") or 0))
    await _apply_classify(state, c["sentiment"], c["theme_tag"])


def register_continuations() -> None:
    """Register reviews CPS sinks. Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("reviews.classify.resume", _classify_resume)
        register_resume("reviews.classify.resume_err", _classify_resume_err)
        register_resume("reviews.draft_reply.resume", _draft_reply_resume)
        register_resume("reviews.draft_reply.resume_err", _draft_reply_resume_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("reviews continuation registration deferred: %s", exc)


# draft_reply sinks land in Task 3; define stubs so register_continuations
# imports cleanly until then.
async def _draft_reply_resume(child_task_id: int, result: dict, state: dict) -> None:  # noqa: D401
    raise NotImplementedError  # Task 3


async def _draft_reply_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    raise NotImplementedError  # Task 3


register_continuations()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py -k classify_resume -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/reviews_continuations.py tests/reviews/test_reviews_cps.py
rtk git commit -m "feat(reviews): classify mechanical sink + heuristic on_error fallback"
```

---

## Task 3: draft_reply producer + sink (never-auto-post)

**Files:**
- Modify: `src/reviews/producers.py`
- Modify: `packages/mr_roboto/src/mr_roboto/executors/reviews_continuations.py`
- Test: `tests/reviews/test_reviews_cps.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/reviews/test_reviews_cps.py

@pytest.mark.asyncio
async def test_enqueue_draft_reply_builds_child():
    from src.reviews import producers
    cap = {}
    async def fake_enqueue(spec, **kw):
        cap["spec"] = spec; cap["kw"] = kw; return 555
    fake_row = (7, "appstore", "Bo", 2, "slow and buggy")
    class _Cur:
        async def fetchone(self): return fake_row
    class _DB:
        async def execute(self, *a, **k): return _Cur()
    async def fake_get_db(): return _DB()
    with patch.object(producers, "enqueue", fake_enqueue), \
         patch("src.infra.db.get_db", fake_get_db):
        tid = await producers.enqueue_draft_reply(review_id=7, product_id="p")
    assert tid == 555
    assert cap["kw"]["on_complete"] == "reviews.draft_reply.resume"
    assert "appstore" in cap["spec"]["context"]["llm_call"]["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_draft_reply_resume_surfaces_draft_never_autoposts():
    from mr_roboto.executors import reviews_continuations as rc
    fa = {}
    async def fake_fa(**kw):
        fa.update(kw)
        class _R:  # noqa
            id = 12
        return _R()
    state = {"review_id": 7, "product_id": "p", "platform": "g2",
             "author": "Ada", "rating": 5}
    result = {"result": {"content": "Thanks so much for the kind words!"}}
    writes = []
    class _DB:
        async def execute(self, sql, params=()):
            writes.append(sql); return None
        async def commit(self): return None
    async def fake_get_db(): return _DB()
    with patch("src.founder_actions.create", fake_fa), \
         patch("src.infra.db.get_db", fake_get_db):
        await rc._draft_reply_resume(99, result, state)
    assert "Thanks so much" in str(fa)            # draft surfaced to founder
    # never-auto-post: NO write to replied_at / reply_body_md
    assert not any("replied_at" in w or "reply_body_md" in w for w in writes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py -k draft_reply -v`
Expected: FAIL — `enqueue_draft_reply` missing / `NotImplementedError`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/reviews/producers.py`:

```python
_PLATFORM_CONVENTIONS = {
    "g2": "G2 replies are read by B2B buyers. Be professional, specific, acknowledge feedback. Offer to connect privately for issues.",
    "appstore": "AppStore replies are brief (<=1000 chars). Be warm, thank the user; for negatives offer a specific resolution path (email/support).",
    "playstore": "PlayStore replies are brief. Acknowledge feedback, be empathetic, direct to support for technical issues.",
    "producthunt": "ProductHunt replies are community-facing. Authentic, founder-to-user; for criticism acknowledge openly and describe what's next.",
    "trustpilot": "Trustpilot replies are public and formal. Use company name, address the complaint, offer a resolution path.",
    "capterra": "Capterra is B2B. Professional, acknowledge the point, highlight roadmap/workarounds for feature requests.",
}
_DEFAULT_CONVENTION = (
    "Be polite, genuine, specific to the review. For negatives acknowledge the issue "
    "and offer a resolution path; for positives thank the reviewer warmly."
)


async def enqueue_draft_reply(*, review_id: int, product_id: str) -> int | None:
    row = await _load_review(review_id)
    if row is None:
        logger.warning("enqueue_draft_reply: review_id=%s not found", review_id)
        return None
    _, platform, author, rating, body_md = row
    rating = int(rating or 3); author = author or "Anonymous"; body_md = body_md or ""
    convention = _PLATFORM_CONVENTIONS.get(platform, _DEFAULT_CONVENTION)
    star = f"{rating}/5 star{'s' if rating != 1 else ''}"
    prompt = (
        f"You are drafting a reply to a {star} review on {platform}.\n"
        f"Reviewer: {author}\nReview content: {body_md[:600]}\n\n"
        f"Platform conventions:\n{convention}\n\n"
        "Write a reply in first person from the product founder's perspective.\n"
        "Rules:\n- No promises about specific features/timelines.\n"
        "- No refunds unless the review mentions billing.\n"
        "- Concise: 2-4 sentences positive, 3-6 negative.\n"
        "- Do NOT start with 'Hi,' or 'Dear,'\nDraft the reply only."
    )
    spec = {
        "title": f"reviews_draft_reply:llm:{_suffix()}",
        "description": f"Draft reply for {platform} review.",
        "agent_type": "reviewer", "kind": "overhead", "priority": 2,
        "context": {"llm_call": {
            "raw_dispatch": True, "call_category": "overhead",
            "task": "reviewer", "agent_type": "reviewer", "difficulty": 3,
            "messages": [{"role": "user", "content": prompt}], "failures": [],
            "estimated_input_tokens": 350, "estimated_output_tokens": 150,
        }},
    }
    return await enqueue(
        spec, lane=LANE_ONESHOT,
        on_complete="reviews.draft_reply.resume",
        on_error="reviews.draft_reply.resume_err",
        cont_state={"review_id": review_id, "product_id": product_id,
                    "platform": platform, "author": author, "rating": rating},
    )
```

Replace the Task-2 stubs in `reviews_continuations.py` with:

```python
async def _surface_draft(state: dict, draft: str) -> None:
    """Mechanical: surface the draft to the founder. NEVER auto-posts —
    replied_at / reply_body_md stay NULL until the founder manually confirms."""
    from src.founder_actions import create as fa_create
    review_id = state["review_id"]; platform = state.get("platform") or ""
    await fa_create(
        mission_id=None, kind="generic",
        title=f"Draft reply ready for {platform} review (id={review_id}) — review before posting",
        why=("A reply draft was generated. NEVER auto-posted — review/edit, then "
             "post manually via the platform. Mark done when sent."),
        instructions=[f"Draft:\n\n{draft[:1000]}",
                      "Edit if needed, then post manually on the platform.",
                      "NEVER reply automatically — this is a draft only."],
        expected_output_kind="ack_only", notify_telegram=True,
    )


async def _draft_reply_resume(child_task_id: int, result: dict, state: dict) -> None:
    draft = _extract_content(result).strip()
    if not draft:
        draft = _fallback_draft(state.get("platform") or "", state.get("author") or "Anonymous",
                                int(state.get("rating") or 3))
    await _surface_draft(state, draft)


async def _draft_reply_resume_err(child_task_id: int, result: dict, state: dict) -> None:
    logger.warning("reviews draft_reply child failed (%s) — fallback draft",
                   (result or {}).get("error"))
    draft = _fallback_draft(state.get("platform") or "", state.get("author") or "Anonymous",
                            int(state.get("rating") or 3))
    await _surface_draft(state, draft)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py -k draft_reply -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
rtk git add src/reviews/producers.py packages/mr_roboto/src/mr_roboto/executors/reviews_continuations.py tests/reviews/test_reviews_cps.py
rtk git commit -m "feat(reviews): draft_reply CPS producer + sink (surfaces draft, never auto-posts)"
```

---

## Task 4: register the sink module for restart-recovery

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/continuations.py:175-186`
- Test: `tests/reviews/test_reviews_cps.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/reviews/test_reviews_cps.py

def test_reviews_continuations_in_handler_modules():
    from general_beckman import continuations as c
    assert "mr_roboto.executors.reviews_continuations" in c._HANDLER_MODULES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py::test_reviews_continuations_in_handler_modules -v`
Expected: FAIL — not in list.

- [ ] **Step 3: Write minimal implementation**

In `continuations.py` `_HANDLER_MODULES`, append after the SP3 entry:

```python
    # CPS SP3 - in-task deadlock set:
    "general_beckman.posthook_continuations",
    # CPS SP4b - reviews CPS mechanical sinks:
    "mr_roboto.executors.reviews_continuations",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py::test_reviews_continuations_in_handler_modules -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/continuations.py
rtk git commit -m "feat(reviews): register reviews CPS sinks in _HANDLER_MODULES (restart-recovery)"
```

---

## Task 5: rewire the two router branches (async enqueue, no inline LLM)

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py:4642-4663`
- Test: `tests/reviews/test_reviews_cps.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/reviews/test_reviews_cps.py

@pytest.mark.asyncio
async def test_router_classify_enqueues_producer():
    import mr_roboto
    with patch("src.reviews.producers.enqueue_classify",
               AsyncMock(return_value=4321)) as m:
        act = await mr_roboto.run({"id": 1, "context": {}},
                                  action="reviews/classify",
                                  payload={"review_id": 7, "product_id": "p"})
    m.assert_awaited_once()
    assert act.status == "completed"
    assert act.result.get("enqueued") == 4321
```

(Confirm `mr_roboto.run(...)` dispatch signature against `__init__.py` — adapt the call to the real entrypoint if it differs; the assertion on `enqueue_classify` + `enqueued` result is the contract.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py::test_router_classify_enqueues_producer -v`
Expected: FAIL — branch still calls `reviews_classify.run` (no `enqueued`).

- [ ] **Step 3: Write minimal implementation**

Replace the `reviews/classify` branch (`__init__.py:4642`):

```python
    if action == "reviews/classify":
        # SP4b: LLM extracted -> admitted producer task on the pump; the
        # reviews.classify.resume continuation does the mechanical persist.
        try:
            from src.reviews.producers import enqueue_classify
            tid = await enqueue_classify(
                review_id=int(payload.get("review_id")),
                product_id=str(payload.get("product_id") or ""),
            )
            if tid is None:
                return Action(status="failed", error="reviews/classify: review not found")
            return Action(status="completed", result={"enqueued": tid})
        except Exception as e:
            return Action(status="failed", error=str(e))
```

Replace the `reviews/draft_reply` branch (`__init__.py:4653`):

```python
    if action == "reviews/draft_reply":
        # SP4b: LLM extracted -> producer task; reviews.draft_reply.resume
        # surfaces the draft via founder_action. NEVER auto-posts.
        try:
            from src.reviews.producers import enqueue_draft_reply
            tid = await enqueue_draft_reply(
                review_id=int(payload.get("review_id")),
                product_id=str(payload.get("product_id") or ""),
            )
            if tid is None:
                return Action(status="failed", error="reviews/draft_reply: review not found")
            return Action(status="completed", result={"enqueued": tid, "auto_posted": False})
        except Exception as e:
            return Action(status="failed", error=str(e))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py::test_router_classify_enqueues_producer -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/__init__.py tests/reviews/test_reviews_cps.py
rtk git commit -m "refactor(reviews): router branches enqueue CPS producers (no inline LLM)"
```

---

## Task 6: strip the LLM from the verb modules + rewire the cron

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/reviews_classify.py` (delete `_call_llm_classify`)
- Modify: `packages/mr_roboto/src/mr_roboto/reviews_draft_reply.py` (delete `_call_llm_draft_reply`)
- Modify: `src/app/jobs/reviews_poll_daily.py:101-128`
- Test: `tests/reviews/test_reviews_cps.py`, existing `tests/z7/test_b8_reviews_harvest.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/reviews/test_reviews_cps.py

def test_no_llm_left_in_verb_modules():
    import inspect, mr_roboto.reviews_classify as rc, mr_roboto.reviews_draft_reply as rd
    assert not hasattr(rc, "_call_llm_classify")
    assert not hasattr(rd, "_call_llm_draft_reply")
    # neither module imports the dispatcher/husam or enqueues an inline LLM
    for mod in (rc, rd):
        src = inspect.getsource(mod)
        assert "await_inline=True" not in src
        assert "LLMDispatcher" not in src


@pytest.mark.asyncio
async def test_cron_enqueues_producer_per_review():
    from src.app.jobs import reviews_poll_daily as job
    calls = []
    async def fake_enq(*, review_id, product_id):
        calls.append((review_id, product_id)); return 1
    # one unclassified review
    class _Cur:
        async def fetchall(self): return [(7, "p")]
    class _DB:
        async def execute(self, *a, **k): return _Cur()
    async def fake_get_db(): return _DB()
    with patch("src.reviews.producers.enqueue_classify", fake_enq), \
         patch("src.infra.db.get_db", fake_get_db):
        res = await job.run_reviews_poll_daily({"products": []})
    assert (7, "p") in calls
    assert res["total_enqueued"] == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py -k "no_llm_left or cron_enqueues" -v`
Expected: FAIL — `_call_llm_*` still present; cron has no `total_enqueued`.

- [ ] **Step 3: Implementation**

In `reviews_classify.py`: delete `_call_llm_classify` entirely and drop it from `__all__`. Keep `_parse_llm_response`, `_heuristic_classify`, side-effect helpers, `VALID_*`, `LOW_STAR_THRESHOLD`. Change `run()` to delegate (kept for any direct/legacy caller) — replace its `_call_llm_classify` use:

```python
async def run(payload: dict) -> dict:
    """DEPRECATED direct entry — enqueues the CPS producer and returns its id.
    Kept for legacy callers; new callers use the router action / cron path."""
    review_id = payload.get("review_id"); product_id = str(payload.get("product_id") or "")
    if review_id is None:
        return {"status": "error", "error": "review_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}
    from src.reviews.producers import enqueue_classify
    tid = await enqueue_classify(review_id=int(review_id), product_id=product_id)
    if tid is None:
        return {"status": "error", "error": f"review_id={review_id} not found"}
    return {"status": "ok", "enqueued": tid}
```

In `reviews_draft_reply.py`: delete `_call_llm_draft_reply`, drop from `__all__`; keep `_fallback_draft`. Rewrite `run()` analogously to delegate to `enqueue_draft_reply` and return `{"status":"ok","enqueued":tid,"auto_posted":False}`. (`_PLATFORM_CONVENTIONS`/`_DEFAULT_CONVENTION` now live in the producer; remove the now-unused copies.)

In `reviews_poll_daily.py` Phase 2 loop (replace the `classify_run` call):

```python
    # Phase 2: enqueue a CPS classify producer per unclassified review.
    total_enqueued = 0
    from src.reviews.producers import enqueue_classify
    for row in unclassified:
        review_id, product_id = row[0], row[1]
        try:
            tid = await enqueue_classify(review_id=review_id, product_id=product_id)
            if tid:
                total_enqueued += 1
        except Exception as exc:
            errors.append(f"classify review_id={review_id}: {exc}")
            logger.error("reviews_poll_daily: classify enqueue failed id=%d: %s", review_id, exc)
```

Update the return dict: replace `"total_classified": total_classified` with `"total_enqueued": total_enqueued` (and drop the now-unused `classify_run` import + `total_classified`). Update the closing `logger.info` accordingly.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/reviews/test_reviews_cps.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/reviews_classify.py packages/mr_roboto/src/mr_roboto/reviews_draft_reply.py src/app/jobs/reviews_poll_daily.py tests/reviews/test_reviews_cps.py
rtk git commit -m "refactor(reviews): delete inline LLM from verbs; cron enqueues CPS producer"
```

---

## Task 7: regression sweep + fix fallout

**Files:** existing reviews tests.

- [ ] **Step 1: Run the existing reviews suite**

Run: `.venv/Scripts/python.exe -m pytest tests/z7/test_b8_reviews_harvest.py -v`
Expected: failures where tests monkeypatched `_call_llm_classify` / `_call_llm_draft_reply` or asserted synchronous `reply_draft`/`sentiment` in the verb result.

- [ ] **Step 2: Update those tests** to the CPS contract: patch `src.reviews.producers.enqueue_classify` / `enqueue_draft_reply` (or the sink), assert `enqueued`/async behavior, and exercise the sink directly for persistence/side-effects. Show the diff per test as you go (no blanket skips).

- [ ] **Step 3: Full mr_roboto + beckman + reviews suites, SEQUENTIALLY (conftest-collision landmine)**

Run each in its OWN invocation (never together):
```
.venv/Scripts/python.exe -m pytest tests/reviews/ -q
.venv/Scripts/python.exe -m pytest tests/z7/test_b8_reviews_harvest.py -q
.venv/Scripts/python.exe -m pytest packages/mr_roboto/tests -q
.venv/Scripts/python.exe -m pytest packages/general_beckman/tests -q
```
Expected: all green.

- [ ] **Step 4: Commit any test fixes**

```bash
rtk git add tests/
rtk git commit -m "test(reviews): update legacy reviews tests to the CPS contract"
```

---

## Self-review (done by plan author)

- **Spec coverage:** §4 reviews rows (classify+draft_reply → CPS) = Tasks 1-6; §5.3 (cron drives, bug fire-and-forget untouched) = Task 6; §6 fork #2 (prompt in producer outside mr_roboto) = Tasks 1/3; §7 fallback (sink-owned, on_error) = Tasks 2/3; §9 landmines (lane=oneshot = Tasks 1/3; `_HANDLER_MODULES` = Task 4; sequential pytest = Task 7). Covered.
- **Placeholder scan:** none — all code shown. The one adaptation note (Task 5 `mr_roboto.run` signature) is a verify-against-source instruction with the contract pinned, not a placeholder.
- **Type/name consistency:** handler names `reviews.classify.resume[/_err]`, `reviews.draft_reply.resume[/_err]` consistent across producer `on_complete/on_error` (Tasks 1,3), registration (Task 2), `_HANDLER_MODULES` (Task 4). cont_state keys consistent producer↔sink. `enqueue` patched at `src.reviews.producers.enqueue` (imported name) consistently.

## Open follow-ups (Plan 2 — workflow splits)

demo_storyboard, incident_draft_update, press_kit_assemble, crisis_draft_holding → separate plan; needs workflow-engine investigation (agent-step prompt build, expander fallback/degraded-emit, workflow launch from `/`-command/router, fan-in artifact passing). Spec §4-5 + §12 carry the design.
