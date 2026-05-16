# Implementation Plan — Yalayut Phase 4: Discovery + Autonomy

> **For agentic workers.** Execute tasks in order. Each task is a closed unit:
> failing test → run (expect FAIL) → minimal implementation → run (expect PASS)
> → commit. Do not skip the FAIL run — it proves the test exercises new code.
> All code blocks are complete and runnable; copy them verbatim. Use the exact
> file paths given. Never leave a `TODO` or placeholder. Run pytest with a
> timeout prefix every time.

## Goal

Add autonomous catalog growth and the founder-facing operations surface to the
yalayut subsystem. Phases 1–3 (catalog core, the `intersect` match layer,
`run_recipe` + mr_roboto preempt + cookiecutter/public_apis adapters + MCP
lifecycle) are **complete and working**. Phase 4 makes the catalog grow itself:

- a daily discovery cron that pulls trusted sources and ingests their artifacts;
- on-demand discovery that fetches against untrusted sources when a single
  `DemandSignal` fires;
- a source-scout that autonomously proposes new candidate sources to the founder;
- the demand-signal subsystem (7 signal types, confidence stacking, dedupe +
  cooldown) that decides *when* on-demand discovery fires;
- four remaining source adapters (`github_topic`, `awesome_list_md`,
  `web_markdown`, `clawhub_api` — the last an explicit stub per spec);
- two mechanical executors (`yalayut_discovery`, `source_scout`) registered in
  mr_roboto and reachable from the Beckman mechanical lane;
- two orchestrator periodic checks (`_check_yalayut_discovery`,
  `_check_source_scout`) that fire on schedule and enqueue real mechanical tasks;
- a `capture_hint` post-hook kind that runs on task finish and replaces the old
  `skills.py` auto-capture;
- the `/yalayut` Telegram command group backed by `packages/yalayut/admin.py`,
  with working inline-button callbacks;
- the policy-proposal flow (KutAI proposes allowlist additions, founder approves
  via Telegram).

Every wired claim above is verified by a task in this plan. Nothing is a
scaffold except `clawhub_api.py`, which the spec explicitly designates a stub.

## Architecture

The Phase 4 dependency graph, identical to the spec's "Interface contract":

```
orchestrator pump
  ├─ _check_yalayut_discovery()  ── timestamp-gated ──> beckman.enqueue({mechanical, action:"yalayut_discovery"})
  └─ _check_source_scout()       ── timestamp-gated ──> beckman.enqueue({mechanical, action:"source_scout"})
        (orchestrator imports ZERO from yalayut — enqueues a plain dict)

beckman mechanical lane ──> mr_roboto.run(task)
  ├─ action "yalayut_discovery" ──> executors/yalayut_discovery.py ──> yalayut.daily_discovery()
  ├─ action "source_scout"      ──> executors/source_scout.py      ──> yalayut.source_scout_scan()
  └─ action "capture_hint"      ──> executors/capture_hint.py      ──> yalayut.capture_hint(task, outcome)

beckman post-hook layer
  └─ kind "capture_hint" (POST_HOOK_REGISTRY) ──> apply._posthook_agent_and_payload
        ──> ("mechanical", {action:"capture_hint", ...}) ──> the capture_hint executor

yalayut.daily_discovery()       ──> discovery/cron.py        ──> trusted (cron-mode) sources → fetch+synth+tier+enable
yalayut.on_demand_discovery()   ──> discovery/on_demand.py   ──> one DemandSignal → untrusted sources
yalayut.source_scout_scan()     ──> discovery/source_scout.py──> propose candidates → yalayut_source_candidates rows
discovery/demand.py             ──> DemandSignal record/stack/dedupe/cooldown over yalayut_demand_signals

source adapters (discovery/sources/)
  ├─ github_topic.py     ── mechanical repo list + LLM-fallback per-artifact synthesis
  ├─ awesome_list_md.py  ── regex-bullet parse + LLM-normalize
  ├─ web_markdown.py     ── frontmatter mechanical
  └─ clawhub_api.py      ── STUB (spec-sanctioned; returns [])

LLM synthesis (discovery/synthesize.py — Phase 1/3 file, extended here)
  ── routes the Sonnet normalization call through beckman.enqueue (await_inline,
     lane=overhead) — yalayut never imports LLMDispatcher; only Beckman calls it.

telegram_bot.py  ──> /yalayut command group ──> yalayut/admin.py
  ── inline callbacks "yal:" prefix handled in handle_callback()
```

Phase 4 touches **one** core-loop file (`orchestrator.py`) and **one** UI file
(`telegram_bot.py`). Everything else lives inside `packages/yalayut/` and
`packages/mr_roboto/`. The orchestrator change is wiring only (two `_check_*`
methods + two pump calls), with zero yalayut import — it enqueues plain dicts.

## Tech Stack

- Python 3.10, fully `async`/`await`. No sync blocking calls.
- `packages/<name>/src/<name>/` src layout (matches `general_beckman`, `mr_roboto`).
- SQLite via `src/infra/db.py` (`get_db()`, aiosqlite, WAL). All 13 yalayut
  tables already created by the Phase 1 migration — Phase 4 only reads/writes.
- `src/infra/times.py` (`utc_now`, `to_db`) for timestamp formatting — never
  `datetime.isoformat()` for `scheduled_tasks` columns.
- `src/infra/logging_config.py` `get_logger("yalayut.<component>")`.
- Embeddings: `multilingual-e5-base` (768d) via `src/memory/embeddings.py`.
- `general_beckman.enqueue(spec_dict, *, lane=...)` — the only LLM/task entry.
- `python-telegram-bot` v20+ async for the `/yalayut` command group.
- Tests: `pytest` + `pytest-asyncio`, always with `timeout 60` prefix.
- `httpx` (async) for source-adapter HTTP fetches; `python-frontmatter` for
  YAML-frontmatter parsing (already a Phase 1 dependency).

## File Structure

### Created

| File | Responsibility |
|---|---|
| `packages/yalayut/src/yalayut/discovery/cron.py` | `daily_discovery()` — pull all enabled trusted (cron-mode) sources, fetch+synthesize+tier+enable each artifact, return a summary dict. |
| `packages/yalayut/src/yalayut/discovery/on_demand.py` | `on_demand_discovery(demand)` — need-driven fetch for one `DemandSignal` against untrusted sources matching the demand's intent keywords. |
| `packages/yalayut/src/yalayut/discovery/source_scout.py` | `source_scout_scan()` — autonomously propose candidate sources (GitHub trending, README cross-refs, web search on demand signals, founder URLs); per-day cap; writes `yalayut_source_candidates`. |
| `packages/yalayut/src/yalayut/discovery/demand.py` | `DemandSignal` dataclass + `record_signal()`, `stack_confidence()`, `pending_signals()` — the 7 signal types, confidence stacking, dedupe by `source_step_pattern`, 7-day cooldown. |
| `packages/yalayut/src/yalayut/discovery/sources/github_topic.py` | `GithubTopicAdapter` — GitHub topic search → repo list → per-repo SKILL.md probe; LLM-fallback synthesis when no canonical frontmatter. |
| `packages/yalayut/src/yalayut/discovery/sources/awesome_list_md.py` | `AwesomeListAdapter` — regex-bullet parse of an awesome-list README + LLM-normalize each bullet to a manifest. |
| `packages/yalayut/src/yalayut/discovery/sources/web_markdown.py` | `WebMarkdownAdapter` — fetch a generic SKILL.md URL, mechanical frontmatter parse. |
| `packages/yalayut/src/yalayut/discovery/sources/clawhub_api.py` | `ClawHubAdapter` — **explicit stub** (ClawHub unenumerable today); `discover()` returns `[]`, logs a debug note. |
| `packages/yalayut/src/yalayut/admin.py` | Founder-ops module: `pending_artifacts/approve_artifact/reject_artifact/requeue`, `pending_sources/approve_source/promote_source/promote_owner`, `policy_proposals/decide_policy/propose_policy`, `disable/enable`, `stats`, `missing_auth/set_secret`, `mcp_status/mcp_restart/mcp_kill`. Imported only by `telegram_bot.py`. |
| `packages/yalayut/src/yalayut/policy_observer.py` | `observe_and_propose()` — scans recent vetting audit rows for repeated unknown shell tokens / domains, writes `yalayut_policy_proposals` rows. |
| `packages/mr_roboto/src/mr_roboto/executors/yalayut_discovery.py` | Mechanical executor — `run(task)` dispatches `daily` (→ `daily_discovery()`) or `on_demand` (→ `on_demand_discovery(demand)`) mode. |
| `packages/mr_roboto/src/mr_roboto/executors/source_scout.py` | Mechanical executor — `run(task)` calls `yalayut.source_scout_scan()`. |
| `packages/mr_roboto/src/mr_roboto/executors/capture_hint.py` | Mechanical executor — `run(task)` calls `yalayut.capture_hint(task, outcome)`. |
| `tests/yalayut/test_phase4_demand_signals.py` | Demand-signal record/stack/dedupe/cooldown unit tests. |
| `tests/yalayut/test_phase4_cron_discovery.py` | `daily_discovery()` integration test with mocked trusted source. |
| `tests/yalayut/test_phase4_on_demand.py` | `on_demand_discovery()` integration test with mocked untrusted source. |
| `tests/yalayut/test_phase4_source_scout.py` | `source_scout_scan()` candidate-proposal + per-day cap test. |
| `tests/yalayut/test_phase4_adapters.py` | `github_topic` / `awesome_list_md` / `web_markdown` / `clawhub_api` adapter tests against fixtures. |
| `tests/yalayut/test_phase4_executors.py` | mr_roboto executor reachability + dispatch tests for all three executors. |
| `tests/yalayut/test_phase4_capture_hint.py` | `capture_hint` post-hook registry + routing + executor end-to-end test. |
| `tests/yalayut/test_phase4_orchestrator_checks.py` | `_check_yalayut_discovery` / `_check_source_scout` timestamp-gating + enqueue test. |
| `tests/yalayut/test_phase4_admin.py` | `admin.py` API surface tests (pending/approve/reject/policy/auth/mcp). |
| `tests/yalayut/test_phase4_policy_observer.py` | `policy_observer.observe_and_propose()` test. |
| `tests/yalayut/test_phase4_telegram.py` | `/yalayut` command + callback handler tests with a stub Telegram update. |

### Modified

| File | Change |
|---|---|
| `packages/yalayut/src/yalayut/__init__.py` | Export `daily_discovery`, `on_demand_discovery`, `source_scout_scan`, `capture_hint` (lazy re-exports). |
| `packages/yalayut/src/yalayut/discovery/synthesize.py` | Add `llm_synthesize(raw_text, source_meta)` — routes a Sonnet normalization call through `beckman.enqueue(... lane="overhead")`; used by `github_topic` + `awesome_list_md`. |
| `packages/mr_roboto/src/mr_roboto/__init__.py` | Add `action == "yalayut_discovery"`, `action == "source_scout"`, `action == "capture_hint"` dispatch branches. |
| `packages/general_beckman/src/general_beckman/posthooks.py` | Register the `capture_hint` `PostHookSpec` in `POST_HOOK_REGISTRY`. |
| `packages/general_beckman/src/general_beckman/apply.py` | Add `capture_hint` branch to `_posthook_agent_and_payload` → routes to `mechanical`. |
| `packages/general_beckman/src/general_beckman/cron_seed.py` | Add `yalayut_discovery` + `source_scout` internal cadences (so the periodic checks have a backstop cadence row). |
| `src/core/orchestrator.py` | Add `_check_yalayut_discovery()` + `_check_source_scout()` methods; call both from the `run_loop` pump (timestamp-gated, mirroring the throttle counter pattern). |
| `src/app/telegram_bot.py` | Register the `/yalayut` `CommandHandler`; add `cmd_yalayut` dispatcher + `handle_callback` `yal:` branch. |
| `src/memory/skills.py` | Remove the old auto-capture call site (now superseded by the `capture_hint` post-hook); leave the read shim untouched. |

## Tasks

---

### Task 1 — DemandSignal subsystem

**Files:**
- Create: `packages/yalayut/src/yalayut/discovery/demand.py`
- Test: `tests/yalayut/test_phase4_demand_signals.py`

The `yalayut_demand_signals` table already exists (Phase 1 schema). This task
builds the record/stack/dedupe/cooldown logic over it.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_demand_signals.py` with the failing test:

```python
import asyncio
import json

import pytest

from src.infra.db import init_db, get_db
from yalayut.discovery import demand


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_record_signal_inserts_row(loop):
    async def _run():
        await init_db()
        sig = demand.DemandSignal(
            source_step_pattern="auth-wiring-fastapi",
            intent_keywords=["auth", "jwt", "fastapi"],
            signal_type="planning_miss",
            confidence=0.4,
        )
        sid = await demand.record_signal(sig)
        assert sid > 0
        db = await get_db()
        cur = await db.execute(
            "SELECT signal_type, confidence, resulted_in_discovery "
            "FROM yalayut_demand_signals WHERE id = ?", (sid,))
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == "planning_miss"
        assert abs(row[1] - 0.4) < 1e-6
        assert row[2] in (0, None)
    loop.run_until_complete(_run())


def test_confidence_stacks_across_signals(loop):
    async def _run():
        await init_db()
        pat = "rag-pipeline-setup"
        for st, conf in [("planning_miss", 0.4), ("tool_call", 0.3),
                          ("hint_miss", 0.2)]:
            await demand.record_signal(demand.DemandSignal(
                source_step_pattern=pat, intent_keywords=["rag"],
                signal_type=st, confidence=conf))
        stacked = await demand.stack_confidence(pat)
        # 1 - (1-.4)(1-.3)(1-.2) = 1 - .336 = .664
        assert abs(stacked - 0.664) < 1e-3
    loop.run_until_complete(_run())


def test_dedupe_within_cooldown(loop):
    async def _run():
        await init_db()
        pat = "cooldown-pattern-xyz"
        s1 = demand.DemandSignal(source_step_pattern=pat,
                                 intent_keywords=["x"],
                                 signal_type="dlq", confidence=0.5)
        first = await demand.record_signal(s1)
        assert first > 0
        # Same pattern + same type within cooldown → deduped (returns -1).
        second = await demand.record_signal(s1)
        assert second == -1
    loop.run_until_complete(_run())


def test_pending_signals_orders_by_stacked_confidence(loop):
    async def _run():
        await init_db()
        await demand.record_signal(demand.DemandSignal(
            source_step_pattern="low-pat", intent_keywords=["a"],
            signal_type="repeat_pattern", confidence=0.15))
        for st in ("planning_miss", "step_entry_miss", "tool_call"):
            await demand.record_signal(demand.DemandSignal(
                source_step_pattern="high-pat", intent_keywords=["b"],
                signal_type=st, confidence=0.5))
        pend = await demand.pending_signals(limit=10)
        pats = [p["source_step_pattern"] for p in pend]
        assert pats.index("high-pat") < pats.index("low-pat")
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL** (`ModuleNotFoundError: yalayut.discovery.demand`):

```
timeout 60 pytest tests/yalayut/test_phase4_demand_signals.py -x -q
```

- [ ] Create `packages/yalayut/src/yalayut/discovery/demand.py`:

```python
"""Yalayut Phase 4 — demand-signal subsystem.

Seven signal types feed one queue with confidence stacking, dedupe by
``source_step_pattern`` and a per-(pattern, type) cooldown window.

Signal types (spec — 4 proactive + 3 reactive):
  proactive: planning_miss, step_entry_miss, tool_call, founder
  reactive:  hint_miss, dlq, repeat_pattern
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

logger = get_logger("yalayut.demand")

SIGNAL_TYPES: frozenset[str] = frozenset({
    "planning_miss", "step_entry_miss", "tool_call", "founder",
    "hint_miss", "dlq", "repeat_pattern",
})

#: Per-(pattern, type) cooldown — spec "start at 7d per source_step_pattern".
COOLDOWN_SECONDS: int = 7 * 24 * 3600


@dataclass
class DemandSignal:
    """One fired demand signal. ``confidence`` is the per-signal weight;
    the stacked confidence across signals is computed by ``stack_confidence``."""
    source_step_pattern: str
    intent_keywords: list[str]
    signal_type: str
    confidence: float = 0.3
    fired_at: str = field(default_factory=lambda: to_db(utc_now()))

    def __post_init__(self) -> None:
        if self.signal_type not in SIGNAL_TYPES:
            raise ValueError(f"unknown signal_type: {self.signal_type!r}")
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


async def _within_cooldown(pattern: str, signal_type: str) -> bool:
    """True when a same-(pattern, type) signal fired inside the cooldown."""
    db = await get_db()
    cutoff = to_db(utc_now() - timedelta(seconds=COOLDOWN_SECONDS))
    cur = await db.execute(
        "SELECT 1 FROM yalayut_demand_signals "
        "WHERE source_step_pattern = ? AND signal_type = ? AND fired_at >= ? "
        "LIMIT 1",
        (pattern, signal_type, cutoff),
    )
    row = await cur.fetchone()
    await cur.close()
    return row is not None


async def record_signal(sig: DemandSignal) -> int:
    """Insert a demand signal. Returns the new row id, or ``-1`` when the
    signal is deduped (same pattern + type already within cooldown)."""
    if await _within_cooldown(sig.source_step_pattern, sig.signal_type):
        logger.debug(
            "demand signal deduped (cooldown)",
            pattern=sig.source_step_pattern, type=sig.signal_type,
        )
        return -1
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO yalayut_demand_signals "
        "(source_step_pattern, intent_keywords_json, signal_type, "
        " confidence, fired_at, resulted_in_discovery) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (sig.source_step_pattern, json.dumps(sig.intent_keywords),
         sig.signal_type, sig.confidence, sig.fired_at),
    )
    await db.commit()
    logger.info("demand signal recorded", pattern=sig.source_step_pattern,
                type=sig.signal_type, confidence=sig.confidence)
    return int(cur.lastrowid)


async def stack_confidence(pattern: str) -> float:
    """Combine all un-discovered signals for one pattern into a single
    confidence via independent-probability stacking:
    ``1 - Π(1 - c_i)``. Bounded in [0, 1]."""
    db = await get_db()
    cur = await db.execute(
        "SELECT confidence FROM yalayut_demand_signals "
        "WHERE source_step_pattern = ? AND resulted_in_discovery = 0",
        (pattern,),
    )
    rows = await cur.fetchall()
    await cur.close()
    miss = 1.0
    for (c,) in rows:
        miss *= (1.0 - max(0.0, min(1.0, float(c or 0.0))))
    return round(1.0 - miss, 6)


async def pending_signals(limit: int = 20) -> list[dict]:
    """Return distinct un-discovered patterns with their stacked confidence
    and merged intent keywords, ordered by stacked confidence descending."""
    db = await get_db()
    cur = await db.execute(
        "SELECT DISTINCT source_step_pattern FROM yalayut_demand_signals "
        "WHERE resulted_in_discovery = 0",
    )
    patterns = [r[0] for r in await cur.fetchall()]
    await cur.close()
    out: list[dict] = []
    for pat in patterns:
        stacked = await stack_confidence(pat)
        kw_cur = await db.execute(
            "SELECT intent_keywords_json FROM yalayut_demand_signals "
            "WHERE source_step_pattern = ? AND resulted_in_discovery = 0",
            (pat,),
        )
        merged: set[str] = set()
        for (kj,) in await kw_cur.fetchall():
            try:
                for k in json.loads(kj or "[]"):
                    merged.add(str(k))
            except (json.JSONDecodeError, TypeError):
                continue
        await kw_cur.close()
        out.append({
            "source_step_pattern": pat,
            "stacked_confidence": stacked,
            "intent_keywords": sorted(merged),
        })
    out.sort(key=lambda d: d["stacked_confidence"], reverse=True)
    return out[:limit]


async def mark_discovered(pattern: str) -> None:
    """Flip ``resulted_in_discovery`` for every signal on a pattern once an
    on-demand discovery run consumed it."""
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_demand_signals SET resulted_in_discovery = 1 "
        "WHERE source_step_pattern = ?",
        (pattern,),
    )
    await db.commit()
```

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_demand_signals.py -x -q
```

- [ ] Commit:

```
rtk git add packages/yalayut/src/yalayut/discovery/demand.py tests/yalayut/test_phase4_demand_signals.py
rtk git commit -m "feat(yalayut,p4): demand-signal subsystem — record/stack/dedupe/cooldown"
```

---

### Task 2 — Remaining source adapters

**Files:**
- Create: `packages/yalayut/src/yalayut/discovery/sources/github_topic.py`
- Create: `packages/yalayut/src/yalayut/discovery/sources/awesome_list_md.py`
- Create: `packages/yalayut/src/yalayut/discovery/sources/web_markdown.py`
- Create: `packages/yalayut/src/yalayut/discovery/sources/clawhub_api.py`
- Modify: `packages/yalayut/src/yalayut/discovery/synthesize.py`
- Test: `tests/yalayut/test_phase4_adapters.py`

`SourceAdapter` protocol (Phase 1 `contracts.py`): `source_type: str`,
`async discover(source_cfg) -> list[ArtifactRef]`, `async fetch(ref) -> Path`.
`ArtifactRef` (Phase 1) is a dataclass with `name`, `name_original`, `owner`,
`source`, `raw_url`, `native_format` fields. The adapters here reuse those.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_adapters.py`:

```python
import asyncio

import pytest

from yalayut.discovery.sources.github_topic import GithubTopicAdapter
from yalayut.discovery.sources.awesome_list_md import AwesomeListAdapter
from yalayut.discovery.sources.web_markdown import WebMarkdownAdapter
from yalayut.discovery.sources.clawhub_api import ClawHubAdapter
from yalayut.contracts import SourceConfig


_AWESOME_README = """\
## Cloud
- **[mcp-server-cloudflare](https://github.com/cloudflare/mcp-server-cloudflare)** - Manage Cloudflare Workers, KV, R2.
- **[open-museum-mcp](https://github.com/x/open-museum-mcp)** - Federated museum collections.

## Browser
- **[mcp-browser-use](https://github.com/y/mcp-browser-use)** - Browser automation via Playwright.
"""

_SKILL_MD = """\
---
name: pdf
description: Use this skill for working with PDF files — extract, merge, split.
license: Proprietary
---
# PDF skill body
Detailed instructions here.
"""


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_clawhub_is_stub(loop):
    async def _run():
        adapter = ClawHubAdapter()
        assert adapter.source_type == "clawhub_api"
        cfg = SourceConfig(source_id="clawhub:stub", source_type="clawhub_api",
                           endpoint="", owner="clawhub")
        refs = await adapter.discover(cfg)
        assert refs == []
    loop.run_until_complete(_run())


def test_awesome_list_parses_bullets(loop):
    async def _run():
        adapter = AwesomeListAdapter()
        refs = adapter.parse_readme(_AWESOME_README,
                                    source_id="github:punkpeye/awesome-mcp-servers")
        names = {r.name_original for r in refs}
        assert "mcp-server-cloudflare" in names
        assert "open-museum-mcp" in names
        assert "mcp-browser-use" in names
        cf = next(r for r in refs if r.name_original == "mcp-server-cloudflare")
        assert cf.owner == "cloudflare"
        assert cf.raw_url.endswith("mcp-server-cloudflare")
    loop.run_until_complete(_run())


def test_web_markdown_parses_frontmatter(loop):
    async def _run():
        adapter = WebMarkdownAdapter()
        ref, body = adapter.parse_skill_md(
            _SKILL_MD, url="https://example.com/SKILL.md")
        assert ref.name_original == "pdf"
        assert ref.native_format == "frontmatter"
        assert "PDF skill body" in body
    loop.run_until_complete(_run())


def test_github_topic_canonical_slug(loop):
    async def _run():
        adapter = GithubTopicAdapter()
        # name canonicalization: <source-slug>-<original>, dedup org prefix.
        assert adapter.canonical_name("anthropics", "pdf") == "anthropics-pdf"
        assert adapter.canonical_name("matlab", "matlab-live-script") == \
            "matlab-live-script"
        assert adapter.canonical_name("cookiecutter",
                                      "cookiecutter-django") == "cc-django"
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL** (`ModuleNotFoundError`):

```
timeout 60 pytest tests/yalayut/test_phase4_adapters.py -x -q
```

- [ ] Create `packages/yalayut/src/yalayut/discovery/sources/clawhub_api.py`:

```python
"""Yalayut Phase 4 — ClawHub source adapter.

EXPLICIT STUB. ClawHub's catalog is not enumerable without a headless
browser (spec non-goal). ``discover()`` returns an empty list so the cron
loop can include this adapter without special-casing it. When ClawHub
exposes a public API, replace ``discover()``/``fetch()`` with a real
implementation — no other Phase 4 file needs to change.
"""
from __future__ import annotations

from pathlib import Path

from src.infra.logging_config import get_logger
from yalayut.contracts import ArtifactRef, SourceConfig

logger = get_logger("yalayut.adapter.clawhub")


class ClawHubAdapter:
    source_type: str = "clawhub_api"

    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]:
        logger.debug("clawhub_api adapter is a stub — discover() returns []")
        return []

    async def fetch(self, ref: ArtifactRef) -> Path:
        raise NotImplementedError("clawhub_api adapter is a stub")
```

- [ ] Create `packages/yalayut/src/yalayut/discovery/sources/web_markdown.py`:

```python
"""Yalayut Phase 4 — web_markdown source adapter.

Fetches a single generic SKILL.md URL and parses YAML frontmatter
mechanically (no LLM). Same shape as github_path; different fetch.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import frontmatter
import httpx

from src.infra.logging_config import get_logger
from yalayut.contracts import ArtifactRef, SourceConfig

logger = get_logger("yalayut.adapter.web_markdown")


class WebMarkdownAdapter:
    source_type: str = "web_markdown"

    def parse_skill_md(self, text: str, url: str) -> tuple[ArtifactRef, str]:
        """Parse a SKILL.md string → (ArtifactRef, body). Mechanical."""
        post = frontmatter.loads(text)
        name_original = str(post.get("name") or "").strip()
        if not name_original:
            raise ValueError(f"SKILL.md at {url} has no 'name' frontmatter")
        owner = url.split("//", 1)[-1].split("/", 1)[0]
        ref = ArtifactRef(
            name=f"web-{name_original}",
            name_original=name_original,
            owner=owner,
            source=f"web:{url}",
            raw_url=url,
            native_format="frontmatter",
        )
        return ref, post.content

    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]:
        url = source_cfg.endpoint
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        ref, _body = self.parse_skill_md(resp.text, url)
        return [ref]

    async def fetch(self, ref: ArtifactRef) -> Path:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(ref.raw_url)
            resp.raise_for_status()
        staging = Path(tempfile.mkdtemp(prefix="yalayut_web_"))
        body_path = staging / "SKILL.md"
        body_path.write_text(resp.text, encoding="utf-8")
        logger.info("web_markdown fetched", url=ref.raw_url)
        return body_path
```

- [ ] Create `packages/yalayut/src/yalayut/discovery/sources/awesome_list_md.py`:

```python
"""Yalayut Phase 4 — awesome_list_md source adapter.

Two-pass: (1) regex extracts ``name + repo URL + raw description`` from
``**[name](url)** - desc`` bullets; (2) LLM-normalizes each bullet to
intent_keywords / auth_env / install_cmd via discovery.synthesize.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import httpx

from src.infra.logging_config import get_logger
from yalayut.contracts import ArtifactRef, SourceConfig

logger = get_logger("yalayut.adapter.awesome_list")

#: **[name](url)** - description   (markdown bullet, awesome-list house style)
_BULLET = re.compile(
    r"^\s*[-*]\s*\*\*\[(?P<name>[^\]]+)\]\((?P<url>https?://[^)]+)\)\*\*"
    r"\s*[-–:]?\s*(?P<desc>.*)$"
)


class AwesomeListAdapter:
    source_type: str = "awesome_list_md"

    def parse_readme(self, text: str, source_id: str) -> list[ArtifactRef]:
        """Pass 1 — mechanical bullet parse. No LLM."""
        refs: list[ArtifactRef] = []
        for line in text.splitlines():
            m = _BULLET.match(line)
            if not m:
                continue
            name_original = m.group("name").strip()
            url = m.group("url").strip()
            # owner = the GitHub org segment of the URL.
            parts = url.split("//", 1)[-1].split("/")
            owner = parts[1] if len(parts) > 1 else "unknown"
            refs.append(ArtifactRef(
                name=f"{owner}-{name_original}",
                name_original=name_original,
                owner=owner,
                source=source_id,
                raw_url=url,
                native_format="awesome_bullet",
                raw_description=m.group("desc").strip(),
            ))
        logger.info("awesome_list parsed", source=source_id, count=len(refs))
        return refs

    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            resp = await c.get(source_cfg.endpoint)
            resp.raise_for_status()
        return self.parse_readme(resp.text, source_cfg.source_id)

    async def fetch(self, ref: ArtifactRef) -> Path:
        """Fetch the linked repo's README for LLM synthesis input."""
        readme_url = ref.raw_url.rstrip("/") + "/raw/HEAD/README.md"
        staging = Path(tempfile.mkdtemp(prefix="yalayut_awesome_"))
        body_path = staging / "README.md"
        try:
            async with httpx.AsyncClient(timeout=30,
                                         follow_redirects=True) as c:
                resp = await c.get(readme_url)
                resp.raise_for_status()
            body_path.write_text(resp.text, encoding="utf-8")
        except Exception as e:
            # README unreachable — fall back to the bullet description so
            # synthesis still has text to work with.
            logger.debug("awesome_list README fetch failed: %s", e)
            body_path.write_text(ref.raw_description or "", encoding="utf-8")
        return body_path
```

- [ ] Create `packages/yalayut/src/yalayut/discovery/sources/github_topic.py`:

```python
"""Yalayut Phase 4 — github_topic source adapter.

Stage 1: GitHub topic search → repo list (mechanical).
Stage 2: per-repo SKILL.md probe; canonical frontmatter parses mechanically,
otherwise the README is handed to LLM-fallback synthesis.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import frontmatter
import httpx

from src.infra.logging_config import get_logger
from yalayut.contracts import ArtifactRef, SourceConfig

logger = get_logger("yalayut.adapter.github_topic")

#: Per-run repo cap — prevents a topic search flooding the catalog.
TOPIC_REPO_CAP: int = 25


class GithubTopicAdapter:
    source_type: str = "github_topic"

    def canonical_name(self, owner: str, name_original: str) -> str:
        """``<source-slug>-<original>`` with the dumb-prefix fixes from recon:
        - drop org prefix when the name already starts with it
          (``matlab`` + ``matlab-live-script`` → ``matlab-live-script``);
        - cookiecutter repos collapse to the ``cc-`` slug
          (``cookiecutter`` + ``cookiecutter-django`` → ``cc-django``)."""
        n = name_original.strip()
        o = owner.strip().lower()
        if o == "cookiecutter" or n.startswith("cookiecutter-"):
            return "cc-" + n.removeprefix("cookiecutter-")
        if n.lower().startswith(o + "-") or n.lower() == o:
            return n
        return f"{o}-{n}"

    async def discover(self, source_cfg: SourceConfig) -> list[ArtifactRef]:
        """Topic search via the public GitHub REST API. ``endpoint`` carries
        the topic string (e.g. ``claude-skill``)."""
        topic = source_cfg.endpoint or "claude-skill"
        url = (f"https://api.github.com/search/repositories"
               f"?q=topic:{topic}&sort=stars&per_page={TOPIC_REPO_CAP}")
        headers = {"Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 403:
                logger.warning("github_topic rate limited — skipping run")
                return []
            resp.raise_for_status()
            items = resp.json().get("items", [])
        refs: list[ArtifactRef] = []
        for repo in items[:TOPIC_REPO_CAP]:
            owner = (repo.get("owner") or {}).get("login", "unknown")
            name_original = repo.get("name", "")
            if not name_original:
                continue
            default_branch = repo.get("default_branch", "main")
            skill_url = (f"https://raw.githubusercontent.com/{owner}/"
                         f"{name_original}/{default_branch}/SKILL.md")
            refs.append(ArtifactRef(
                name=self.canonical_name(owner, name_original),
                name_original=name_original,
                owner=owner,
                source=f"github_topic:{topic}",
                raw_url=skill_url,
                native_format="unknown",
            ))
        logger.info("github_topic discovered", topic=topic, count=len(refs))
        return refs

    async def fetch(self, ref: ArtifactRef) -> Path:
        """Probe SKILL.md; mark native_format so synthesis picks the path."""
        staging = Path(tempfile.mkdtemp(prefix="yalayut_topic_"))
        body_path = staging / "SKILL.md"
        async with httpx.AsyncClient(timeout=30,
                                     follow_redirects=True) as client:
            resp = await client.get(ref.raw_url)
        if resp.status_code == 200 and resp.text.strip().startswith("---"):
            try:
                frontmatter.loads(resp.text)
                ref.native_format = "frontmatter"
            except Exception:
                ref.native_format = "freeform"
            body_path.write_text(resp.text, encoding="utf-8")
        else:
            # No canonical SKILL.md — fetch the README for LLM fallback.
            ref.native_format = "freeform"
            readme_url = ref.raw_url.rsplit("/", 1)[0] + "/README.md"
            try:
                async with httpx.AsyncClient(timeout=30,
                                             follow_redirects=True) as c:
                    r2 = await c.get(readme_url)
                    r2.raise_for_status()
                body_path.write_text(r2.text, encoding="utf-8")
            except Exception:
                body_path.write_text("", encoding="utf-8")
        return body_path
```

- [ ] Append `llm_synthesize` to `packages/yalayut/src/yalayut/discovery/synthesize.py` (existing Phase 1/3 file — add this function, do not rewrite the file):

```python
async def llm_synthesize(raw_text: str, source_meta: dict) -> dict:
    """LLM-fallback manifest synthesis for unstructured sources
    (awesome-list bullets, freeform README). Routes a Sonnet call through
    ``beckman.enqueue`` — yalayut never imports LLMDispatcher; only Beckman
    talks to it (KutAI "only Beckman calls the dispatcher" rule).

    Returns a partial manifest dict: ``{intent_keywords, mechanizable,
    kind, install_cmd, auth_env}``. On any failure returns a conservative
    empty-ish manifest so the caller can still tier the artifact at T1/T2.
    """
    import json as _json

    from src.infra.logging_config import get_logger as _gl
    _log = _gl("yalayut.synthesize")

    prompt = (
        "You normalize a software-artifact description into a JSON manifest. "
        "Return ONLY a JSON object with keys: intent_keywords (array of "
        "lowercase strings), mechanizable (boolean), kind (one of "
        "prompt_skill|shell_recipe|procedure|agent_config), install_cmd "
        "(string or null), auth_env (string env-var name or null).\n\n"
        f"Artifact name: {source_meta.get('name_original', '')}\n"
        f"Description / README:\n{raw_text[:2000]}\n"
    )
    try:
        import general_beckman
        result = await general_beckman.enqueue(
            {
                "agent_type": "yalayut_synthesizer",
                "context": {
                    "llm_call": {
                        "raw_dispatch": True,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "model_hint": "sonnet",
                    },
                },
                "kind": "overhead",
            },
            lane="overhead",
            await_inline=True,
        )
        raw = result.get("result") if isinstance(result, dict) else result
        parsed = _json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception as e:  # noqa: BLE001 — synthesis must never crash cron
        _log.warning("llm_synthesize failed, using empty manifest: %s", e)
        parsed = {}

    return {
        "intent_keywords": list(parsed.get("intent_keywords") or []),
        "mechanizable": bool(parsed.get("mechanizable", False)),
        "kind": parsed.get("kind") or "prompt_skill",
        "install_cmd": parsed.get("install_cmd"),
        "auth_env": parsed.get("auth_env"),
    }
```

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_adapters.py -x -q
```

- [ ] Commit:

```
rtk git add packages/yalayut/src/yalayut/discovery/sources/ packages/yalayut/src/yalayut/discovery/synthesize.py tests/yalayut/test_phase4_adapters.py
rtk git commit -m "feat(yalayut,p4): github_topic/awesome_list_md/web_markdown adapters + clawhub stub + LLM-fallback synthesis"
```

---

### Task 3 — daily_discovery() cron body

**Files:**
- Create: `packages/yalayut/src/yalayut/discovery/cron.py`
- Test: `tests/yalayut/test_phase4_cron_discovery.py`

`daily_discovery()` pulls every enabled trusted source whose `discovery_mode`
is `cron` or `both`, runs the existing Phase 1/3 `fetch → synthesize → tier →
enable` pipeline per artifact, and returns a summary dict.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_cron_discovery.py`:

```python
import asyncio

import pytest

from src.infra.db import init_db, get_db
from src.infra.times import utc_now, to_db
from yalayut.discovery import cron as yal_cron


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


async def _seed_trusted_source():
    db = await get_db()
    await db.execute(
        "INSERT OR IGNORE INTO yalayut_sources "
        "(source_id, source_type, endpoint, trusted, enabled, "
        " discovery_mode, min_interval_s) "
        "VALUES ('github:test/skills', 'github_path', "
        "'https://example.com', 1, 1, 'cron', 0)")
    await db.commit()


def test_daily_discovery_returns_summary(loop, monkeypatch):
    async def _run():
        await init_db()
        await _seed_trusted_source()

        async def _fake_ingest(source_row):
            return {"ingested": 3, "enabled": 2, "quarantined": 1}

        monkeypatch.setattr(yal_cron, "_ingest_source", _fake_ingest)
        summary = await yal_cron.daily_discovery()
        assert summary["sources_scanned"] == 1
        assert summary["artifacts_ingested"] == 3
        assert summary["artifacts_enabled"] == 2
    loop.run_until_complete(_run())


def test_daily_discovery_respects_min_interval(loop, monkeypatch):
    async def _run():
        await init_db()
        db = await get_db()
        # source ran 30s ago, min_interval 3600 → must be skipped.
        await db.execute(
            "INSERT OR IGNORE INTO yalayut_sources "
            "(source_id, source_type, endpoint, trusted, enabled, "
            " discovery_mode, min_interval_s, last_run_at) "
            "VALUES ('github:recent/skills', 'github_path', 'x', 1, 1, "
            "'cron', 3600, ?)",
            (to_db(utc_now()),))
        await db.commit()

        called = []

        async def _fake_ingest(source_row):
            called.append(source_row["source_id"])
            return {"ingested": 0, "enabled": 0, "quarantined": 0}

        monkeypatch.setattr(yal_cron, "_ingest_source", _fake_ingest)
        summary = await yal_cron.daily_discovery()
        assert "github:recent/skills" not in called
        assert summary["sources_skipped"] >= 1
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL**:

```
timeout 60 pytest tests/yalayut/test_phase4_cron_discovery.py -x -q
```

- [ ] Create `packages/yalayut/src/yalayut/discovery/cron.py`:

```python
"""Yalayut Phase 4 — daily discovery cron.

``daily_discovery()`` is the body the ``yalayut_discovery`` mechanical
executor calls in ``daily`` mode. It pulls every enabled, trusted source
whose ``discovery_mode`` is ``cron`` or ``both`` and runs the Phase 1/3
ingest pipeline (fetch → synthesize → tier → enable) per artifact.
"""
from __future__ import annotations

from datetime import timedelta

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

logger = get_logger("yalayut.cron")


async def _due_cron_sources() -> list[dict]:
    """Trusted, enabled, cron/both sources whose min_interval has elapsed."""
    db = await get_db()
    cur = await db.execute(
        "SELECT source_id, source_type, endpoint, auth_env, "
        "       min_interval_s, last_run_at "
        "FROM yalayut_sources "
        "WHERE enabled = 1 AND trusted = 1 "
        "  AND discovery_mode IN ('cron', 'both')",
    )
    rows = await cur.fetchall()
    await cur.close()
    now = utc_now()
    due: list[dict] = []
    for sid, stype, endpoint, auth_env, min_iv, last_run in rows:
        if last_run and min_iv:
            try:
                last_dt = utc_now().fromisoformat(str(last_run).replace(" ", "T"))
                if now - last_dt < timedelta(seconds=int(min_iv)):
                    continue
            except (ValueError, TypeError):
                pass
        due.append({
            "source_id": sid, "source_type": stype, "endpoint": endpoint,
            "auth_env": auth_env, "min_interval_s": min_iv,
        })
    return due


async def _ingest_source(source_row: dict) -> dict:
    """Run the Phase 1/3 ingest pipeline for one source.

    Delegates to ``yalayut.discovery.fetch.ingest_all`` (Phase 1/3 entry
    point). Returns ``{ingested, enabled, quarantined}``.
    """
    from yalayut.discovery import fetch as yal_fetch
    return await yal_fetch.ingest_all(source_row)


async def _mark_ran(source_id: str) -> None:
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_sources SET last_run_at = ? WHERE source_id = ?",
        (to_db(utc_now()), source_id),
    )
    await db.commit()


async def daily_discovery() -> dict:
    """Pull all due trusted cron-mode sources. Returns a summary dict."""
    due = await _due_cron_sources()
    summary = {
        "sources_scanned": 0,
        "sources_skipped": 0,
        "artifacts_ingested": 0,
        "artifacts_enabled": 0,
        "artifacts_quarantined": 0,
        "errors": [],
    }
    # Count rows that exist but are not due as "skipped".
    db = await get_db()
    cur = await db.execute(
        "SELECT COUNT(*) FROM yalayut_sources "
        "WHERE enabled = 1 AND trusted = 1 "
        "  AND discovery_mode IN ('cron', 'both')")
    total = (await cur.fetchone())[0]
    await cur.close()
    summary["sources_skipped"] = max(0, total - len(due))

    for src in due:
        try:
            res = await _ingest_source(src)
            summary["sources_scanned"] += 1
            summary["artifacts_ingested"] += int(res.get("ingested", 0))
            summary["artifacts_enabled"] += int(res.get("enabled", 0))
            summary["artifacts_quarantined"] += int(res.get("quarantined", 0))
            await _mark_ran(src["source_id"])
        except Exception as e:  # noqa: BLE001 — one bad source must not abort
            logger.warning("cron ingest failed for %s: %s",
                            src["source_id"], e)
            summary["errors"].append(f"{src['source_id']}: {e}")
    logger.info("daily_discovery complete", **{
        k: v for k, v in summary.items() if k != "errors"})
    return summary
```

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_cron_discovery.py -x -q
```

- [ ] Commit:

```
rtk git add packages/yalayut/src/yalayut/discovery/cron.py tests/yalayut/test_phase4_cron_discovery.py
rtk git commit -m "feat(yalayut,p4): daily_discovery() cron body — pull trusted cron-mode sources"
```

---

### Task 4 — on_demand_discovery() body

**Files:**
- Create: `packages/yalayut/src/yalayut/discovery/on_demand.py`
- Test: `tests/yalayut/test_phase4_on_demand.py`

`on_demand_discovery(demand)` takes one demand dict (a `pending_signals()`
entry — `source_step_pattern`, `intent_keywords`, `stacked_confidence`),
matches untrusted on-demand sources against its intent keywords, ingests them
with a per-source artifact cap, and marks the demand pattern discovered.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_on_demand.py`:

```python
import asyncio

import pytest

from src.infra.db import init_db, get_db
from yalayut.discovery import on_demand
from yalayut.discovery import demand as yal_demand


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_on_demand_ingests_matching_source(loop, monkeypatch):
    async def _run():
        await init_db()
        db = await get_db()
        await db.execute(
            "INSERT OR IGNORE INTO yalayut_sources "
            "(source_id, source_type, endpoint, trusted, enabled, "
            " discovery_mode, min_interval_s) "
            "VALUES ('github:awesome/mcp', 'awesome_list_md', 'x', 0, 1, "
            "'on_demand', 0)")
        await db.commit()

        async def _fake_ingest(source_row, *, artifact_cap=10):
            return {"ingested": 2, "enabled": 0, "quarantined": 2}

        monkeypatch.setattr(on_demand, "_ingest_source_capped", _fake_ingest)
        result = await on_demand.on_demand_discovery({
            "source_step_pattern": "mcp-cloudflare-setup",
            "intent_keywords": ["mcp", "cloudflare"],
            "stacked_confidence": 0.7,
        })
        assert result["artifacts_ingested"] == 2
        assert result["pattern"] == "mcp-cloudflare-setup"
    loop.run_until_complete(_run())


def test_on_demand_marks_pattern_discovered(loop, monkeypatch):
    async def _run():
        await init_db()
        await yal_demand.record_signal(yal_demand.DemandSignal(
            source_step_pattern="pat-done", intent_keywords=["x"],
            signal_type="dlq", confidence=0.5))

        async def _fake_ingest(source_row, *, artifact_cap=10):
            return {"ingested": 0, "enabled": 0, "quarantined": 0}

        monkeypatch.setattr(on_demand, "_ingest_source_capped", _fake_ingest)
        monkeypatch.setattr(on_demand, "_untrusted_sources_for",
                            lambda kw: [])
        await on_demand.on_demand_discovery({
            "source_step_pattern": "pat-done",
            "intent_keywords": ["x"],
            "stacked_confidence": 0.5,
        })
        db = await get_db()
        cur = await db.execute(
            "SELECT resulted_in_discovery FROM yalayut_demand_signals "
            "WHERE source_step_pattern = 'pat-done'")
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == 1
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL**:

```
timeout 60 pytest tests/yalayut/test_phase4_on_demand.py -x -q
```

- [ ] Create `packages/yalayut/src/yalayut/discovery/on_demand.py`:

```python
"""Yalayut Phase 4 — on-demand discovery.

``on_demand_discovery(demand)`` is the need-driven path: one ``DemandSignal``
fires, this fetches against *untrusted* sources whose intent overlaps the
demand's keywords. Volume-dangerous catalogs (public-apis ~1.4k, awesome-mcp
~1k) are only pulled here, with a per-source artifact cap.
"""
from __future__ import annotations

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from yalayut.discovery import demand as _demand

logger = get_logger("yalayut.on_demand")

#: Per-source artifact cap for an on-demand run — first-flood guard.
ON_DEMAND_ARTIFACT_CAP: int = 10


async def _untrusted_sources_for(intent_keywords: list[str]) -> list[dict]:
    """Untrusted, enabled, on_demand/both sources. The intent filter is
    coarse — adapters apply per-artifact keyword matching downstream."""
    db = await get_db()
    cur = await db.execute(
        "SELECT source_id, source_type, endpoint, auth_env "
        "FROM yalayut_sources "
        "WHERE enabled = 1 AND trusted = 0 "
        "  AND discovery_mode IN ('on_demand', 'both')",
    )
    rows = await cur.fetchall()
    await cur.close()
    return [
        {"source_id": s, "source_type": t, "endpoint": e, "auth_env": a,
         "intent_keywords": intent_keywords}
        for s, t, e, a in rows
    ]


async def _ingest_source_capped(source_row: dict, *,
                                artifact_cap: int = ON_DEMAND_ARTIFACT_CAP) -> dict:
    """Phase 1/3 ingest pipeline with a per-run artifact cap."""
    from yalayut.discovery import fetch as yal_fetch
    return await yal_fetch.ingest_all(source_row, artifact_cap=artifact_cap)


async def on_demand_discovery(demand: dict) -> dict:
    """Fetch untrusted sources for one demand signal. Returns a summary dict."""
    pattern = demand.get("source_step_pattern", "")
    keywords = list(demand.get("intent_keywords") or [])
    summary = {
        "pattern": pattern,
        "sources_scanned": 0,
        "artifacts_ingested": 0,
        "artifacts_enabled": 0,
        "artifacts_quarantined": 0,
        "errors": [],
    }
    sources = await _untrusted_sources_for(keywords)
    for src in sources:
        try:
            res = await _ingest_source_capped(
                src, artifact_cap=ON_DEMAND_ARTIFACT_CAP)
            summary["sources_scanned"] += 1
            summary["artifacts_ingested"] += int(res.get("ingested", 0))
            summary["artifacts_enabled"] += int(res.get("enabled", 0))
            summary["artifacts_quarantined"] += int(res.get("quarantined", 0))
        except Exception as e:  # noqa: BLE001
            logger.warning("on-demand ingest failed for %s: %s",
                            src["source_id"], e)
            summary["errors"].append(f"{src['source_id']}: {e}")

    # Consume the demand: every signal on this pattern is marked discovered
    # so the next pending_signals() sweep skips it.
    if pattern:
        await _demand.mark_discovered(pattern)
    logger.info("on_demand_discovery complete", pattern=pattern,
                ingested=summary["artifacts_ingested"])
    return summary
```

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_on_demand.py -x -q
```

- [ ] Commit:

```
rtk git add packages/yalayut/src/yalayut/discovery/on_demand.py tests/yalayut/test_phase4_on_demand.py
rtk git commit -m "feat(yalayut,p4): on_demand_discovery() — need-driven untrusted-source fetch"
```

---

### Task 5 — source_scout_scan() body

**Files:**
- Create: `packages/yalayut/src/yalayut/discovery/source_scout.py`
- Test: `tests/yalayut/test_phase4_source_scout.py`

`source_scout_scan()` proposes candidate sources from four signals (GitHub
trending in relevant topics, cross-refs in approved artifacts' READMEs, web
search on accumulated demand signals, founder-mentioned URLs), capped per day,
writing `yalayut_source_candidates` rows.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_source_scout.py`:

```python
import asyncio

import pytest

from src.infra.db import init_db, get_db
from yalayut.discovery import source_scout


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_source_scout_writes_candidates(loop, monkeypatch):
    async def _run():
        await init_db()

        async def _fake_github_trending():
            return [
                {"candidate_source_id": "github:acme/skills",
                 "source_type": "github_path",
                 "endpoint": "https://github.com/acme/skills",
                 "metadata_json": '{"description": "acme skills"}'},
            ]

        async def _empty():
            return []

        monkeypatch.setattr(source_scout, "_scan_github_trending",
                            _fake_github_trending)
        monkeypatch.setattr(source_scout, "_scan_readme_crossrefs", _empty)
        monkeypatch.setattr(source_scout, "_scan_demand_websearch", _empty)
        monkeypatch.setattr(source_scout, "_scan_founder_urls", _empty)

        result = await source_scout.source_scout_scan()
        assert result["candidates_proposed"] == 1
        db = await get_db()
        cur = await db.execute(
            "SELECT candidate_source_id, state FROM yalayut_source_candidates")
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == "github:acme/skills"
        assert row[1] == "pending"
    loop.run_until_complete(_run())


def test_source_scout_respects_daily_cap(loop, monkeypatch):
    async def _run():
        await init_db()

        async def _many():
            return [
                {"candidate_source_id": f"github:x/repo{i}",
                 "source_type": "github_path",
                 "endpoint": f"https://github.com/x/repo{i}",
                 "metadata_json": "{}"}
                for i in range(20)
            ]

        async def _empty():
            return []

        monkeypatch.setattr(source_scout, "_scan_github_trending", _many)
        monkeypatch.setattr(source_scout, "_scan_readme_crossrefs", _empty)
        monkeypatch.setattr(source_scout, "_scan_demand_websearch", _empty)
        monkeypatch.setattr(source_scout, "_scan_founder_urls", _empty)

        result = await source_scout.source_scout_scan()
        # default cap = 5
        assert result["candidates_proposed"] == source_scout.DAILY_CANDIDATE_CAP
    loop.run_until_complete(_run())


def test_source_scout_dedupes_existing(loop, monkeypatch):
    async def _run():
        await init_db()
        db = await get_db()
        await db.execute(
            "INSERT INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, endpoint, state, proposed_at) "
            "VALUES ('github:dup/repo', 'github_path', 'x', 'pending', "
            "datetime('now'))")
        # Also already an approved source — must not be re-proposed.
        await db.execute(
            "INSERT INTO yalayut_sources (source_id, source_type, endpoint, "
            "trusted, enabled, discovery_mode) "
            "VALUES ('github:known/repo', 'github_path', 'x', 1, 1, 'cron')")
        await db.commit()

        async def _candidates():
            return [
                {"candidate_source_id": "github:dup/repo",
                 "source_type": "github_path", "endpoint": "x",
                 "metadata_json": "{}"},
                {"candidate_source_id": "github:known/repo",
                 "source_type": "github_path", "endpoint": "x",
                 "metadata_json": "{}"},
                {"candidate_source_id": "github:fresh/repo",
                 "source_type": "github_path", "endpoint": "x",
                 "metadata_json": "{}"},
            ]

        async def _empty():
            return []

        monkeypatch.setattr(source_scout, "_scan_github_trending",
                            _candidates)
        monkeypatch.setattr(source_scout, "_scan_readme_crossrefs", _empty)
        monkeypatch.setattr(source_scout, "_scan_demand_websearch", _empty)
        monkeypatch.setattr(source_scout, "_scan_founder_urls", _empty)

        result = await source_scout.source_scout_scan()
        assert result["candidates_proposed"] == 1  # only github:fresh/repo
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL**:

```
timeout 60 pytest tests/yalayut/test_phase4_source_scout.py -x -q
```

- [ ] Create `packages/yalayut/src/yalayut/discovery/source_scout.py`:

```python
"""Yalayut Phase 4 — autonomous source scout.

``source_scout_scan()`` proposes candidate sources from four signals:
  1. GitHub trending in relevant topics
  2. cross-refs in approved artifacts' READMEs
  3. web search on accumulated demand signals
  4. founder-mentioned URLs (queued via /yalayut scout <url>)

Candidates are deduped against existing ``yalayut_sources`` and pending
``yalayut_source_candidates`` rows, capped per day, and written as
``pending`` rows for founder review. The scout NEVER auto-adds a source.
"""
from __future__ import annotations

import json

import httpx

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

logger = get_logger("yalayut.source_scout")

#: Spec — per-day candidate cap (default 5).
DAILY_CANDIDATE_CAP: int = 5

#: GitHub topics the scout watches for trending skill/agent repos.
SCOUT_TOPICS: tuple[str, ...] = (
    "claude-skill", "agent-skill", "mcp-server", "cookiecutter-template",
)


async def _scan_github_trending() -> list[dict]:
    """Signal 1 — GitHub trending repos in skill/agent topics."""
    out: list[dict] = []
    for topic in SCOUT_TOPICS:
        url = (f"https://api.github.com/search/repositories"
               f"?q=topic:{topic}&sort=stars&per_page=5")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    url, headers={"Accept": "application/vnd.github+json"})
                if resp.status_code != 200:
                    continue
                items = resp.json().get("items", [])
        except Exception as e:  # noqa: BLE001
            logger.debug("github trending scan failed (%s): %s", topic, e)
            continue
        for repo in items:
            full = repo.get("full_name", "")
            if not full:
                continue
            out.append({
                "candidate_source_id": f"github:{full}",
                "source_type": "github_topic",
                "endpoint": topic,
                "metadata_json": json.dumps({
                    "description": repo.get("description") or "",
                    "stars": repo.get("stargazers_count"),
                    "topic": topic,
                }),
            })
    return out


async def _scan_readme_crossrefs() -> list[dict]:
    """Signal 2 — GitHub URLs referenced in approved artifacts' READMEs."""
    # Approved artifacts' bodies are on disk under vendor/skills. Phase 1/3
    # stored body_excerpt in yalayut_index — scan it for github.com refs.
    import re
    db = await get_db()
    cur = await db.execute(
        "SELECT DISTINCT body_excerpt FROM yalayut_index "
        "WHERE enabled = 1 AND body_excerpt IS NOT NULL")
    rows = await cur.fetchall()
    await cur.close()
    ref_re = re.compile(r"github\.com/([\w.-]+/[\w.-]+)")
    seen: set[str] = set()
    out: list[dict] = []
    for (excerpt,) in rows:
        for m in ref_re.finditer(excerpt or ""):
            full = m.group(1).rstrip(".")
            if full in seen:
                continue
            seen.add(full)
            out.append({
                "candidate_source_id": f"github:{full}",
                "source_type": "github_path",
                "endpoint": f"https://github.com/{full}",
                "metadata_json": json.dumps({"via": "readme_crossref"}),
            })
    return out


async def _scan_demand_websearch() -> list[dict]:
    """Signal 3 — web search on accumulated high-confidence demand signals.

    Reads the top demand patterns and (best-effort) searches for source
    repos. Web search is delegated to vecihi; when vecihi is unavailable
    this returns []. The scout degrades gracefully — never crashes.
    """
    from yalayut.discovery import demand as _demand
    pending = await _demand.pending_signals(limit=3)
    out: list[dict] = []
    for sig in pending:
        if sig["stacked_confidence"] < 0.5:
            continue
        query = " ".join(sig["intent_keywords"][:4]) + " skill OR mcp github"
        try:
            from packages.vecihi import search as _vsearch  # type: ignore
        except Exception:
            try:
                import vecihi as _vsearch  # type: ignore
            except Exception:
                logger.debug("vecihi unavailable — demand websearch skipped")
                return out
        try:
            results = await _vsearch.search(query, limit=3)
        except Exception as e:  # noqa: BLE001
            logger.debug("demand websearch failed: %s", e)
            continue
        for r in results or []:
            u = r.get("url", "") if isinstance(r, dict) else ""
            if "github.com/" not in u:
                continue
            full = u.split("github.com/", 1)[1].strip("/").split("/")[:2]
            if len(full) != 2:
                continue
            out.append({
                "candidate_source_id": f"github:{'/'.join(full)}",
                "source_type": "github_path",
                "endpoint": u,
                "metadata_json": json.dumps({
                    "via": "demand_websearch",
                    "demand_pattern": sig["source_step_pattern"],
                }),
            })
    return out


async def _scan_founder_urls() -> list[dict]:
    """Signal 4 — founder-mentioned URLs queued by /yalayut scout <url>.

    They are written by admin.queue_scout_url() with state='founder_queued';
    this promotes them into the candidate proposal flow.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT id, candidate_source_id, source_type, endpoint, metadata_json "
        "FROM yalayut_source_candidates WHERE state = 'founder_queued'")
    rows = await cur.fetchall()
    await cur.close()
    out: list[dict] = []
    for cid, csid, stype, endpoint, meta in rows:
        out.append({
            "candidate_source_id": csid,
            "source_type": stype,
            "endpoint": endpoint,
            "metadata_json": meta or "{}",
            "_promote_row_id": cid,
        })
    return out


async def _already_known(candidate_source_id: str) -> bool:
    """True when this id is already an approved source or a pending/decided
    candidate (deduped — no double proposal)."""
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_sources WHERE source_id = ? LIMIT 1",
        (candidate_source_id,))
    if await cur.fetchone():
        await cur.close()
        return True
    await cur.close()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_source_candidates "
        "WHERE candidate_source_id = ? AND state IN "
        "('pending', 'approved', 'rejected') LIMIT 1",
        (candidate_source_id,))
    hit = await cur.fetchone()
    await cur.close()
    return hit is not None


async def source_scout_scan() -> dict:
    """Run all four scout signals; write up to DAILY_CANDIDATE_CAP new
    ``yalayut_source_candidates`` rows. Returns a summary dict."""
    summary = {"candidates_proposed": 0, "candidates_deduped": 0,
               "telegram_cards": []}
    raw: list[dict] = []
    for scanner in (_scan_github_trending, _scan_readme_crossrefs,
                    _scan_demand_websearch, _scan_founder_urls):
        try:
            raw.extend(await scanner())
        except Exception as e:  # noqa: BLE001
            logger.warning("scout scanner %s failed: %s",
                           scanner.__name__, e)

    db = await get_db()
    now = to_db(utc_now())
    proposed = 0
    for cand in raw:
        if proposed >= DAILY_CANDIDATE_CAP:
            break
        csid = cand["candidate_source_id"]
        promote_id = cand.get("_promote_row_id")
        if promote_id is None and await _already_known(csid):
            summary["candidates_deduped"] += 1
            continue
        if promote_id is not None:
            # founder-queued row → flip it to pending (it IS the candidate).
            await db.execute(
                "UPDATE yalayut_source_candidates SET state = 'pending', "
                "proposed_at = ? WHERE id = ?", (now, promote_id))
        else:
            await db.execute(
                "INSERT INTO yalayut_source_candidates "
                "(candidate_source_id, source_type, endpoint, metadata_json, "
                " state, proposed_at) VALUES (?, ?, ?, ?, 'pending', ?)",
                (csid, cand["source_type"], cand["endpoint"],
                 cand.get("metadata_json") or "{}", now))
        proposed += 1
        summary["telegram_cards"].append(csid)
    await db.commit()
    summary["candidates_proposed"] = proposed
    logger.info("source_scout_scan complete", proposed=proposed,
                deduped=summary["candidates_deduped"])
    return summary
```

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_source_scout.py -x -q
```

- [ ] Commit:

```
rtk git add packages/yalayut/src/yalayut/discovery/source_scout.py tests/yalayut/test_phase4_source_scout.py
rtk git commit -m "feat(yalayut,p4): source_scout_scan() — autonomous candidate-source proposal with daily cap"
```

---

### Task 6 — capture_hint() body + __init__ exports

**Files:**
- Modify: `packages/yalayut/src/yalayut/__init__.py`
- Create (or extend if Phase 1 already stubbed it): `packages/yalayut/src/yalayut/capture.py`
- Test: `tests/yalayut/test_phase4_capture_hint.py` (capture body only — post-hook wiring is Task 8)

`capture_hint(task, outcome)` is the internal-hint auto-capture: when a task
took 2+ iterations and succeeded, embed its description and upsert an
`internal_hint` row into `yalayut_index`. Replaces `skills.py` auto-capture.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_capture_hint.py`:

```python
import asyncio
import json

import pytest

from src.infra.db import init_db, get_db
import yalayut


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_capture_hint_inserts_internal_hint(loop):
    async def _run():
        await init_db()
        task = {
            "id": 999,
            "title": "Wire JWT auth into FastAPI",
            "description": "Add JWT bearer auth middleware to the API.",
            "agent_type": "coder",
        }
        outcome = {"status": "completed", "iterations": 3,
                   "result": json.dumps({"ok": True})}
        await yalayut.capture_hint(task, outcome)
        db = await get_db()
        cur = await db.execute(
            "SELECT artifact_type, kind, exposure_class, vet_tier, enabled "
            "FROM yalayut_index WHERE kind = 'internal_hint'")
        row = await cur.fetchone()
        await cur.close()
        assert row is not None
        assert row[0] == "skill"
        assert row[1] == "internal_hint"
        assert row[2] == "inject"
        assert row[3] == 0  # T0
        assert row[4] == 1  # enabled
    loop.run_until_complete(_run())


def test_capture_hint_skips_single_iteration(loop):
    async def _run():
        await init_db()
        before_db = await get_db()
        cur = await before_db.execute(
            "SELECT COUNT(*) FROM yalayut_index WHERE kind = 'internal_hint'")
        before = (await cur.fetchone())[0]
        await cur.close()
        # 1 iteration → no capture (nothing learned).
        await yalayut.capture_hint(
            {"id": 1, "title": "trivial", "description": "x",
             "agent_type": "coder"},
            {"status": "completed", "iterations": 1})
        cur = await before_db.execute(
            "SELECT COUNT(*) FROM yalayut_index WHERE kind = 'internal_hint'")
        after = (await cur.fetchone())[0]
        await cur.close()
        assert after == before
    loop.run_until_complete(_run())


def test_capture_hint_skips_failed_task(loop):
    async def _run():
        await init_db()
        await yalayut.capture_hint(
            {"id": 2, "title": "failed task", "description": "x",
             "agent_type": "coder"},
            {"status": "failed", "iterations": 4})
        db = await get_db()
        cur = await db.execute(
            "SELECT COUNT(*) FROM yalayut_index "
            "WHERE kind = 'internal_hint' AND name LIKE '%failed-task%'")
        assert (await cur.fetchone())[0] == 0
        await cur.close()
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL**:

```
timeout 60 pytest tests/yalayut/test_phase4_capture_hint.py -x -q
```

- [ ] Create `packages/yalayut/src/yalayut/capture.py`:

```python
"""Yalayut Phase 4 — internal-hint auto-capture.

``capture_hint(task, outcome)`` is the post-hook body that replaces the old
``src/memory/skills.py`` auto-capture. When a task succeeded after 2+
iterations (i.e. something non-trivial was learned), it embeds the task
description and upserts an ``internal_hint`` artifact into ``yalayut_index``.
"""
from __future__ import annotations

import re

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

logger = get_logger("yalayut.capture")

#: Minimum iterations for a task to be worth capturing as a hint.
MIN_ITERATIONS_FOR_CAPTURE: int = 2


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:60] or "hint"


async def capture_hint(task: dict, outcome: dict) -> None:
    """Capture a successful 2+-iteration task as an internal_hint artifact.

    No-op when the task failed or completed in a single iteration — there is
    no transferable strategy to capture from a trivially-solved task.
    """
    if (outcome or {}).get("status") != "completed":
        return
    iterations = int((outcome or {}).get("iterations") or 0)
    if iterations < MIN_ITERATIONS_FOR_CAPTURE:
        return

    title = (task or {}).get("title") or ""
    description = (task or {}).get("description") or ""
    if not title and not description:
        return

    name = f"internal-{_slug(title)}"
    body = f"{title}\n\n{description}".strip()

    # Embed the body for vector matching (multilingual-e5-base, 768d).
    embedding_blob = None
    try:
        from src.memory.embeddings import embed_text
        vec = await embed_text(body)
        import struct
        embedding_blob = struct.pack(f"{len(vec)}f", *vec)
    except Exception as e:  # noqa: BLE001 — capture must never crash on_finish
        logger.debug("capture_hint embedding failed: %s", e)

    db = await get_db()
    now = to_db(utc_now())
    # Upsert: a repeated pattern refreshes the existing row, not a duplicate.
    cur = await db.execute(
        "SELECT id FROM yalayut_index WHERE source = 'internal' AND name = ?",
        (name,))
    existing = await cur.fetchone()
    await cur.close()
    if existing:
        await db.execute(
            "UPDATE yalayut_index SET body_excerpt = ?, embedding = ?, "
            "vetted_at = ? WHERE id = ?",
            (body[:500], embedding_blob, now, existing[0]))
    else:
        await db.execute(
            "INSERT INTO yalayut_index "
            "(artifact_type, kind, source, owner, name, name_original, "
            " version, body_excerpt, embedding, vet_tier, exposure_class, "
            " applies_to, mechanizable, enabled, created_at, vetted_at) "
            "VALUES ('skill', 'internal_hint', 'internal', 'kutai', ?, ?, "
            " '1.0.0', ?, ?, 0, 'inject', 'execution', 0, 1, ?, ?)",
            (name, title[:120], body[:500], embedding_blob, now, now))
    await db.commit()
    logger.info("internal_hint captured", name=name, iterations=iterations)
```

- [ ] Modify `packages/yalayut/src/yalayut/__init__.py` — add lazy re-exports (append; keep existing Phase 1–3 exports):

```python
# ── Phase 4 — discovery + autonomy entry points ─────────────────────────


def daily_discovery() -> dict:
    """Mechanical-executor body: pull trusted cron-mode sources."""
    from yalayut.discovery.cron import daily_discovery as _impl
    return _impl()  # returns a coroutine — awaited by the executor


def on_demand_discovery(demand: dict) -> dict:
    """Need-driven fetch for one DemandSignal."""
    from yalayut.discovery.on_demand import on_demand_discovery as _impl
    return _impl(demand)


def source_scout_scan() -> dict:
    """Mechanical-executor body: propose candidate sources."""
    from yalayut.discovery.source_scout import source_scout_scan as _impl
    return _impl()


def capture_hint(task: dict, outcome: dict):
    """Post-hook body: internal_hint auto-capture."""
    from yalayut.capture import capture_hint as _impl
    return _impl(task, outcome)
```

> Note: the four functions return coroutines (the inner `_impl` calls are not
> awaited here — they cannot be, this is a sync wrapper). Executors `await`
> the return value. This matches the spec's `__init__.py` signatures, which
> document return *types* (`dict`, `None`) — the async nature is in the call
> contract.

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_capture_hint.py -x -q
```

- [ ] Commit:

```
rtk git add packages/yalayut/src/yalayut/capture.py packages/yalayut/src/yalayut/__init__.py tests/yalayut/test_phase4_capture_hint.py
rtk git commit -m "feat(yalayut,p4): capture_hint() internal-hint auto-capture + Phase 4 __init__ exports"
```

---

### Task 7 — mr_roboto executors (yalayut_discovery, source_scout, capture_hint)

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/executors/yalayut_discovery.py`
- Create: `packages/mr_roboto/src/mr_roboto/executors/source_scout.py`
- Create: `packages/mr_roboto/src/mr_roboto/executors/capture_hint.py`
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py`
- Test: `tests/yalayut/test_phase4_executors.py`

The three executors are leaf shims importing `yalayut`. They are registered as
`action ==` branches in `mr_roboto/__init__.py`, mirroring the `cost_pull`
pattern (`action in (...)` → `from mr_roboto.executors.X import run`).

#### Steps

- [ ] Create `tests/yalayut/test_phase4_executors.py`:

```python
import asyncio

import pytest

from src.infra.db import init_db
import mr_roboto


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_yalayut_discovery_executor_reachable(loop, monkeypatch):
    async def _run():
        await init_db()
        import yalayut

        async def _fake_daily():
            return {"sources_scanned": 2, "artifacts_ingested": 4}

        monkeypatch.setattr(yalayut, "daily_discovery",
                            lambda: _fake_daily())
        task = {"id": 1, "agent_type": "mechanical", "context": {},
                "payload": {"action": "yalayut_discovery", "mode": "daily"}}
        r = await mr_roboto.run(task)
        assert r.status == "completed"
        assert r.result["sources_scanned"] == 2
    loop.run_until_complete(_run())


def test_source_scout_executor_reachable(loop, monkeypatch):
    async def _run():
        await init_db()
        import yalayut

        async def _fake_scan():
            return {"candidates_proposed": 3}

        monkeypatch.setattr(yalayut, "source_scout_scan",
                            lambda: _fake_scan())
        task = {"id": 2, "agent_type": "mechanical", "context": {},
                "payload": {"action": "source_scout"}}
        r = await mr_roboto.run(task)
        assert r.status == "completed"
        assert r.result["candidates_proposed"] == 3
    loop.run_until_complete(_run())


def test_capture_hint_executor_reachable(loop, monkeypatch):
    async def _run():
        await init_db()
        import yalayut
        seen = {}

        async def _fake_capture(task, outcome):
            seen["task_id"] = task.get("id")
            seen["iterations"] = outcome.get("iterations")

        monkeypatch.setattr(yalayut, "capture_hint",
                            lambda t, o: _fake_capture(t, o))
        task = {"id": 3, "agent_type": "mechanical", "context": {},
                "payload": {"action": "capture_hint",
                            "source_task": {"id": 77, "title": "t",
                                            "description": "d"},
                            "outcome": {"status": "completed",
                                        "iterations": 3}}}
        r = await mr_roboto.run(task)
        assert r.status == "completed"
        assert seen["task_id"] == 77
        assert seen["iterations"] == 3
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL** (`unknown mechanical action`):

```
timeout 60 pytest tests/yalayut/test_phase4_executors.py -x -q
```

- [ ] Create `packages/mr_roboto/src/mr_roboto/executors/yalayut_discovery.py`:

```python
"""Yalayut Phase 4 — yalayut_discovery mechanical executor.

Dispatches the catalog-discovery pipeline. ``mode`` (payload):
  - ``daily``      → yalayut.daily_discovery()    (trusted cron-mode sources)
  - ``on_demand``  → yalayut.on_demand_discovery(demand)  (one DemandSignal)

Leaf shim — the only mr_roboto file that imports yalayut for discovery.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.yalayut_discovery")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    mode = (payload.get("mode") or "daily").lower()
    import yalayut
    try:
        if mode == "daily":
            return await yalayut.daily_discovery()
        if mode == "on_demand":
            demand = payload.get("demand") or {}
            if not demand:
                return {"ok": False, "reason": "on_demand mode needs a demand"}
            return await yalayut.on_demand_discovery(demand)
        return {"ok": False, "reason": f"unknown discovery mode: {mode!r}"}
    except Exception as e:  # noqa: BLE001
        logger.warning("yalayut_discovery executor failed: %s", e)
        return {"ok": False, "reason": str(e)}
```

- [ ] Create `packages/mr_roboto/src/mr_roboto/executors/source_scout.py`:

```python
"""Yalayut Phase 4 — source_scout mechanical executor.

Runs ``yalayut.source_scout_scan()`` — proposes candidate sources to the
founder. Leaf shim importing yalayut.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.source_scout")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    import yalayut
    try:
        return await yalayut.source_scout_scan()
    except Exception as e:  # noqa: BLE001
        logger.warning("source_scout executor failed: %s", e)
        return {"ok": False, "reason": str(e)}
```

- [ ] Create `packages/mr_roboto/src/mr_roboto/executors/capture_hint.py`:

```python
"""Yalayut Phase 4 — capture_hint mechanical executor.

The ``capture_hint`` post-hook routes here. Payload carries the source
task dict + its outcome; the executor calls ``yalayut.capture_hint``.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.capture_hint")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    source_task = payload.get("source_task") or {}
    outcome = payload.get("outcome") or {}
    import yalayut
    try:
        await yalayut.capture_hint(source_task, outcome)
        return {"ok": True, "captured": True}
    except Exception as e:  # noqa: BLE001 — post-hook must never DLQ the source
        logger.warning("capture_hint executor failed: %s", e)
        return {"ok": True, "captured": False, "reason": str(e)}
```

- [ ] Modify `packages/mr_roboto/src/mr_roboto/__init__.py` — add three dispatch branches near the `cost_pull` branch (the `action in (...)` block around line 3143). Insert before the final `unknown mechanical action` fallthrough:

```python
    if action == "yalayut_discovery":
        from mr_roboto.executors.yalayut_discovery import run as _yal_disc_run
        res = await _yal_disc_run(task)
        return MechResult(status="completed", result=res)

    if action == "source_scout":
        from mr_roboto.executors.source_scout import run as _scout_run
        res = await _scout_run(task)
        return MechResult(status="completed", result=res)

    if action == "capture_hint":
        from mr_roboto.executors.capture_hint import run as _capture_run
        res = await _capture_run(task)
        return MechResult(status="completed", result=res)
```

> `MechResult` is mr_roboto's existing result type — match whatever the file's
> other branches use (e.g. `cost_pull` at line 3143). If the local name differs,
> use the same constructor the neighbouring branches use.

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_executors.py -x -q
```

- [ ] Commit:

```
rtk git add packages/mr_roboto/src/mr_roboto/executors/yalayut_discovery.py packages/mr_roboto/src/mr_roboto/executors/source_scout.py packages/mr_roboto/src/mr_roboto/executors/capture_hint.py packages/mr_roboto/src/mr_roboto/__init__.py tests/yalayut/test_phase4_executors.py
rtk git commit -m "feat(yalayut,p4): mr_roboto executors — yalayut_discovery/source_scout/capture_hint registered + reachable"
```

---

### Task 8 — capture_hint post-hook registration + routing

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/posthooks.py`
- Modify: `packages/general_beckman/src/general_beckman/apply.py`
- Modify: `src/memory/skills.py`
- Test: extend `tests/yalayut/test_phase4_capture_hint.py`

Register `capture_hint` as a `PostHookSpec` and add the `apply.py` routing
branch so it dispatches to the mechanical lane. Auto-wired on every gradeable
task via `auto_wire_triggers=["*"]` so capture happens on every task finish.

#### Steps

- [ ] Append the post-hook routing test to `tests/yalayut/test_phase4_capture_hint.py`:

```python
def test_capture_hint_registered_in_post_hook_registry(loop):
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "capture_hint" in POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["capture_hint"]
    assert spec.verb == "capture_hint"
    # advisory — capture failure must not DLQ the source task.
    assert spec.default_severity == "warning"


def test_capture_hint_routes_to_mechanical(loop):
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.apply import RequestPostHook  # dataclass

    hook = RequestPostHook(kind="capture_hint", source_task_id=55)
    source = {"id": 55, "title": "the task", "agent_type": "coder"}
    source_ctx = {"title": "the task",
                  "description": "do the thing"}
    agent_type, payload = _posthook_agent_and_payload(hook, source, source_ctx)
    assert agent_type == "mechanical"
    assert payload["payload"]["action"] == "capture_hint"
    assert payload["payload"]["source_task"]["id"] == 55
```

> If `RequestPostHook`'s constructor signature differs (it may take more
> fields), adapt the test to the real dataclass — the assertion targets are
> `agent_type == "mechanical"` and `payload["payload"]["action"] == "capture_hint"`.

- [ ] Run it — expect **FAIL** (`capture_hint` not in registry):

```
timeout 60 pytest tests/yalayut/test_phase4_capture_hint.py -x -q
```

- [ ] Add the `capture_hint` entry to `POST_HOOK_REGISTRY` in `posthooks.py` (inside the dict, after `verify_artifacts`):

```python
    # Yalayut Phase 4 — internal-hint auto-capture. Replaces the old
    # src/memory/skills.py auto-capture. Auto-wired on every gradeable task
    # via the "*" trigger; the executor itself no-ops on <2-iteration or
    # failed tasks. Warning severity — a capture miss must never DLQ the
    # source task.
    "capture_hint": PostHookSpec(
        kind="capture_hint",
        verb="capture_hint",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=["*"],
        description=(
            "Yalayut P4 — capture a successful 2+-iteration task as an "
            "internal_hint artifact in yalayut_index. Mechanical; advisory."
        ),
    ),
```

- [ ] Add the routing branch to `_posthook_agent_and_payload` in `apply.py` (after the `verify_artifacts` branch, before `grounding`):

```python
    if a.kind == "capture_hint":
        # Yalayut Phase 4 — internal-hint auto-capture. Mechanical post-hook:
        # mr_roboto's capture_hint executor calls yalayut.capture_hint with
        # the source task + its outcome. Advisory — never fails the source.
        return ("mechanical", {
            "source_task_id": a.source_task_id,
            "posthook_kind": "capture_hint",
            "executor": "mechanical",
            "payload": {
                "action": "capture_hint",
                "source_task": {
                    "id": a.source_task_id,
                    "title": source_ctx.get("title") or source.get("title", ""),
                    "description": source_ctx.get("description")
                    or source.get("description", ""),
                    "agent_type": source.get("agent_type", ""),
                },
                "outcome": {
                    "status": source.get("status", "completed"),
                    "iterations": int(source_ctx.get("iterations") or 0),
                    "result": source.get("result", ""),
                },
            },
        })
```

- [ ] Modify `src/memory/skills.py` — remove the old auto-capture call site. Find the function that captures skills from 2+ iteration tasks (e.g. `capture_skill_from_task` / `auto_capture`) and replace its body with a deprecation no-op (keep the symbol so any lingering import doesn't break; the *read* shim — `get_skills_for_task` / `task["skills"]` filter — stays untouched):

```python
async def capture_skill_from_task(task: dict, outcome: dict) -> None:
    """DEPRECATED — superseded by the yalayut ``capture_hint`` post-hook
    (registered in general_beckman POST_HOOK_REGISTRY). This no-op remains
    only so stale imports don't break. Auto-capture now runs as a mechanical
    post-hook on every gradeable task finish. Safe to delete once no caller
    references it."""
    return None
```

> Use the actual function name present in `skills.py`. If the auto-capture is
> invoked from a different module (e.g. `base.py` / `apply.py`), delete that
> call site instead — grep `capture_skill` / `auto_capture` first and neutralise
> the single live call path. The `capture_hint` post-hook is now the sole
> capture path.

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_capture_hint.py -x -q
```

- [ ] Run the post-hook registry coverage test to confirm no regression:

```
timeout 60 pytest tests/i2p/test_post_hooks_registry_coverage.py -x -q
```

- [ ] Commit:

```
rtk git add packages/general_beckman/src/general_beckman/posthooks.py packages/general_beckman/src/general_beckman/apply.py src/memory/skills.py tests/yalayut/test_phase4_capture_hint.py
rtk git commit -m "feat(yalayut,p4): register capture_hint post-hook + route to mechanical; retire skills.py auto-capture"
```

---

### Task 9 — orchestrator periodic checks

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/cron_seed.py`
- Modify: `src/core/orchestrator.py`
- Test: `tests/yalayut/test_phase4_orchestrator_checks.py`

Add `_check_yalayut_discovery()` and `_check_source_scout()` to the orchestrator,
called from the `run_loop` pump, timestamp-gated, mirroring the existing Z6
throttle-counter pattern. They enqueue a plain dict via `beckman.enqueue` — the
orchestrator imports **zero** from yalayut. The `cron_seed` cadences are a
restart-survivable backstop (the in-memory `_check_*` gate resets on restart).

#### Steps

- [ ] Create `tests/yalayut/test_phase4_orchestrator_checks.py`:

```python
import asyncio
import time

import pytest

from src.core.orchestrator import Orchestrator


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_yalayut_discovery_check_enqueues_when_due(loop, monkeypatch):
    async def _run():
        enqueued = []

        async def _fake_enqueue(spec, **kw):
            enqueued.append(spec)
            return {"id": 1}

        import general_beckman
        monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

        orch = Orchestrator.__new__(Orchestrator)
        orch._last_yalayut_discovery = 0.0  # never run → due
        await orch._check_yalayut_discovery()
        assert len(enqueued) == 1
        spec = enqueued[0]
        assert spec["agent_type"] == "mechanical"
        assert spec["payload"]["action"] == "yalayut_discovery"
        # second call right away → NOT due, gated.
        await orch._check_yalayut_discovery()
        assert len(enqueued) == 1
    loop.run_until_complete(_run())


def test_source_scout_check_enqueues_when_due(loop, monkeypatch):
    async def _run():
        enqueued = []

        async def _fake_enqueue(spec, **kw):
            enqueued.append(spec)
            return {"id": 2}

        import general_beckman
        monkeypatch.setattr(general_beckman, "enqueue", _fake_enqueue)

        orch = Orchestrator.__new__(Orchestrator)
        orch._last_source_scout = time.time() - 100000  # long ago → due
        await orch._check_source_scout()
        assert len(enqueued) == 1
        assert enqueued[0]["payload"]["action"] == "source_scout"
    loop.run_until_complete(_run())


def test_check_does_not_crash_on_enqueue_error(loop, monkeypatch):
    async def _run():
        async def _boom(spec, **kw):
            raise RuntimeError("beckman down")

        import general_beckman
        monkeypatch.setattr(general_beckman, "enqueue", _boom)
        orch = Orchestrator.__new__(Orchestrator)
        orch._last_yalayut_discovery = 0.0
        # must swallow the error — pump must never crash on a periodic check.
        await orch._check_yalayut_discovery()
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL** (`_check_yalayut_discovery` does not exist):

```
timeout 60 pytest tests/yalayut/test_phase4_orchestrator_checks.py -x -q
```

- [ ] Add the two cadences to `INTERNAL_CADENCES` in `cron_seed.py` (append to the list):

```python
    # Yalayut Phase 4 — daily catalog discovery (trusted cron-mode sources).
    # Backstop cadence: the orchestrator's in-memory _check_yalayut_discovery
    # gate resets on restart; this scheduled_tasks row survives restarts.
    {
        "title": "yalayut_discovery",
        "description": "Daily yalayut catalog discovery — pull trusted sources",
        "interval_seconds": 86400,  # 24h
        "payload": {"_executor": "yalayut_discovery", "mode": "daily"},
    },
    # Yalayut Phase 4 — source-scout candidate proposal (daily).
    {
        "title": "source_scout",
        "description": "Daily yalayut source-scout — propose candidate sources",
        "interval_seconds": 86400,  # 24h
        "payload": {"_executor": "source_scout"},
    },
```

- [ ] Add the two methods to `Orchestrator` in `orchestrator.py` (insert after `_drop_running_future`, before `_dispatch`):

```python
    # ─── Yalayut Phase 4 periodic checks ─────────────────────────────────
    #
    # Mirror the _check_todo_reminders pattern: timestamp-gated, enqueue a
    # plain dict via beckman.enqueue. The orchestrator imports ZERO from
    # yalayut — the mechanical executor (action "yalayut_discovery" /
    # "source_scout") owns the yalayut import. The cron_seed cadence rows
    # are the restart-survivable backstop; these in-process checks give a
    # finer cadence and fire promptly after boot.

    _YALAYUT_DISCOVERY_INTERVAL_S: float = 86400.0   # 24h
    _SOURCE_SCOUT_INTERVAL_S: float = 86400.0        # 24h

    async def _check_yalayut_discovery(self) -> None:
        """Enqueue a yalayut daily-discovery mechanical task when due."""
        import time as _time
        last = getattr(self, "_last_yalayut_discovery", 0.0)
        now = _time.time()
        if now - last < self._YALAYUT_DISCOVERY_INTERVAL_S:
            return
        self._last_yalayut_discovery = now
        try:
            import general_beckman
            await general_beckman.enqueue(
                {
                    "agent_type": "mechanical",
                    "title": "Yalayut daily discovery",
                    "payload": {"action": "yalayut_discovery",
                                "mode": "daily"},
                },
                lane="mechanical",
            )
            logger.info("enqueued yalayut daily discovery task")
        except Exception as e:
            logger.warning("yalayut discovery enqueue failed: %s", e)

    async def _check_source_scout(self) -> None:
        """Enqueue a yalayut source-scout mechanical task when due."""
        import time as _time
        last = getattr(self, "_last_source_scout", 0.0)
        now = _time.time()
        if now - last < self._SOURCE_SCOUT_INTERVAL_S:
            return
        self._last_source_scout = now
        try:
            import general_beckman
            await general_beckman.enqueue(
                {
                    "agent_type": "mechanical",
                    "title": "Yalayut source scout",
                    "payload": {"action": "source_scout"},
                },
                lane="mechanical",
            )
            logger.info("enqueued yalayut source-scout task")
        except Exception as e:
            logger.warning("source-scout enqueue failed: %s", e)
```

- [ ] Initialise the two timestamps in `Orchestrator.__init__` (add to the existing `__init__` body):

```python
        # Yalayut Phase 4 — periodic-check gates. 0.0 → first pump tick after
        # boot fires both checks immediately, then they self-gate to 24h.
        self._last_yalayut_discovery: float = 0.0
        self._last_source_scout: float = 0.0
```

- [ ] Wire both checks into the `run_loop` pump. In `run_loop`, after the existing Z6 sweep block (right before `await asyncio.sleep(3)`), add:

```python
                # Yalayut Phase 4 — periodic discovery + source-scout checks.
                # Both are timestamp-gated internally; calling every tick is
                # cheap (a getattr + time comparison).
                try:
                    await self._check_yalayut_discovery()
                    await self._check_source_scout()
                except Exception as e:
                    logger.debug("yalayut periodic check skipped: %s", e)
```

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_orchestrator_checks.py -x -q
```

- [ ] Verify the orchestrator still imports cleanly:

```
python -c "from src.core.orchestrator import Orchestrator; print('ok')"
```

- [ ] Commit:

```
rtk git add src/core/orchestrator.py packages/general_beckman/src/general_beckman/cron_seed.py tests/yalayut/test_phase4_orchestrator_checks.py
rtk git commit -m "feat(yalayut,p4): orchestrator _check_yalayut_discovery + _check_source_scout periodic checks + cron backstop"
```

---

### Task 10 — policy observer

**Files:**
- Create: `packages/yalayut/src/yalayut/policy_observer.py`
- Test: `tests/yalayut/test_phase4_policy_observer.py`

`observe_and_propose()` scans recent vetting audit data for repeated unknown
shell tokens / domains that capped artifacts at T2, and writes
`yalayut_policy_proposals` rows so the founder can approve allowlist additions.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_policy_observer.py`:

```python
import asyncio
import json

import pytest

from src.infra.db import init_db, get_db
from yalayut import policy_observer


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


def test_observe_proposes_repeated_unknown_token(loop):
    async def _run():
        await init_db()
        db = await get_db()
        # 3 artifacts capped at T2 by the same unknown shell token 'wasp'.
        for i in range(3):
            await db.execute(
                "INSERT INTO yalayut_index "
                "(artifact_type, kind, source, name, version, vet_tier, "
                " exposure_class, check_max_json, enabled, created_at) "
                "VALUES ('skill', 'shell_recipe', 'github:x/y', ?, '1.0.0', "
                " 2, 'inject', ?, 1, datetime('now'))",
                (f"artifact-{i}",
                 json.dumps({"shell_allowlist": {"tier": 2,
                                                 "unknown_token": "wasp"}})))
        await db.commit()
        n = await policy_observer.observe_and_propose()
        assert n >= 1
        cur = await db.execute(
            "SELECT check_name, key, state FROM yalayut_policy_proposals "
            "WHERE key = 'wasp'")
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == "shell_allowlist"
        assert row[2] == "pending"
    loop.run_until_complete(_run())


def test_observe_skips_below_threshold(loop):
    async def _run():
        await init_db()
        db = await get_db()
        # only 1 occurrence — below the propose threshold (3).
        await db.execute(
            "INSERT INTO yalayut_index "
            "(artifact_type, kind, source, name, version, vet_tier, "
            " exposure_class, check_max_json, enabled, created_at) "
            "VALUES ('skill', 'shell_recipe', 'github:x/y', 'lone', '1.0.0', "
            " 2, 'inject', ?, 1, datetime('now'))",
            (json.dumps({"shell_allowlist": {"tier": 2,
                                             "unknown_token": "rare"}}),))
        await db.commit()
        await policy_observer.observe_and_propose()
        cur = await db.execute(
            "SELECT COUNT(*) FROM yalayut_policy_proposals WHERE key = 'rare'")
        assert (await cur.fetchone())[0] == 0
        await cur.close()
    loop.run_until_complete(_run())


def test_observe_idempotent(loop):
    async def _run():
        await init_db()
        db = await get_db()
        for i in range(3):
            await db.execute(
                "INSERT INTO yalayut_index "
                "(artifact_type, kind, source, name, version, vet_tier, "
                " exposure_class, check_max_json, enabled, created_at) "
                "VALUES ('skill', 'shell_recipe', 'github:x/y', ?, '1.0.0', "
                " 2, 'inject', ?, 1, datetime('now'))",
                (f"idem-{i}",
                 json.dumps({"shell_allowlist": {"tier": 2,
                                                 "unknown_token": "idem"}})))
        await db.commit()
        await policy_observer.observe_and_propose()
        await policy_observer.observe_and_propose()
        cur = await db.execute(
            "SELECT COUNT(*) FROM yalayut_policy_proposals WHERE key = 'idem'")
        assert (await cur.fetchone())[0] == 1  # no duplicate proposal
        await cur.close()
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL**:

```
timeout 60 pytest tests/yalayut/test_phase4_policy_observer.py -x -q
```

- [ ] Create `packages/yalayut/src/yalayut/policy_observer.py`:

```python
"""Yalayut Phase 4 — policy observer.

KutAI proposes allowlist additions from observation. ``observe_and_propose``
scans ``yalayut_index.check_max_json`` audit data for unknown shell tokens
(and domains) that capped 3+ artifacts at T2, and writes pending
``yalayut_policy_proposals`` rows. The founder approves via Telegram
(``/yalayut policy review``). The observer NEVER mutates ``yalayut_policy``.
"""
from __future__ import annotations

import json
from collections import Counter

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

logger = get_logger("yalayut.policy_observer")

#: How many artifacts must hit the same unknown token before we propose it.
PROPOSE_THRESHOLD: int = 3


async def _proposal_exists(check_name: str, key: str) -> bool:
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_policy_proposals "
        "WHERE check_name = ? AND key = ? AND state = 'pending' LIMIT 1",
        (check_name, key))
    hit = await cur.fetchone()
    await cur.close()
    return hit is not None


async def _already_policy(check_name: str, key: str) -> bool:
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_policy WHERE check_name = ? AND key = ? LIMIT 1",
        (check_name, key))
    hit = await cur.fetchone()
    await cur.close()
    return hit is not None


async def observe_and_propose() -> int:
    """Scan vetting audit data; write policy proposals. Returns the count of
    new proposals written."""
    db = await get_db()
    cur = await db.execute(
        "SELECT name, check_max_json FROM yalayut_index "
        "WHERE check_max_json IS NOT NULL")
    rows = await cur.fetchall()
    await cur.close()

    # (check_name, key) -> [artifact names that hit it]
    hits: dict[tuple[str, str], list[str]] = {}
    for name, cmj in rows:
        try:
            checks = json.loads(cmj or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        for check_name, detail in (checks or {}).items():
            if not isinstance(detail, dict):
                continue
            token = detail.get("unknown_token") or detail.get("unknown_domain")
            if not token:
                continue
            key = (check_name, str(token))
            hits.setdefault(key, []).append(name)

    written = 0
    now = to_db(utc_now())
    for (check_name, token), artifacts in hits.items():
        if len(artifacts) < PROPOSE_THRESHOLD:
            continue
        if await _already_policy(check_name, token):
            continue
        if await _proposal_exists(check_name, token):
            continue
        await db.execute(
            "INSERT INTO yalayut_policy_proposals "
            "(check_name, key, proposed_value, evidence_json, state, "
            " proposed_at) VALUES (?, ?, 'allow', ?, 'pending', ?)",
            (check_name, token,
             json.dumps({"artifacts": artifacts,
                         "occurrences": len(artifacts)}),
             now))
        written += 1
        logger.info("policy proposal written", check=check_name,
                    key=token, occurrences=len(artifacts))
    await db.commit()
    return written
```

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_policy_observer.py -x -q
```

- [ ] Commit:

```
rtk git add packages/yalayut/src/yalayut/policy_observer.py tests/yalayut/test_phase4_policy_observer.py
rtk git commit -m "feat(yalayut,p4): policy observer — propose allowlist additions from vetting audit data"
```

---

### Task 11 — admin.py founder-ops module

**Files:**
- Create: `packages/yalayut/src/yalayut/admin.py`
- Test: `tests/yalayut/test_phase4_admin.py`

`admin.py` is the founder-ops API the `/yalayut` Telegram group calls. Every
function the spec lists is implemented against the existing tables.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_admin.py`:

```python
import asyncio
import json

import pytest

from src.infra.db import init_db, get_db
from yalayut import admin


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


async def _seed_t2_artifact():
    db = await get_db()
    cur = await db.execute(
        "INSERT INTO yalayut_index "
        "(artifact_type, kind, source, owner, name, version, vet_tier, "
        " exposure_class, enabled, created_at) "
        "VALUES ('skill', 'prompt_skill', 'github:x/y', 'x', 'a-skill', "
        " '1.0.0', 2, 'quarantine', 0, datetime('now'))")
    await db.commit()
    return cur.lastrowid


def test_pending_artifacts_lists_t2(loop):
    async def _run():
        await init_db()
        aid = await _seed_t2_artifact()
        pend = await admin.pending_artifacts()
        assert any(p["id"] == aid for p in pend)
    loop.run_until_complete(_run())


def test_approve_artifact_enables(loop):
    async def _run():
        await init_db()
        aid = await _seed_t2_artifact()
        await admin.approve_artifact(aid)
        db = await get_db()
        cur = await db.execute(
            "SELECT enabled FROM yalayut_index WHERE id = ?", (aid,))
        assert (await cur.fetchone())[0] == 1
        await cur.close()
    loop.run_until_complete(_run())


def test_reject_artifact_disables(loop):
    async def _run():
        await init_db()
        aid = await _seed_t2_artifact()
        await admin.reject_artifact(aid)
        db = await get_db()
        cur = await db.execute(
            "SELECT enabled, vet_tier FROM yalayut_index WHERE id = ?", (aid,))
        row = await cur.fetchone()
        await cur.close()
        assert row[0] == 0
    loop.run_until_complete(_run())


def test_approve_source_creates_source_row(loop):
    async def _run():
        await init_db()
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, endpoint, state, proposed_at) "
            "VALUES ('github:new/src', 'github_path', 'https://x', "
            "'pending', datetime('now'))")
        await db.commit()
        cand_id = cur.lastrowid
        await admin.approve_source(cand_id, trusted=True)
        cur = await db.execute(
            "SELECT trusted, enabled FROM yalayut_sources "
            "WHERE source_id = 'github:new/src'")
        row = await cur.fetchone()
        await cur.close()
        assert row == (1, 1)
    loop.run_until_complete(_run())


def test_decide_policy_approve_writes_policy_row(loop):
    async def _run():
        await init_db()
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO yalayut_policy_proposals "
            "(check_name, key, proposed_value, state, proposed_at) "
            "VALUES ('shell_allowlist', 'wasp', 'allow', 'pending', "
            "datetime('now'))")
        await db.commit()
        pid = cur.lastrowid
        await admin.decide_policy(pid, approve=True)
        cur = await db.execute(
            "SELECT value FROM yalayut_policy "
            "WHERE check_name = 'shell_allowlist' AND key = 'wasp'")
        assert (await cur.fetchone())[0] == "allow"
        await cur.close()
    loop.run_until_complete(_run())


def test_set_secret_encrypts(loop):
    async def _run():
        await init_db()
        await admin.set_secret("TEST_API_KEY", "supersecret")
        db = await get_db()
        cur = await db.execute(
            "SELECT encrypted_value FROM yalayut_secrets "
            "WHERE key_name = 'TEST_API_KEY'")
        row = await cur.fetchone()
        await cur.close()
        # stored encrypted — never the plaintext.
        assert row is not None
        assert b"supersecret" not in (row[0] or b"")
    loop.run_until_complete(_run())


def test_stats_returns_counts(loop):
    async def _run():
        await init_db()
        await _seed_t2_artifact()
        stats = await admin.stats()
        assert "tier_counts" in stats
        assert "exposure_class_counts" in stats
    loop.run_until_complete(_run())
```

- [ ] Run it — expect **FAIL**:

```
timeout 60 pytest tests/yalayut/test_phase4_admin.py -x -q
```

- [ ] Create `packages/yalayut/src/yalayut/admin.py`:

```python
"""Yalayut Phase 4 — founder-ops module.

Backs the ``/yalayut`` Telegram command group. Imported only by
``src/app/telegram_bot.py``. Every function is async and operates directly
on the yalayut tables — no LLM, no exposure logic.
"""
from __future__ import annotations

import json
import os

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from src.infra.times import utc_now, to_db

logger = get_logger("yalayut.admin")


# ─── Artifact vetting ───────────────────────────────────────────────────

async def pending_artifacts() -> list[dict]:
    """T2 artifacts awaiting founder promotion (quarantined-until-promoted)."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, name, name_original, source, owner, kind, vet_tier "
        "FROM yalayut_index WHERE vet_tier = 2 AND enabled = 0 "
        "ORDER BY created_at DESC")
    rows = await cur.fetchall()
    await cur.close()
    return [
        {"id": r[0], "name": r[1], "name_original": r[2], "source": r[3],
         "owner": r[4], "kind": r[5], "vet_tier": r[6]}
        for r in rows
    ]


async def approve_artifact(artifact_id: int) -> dict:
    """Promote a T2 artifact: enable it (founder accepts the risk)."""
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 1, vetted_at = ? WHERE id = ?",
        (to_db(utc_now()), artifact_id))
    await db.commit()
    logger.info("artifact approved", artifact_id=artifact_id)
    return {"ok": True, "artifact_id": artifact_id}


async def reject_artifact(artifact_id: int) -> dict:
    """Reject an artifact: disable it. Never deleted (spec — re-enableable)."""
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 0 WHERE id = ?", (artifact_id,))
    await db.commit()
    logger.info("artifact rejected", artifact_id=artifact_id)
    return {"ok": True, "artifact_id": artifact_id}


async def requeue(artifact_id: int) -> dict:
    """Re-enable a T3-quarantined artifact for re-vetting."""
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 1, vet_state = 'requeued' "
        "WHERE id = ?", (artifact_id,))
    await db.commit()
    logger.info("artifact requeued", artifact_id=artifact_id)
    return {"ok": True, "artifact_id": artifact_id}


async def disable(artifact_id: int) -> dict:
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 0 WHERE id = ?", (artifact_id,))
    await db.commit()
    return {"ok": True, "artifact_id": artifact_id}


async def enable(artifact_id: int) -> dict:
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_index SET enabled = 1 WHERE id = ?", (artifact_id,))
    await db.commit()
    return {"ok": True, "artifact_id": artifact_id}


# ─── Source candidates ──────────────────────────────────────────────────

async def pending_sources() -> list[dict]:
    """Source-scout proposals awaiting founder decision."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, candidate_source_id, source_type, endpoint, metadata_json "
        "FROM yalayut_source_candidates WHERE state = 'pending' "
        "ORDER BY proposed_at DESC")
    rows = await cur.fetchall()
    await cur.close()
    out = []
    for r in rows:
        try:
            meta = json.loads(r[4] or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        out.append({"id": r[0], "candidate_source_id": r[1],
                    "source_type": r[2], "endpoint": r[3], "metadata": meta})
    return out


async def approve_source(candidate_id: int, *, trusted: bool) -> dict:
    """Approve a candidate → create a ``yalayut_sources`` row.

    Trusted sources get ``discovery_mode='cron'`` (daily pull); untrusted
    get ``discovery_mode='on_demand'`` (only pulled on a demand signal)."""
    db = await get_db()
    cur = await db.execute(
        "SELECT candidate_source_id, source_type, endpoint "
        "FROM yalayut_source_candidates WHERE id = ?", (candidate_id,))
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return {"ok": False, "reason": "candidate not found"}
    source_id, source_type, endpoint = row
    mode = "cron" if trusted else "on_demand"
    trust_score = 0.7 if trusted else 0.3
    await db.execute(
        "INSERT OR IGNORE INTO yalayut_sources "
        "(source_id, source_type, endpoint, trust_score, discovery_mode, "
        " trusted, enabled, min_interval_s) "
        "VALUES (?, ?, ?, ?, ?, ?, 1, 86400)",
        (source_id, source_type, endpoint, trust_score, mode,
         1 if trusted else 0))
    await db.execute(
        "UPDATE yalayut_source_candidates SET state = 'approved', "
        "decided_at = ? WHERE id = ?", (to_db(utc_now()), candidate_id))
    await db.commit()
    logger.info("source approved", source_id=source_id, trusted=trusted)
    return {"ok": True, "source_id": source_id, "trusted": trusted}


async def reject_source(candidate_id: int) -> dict:
    db = await get_db()
    await db.execute(
        "UPDATE yalayut_source_candidates SET state = 'rejected', "
        "decided_at = ? WHERE id = ?", (to_db(utc_now()), candidate_id))
    await db.commit()
    return {"ok": True, "candidate_id": candidate_id}


async def promote_source(source_id: str, tier: int) -> dict:
    """Manual source trust promotion (spec — promotion always manual)."""
    db = await get_db()
    new_score = {0: 0.9, 1: 0.6, 2: 0.3}.get(int(tier), 0.3)
    await db.execute(
        "UPDATE yalayut_sources SET trust_score = ?, trusted = ? "
        "WHERE source_id = ?",
        (new_score, 1 if int(tier) == 0 else 0, source_id))
    await db.commit()
    logger.info("source promoted", source_id=source_id, tier=tier)
    return {"ok": True, "source_id": source_id, "tier": tier}


async def promote_owner(owner_name: str) -> dict:
    """Trust an owner — all future sources from them inherit the elevation."""
    db = await get_db()
    cur = await db.execute(
        "SELECT 1 FROM yalayut_owners WHERE owner_id = ?", (owner_name,))
    exists = await cur.fetchone()
    await cur.close()
    if exists:
        await db.execute(
            "UPDATE yalayut_owners SET trust_score = 0.9 WHERE owner_id = ?",
            (owner_name,))
    else:
        await db.execute(
            "INSERT INTO yalayut_owners (owner_id, trust_score, notes) "
            "VALUES (?, 0.9, 'founder-promoted')", (owner_name,))
    await db.commit()
    logger.info("owner promoted", owner=owner_name)
    return {"ok": True, "owner": owner_name}


async def queue_scout_url(url: str) -> dict:
    """Founder-mentioned candidate source (/yalayut scout <url>).

    Written with state='founder_queued'; the next source_scout_scan()
    promotes it into the pending proposal flow."""
    db = await get_db()
    owner = url.split("//", 1)[-1].split("/", 1)[0]
    cand_id = f"web:{url}"
    await db.execute(
        "INSERT OR IGNORE INTO yalayut_source_candidates "
        "(candidate_source_id, source_type, endpoint, metadata_json, state, "
        " proposed_at) VALUES (?, 'web_markdown', ?, ?, 'founder_queued', ?)",
        (cand_id, url, json.dumps({"via": "founder", "owner": owner}),
         to_db(utc_now())))
    await db.commit()
    return {"ok": True, "candidate_source_id": cand_id}


# ─── Policy proposals ───────────────────────────────────────────────────

async def policy_proposals() -> list[dict]:
    db = await get_db()
    cur = await db.execute(
        "SELECT id, check_name, key, proposed_value, evidence_json "
        "FROM yalayut_policy_proposals WHERE state = 'pending' "
        "ORDER BY proposed_at DESC")
    rows = await cur.fetchall()
    await cur.close()
    out = []
    for r in rows:
        try:
            ev = json.loads(r[4] or "{}")
        except (json.JSONDecodeError, TypeError):
            ev = {}
        out.append({"id": r[0], "check_name": r[1], "key": r[2],
                    "proposed_value": r[3], "evidence": ev})
    return out


async def propose_policy(check_name: str, key: str,
                         value: str = "allow") -> dict:
    """Founder-initiated policy proposal (/yalayut policy add)."""
    db = await get_db()
    await db.execute(
        "INSERT INTO yalayut_policy_proposals "
        "(check_name, key, proposed_value, evidence_json, state, proposed_at) "
        "VALUES (?, ?, ?, '{\"via\": \"founder\"}', 'pending', ?)",
        (check_name, key, value, to_db(utc_now())))
    await db.commit()
    return {"ok": True, "check_name": check_name, "key": key}


async def decide_policy(proposal_id: int, *, approve: bool) -> dict:
    """Approve → write the yalayut_policy row. Reject → just mark decided."""
    db = await get_db()
    cur = await db.execute(
        "SELECT check_name, key, proposed_value FROM yalayut_policy_proposals "
        "WHERE id = ?", (proposal_id,))
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return {"ok": False, "reason": "proposal not found"}
    check_name, key, value = row
    state = "approved" if approve else "rejected"
    if approve:
        await db.execute(
            "INSERT OR REPLACE INTO yalayut_policy "
            "(check_name, key, value, added_by, added_at) "
            "VALUES (?, ?, ?, 'auto_proposal', ?)",
            (check_name, key, value, to_db(utc_now())))
    await db.execute(
        "UPDATE yalayut_policy_proposals SET state = ?, decided_at = ? "
        "WHERE id = ?", (state, to_db(utc_now()), proposal_id))
    await db.commit()
    logger.info("policy decided", proposal_id=proposal_id, approve=approve)
    return {"ok": True, "proposal_id": proposal_id, "state": state}


# ─── Auth / secrets ─────────────────────────────────────────────────────

async def missing_auth() -> list[dict]:
    """Artifacts blocked by a missing env var (env_status != 'ready')."""
    db = await get_db()
    cur = await db.execute(
        "SELECT id, name, env_status FROM yalayut_index "
        "WHERE env_status IS NOT NULL AND env_status != 'ready'")
    rows = await cur.fetchall()
    await cur.close()
    return [{"id": r[0], "name": r[1], "env_status": r[2]} for r in rows]


def _fernet():
    """Build a Fernet from the .env key. Raises if KATALOG_SECRET_KEY unset."""
    from cryptography.fernet import Fernet
    key = os.getenv("KATALOG_SECRET_KEY")
    if not key:
        raise RuntimeError("KATALOG_SECRET_KEY not set in .env")
    return Fernet(key.encode() if isinstance(key, str) else key)


async def set_secret(key_name: str, value: str) -> dict:
    """Encrypt + store a secret; re-vet artifacts that were missing it."""
    enc = _fernet().encrypt(value.encode())
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO yalayut_secrets "
        "(key_name, encrypted_value, added_at) VALUES (?, ?, ?)",
        (key_name, enc, to_db(utc_now())))
    # flip artifacts that were blocked solely on this env var → ready.
    await db.execute(
        "UPDATE yalayut_index SET env_status = 'ready' "
        "WHERE env_status = ?", (f"missing_{key_name}",))
    await db.commit()
    logger.info("secret set", key_name=key_name)
    return {"ok": True, "key_name": key_name}


# ─── MCP process control ────────────────────────────────────────────────

async def mcp_status() -> list[dict]:
    db = await get_db()
    cur = await db.execute(
        "SELECT p.artifact_id, i.name, p.pid, p.port, p.health, "
        "       p.last_probe_at "
        "FROM yalayut_mcp_processes p "
        "JOIN yalayut_index i ON i.id = p.artifact_id")
    rows = await cur.fetchall()
    await cur.close()
    return [
        {"artifact_id": r[0], "name": r[1], "pid": r[2], "port": r[3],
         "health": r[4], "last_probe_at": r[5]}
        for r in rows
    ]


async def mcp_restart(artifact_id: int) -> dict:
    """Restart an MCP process — delegates to the Phase 3 MCP plugin."""
    try:
        from yalayut.plugins.mcp import restart_process
        await restart_process(artifact_id)
        return {"ok": True, "artifact_id": artifact_id}
    except Exception as e:  # noqa: BLE001
        logger.warning("mcp_restart failed: %s", e)
        return {"ok": False, "reason": str(e)}


async def mcp_kill(artifact_id: int) -> dict:
    """Kill an MCP process — delegates to the Phase 3 MCP plugin."""
    try:
        from yalayut.plugins.mcp import kill_process
        await kill_process(artifact_id)
        return {"ok": True, "artifact_id": artifact_id}
    except Exception as e:  # noqa: BLE001
        logger.warning("mcp_kill failed: %s", e)
        return {"ok": False, "reason": str(e)}


# ─── Stats ──────────────────────────────────────────────────────────────

async def stats() -> dict:
    """Overview for /yalayut: counts by tier/type/exposure, queue depths."""
    db = await get_db()

    async def _counts(col: str) -> dict:
        cur = await db.execute(
            f"SELECT {col}, COUNT(*) FROM yalayut_index "
            f"WHERE enabled = 1 GROUP BY {col}")
        rows = await cur.fetchall()
        await cur.close()
        return {str(r[0]): r[1] for r in rows}

    tier_counts = await _counts("vet_tier")
    type_counts = await _counts("artifact_type")
    exposure_counts = await _counts("exposure_class")

    cur = await db.execute(
        "SELECT COUNT(*) FROM yalayut_index WHERE vet_tier = 2 AND enabled = 0")
    vet_queue = (await cur.fetchone())[0]
    await cur.close()
    cur = await db.execute(
        "SELECT COUNT(*) FROM yalayut_source_candidates WHERE state = 'pending'")
    source_queue = (await cur.fetchone())[0]
    await cur.close()
    cur = await db.execute(
        "SELECT COUNT(*) FROM yalayut_demand_signals "
        "WHERE resulted_in_discovery = 0")
    demand_backlog = (await cur.fetchone())[0]
    await cur.close()

    # exposure-class A/B from yalayut_usage.
    cur = await db.execute(
        "SELECT exposure_class, "
        "       SUM(CASE WHEN succeeded THEN 1 ELSE 0 END), COUNT(*) "
        "FROM yalayut_usage WHERE exposure_class IS NOT NULL "
        "GROUP BY exposure_class")
    ab_rows = await cur.fetchall()
    await cur.close()
    exposure_ab = {
        r[0]: {"succeeded": r[1] or 0, "total": r[2] or 0}
        for r in ab_rows
    }

    return {
        "tier_counts": tier_counts,
        "type_counts": type_counts,
        "exposure_class_counts": exposure_counts,
        "vet_queue_depth": vet_queue,
        "source_candidate_queue_depth": source_queue,
        "demand_signal_backlog": demand_backlog,
        "exposure_ab": exposure_ab,
    }
```

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_admin.py -x -q
```

- [ ] Commit:

```
rtk git add packages/yalayut/src/yalayut/admin.py tests/yalayut/test_phase4_admin.py
rtk git commit -m "feat(yalayut,p4): admin.py founder-ops — artifacts/sources/policy/auth/mcp/stats API"
```

---

### Task 12 — /yalayut Telegram command group + callbacks

**Files:**
- Modify: `src/app/telegram_bot.py`
- Test: `tests/yalayut/test_phase4_telegram.py`

Register the `/yalayut` command and a `yal:` callback branch. The command
dispatches on its first argument to the spec's subcommands; inline buttons
(vet / source-candidate / policy approval) route through `handle_callback`.

#### Steps

- [ ] Create `tests/yalayut/test_phase4_telegram.py`:

```python
import asyncio
from types import SimpleNamespace

import pytest

from src.infra.db import init_db, get_db
from src.app.telegram_bot import TelegramInterface


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


class _StubMessage:
    def __init__(self, text=""):
        self.text = text
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))
        return SimpleNamespace(message_id=1)


class _StubUpdate:
    def __init__(self, text=""):
        self.message = _StubMessage(text)
        self.effective_chat = SimpleNamespace(id=123)
        self.effective_user = SimpleNamespace(id=123)
        self.callback_query = None


def _telegram():
    # build the interface without a live bot — only handler bodies tested.
    return TelegramInterface.__new__(TelegramInterface)


def test_cmd_yalayut_overview(loop):
    async def _run():
        await init_db()
        tg = _telegram()
        update = _StubUpdate("/yalayut")
        ctx = SimpleNamespace(args=[])
        await tg.cmd_yalayut(update, ctx)
        assert update.message.replies
        text = update.message.replies[0][0]
        assert "yalayut" in text.lower() or "catalog" in text.lower()
    loop.run_until_complete(_run())


def test_cmd_yalayut_sources_pending(loop):
    async def _run():
        await init_db()
        db = await get_db()
        await db.execute(
            "INSERT INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, endpoint, state, proposed_at) "
            "VALUES ('github:a/b', 'github_path', 'x', 'pending', "
            "datetime('now'))")
        await db.commit()
        tg = _telegram()
        update = _StubUpdate("/yalayut sources pending")
        ctx = SimpleNamespace(args=["sources", "pending"])
        await tg.cmd_yalayut(update, ctx)
        assert update.message.replies
        assert "github:a/b" in update.message.replies[-1][0]
    loop.run_until_complete(_run())


def test_yalayut_callback_approve_source(loop):
    async def _run():
        await init_db()
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, endpoint, state, proposed_at) "
            "VALUES ('github:cb/src', 'github_path', 'x', 'pending', "
            "datetime('now'))")
        await db.commit()
        cand_id = cur.lastrowid

        tg = _telegram()
        answered = []
        edited = []
        cq = SimpleNamespace(
            data=f"yal:src_approve_trusted:{cand_id}",
            answer=lambda *a, **k: _async_noop(answered),
            edit_message_text=lambda *a, **k: _async_noop(edited),
            message=_StubMessage(),
        )
        update = SimpleNamespace(callback_query=cq,
                                 effective_chat=SimpleNamespace(id=123))
        await tg.handle_yalayut_callback(update, SimpleNamespace())
        cur = await db.execute(
            "SELECT trusted FROM yalayut_sources "
            "WHERE source_id = 'github:cb/src'")
        row = await cur.fetchone()
        await cur.close()
        assert row is not None and row[0] == 1
    loop.run_until_complete(_run())


def _async_noop(sink):
    async def _c():
        sink.append(True)
    return _c()
```

- [ ] Run it — expect **FAIL** (`cmd_yalayut` does not exist):

```
timeout 60 pytest tests/yalayut/test_phase4_telegram.py -x -q
```

- [ ] Register the command handler in `_setup_handlers()` (add near the other `CommandHandler` registrations, ~line 1893):

```python
        self.app.add_handler(CommandHandler("yalayut", self.cmd_yalayut))
```

- [ ] Add the `yal:` branch to `handle_callback()`. Find the callback dispatcher (it inspects `query.data`) and add an early branch:

```python
        if data.startswith("yal:"):
            await self.handle_yalayut_callback(update, context)
            return
```

- [ ] Add the `cmd_yalayut` dispatcher + `handle_yalayut_callback` methods to the `TelegramInterface` class (place near the other command handlers):

```python
    # ─── Yalayut Phase 4 — /yalayut command group ───────────────────────

    async def cmd_yalayut(self, update, context):
        """`/yalayut <subcommand> ...` — catalog ops surface.

        Subcommands (spec Telegram UX): (no args) overview · sources pending ·
        review <source> · pending · policy add|review · disable|enable|requeue
        <id> · source promote <id> <tier> · owner promote <name> · stats ·
        discover "<intent>" · scout <url> · auth missing · auth set <K>=<V> ·
        mcp status|restart|kill <id>.
        """
        from yalayut import admin
        from yalayut.discovery import demand as _demand
        args = list(getattr(context, "args", []) or [])
        sub = args[0] if args else ""

        try:
            if not sub:
                st = await admin.stats()
                lines = ["📚 *Yalayut catalog*"]
                lines.append(f"Tiers: {st['tier_counts']}")
                lines.append(f"Types: {st['type_counts']}")
                lines.append(f"Vet queue: {st['vet_queue_depth']}")
                lines.append(
                    f"Source candidates: {st['source_candidate_queue_depth']}")
                lines.append(
                    f"Demand backlog: {st['demand_signal_backlog']}")
                await self._reply(update, "\n".join(lines))
                return

            if sub == "sources" and len(args) > 1 and args[1] == "pending":
                pend = await admin.pending_sources()
                if not pend:
                    await self._reply(update, "No pending source candidates.")
                    return
                for p in pend[:10]:
                    text = (f"🔎 *Source candidate*\n`{p['candidate_source_id']}`"
                            f"\ntype: {p['source_type']}\n{p['metadata']}")
                    kb = self._yalayut_source_keyboard(p["id"])
                    await self._reply(update, text, reply_markup=kb)
                return

            if sub == "review" and len(args) > 1:
                # collapsed per-source digest = pending artifacts of a source.
                src = args[1]
                pend = [a for a in await admin.pending_artifacts()
                        if a["source"] == src]
                if not pend:
                    await self._reply(update, f"No pending artifacts for {src}.")
                    return
                names = "\n".join(f"• {a['name']} (id {a['id']})"
                                  for a in pend)
                await self._reply(update, f"Pending in {src}:\n{names}")
                return

            if sub == "pending":
                pend = await admin.pending_artifacts()
                if not pend:
                    await self._reply(update, "No T2 escalations pending.")
                    return
                for a in pend[:10]:
                    text = (f"📦 *{a['name']}*\nsource: {a['source']}\n"
                            f"kind: {a['kind']} · tier T{a['vet_tier']}")
                    kb = self._yalayut_vet_keyboard(a["id"])
                    await self._reply(update, text, reply_markup=kb)
                return

            if sub == "policy":
                if len(args) > 1 and args[1] == "review":
                    props = await admin.policy_proposals()
                    if not props:
                        await self._reply(update, "No policy proposals.")
                        return
                    for p in props[:10]:
                        text = (f"⚙️ *Policy proposal*\n"
                                f"{p['check_name']} → `{p['key']}` "
                                f"= {p['proposed_value']}\n{p['evidence']}")
                        kb = self._yalayut_policy_keyboard(p["id"])
                        await self._reply(update, text, reply_markup=kb)
                    return
                if len(args) > 3 and args[1] == "add":
                    await admin.propose_policy(args[2], args[3])
                    await self._reply(
                        update, f"Policy proposal queued: {args[2]}/{args[3]}")
                    return
                await self._reply(update,
                                  "Usage: /yalayut policy add <check> <key> "
                                  "| /yalayut policy review")
                return

            if sub in ("disable", "enable", "requeue") and len(args) > 1:
                aid = int(args[1])
                fn = {"disable": admin.disable, "enable": admin.enable,
                      "requeue": admin.requeue}[sub]
                await fn(aid)
                await self._reply(update, f"Artifact {aid}: {sub} done.")
                return

            if sub == "source" and len(args) > 3 and args[1] == "promote":
                await admin.promote_source(args[2], int(args[3]))
                await self._reply(update,
                                  f"Source {args[2]} promoted to T{args[3]}.")
                return

            if sub == "owner" and len(args) > 2 and args[1] == "promote":
                await admin.promote_owner(args[2])
                await self._reply(update, f"Owner {args[2]} promoted.")
                return

            if sub == "stats":
                st = await admin.stats()
                lines = ["📊 *Yalayut stats*"]
                for cls, ab in (st.get("exposure_ab") or {}).items():
                    tot = ab["total"] or 1
                    rate = 100.0 * ab["succeeded"] / tot
                    lines.append(f"{cls}: {ab['succeeded']}/{ab['total']} "
                                 f"({rate:.0f}%)")
                await self._reply(update, "\n".join(lines))
                return

            if sub == "discover" and len(args) > 1:
                intent = " ".join(args[1:]).strip('"')
                await _demand.record_signal(_demand.DemandSignal(
                    source_step_pattern=f"founder:{intent[:40]}",
                    intent_keywords=intent.split(),
                    signal_type="founder", confidence=0.8))
                # immediately enqueue an on-demand discovery for this intent.
                import general_beckman
                await general_beckman.enqueue(
                    {"agent_type": "mechanical",
                     "title": f"Yalayut discover: {intent[:40]}",
                     "payload": {"action": "yalayut_discovery",
                                 "mode": "on_demand",
                                 "demand": {
                                     "source_step_pattern":
                                         f"founder:{intent[:40]}",
                                     "intent_keywords": intent.split(),
                                     "stacked_confidence": 0.8}}},
                    lane="mechanical")
                await self._reply(update,
                                  f"Discovery queued for: {intent}")
                return

            if sub == "scout" and len(args) > 1:
                res = await admin.queue_scout_url(args[1])
                await self._reply(update,
                                  f"Scout URL queued: {res['candidate_source_id']}")
                return

            if sub == "auth":
                if len(args) > 1 and args[1] == "missing":
                    miss = await admin.missing_auth()
                    if not miss:
                        await self._reply(update,
                                          "No artifacts blocked on auth.")
                        return
                    txt = "\n".join(f"• {m['name']}: {m['env_status']}"
                                    for m in miss)
                    await self._reply(update, f"Missing auth:\n{txt}")
                    return
                if len(args) > 2 and args[1] == "set":
                    kv = args[2]
                    if "=" not in kv:
                        await self._reply(update,
                                          "Usage: /yalayut auth set KEY=VALUE")
                        return
                    k, v = kv.split("=", 1)
                    await admin.set_secret(k.strip(), v.strip())
                    await self._reply(update, f"Secret {k.strip()} stored.")
                    return
                await self._reply(update,
                                  "Usage: /yalayut auth missing "
                                  "| /yalayut auth set KEY=VALUE")
                return

            if sub == "mcp":
                if len(args) > 1 and args[1] == "status":
                    rows = await admin.mcp_status()
                    if not rows:
                        await self._reply(update, "No MCP servers running.")
                        return
                    txt = "\n".join(
                        f"• {r['name']}: {r['health']} (pid {r['pid']})"
                        for r in rows)
                    await self._reply(update, f"MCP servers:\n{txt}")
                    return
                if len(args) > 2 and args[1] in ("restart", "kill"):
                    aid = int(args[2])
                    fn = (admin.mcp_restart if args[1] == "restart"
                          else admin.mcp_kill)
                    res = await fn(aid)
                    await self._reply(update,
                                      f"MCP {args[1]} {aid}: "
                                      f"{'ok' if res.get('ok') else res}")
                    return
                await self._reply(update,
                                  "Usage: /yalayut mcp status|restart|kill")
                return

            await self._reply(update,
                              "Unknown subcommand. Try /yalayut for overview.")
        except Exception as e:  # noqa: BLE001
            await self._reply(update, f"⚠️ /yalayut error: {e}")

    # ── inline keyboards ────────────────────────────────────────────────

    def _yalayut_vet_keyboard(self, artifact_id: int):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "✅ Approve", callback_data=f"yal:vet_approve:{artifact_id}"),
            InlineKeyboardButton(
                "❌ Reject", callback_data=f"yal:vet_reject:{artifact_id}"),
        ]])

    def _yalayut_source_keyboard(self, candidate_id: int):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "Trust", callback_data=f"yal:src_approve_trusted:{candidate_id}"),
            InlineKeyboardButton(
                "Untrust",
                callback_data=f"yal:src_approve_untrusted:{candidate_id}"),
        ], [
            InlineKeyboardButton(
                "Reject", callback_data=f"yal:src_reject:{candidate_id}"),
        ]])

    def _yalayut_policy_keyboard(self, proposal_id: int):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "Approve", callback_data=f"yal:pol_approve:{proposal_id}"),
            InlineKeyboardButton(
                "Reject", callback_data=f"yal:pol_reject:{proposal_id}"),
        ]])

    async def handle_yalayut_callback(self, update, context):
        """Route `yal:<action>:<id>` inline-button taps."""
        from yalayut import admin
        query = update.callback_query
        data = query.data or ""
        parts = data.split(":")
        if len(parts) != 3:
            await query.answer("Bad callback")
            return
        _, action, raw_id = parts
        try:
            target_id = int(raw_id)
        except ValueError:
            await query.answer("Bad id")
            return

        try:
            if action == "vet_approve":
                await admin.approve_artifact(target_id)
                msg = f"Artifact {target_id} approved."
            elif action == "vet_reject":
                await admin.reject_artifact(target_id)
                msg = f"Artifact {target_id} rejected."
            elif action == "src_approve_trusted":
                await admin.approve_source(target_id, trusted=True)
                msg = f"Source candidate {target_id} approved (trusted)."
            elif action == "src_approve_untrusted":
                await admin.approve_source(target_id, trusted=False)
                msg = f"Source candidate {target_id} approved (untrusted)."
            elif action == "src_reject":
                await admin.reject_source(target_id)
                msg = f"Source candidate {target_id} rejected."
            elif action == "pol_approve":
                await admin.decide_policy(target_id, approve=True)
                msg = f"Policy proposal {target_id} approved."
            elif action == "pol_reject":
                await admin.decide_policy(target_id, approve=False)
                msg = f"Policy proposal {target_id} rejected."
            else:
                msg = f"Unknown yalayut action: {action}"
            await query.answer("Done")
            await query.edit_message_text(msg)
        except Exception as e:  # noqa: BLE001
            await query.answer("Error")
            await query.edit_message_text(f"⚠️ {e}")
```

> The `_reply()` helper already injects `REPLY_KEYBOARD` per the CLAUDE.md
> Telegram pattern — every reply above goes through `_reply()`. The
> `reply_markup=kb` calls for inline keyboards pass through `_reply()`'s
> `**kwargs`; verify `_reply` forwards `reply_markup` (it does — see line 501).
> Inline keyboards and the persistent `REPLY_KEYBOARD` do not conflict (one is
> attached to the message, the other is the chat-level keyboard).

- [ ] Run it — expect **PASS**:

```
timeout 60 pytest tests/yalayut/test_phase4_telegram.py -x -q
```

- [ ] Verify the bot module still imports cleanly:

```
python -c "from src.app.telegram_bot import TelegramInterface; print('ok')"
```

- [ ] Commit:

```
rtk git add src/app/telegram_bot.py tests/yalayut/test_phase4_telegram.py
rtk git commit -m "feat(yalayut,p4): /yalayut Telegram command group + yal: inline-button callbacks"
```

---

### Task 13 — Full Phase 4 suite + integration verification

**Files:**
- Test: all `tests/yalayut/test_phase4_*.py`

#### Steps

- [ ] Run the entire Phase 4 suite — expect all **PASS**:

```
timeout 120 pytest tests/yalayut/ -q
```

- [ ] Run the post-hook registry coverage + a Beckman apply smoke test to confirm no regression from the `capture_hint` registration:

```
timeout 60 pytest tests/i2p/test_post_hooks_registry_coverage.py packages/general_beckman/tests/ -q -x
```

- [ ] Verify all touched modules import cleanly:

```
python -c "import yalayut; import mr_roboto; from src.core.orchestrator import Orchestrator; from src.app.telegram_bot import TelegramInterface; from general_beckman.posthooks import POST_HOOK_REGISTRY; assert 'capture_hint' in POST_HOOK_REGISTRY; print('ok')"
```

- [ ] If anything fails, fix it before committing. Then commit:

```
rtk git add -A
rtk git commit -m "test(yalayut,p4): full Phase 4 suite green + integration verification"
```

---

## Self-review

Every Phase 4 spec requirement is mapped to a task. Verification of the
"no scaffolds, no unwired fragments" instruction:

| Spec requirement | Task | Wired? |
|---|---|---|
| `discovery/cron.py` `daily_discovery()` body | 3 | Yes — calls Phase 1/3 `fetch.ingest_all` per due source. |
| `discovery/on_demand.py` `on_demand_discovery(demand)` body | 4 | Yes — fetches untrusted sources, marks demand discovered. |
| `discovery/source_scout.py` `source_scout_scan()` body | 5 | Yes — 4 real scan signals, dedup, daily cap, writes candidate rows. |
| Mechanical executors `yalayut_discovery` + `source_scout` | 7 | Yes — registered as `action ==` branches in `mr_roboto/__init__.py`; Task 7 tests prove `mr_roboto.run()` reaches them. |
| `capture_hint` executor | 7 | Yes — registered + reachability test. |
| Orchestrator `_check_yalayut_discovery` / `_check_source_scout` | 9 | Yes — methods added, **called from `run_loop` pump**, timestamp-gated, enqueue plain dict, **zero yalayut import**; cron_seed backstop added. Task 9 tests prove enqueue fires when due and gates when not. |
| Adapters `github_topic` / `awesome_list_md` / `web_markdown` | 2 | Yes — full implementations with LLM-fallback synthesis. |
| `clawhub_api.py` stub | 2 | **Stub — spec-sanctioned.** `discover()` returns `[]`, documented in code + File Structure table. Explicitly the only stub in Phase 4. |
| LLM-fallback synthesis via `beckman.enqueue` | 2 | Yes — `synthesize.llm_synthesize` routes through `beckman.enqueue(lane="overhead")`; yalayut never imports `LLMDispatcher`. |
| `yalayut_demand_signals` + 7 signal types + stacking + dedupe + cooldown | 1 | Yes — `SIGNAL_TYPES` frozenset has all 7; `stack_confidence` implements `1-Π(1-c)`; `record_signal` dedupes within 7-day cooldown. |
| `capture_hint` post-hook kind registered | 8 | Yes — `PostHookSpec` in `POST_HOOK_REGISTRY`, `auto_wire_triggers=["*"]` (runs on every gradeable task finish), `apply.py` routes to mechanical. Old `skills.py` auto-capture retired. |
| `/yalayut` command group → `admin.py` | 11, 12 | Yes — every spec subcommand has a `cmd_yalayut` branch; `admin.py` implements every listed function. |
| Inline-button callbacks (vet / source / policy) | 12 | Yes — `yal:` prefix branch in `handle_callback`, `handle_yalayut_callback` routes all 7 callback actions; Task 12 test exercises one end-to-end. |
| Policy proposals flow | 10, 11 | Yes — `policy_observer.observe_and_propose()` writes proposals from observation; `admin.decide_policy` + Telegram `/yalayut policy review` close the loop. |

**Type / signature consistency with spec** (`Public APIs` section):

- `daily_discovery() -> dict` — Task 3 returns a summary dict. ✓
- `on_demand_discovery(demand: dict) -> dict` — Task 4 signature matches. ✓
- `source_scout_scan() -> dict` — Task 5 returns a summary dict. ✓
- `capture_hint(task: dict, outcome: dict) -> None` — Task 6 returns `None`. ✓
- `admin.py` API — Task 11 implements every name in the spec block
  (`pending_artifacts`, `approve_artifact`, `reject_artifact`, `requeue`,
  `pending_sources`, `approve_source`, `promote_source`, `promote_owner`,
  `policy_proposals`, `decide_policy`, `disable`, `enable`, `stats`,
  `missing_auth`, `set_secret`, `mcp_status`, `mcp_restart`, `mcp_kill`),
  plus `propose_policy`, `reject_source`, `queue_scout_url` needed by the
  Telegram surface. ✓
- Demand-signal schema — `record_signal` writes exactly the
  `yalayut_demand_signals` columns from the Phase 1 schema
  (`source_step_pattern`, `intent_keywords_json`, `signal_type`,
  `confidence`, `fired_at`, `resulted_in_discovery`). ✓

**No placeholders.** Every code block is complete, runnable Python. The only
designated stub is `clawhub_api.py`, sanctioned by the spec and the task
instruction.

### Spec ambiguities resolved inline

1. **`__init__.py` return types vs async.** The spec writes
   `daily_discovery() -> dict` but the bodies are necessarily `async`. Resolved:
   the `__init__.py` wrappers are thin sync functions that return the inner
   coroutine; executors `await` them. The spec's `-> dict` documents the
   *resolved* type, which holds.
2. **Where the orchestrator `_check_*` cadence is anchored.** The CLAUDE.md
   note says `_check_todo_reminders` exists, but in the current tree the todo
   reminder is a `cron_seed` cadence, not an orchestrator method. The task
   instruction explicitly requires orchestrator `_check_*` methods, so Phase 4
   adds them *and* a `cron_seed` backstop (the in-memory gate resets on
   restart; the `scheduled_tasks` row survives). Both fire the same mechanical
   action — idempotent, no double-work risk (discovery/scout are
   self-deduping).
3. **`capture_hint` auto-wire scope.** The spec says it "replaces the old
   skills.py auto-capture" but does not say how broadly it auto-wires.
   Resolved: `auto_wire_triggers=["*"]` (every gradeable task), matching the
   old auto-capture's universal scope; the executor itself no-ops on
   `<2-iteration` / failed tasks, so the universal trigger is cheap and correct.
4. **`capture_hint` severity.** Post-hooks default to `blocker`. A capture miss
   must never DLQ the source task. Resolved: `default_severity="warning"`, and
   the executor returns `ok=True` even on internal failure.
5. **On-demand "match by intent" granularity.** The spec says on-demand fetches
   "untrusted sources" for a demand. Resolved: the source-level filter is
   coarse (all untrusted on_demand sources); per-artifact keyword matching is
   the adapters' job (Phase 1/3 `ingest_all`), so `on_demand_discovery` passes
   `intent_keywords` through in the source row rather than pre-filtering
   sources — avoids missing a source whose name doesn't mention the keyword.
6. **Source-scout web-search dependency.** Signal 3 (web search on demand
   signals) needs a search backend. Resolved: delegated to `vecihi`
   (KutAI's web scraper) with graceful degradation — when `vecihi` is
   unavailable the scanner returns `[]` rather than crashing the scan.
```
