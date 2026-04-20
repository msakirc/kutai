# General Beckman

**Task master for autonomous AI pipelines. Decides what runs next, owns every task row.**

*Three-method public API. Pure rewrite rules. Unified cron table. No lanes, no tick, no handler registry.*

[English](#english) | [Turkce](#turkce)

---

<a id="english"></a>

## What is General Beckman?

General Beckman is the task-row authority for KutAI. It decides **which task to release next** from the queue and **what to do with each result** — including spawning follow-up tasks (subtasks, retries, clarifications, mission progression). Named after General Beckman in *Chuck*, the NSA commander who hands out missions.

```python
import general_beckman as beckman

# Called every orchestrator cycle (~3s). Sweeps the queue, fires due crons,
# picks one dispatchable task. Returns None when the queue is empty or
# the system is saturated.
task = await beckman.next_task()

# Called after every dispatched task returns. Routes the result through
# rewrite rules, applies actions to the DB. Every follow-up row originates here.
await beckman.on_task_finished(task_id, result)

# The one external write path — for user- or bot-initiated tasks
# (/task, /shop, etc.). Not used for result-driven or cron-driven tasks.
new_task_id = await beckman.enqueue(spec)
```

That's the entire surface. No public classes, no registries, no pub/sub, no `tick()`. Orchestrator calls those three functions and nothing more.

## Why General Beckman?

| | Beckman | Celery | RQ | Dramatiq |
|---|---|---|---|---|
| **Scope** | Single-process async task master | Distributed broker + workers | Redis-backed queue | Distributed with brokers |
| **Result handling** | Pure rewrite rules → DB actions | Callbacks / chains / groups | Return value / job status | Middleware chain |
| **Retry policy** | Typed `decide_retry` + quality bonus | Celery retry, exponential | `retry()` + `max_retries` | `Retries` middleware |
| **Cron** | Unified table (crontab + interval) | `celery beat` (separate process) | `rq-scheduler` | Periodic tasks |
| **Deps** | sqlite + salako + nerd_herd | Redis/RabbitMQ + kombu + celery | Redis | Redis/RabbitMQ |
| **Mission/workflow awareness** | Yes, via `MissionAdvance` action | Via chains/groups | Via dependent jobs | Via groups |
| **Queue hygiene** | Integrated sweep (stuck, dep cascade, escalation) | None (outside scope) | None | None |

Beckman is deliberately narrow: one SQLite DB, one async process, one pump. It does not solve distribution, multi-host scheduling, or cross-worker fairness. It solves **"given my current capacity signal, what task should run next and what should I do when it finishes?"**.

## Design principle

> **Orchestrator orchestrates. Beckman owns tasks. Hoca owns models. Nerd Herd owns utilization.**

Beckman answers *"should I dispatch one more?"* via a single system-busy bit from `nerd_herd.snapshot()`. No lanes, no partitioning. Model-aware concerns (swap budget, loaded-model affinity) live per-call inside `fatih_hoca.select()`. Workflow engine lives in its own package and is invoked via a thin `salako.workflow_advance` mechanical executor — never imported from here.

## Features

### Task selection (`next_task`)
- Age-based priority boost (+0.1/hour, cap +1.0) to prevent starvation
- Paused-pattern filter (DLQ category skip — `/dlq pause category:quality` excludes those tasks)
- Single system-busy bit from `nerd_herd.snapshot()` — mechanical tasks still dispatch under busy, LLM tasks hold
- Opportunistic sweep and cron fire in the same call (throttled internally)

### Result handling (`on_task_finished`)
- `route_result(task, result)` produces typed Action dataclasses
- `rewrite_actions(task, ctx, actions)` — pure policy rewrites (mission-task complete injects `MissionAdvance`, silent task clarify becomes `Failed`, workflow step subtasks blocked, clarification history reused)
- `apply_actions(task, actions)` — action → DB rows, one branch per type
- Retry / DLQ decisions via `retry.decide_retry` — typed output, pure policy

### Cron (unified table)
- Both user crons (`cron_expression`) and internal cadences (`interval_seconds`) live in one `scheduled_tasks` table
- 6 internal cadences seeded on first `next_task()`: `beckman_sweep`, `hoca_benchmark_refresh`, `nerd_herd_health_alert`, `todo_reminder`, `daily_digest`, `api_discovery`
- Marker payloads (`sweep`, `benchmark_refresh`, `nerd_herd_health`) dispatch internally; non-markers insert concrete task rows

### Queue hygiene (`sweep_queue` via marker)
- Stuck `processing` tasks (>5 min) → infra-reset up to 3 times, then fail
- Ungraded stuck >30 min → promote to completed (safety net)
- Pending tasks with all deps failed → cascade fail (unless any dep is in DLQ)
- `waiting_subtasks` with all children terminal → mark complete/failed
- Pending tasks with `next_retry_at` >1h overdue → clear the gate
- `waiting_human` escalation tiers (4h nudge / 24h tier 1 / 48h tier 2 / 72h cancel) — notifications via `salako.notify_user` tasks
- Workflow-level timeout → pause mission

### Retry policy (`retry.decide_retry`)
- Backoff table: `[0, 10, 30, 120, 600]` seconds — first retry immediate, subsequent back off
- Exhausted (attempts ≥ max) → DLQ, unless category is `quality` AND progress ≥ 0.5 AND fewer than 2 bonus attempts granted (then one bonus, `bonus_used=True`)
- DLQ writes to `dead_letter_tasks` table + spawn a `salako.notify_user` task with failure summary (no inline Telegram)

## Install

```bash
pip install -e ./packages/general_beckman
```

Requires `sqlite3`, `nerd_herd`, `salako` (listed in pyproject.toml).

## API

### Public functions

```python
async def next_task() -> Task | None
async def on_task_finished(task_id: int, result: dict) -> None
async def enqueue(spec: dict) -> int
```

### Action types (`result_router`)

```python
Complete(task_id, result, iterations, metadata, raw)
CompleteWithReusedAnswer(task_id, result, raw)
SpawnSubtasks(parent_task_id, subtasks, raw)
RequestClarification(task_id, question, chat_id, raw)
RequestReview(task_id, summary, raw)
Exhausted(task_id, error, raw)
Failed(task_id, error, raw)
MissionAdvance(task_id, mission_id, completed_task_id, raw)
```

### Retry decision

```python
decide_retry(failure: dict, progress: float | None = None,
             bonus_count: int = 0) -> RetryDecision | DLQAction

RetryDecision(action="immediate" | "delayed", delay_seconds=0, bonus_used=False)
DLQAction(category="unknown", reason="")
```

### Paused patterns (DLQ pause filter)

```python
from general_beckman.paused_patterns import pause, unpause, all_paused, is_paused

pause("category:quality")       # exclude tasks with error_category="quality"
unpause("category:quality")
patterns = all_paused()         # set[str]
is_paused(task_row)             # bool
```

## Architecture

```
general_beckman/
  ├── __init__.py          — public API (next_task / on_task_finished / enqueue)
  ├── types.py             — Task + AgentResult dataclass re-exports
  ├── queue.py             — pick_ready_task (priority boost + paused-pattern filter)
  ├── cron.py              — fire_due: scheduled_tasks processor, marker dispatch
  ├── cron_seed.py         — lazy-seeds 6 canonical internal cadences
  ├── sweep.py             — queue hygiene (stuck / cascade / escalation / workflow timeout)
  ├── rewrite.py           — pure action-rewriting rules (replaced result_guards.py)
  ├── apply.py             — action → DB rows (one branch per Action type)
  ├── retry.py             — decide_retry policy + DLQ decision
  ├── paused_patterns.py   — DLQ pause-pattern module state
  ├── result_router.py     — agent result → Action dataclasses
  └── task_context.py      — parse_context helper
```

Each module has one responsibility. `sweep.py` is the largest (~370 lines, seven distinct hygiene branches). Everything else is under 260 lines.

### Task flow — dispatch pump

```
orchestrator.run_loop (3s cycle)
  └─ task = await beckman.next_task()
       ├─ cron.fire_due()             — processes scheduled_tasks; markers dispatch
       │                                internally (sweep, benchmark refresh, health),
       │                                non-markers insert concrete task rows
       ├─ queue.pick_ready_task(busy) — age-boost sort → paused filter → claim first
       └─ returns Task | None
```

### Task flow — result handling

```
orchestrator._dispatch runs the agent/salako, then:
  beckman.on_task_finished(task_id, result)
    ├─ route_result(task, result)          — Action dataclasses
    ├─ rewrite_actions(task, ctx, actions) — pure policy:
    │                                         mission-task complete → +MissionAdvance
    │                                         silent task clarify → Failed
    │                                         workflow step subtasks → blocked
    │                                         clarification history → reused answer
    └─ apply_actions(task, actions)        — DB side-effects per action type
         ├─ Complete / CompleteWithReusedAnswer → update_task(status=completed)
         ├─ SpawnSubtasks → add_task × N + waiting_subtasks
         ├─ RequestClarification → salako.clarify task
         ├─ RequestReview → reviewer task (deduped)
         ├─ Exhausted / Failed → decide_retry → pending+backoff | DLQ+notify
         └─ MissionAdvance → salako.workflow_advance task
```

### Mission progression (no workflow-engine imports)

```
coder task #500 completes (mission_id=M)
  → on_task_finished → rewrite injects MissionAdvance
  → apply spawns mechanical task: {executor: workflow_advance, mission_id: M, ...}

Next cycle:
  orchestrator picks that task → salako.run(it)
  → salako.workflow_advance delegates to workflow_engine.advance()
  → advance() runs post_execute_workflow_step (artifact capture, phase check)
  → returns subtasks for phase N+1 → salako envelope {status: needs_subtasks}
  → on_task_finished → apply spawns those as SpawnSubtasks → DB rows
```

## What General Beckman does NOT do

- **LLM model selection** — that's `fatih_hoca`'s job.
- **Process management / llama-server** — that's `dallama`.
- **Cloud rate-limit tracking** — that's `kuleden_donen_var`.
- **Resource health checks** — that's `nerd_herd.health_summary`.
- **Workflow recipe / phase computation** — that's `workflow_engine`, invoked via `salako.workflow_advance`.
- **Telemetry push** — that lives at the dispatch observation point in `src/core/metrics_push.py`.
- **Telegram I/O** — all outbound messages go through `salako.clarify` / `salako.notify_user` mechanical tasks.

## License

MIT

---

<a id="turkce"></a>

## General Beckman nedir?

General Beckman, KutAI icin gorev satiri otoritesidir. Kuyruktan **hangi gorevi serbest birakacagini** ve **her sonucla ne yapacagini** belirler — alt gorevler, yeniden denemeler, soru-sorma ve misyon ilerletme dahil tum takip gorevlerini olusturur. Adini *Chuck* dizisindeki Genera Beckman'dan alir — NSA komutani, ajanlara gorev dagitir.

```python
import general_beckman as beckman

# Orchestrator dongusunde (~3s) her cagrida calisir. Kuyrugu temizler, vadesi
# gelmis cronlari ateslar, bir gorev secer. Kuyruk bossa veya sistem doluysa None.
task = await beckman.next_task()

# Her sonuclanan gorevden sonra cagrilir. Sonucu rewrite kurallarindan gecirir,
# DB'ye uygular. Tum takip satirlari buradan dogar.
await beckman.on_task_finished(task_id, result)

# Tek disaridan yazma yolu — kullanici veya bot baslatmali gorevler icin
# (/task, /shop vb.). Sonuc- veya cron-kaynakli gorevler icin kullanilmaz.
new_task_id = await beckman.enqueue(spec)
```

API yuzeyi bundan ibaret. Public class yok, registry yok, pub/sub yok, `tick()` yok. Orchestrator bu uc fonksiyonu cagirir, baska hicbirini.

## Neden General Beckman?

- **Uc metotluk dar yuzey** — Celery/RQ gibi full framework degil. Tek islemde, SQLite uzerinde, asenkron.
- **Saf rewrite kurallari** — sonuc-islem mantigi pure function, test edilebilir, side effect'siz.
- **Birlesik cron tablosu** — kullanici crontab ifadeleri ve dahili cadence'lar ayni `scheduled_tasks` tablosunda.
- **Misyon farkindaligi** — misyon gorevi tamamlandiginda `MissionAdvance` aksiyonu uretir, `salako.workflow_advance` mechanical gorevi dogurur. Workflow engine'i import etmez.
- **Entegre kuyruk temizligi** — sikismis task'lar, cascade basarisizliklar, waiting_human tirmandirma, workflow timeout — hepsi dahili `sweep_queue` ile.

## Tasarim ilkesi

> **Orchestrator orchestrate eder. Beckman gorevleri sahiplenir. Hoca modelleri bilir. Nerd Herd utilization'i bilir.**

Beckman *"bir tane daha gonderebilir miyim?"* sorusunu `nerd_herd.snapshot()`'tan gelen tek bir busy biti ile cevaplar. Lane yok, bolmecesiz. Model-farkindali kaygilar (swap budget, yuklu model affinity) per-call olarak `fatih_hoca.select()` icinde yasar. Workflow engine kendi paketinde, thin bir `salako.workflow_advance` mekanik executor araciligiyla cagrilir — buraya hic import edilmez.

## Ozellikler

### Gorev secimi (`next_task`)
- Yas tabanli oncelik artirma (+0.1/saat, max +1.0) — aclik onlenir
- Duraklamis desen filtresi (DLQ kategori atlama — `/dlq pause category:quality`)
- Nerd Herd'den tek busy biti — sistem mesgulken mechanical gorevler yine dispatch edilir, LLM gorevleri bekler
- Ayni cagri icinde firsatci sweep ve cron atesi (dahili throttled)

### Sonuc islem (`on_task_finished`)
- `route_result(task, result)` — typed Action dataclass'lari uretir
- `rewrite_actions(task, ctx, actions)` — pure policy rewrites
- `apply_actions(task, actions)` — her Action turu icin bir DB dali
- `retry.decide_retry` — typed cikti, pure policy

### Cron (birlesik tablo)
- Kullanici cronlari (`cron_expression`) ve dahili cadence'lar (`interval_seconds`) ayni `scheduled_tasks` tablosunda
- 6 dahili cadence ilk `next_task()` cagrisinda seed edilir: `beckman_sweep`, `hoca_benchmark_refresh`, `nerd_herd_health_alert`, `todo_reminder`, `daily_digest`, `api_discovery`
- Marker payload'lar (`sweep`, `benchmark_refresh`, `nerd_herd_health`) dahili dispatch, digerleri somut gorev satiri ekler

### Kuyruk temizligi (`sweep_queue` marker ile)
- `processing`'de sikismis (>5 dk) → 3 kez infra-reset, sonra fail
- Ungraded >30 dk sikismis → safety-net ile completed'a yukselt
- Tum bagimliliklari fail olmus bekleyen → cascade fail (bagimliliklardan biri DLQ'da ise ertelenir)
- `waiting_subtasks`'ta tum cocuklari terminal olmus → complete/failed isaretle
- `next_retry_at`'i 1 saatten fazla gecmis bekleyen → kapiyi sifirla
- `waiting_human` tirmandirma kademeleri (4s dipnot / 24s kademe 1 / 48s kademe 2 / 72s iptal) — bildirim `salako.notify_user` gorevleri ile
- Workflow-seviye timeout → misyonu pause et

### Yeniden deneme policy'si (`retry.decide_retry`)
- Backoff tablosu: `[0, 10, 30, 120, 600]` saniye — ilk deneme anlik, sonrakiler artar
- Exhausted (denemeler >= max) → DLQ, **eger** kategori `quality` VE progress >= 0.5 VE 2'den az bonus verilmisse bir bonus deneme (`bonus_used=True`)
- DLQ yazimi `dead_letter_tasks` tablosuna + `salako.notify_user` gorevi olustur (inline Telegram yok)

## Kurulum

```bash
pip install -e ./packages/general_beckman
```

Bagimliliklar: `sqlite3`, `nerd_herd`, `salako` (pyproject.toml'da).

## API

### Public fonksiyonlar

```python
async def next_task() -> Task | None
async def on_task_finished(task_id: int, result: dict) -> None
async def enqueue(spec: dict) -> int
```

### Action turleri (`result_router`)

```python
Complete, CompleteWithReusedAnswer, SpawnSubtasks,
RequestClarification, RequestReview, Exhausted, Failed, MissionAdvance
```

### Paused patterns

```python
from general_beckman.paused_patterns import pause, unpause, all_paused, is_paused
```

## Mimari

```
general_beckman/
  ├── __init__.py          — public API
  ├── types.py             — Task + AgentResult
  ├── queue.py             — pick_ready_task
  ├── cron.py              — fire_due: scheduled_tasks processor
  ├── cron_seed.py         — 6 dahili cadence'i seed eder
  ├── sweep.py             — kuyruk temizligi
  ├── rewrite.py           — pure action-rewrite kurallari
  ├── apply.py             — action -> DB satirlari
  ├── retry.py             — decide_retry + DLQ karari
  ├── paused_patterns.py   — DLQ pause module state
  ├── result_router.py     — agent sonucu -> Action dataclass'lari
  └── task_context.py      — parse_context helper
```

Her modul tek bir sorumluluga sahip. `sweep.py` en buyugu (~370 satir, 7 ayri temizlik dali). Digerleri 260 satirin altinda.

### Misyon ilerleme (workflow_engine import etmez)

```
coder gorevi #500 tamamlandi (mission_id=M)
  -> on_task_finished -> rewrite MissionAdvance enjekte eder
  -> apply mechanical gorev dogurur: {executor: workflow_advance, mission_id: M, ...}

Sonraki cycle:
  orchestrator o gorevi secer -> salako.run(it)
  -> salako.workflow_advance, workflow_engine.advance() cagirir
  -> advance() post_execute_workflow_step calistirir (artifact capture, phase check)
  -> N+1 faz icin subtask'lar doner -> salako envelope {status: needs_subtasks}
  -> on_task_finished -> apply bunlari SpawnSubtasks olarak DB satirlarina cevirir
```

## General Beckman'in Yapmadigi Seyler

- **LLM model secimi** — o `fatih_hoca`'nin isi.
- **Process yonetimi / llama-server** — o `dallama`.
- **Cloud rate-limit takibi** — o `kuleden_donen_var`.
- **Kaynak saglik kontrolu** — o `nerd_herd.health_summary`.
- **Workflow recipe / faz hesabi** — o `workflow_engine`, `salako.workflow_advance` ile cagrilir.
- **Telemetry push** — o dispatch gozlem noktasinda, `src/core/metrics_push.py`'da.
- **Telegram I/O** — tum disariya mesajlar `salako.clarify` / `salako.notify_user` mechanical gorevleri uzerinden.

## Lisans

MIT
