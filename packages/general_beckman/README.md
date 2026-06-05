# General Beckman — Task master (the queue's mission commander)

> Named for General Beckman in *Chuck* — the commander who decides which
> mission goes out and what happens when an agent reports back. This package
> is that authority for the task queue: it owns every task row's lifecycle.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** An autonomous system generates far more work than it can
run at once — user requests, mission steps, retries, clarifications, post-hook
graders, cron cadences — all against scarce GPU and metered cloud quota.
General Beckman is the single place that answers two questions so nothing else
has to: *"which one task should I release right now?"* and *"this one just
finished — what does its result imply?"*. Callers enqueue work and report
outcomes; Beckman owns admission, retry/backoff, dead-lettering, queue hygiene,
mission cost ceilings, and the spawning of every follow-up row.

**What it really does.** `next_task()` is an *admission loop*, not a dumb pop: it
takes the top-K ready tasks by urgency within a lane, asks the model selector
for a Pick per candidate, and admits the first whose chosen model clears its
pool-pressure threshold — claiming it, reserving its in-flight slot, and
stamping the Pick so the worker reuses it. `on_task_finished()` runs a pure
pipeline — `route_result → rewrite_actions → apply_actions` — that turns an
agent's envelope into typed Action dataclasses and applies them as DB rows:
complete, spawn subtasks, retry with backoff, dead-letter, advance a mission,
fire a durable continuation. `enqueue()` is the one external write path.

**It does NOT** pick models, run LLM/HTTP calls, manage GPU or processes, or
compute workflow phases — it asks a *model selector* for a Pick, reads a
*capacity snapshot* for pressure, and routes mission progression through a
*mechanical executor* (it never imports the workflow engine). It does NOT talk
to Telegram directly: every outbound message is a mechanical `notify_user` /
`clarify` task. There is no distribution, multi-host scheduling, or
cross-worker fairness — this is one SQLite DB, one async process, one pump.

## Public API

The pump calls three async functions; everything else is internal. Two more
exports (`resolve_inline`, `on_model_swap`) are wake-up hooks the host fires
from elsewhere.

```python
import general_beckman as beckman

# 1. Admission — called every orchestrator pump cycle, per lane.
#    Fires due crons, scans top-K ready tasks by urgency, admits the first
#    whose selected model clears its pool-pressure gate. None = nothing
#    admissible this tick (queue empty, lane saturated, or no model has room).
task = await beckman.next_task(lane=None)          # -> Task | None  (Task = dict)

# 2. Result handling — called after every dispatched task returns.
#    route_result -> rewrite_actions -> apply_actions. Every follow-up row
#    (subtask, retry, DLQ, MissionAdvance, continuation) originates here.
await beckman.on_task_finished(task_id, result)    # -> None

# 3. The single external write path — user/bot tasks, subtasks, cron rows.
new_id = await beckman.enqueue(spec)               # -> int (task id)
```

`enqueue` carries the full lifecycle contract via keyword-only options:

```python
new_id = await beckman.enqueue(
    spec,                       # dict of add_task kwargs; spec["kind"] defaults "main_work"
    parent_id=None,             # stored as tasks.parent_task_id
    lane=None,                  # "oneshot" (default) | "ongoing"; else derived from agent_type
    on_complete=None,           # name of a durable continuation handler (survives restart)
    on_error=None,              # continuation handler fired on failure
    cont_state=None,            # dict passed back to the handler when it fires
    next_task_spec=None,        # fire-and-forget follow-up spec (context-based, NOT durable)
)
# await_inline=True instead RETURNS a TaskResult, blocking until the task
# reaches a terminal state (resolved via resolve_inline). Mutually exclusive
# with on_complete/on_error.
result = await beckman.enqueue(spec, await_inline=True)   # -> TaskResult(status, result, error)
```

Wake-up hooks the host fires (not part of the pump):

```python
beckman.resolve_inline(task_id, TaskResult(...))   # wake an await_inline waiter
await beckman.on_model_swap(old_model, new_model)  # accelerate retries deferred on model load
```

Top-level exports (`__all__`): functions `next_task`, `on_task_finished`,
`enqueue`, `on_model_swap`, `resolve_inline`, `notify_threshold`; types `Task`,
`AgentResult`, `TaskResult`; constants `INLINE_TIMEOUT`, `THRESHOLDS_PCT`.
`Task` and `AgentResult` are both `dict[str, Any]` aliases — a task is a raw DB
row, not a dataclass.

### Action types (`result_router`)

`route_result(task, agent_result) -> list[Action]` maps an agent envelope's
`status` to one of these frozen dataclasses; `apply_actions` has one DB branch
per type:

```python
Complete(task_id, result, iterations, metadata, raw)
CompleteWithReusedAnswer(task_id, result, raw)     # clarification reused from history
SpawnSubtasks(parent_task_id, subtasks, raw)
RequestClarification(task_id, question, chat_id, raw)
RequestReview(task_id, summary, raw)
Exhausted(task_id, error, raw)                     # -> decide_retry
Failed(task_id, error, raw)                        # -> decide_retry
MissionAdvance(task_id, mission_id, completed_task_id, raw)
RequestPostHook(source_task_id, kind, source_ctx)  # spawn grader / summary
PostHookVerdict(source_task_id, kind, passed, raw, action="gate"|"rewrite", new_result=None)
```

### Retry decision (`retry`)

```python
from general_beckman.retry import decide_retry, RetryDecision, DLQAction

decide_retry(failure: dict, progress: float | None = None, bonus_count: int = 0)
#   -> RetryDecision(action="immediate"|"delayed", delay_seconds, bonus_used)
#   -> DLQAction(action="dlq", category, reason)
```

## Architecture

```
orchestrator pump (per lane, per cycle)
  └─ next_task(lane) ── admission loop
       ├─ fire_due()                  cron: markers dispatch internally,
       │                              non-markers insert concrete task rows
       ├─ lane cap check              (mechanicals exempt — CPU-only, no GPU/cloud)
       ├─ capacity snapshot           pool pressure, overlaid with in-process truth
       ├─ pick_ready_top_k(urgency)   age-boosted, paused-pattern-filtered
       └─ per candidate:
            mechanical?  → claim & return (no model, no pressure gate)
            else → selector.select(urgency, …) → pick clears pressure gate?
                   → cost-ceiling ok? → claim → reserve in-flight slot → return

orchestrator runs the agent / mechanical, then:
  on_task_finished(task_id, result)
    ├─ accumulate mission spent_usd, fire 50/75/90% ceiling notifies
    ├─ post_execute_workflow_step  (may flip status before routing)
    ├─ route_result   → Action dataclasses
    ├─ rewrite_actions → pure policy (mission complete → +MissionAdvance,
    │                    silent clarify → Failed, clarification reuse, …)
    ├─ apply_actions  → DB side-effects, one branch per Action
    └─ fire durable continuations on TRUE terminal (re-read DB status)
```

Mission progression carries no workflow-engine import: a completed mission task
gets a `MissionAdvance` rewritten in, which spawns a mechanical
`workflow_advance` task; the mechanical executor delegates to the workflow
engine and returns the next phase's subtasks as a normal result, which
`on_task_finished` spawns as `SpawnSubtasks`.

## Key Modules

| module | role |
|---|---|
| `__init__.py` | public API: `next_task` admission loop, `on_task_finished` pipeline, `enqueue`, cost-ceiling + threshold notifies |
| `queue.py` | `pick_ready_top_k` — age-boost ordering + paused-pattern filter |
| `admission.py` | per-task urgency computation (stamped on the row for worker reuse) |
| `lanes.py` | `oneshot` / `ongoing` lane policy, per-lane concurrency caps, mechanical exemption |
| `result_router.py` | `route_result` — agent envelope → typed Action dataclasses |
| `rewrite.py` | pure action-rewriting policy (no DB) |
| `apply.py` | `apply_actions` — Action → DB rows; the largest module (post-hook chain machinery) |
| `retry.py` | `decide_retry` policy + transient/quality backoff ladder + DLQ |
| `sweep.py` | queue hygiene: stuck / ungraded / dep-cascade / rollup / overdue gates / `waiting_human` escalation / workflow timeout |
| `cron.py` + `cron_seed.py` | unified `scheduled_tasks` processor; lazily seeds internal cadences |
| `continuations.py` | durable continuation substrate (`continuations` table, survives restart) |
| `posthooks.py` + `posthook_handlers/` | post-hook spec registry + handler implementations |
| `paused_patterns.py` | DLQ pause-pattern module state (`/dlq pause category:…`) |

## Gotchas

- **`next_task` admits at most one task per call.** It returns the first
  candidate that clears the gate, not a batch. The pump calls it repeatedly.
- **Admission is the model-selection point.** The Pick chosen here is stamped as
  `preselected_pick` and reused verbatim by the worker on the happy path —
  including hoisted OVERHEAD hints from `context.llm_call`. Selection bugs are
  not Beckman bugs; they live in the model selector.
- **A stale-state cache can short-circuit the scan.** When input state is
  unchanged since the last tick *and* that tick admitted nothing, `next_task`
  returns `None` without re-scanning. A 30s wall-clock bucket and per-model
  availability flags are mixed into the fingerprint specifically to break a
  deadlock where every candidate sits at full pressure forever. Mechanical
  tasks bypass the cache.
- **Mechanical tasks are unbounded.** No model, no Pick, no pressure gate, no
  lane cap — they reach the DB even under full LLM saturation, so git commits /
  blackboard writes / `notify_user` never stall behind slow LLM work.
- **Continuations fire on the *DB* terminal status, not the in-memory result.**
  `apply_actions` may re-pend a transient failure; firing on the raw result
  would latch the failed first attempt. `on_task_finished` re-reads the row and
  only fires on a true `completed` / `failed`.
- **Transient vs quality retries differ.** Availability/quota/timeout failures
  ride the full backoff ladder (tail = 24h, past a daily-quota reset); quality
  failures retry immediately (waiting can't change a deterministic output). The
  admission cap-guard, `decide_retry`, and sweep all share
  `effective_max_attempts` so they agree on when a task is truly exhausted.
- **`await_inline` and `on_complete`/`on_error` are mutually exclusive** — a
  blocking wait can't also fire a continuation; `enqueue` raises `ValueError`.

## Dependencies

Declared (pyproject) hard peers — Beckman imports their types/functions
directly:

- **The capacity tracker** (`nerd_herd`): `next_task` awaits its snapshot for
  pool pressure, and `queue_profile_push.py` imports its `QueueProfile` type at
  module load and pushes a live queue profile after each result. This is an
  import-time coupling, not best-effort.
- **The mechanical executor** (`mr_roboto`): all non-LLM follow-ups — clarify,
  notify_user, workflow_advance, schema-gate, semgrep, signature extraction —
  are created as mechanical rows it dispatches. `apply.py` imports several of
  its verbs directly.

Best-effort / lazy (NOT declared deps — guarded by `try/except`, system runs
without them): the **model selector** (queried per candidate for a Pick), the
**cloud capacity adapter** (overlays in-process rate-limit truth onto the
snapshot), and a **quality checker** (rejects degenerate clarification text).

Host coupling: Beckman reads and writes the host's SQLite DB throughout
(`src.infra.db`) and runs the workflow-step post-hook (`src.workflows`) inside
`on_task_finished`. It is not a standalone library — it is the host's task
authority.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\general_beckman\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Adını *Chuck* dizisindeki General Beckman'dan alır — hangi görevin sahaya
> çıkacağına ve ajan rapor verdiğinde ne olacağına karar veren komutan. Bu
> paket, görev kuyruğu için o otoritedir: her görev satırının yaşam döngüsünü
> sahiplenir.

### Amaç

**Ne işe yarar.** Otonom bir sistem, aynı anda çalıştırabileceğinden çok daha
fazla iş üretir — kullanıcı istekleri, misyon adımları, yeniden denemeler,
soru-sormalar, post-hook değerlendiriciler, cron cadence'ları — hepsi kıt GPU
ve sayaçlı bulut kotasına karşı. General Beckman, başka hiçbir bileşenin
uğraşmaması için iki soruyu tek noktada yanıtlar: *"şu anda hangi tek görevi
serbest bırakmalıyım?"* ve *"bu görev az önce bitti — sonucu neyi gerektiriyor?"*.
Çağıranlar iş ekler ve sonuç bildirir; admission, yeniden deneme/backoff,
dead-letter, kuyruk temizliği, misyon maliyet tavanları ve her takip satırının
oluşturulması Beckman'a aittir.

**Gerçekte ne yapar.** `next_task()` aptal bir pop değil, bir *admission
döngüsüdür*: bir lane içinde aciliyete göre ilk-K hazır görevi alır, her aday
için model seçiciden bir Pick ister ve seçilen modeli kendi pool-pressure
eşiğini geçen ilk görevi kabul eder — onu claim eder, in-flight slotunu rezerve
eder ve worker'ın aynısını kullanması için Pick'i damgalar. `on_task_finished()`
saf bir hat işletir — `route_result → rewrite_actions → apply_actions` — bir
ajanın envelope'unu tipli Action dataclass'larına çevirip DB satırları olarak
uygular: tamamla, alt görev doğur, backoff'la yeniden dene, dead-letter,
misyonu ilerlet, kalıcı bir continuation ateşle. `enqueue()` tek dış yazma
yoludur.

**Yapmadıkları**: model seçmez, LLM/HTTP çağrısı yapmaz, GPU veya süreç
yönetmez, workflow fazı hesaplamaz — *model seçicisinden* bir Pick ister, basınç
için bir *kapasite anlık görüntüsü* okur ve misyon ilerlemesini bir *mekanik
yürütücü* üzerinden geçirir (workflow engine'i asla import etmez). Telegram'la
doğrudan konuşmaz: tüm giden mesajlar mekanik `notify_user` / `clarify`
görevidir. Dağıtım, çok-makineli zamanlama veya worker'lar arası adalet yoktur —
bu tek bir SQLite DB, tek bir asenkron süreç, tek bir pompadır.

### Genel API

Pompa üç asenkron fonksiyon çağırır; gerisi tamamen iç işleyiştir. İki ek export
(`resolve_inline`, `on_model_swap`) host'un başka yerden ateşlediği uyandırma
kancalarıdır.

```python
import general_beckman as beckman

# 1. Admission — her orchestrator pompa döngüsünde, lane başına çağrılır.
#    Vadesi gelmiş cronları ateşler, aciliyete göre ilk-K hazır görevi tarar,
#    seçilen modeli pool-pressure kapısını geçen ilk görevi kabul eder. None =
#    bu tick'te kabul edilebilir görev yok (kuyruk boş, lane dolu ya da hiçbir
#    modelde yer yok).
task = await beckman.next_task(lane=None)          # -> Task | None  (Task = dict)

# 2. Sonuç işleme — sonuçlanan her görevden sonra çağrılır.
#    route_result -> rewrite_actions -> apply_actions. Her takip satırı
#    (alt görev, yeniden deneme, DLQ, MissionAdvance, continuation) buradan doğar.
await beckman.on_task_finished(task_id, result)    # -> None

# 3. Tek dış yazma yolu — kullanıcı/bot görevleri, alt görevler, cron satırları.
new_id = await beckman.enqueue(spec)               # -> int (görev id'si)
```

`enqueue`, yaşam döngüsü sözleşmesinin tamamını anahtar-kelime seçenekleriyle
taşır:

```python
new_id = await beckman.enqueue(
    spec,                       # add_task kwarg'ları; spec["kind"] varsayılan "main_work"
    parent_id=None,             # tasks.parent_task_id olarak saklanır
    lane=None,                  # "oneshot" (varsayılan) | "ongoing"; yoksa agent_type'tan türetilir
    on_complete=None,           # kalıcı bir continuation handler adı (restart'tan sağ çıkar)
    on_error=None,              # hata durumunda ateşlenen continuation handler
    cont_state=None,            # handler ateşlendiğinde geri verilen dict
    next_task_spec=None,        # ateşle-unut takip spec'i (context tabanlı, kalıcı DEĞİL)
)
# await_inline=True ise bunun yerine bir TaskResult DÖNDÜRÜR, görev terminal
# duruma ulaşana dek bloklar (resolve_inline ile çözülür). on_complete/on_error
# ile birlikte kullanılamaz.
result = await beckman.enqueue(spec, await_inline=True)   # -> TaskResult(status, result, error)
```

Host'un ateşlediği uyandırma kancaları (pompanın parçası değil):

```python
beckman.resolve_inline(task_id, TaskResult(...))   # bir await_inline bekleyicisini uyandır
await beckman.on_model_swap(old_model, new_model)  # model yüklemesini bekleyen retry'leri hızlandır
```

Üst düzey export'lar (`__all__`): `next_task`, `on_task_finished`, `enqueue`,
`on_model_swap`, `resolve_inline`, `notify_threshold` fonksiyonları; `Task`,
`AgentResult`, `TaskResult` tipleri; `INLINE_TIMEOUT`, `THRESHOLDS_PCT`
sabitleri. `Task` ve `AgentResult` ikisi de `dict[str, Any]` takma adıdır — bir
görev, dataclass değil ham bir DB satırıdır.

### Action türleri (`result_router`)

`route_result(task, agent_result) -> list[Action]`, bir ajan envelope'unun
`status`'unu aşağıdaki frozen dataclass'lardan birine eşler; `apply_actions`
her tür için bir DB dalına sahiptir:

```python
Complete(task_id, result, iterations, metadata, raw)
CompleteWithReusedAnswer(task_id, result, raw)     # soru-sorma geçmişten yeniden kullanıldı
SpawnSubtasks(parent_task_id, subtasks, raw)
RequestClarification(task_id, question, chat_id, raw)
RequestReview(task_id, summary, raw)
Exhausted(task_id, error, raw)                     # -> decide_retry
Failed(task_id, error, raw)                        # -> decide_retry
MissionAdvance(task_id, mission_id, completed_task_id, raw)
RequestPostHook(source_task_id, kind, source_ctx)  # grader / summary doğur
PostHookVerdict(source_task_id, kind, passed, raw, action="gate"|"rewrite", new_result=None)
```

### Yeniden deneme kararı (`retry`)

```python
from general_beckman.retry import decide_retry, RetryDecision, DLQAction

decide_retry(failure: dict, progress: float | None = None, bonus_count: int = 0)
#   -> RetryDecision(action="immediate"|"delayed", delay_seconds, bonus_used)
#   -> DLQAction(action="dlq", category, reason)
```

### Mimari

```
orchestrator pompası (lane başına, döngü başına)
  └─ next_task(lane) ── admission döngüsü
       ├─ fire_due()                  cron: marker'lar dahili dispatch,
       │                              marker olmayanlar somut görev satırı ekler
       ├─ lane cap kontrolü           (mekanikler muaf — sadece CPU, GPU/bulut yok)
       ├─ kapasite anlık görüntüsü     pool basıncı, süreç-içi gerçekle örtülür
       ├─ pick_ready_top_k(urgency)   yaş-artırımlı, duraklatılmış-desen filtreli
       └─ aday başına:
            mekanik mi?  → claim & döndür (model yok, basınç kapısı yok)
            değilse → seçici.select(urgency, …) → pick basınç kapısını geçti mi?
                      → maliyet tavanı uygun mu? → claim → in-flight slot rezerve → döndür

orchestrator ajanı / mekaniği çalıştırır, sonra:
  on_task_finished(task_id, result)
    ├─ misyon spent_usd biriktir, %50/75/90 tavan bildirimi ateşle
    ├─ post_execute_workflow_step  (routing öncesi status'u çevirebilir)
    ├─ route_result   → Action dataclass'ları
    ├─ rewrite_actions → saf politika (misyon tamamlandı → +MissionAdvance,
    │                    sessiz clarify → Failed, soru-sorma yeniden kullanımı, …)
    ├─ apply_actions  → DB yan etkileri, Action başına bir dal
    └─ GERÇEK terminal'de kalıcı continuation ateşle (DB status'unu yeniden oku)
```

Misyon ilerlemesi hiçbir workflow-engine import'u taşımaz: tamamlanan bir misyon
görevine bir `MissionAdvance` rewrite ile eklenir, bu da mekanik bir
`workflow_advance` görevi doğurur; mekanik yürütücü işi workflow engine'e
devreder ve bir sonraki fazın alt görevlerini normal bir sonuç olarak döndürür,
`on_task_finished` da bunları `SpawnSubtasks` olarak doğurur.

### Ana Modüller

| modül | rolü |
|---|---|
| `__init__.py` | genel API: `next_task` admission döngüsü, `on_task_finished` hattı, `enqueue`, maliyet tavanı + bildirimler |
| `queue.py` | `pick_ready_top_k` — yaş-artırımlı sıralama + duraklatılmış-desen filtresi |
| `admission.py` | görev başına aciliyet hesabı (worker yeniden kullanımı için satıra damgalanır) |
| `lanes.py` | `oneshot` / `ongoing` lane politikası, lane başına eşzamanlılık tavanı, mekanik muafiyeti |
| `result_router.py` | `route_result` — ajan envelope'u → tipli Action dataclass'ları |
| `rewrite.py` | saf action-rewrite politikası (DB yok) |
| `apply.py` | `apply_actions` — Action → DB satırları; en büyük modül (post-hook zincir makinesi) |
| `retry.py` | `decide_retry` politikası + transient/quality backoff merdiveni + DLQ |
| `sweep.py` | kuyruk temizliği: sıkışmış / değerlendirilmemiş / dep-cascade / rollup / vadesi geçmiş kapılar / `waiting_human` tırmandırma / workflow timeout |
| `cron.py` + `cron_seed.py` | birleşik `scheduled_tasks` işleyicisi; dahili cadence'ları tembel seed eder |
| `continuations.py` | kalıcı continuation altyapısı (`continuations` tablosu, restart'tan sağ çıkar) |
| `posthooks.py` + `posthook_handlers/` | post-hook spec kaydı + handler uygulamaları |
| `paused_patterns.py` | DLQ duraklatma-deseni modül durumu (`/dlq pause category:…`) |

### Tuzaklar

- **`next_task` çağrı başına en fazla bir görev kabul eder.** Bir parti değil,
  kapıyı geçen ilk adayı döndürür. Pompa onu tekrar tekrar çağırır.
- **Admission, model seçim noktasıdır.** Burada seçilen Pick `preselected_pick`
  olarak damgalanır ve mutlu yolda worker tarafından aynen yeniden kullanılır —
  `context.llm_call`'tan kaldırılan OVERHEAD ipuçları dahil. Seçim hataları
  Beckman hatası değildir; model seçicide yaşar.
- **Bayat durum cache'i taramayı kısa devre yapabilir.** Giriş durumu son
  tick'ten beri değişmediyse *ve* o tick hiçbir şey kabul etmediyse, `next_task`
  yeniden taramadan `None` döner. Fingerprint'e 30s'lik bir duvar-saati kovası
  ve model başına erişilebilirlik bayrakları, her adayın sonsuza dek tam basınçta
  takılı kaldığı bir deadlock'u kırmak için özellikle karıştırılmıştır. Mekanik
  görevler cache'i atlar.
- **Mekanik görevler sınırsızdır.** Model yok, Pick yok, basınç kapısı yok, lane
  tavanı yok — tam LLM doygunluğunda bile DB'ye ulaşırlar, böylece git
  commit'ler / blackboard yazımları / `notify_user` yavaş LLM işinin arkasında
  takılmaz.
- **Continuation'lar *DB* terminal status'unda ateşlenir, bellekteki sonuçta
  değil.** `apply_actions` bir transient hatayı yeniden pending'e alabilir; ham
  sonuçta ateşlemek başarısız ilk denemeyi mandallar. `on_task_finished` satırı
  yeniden okur ve yalnızca gerçek bir `completed` / `failed`'de ateşler.
- **Transient ve quality retry'leri farklıdır.** Erişilebilirlik/kota/timeout
  hataları tüm backoff merdivenini biner (kuyruk = 24s, günlük kota sıfırlamasını
  geçer); quality hataları hemen yeniden denenir (beklemek deterministik bir
  çıktıyı değiştiremez). Admission cap-guard, `decide_retry` ve sweep, bir görevin
  gerçekten tükendiğine karar vermede aynı `effective_max_attempts`'i paylaşır.
- **`await_inline` ile `on_complete`/`on_error` birlikte kullanılamaz** —
  bloklayan bir bekleme aynı anda bir continuation ateşleyemez; `enqueue`
  `ValueError` fırlatır.

### Bağımlılıklar

Bildirilen (pyproject) sıkı dengler — Beckman onların tiplerini/fonksiyonlarını
doğrudan import eder:

- **Kapasite izleyici** (`nerd_herd`): `next_task`, pool basıncı için onun anlık
  görüntüsünü bekler ve `queue_profile_push.py`, `QueueProfile` tipini modül
  yüklenirken import edip her sonuçtan sonra canlı bir kuyruk profili iter. Bu,
  best-effort değil, import-zamanı bir bağlantıdır.
- **Mekanik yürütücü** (`mr_roboto`): tüm LLM-olmayan takipler — clarify,
  notify_user, workflow_advance, schema-gate, semgrep, imza çıkarımı — onun
  dispatch ettiği mekanik satırlar olarak oluşturulur. `apply.py` onun verb'lerini
  doğrudan import eder.

Best-effort / tembel (bildirilen dep DEĞİL — `try/except` ile korunur, sistem
bunlarsız da çalışır): **model seçici** (aday başına Pick için sorgulanır),
**bulut kapasite adaptörü** (süreç-içi rate-limit gerçeğini anlık görüntünün
üzerine bindirir) ve bir **kalite denetleyici** (dejenere soru-sorma metnini
reddeder).

Host bağlantısı: Beckman baştan sona host'un SQLite DB'sini okur ve yazar
(`src.infra.db`) ve `on_task_finished` içinde workflow-adım post-hook'unu
(`src.workflows`) çalıştırır. Bağımsız bir kütüphane değildir — host'un görev
otoritesidir.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\general_beckman\ -q
```
