# Mr. Roboto — Mechanical (non-LLM) task dispatcher

> Named for the steadfast worker who just does the grunt work without asking.
> Mr. Roboto is the single entry point for every task that should run **without
> a model** — deterministic work that must not burn LLM tokens or swap budget.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** A mission produces a constant stream of work that has no
business going near an LLM: snapshot the workspace, commit the diff, shape-check
an artifact, run a test, render a legal doc, post a status update, fire a
Telegram nudge. Routing all of that through one mechanical dispatcher means the
caller hands over a plain task dict and gets back a uniform result — no prompt,
no model load, no swap, no retry loop. **Adding a new mechanical capability is
one new verb branch, not a new agent.**

**What it really does.** `run(task)` reads `task["payload"]["action"]` — a verb
string — and routes it to the matching deterministic executor, of which there
are well over a hundred spanning git/workspace, artifact shape-checkers,
regen/preview, scanners and reviewers, mobile build adapters, growth/CRM/comms,
and cron jobs. Every executor returns the same `Action` envelope. Before
dispatch, two cross-cutting layers run: a **safety gate** for shell-executing
verbs, and a **reversibility + confirmation gate** that can park an irreversible
verb on a founder yes/no without holding a worker slot. Every dispatch also
lands an audit row.

**It does NOT** make LLM calls, pick models, or run a ReAct loop — anything that
needs a model is enqueued back as a regular agent task (a few verbs *orchestrate*
an LLM step by enqueuing it; they never call the model themselves). It does NOT
own the queue, schedule tasks, or decide *when* a verb runs — the caller (the
host orchestrator / task master) does that and invokes `run()` per task.

## Public API

The whole package is one function plus a result type. Everything else is an
internal executor reached only through the verb string.

```python
import mr_roboto

# Route a mechanical task by its payload.action verb:
action = await mr_roboto.run(task)        # -> mr_roboto.Action

# action.status  -> "completed" | "failed" | "skipped" | "blocked"
#                   | "waiting_human" | "needs_clarification" | "rejected" | "partial"
# action.result  -> dict (executor-specific payload)
# action.error   -> str | None  (populated on failure paths)
# action.reversibility -> "full" | "partial" | "irreversible"  (from the verb registry)

# Reversibility lookup (the live source of truth for what verbs exist):
from mr_roboto import get_reversibility, VERB_REVERSIBILITY
tag = get_reversibility("git_commit")     # -> "full"

# A couple of executors are also exported as direct helpers for callers that
# want them without building a task dict:
snap = await mr_roboto.snapshot_workspace(mission_id, task_id, workspace_path, repo_path=None)
info = await mr_roboto.auto_commit(task, result)   # -> commit-info dict
```

A task dict for `run()`:

```python
task = {
    "id": 42,
    "mission_id": 7,
    "payload": {                       # also accepted at task["context"]["payload"]
        "action": "workspace_snapshot",
        "workspace_path": "/path/to/ws",
    },
}
```

An **unknown** verb is not an exception — it returns
`Action(status="failed", error="unknown mechanical action: ...")` and the caller
marks the task failed.

## Architecture

`run()` is a thin pre-flight wrapper around a long verb-dispatch switch:

```
run(task)
  ├─ safety gate     — only for shell verbs (run_cmd / run_pytest):
  │                    asks safety_guard.pre_action → may return
  │                    blocked / waiting_human before any executor runs
  ├─ reversibility   — get_reversibility(verb, override) tags the Action;
  │                    fastlane derives its tag from the build lane
  ├─ confirmation gate — if the verb is partial/irreversible AND confirmation
  │                    is armed (payload flag or KUTAI_CONFIRM_POLICY), park
  │                    the task on a founder yes/no via the clarify path
  │                    (no busy-poll, no worker-slot hold)
  ├─ dispatch        — _run_dispatch() routes payload.action → executor
  │                    → returns Action(status, result, error)
  └─ audit           — record one registry_events row per dispatch; external
                       publish verbs also land an external_comms_log row
```

## Verb families

The dispatcher routes **150+ verbs**. They are not listed individually here —
that table would rot. The authoritative, code-checked enumeration is
`VERB_REVERSIBILITY` in `reversibility.py` (cross-checked against the dispatcher
by `tests/`). The families:

| family | examples | typical reversibility |
|---|---|---|
| Workspace & git | `workspace_snapshot`, `git_commit` | full |
| Artifact shape-checkers | `verify_artifacts`, `verify_charter_shape`, `verify_adr_shape`, `verify_schema_version`, `check_grounding` | full |
| Regen & preview | `regen_artifact`, `regen_bundle`, `annotate_html_oids`, `emit_preview_url`, `publish_preview_pages` | full / irreversible |
| Scanners & reviewers | `run_pytest`, `run_semgrep`, `run_bandit`, `run_axe`, `security_review`, `performance_review`, `visual_review` | full |
| Shell & migration | `run_cmd`, `apply_migration`, `yalayut_recipe` | partial |
| Mobile build/dist | `expo_cli`, `eas_build`, `eas_submit`, `gen_mobile_ci`, `fastlane`, `maestro` | full / irreversible |
| Growth & analytics | `record_hypothesis`, `record_verdict`, `assign_variant`, `score_backlog`, `analytics_digest` | full / partial |
| Comms & CRM | `notify_user`, `clarify`, `crm/add_contact`, `email/send_via_provider`, `changelog/publish`, `incident/publish_status` | irreversible-leaning |
| Datastore & state | `kdv_persist`, `cloud_refresh`, `workflow_advance`, `mission_event_drain`, `vector_maint_*` | full |
| Cron jobs | `backup_verify`, `cve_scan`, `secret_scan`, `cost_pull`, `reviews_poll_daily` | full |

The reversibility tag is load-bearing: it (with the confirmation policy) decides
whether a verb can run unattended or must wait for the founder.

## Gotchas

- **`Action.status` has more than three values.** Besides
  `completed` / `failed` / `skipped`, the gates emit `blocked`,
  `waiting_human`, `needs_clarification`, `rejected`, and `partial`. A caller
  that only checks `== "completed"` will mishandle a parked task. In
  particular `needs_clarification` means the task is **parked, not finished** —
  the caller must leave the row alone and re-dispatch when the founder replies.
- **The confirmation gate is a park/resume loop, not a blocking wait.** First
  entry sends a Telegram question and returns `needs_clarification`; the founder's
  typed reply re-dispatches the same task with `context["user_clarification"]`
  set, and the gate runs again to approve/reject. An ambiguous reply
  **fails closed** (rejected) — an irreversible verb never proceeds on an answer
  it can't parse. A gate that can't reach the founder returns `failed`, never a
  silent skip.
- **Payload location drifts.** `run()` reads the payload from `task["payload"]`
  *or* `task["context"]["payload"]` — raw expander shape vs. the orchestrator's
  copied shape. Put it in either; don't assume only one.
- **Audit failures are swallowed by design.** The `registry_events` /
  `external_comms_log` writes never raise into the dispatch path — an action
  must not fail because its audit row didn't write.
- **Submodule import discipline.** Several executors (`critic_gate`,
  `init_mission_github_repo`, `mark_green`, `record_demo`, …) are imported as
  *submodules*, not symbols, so `monkeypatch.setattr("mr_roboto.<mod>._x", ...)`
  keeps working. Don't "tidy" those into `from … import name` — it breaks the
  test mocks.

## Dependencies

`mr_roboto` declares no third-party dependencies; it is built to run inside the
host repo (repo root on `sys.path`) and leans on it heavily:

- **The host repo** — executors import `src.infra.db`, `src.tools.*`,
  `src.app.telegram_bot`, `src.infra.mission_lessons`, and more. This is
  load-bearing and intentional: Mr. Roboto is the repo's mechanical arm, not a
  standalone library. If it were ever used standalone, those would have to be
  injected.
- **`safety_guard`** — the one genuine sibling-package dependency with a *hard*
  runtime coupling. For the shell verbs (`run_cmd`, `run_pytest`) `run()`
  imports `safety_guard.pre_action` unconditionally and honors its
  `Allow` / `WaitForFounder` / `Block` decision before the executor runs. There
  is no try/except fallback on that path — a shell verb cannot dispatch without
  it.
- **The host task master** — a few executors enqueue follow-up tasks back onto
  the queue (e.g. surfacing a spec-patch proposal, or orchestrating an LLM
  step). That call is best-effort and try-wrapped: a failed enqueue is logged,
  never fatal.
- **Env**: `KUTAI_CONFIRM_POLICY` (`off` / `irreversible_only` /
  `partial_or_worse`) auto-arms the confirmation gate; `WORKSPACE_ROOT` resolves
  the safety-gate workspace.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\mr_roboto\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Adını, hiç soru sormadan kaba işi halleden o sebatkâr işçiden alır.
> Mr. Roboto, **model kullanmadan** çalışması gereken her görevin tek giriş
> noktasıdır — LLM token'ı ya da swap bütçesi harcamaması gereken belirlenimci iş.

### Amaç

**Ne işe yarar.** Bir görev, bir LLM'e hiç uğramaması gereken sürekli bir iş
akışı üretir: çalışma alanını anlık görüntüle, diff'i commit'le, bir artefaktın
şeklini denetle, test çalıştır, hukuki belge oluştur, durum güncellemesi yayınla,
Telegram bildirimi gönder. Bunların hepsini tek bir mekanik dağıtıcıdan geçirmek,
çağıran tarafın düz bir görev sözlüğü verip karşılığında tek tip bir sonuç alması
demektir — prompt yok, model yüklemesi yok, swap yok, retry döngüsü yok. **Yeni
bir mekanik yetenek eklemek bir yeni ajan değil, bir yeni fiil dalıdır.**

**Gerçekte ne yapar.** `run(task)`, `task["payload"]["action"]` değerini — bir
fiil dizesini — okur ve eşleşen belirlenimci yürütücüye yönlendirir; bunlardan
yüzü aşkın vardır: git/çalışma-alanı, artefakt şekil denetçileri, regen/önizleme,
tarayıcılar ve incelemeciler, mobil derleme adaptörleri, growth/CRM/iletişim ve
cron işleri. Her yürütücü aynı `Action` zarfını döndürür. Dağıtımdan önce iki
kesişen katman çalışır: shell çalıştıran fiiller için bir **güvenlik kapısı** ve
geri-alınamaz bir fiili, bir işçi yuvasını tutmadan, founder'ın evet/hayır'ına
park edebilen bir **geri-alınabilirlik + onay kapısı**. Her dağıtım ayrıca bir
denetim satırı bırakır.

**Yapmadıkları**: LLM çağrısı yapmaz, model seçmez, ReAct döngüsü çalıştırmaz —
modele ihtiyaç duyan her şey normal bir ajan görevi olarak kuyruğa geri eklenir
(birkaç fiil bir LLM adımını *düzenler* ama onu kuyruğa ekleyerek; kendileri
modeli hiç çağırmaz). Kuyruğa sahip değildir, görev zamanlamaz, bir fiilin *ne
zaman* çalışacağına karar vermez — onu çağıran taraf (host düzenleyici / görev
ustası) yapar ve görev başına `run()` çağırır.

### Genel API

Tüm paket tek bir fonksiyon artı bir sonuç tipidir. Geri kalan her şey yalnızca
fiil dizesiyle ulaşılan içsel bir yürütücüdür.

```python
import mr_roboto

# Mekanik bir görevi payload.action fiiline göre yönlendir:
action = await mr_roboto.run(task)        # -> mr_roboto.Action

# action.status  -> "completed" | "failed" | "skipped" | "blocked"
#                   | "waiting_human" | "needs_clarification" | "rejected" | "partial"
# action.result  -> dict (yürütücüye özgü içerik)
# action.error   -> str | None  (yalnızca hata yollarında dolar)
# action.reversibility -> "full" | "partial" | "irreversible"  (fiil kaydından)

# Geri-alınabilirlik sorgusu (hangi fiillerin var olduğunun canlı doğruluk kaynağı):
from mr_roboto import get_reversibility, VERB_REVERSIBILITY
tag = get_reversibility("git_commit")     # -> "full"

# Birkaç yürütücü, görev sözlüğü kurmadan kullanmak isteyen çağıranlar için
# doğrudan yardımcı olarak da dışa verilir:
snap = await mr_roboto.snapshot_workspace(mission_id, task_id, workspace_path, repo_path=None)
info = await mr_roboto.auto_commit(task, result)   # -> commit-bilgisi sözlüğü
```

`run()` için bir görev sözlüğü:

```python
task = {
    "id": 42,
    "mission_id": 7,
    "payload": {                       # task["context"]["payload"] de kabul edilir
        "action": "workspace_snapshot",
        "workspace_path": "/path/to/ws",
    },
}
```

**Bilinmeyen** bir fiil bir istisna değildir —
`Action(status="failed", error="unknown mechanical action: ...")` döner ve çağıran
taraf görevi başarısız işaretler.

### Mimari

`run()`, uzun bir fiil-dağıtım anahtarının önüne konmuş ince bir ön-uçuş
sarmalayıcısıdır:

```
run(task)
  ├─ güvenlik kapısı   — yalnızca shell fiilleri (run_cmd / run_pytest) için:
  │                      safety_guard.pre_action'a sorar → yürütücüden önce
  │                      blocked / waiting_human dönebilir
  ├─ geri-alınabilirlik — get_reversibility(verb, override) Action'ı etiketler;
  │                      fastlane etiketini derleme lane'inden türetir
  ├─ onay kapısı        — fiil partial/irreversible İSE VE onay kuruluysa
  │                      (payload bayrağı ya da KUTAI_CONFIRM_POLICY), görevi
  │                      clarify yolu üzerinden founder evet/hayır'ına park et
  │                      (busy-poll yok, işçi-yuvası tutmak yok)
  ├─ dağıtım           — _run_dispatch() payload.action → yürütücü
  │                      → Action(status, result, error) döndürür
  └─ denetim           — dağıtım başına bir registry_events satırı yaz; dış
                         yayın fiilleri ayrıca bir external_comms_log satırı bırakır
```

### Fiil aileleri

Dağıtıcı **150'den fazla fiili** yönlendirir. Burada tek tek listelenmezler — o
tablo çürür. Yetkili, kodla denetlenmiş tam sayım `reversibility.py` içindeki
`VERB_REVERSIBILITY`'dir (dağıtıcıya karşı `tests/` ile çapraz denetlenir).
Aileler:

| aile | örnekler | tipik geri-alınabilirlik |
|---|---|---|
| Çalışma alanı & git | `workspace_snapshot`, `git_commit` | full |
| Artefakt şekil denetçileri | `verify_artifacts`, `verify_charter_shape`, `verify_adr_shape`, `verify_schema_version`, `check_grounding` | full |
| Regen & önizleme | `regen_artifact`, `regen_bundle`, `annotate_html_oids`, `emit_preview_url`, `publish_preview_pages` | full / irreversible |
| Tarayıcılar & incelemeciler | `run_pytest`, `run_semgrep`, `run_bandit`, `run_axe`, `security_review`, `performance_review`, `visual_review` | full |
| Shell & migrasyon | `run_cmd`, `apply_migration`, `yalayut_recipe` | partial |
| Mobil derleme/dağıtım | `expo_cli`, `eas_build`, `eas_submit`, `gen_mobile_ci`, `fastlane`, `maestro` | full / irreversible |
| Growth & analitik | `record_hypothesis`, `record_verdict`, `assign_variant`, `score_backlog`, `analytics_digest` | full / partial |
| İletişim & CRM | `notify_user`, `clarify`, `crm/add_contact`, `email/send_via_provider`, `changelog/publish`, `incident/publish_status` | çoğunlukla geri-alınamaz |
| Veri deposu & durum | `kdv_persist`, `cloud_refresh`, `workflow_advance`, `mission_event_drain`, `vector_maint_*` | full |
| Cron işleri | `backup_verify`, `cve_scan`, `secret_scan`, `cost_pull`, `reviews_poll_daily` | full |

Geri-alınabilirlik etiketi yük taşır: (onay politikasıyla birlikte) bir fiilin
gözetimsiz mi çalışabileceğine yoksa founder'ı mı beklemesi gerektiğine karar
verir.

### Tuzaklar

- **`Action.status`'un üçten fazla değeri vardır.**
  `completed` / `failed` / `skipped`'in yanında kapılar `blocked`,
  `waiting_human`, `needs_clarification`, `rejected` ve `partial` üretir. Yalnızca
  `== "completed"` kontrol eden bir çağıran, park edilmiş bir görevi yanlış
  ele alır. Özellikle `needs_clarification`, görevin **park edildiği, bitmediği**
  anlamına gelir — çağıran satırı rahat bırakmalı ve founder yanıtlayınca yeniden
  dağıtmalıdır.
- **Onay kapısı bloklayan bir bekleme değil, bir park/devam döngüsüdür.** İlk
  giriş bir Telegram sorusu gönderir ve `needs_clarification` döner; founder'ın
  yazdığı yanıt aynı görevi `context["user_clarification"]` ayarlanmış olarak
  yeniden dağıtır ve kapı onay/ret için tekrar çalışır. Belirsiz bir yanıt
  **kapalı tarafa düşer** (rejected) — geri-alınamaz bir fiil, çözemediği bir
  yanıt üzerine asla ilerlemez. Founder'a ulaşamayan bir kapı `failed` döner,
  asla sessiz bir atlama yapmaz.
- **Payload konumu kayar.** `run()`, payload'ı `task["payload"]` *ya da*
  `task["context"]["payload"]` üzerinden okur — ham expander şekli ile
  düzenleyicinin kopyaladığı şekil. İkisinden birine koy; yalnızca birinin
  olduğunu varsayma.
- **Denetim hataları tasarımca yutulur.** `registry_events` /
  `external_comms_log` yazımları dağıtım yoluna asla istisna fırlatmaz — bir
  eylem, denetim satırı yazılmadı diye başarısız olmamalıdır.
- **Submodül import disiplini.** Birçok yürütücü (`critic_gate`,
  `init_mission_github_repo`, `mark_green`, `record_demo`, …) sembol olarak değil,
  *submodül* olarak import edilir; böylece
  `monkeypatch.setattr("mr_roboto.<mod>._x", ...)` çalışmaya devam eder. Bunları
  `from … import name` haline "düzeltme" — test mock'larını bozar.

### Bağımlılıklar

`mr_roboto` hiçbir üçüncü-parti bağımlılık bildirmez; host repo'nun içinde (repo
kökü `sys.path`'te) çalışacak şekilde kurulmuştur ve ona yoğun yaslanır:

- **Host repo** — yürütücüler `src.infra.db`, `src.tools.*`,
  `src.app.telegram_bot`, `src.infra.mission_lessons` ve fazlasını import eder.
  Bu yük taşır ve bilinçlidir: Mr. Roboto repo'nun mekanik koludur, bağımsız bir
  kütüphane değil. Bağımsız kullanılacak olsaydı bunların enjekte edilmesi
  gerekirdi.
- **`safety_guard`** — *sert* çalışma-zamanı bağı olan tek gerçek
  sibling-paket bağımlılığı. Shell fiilleri (`run_cmd`, `run_pytest`) için
  `run()`, `safety_guard.pre_action`'ı koşulsuz import eder ve yürütücüden önce
  onun `Allow` / `WaitForFounder` / `Block` kararına uyar. O yolda try/except
  yedeği yoktur — bir shell fiili onsuz dağıtılamaz.
- **Host görev ustası** — birkaç yürütücü kuyruğa takip görevleri ekler (ör. bir
  spec-patch önerisini yüzeye çıkarmak ya da bir LLM adımını düzenlemek). O çağrı
  best-effort'tur ve try ile sarılıdır: başarısız bir enqueue loglanır, asla
  ölümcül değildir.
- **Env**: `KUTAI_CONFIRM_POLICY` (`off` / `irreversible_only` /
  `partial_or_worse`) onay kapısını otomatik kurar; `WORKSPACE_ROOT` güvenlik
  kapısının çalışma alanını çözer.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\mr_roboto\ -q
```
