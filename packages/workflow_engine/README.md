# Workflow Engine — Post-step mission advancer

> One job: take a workflow step that just finished and move the mission forward
> from it — capture its artifacts, run its finish-gate, and close the mission if
> nothing else is left.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** When a workflow step completes, several things have to
happen before the mission can move on: its output files have to be captured as
artifacts, a post-step hook has to validate the result (and possibly demand
clarification or fail the step), the next phase's work has to be queued, and — if
nothing is left — the mission has to be closed and the user told. Scattering that
across the caller is how missions silently stall or deliver a reviewer's JSON
verdict instead of the real answer. This package collapses all of it into **one
awaitable call with a single, predictable result object**, so the caller just
hands over a finished step and reads back a verdict.

**What it really does.** `advance(mission_id, completed_task_id, previous_result)`
loads the completed task, confirms it is a workflow step, then runs four stages in
order: (1) extract and store pipeline artifacts, (2) run the post-execute finish
hook — which may flip the step to `needs_clarification` or `failed`, (3) emit any
next-phase subtasks, and (4) if no subtasks remain and every other mission task is
terminal, mark the mission complete and deliver its final result. It returns an
`AdvanceResult` carrying the resolved status, any error/question text, the queued
subtasks, and the captured artifacts. Each stage is defensively wrapped so a
failure in artifact capture or mission completion never corrupts the verdict.

**It does NOT** run agents or make LLM calls, pick or score the next model, or own
phase-transition logic (it asks the engine for next subtasks; when that primitive
is absent it is a clean no-op, and phase routing stays in the caller). It also
does not dispatch itself — it is invoked by the host's mechanical executor for the
`workflow_advance` action, not on the agent path.

## Public API

The whole package is one function and its result type. The caller awaits it once
per completed workflow step and branches on `.status`:

```python
from workflow_engine import advance, AdvanceResult

result: AdvanceResult = await advance(
    mission_id=46,
    completed_task_id=3797,
    previous_result={"status": "completed", "result": "<markdown>"},
)

# AdvanceResult fields:
#   status: str            -> "completed" | "needs_clarification" | "failed"
#   error: str             -> error text, or the clarification question
#   next_subtasks: list[dict]  -> phase work to enqueue (empty today; see Gotchas)
#   artifacts: dict[str, Any]  -> {artifact_name: content} captured this step

if result.status == "needs_clarification":
    ...  # ask result.error, pause the mission
elif result.status == "failed":
    ...  # DLQ with result.error
elif result.next_subtasks:
    ...  # enqueue result.next_subtasks
else:
    ...  # mission advanced (and possibly completed) cleanly
```

`previous_result` is the finished step's payload. The post-hook reads its
`result` key for schema validation; if the carried payload is a condensed summary
without one, `advance` back-fills the full result string from the task's DB row so
the finish-gate sees real content (see Gotchas).

## Architecture

A thin façade. `advance` owns the ordering and the defensive boundaries; each
stage delegates to a primitive in the host's workflow-engine internals. Any stage
can short-circuit the result:

```
advance(mission_id, completed_task_id, previous_result)
  │
  ├─ load task; not found            → AdvanceResult(failed)
  ├─ not a workflow step             → AdvanceResult(completed, no-op)
  │
  ├─ 1. capture artifacts            (best-effort; populates .artifacts)
  ├─ 2. post-execute finish hook     → may flip → needs_clarification | failed
  ├─ 3. next-phase subtasks          (no-op until the recipe primitive exists)
  └─ 4. mission completion check     (only if no subtasks & status completed)
          └─ all tasks terminal → mark mission done
                                 → tear down mission sandbox (best-effort)
                                 → deliver final result + summary (best-effort)
```

## Dependencies

This package carries **no third-party dependencies** of its own. Its real
coupling is to the host application's internals, imported lazily inside `advance`
so the module loads even where those internals are absent:

- **Host workflow-engine internals** — the four stages delegate to the host's
  finish-hook (`is_workflow_step`, `post_execute_workflow_step`,
  `get_artifact_store`), its pipeline-artifact extractor, and — when present — its
  recipe-advance primitive. This is the load-bearing dependency: without these,
  `advance` degrades to a no-op rather than failing.
- **Host database** — reads the completed task and its mission/sibling tasks, and
  writes the mission's `completed` status.
- **Host workspace + shell** — resolves the per-mission workspace for artifact
  capture and tears down the mission sandbox on completion (best-effort).
- **Host Telegram interface** — delivers the final result and a completion summary
  when a mission closes (best-effort; skipped silently if the bot isn't up).

The **caller is the host's mechanical executor**: a step with the
`workflow_advance` action routes its payload (`mission_id`, `completed_task_id`,
`previous_result`) here and maps the returned `AdvanceResult` back onto the
mechanical result contract (`needs_subtasks` / `needs_clarification` / `failed` /
`completed`).

## Gotchas

- **Next-phase subtasks are a no-op today.** Stage 3 imports a recipe-advance
  primitive that isn't wired yet, so the `ImportError` branch always runs and
  `next_subtasks` comes back empty. Phase-transition logic still lives in the
  caller — don't assume `advance` queues the next phase.
- **Condensed payloads trip the finish-gate.** The mechanical advance task often
  carries `previous_result` as a summary (e.g. `{"summary": "..."}`) without the
  `result` key the post-hook validates against. `advance` back-fills the full
  result string from the task's DB row when the payload lacks one — drop that and
  producer steps DLQ with "schema requires X" despite clean sibling output.
- **Completion delivery skips bookkeeping agents.** When picking the message to
  send the user, mechanical / reviewer / summarizer tasks are skipped so a grading
  verdict or structural summary never leaks out as the mission's answer. The last
  completed *content* task with a non-trivial result wins.
- **Everything past validation is best-effort.** Artifact capture, mission
  completion, sandbox teardown, and Telegram delivery are each wrapped so their
  failure never breaks the returned verdict — a missing artifact store or a
  not-yet-initialised bot leaves the mission's status intact.
- **It defends the guard it expects callers to do.** `advance` re-checks
  `is_workflow_step` and returns a clean no-op for non-workflow tasks, even though
  callers should gate before calling.

## Tests

Tests live in the repo test suite, not inside the package:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_workflow_engine_advance.py -q
```

---
<a id="türkçe"></a>

## Türkçe

> Tek bir işi var: yeni biten bir iş akışı adımını alıp görevi oradan ileri
> taşımak — adımın çıktılarını yakalamak, bitiş kapısını çalıştırmak ve geriye iş
> kalmadıysa görevi kapatmak.

### Amaç

**Ne işe yarar.** Bir iş akışı adımı bittiğinde, görev ilerleyebilmeden önce
birkaç şey olmalı: adımın ürettiği dosyalar artefakt olarak yakalanmalı, bir
adım-sonrası kanca sonucu doğrulamalı (ve gerekirse açıklama istemeli ya da adımı
başarısız saymalı), sonraki fazın işi kuyruğa alınmalı ve geriye iş kalmadıysa
görev kapatılıp kullanıcıya haber verilmeli. Bunu çağıran tarafa dağıtmak,
görevlerin sessizce takılmasının ya da gerçek cevap yerine bir değerlendiricinin
JSON kararını sunmasının yoludur. Bu paket, tüm bunları **tek bir awaitable çağrı
ve tek, öngörülebilir bir sonuç nesnesi** içinde toplar; böylece çağıran taraf
sadece biten bir adımı verir ve geriye bir karar okur.

**Gerçekte ne yapar.** `advance(mission_id, completed_task_id, previous_result)`
biten görevi yükler, bunun bir iş akışı adımı olduğunu doğrular, sonra dört aşamayı
sırayla çalıştırır: (1) pipeline artefaktlarını çıkarıp saklar, (2) adım-sonrası
bitiş kancasını çalıştırır — bu adımı `needs_clarification` ya da `failed`'a
çevirebilir, (3) varsa sonraki fazın alt görevlerini üretir ve (4) geriye alt
görev kalmadıysa ve diğer tüm görev adımları sonlandıysa görevi tamamlandı olarak
işaretler ve nihai sonucu teslim eder. Geriye, çözümlenen durumu, hata/soru
metnini, kuyruğa alınan alt görevleri ve yakalanan artefaktları taşıyan bir
`AdvanceResult` döndürür. Her aşama savunmacı biçimde sarılmıştır; artefakt
yakalama ya da görev tamamlamadaki bir hata kararı asla bozmaz.

**Yapmadıkları**: ajan çalıştırmaz veya LLM çağrısı yapmaz, sonraki modeli seçmez
ya da puanlamaz, faz-geçiş mantığına sahip değildir (sonraki alt görevleri
engine'den ister; bu ilkel yoksa temiz bir no-op olur ve faz yönlendirme çağıran
tarafta kalır). Kendini de göndermez — ajan yolunda değil, host'un `workflow_advance`
eylemi için mekanik yürütücüsü tarafından çağrılır.

### Genel API

Tüm paket tek bir fonksiyon ve sonuç tipinden ibarettir. Çağıran taraf, biten her
iş akışı adımı için bir kez bunu await eder ve `.status`'a göre dallanır:

```python
from workflow_engine import advance, AdvanceResult

result: AdvanceResult = await advance(
    mission_id=46,
    completed_task_id=3797,
    previous_result={"status": "completed", "result": "<markdown>"},
)

# AdvanceResult alanları:
#   status: str            -> "completed" | "needs_clarification" | "failed"
#   error: str             -> hata metni ya da açıklama sorusu
#   next_subtasks: list[dict]  -> kuyruğa alınacak faz işi (bugün boş; Tuzaklar'a bak)
#   artifacts: dict[str, Any]  -> {artefakt_adı: içerik} bu adımda yakalananlar

if result.status == "needs_clarification":
    ...  # result.error'ı sor, görevi duraklat
elif result.status == "failed":
    ...  # result.error ile DLQ
elif result.next_subtasks:
    ...  # result.next_subtasks'i kuyruğa al
else:
    ...  # görev temiz ilerledi (ve muhtemelen tamamlandı)
```

`previous_result`, biten adımın yüküdür. Adım-sonrası kanca, şema doğrulaması için
onun `result` anahtarını okur; taşınan yük bunu içermeyen kısaltılmış bir özetse,
`advance` tam sonuç dizgesini görevin DB satırından geri doldurur; böylece bitiş
kapısı gerçek içeriği görür (bkz. Tuzaklar).

### Mimari

İnce bir cephe. `advance` sıralamayı ve savunmacı sınırları sahiplenir; her aşama
host'un iş akışı motoru iç bileşenlerindeki bir ilkele devreder. Herhangi bir aşama
sonucu kısa devre yaptırabilir:

```
advance(mission_id, completed_task_id, previous_result)
  │
  ├─ görevi yükle; bulunamadı        → AdvanceResult(failed)
  ├─ iş akışı adımı değil            → AdvanceResult(completed, no-op)
  │
  ├─ 1. artefaktları yakala          (best-effort; .artifacts'ı doldurur)
  ├─ 2. adım-sonrası bitiş kancası   → çevirebilir → needs_clarification | failed
  ├─ 3. sonraki faz alt görevleri    (recipe ilkeli yoksa no-op)
  └─ 4. görev tamamlama kontrolü     (yalnız alt görev yoksa & status completed)
          └─ tüm görevler sonlandı → görevi tamamlandı işaretle
                                    → görev sandbox'ını yık (best-effort)
                                    → nihai sonuç + özet teslim (best-effort)
```

### Bağımlılıklar

Bu paketin kendine ait **üçüncü taraf bağımlılığı yoktur**. Gerçek bağı, host
uygulamasının iç bileşenlerinedir; bunlar `advance` içinde lazy import edilir,
böylece o iç bileşenler yokken bile modül yüklenir:

- **Host iş akışı motoru iç bileşenleri** — dört aşama, host'un bitiş kancasına
  (`is_workflow_step`, `post_execute_workflow_step`, `get_artifact_store`),
  pipeline-artefakt çıkarıcısına ve — varsa — recipe-advance ilkeline devreder. Yük
  taşıyan bağımlılık budur: bunlar olmadan `advance` başarısız olmak yerine bir
  no-op'a iner.
- **Host veritabanı** — biten görevi ve onun görev/kardeş görevlerini okur, görevin
  `completed` durumunu yazar.
- **Host çalışma alanı + shell** — artefakt yakalama için görev başına çalışma
  alanını çözer ve tamamlamada görev sandbox'ını yıkar (best-effort).
- **Host Telegram arayüzü** — bir görev kapandığında nihai sonucu ve bir tamamlama
  özetini teslim eder (best-effort; bot ayakta değilse sessizce atlanır).

**Çağıran taraf host'un mekanik yürütücüsüdür**: `workflow_advance` eylemli bir
adım, yükünü (`mission_id`, `completed_task_id`, `previous_result`) buraya
yönlendirir ve dönen `AdvanceResult`'ı tekrar mekanik sonuç sözleşmesine eşler
(`needs_subtasks` / `needs_clarification` / `failed` / `completed`).

### Tuzaklar

- **Sonraki faz alt görevleri bugün bir no-op'tur.** 3. aşama, henüz bağlanmamış bir
  recipe-advance ilkelini import eder; bu yüzden `ImportError` dalı daima çalışır ve
  `next_subtasks` boş döner. Faz-geçiş mantığı hâlâ çağıran tarafta yaşar —
  `advance`'in sonraki fazı kuyruğa aldığını varsayma.
- **Kısaltılmış yükler bitiş kapısını takar.** Mekanik advance görevi çoğu zaman
  `previous_result`'ı, kancanın doğruladığı `result` anahtarı olmadan bir özet
  (örn. `{"summary": "..."}`) olarak taşır. `advance`, yük bunu içermediğinde tam
  sonuç dizgesini görevin DB satırından geri doldurur — bunu kaldır, üretici
  adımlar temiz kardeş çıktısına rağmen "schema requires X" ile DLQ'ya düşer.
- **Tamamlama teslimi defter-tutma ajanlarını atlar.** Kullanıcıya gönderilecek
  mesaj seçilirken mekanik / değerlendirici / özetleyici görevleri atlanır; böylece
  bir değerlendirme kararı ya da yapısal özet görevin cevabı olarak sızmaz. Önemsiz
  olmayan bir sonuca sahip son tamamlanmış *içerik* görevi kazanır.
- **Doğrulamadan sonrası tamamen best-effort'tur.** Artefakt yakalama, görev
  tamamlama, sandbox yıkımı ve Telegram teslimi; her biri, hatalarının dönen kararı
  asla bozmaması için sarılmıştır — eksik bir artefakt deposu ya da henüz başlamamış
  bir bot, görevin durumunu olduğu gibi bırakır.
- **Çağıranın yapması beklenen kontrolü kendi savunur.** `advance`,
  `is_workflow_step`'i yeniden kontrol eder ve iş akışı olmayan görevler için temiz
  bir no-op döner — çağıranların çağrıdan önce kontrol etmesi gerekse de.

### Testler

Testler paketin içinde değil, repo test paketinde bulunur:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests\test_workflow_engine_advance.py -q
```
