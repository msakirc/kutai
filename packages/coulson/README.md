# Coulson — Agent runtime (one task → many LLM calls)

> Named for the level-headed handler who runs the field agents. Coulson is that
> handler for KutAI's agents: hand it a profile and a task, it drives the whole
> reasoning loop to a finished answer.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** A single agent task is almost never one LLM call — it's
think, call a tool, read the result, think again, maybe self-correct, maybe ask
for clarification, then answer. Coulson owns that *whole* middle. The caller
admits a task and gets back a finished result dict; it never has to orchestrate
the iteration, count tokens against the context window, parse the model's
tool-call DSL, or decide when the agent has gone off the rails. Coulson fills the
gap between *deciding what to run* (the host's task master) and *making one LLM
call* (the host's per-call layer).

**What it really does.** Given a duck-typed **profile** (its tools, iteration
budget, prompts) and a **task** dict, it assembles the system + user prompt
(RAG, skills, workflow config), then runs one of two patterns: a **ReAct loop**
(think → tool → observe, repeated with per-iteration model selection and
checkpointed state) or a **single-shot** call (plan/classify, no tool loop).
Between iterations it runs sub-iteration **guards** that catch the agent
answering without using a required tool, hallucinating, skipping a declared web
search, or claiming a build step is done without writing the files it promised —
and feeds corrective nudges back into the loop. It can self-reflect on its own
final answer, recover from a checkpoint after a crash, and short-circuit a task
that needs real-world side effects it isn't cleared for.

**It does NOT** decide *which* task runs or enqueue work (the host's task master
does that and calls `execute`), make the LLM call itself or pick the model (it
asks the host's selector for a pick and the host's call layer to execute it),
manage llama-server or GPU, or define the agents (profiles live in the host;
Coulson only drives whatever shape it's handed).

## Public API

One entry point. The host's task master calls it once per task:

```python
from coulson import execute

result = await execute(profile, task, progress_callback=None)   # -> dict
```

- **`profile`** — duck-typed agent config (see shape below). Not constructed by
  Coulson; the host hands it in.
- **`task`** — a task dict with at least `id`, `description`, and a `context`
  field (dict, or JSON string Coulson will parse). Workflow-step tasks may also
  carry `mission_id`, `needs_real_tools`, `reversibility`, `tools_hint`, etc.
- **`progress_callback`** — optional `async fn(task_id, iteration, max_iter, summary)`.

**Return** is a result dict whose `status` is one of: `completed`, `exhausted`
(iteration budget spent), `needs_subtasks`, `needs_clarification`,
`needs_review`, `cancelled`, `failed`, or `blocked_on_founder_action` (the task
needed real-world side effects it wasn't cleared for). Always present:
`status` and `result`; completed/exhausted results also carry `model`, `cost`,
`difficulty`, `iterations`, and (completed) `tools_used_names` / `tool_calls`.

### The profile shape

`profile` is anything matching the attribute surface of the host's base agent
class — it is **not** imported from Coulson. Required attributes:

| attribute | type | role |
|---|---|---|
| `name` | `str` | agent identity (prompt override lookup, requirements) |
| `description` | `str` | human-readable role |
| `allowed_tools` | `list[str] \| None` | tool whitelist (`None` = all tools) |
| `max_iterations` | `int` | ReAct loop budget |
| `execution_pattern` | `str` | `"react_loop"` or `"single_shot"` |
| `enable_self_reflection` | `bool` | run the post-final self-review pass |
| `min_confidence` | `int` | confidence gate (`0` = disabled) |
| `can_create_subtasks` | `bool` | may emit `needs_subtasks` |
| `_suppress_clarification` | `bool` | block clarification requests |
| `default_tier`, `min_tier` | `str` | model-tier hints |
| `get_system_prompt(task) -> str` | method | base system prompt |
| `_build_context(task) -> str` | async method | builds the user-message context |

### Secondary surface — oncall handler registry

A small pluggable registry lets host domains register oncall-action verbs without
editing a hardcoded whitelist:

```python
from coulson.agent_handlers.registry import register_handler, lookup_handler

register_handler("ops", "restart_service", _my_async_handler)   # fn(verb, params, mission_id) -> dict
```

Also exports `lookup_handler`, `list_verbs(domain=None)`, `get_whitelist(domain)`,
`is_registered(domain, name)`.

## Architecture

`execute` is a thin setup-and-route shell; the two execution patterns do the work:

```
execute(profile, task)
  ├─ setup: DB prompt override · tools_hint override · auto-strip file/web/write
  │         tools · _suppress_clarification flag · live workflow-step refresh
  ├─ real-world gate: needs_real_tools? → re-check admission
  │         · not cleared → return status="blocked_on_founder_action" (no LLM call)
  │         · cleared     → inject a "use the real tool, don't fabricate" warning
  └─ route by execution_pattern
        ├─ single_shot → one call → parse → return
        └─ react_loop  → for each iteration:
              build/restore messages → pick model → call → parse action
                ├─ tool_call      → run tool (incl. host api/mcp tools) → observe
                ├─ final_answer   → sub-iter guards (hallucination · search ·
                │                   grounding · self_critique · format)
                │                   → [self_reflect] → return "completed"
                ├─ clarify        → "needs_clarification"
                └─ decompose      → "needs_subtasks"
              (checkpoint state each iter; trim/prune on context-window pressure)
        budget spent → return "exhausted"
  finally: restore any allowed_tools the setup phase overrode
```

## Key Modules

| module | role |
|---|---|
| `__init__.py` | public `execute()` — setup phase, real-world gate, pattern routing |
| `react.py` | the ReAct loop: iterate, select-per-iter, tool dispatch, guards, finish |
| `single_shot.py` | one-call path for plan/classify profiles |
| `dispatch_helpers.py` | per-iteration model pick + map the call result to a response dict |
| `context.py` | system + user prompt assembly (RAG, skills, workflow config) |
| `parsing.py` | ReAct DSL parser, tool-name alias map, function-call response handling |
| `window.py` | context-window token count / trim / prune |
| `guards.py` | the five sub-iteration guards + task-type heuristics |
| `grounding.py` | declared-`produces` vs actual-`write_file` comparison (pure) |
| `self_critique.py` | the agent's one-shot review of its own diff |
| `streaming_guards.py` | mid-stream degeneracy / secret-leak guards |
| `escalation.py` | message trim when a task escalates mid-flight |
| `reflection.py` | post-final self-review pass (re-exports the host's reflection blocks) |
| `checkpoint.py` | per-task state save / restore + log |
| `validation.py` | refusal / length / empty checks + quality-checker wrappers |
| `system_prompt_blocks.py` | injectable warning blocks (e.g. real-world side effects) |
| `skill_render.py` | render matched skills into the prompt |
| `agent_handlers/registry.py` | pluggable oncall-action verb registry |

## Dependencies

Coulson sits squarely in the LLM-execution stack. Its hard runtime dependencies
are three sibling packages (declared in `pyproject.toml`):

- **`fatih_hoca`** (top-level import) — Coulson asks it to build per-task
  requirements (`requirements_for`) and to pick a model per iteration (`select`,
  `is_servable`, `mid_task_urgency`), and imports its `ModelRequirements` /
  `Failure` / `Pick` types directly. The per-iteration retry surface lives in
  Coulson; the *pick* is delegated.
- **`hallederiz_kadir`** — imported inside the loop (circular-import hygiene) to
  execute the chosen model's call.
- **`nerd_herd`** — declared dependency, pulled in transitively for the snapshots
  that feed model selection.

Softer couplings are imported lazily and described by role, not named: a
degenerate-output quality checker (called before accepting a result or a
self-reflection correction), the host's real-world-action admission gate, and
`src/` infra (`db`, `tools`, `workspace`, logging, prompt versions) for
config-refresh, tool execution, and context. The demand-signal source and the
host's api/mcp tool catalog are optional, imported only when present.

## Gotchas

- **`profile` is a contract, not a class.** Coulson never imports the agent base
  class. Anything with the attribute surface above works; a missing attribute
  fails at runtime, not import.
- **`execute` mutates `profile.allowed_tools`** during setup (tools_hint,
  auto-strip) and restores it in a `finally`. Don't share one profile instance
  across concurrent `execute` calls — the snapshot/restore is per-call, not
  re-entrant.
- **Auto-strip is silent.** For a step with a structured-output `artifact_schema`,
  write tools (`write_file`, `apply_diff`, …) are stripped unless the step sets
  `_allow_write_tools` — small models stuff output into `write_file` args and the
  parser chokes. File/web tools strip on `_strip_file_tools` / `_strip_web_tools`.
- **Checkpoints are skipped on retry.** When the task context carries
  `_schema_error`, Coulson rebuilds the prompt from scratch instead of resuming —
  resuming would replay the old (bad) prompt and never show the model the retry
  nudge.
- **`needs_real_tools` short-circuits before any LLM cost.** A task flagged for
  real-world side effects is re-admitted first; if not cleared it returns
  `blocked_on_founder_action` with zero iterations and no model call.
- **Guards run only on `final_answer`.** They fire *inside* one iteration and feed
  a correction back; `self_critique` uses a separate budget so it doesn't eat the
  guard-correction slots.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\coulson\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Adını, saha ajanlarını yöneten soğukkanlı amirden alır. Coulson, KutAI'nin
> ajanları için o amirdir: ona bir profil ve bir görev verin, tüm akıl yürütme
> döngüsünü bitmiş bir cevaba kadar sürer.

## Amaç

**Ne işe yarar.** Tek bir ajan görevi neredeyse hiçbir zaman tek bir LLM çağrısı
değildir — düşün, bir araç çağır, sonucu oku, tekrar düşün, belki kendini düzelt,
belki açıklama iste, sonra cevap ver. Coulson bu *tüm* orta kısmı üstlenir. Çağıran
taraf bir görevi kabul ettirir ve bitmiş bir sonuç dict'i geri alır; iterasyonu
yönetmek, token'ı bağlam penceresine karşı saymak, modelin araç-çağrı DSL'ini
ayrıştırmak ya da ajanın ne zaman yoldan çıktığına karar vermek zorunda kalmaz.
Coulson, *neyin çalışacağına karar vermek* (host'un görev ustası) ile *tek bir LLM
çağrısı yapmak* (host'un çağrı katmanı) arasındaki boşluğu doldurur.

**Gerçekte ne yapar.** Duck-typed bir **profil** (araçları, iterasyon bütçesi,
prompt'ları) ve bir **görev** dict'i verildiğinde, önce sistem + kullanıcı
prompt'unu kurar (RAG, beceriler, workflow yapılandırması), sonra iki kalıptan
birini çalıştırır: bir **ReAct döngüsü** (düşün → araç → gözlemle; iterasyon
başına model seçimi ve checkpoint'lenen durumla tekrarlanır) ya da **tek-atış** bir
çağrı (planla/sınıflandır, araç döngüsü yok). İterasyonlar arasında, ajanın gerekli
bir aracı kullanmadan cevap vermesini, halüsinasyon görmesini, beyan edilen bir web
aramasını atlamasını ya da söz verdiği dosyaları yazmadan bir yapım adımını bitmiş
ilan etmesini yakalayan **alt-iterasyon korumalarını** çalıştırır — ve düzeltici
uyarıları döngüye geri besler. Kendi nihai cevabı üzerine öz-değerlendirme yapabilir,
bir çökmeden sonra checkpoint'ten toparlanabilir ve yetkili olmadığı gerçek-dünya yan
etkileri gerektiren bir görevi kısa devre yapabilir.

**Yapmadıkları**: hangi görevin çalışacağına karar vermez veya iş kuyruğa almaz
(bunu host'un görev ustası yapar ve `execute`'u çağırır), LLM çağrısını kendisi
yapmaz veya modeli seçmez (host'un seçicisinden bir seçim ister ve host'un çağrı
katmanından yürütmesini ister), llama-server veya GPU yönetmez, ajanları tanımlamaz
(profiller host'ta yaşar; Coulson yalnızca kendisine verilen şekli sürer).

## Genel API

Tek bir giriş noktası. Host'un görev ustası onu görev başına bir kez çağırır:

```python
from coulson import execute

result = await execute(profile, task, progress_callback=None)   # -> dict
```

- **`profile`** — duck-typed ajan yapılandırması (şekli aşağıda). Coulson tarafından
  kurulmaz; host onu içeri verir.
- **`task`** — en azından `id`, `description` ve bir `context` alanı (dict ya da
  Coulson'un ayrıştıracağı JSON string) taşıyan bir görev dict'i. Workflow-adımı
  görevleri ayrıca `mission_id`, `needs_real_tools`, `reversibility`, `tools_hint`
  vb. taşıyabilir.
- **`progress_callback`** — opsiyonel `async fn(task_id, iteration, max_iter, summary)`.

**Dönüş**, `status`'u şunlardan biri olan bir sonuç dict'idir: `completed`,
`exhausted` (iterasyon bütçesi bitti), `needs_subtasks`, `needs_clarification`,
`needs_review`, `cancelled`, `failed` ya da `blocked_on_founder_action` (görev,
yetkili olmadığı gerçek-dünya yan etkileri gerektirdi). Her zaman bulunur: `status`
ve `result`; completed/exhausted sonuçları ayrıca `model`, `cost`, `difficulty`,
`iterations` ve (completed) `tools_used_names` / `tool_calls` taşır.

### Profil şekli

`profile`, host'un temel ajan sınıfının nitelik yüzeyine uyan herhangi bir şeydir —
Coulson'dan **import edilmez**. Gerekli nitelikler:

| nitelik | tip | rolü |
|---|---|---|
| `name` | `str` | ajan kimliği (prompt override araması, gereksinimler) |
| `description` | `str` | okunabilir rol |
| `allowed_tools` | `list[str] \| None` | araç beyaz listesi (`None` = tüm araçlar) |
| `max_iterations` | `int` | ReAct döngü bütçesi |
| `execution_pattern` | `str` | `"react_loop"` ya da `"single_shot"` |
| `enable_self_reflection` | `bool` | nihai-sonrası öz-değerlendirme geçişini çalıştır |
| `min_confidence` | `int` | güven kapısı (`0` = kapalı) |
| `can_create_subtasks` | `bool` | `needs_subtasks` yayabilir |
| `_suppress_clarification` | `bool` | açıklama isteklerini engelle |
| `default_tier`, `min_tier` | `str` | model-katman ipuçları |
| `get_system_prompt(task) -> str` | metot | temel sistem prompt'u |
| `_build_context(task) -> str` | async metot | kullanıcı-mesajı bağlamını kurar |

### İkincil yüzey — oncall handler kayıt defteri

Küçük, takılabilir bir kayıt defteri, host alanlarının sabit-kodlu bir beyaz liste
düzenlemeden oncall-aksiyon fiilleri kaydetmesine izin verir:

```python
from coulson.agent_handlers.registry import register_handler, lookup_handler

register_handler("ops", "restart_service", _my_async_handler)   # fn(verb, params, mission_id) -> dict
```

Ayrıca `lookup_handler`, `list_verbs(domain=None)`, `get_whitelist(domain)`,
`is_registered(domain, name)` export eder.

## Mimari

`execute` ince bir kurulum-ve-yönlendirme kabuğudur; işi iki yürütme kalıbı yapar:

```
execute(profile, task)
  ├─ kurulum: DB prompt override · tools_hint override · file/web/write
  │           araç-strip'i · _suppress_clarification bayrağı · canlı workflow-adım yenileme
  ├─ gerçek-dünya kapısı: needs_real_tools? → kabulü yeniden kontrol et
  │           · yetkisiz → status="blocked_on_founder_action" döndür (LLM çağrısı yok)
  │           · yetkili  → "gerçek aracı kullan, uydurma" uyarısı enjekte et
  └─ execution_pattern'a göre yönlendir
        ├─ single_shot → tek çağrı → ayrıştır → döndür
        └─ react_loop  → her iterasyon için:
              mesaj kur/geri yükle → model seç → çağır → aksiyon ayrıştır
                ├─ tool_call      → aracı çalıştır (host api/mcp araçları dahil) → gözlemle
                ├─ final_answer   → alt-iter korumaları (halüsinasyon · arama ·
                │                   grounding · self_critique · format)
                │                   → [self_reflect] → "completed" döndür
                ├─ clarify        → "needs_clarification"
                └─ decompose      → "needs_subtasks"
              (her iter durumu checkpoint'le; pencere baskısında trim/prune)
        bütçe bitti → "exhausted" döndür
  finally: kurulum fazının ezdiği allowed_tools'u geri yükle
```

## Ana Modüller

| modül | rolü |
|---|---|
| `__init__.py` | genel `execute()` — kurulum fazı, gerçek-dünya kapısı, kalıp yönlendirme |
| `react.py` | ReAct döngüsü: iterasyon, iter-başı seçim, araç dispatch, korumalar, bitiş |
| `single_shot.py` | plan/sınıflandır profilleri için tek-çağrı yolu |
| `dispatch_helpers.py` | iterasyon başına model seçimi + çağrı sonucunu yanıt dict'ine eşle |
| `context.py` | sistem + kullanıcı prompt kurulumu (RAG, beceriler, workflow yapılandırması) |
| `parsing.py` | ReAct DSL ayrıştırıcı, araç-adı alias haritası, function-call yanıt işleme |
| `window.py` | bağlam-penceresi token sayımı / trim / prune |
| `guards.py` | beş alt-iterasyon koruması + görev-tipi sezgileri |
| `grounding.py` | beyan edilen `produces` vs gerçek `write_file` karşılaştırması (saf) |
| `self_critique.py` | ajanın kendi diff'i üzerine tek-atış incelemesi |
| `streaming_guards.py` | akış-içi yozlaşma / sır-sızıntısı korumaları |
| `escalation.py` | görev akış-içi yükseldiğinde mesaj trim'i |
| `reflection.py` | nihai-sonrası öz-değerlendirme geçişi (host'un reflection bloklarını re-export eder) |
| `checkpoint.py` | görev başına durum kaydet / geri yükle + log |
| `validation.py` | ret / uzunluk / boş kontrolleri + kalite-denetçisi sarmalayıcıları |
| `system_prompt_blocks.py` | enjekte edilebilir uyarı blokları (ör. gerçek-dünya yan etkileri) |
| `skill_render.py` | eşleşen becerileri prompt'a render et |
| `agent_handlers/registry.py` | takılabilir oncall-aksiyon fiil kayıt defteri |

## Bağımlılıklar

Coulson tam olarak LLM-yürütme yığınında oturur. Sıkı çalışma-zamanı
bağımlılıkları üç kardeş pakettir (`pyproject.toml`'da beyan edilir):

- **`fatih_hoca`** (üst düzey import) — Coulson ondan görev başına gereksinimleri
  kurmasını (`requirements_for`) ve iterasyon başına bir model seçmesini (`select`,
  `is_servable`, `mid_task_urgency`) ister; `ModelRequirements` / `Failure` / `Pick`
  tiplerini doğrudan import eder. İterasyon başına yeniden-deneme yüzeyi Coulson'da
  yaşar; *seçim* delege edilir.
- **`hallederiz_kadir`** — seçilen modelin çağrısını yürütmek için döngü içinde
  (döngüsel-import hijyeni) import edilir.
- **`nerd_herd`** — beyan edilen bağımlılık; model seçimini besleyen anlık
  görüntüler için geçişli olarak çekilir.

Daha gevşek bağlar fonksiyon içinde lazy import edilir ve isimle değil rolle
anılır: yozlaşmış çıktıyı reddeden bir kalite denetçisi (bir sonucu ya da
öz-değerlendirme düzeltmesini kabul etmeden önce çağrılır), host'un gerçek-dünya
eylem admission kapısı, ve yapılandırma yenileme / araç yürütme / bağlam için
`src/` altyapısı (`db`, `tools`, `workspace`, logging, prompt sürümleri).
Talep-sinyali kaynağı ve host'un api/mcp araç kataloğu opsiyoneldir, yalnızca
mevcutsa import edilir.

## Tuzaklar

- **`profile` bir sözleşmedir, bir sınıf değil.** Coulson ajan temel sınıfını asla
  import etmez. Yukarıdaki nitelik yüzeyine sahip her şey çalışır; eksik bir nitelik
  import'ta değil çalışma-zamanında patlar.
- **`execute`, `profile.allowed_tools`'u değiştirir** kurulum sırasında (tools_hint,
  auto-strip) ve bir `finally`'de geri yükler. Tek bir profil örneğini eşzamanlı
  `execute` çağrıları arasında paylaşmayın — snapshot/geri-yükleme çağrı başınadır,
  yeniden-girişli değildir.
- **Auto-strip sessizdir.** Yapılandırılmış-çıktı `artifact_schema`'sı olan bir adımda,
  adım `_allow_write_tools` ayarlamadıkça yazma araçları (`write_file`, `apply_diff`,
  …) çıkarılır — küçük modeller çıktıyı `write_file` argümanlarına tıkar ve ayrıştırıcı
  boğulur. File/web araçları `_strip_file_tools` / `_strip_web_tools` üzerine çıkar.
- **Checkpoint'ler yeniden-denemede atlanır.** Görev bağlamı `_schema_error` taşıdığında
  Coulson, devam etmek yerine prompt'u sıfırdan kurar — devam etmek eski (bozuk)
  prompt'u tekrar oynatır ve modele yeniden-deneme uyarısını asla göstermez.
- **`needs_real_tools` herhangi bir LLM maliyetinden önce kısa devre yapar.**
  Gerçek-dünya yan etkileri için işaretlenmiş bir görev önce yeniden kabul edilir;
  yetkili değilse sıfır iterasyon ve model çağrısı olmadan `blocked_on_founder_action`
  döndürür.
- **Korumalar yalnızca `final_answer`'da çalışır.** Bir iterasyon *içinde* tetiklenir
  ve bir düzeltmeyi geri beslerler; `self_critique` ayrı bir bütçe kullanır, böylece
  koruma-düzeltme slotlarını yemez.

## Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\coulson\ -q
```
