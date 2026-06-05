# Intersect — Per-task skill match-and-expose gate

> Where a task and the skill catalog *intersect*: once per task, just before
> dispatch, it decides which catalogued skills the task should actually see and
> in what form.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** A task is about to run, and somewhere in the skill
catalog there may be a recipe, a prompt hint, an API verb, or an MCP tool that
makes the task go better. Intersect removes the problem of *deciding what to do
with those matches*. The caller hands it the raw task; it hands back the same
task, now carrying a `skills` envelope of exactly the right matches in exactly
the right shape — or, for a high-confidence mechanizable recipe, it reroutes the
task to run the recipe directly. The caller never reasons about trust scores,
exposure tiers, budget caps, or arg-binding.

**What it really does.** One async function, `flash(task)`. It asks the catalog
for candidate skills, scores each one (vector match × source trust × owner trust
× keyword-hint bonus), resolves same-kind conflicts down to one winner per slot,
and assigns every survivor an **exposure class** from its vet tier and
confidence: `inject` (prose or pre-bound text injected into the agent's
context), `tool` (a callable API/MCP surface, budget-capped), `preempt` (a
trusted mechanizable recipe that *replaces* the task), or `quarantine`
(suppressed). For parametric skills it statically binds arguments from the task
context, falling back to an embedding-keyed bind cache. Then it writes usage
telemetry and returns.

**It does NOT** make any LLM call — binding is static-or-cached, never
model-driven (a hard rule: only the task master calls the dispatcher). It does
NOT discover, fetch, vet, or rank the catalog (that is the catalog package's
job; intersect only reads what `query` returns). It does NOT execute the recipe
it routes to — it only sets the routing fields and lets the mechanical lane run
it. And it **never raises**: any failure degrades to `task["skills"] = []` and
returns the task untouched, so dispatch is never blocked.

## Public API

One entry point, invoked once per task before dispatch. The four `THETA_*`
constants are the confidence thresholds the exposure classifier floors against
(exported so the host can read them; tuning lives in `exposure.py`).

```python
from intersect import flash, THETA_PREEMPT, THETA_INJECT, THETA_TOOL, THETA_MIN

task = await flash(task)   # -> the same dict, mutated; always has task["skills"]
```

`flash(task: dict) -> dict` always returns the **same** task dict it was given,
mutated in place, with one of two outcomes:

- **Envelope attached** — `task["skills"]` is a `list[dict]`, each entry an
  exposed skill with `artifact_id`, `name`, `exposure_class`
  (`inject` / `tool`), `applies_to` (`"execution"`), `render`
  (`"prose"` / `"prebind"`), `payload` (`body` / `kind` / `bound_args`), and
  `confidence`. Empty list = no match, or graceful degrade.
- **Preempt reroute** — when a trusted high-confidence mechanizable recipe wins,
  `task["skills"]` is `[]` and the task is instead routed to the mechanical lane:
  `task["runner"] = "mechanical"` and
  `task["payload"] = {"action": "yalayut_recipe", "recipe_id": ..., "args": ...}`.

Top-level exports (`__all__`): the function `flash`; the thresholds
`THETA_PREEMPT` (0.80), `THETA_INJECT` (0.55), `THETA_TOOL` (0.45),
`THETA_MIN` (0.30). Everything else (`scoring`, `binding`, `budget`,
`exposure`, `telemetry`) is internal.

## Architecture

A single linear pass per task. Any step's failure short-circuits to the graceful
degrade (`skills = []`):

```
flash(task)
  │  parse task["context"]; if recipe_lookup is False → return untouched
  ├─ candidates = catalog query(task)        (raw task dict, not the bind ctx)
  │     └─ none → fire a demand-miss signal, return
  ├─ score    each: confidence = match × source_trust × owner_trust × hint_bonus
  │            (env-gated / not-ready artifacts skipped silently)
  ├─ conflict resolve: highest confidence per slot; same-kind agent_configs
  │            compete, prompt hints stack; losers logged as conflict_losers
  ├─ classify each by (tier × kind × confidence) → inject | tool | preempt | quarantine
  ├─ bind     parametric args statically; incomplete → bind-cache lookup;
  │            newly-complete → seed the bind cache
  ├─ preempt? → set runner=mechanical + payload, skills=[], emit telemetry, return
  ├─ budget   caps on tool class (api ≤3/step; mcp ≤3/server, ≤6/step)
  ├─ attach   the trimmed skills envelope
  └─ emit     yalayut_usage telemetry (exposed + conflict-losers + budget drops)
```

## Key Modules

| module | role |
|---|---|
| `flash.py` | the `flash(task)` orchestration — the only public surface; owns the pass above and the graceful-degrade boundary |
| `scoring.py` | `score_artifact` (match × trusts × hint, clamped 0–1) and `compute_hint_bonus` (keyword overlap vs the step's recipe hint, ≤1.30×) |
| `exposure.py` | `classify` (tier-ceiling + θ-threshold → exposure class), `render_variant` (prose vs prebind), and the `THETA_*` constants |
| `binding.py` | `static_bind` (walk `bind_from` dotted paths), plus the embedding-keyed `lookup_bind_cache` / `write_bind_cache` |
| `budget.py` | `apply_caps` — per-step API/MCP tool caps; inject/preempt pass through uncapped |
| `telemetry.py` | `record_usage` — one `yalayut_usage` row per considered artifact; never raises |

## Dependencies

Intersect's one genuine peer is the **catalog package** (`yalayut`) — a hard
runtime import, not a generic seam:

- **Catalog reads** — `flash` calls `yalayut.query(task)` for candidates and
  consumes the `Artifact` attributes it returns (`vet_tier`, `kind`, `score`,
  `inputs_schema`, `body_excerpt`, `env_status`, …). On an empty result it calls
  `yalayut.record_demand_signal(...)` to log a proactive demand miss. This is the
  only sibling package intersect couples to, and it is declared in `pyproject`.
- **Host DB** (`src.infra.db.get_db`) — reads trust scores from `yalayut_sources`
  / `yalayut_owners`, reads/writes the `yalayut_bind_cache`, and writes
  `yalayut_usage` telemetry. Every DB touch is wrapped in `try/except` and
  degrades to a neutral default (missing trust row → trust `1.0`).
- **Embeddings** (`src.memory.embeddings.get_embedding`) — only for the bind
  cache key (multilingual-e5-base, 768d). A failed embedding simply means every
  cache lookup misses — safe.

It does **not** import the LLM dispatcher, the model selector, or the
orchestrator. The host calls `flash` once per task in the dispatch pump; nothing
calls back into intersect.

## Gotchas

- **Pass the raw task to the catalog.** `flash` builds a nested binding context
  internally (`_build_task_ctx`) for arg resolution, but `yalayut.query` receives
  the **raw** task dict (top-level `id` / `title`). Passing the nested dict yields
  an empty query and zero matches — there is a regression test guarding exactly
  this.
- **`recipe_lookup: False` opts the step out entirely** — design/architecture/
  debug steps that set it in their context get the task back untouched, no query.
- **Preempt is gated and can downgrade.** A preempt-classified recipe with any
  unbound required field, or when the preempt feature flag is off, silently
  downgrades to `inject` so the recipe body still surfaces in context instead of
  routing to a lane that can't run it. The **first** complete preempt wins and
  owns the whole task (loop breaks).
- **Tier ceiling is hard.** Tier 2 and 3 artifacts are always quarantined
  (sandbox exposure is not implemented); only Tier 0 can ever `preempt`.
- **Trust defaults to neutral, not zero.** A missing source/owner trust row
  returns `1.0` — an unseeded source must not silently zero out every confidence.
- **Bind args are a contract for prebind.** `render_variant` only returns
  `"prebind"` when the artifact is parametric *and* every schema field is bound;
  any missing field falls back to `"prose"`.
- **Telemetry and demand signals never raise** — they swallow their own errors so
  dispatch is never disturbed.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\intersect\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Bir görev ile yetenek kataloğunun *kesiştiği* yer: her görev için bir kez, tam
> sevk öncesinde, görevin hangi katalog yeteneklerini gerçekten görmesi
> gerektiğine ve bunları hangi biçimde göreceğine karar verir.

### Amaç

**Ne işe yarar.** Bir görev çalışmak üzeredir ve yetenek kataloğunda bir yerde,
görevi daha iyi yürütecek bir reçete, bir prompt ipucu, bir API fiili ya da bir
MCP aracı olabilir. Intersect, *bu eşleşmelerle ne yapılacağına karar verme*
sorununu ortadan kaldırır. Çağıran taraf ham görevi verir; intersect aynı görevi,
artık doğru eşleşmeleri doğru biçimde taşıyan bir `skills` zarfıyla geri verir —
ya da yüksek güvenli mekanize edilebilir bir reçete için görevi reçeteyi doğrudan
çalıştıracak şekilde yeniden yönlendirir. Çağıran taraf güven skorlarını, ifşa
katmanlarını, bütçe sınırlarını ya da argüman bağlamayı hiç düşünmez.

**Gerçekte ne yapar.** Tek bir async fonksiyon, `flash(task)`. Kataloğa aday
yetenekleri sorar, her birini puanlar (vektör eşleşmesi × kaynak güveni × sahip
güveni × anahtar-kelime ipucu bonusu), aynı türdeki çakışmaları slot başına tek
kazanana indirir ve hayatta kalan her adaya, vet katmanı ve güvenine göre bir
**ifşa sınıfı** atar: `inject` (ajanın bağlamına enjekte edilen düz ya da önceden
bağlanmış metin), `tool` (çağrılabilir bir API/MCP yüzeyi, bütçeyle sınırlı),
`preempt` (görevin *yerine geçen* güvenilir, mekanize edilebilir bir reçete) ya da
`quarantine` (bastırılmış). Parametrik yetenekler için argümanları görev
bağlamından statik olarak bağlar, başarısız olursa embedding anahtarlı bir bağlama
önbelleğine düşer. Sonra kullanım telemetrisini yazar ve döner.

**Yapmadıkları**: hiçbir LLM çağrısı yapmaz — bağlama statik ya da önbellekten
gelir, asla modelden değil (katı kural: dispatcher'ı yalnızca görev ustası
çağırır). Kataloğu keşfetmez, çekmez, denetlemez ya da sıralamaz (bu, katalog
paketinin işidir; intersect yalnızca `query`'nin döndürdüğünü okur). Yönlendirdiği
reçeteyi çalıştırmaz — yalnızca yönlendirme alanlarını ayarlar, çalıştırmayı
mekanik şerit yapar. Ve **asla hata fırlatmaz**: herhangi bir başarısızlık
`task["skills"] = []`'e düşer ve görevi olduğu gibi geri verir, böylece sevk asla
engellenmez.

### Genel API

Tek bir giriş noktası, her görev için sevk öncesi bir kez çağrılır. Dört `THETA_*`
sabiti, ifşa sınıflandırıcısının güveni karşılaştırdığı eşiklerdir (host
okuyabilsin diye export edilir; ayar `exposure.py`'de yaşar).

```python
from intersect import flash, THETA_PREEMPT, THETA_INJECT, THETA_TOOL, THETA_MIN

task = await flash(task)   # -> aynı dict, yerinde değiştirilir; her zaman task["skills"] var
```

`flash(task: dict) -> dict` her zaman kendisine verilen **aynı** görev dict'ini
yerinde değiştirerek döndürür; iki sonuçtan biriyle:

- **Zarf eklenir** — `task["skills"]` bir `list[dict]`'tir, her giriş ifşa edilmiş
  bir yetenektir: `artifact_id`, `name`, `exposure_class` (`inject` / `tool`),
  `applies_to` (`"execution"`), `render` (`"prose"` / `"prebind"`), `payload`
  (`body` / `kind` / `bound_args`) ve `confidence`. Boş liste = eşleşme yok ya da
  zarif düşüş.
- **Preempt yönlendirme** — güvenilir, yüksek güvenli, mekanize edilebilir bir
  reçete kazandığında `task["skills"]` `[]` olur ve görev bunun yerine mekanik
  şeride yönlendirilir: `task["runner"] = "mechanical"` ve
  `task["payload"] = {"action": "yalayut_recipe", "recipe_id": ..., "args": ...}`.

Üst düzey export'lar (`__all__`): `flash` fonksiyonu; eşikler `THETA_PREEMPT`
(0.80), `THETA_INJECT` (0.55), `THETA_TOOL` (0.45), `THETA_MIN` (0.30). Geri kalan
her şey (`scoring`, `binding`, `budget`, `exposure`, `telemetry`) iç kullanımdır.

### Mimari

Görev başına tek bir doğrusal geçiş. Herhangi bir adımın başarısızlığı zarif
düşüşe kısa devre yapar (`skills = []`):

```
flash(task)
  │  task["context"]'i parse et; recipe_lookup False ise → dokunmadan dön
  ├─ candidates = katalog query(task)        (ham task dict, bağlama ctx'i değil)
  │     └─ yok → talep-kaçırma sinyali at, dön
  ├─ puanla  her biri: güven = eşleşme × kaynak_güveni × sahip_güveni × ipucu_bonusu
  │            (env-kapılı / hazır-olmayan artifact'lar sessizce atlanır)
  ├─ çakışma çöz: slot başına en yüksek güven; aynı tür agent_config'ler
  │            yarışır, prompt ipuçları üst üste yığılır; kaybedenler loglanır
  ├─ sınıflandır (katman × tür × güven) → inject | tool | preempt | quarantine
  ├─ bağla   parametrik argümanları statik; eksikse → bağlama-önbelleği bakışı;
  │            yeni-tamamlanan → bağlama önbelleğine tohum at
  ├─ preempt? → runner=mechanical + payload ayarla, skills=[], telemetri, dön
  ├─ bütçe   tool sınıfına sınırlar (api ≤3/adım; mcp ≤3/sunucu, ≤6/adım)
  ├─ ekle    kırpılmış skills zarfını
  └─ yaz     yalayut_usage telemetrisi (ifşa + çakışma-kaybedenleri + bütçe düşenleri)
```

### Ana Modüller

| modül | rolü |
|---|---|
| `flash.py` | `flash(task)` orkestrasyonu — tek genel yüzey; yukarıdaki geçişi ve zarif-düşüş sınırını yönetir |
| `scoring.py` | `score_artifact` (eşleşme × güvenler × ipucu, 0–1 kıskaçlı) ve `compute_hint_bonus` (adımın reçete ipucuyla anahtar-kelime örtüşmesi, ≤1.30×) |
| `exposure.py` | `classify` (katman tavanı + θ eşiği → ifşa sınıfı), `render_variant` (prose vs prebind) ve `THETA_*` sabitleri |
| `binding.py` | `static_bind` (`bind_from` noktalı yolları yürür) ve embedding anahtarlı `lookup_bind_cache` / `write_bind_cache` |
| `budget.py` | `apply_caps` — adım başına API/MCP araç sınırları; inject/preempt sınırsız geçer |
| `telemetry.py` | `record_usage` — değerlendirilen her artifact için bir `yalayut_usage` satırı; asla hata fırlatmaz |

### Bağımlılıklar

Intersect'in tek gerçek dengi **katalog paketidir** (`yalayut`) — genel bir dikiş
değil, katı bir çalışma-zamanı import'u:

- **Katalog okumaları** — `flash`, adaylar için `yalayut.query(task)` çağırır ve
  döndürdüğü `Artifact` özniteliklerini tüketir (`vet_tier`, `kind`, `score`,
  `inputs_schema`, `body_excerpt`, `env_status`, …). Boş sonuçta, proaktif bir
  talep kaçırmasını loglamak için `yalayut.record_demand_signal(...)` çağırır.
  Intersect'in bağlandığı tek kardeş paket budur ve `pyproject`'te bildirilmiştir.
- **Host DB** (`src.infra.db.get_db`) — `yalayut_sources` / `yalayut_owners`'tan
  güven skorlarını okur, `yalayut_bind_cache`'i okur/yazar ve `yalayut_usage`
  telemetrisini yazar. Her DB dokunuşu `try/except` ile sarılıdır ve nötr bir
  varsayılana düşer (eksik güven satırı → güven `1.0`).
- **Embeddings** (`src.memory.embeddings.get_embedding`) — yalnızca bağlama
  önbelleği anahtarı için (multilingual-e5-base, 768d). Başarısız bir embedding
  yalnızca her önbellek bakışının ıskalaması demektir — güvenli.

LLM dispatcher'ı, model seçiciyi ya da orchestrator'ı import etmez. Host, sevk
pompasında her görev için `flash`'i bir kez çağırır; hiçbir şey intersect'e geri
çağrı yapmaz.

### Tuzaklar

- **Kataloğa ham görevi geçir.** `flash` argüman çözümü için içeride iç içe bir
  bağlama bağlamı kurar (`_build_task_ctx`), ama `yalayut.query` **ham** görev
  dict'ini alır (üst düzey `id` / `title`). İç içe dict'i geçmek boş bir sorgu ve
  sıfır eşleşme verir — tam bunu koruyan bir regresyon testi vardır.
- **`recipe_lookup: False` adımı tümüyle dışarıda bırakır** — bağlamında bunu
  ayarlayan tasarım/mimari/hata-ayıklama adımları görevi sorgusuz, dokunulmamış
  geri alır.
- **Preempt kapılıdır ve düşürülebilir.** Bağlanmamış zorunlu alanı olan ya da
  preempt özellik bayrağı kapalıyken preempt sınıflı bir reçete, sessizce
  `inject`'e düşer; böylece reçete gövdesi onu çalıştıramayacak bir şeride
  yönlenmek yerine yine de bağlamda görünür. **İlk** tam preempt kazanır ve tüm
  görevi sahiplenir (döngü kırılır).
- **Katman tavanı katıdır.** Katman 2 ve 3 artifact'ları her zaman karantinaya
  alınır (sandbox ifşası uygulanmamıştır); yalnızca Katman 0 `preempt` olabilir.
- **Güven sıfıra değil nötre düşer.** Eksik bir kaynak/sahip güven satırı `1.0`
  döner — denetlenmemiş bir kaynak her güveni sessizce sıfırlamamalıdır.
- **Bağlı argümanlar prebind için bir sözleşmedir.** `render_variant` yalnızca
  artifact parametrikken *ve* her şema alanı bağlıyken `"prebind"` döner; eksik
  herhangi bir alan `"prose"`'a düşer.
- **Telemetri ve talep sinyalleri asla hata fırlatmaz** — kendi hatalarını yutar,
  böylece sevk asla rahatsız olmaz.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\intersect\ -q
```
