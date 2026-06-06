# Fatih Hoca — Model-selection brain

> Turkish for "Teacher Fatih": the one who knows every model's strengths and
> assigns the right one to each job. Given a task and the current state of the
> machine, Fatih Hoca hands back the single best model to run it.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** Picking *which* model runs a given task is a hard,
many-axis judgement call: a local 9B that's already in VRAM vs. a stronger cloud
model that costs money and quota; an easy task that shouldn't burn an expensive
model vs. a hard one that needs the best; free quota that resets at midnight vs.
a swap budget that forbids thrashing the GPU. Fatih Hoca removes that judgement
from every caller. Hand it the task and it returns one ranked answer, so the
caller never has to reason about catalogs, capabilities, rate limits, or GPU
pressure.

**What it really does.** It owns all model *knowledge* and turns it into one
decision. It builds the catalog (YAML + a scan of local GGUF files + live cloud
discovery), enriches each model with benchmark and capability scores, and runs
a three-layer pipeline: hard eligibility gates, a weighted capability/speed/cost
composite, and a utilization adjustment that balances "don't waste a strong
model on an easy task" against "burn free quota before it expires." The output
is a `Pick` — the chosen model plus a timing estimate — or a clean refusal when
nothing fits.

**It does NOT** make LLM or HTTP calls (the host's caller does that and reports
outcomes back), manage the GPU or load/unload models (the host's model manager
does, after a `Pick`), or persist its own picks (it returns a pure value; the
host writes telemetry). It only *decides*.

## Public API

The package keeps process-wide singletons. The host calls `init()` once at boot,
then `select()` per task:

```python
import fatih_hoca

# Boot — builds catalog, runs cloud discovery, wires the system-state source.
names: list[str] = fatih_hoca.init(
    models_dir=".models",          # scan for local GGUF files
    catalog_path="models.yaml",    # YAML-declared cloud + local models
    nerd_herd=system_state_source, # provides .snapshot() / .recent_swap_count()
    api_keys={"groq": "..."},      # runs per-provider auth probe + discovery
)

# Per task — the hot path.
pick = fatih_hoca.select(
    task="coder", agent_type="coder", difficulty=7,
    needs_function_calling=True, estimated_output_tokens=4000,
    call_category="main_work", urgency=0.5,
    remaining_budget_usd=2.50,
)
# -> Pick(model, min_time_seconds, estimated_load_seconds, score, top_summary)
#    or None  (no eligible model / pressure too high / swap budget spent)
#    or SelectionFailure(reason, detail)  (e.g. nothing fits the budget)

# Continuation gate — can a task keep running on the model it already holds?
ok: bool = fatih_hoca.is_servable(model=pick.model, reqs=reqs)   # -> bool

# Build requirements from a task dict (async; reads agent profiles + context).
reqs = await fatih_hoca.requirements_for(task, task_ctx, agent_name="coder")
# -> ModelRequirements

models: list[fatih_hoca.ModelInfo] = fatih_hoca.all_models()
```

Top-level exports (`__all__`): functions `init`, `select`, `is_servable`,
`all_models`, `requirements_for`, `mid_task_urgency`; dataclasses / types `Pick`,
`Failure`, `SelectionFailure`, `ModelInfo`, `ModelRequirements`, `ScoredModel`,
`Cap`; the constants `AGENT_REQUIREMENTS`, `CAPABILITY_TO_TASK`,
`ALL_CAPABILITIES`, `TASK_PROFILES`; and `discovery_results` (populated by
`init()` so the boot caller can wire cloud capacity tracking).

`select(**kwargs)` forwards to `Selector.select`. The keywords callers actually
pass:

| keyword | meaning |
|---|---|
| `task` / `agent_type` | route to a task profile (`coder`, `planner`, `analyst`, …) |
| `difficulty` | 1–10; sets the capability bar (`cap_needed_for_difficulty`) |
| `needs_function_calling` / `needs_vision` / `needs_json_mode` / `needs_thinking` | hard capability gates |
| `estimated_input_tokens` / `estimated_output_tokens` / `min_context_length` | sizing → per-call / context gates + timing estimate |
| `call_category` | `"main_work"` (may swap models) vs `"overhead"` (sticky to loaded) |
| `urgency` | 0–1; lowers the pool-pressure admission threshold |
| `local_only` / `prefer_local` / `prefer_speed` / `prefer_quality` | routing preferences |
| `remaining_budget_usd` / `max_cost` | budget gate; `0.0` admits only free models, `None` disables |
| `failures` / `exclude_models` | exclude models that already failed this task |
| `diag_out` | optional dict filled with the empty-pool stage + filter histogram |

## Architecture

Three layers, best-first output:

```
select()  → builds ModelRequirements, pulls a system snapshot
   │
   ▼ Layer 1 — eligibility (hard gates)
   │   demoted · dead-id · provider-dead · local_only · no-api-key ·
   │   context · function-calling · json · vision · cost · per-call TPM ·
   │   circuit-breaker · daily-exhausted · rpm-cooldown · no-VRAM
   │   + budget filter (per-call $ cap)
   │
   ▼ Layer 2 — weighted composite (capability · speed · cost · failure penalty)
   │
   ▼ Layer 3 — utilization adjustment
   │   composite *= 1 + K · scarcity · fit_dampener      (K = UTILIZATION_K)
   │   pool-pressure gate: drop candidates below the urgency-derived threshold
   │   swap-budget gate: if best needs a GPU swap and budget is spent,
   │                     fall back to a loaded/cloud model or refuse
   ▼
Pick(model, min_time_seconds, …)   |   None   |   SelectionFailure
```

The utilization layer is the heart of the package: selection is a balance
problem with two faces — "don't waste a strong model on an easy task" and "burn
free quota before it expires" — both expressed by one signed `scarcity` scalar
per pool (local / time-bucketed / per-call). The full design record, including
scarcity semantics per pool and the dampener's asymmetry rationale, lives in
`docs/architecture/fatih-hoca-phase2d-equilibrium.md`.

## Key Modules

| module | role |
|---|---|
| `selector.py` | `Selector` — Layer 1 eligibility, pressure/swap gates, `select()`, `is_servable()` |
| `ranking.py` | `rank_candidates` + `_apply_utilization_layer`; `ScoredModel` |
| `pools.py` | `Pool` enum, `classify_pool`, `compute_urgency` (signed scarcity per pool), `UTILIZATION_K` |
| `capability_curve.py` | `cap_needed_for_difficulty(d)` — the capability bar per difficulty |
| `capabilities.py` | `score_model_for_task`, `Cap`, `TASK_PROFILES` — 0–10 capability fit |
| `registry.py` | `ModelRegistry`, `ModelInfo` — catalog (YAML + GGUF scan + cloud), dead/revive state |
| `requirements.py` | `ModelRequirements`, `AGENT_REQUIREMENTS`, `QuotaPlanner` |
| `requirements_builder.py` | `requirements_for` — task dict → `ModelRequirements` (profiles + classification + retry escalation) |
| `profiles.py` | per-agent task profiles |
| `grading.py` | `grading_perf_score` from recorded success rates |
| `swap_policy.py` | `can_swap` — GPU swap budget rule |
| `urgency.py` | `mid_task_urgency` — admission urgency + finish-bias (no stricter mid-task) |
| `estimates.py` | input/output token estimates feeding the per-call gates |
| `cloud/` | per-provider discovery adapters (groq, openai, anthropic, gemini, cerebras, sambanova, openrouter) + cache + alert throttle |
| `counterfactual.py` · `simulate_i2p.py` · `debug/` | dev tools: replay picks, simulate a workflow, dump pressure |
| `types.py` | `Pick`, `Failure`, `SelectionFailure`, `SwapBudget` |

## Dependencies

- **Nerd Herd** is a genuine hard runtime dependency (declared in
  `pyproject.toml`, imported at module level: `ranking.py` and `selector.py` use
  `nerd_herd.types`). Fatih Hoca queries the injected `nerd_herd` object for
  `snapshot()` (GPU/VRAM, in-flight calls, cloud capacity) and
  `recent_swap_count()` (swap-budget input). When `init()` is called without
  one, a no-op stub returns an empty snapshot, so selection still runs.
- **The host** owns everything else through lazy, inside-function imports
  (catalog enrichment, benchmark cache, registry-store persistence, retry
  constraints, in-flight truth). These are soft couplings: each is wrapped in
  `try`/lazy-import and degrades cleanly if absent. Cloud discovery and the
  alert bridge (`alert_fn`) are injected by the boot caller — Fatih Hoca never
  imports Telegram or any notifier itself.

## Gotchas

- **`select()` is pure** — it returns a `Pick` and writes nothing. The host
  persists telemetry and records the actual swap *after* a successful run. Don't
  expect side effects.
- **Three return types, three meanings.** `Pick` = run it. `None` = no eligible
  model, pressure too high, or swap budget spent — back off or escalate
  `urgency`. `SelectionFailure` = a named structural refusal (e.g. budget). A
  caller that treats `None` and `SelectionFailure` the same loses the reason.
- **`is_servable()` skips the pressure gate.** It answers "can a task *continue*
  on the model it already holds," so a loaded local survives `vram=0` (its own
  residency consumed that VRAM). Don't use it to start *new* work.
- **Profile floors the request.** `select()` raises `needs_function_calling` /
  `needs_vision` to the agent profile's value — a caller can escalate
  `False→True` but never relax `True→False`.
- **Don't re-judge a task stricter mid-flight.** Use `mid_task_urgency()` for
  mid-task re-selection; judging a running task harder than at admission can veto
  it off the band that admitted it and surface "no candidates."
- **`init()` must run first.** Before it, `select()`/`all_models()` return
  `None`/`[]` and `is_servable()` returns `False` (fail-closed, so the caller
  re-selects rather than reusing a stale pick).

## Runbook — tuning the utilization equilibrium

The scenario suite *is* the equilibrium specification — do not tune by eyeball on
a single trace.

1. Make the change (usually `pools.py` scarcity math, `pools.py::UTILIZATION_K`,
   or `ranking.py::_apply_utilization_layer`'s stickiness block).
2. Run the scenario table — hard tasks should stay saturated on capable models,
   easy tasks should not waste strong models, and diverse pools should keep free
   quota healthy:
   ```powershell
   & .\.venv\Scripts\python.exe packages\fatih_hoca\tests\sim\run_scenarios.py
   ```
3. Run the swap-storm check against the real GGUF registry — the swap rate must
   stay low across the starting configurations:
   ```powershell
   & .\.venv\Scripts\python.exe packages\fatih_hoca\tests\sim\run_swap_storm_check.py
   ```
4. Run the package tests — no unit regressions (see below).
5. Commit with a note explaining the tuning rationale.

`packages/fatih_hoca/tests/sim/README.md` has extension guidance, metric
semantics, and the swap-storm reasoning.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\fatih_hoca\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Türkçe karşılığı "Fatih Hoca"dır: her modelin güçlü yanlarını bilen ve her işe
> doğru olanı atayan kişi. Bir görev ile makinenin anlık durumunu alır, o işi
> çalıştıracak tek en iyi modeli geri verir.

### Amaç

**Ne işe yarar.** Bir görevi *hangi* modelin çalıştıracağını seçmek, çok eksenli
ve zor bir karardır: zaten VRAM'de duran yerel bir 9B mi, yoksa para ve kota
harcayan daha güçlü bir bulut modeli mi; pahalı bir modeli yakmaması gereken
kolay bir görev mi, yoksa en iyisine ihtiyaç duyan zor bir görev mi; gece yarısı
sıfırlanan ücretsiz kota mı, yoksa GPU'yu yıpratmaya izin vermeyen bir swap
bütçesi mi. Fatih Hoca bu kararı her çağıran taraftan alır. Görevi verirsiniz,
size sıralanmış tek bir cevap döner; böylece çağıran taraf katalogları,
yetenekleri, rate limit'leri veya GPU baskısını hiç düşünmek zorunda kalmaz.

**Gerçekte ne yapar.** Tüm model *bilgisine* sahiptir ve bunu tek bir karara
dönüştürür. Kataloğu kurar (YAML + yerel GGUF dosyalarının taranması + canlı
bulut keşfi), her modeli benchmark ve yetenek skorlarıyla zenginleştirir ve üç
katmanlı bir hat çalıştırır: sert uygunluk kapıları, ağırlıklı bir
yetenek/hız/maliyet bileşeni ve "kolay bir göreve güçlü bir modeli boşa harcama"
ile "ücretsiz kotayı bitmeden yak" arasını dengeleyen bir kullanım ayarı.
Çıktısı bir `Pick`'tir — seçilen model artı bir zaman tahmini — ya da hiçbir şey
uymuyorsa temiz bir ret.

**Yapmadıkları**: LLM veya HTTP çağrısı yapmaz (onu host'un çağırıcısı yapar ve
sonuçları geri bildirir), GPU yönetmez veya model yükleyip boşaltmaz (onu bir
`Pick`'ten sonra host'un model yöneticisi yapar) ve kendi seçimlerini
kalıcılaştırmaz (saf bir değer döndürür; telemetriyi host yazar). O yalnızca
*karar verir*.

### Genel API

Paket, süreç çapında singleton'lar tutar. Host, boot'ta bir kez `init()` çağırır,
sonra görev başına `select()`:

```python
import fatih_hoca

# Boot — kataloğu kurar, bulut keşfini çalıştırır, sistem-durumu kaynağını bağlar.
names: list[str] = fatih_hoca.init(
    models_dir=".models",          # yerel GGUF dosyalarını tara
    catalog_path="models.yaml",    # YAML ile tanımlı bulut + yerel modeller
    nerd_herd=system_state_source, # .snapshot() / .recent_swap_count() sağlar
    api_keys={"groq": "..."},      # sağlayıcı başına auth probe + keşif çalıştırır
)

# Görev başına — sıcak yol.
pick = fatih_hoca.select(
    task="coder", agent_type="coder", difficulty=7,
    needs_function_calling=True, estimated_output_tokens=4000,
    call_category="main_work", urgency=0.5,
    remaining_budget_usd=2.50,
)
# -> Pick(model, min_time_seconds, estimated_load_seconds, score, top_summary)
#    ya da None  (uygun model yok / baskı çok yüksek / swap bütçesi bitti)
#    ya da SelectionFailure(reason, detail)  (örn. bütçeye hiçbir şey sığmıyor)

# Devam kapısı — bir görev zaten elindeki modelde çalışmaya devam edebilir mi?
ok: bool = fatih_hoca.is_servable(model=pick.model, reqs=reqs)   # -> bool

# Bir görev dict'inden gereksinim kur (async; ajan profillerini + bağlamı okur).
reqs = await fatih_hoca.requirements_for(task, task_ctx, agent_name="coder")
# -> ModelRequirements

models: list[fatih_hoca.ModelInfo] = fatih_hoca.all_models()
```

Üst düzey export'lar (`__all__`): fonksiyonlar `init`, `select`, `is_servable`,
`all_models`, `requirements_for`, `mid_task_urgency`; dataclass / tipler `Pick`,
`Failure`, `SelectionFailure`, `ModelInfo`, `ModelRequirements`, `ScoredModel`,
`Cap`; sabitler `AGENT_REQUIREMENTS`, `CAPABILITY_TO_TASK`, `ALL_CAPABILITIES`,
`TASK_PROFILES`; ve `discovery_results` (`init()` tarafından doldurulur, böylece
boot çağırıcısı bulut kapasite takibini bağlayabilir).

`select(**kwargs)`, `Selector.select`'e iletir. Çağıranların gerçekte geçtiği
anahtarlar:

| anahtar | anlamı |
|---|---|
| `task` / `agent_type` | bir görev profiline yönlendir (`coder`, `planner`, `analyst`, …) |
| `difficulty` | 1–10; yetenek barını belirler (`cap_needed_for_difficulty`) |
| `needs_function_calling` / `needs_vision` / `needs_json_mode` / `needs_thinking` | sert yetenek kapıları |
| `estimated_input_tokens` / `estimated_output_tokens` / `min_context_length` | boyutlama → per-call / bağlam kapıları + zaman tahmini |
| `call_category` | `"main_work"` (swap yapabilir) vs `"overhead"` (yüklü modele yapışkan) |
| `urgency` | 0–1; havuz-baskısı kabul eşiğini düşürür |
| `local_only` / `prefer_local` / `prefer_speed` / `prefer_quality` | yönlendirme tercihleri |
| `remaining_budget_usd` / `max_cost` | bütçe kapısı; `0.0` yalnızca ücretsiz modelleri alır, `None` devre dışı bırakır |
| `failures` / `exclude_models` | bu görevde zaten başarısız olan modelleri dışla |
| `diag_out` | boş-havuz aşaması + filtre histogramıyla doldurulan opsiyonel dict |

### Mimari

Üç katman, en-iyi-önce çıktı:

```
select()  → ModelRequirements kurar, bir sistem anlık görüntüsü çeker
   │
   ▼ Katman 1 — uygunluk (sert kapılar)
   │   demoted · ölü-id · sağlayıcı-ölü · local_only · api-key-yok ·
   │   bağlam · function-calling · json · vision · maliyet · per-call TPM ·
   │   devre kesici · günlük-tükeniş · rpm-cooldown · VRAM-yok
   │   + bütçe filtresi (per-call $ tavanı)
   │
   ▼ Katman 2 — ağırlıklı bileşen (yetenek · hız · maliyet · hata cezası)
   │
   ▼ Katman 3 — kullanım ayarı
   │   composite *= 1 + K · scarcity · fit_dampener      (K = UTILIZATION_K)
   │   havuz-baskısı kapısı: urgency'den türeyen eşiğin altındaki adayları düşür
   │   swap-bütçesi kapısı: en iyi aday GPU swap gerektiriyor ve bütçe bittiyse,
   │                        yüklü/bulut bir modele geri düş ya da reddet
   ▼
Pick(model, min_time_seconds, …)   |   None   |   SelectionFailure
```

Kullanım katmanı paketin kalbidir: seçim, iki yüzü olan bir denge sorunudur —
"kolay bir göreve güçlü modeli boşa harcama" ve "ücretsiz kotayı bitmeden yak" —
ve her ikisi de havuz başına (yerel / zaman-kovalı / per-call) tek bir işaretli
`scarcity` skaları ile ifade edilir. Havuz başına scarcity anlamları ve
dampener'ın asimetri gerekçesi dahil tam tasarım kaydı
`docs/architecture/fatih-hoca-phase2d-equilibrium.md` dosyasındadır.

### Ana Modüller

| modül | rolü |
|---|---|
| `selector.py` | `Selector` — Katman 1 uygunluk, baskı/swap kapıları, `select()`, `is_servable()` |
| `ranking.py` | `rank_candidates` + `_apply_utilization_layer`; `ScoredModel` |
| `pools.py` | `Pool` enum, `classify_pool`, `compute_urgency` (havuz başına işaretli scarcity), `UTILIZATION_K` |
| `capability_curve.py` | `cap_needed_for_difficulty(d)` — zorluk başına yetenek barı |
| `capabilities.py` | `score_model_for_task`, `Cap`, `TASK_PROFILES` — 0–10 yetenek uyumu |
| `registry.py` | `ModelRegistry`, `ModelInfo` — katalog (YAML + GGUF tarama + bulut), ölü/diriltme durumu |
| `requirements.py` | `ModelRequirements`, `AGENT_REQUIREMENTS`, `QuotaPlanner` |
| `requirements_builder.py` | `requirements_for` — görev dict'i → `ModelRequirements` (profil + sınıflandırma + retry tırmanışı) |
| `profiles.py` | ajan başına görev profilleri |
| `grading.py` | kayıtlı başarı oranlarından `grading_perf_score` |
| `swap_policy.py` | `can_swap` — GPU swap bütçe kuralı |
| `urgency.py` | `mid_task_urgency` — kabul urgency'si + bitirme-yanlılığı (görev içinde daha sıkı değil) |
| `estimates.py` | per-call kapılarını besleyen girdi/çıktı token tahminleri |
| `cloud/` | sağlayıcı başına keşif adaptörleri (groq, openai, anthropic, gemini, cerebras, sambanova, openrouter) + cache + uyarı kısması |
| `counterfactual.py` · `simulate_i2p.py` · `debug/` | geliştirici araçları: seçimleri yeniden oynat, iş akışı simüle et, baskı dök |
| `types.py` | `Pick`, `Failure`, `SelectionFailure`, `SwapBudget` |

### Bağımlılıklar

- **Nerd Herd** gerçek bir sert çalışma-zamanı bağımlılığıdır (`pyproject.toml`'da
  tanımlı, modül düzeyinde import edilir: `ranking.py` ve `selector.py`
  `nerd_herd.types` kullanır). Fatih Hoca, enjekte edilen `nerd_herd` nesnesine
  `snapshot()` (GPU/VRAM, uçuştaki çağrılar, bulut kapasitesi) ve
  `recent_swap_count()` (swap-bütçe girdisi) sorar. `init()` bunsuz çağrılırsa,
  boş bir anlık görüntü döndüren no-op bir stub kullanılır; seçim yine çalışır.
- **Host**, geri kalan her şeye lazy, fonksiyon-içi import'lar üzerinden sahiptir
  (katalog zenginleştirme, benchmark cache, registry-store kalıcılığı, retry
  kısıtları, uçuştaki gerçek durum). Bunlar yumuşak bağlardır: her biri
  `try`/lazy-import ile sarılıdır ve yoksa temiz şekilde bozulur. Bulut keşfi ve
  uyarı köprüsü (`alert_fn`) boot çağırıcısı tarafından enjekte edilir — Fatih
  Hoca asla Telegram veya herhangi bir bildirici import etmez.

### Tuzaklar

- **`select()` saftır** — bir `Pick` döner ve hiçbir şey yazmaz. Host
  telemetriyi kalıcılaştırır ve gerçek swap'ı başarılı bir çalışmadan *sonra*
  kaydeder. Yan etki beklemeyin.
- **Üç dönüş tipi, üç anlam.** `Pick` = çalıştır. `None` = uygun model yok, baskı
  çok yüksek ya da swap bütçesi bitti — geri çekil veya `urgency`'yi tırmandır.
  `SelectionFailure` = adlandırılmış yapısal bir ret (örn. bütçe). `None` ile
  `SelectionFailure`'ı aynı sayan bir çağıran, nedeni kaybeder.
- **`is_servable()` baskı kapısını atlar.** "Bir görev zaten elindeki modelde
  *devam* edebilir mi" sorusunu yanıtlar; bu yüzden yüklü bir yerel model
  `vram=0`'a dayanır (o VRAM'i kendi varlığı tüketmiştir). Onu *yeni* iş
  başlatmak için kullanmayın.
- **Profil isteği tabanlar.** `select()`, `needs_function_calling` /
  `needs_vision`'ı ajan profilinin değerine yükseltir — çağıran `False→True`
  tırmandırabilir ama asla `True→False` gevşetemez.
- **Bir görevi görev içinde daha sıkı yargılamayın.** Görev-içi yeniden seçim
  için `mid_task_urgency()` kullanın; çalışan bir görevi kabul anındakinden daha
  sert yargılamak, onu kendisini kabul eden bandın dışına veto edebilir ve "aday
  yok" hatasını yüzeye çıkarır.
- **`init()` önce çalışmalı.** Ondan önce `select()`/`all_models()`
  `None`/`[]` döner ve `is_servable()` `False` döner (fail-closed, böylece
  çağıran bayat bir seçimi yeniden kullanmak yerine yeniden seçim yapar).

### Çalıştırma Kılavuzu — kullanım dengesini ayarlama

Senaryo takımı dengenin *spesifikasyonudur* — tek bir izlemeye bakıp göz kararı
ayarlama yapmayın.

1. Değişikliği yap (genelde `pools.py` scarcity matematiği,
   `pools.py::UTILIZATION_K` ya da `ranking.py::_apply_utilization_layer`'ın
   yapışkanlık bloğu).
2. Senaryo tablosunu çalıştır — zor görevler yetenekli modellerde doygun
   kalmalı, kolay görevler güçlü modelleri boşa harcamamalı ve çeşitli havuzlar
   ücretsiz kotayı sağlıklı tutmalı:
   ```powershell
   & .\.venv\Scripts\python.exe packages\fatih_hoca\tests\sim\run_scenarios.py
   ```
3. Gerçek GGUF registry'sine karşı swap-storm kontrolünü çalıştır — swap oranı
   başlangıç konfigürasyonları boyunca düşük kalmalı:
   ```powershell
   & .\.venv\Scripts\python.exe packages\fatih_hoca\tests\sim\run_swap_storm_check.py
   ```
4. Paket testlerini çalıştır — birim regresyonu olmamalı (aşağıya bakın).
5. Ayarlama gerekçesini açıklayan bir notla commit et.

`packages/fatih_hoca/tests/sim/README.md`'de genişletme rehberi, metrik
anlamları ve swap-storm gerekçesi bulunur.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\fatih_hoca\ -q
```
