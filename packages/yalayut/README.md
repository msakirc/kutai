# Yalayut — Vetted external-skill catalog & matcher

> The host's librarian for outside knowledge: skills, web APIs, and MCP servers
> pulled in from the wider ecosystem, vetted for safety, and handed back —
> ranked — when a task needs one. Yalayut owns the shelf; the caller decides
> what to read off it.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** A task often needs capability the host doesn't ship —
a niche web API, a community skill, an MCP tool server. Pulling those in by hand
is slow and unsafe: every source names things differently, trust varies wildly,
and a stray shell command or prompt-injection string can ride along. Yalayut
removes that friction. It discovers candidate artifacts from many sources, runs
each through automated safety checks, assigns a trust **tier**, stores the safe
ones in a searchable index, and — at task time — returns the best matches ranked
by semantic similarity. Adding a new upstream is writing one source adapter, not
re-plumbing the catalog.

**What it really does.** For every artifact (skill / api / mcp) it keeps one row
in `yalayut_index` carrying a 768-dim embedding, a vetting tier (T0 best … T3
worst), and an exposure ceiling. Discovery (`discover → fetch → synthesize →
vet → tier-classify → embed → store`) feeds the index from trusted cron sources
and need-driven on-demand sources. The hot read, `query(task_ctx)`, embeds the
task text and ranks the enabled index by cosine similarity. Separately it can
**execute** the safe-by-construction artifact kinds itself: `run_recipe` runs an
allowlisted shell recipe with no shell, and `dispatch_tool` routes a namespaced
api/mcp tool-call to its plugin. A demand-signal subsystem learns what was
missing and drives autonomous re-discovery.

**It does NOT decide exposure or render prompts** — `query` returns ranked
`Artifact` dataclasses; the caller picks which to inject/expose and how to phrase
them. It does **not** select models, run the agent loop, or talk to Telegram
directly (a founder-ops module backs `/yalayut`, but the bot owns the UI). It
never auto-adds an untrusted source or auto-enables a T2/T3 artifact — promotion
is always a founder decision. And it never starts an MCP server at boot — servers
launch lazily only when a matched task needs one.

## Public API

The contract is two hot entry points plus a small set of mechanical-executor
bodies. Everything is async and re-exported from the top-level package.

```python
from yalayut import query, dispatch_tool, run_recipe, Artifact

# 1. Hot read — the only call on the task-dispatch path.
#    task_ctx is a raw KutAI task dict; embeds its text, ranks the index.
artifacts = await query(task_ctx, top_k=12)        # -> list[Artifact]
top = artifacts[0]                                  # .name .artifact_type .vet_tier
                                                    # .score .exposure_class .env_status

# 2. Run an allowlisted shell recipe (yalayut_recipe mechanical executor body).
#    args carry intersect-bound inputs incl. an optional workspace_path.
result = await run_recipe(recipe_id, args)          # -> dict: ok / steps / artifacts_*

# 3. Route a namespaced api_/mcp_ tool-call to its plugin executor.
#    registry is the caller's per-task tool-spec map.
out = await dispatch_tool(tool_name, args, registry)  # -> dict: ok / response / error
```

`Artifact` (returned by `query`) is the ranked-result dataclass: `artifact_id,
name, name_original, artifact_type, kind, vet_tier, score, exposure_class,
applies_to, mechanizable, body_excerpt, payload, source, owner, env_status,
intent_keywords, inputs_schema`.

### Discovery & autonomy entry points

These are the bodies behind the host's mechanical executors / post-hooks. All are
re-exported from `yalayut`:

| function | role |
|---|---|
| `daily_discovery() -> dict` | pull every due trusted cron source end-to-end; index the survivors |
| `on_demand_discovery(demand: dict) -> dict` | need-driven fetch against untrusted sources for one demand signal |
| `source_scout_scan() -> dict` | propose candidate sources for founder review (never auto-adds) |
| `run_demand_drain() -> dict` | derive `repeat_pattern` + run on-demand discovery for every high-confidence pattern |
| `record_demand_signal(*, source_step_pattern, intent_keywords, signal_type, confidence=0.3) -> int` | record one demand signal; returns row id, or `-1` if deduped by cooldown |
| `observe_and_propose() -> int` | scan vetting audit data; write founder policy proposals; returns count |
| `capture_hint(task: dict, outcome: dict) -> None` | post-hook: capture a successful 2+-iteration task as an `internal_hint` artifact |

`record_demand_signal`'s `signal_type` must be one of seven: `planning_miss`,
`step_entry_miss`, `tool_call`, `founder` (proactive); `hint_miss`, `dlq`,
`repeat_pattern` (reactive).

## Architecture

Discovery is a per-source pipeline; the read path is a single cosine scan:

```
sources ─► discover ─► fetch ─► synthesize ─► vet (auto_checks)
                                                 │
                              trust caps (source/owner) ─┐
                                                         ▼
                                              tier-classify (T0..T3)
                                                         │
                                       embed (multilingual-e5-base, 768d)
                                                         ▼
                                          store → yalayut_index (enabled?)
                                                         │
   task_ctx ─► query() ─► embed text ─► cosine vs enabled rows ─► top_k Artifacts
                                                         │
                exposure ceiling per tier ──────────────┘ (caller refines)

execution: run_recipe (shell, allowlisted, no-shell) · dispatch_tool (api | mcp)
autonomy:  demand signals ─► stack confidence ─► drain ─► on_demand_discovery
```

Tiering: `trust_cap = min(source_max, owner_max)` (a trusted owner *elevates* a
sketchy source); `final = max(trust_cap, worst_check)` (auto-checks always cap,
never elevate). T0/T1 auto-enable; T2 is quarantined-until-founder-promotes; T3
stays quarantined.

## Key Modules

| module | role |
|---|---|
| `__init__.py` | the public async surface (`query`, `dispatch_tool`, `run_recipe`, discovery/autonomy entry points) |
| `_query_engine.py` | the hot read — cosine rank of enabled index vs task embedding (`query_db` is the I/O-free testable core) |
| `index.py` | `yalayut_index` store/read; float32 BLOB embeddings; tier→exposure default; enable policy |
| `schema.py` | all yalayut tables (`ensure_yalayut_schema`), idempotent on every boot |
| `contracts.py` | in-process dataclasses + plugin protocols (`Manifest`, `Artifact`, `TaskContext`, `IndexRow`, `*Plugin`) |
| `manifest.py` | manifest parse + validation |
| `trust.py` | source/owner trust → tier ceiling |
| `tier_classifier.py` | combine trust caps + auto-check caps → final tier + audit |
| `vetting/auto_checks.py` · `vetting/policy.py` | safety checks + DB-backed allowlists (shell bins, domains, injection regexes) |
| `executor.py` | `run_recipe` — allowlisted, no-shell, pre-flight-checked shell-recipe runner |
| `shell_safety.py` | tokenize + allowlist + Windows-incompat gate for every command |
| `plugins/api.py` · `plugins/mcp.py` · `plugins/skill.py` | per-artifact-type binding + tool execution |
| `mcp_manager.py` | MCP stdio process lifecycle — lazy start, JSON-RPC, health probe, idle sweep |
| `secrets.py` | fernet-encrypted auth store + `env_status` lifecycle |
| `discovery/` | source adapters, `cron`, `on_demand`, `source_scout`, `demand`, `demand_drain`, `synthesize` |
| `capture.py` | `internal_hint` auto-capture post-hook |
| `policy_observer.py` | propose allowlist additions from audit data (never mutates policy) |
| `admin.py` | founder-ops bodies behind `/yalayut` (vet queue, sources, policy, auth, MCP, stats) |
| `seed/` | seed sources, owners, policy, and bundled manifests |

## Dependencies

Yalayut's one genuine hard runtime coupling is to the **host application's
shared infrastructure** under `src/` — it is not standalone:

- **Embeddings** — `query`, discovery, and `capture_hint` lazily import
  `src.memory.embeddings` (multilingual-e5-base, 768d). The index BLOBs are only
  comparable to query vectors from this exact model; mixing models silently
  zeroes every cosine score.
- **DB** — every persistent path uses `src.infra.db.get_db()` (the shared
  aiosqlite handle); `src.infra.times` for DB-format timestamps;
  `src.infra.logging_config` for logging.
- **API execution** — `plugins/api.py` delegates HTTP to `src/tools/free_apis.py`
  (it already handles auth-header / apikey substitution and truncation).

The consuming **matcher/exposer** decides per-task exposure off `query`'s output;
the **host's mechanical-executor layer** invokes `run_recipe` / `dispatch_tool` /
the discovery bodies; the **Telegram bot** drives `admin.py`. Those are seams to
the rest of the host, not in-package dependencies — Yalayut does not import them.

Third-party (`pyproject`): `python-frontmatter`, `PyYAML`, `httpx`, `numpy`.
MCP-server transport, secrets, and free-API execution add `cryptography`
(fernet) and the host's HTTP tooling at use time.

Env: `YALAYUT_SECRET_KEY` (fernet key for the encrypted auth store — required
only when storing secrets).

## Gotchas

- **Embedding model is load-bearing.** Index BLOBs and query vectors must both
  come from multilingual-e5-base. A different model gives `_cosine` mismatched
  lengths → `0.0` for every row → the catalog silently returns nothing.
- **`TaskContext.from_task` parses a JSON-string `context`.** DB-sourced tasks
  carry `context` as a JSON string; the dataclass parses it before `.get()`.
  Hand-building a task dict, leave `context` as a real dict or a JSON string —
  not some other shape — or the row degrades to empty text and `query` returns
  `[]`.
- **`query` returns ranked candidates, not decisions.** It never filters by tier
  or exposure or `env_status` beyond ranking — the caller must respect
  `vet_tier`, `exposure_class`, and `env_status` itself. Defence-in-depth exists
  in the plugins, but the read path trusts the caller.
- **Enable policy is tier-driven.** Only T0/T1 land `enabled=1`. A freshly
  discovered T2/T3 artifact exists in the index but is invisible to `query`
  until a founder promotes it via `admin.approve_artifact`.
- **Recipes are no-shell and allowlisted.** `run_recipe` tokenizes every command,
  rejects any first-token binary not in `yalayut_policy.shell_allowlist`, and
  fails the whole recipe pre-flight if any step is Windows-incompatible —
  nothing runs until every step passes. `mechanizable` must be set or it refuses.
- **MCP servers are lazy and idle-swept.** None start at boot (`no_auto_connect`).
  v1 supports the `stdio` transport only; a server is killed after its
  `idle_timeout_s`, and disabled in the index after repeated health-probe fails.
- **Discovery never auto-trusts.** `source_scout_scan` only writes `pending`
  candidate rows; `observe_and_propose` only writes `pending` policy proposals.
  Both wait for a founder decision through `admin.py`.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\yalayut\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Host uygulamanın dış dünya için kütüphanecisi: daha geniş ekosistemden çekilen
> beceriler (skill), web API'leri ve MCP sunucuları — güvenlik için elenir,
> indekslenir ve bir görev birine ihtiyaç duyduğunda sıralanmış olarak geri
> verilir. Rafı Yalayut tutar; raftan ne okunacağına çağıran taraf karar verir.

### Amaç

**Ne işe yarar.** Bir görev çoğu zaman host'un içinde gelmeyen bir yeteneğe
ihtiyaç duyar — niş bir web API'si, topluluk becerisi, MCP araç sunucusu. Bunları
elle çekmek hem yavaş hem tehlikelidir: her kaynak şeyleri farklı adlandırır,
güven düzeyi çılgınca değişir ve başıboş bir shell komutu ya da prompt-injection
metni yanında sızabilir. Yalayut bu sürtünmeyi ortadan kaldırır. Birçok kaynaktan
aday artifaktları keşfeder, her birini otomatik güvenlik denetimlerinden geçirir,
bir güven **kademesi** (tier) atar, güvenli olanları aranabilir bir indekste
saklar ve görev anında en iyi eşleşmeleri anlamsal benzerliğe göre sıralayıp
döndürür. Yeni bir üst kaynak eklemek, katalogu baştan kurmak değil, tek bir
kaynak adaptörü yazmaktır.

**Gerçekte ne yapar.** Her artifakt (skill / api / mcp) için `yalayut_index`'te
tek bir satır tutar; bu satır 768 boyutlu bir embedding, bir eleme kademesi (T0
en iyi … T3 en kötü) ve bir maruz-bırakma (exposure) tavanı taşır. Keşif zinciri
(`keşfet → çek → sentezle → ele → kademe-belirle → embed → sakla`) indeksi güvenli
cron kaynaklarından ve ihtiyaç-güdümlü on-demand kaynaklardan besler. Sıcak okuma
yolu `query(task_ctx)`, görev metnini embed eder ve etkin indeksi kosinüs
benzerliğine göre sıralar. Ayrıca güvenli-tasarımlı artifakt türlerini kendisi
**çalıştırabilir**: `run_recipe` allowlist'li bir shell tarifini shell olmadan
çalıştırır, `dispatch_tool` namespace'li bir api/mcp araç çağrısını eklentisine
yönlendirir. Bir talep-sinyali (demand) altsistemi neyin eksik olduğunu öğrenir ve
otonom yeniden-keşfi sürer.

**Maruz bırakmaya karar vermez, prompt render etmez** — `query`, sıralanmış
`Artifact` dataclass'ları döndürür; hangisinin enjekte/maruz bırakılacağına ve
nasıl ifade edileceğine çağıran taraf karar verir. Model **seçmez**, ajan döngüsünü
**çalıştırmaz** ve doğrudan Telegram ile **konuşmaz** (`/yalayut`'u besleyen bir
founder-ops modülü vardır, ama arayüzü bot tutar). Güvensiz bir kaynağı asla
otomatik eklemez, bir T2/T3 artifaktını asla otomatik etkinleştirmez — terfi her
zaman founder kararıdır. Ve hiçbir MCP sunucusunu açılışta başlatmaz — sunucular
yalnızca eşleşen bir görev ihtiyaç duyduğunda tembel (lazy) başlar.

### Genel API

Sözleşme, iki sıcak giriş noktası ve küçük bir mekanik-yürütücü gövdesi kümesidir.
Hepsi async'tir ve üst düzey paketten yeniden export edilir.

```python
from yalayut import query, dispatch_tool, run_recipe, Artifact

# 1. Sıcak okuma — görev-sevk yolundaki tek çağrı.
#    task_ctx ham bir KutAI görev dict'i; metnini embed eder, indeksi sıralar.
artifacts = await query(task_ctx, top_k=12)        # -> list[Artifact]
top = artifacts[0]                                  # .name .artifact_type .vet_tier
                                                    # .score .exposure_class .env_status

# 2. Allowlist'li bir shell tarifini çalıştır (yalayut_recipe yürütücü gövdesi).
#    args, intersect ile bağlanmış girdileri taşır (opsiyonel workspace_path dâhil).
result = await run_recipe(recipe_id, args)          # -> dict: ok / steps / artifacts_*

# 3. Namespace'li bir api_/mcp_ araç çağrısını eklentisine yönlendir.
#    registry, çağıranın görev başına araç-spec haritasıdır.
out = await dispatch_tool(tool_name, args, registry)  # -> dict: ok / response / error
```

`Artifact` (`query`'nin döndürdüğü) sıralanmış-sonuç dataclass'ıdır:
`artifact_id, name, name_original, artifact_type, kind, vet_tier, score,
exposure_class, applies_to, mechanizable, body_excerpt, payload, source, owner,
env_status, intent_keywords, inputs_schema`.

#### Keşif & otonomi giriş noktaları

Bunlar host'un mekanik yürütücülerinin / post-hook'larının gövdeleridir. Hepsi
`yalayut`'tan yeniden export edilir:

| fonksiyon | rolü |
|---|---|
| `daily_discovery() -> dict` | vadesi gelen her güvenli cron kaynağını uçtan uca çek; hayatta kalanları indeksle |
| `on_demand_discovery(demand: dict) -> dict` | bir talep sinyali için güvensiz kaynaklara ihtiyaç-güdümlü çekim |
| `source_scout_scan() -> dict` | founder incelemesi için aday kaynaklar öner (asla otomatik eklemez) |
| `run_demand_drain() -> dict` | `repeat_pattern` türet + her yüksek-güvenli desen için on-demand keşif çalıştır |
| `record_demand_signal(*, source_step_pattern, intent_keywords, signal_type, confidence=0.3) -> int` | bir talep sinyali kaydet; satır id döner, cooldown ile dedup edilirse `-1` |
| `observe_and_propose() -> int` | eleme denetim verisini tara; founder politika önerileri yaz; sayı döner |
| `capture_hint(task: dict, outcome: dict) -> None` | post-hook: başarılı, 2+ yinelemeli görevi `internal_hint` artifaktı olarak yakala |

`record_demand_signal`'in `signal_type`'ı yedi değerden biri olmalı:
`planning_miss`, `step_entry_miss`, `tool_call`, `founder` (proaktif); `hint_miss`,
`dlq`, `repeat_pattern` (reaktif).

### Mimari

Keşif kaynak-başına bir zincirdir; okuma yolu tek bir kosinüs taramasıdır:

```
kaynaklar ─► keşfet ─► çek ─► sentezle ─► ele (auto_checks)
                                             │
                          güven tavanı (kaynak/owner) ─┐
                                                       ▼
                                          kademe-belirle (T0..T3)
                                                       │
                                  embed (multilingual-e5-base, 768d)
                                                       ▼
                                       sakla → yalayut_index (etkin mi?)
                                                       │
   task_ctx ─► query() ─► metni embed ─► etkin satırlara kosinüs ─► top_k Artifact
                                                       │
              kademe başına maruz-bırakma tavanı ──────┘ (çağıran inceltir)

yürütme: run_recipe (shell, allowlist'li, shell'siz) · dispatch_tool (api | mcp)
otonomi: talep sinyalleri ─► güven yığ ─► drain ─► on_demand_discovery
```

Kademeleme: `trust_cap = min(source_max, owner_max)` (güvenilir bir owner,
şüpheli bir kaynağı *yükseltir*); `final = max(trust_cap, en_kötü_check)`
(otomatik denetimler her zaman tavanlar, asla yükseltmez). T0/T1 otomatik
etkinleşir; T2 founder-terfisine-kadar karantinadadır; T3 karantinada kalır.

### Ana Modüller

| modül | rolü |
|---|---|
| `__init__.py` | genel async yüzey (`query`, `dispatch_tool`, `run_recipe`, keşif/otonomi giriş noktaları) |
| `_query_engine.py` | sıcak okuma — etkin indeksi görev embedding'ine göre kosinüs sıralar (`query_db`, I/O'suz test çekirdeği) |
| `index.py` | `yalayut_index` saklama/okuma; float32 BLOB embedding; kademe→maruz-bırakma varsayılanı; etkinleştirme politikası |
| `schema.py` | tüm yalayut tabloları (`ensure_yalayut_schema`), her açılışta idempotent |
| `contracts.py` | süreç-içi dataclass'lar + eklenti protokolleri (`Manifest`, `Artifact`, `TaskContext`, `IndexRow`, `*Plugin`) |
| `manifest.py` | manifest parse + doğrulama |
| `trust.py` | kaynak/owner güveni → kademe tavanı |
| `tier_classifier.py` | güven tavanları + otomatik-denetim tavanlarını birleştir → final kademe + audit |
| `vetting/auto_checks.py` · `vetting/policy.py` | güvenlik denetimleri + DB-tabanlı allowlist'ler (shell binary, domain, injection regex) |
| `executor.py` | `run_recipe` — allowlist'li, shell'siz, ön-uçuş denetimli shell-tarifi çalıştırıcı |
| `shell_safety.py` | her komut için tokenize + allowlist + Windows-uyumsuzluk kapısı |
| `plugins/api.py` · `plugins/mcp.py` · `plugins/skill.py` | artifakt-türü başına bağlama + araç yürütme |
| `mcp_manager.py` | MCP stdio süreç yaşam döngüsü — tembel başlatma, JSON-RPC, sağlık yoklama, boşta süpürme |
| `secrets.py` | fernet-şifreli auth deposu + `env_status` yaşam döngüsü |
| `discovery/` | kaynak adaptörleri, `cron`, `on_demand`, `source_scout`, `demand`, `demand_drain`, `synthesize` |
| `capture.py` | `internal_hint` otomatik-yakalama post-hook'u |
| `policy_observer.py` | audit verisinden allowlist eklemesi öner (politikayı asla değiştirmez) |
| `admin.py` | `/yalayut` arkasındaki founder-ops gövdeleri (eleme kuyruğu, kaynaklar, politika, auth, MCP, istatistik) |
| `seed/` | seed kaynakları, owner'lar, politika ve paketle gelen manifestler |

### Bağımlılıklar

Yalayut'un tek gerçek sert çalışma-zamanı bağı, **host uygulamanın `src/`
altındaki paylaşılan altyapısıdır** — bağımsız (standalone) değildir:

- **Embedding'ler** — `query`, keşif ve `capture_hint`, `src.memory.embeddings`'i
  tembel import eder (multilingual-e5-base, 768d). İndeks BLOB'ları yalnızca tam
  olarak bu modelden gelen sorgu vektörleriyle karşılaştırılabilir; model
  karıştırmak her kosinüs skorunu sessizce sıfırlar.
- **DB** — her kalıcı yol `src.infra.db.get_db()`'yi (paylaşılan aiosqlite
  handle'ı) kullanır; DB-format zaman damgaları için `src.infra.times`; loglama
  için `src.infra.logging_config`.
- **API yürütme** — `plugins/api.py`, HTTP'yi `src/tools/free_apis.py`'ye devreder
  (auth-header / apikey ikamesini ve kırpmayı zaten o yapar).

Tüketen **eşleştirici/maruz-bırakıcı**, görev başına maruz-bırakmayı `query`
çıktısından kararlaştırır; **host'un mekanik-yürütücü katmanı** `run_recipe` /
`dispatch_tool` / keşif gövdelerini çağırır; **Telegram botu** `admin.py`'yi
sürer. Bunlar host'un geri kalanına giden dikiş yerleridir, paket-içi bağımlılık
değil — Yalayut onları import etmez.

Üçüncü-taraf (`pyproject`): `python-frontmatter`, `PyYAML`, `httpx`, `numpy`.
MCP-sunucu transportu, secrets ve free-API yürütme, kullanım anında
`cryptography` (fernet) ile host'un HTTP araçlarını ekler.

Env: `YALAYUT_SECRET_KEY` (şifreli auth deposu için fernet anahtarı — yalnızca
secret saklarken gereklidir).

### Tuzaklar

- **Embedding modeli yük taşır.** İndeks BLOB'ları ve sorgu vektörlerinin ikisi
  de multilingual-e5-base'den gelmeli. Farklı bir model `_cosine`'a uyumsuz
  uzunluklar verir → her satır için `0.0` → katalog sessizce hiçbir şey döndürmez.
- **`TaskContext.from_task`, JSON-string bir `context` parse eder.** DB-kaynaklı
  görevler `context`'i JSON string olarak taşır; dataclass `.get()`'ten önce onu
  parse eder. Elle bir görev dict'i kurarken `context`'i gerçek bir dict ya da
  JSON string olarak bırak — başka bir şekil değil — yoksa satır boş metne
  düşer ve `query` `[]` döndürür.
- **`query` sıralanmış adaylar döndürür, karar değil.** Sıralamanın ötesinde
  kademeye, maruz-bırakmaya veya `env_status`'a göre filtrelemez — çağıran taraf
  `vet_tier`, `exposure_class` ve `env_status`'a kendisi uymalı. Eklentilerde
  derinlemesine-savunma vardır, ama okuma yolu çağırana güvenir.
- **Etkinleştirme politikası kademe-güdümlüdür.** Yalnızca T0/T1 `enabled=1` olur.
  Yeni keşfedilmiş bir T2/T3 artifaktı indekste vardır ama bir founder
  `admin.approve_artifact` ile terfi edene dek `query`'ye görünmezdir.
- **Tarifler shell'siz ve allowlist'lidir.** `run_recipe` her komutu tokenize
  eder, `yalayut_policy.shell_allowlist`'te olmayan ilk-token binary'yi reddeder
  ve herhangi bir adım Windows-uyumsuzsa tüm tarifi ön-uçuşta düşürür — her adım
  geçmeden hiçbir şey çalışmaz. `mechanizable` set edilmemişse reddeder.
- **MCP sunucuları tembeldir ve boşta süpürülür.** Açılışta hiçbiri başlamaz
  (`no_auto_connect`). v1 yalnızca `stdio` transportunu destekler; bir sunucu
  `idle_timeout_s`'inden sonra öldürülür ve tekrarlı sağlık-yoklama hatalarından
  sonra indekste devre dışı bırakılır.
- **Keşif asla otomatik güvenmez.** `source_scout_scan` yalnızca `pending` aday
  satırları yazar; `observe_and_propose` yalnızca `pending` politika önerileri
  yazar. İkisi de `admin.py` üzerinden bir founder kararını bekler.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\yalayut\ -q
```
