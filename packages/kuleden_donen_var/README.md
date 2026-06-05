# Kuleden Dönen Var — Cloud capacity tracker (KDV)

> Turkish for the air-traffic-control tower's "there's one cleared to land"
> announcement: the tower signals when a runway slot opens up. KDV is that tower
> for cloud LLM providers — it tells the host when a provider's capacity opens up
> or closes.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** Every cloud LLM provider — Groq, Gemini, Anthropic,
OpenAI, Cerebras, SambaNova, and the ones still to come — reports its rate limits
and quotas differently: different headers, different axes, different 429 formats.
KDV exists so integrating any of them is painless. It learns each provider's own
dialect, tracks every limit and quota behind one uniform interface, and absorbs
that hassle so the caller can fire a litellm call without ever reasoning about
rate limits. **Adding a new provider is a parser, not a rewrite.**

**What it really does.** For every (model, provider) it keeps a live, multi-axis
ledger — requests *and* tokens, per-minute *and* per-day, two-tiered (per-model +
per-provider-aggregate) — populated from each provider's own response headers and
429 bodies, decremented locally between responses. Over that ledger sit four
guards (rate limiter, circuit breaker, canary single-flight, daily axis). Before
each call the caller asks `pre_call(...)` and gets a clean go/no-go with a wait
estimate; afterward it reports the outcome so the ledger stays sharp. A callback
fires the instant a provider's capacity reopens.

**It does NOT** discover or pick models (the host pushes config via `register`;
the host's selector picks), make LLM/HTTP calls (the caller does that and feeds
KDV headers + outcomes), or manage local models or GPU. Auth errors are treated
as permanent, so they never trip the breaker.

## Public API

Constructed once per process by the host and driven by the host's LLM caller. The
call lifecycle is an ordered contract:

```python
from kuleden_donen_var import KuledenDonenVar, KuledenConfig

kdv = KuledenDonenVar(KuledenConfig(on_capacity_change=on_change))
kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)   # host pushes config

# Per call — ALWAYS in this order:
res = kdv.pre_call("groq/llama-8b", "groq", estimated_tokens=5000)   # -> PreCallResult
if not res.allowed:
    ...  # honor res.reason / res.wait_seconds / res.daily_exhausted; pick another model
kdv.record_attempt("groq/llama-8b", "groq", estimated_tokens=5000)   # reserve TPM + count RPM
# ...make the call...
kdv.post_call("groq/llama-8b", "groq", headers=hdrs, token_count=3200, reserved_tokens=5000)
# OR, on failure:
kdv.record_failure("groq/llama-8b", "groq", "rate_limited", error_message=body)
kdv.release_reservation("groq/llama-8b", "groq", reserved_tokens=5000)
```

`pre_call → record_attempt → (post_call | record_failure + release_reservation)`.
Skipping `record_attempt` undercounts RPM and keeps `no_data_warnings` firing.

Top-level exports (`__all__`): classes `KuledenDonenVar`, `KuledenConfig`,
`CapacityEvent`, `ModelStatus`, `ProviderStatus`, `PreCallResult`,
`RateLimitSnapshot`, `InFlightHandle`, `InFlightTracker`; the header helper
`parse_rate_limit_headers(provider, headers) -> RateLimitSnapshot | None`; and the
in-flight singleton helpers `begin_call` / `end_call` / `in_flight_count` /
`configure_in_flight_push`.

### Key methods (`KuledenDonenVar`)

| method | purpose |
|---|---|
| `register(model_id, provider, rpm, tpm, provider_aggregate_rpm=None, provider_aggregate_tpm=None)` | host pushes per-model + optional provider-aggregate limits |
| `pre_call(model_id, provider, estimated_tokens=0) -> PreCallResult` | non-blocking go/no-go: `.allowed / .reason / .wait_seconds / .daily_exhausted / .binding_provider` |
| `record_attempt(model_id, provider, headers=None, estimated_tokens=0)` | reserve TPM + count RPM **before** the POST (admission-race close) |
| `post_call(model_id, provider, headers, token_count, reserved_tokens=0)` | success: reconcile reservation to actual, apply header limits, reset breaker |
| `record_failure(model_id, provider, error_type, error_message="")` | `rate_limit` / `rate_limited` / `daily_exhausted` / `server_error` / `timeout` / `auth` |
| `release_reservation(model_id, provider, reserved_tokens)` | roll back the TPM reservation on failure |
| `recent_success_rate(model_id) -> float` · `provider_prior_rate(provider, ...) -> float \| None` | reliability signals (published as metrics) |
| `no_data_warnings(min_age_hours=24.0) -> list[dict]` | providers enabled but never attempted |
| `snapshot_state() -> dict` · `restore_state(snap)` | persist / restore across reboots |
| `restore_limits()` | gradually undo adaptive 429 reductions (host watchdog) |
| `status -> dict[str, ProviderStatus]` | full per-provider / per-model snapshot |

## Architecture

Two-tier limits (per-model + per-provider-aggregate; both must pass), daily axes
anchored to the next UTC midnight, header-derived live limits, plus four guards:

```
pre_call ─┬─ rate limiter   (RPM/TPM/RPD, two-tier, header-overridden)
          ├─ circuit breaker (per-provider; trip → cooldown; auth excluded)
          ├─ canary gate     (single-flight under uncertainty: boot/reset/post-failure)
          └─ daily axis      (RPD ticked locally each attempt → exhaustion before first 429)
record_attempt → reserve TPM + count RPM        (race close, before POST)
post_call / record_failure + release_reservation → reconcile + reliability window
                                                  ↘ on_capacity_change callback
in-flight tracker → begin_call / end_call        (+ optional metrics push)
```

## Key Modules

| module | role |
|---|---|
| `kdv.py` | `KuledenDonenVar` facade — lifecycle, canary, reservations, reliability, callback |
| `rate_limiter.py` | two-tier RPM/TPM/RPD windows, adaptive 429 reduce/restore, post-429 cooldown |
| `header_parser.py` | per-provider header parsing, `parse_429_body`, `Retry-After` |
| `circuit_breaker.py` | per-provider sliding-window failure breaker |
| `in_flight.py` | dispatched-but-unconfirmed call tracker (TTL-pruned) |
| `nerd_herd_adapter.py` | translates KDV state into Nerd Herd's types and pushes pressure + reliability |
| `cost_adapter.py` | stable re-export of the host's vendor-cost DB writer |
| `config.py` | `KuledenConfig`, `PreCallResult`, `ProviderStatus`, `ModelStatus`, `CapacityEvent` |

## Capacity events

`on_capacity_change` fires for exactly four event types:

| event | when |
|---|---|
| `capacity_restored` | headroom recovers (`prev rpm_remaining ≤ 1` → `> 5`) |
| `limit_hit` | 429 received, limits adaptively reduced |
| `circuit_breaker_tripped` | provider disabled after repeated failures |
| `circuit_breaker_reset` | provider re-enabled after a clean call |

Daily exhaustion is **not** a callback event — it surfaces via
`PreCallResult.daily_exhausted` / `reason="rpd"`.

## Gotchas

- **Bookkeeping moved**: RPM is counted in `record_attempt`, not `post_call`.
  Calling only `post_call` undercounts RPM and keeps `no_data_warnings` firing.
- **Reservation contract**: the `estimated_tokens` passed to `record_attempt` must
  come back as `reserved_tokens` to `post_call` (success) or `release_reservation`
  (failure), or TPM accounting drifts.
- **Canary single-flight**: under uncertainty (boot, day/breaker reset, any
  failure) only one call per provider is admitted; others get
  `reason="canary_in_flight"`, `wait_seconds=5.0`. Honor `reason` or you busy-loop.
- **`wait_seconds` is load-bearing**: circuit-breaker refusals return the real
  remaining cooldown (not `0.0`) so the caller actually sleeps before retry.
- **In-flight push is opt-in**: until `configure_in_flight_push()` runs,
  `begin_call`/`end_call` track in-process only and emit no metrics.
- **`auth` vs `auth_failure`**: both spellings are handled — `auth` skips the
  breaker, `auth_failure` is excluded from the reliability window. Easy to confuse.
- **Sync, single-process**: the check-and-reserve race relies on single-process
  asyncio (sync calls don't yield). The rate limiter's async `wait_*` helpers are
  not used by the facade.

## Dependencies

KDV makes no network calls — it only parses headers/bodies the caller hands it.
Its one genuine peer is **Nerd Herd**, where it publishes capacity + reliability
metrics:

- **Core is Nerd-Herd-independent** — `kdv.py` imports `nerd_herd.burn_log` only
  inside `try/except`, so capacity logic runs even if Nerd Herd is absent.
- **The metrics layer is Nerd-Herd-shaped** — `nerd_herd_adapter.py` translates
  KDV's internal state into Nerd Herd's `CloudProviderState` types; it is the
  deliberate seam to Nerd Herd, not a generic interface.
- **The push sink is injected** — `configure_in_flight_push(sink, state_getter)`
  wires the in-flight push; the sink is duck-typed (`push_cloud_state`) and a
  no-op until configured.
- **Host DB** — `cost_adapter.record_vendor_cost` lazily re-exports the host's
  vendor-cost writer.
- **Env**: `KDV_INFLIGHT_TTL_S` (default 180s).

The host constructs KDV once per process, drives it through the call lifecycle,
and persists/restores its state across reboots.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\kuleden_donen_var\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Türkçesi, hava trafik kulesinin "kuleden dönen var" anonsundan gelir: pist
> müsait olduğunda kule haber verir. KDV de bulut LLM sağlayıcılarının kapasitesi
> açıldığında (ya da kapandığında) host uygulamaya haber veren kuledir.

### Amaç

**Ne işe yarar.** Her bulut LLM sağlayıcısı — Groq, Gemini, Anthropic, OpenAI,
Cerebras, SambaNova ve gelecek olanlar — rate limit ve kotalarını farklı bildirir:
farklı header, farklı eksen, farklı 429 formatı. KDV, bunların herhangi birini
entegre etmeyi zahmetsiz kılmak için vardır. Her sağlayıcının kendi lehçesini
öğrenir, tüm limit ve kotaları tek bir tip arayüz arkasında takip eder ve bu
zahmeti soğurur; böylece çağıran taraf rate limit'i hiç düşünmeden bir litellm
çağrısı yapabilir. **Yeni bir sağlayıcı eklemek bir parser yazmaktır, baştan
yazmak değil.**

**Gerçekte ne yapar.** Her (model, sağlayıcı) için canlı, çok eksenli bir defter
tutar — istek *ve* token, dakika başına *ve* gün başına, iki katmanlı (per-model +
sağlayıcı-toplam) — her sağlayıcının kendi yanıt header'larından ve 429
gövdelerinden beslenir, yanıtlar arasında yerelde düşülür. Bu defterin üstünde
dört koruma oturur (rate limiter, devre kesici, canary tek-uçuş, günlük eksen). Her
çağrıdan önce çağıran taraf `pre_call(...)` sorar ve bekleme tahminiyle birlikte
temiz bir git/gitme yanıtı alır; sonra sonucu bildirir, böylece defter keskin
kalır. Bir sağlayıcının kapasitesi açıldığı anda bir callback tetiklenir.

**Yapmadıkları**: model keşfetmez veya seçmez (config'i host `register` ile
gönderir, seçimi host'un seçicisi yapar), LLM/HTTP çağrısı yapmaz (onu çağıran
taraf yapar ve KDV'ye header + sonuç besler), yerel model veya GPU yönetmez. Auth
hataları kalıcı sayılır, bu yüzden devre kesiciyi tetiklemez.

### Genel API

Süreç başına bir kez host tarafından kurulur ve host'un LLM çağırıcısı tarafından
sürülür. Çağrı yaşam döngüsü sıralı bir sözleşmedir:

```python
from kuleden_donen_var import KuledenDonenVar, KuledenConfig

kdv = KuledenDonenVar(KuledenConfig(on_capacity_change=on_change))
kdv.register("groq/llama-8b", "groq", rpm=30, tpm=131072)   # config'i host gönderir

# Her çağrı — DAİMA bu sırayla:
res = kdv.pre_call("groq/llama-8b", "groq", estimated_tokens=5000)   # -> PreCallResult
if not res.allowed:
    ...  # res.reason / res.wait_seconds / res.daily_exhausted'a uy; başka model seç
kdv.record_attempt("groq/llama-8b", "groq", estimated_tokens=5000)   # TPM rezerve + RPM say
# ...çağrıyı yap...
kdv.post_call("groq/llama-8b", "groq", headers=hdrs, token_count=3200, reserved_tokens=5000)
# YA DA, hata durumunda:
kdv.record_failure("groq/llama-8b", "groq", "rate_limited", error_message=body)
kdv.release_reservation("groq/llama-8b", "groq", reserved_tokens=5000)
```

`pre_call → record_attempt → (post_call | record_failure + release_reservation)`.
`record_attempt` atlanırsa RPM eksik sayılır ve `no_data_warnings` sürekli öter.

Üst düzey export'lar (`__all__`): `KuledenDonenVar`, `KuledenConfig`,
`CapacityEvent`, `ModelStatus`, `ProviderStatus`, `PreCallResult`,
`RateLimitSnapshot`, `InFlightHandle`, `InFlightTracker` sınıfları; header yardımcısı
`parse_rate_limit_headers(provider, headers) -> RateLimitSnapshot | None`; ve
in-flight singleton yardımcıları `begin_call` / `end_call` / `in_flight_count` /
`configure_in_flight_push`.

#### Ana metotlar (`KuledenDonenVar`)

| metot | görevi |
|---|---|
| `register(...)` | host'un per-model + opsiyonel sağlayıcı-toplam limitlerini göndermesi |
| `pre_call(...) -> PreCallResult` | bloklamayan git/gitme: `.allowed / .reason / .wait_seconds / .daily_exhausted / .binding_provider` |
| `record_attempt(...)` | POST'tan **önce** TPM rezerve + RPM say (admission-race kapama) |
| `post_call(..., reserved_tokens=0)` | başarı: rezervasyonu gerçek değere düzelt, header limitlerini uygula, devre kesiciyi sıfırla |
| `record_failure(..., error_message="")` | `rate_limit` / `rate_limited` / `daily_exhausted` / `server_error` / `timeout` / `auth` |
| `release_reservation(...)` | hata durumunda TPM rezervasyonunu geri al |
| `recent_success_rate` · `provider_prior_rate` | güvenilirlik sinyalleri (metrik olarak yayınlanır) |
| `no_data_warnings(min_age_hours=24.0)` | etkin ama hiç denenmemiş sağlayıcılar |
| `snapshot_state` · `restore_state` | yeniden başlatmalar arası kaydet / geri yükle |
| `restore_limits()` | adaptif 429 düşüşlerini kademeli geri al (host watchdog) |
| `status` | sağlayıcı/model bazında tam durum anlık görüntüsü |

### Mimari

İki katmanlı limitler (per-model + sağlayıcı-toplam; ikisi de geçmeli), bir
sonraki UTC gece yarısına sabitlenen günlük eksenler, header'dan türetilen canlı
limitler ve dört koruma:

```
pre_call ─┬─ rate limiter   (RPM/TPM/RPD, iki katmanlı, header ile ezilir)
          ├─ devre kesici    (sağlayıcı başına; tetik → soğuma; auth hariç)
          ├─ canary kapısı   (belirsizlikte tek-uçuş: boot/reset/hata-sonrası)
          └─ günlük eksen     (RPD her denemede yerelde düşer → ilk 429'dan önce tükeniş)
record_attempt → TPM rezerve + RPM say          (POST öncesi race kapama)
post_call / record_failure + release_reservation → mutabakat + güvenilirlik penceresi
                                                  ↘ on_capacity_change callback
in-flight izleyici → begin_call / end_call       (+ opsiyonel metrik push)
```

### Ana Modüller

| modül | rolü |
|---|---|
| `kdv.py` | `KuledenDonenVar` cephesi — yaşam döngüsü, canary, rezervasyon, güvenilirlik, callback |
| `rate_limiter.py` | iki katmanlı RPM/TPM/RPD pencereleri, adaptif 429 düşür/geri-al, 429 sonrası soğuma |
| `header_parser.py` | sağlayıcı bazında header parse, `parse_429_body`, `Retry-After` |
| `circuit_breaker.py` | sağlayıcı başına kayan-pencere hata kesici |
| `in_flight.py` | gönderilmiş-ama-onaylanmamış çağrı izleyici (TTL ile budanır) |
| `nerd_herd_adapter.py` | KDV durumunu Nerd Herd'in tiplerine çevirir, baskı + güvenilirliği iter |
| `cost_adapter.py` | host'un vendor-cost DB yazıcısının kararlı re-export'u |
| `config.py` | `KuledenConfig`, `PreCallResult`, `ProviderStatus`, `ModelStatus`, `CapacityEvent` |

### Kapasite olayları

`on_capacity_change` tam olarak dört olay türünde tetiklenir:

| olay | ne zaman |
|---|---|
| `capacity_restored` | başlık (headroom) toparlar (`önceki rpm_remaining ≤ 1` → `> 5`) |
| `limit_hit` | 429 alındı, limitler adaptif düşürüldü |
| `circuit_breaker_tripped` | tekrarlı hatalardan sonra sağlayıcı devre dışı |
| `circuit_breaker_reset` | temiz çağrıdan sonra sağlayıcı tekrar etkin |

Günlük tükeniş callback olayı **değildir** — `PreCallResult.daily_exhausted` /
`reason="rpd"` üzerinden görünür.

### Tuzaklar

- **Sayım taşındı**: RPM `post_call`'da değil `record_attempt`'te sayılır. Sadece
  `post_call` çağırmak RPM'i eksik sayar ve `no_data_warnings`'i sürekli öttürür.
- **Rezervasyon sözleşmesi**: `record_attempt`'e verilen `estimated_tokens`,
  `post_call`'a `reserved_tokens` olarak (başarı) ya da `release_reservation`'a
  (hata) geri dönmeli; yoksa TPM muhasebesi kayar.
- **Canary tek-uçuş**: belirsizlikte (boot, gün/kesici reset, herhangi bir hata)
  sağlayıcı başına yalnızca bir çağrı kabul edilir; diğerleri
  `reason="canary_in_flight"`, `wait_seconds=5.0` alır. `reason`'a uy, yoksa
  busy-loop'a girersin.
- **`wait_seconds` yük taşır**: devre kesici retleri gerçek kalan soğuma süresini
  döndürür (`0.0` değil), böylece çağıran taraf tekrar denemeden önce gerçekten uyur.
- **In-flight push opt-in'dir**: `configure_in_flight_push()` çalışana kadar
  `begin_call`/`end_call` yalnızca süreç-içi izler, metrik yaymaz.
- **`auth` vs `auth_failure`**: iki yazım da işlenir — `auth` kesiciyi atlar,
  `auth_failure` güvenilirlik penceresinden dışlanır. Karıştırması kolaydır.
- **Senkron, tek-süreç**: kontrol-ve-rezerve yarışı tek-süreç asyncio'ya dayanır
  (senkron çağrılar yield etmez). Rate limiter'ın async `wait_*` yardımcıları
  cephe tarafından kullanılmaz.

### Bağımlılıklar

KDV ağ çağrısı yapmaz — yalnızca çağıran tarafın verdiği header/gövdeleri parse
eder. Tek gerçek dengi **Nerd Herd**'tir; kapasite + güvenilirlik metriklerini
oraya yayınlar:

- **Çekirdek Nerd Herd'den bağımsızdır** — `kdv.py`, `nerd_herd.burn_log`'u yalnızca
  `try/except` içinde import eder; Nerd Herd yoksa bile kapasite mantığı çalışır.
- **Metrik katmanı Nerd Herd biçimlidir** — `nerd_herd_adapter.py`, KDV'nin iç
  durumunu Nerd Herd'in `CloudProviderState` tiplerine çevirir; bu, Nerd Herd'e
  giden bilinçli bir dikiş yeridir, genel bir arayüz değil.
- **Push hedefi enjekte edilir** — `configure_in_flight_push(sink, state_getter)`
  in-flight push'u bağlar; hedef duck-typed'dır (`push_cloud_state`) ve
  yapılandırılana dek no-op'tur.
- **Host DB** — `cost_adapter.record_vendor_cost`, host'un vendor-cost yazıcısını
  lazy olarak re-export eder.
- **Env**: `KDV_INFLIGHT_TTL_S` (varsayılan 180s).

Host, KDV'yi süreç başına bir kez kurar, çağrı yaşam döngüsü boyunca sürer ve
durumunu yeniden başlatmalar arasında kalıcılaştırır.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\kuleden_donen_var\ -q
```
