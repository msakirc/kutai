# HaLLederiz Kadir — LLM call executor

> Nicknamed after a TV character who could talk his way out of anything;
> *"hallederiz"* is Turkish for "we'll handle it," and the double-L winks at
> LLM. He is the package that, once a model has been chosen, just makes the
> call land.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** A single LLM provider call is deceptively hairy: local
GGUF servers stream and can stall mid-token, cloud providers throttle and rotate
ids, thinking models bury reasoning in side channels, and every backend phrases
its errors differently. HaLLederiz Kadir is where all of that is absorbed. Hand
it a chosen model plus messages and it returns one of exactly two things — a
clean `CallResult` or a classified `CallError` — having streamed, watch-dogged,
retried, parsed, quality-checked, and metered the call along the way. The caller
never has to reason about httpx timeouts, `<think>` tags, or which provider this
model happens to live behind.

**What it really does.** It is the "I picked a model" → "here is the response"
stage. It builds the completion kwargs from the model's own capabilities,
routes a *local* call through the GPU-server path (always-streaming, with a
stream-inactivity watchdog and degenerate-output abort) and a *cloud* call
through the capacity-gated path (admission check, reserve, stream, reconcile),
runs a bounded retry loop that understands HTTP status before error text, and
hands back a result the caller can act on. On failure it salvages whatever
streamed bytes arrived so the partial work isn't lost.

**It does NOT** pick or score models — it executes against the *one* model it
was handed (the caller picks and, on a `CallError`, tries the next candidate).
It does NOT start, stop, or swap the local model server, and it does NOT own
rate-limit state — it asks the host's capacity tracker for a go/no-go and
reports outcomes back, but the ledger lives elsewhere. It is not a policy layer:
no swap budgets, no call categorization beyond the label it's told to log under.

## Public API

`call(...)` is the single entry point. The caller invokes it once per candidate
model; the return is a tagged union — branch on the type.

```python
from hallederiz_kadir import call, CallResult, CallError

result = await call(
    model,                          # the chosen model object (capabilities + ids)
    messages,                       # chat messages, already redacted/adapted by the caller
    tools,                          # tool-call defs, or None
    timeout,                        # seconds; 0/negative = "no wall-clock cap" (local)
    task,                           # label for sampling profile + logging
    needs_thinking,                 # request thinking for this call
    estimated_input_tokens=0,       # feeds the cloud capacity (TPM) estimate
    estimated_output_tokens=1000,   # feeds max_tokens + the TPM estimate
    response_format=None,           # optional {"type": "json_schema"|"json_object", ...}
    task_obj=None,                  # the host Task dict, for token/cost telemetry
    iteration_n=0,
    call_category="main_work",      # "main_work" | "overhead", for telemetry
)  # -> CallResult | CallError

if isinstance(result, CallResult):
    result.content        # response text
    result.tool_calls     # list[dict] | None  ({"id", "name", "arguments"})
    result.thinking       # reasoning text | None
    result.cost           # 0.0 for local, computed for cloud
    result.latency        # wall-clock seconds
else:                     # CallError
    result.category       # see categories below
    result.retryable      # hint for the caller's candidate loop
    result.partial_content  # bytes that streamed before the failure, if any
```

`__init__.py` exports exactly three names (`__all__`): the coroutine `call` and
the two dataclasses `CallResult` and `CallError`. There is no other public
surface.

### `CallResult` (success)

```python
@dataclass
class CallResult:
    content: str
    tool_calls: list[dict] | None
    thinking: str | None
    usage: dict                 # {"prompt_tokens", "completion_tokens"}
    cost: float                 # 0.0 for local; litellm-computed for cloud
    latency: float              # seconds
    model: str                  # litellm id
    model_name: str             # human-readable name
    is_local: bool
    provider: str               # "local" for local; provider name for cloud
    task: str
```

### `CallError` (classified failure)

```python
@dataclass
class CallError:
    category: str               # classification (see table)
    message: str                # cleaned provider message
    retryable: bool             # caller hint: re-pick vs give up
    partial_content: str | None = None   # salvaged streamed bytes
    headers: dict[str, str] | None = None  # provider rate-limit headers, if any
    status_code: int | None = None         # HTTP status off the exception, if any
```

`headers` and `status_code` are not cosmetic: the caller forwards `headers` back
to the capacity tracker so `x-ratelimit-*` counters stay in sync even on a 4xx,
and `status_code` drives the "this id is gone" handling (a 404 marks the model
dead for the rest of the process).

### Error categories

`category` is one of: `timeout`, `rate_limited`, `daily_exhausted`,
`auth_failure`, `model_not_found`, `server_error`, `connection_error`,
`gpu_busy`, `loading`, `circuit_breaker`, `no_model`, `context_overflow`,
`json_unsupported`, `quality_failure`, or `unknown`. Classification consults the
HTTP **status code first** (stable across providers) and falls back to error
**text** only for locally-raised errors and to disambiguate buckets that share a
status (e.g. minute vs. daily 429).

## Architecture

Two backend paths behind one signature; the caller can't tell which ran:

```
call(model, messages, …)
  │  build completion_kwargs  (sampling · max_tokens · api_key/provider ·
  │                            tools | response_format · reasoning override)
  ├── LOCAL  ─ stream (always) ─ inactivity watchdog + degenerate-abort ─ parse
  │           (in-flight mark so the idle-unload watchdog won't yank the model)
  └── CLOUD  ─ pre_call gate ─ reserve TPM/RPM ─ stream ─ reconcile via post_call
              (on failure: release reservation, feed 429 headers/body back)
        │
        └── execute_with_retry  (status-first classification, Retry-After backoff,
                                 local: loading-wait / swap-wait; partial salvage)
        └── parse_response       (content · tool_calls · thinking · think-tag strip
                                  · reasoning rescue · cost)
        └── quality check ─ metrics ─ token/cost telemetry ─ audit
```

## Key Modules

| module | role |
|---|---|
| `caller.py` | `call()` — kwargs building, local/cloud routing, the streaming accumulator, capacity-tracker handshake, post-failure header/404/auth handling, metrics + telemetry |
| `retry.py` | `execute_with_retry` loop, `classify_error` (status-first), Retry-After backoff, provider-name cleanup in messages |
| `response.py` | `parse_response` — content / tool_calls / thinking extraction, `<think>`-tag stripping, reasoning-content rescue, cost |
| `types.py` | `CallResult`, `CallError` dataclasses |

## Dependencies

- **`litellm`** (declared) — every completion goes through `litellm.acompletion`.
- **The cloud capacity tracker** is a genuine hard peer. Most touch points are
  defensively wrapped (the call still runs if it's absent), but the retry loop
  reads its `Retry-After` parser directly and the cloud admission/reconcile
  handshake (`begin_call`/`end_call`, header + 429-body parsing) is shaped to
  its types — this is a deliberate seam, not a generic interface.
- **The degenerate-output detector** is also hard on the dominant path: the
  streaming accumulator imports its stream callback unguarded to abort
  repetitive generation mid-flight. (The non-streaming quality check that runs
  *after* a call is, by contrast, best-effort.)
- **Everything else is soft / host-provided.** Sampling profiles, the
  streaming-guard pipeline, the local-model manager, the registry, logging,
  metrics, audit, heartbeat, and DB telemetry are all imported inside
  `try/except` (or via lazy accessors that fall back to no-ops), so the package
  imports and a call executes even when the host wiring is missing — that's what
  makes the unit tests runnable in isolation.

## Gotchas

- **Two-value contract, not exceptions.** `call` never raises for a failed LLM
  call — it returns a `CallError`. Always `isinstance`-branch the result.
- **`timeout <= 0` is a sentinel, not a bug.** For local models the caller
  passes `0.0` to mean "no wall-clock cap"; the stream-inactivity watchdog
  (first-chunk 180 s, inter-chunk 20 s) governs hangs instead. A positive
  timeout becomes the HTTP deadline minus 5 s headroom (floored at 10 s).
- **Local always streams; Ollama never does.** Non-streaming + thinking + tools
  made the local server silently drop requests, so local is always streamed and
  the accumulator reassembles tool-call deltas. Ollama stays non-streaming (its
  OpenAI-compat layer mishandled streamed tool calls).
- **Cloud capacity calls must balance.** A cloud admission reserves TPM and an
  RPM in-flight handle; the success path reconciles via `post_call`, the failure
  path releases the reservation and the handle is cleared in a `finally`.
  Short-circuiting around `call` would leak reservations for ~60 s.
- **`GOOGLE_API_KEY` is popped at import time.** Importing the package removes
  `GOOGLE_API_KEY` from the environment (saving it to `GOOGLE_API_KEY_SAVED`) so
  litellm's gemini auto-router doesn't silently switch to a Vertex backend.
  Import order matters: import this before anything that reads that env var.
- **Status code outranks error text.** When tuning classification, change the
  status-code map first; the text matchers exist only for errors that arrive
  without an HTTP status, and several are narrow on purpose (loose `404`/`"not
  found"` matching once mass-marked models dead off request-echo bodies).

## Tests

Run from the repo root (the tests import host `src.*` modules, stubbed by the
package's `conftest.py`):

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\hallederiz_kadir\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Lakabı, her işin içinden konuşarak sıyrılan bir TV karakterinden gelir;
> *"hallederiz"* "biz icabına bakarız" demektir ve içindeki çift L, LLM'e göz
> kırpar. Bir model seçildikten sonra çağrıyı sorunsuz yere indiren paket odur.

### Amaç

**Ne işe yarar.** Tek bir LLM sağlayıcı çağrısı göründüğünden çetrefillidir:
yerel GGUF sunucuları stream eder ve token ortasında takılabilir, bulut
sağlayıcıları throttle eder ve id'leri döndürür, düşünen (thinking) modeller akıl
yürütmeyi yan kanallara gömer ve her backend hatasını farklı ifade eder.
HaLLederiz Kadir, bunların hepsinin soğurulduğu yerdir. Ona seçilmiş bir model
ile mesajları verirsin, sana tam olarak iki şeyden birini döndürür — temiz bir
`CallResult` ya da sınıflandırılmış bir `CallError` — bu arada çağrıyı stream
eder, gözcü (watchdog) ile izler, yeniden dener, ayrıştırır, kalite kontrolünden
geçirir ve ölçer. Çağıran tarafın httpx zaman aşımlarını, `<think>` etiketlerini
ya da bu modelin hangi sağlayıcının ardında olduğunu hiç düşünmesi gerekmez.

**Gerçekte ne yapar.** "Bir model seçtim" → "işte yanıt" aşamasıdır. Completion
parametrelerini modelin kendi yeteneklerinden kurar; bir *yerel* çağrıyı
GPU-sunucu yolundan (her zaman streaming, stream-durağanlık gözcüsü ve dejenere
çıktı durdurması ile), bir *bulut* çağrısını ise kapasite-kapılı yoldan (kabul
kontrolü, rezerve et, stream et, mutabakat) geçirir; HTTP durum kodunu hata
metninden önce dikkate alan sınırlı bir yeniden-deneme döngüsü çalıştırır ve
çağıran tarafın üzerine işlem yapabileceği bir sonuç verir. Hata durumunda,
stream'den gelen ne kadar bayt geldiyse onu kurtarır; böylece kısmi iş kaybolmaz.

**Yapmadıkları.** Model seçmez veya puanlamaz — kendisine verilen *tek* modele
karşı çalışır (seçimi çağıran taraf yapar ve bir `CallError`'da bir sonraki
adayı dener). Yerel model sunucusunu başlatmaz, durdurmaz veya takas (swap)
etmez ve rate-limit durumuna sahip değildir — host'un kapasite takipçisine
git/gitme sorar ve sonuçları geri bildirir, ama defter başka yerde tutulur. Bir
politika katmanı değildir: takas bütçesi yoktur, loglayacağı etiketin ötesinde
çağrı sınıflandırması yapmaz.

### Genel API

`call(...)` tek giriş noktasıdır. Çağıran taraf bunu her aday model için bir kez
çağırır; dönüş etiketli bir birleşimdir (tagged union) — tipe göre dallan.

```python
from hallederiz_kadir import call, CallResult, CallError

result = await call(
    model,                          # seçilmiş model nesnesi (yetenekler + id'ler)
    messages,                       # mesajlar, çağıran tarafça zaten redakte/uyarlanmış
    tools,                          # tool-call tanımları veya None
    timeout,                        # saniye; 0/negatif = "duvar-saati sınırı yok" (yerel)
    task,                           # sampling profili + loglama için etiket
    needs_thinking,                 # bu çağrı için thinking iste
    estimated_input_tokens=0,       # bulut kapasite (TPM) tahminini besler
    estimated_output_tokens=1000,   # max_tokens + TPM tahminini besler
    response_format=None,           # opsiyonel {"type": "json_schema"|"json_object", ...}
    task_obj=None,                  # token/maliyet telemetrisi için host Task sözlüğü
    iteration_n=0,
    call_category="main_work",      # "main_work" | "overhead", telemetri için
)  # -> CallResult | CallError

if isinstance(result, CallResult):
    result.content        # yanıt metni
    result.tool_calls     # list[dict] | None  ({"id", "name", "arguments"})
    result.thinking       # akıl yürütme metni | None
    result.cost           # yerel için 0.0, bulut için hesaplanır
    result.latency        # duvar-saati saniye
else:                     # CallError
    result.category       # aşağıdaki kategorilere bak
    result.retryable      # çağıranın aday döngüsü için ipucu
    result.partial_content  # hatadan önce stream olan baytlar, varsa
```

`__init__.py` tam olarak üç isim export eder (`__all__`): `call` coroutine'i ve
iki dataclass `CallResult` ile `CallError`. Başka açık yüzey yoktur.

#### `CallResult` (başarı)

```python
@dataclass
class CallResult:
    content: str
    tool_calls: list[dict] | None
    thinking: str | None
    usage: dict                 # {"prompt_tokens", "completion_tokens"}
    cost: float                 # yerel için 0.0; bulut için litellm hesabı
    latency: float              # saniye
    model: str                  # litellm id
    model_name: str             # okunabilir isim
    is_local: bool
    provider: str               # yerel için "local"; bulut için sağlayıcı adı
    task: str
```

#### `CallError` (sınıflandırılmış hata)

```python
@dataclass
class CallError:
    category: str               # sınıflandırma (tabloya bak)
    message: str                # temizlenmiş sağlayıcı mesajı
    retryable: bool             # çağırana ipucu: yeniden-seç mi, vazgeç mi
    partial_content: str | None = None   # kurtarılmış stream baytları
    headers: dict[str, str] | None = None  # sağlayıcı rate-limit header'ları, varsa
    status_code: int | None = None         # exception'daki HTTP durum kodu, varsa
```

`headers` ve `status_code` süs değildir: çağıran taraf `headers`'ı kapasite
takipçisine geri iletir ki `x-ratelimit-*` sayaçları bir 4xx'te bile senkron
kalsın; `status_code` ise "bu id artık yok" işlemini sürer (bir 404, modeli
süreç boyunca ölü işaretler).

#### Hata kategorileri

`category` şunlardan biridir: `timeout`, `rate_limited`, `daily_exhausted`,
`auth_failure`, `model_not_found`, `server_error`, `connection_error`,
`gpu_busy`, `loading`, `circuit_breaker`, `no_model`, `context_overflow`,
`json_unsupported`, `quality_failure` veya `unknown`. Sınıflandırma önce HTTP
**durum koduna** bakar (sağlayıcılar arası kararlıdır) ve yalnızca yerelde
oluşan hatalar için ya da aynı durumu paylaşan kovaları ayırmak için (örn.
dakikalık vs. günlük 429) hata **metnine** düşer.

### Mimari

Tek imza ardında iki backend yolu; çağıran taraf hangisinin çalıştığını anlayamaz:

```
call(model, messages, …)
  │  completion_kwargs kur  (sampling · max_tokens · api_key/sağlayıcı ·
  │                          tools | response_format · reasoning ezme)
  ├── YEREL  ─ stream (her zaman) ─ durağanlık gözcüsü + dejenere-durdurma ─ ayrıştır
  │           (modeli in-flight işaretle ki idle-unload gözcüsü onu çekmesin)
  └── BULUT  ─ pre_call kapısı ─ TPM/RPM rezerve ─ stream ─ post_call mutabakatı
              (hata durumunda: rezervasyonu serbest bırak, 429 header/gövdeyi geri besle)
        │
        └── execute_with_retry  (durum-önce sınıflandırma, Retry-After backoff,
                                 yerel: loading-bekle / swap-bekle; kısmi kurtarma)
        └── parse_response       (content · tool_calls · thinking · think-tag temizleme
                                  · reasoning kurtarma · maliyet)
        └── kalite kontrolü ─ metrikler ─ token/maliyet telemetrisi ─ audit
```

### Ana Modüller

| modül | rolü |
|---|---|
| `caller.py` | `call()` — kwargs kurulumu, yerel/bulut yönlendirme, streaming biriktirici, kapasite-takipçisi el sıkışması, hata-sonrası header/404/auth işleme, metrik + telemetri |
| `retry.py` | `execute_with_retry` döngüsü, `classify_error` (durum-önce), Retry-After backoff, mesajlarda sağlayıcı adı temizleme |
| `response.py` | `parse_response` — content / tool_calls / thinking çıkarımı, `<think>`-etiket temizleme, reasoning-content kurtarma, maliyet |
| `types.py` | `CallResult`, `CallError` dataclass'ları |

### Bağımlılıklar

- **`litellm`** (tanımlı) — her completion `litellm.acompletion` üzerinden geçer.
- **Bulut kapasite takipçisi** gerçek bir sıkı dengidir (hard peer). Çoğu temas
  noktası savunmacı şekilde sarmalanmıştır (yokken bile çağrı çalışır), ama
  yeniden-deneme döngüsü onun `Retry-After` parser'ını doğrudan okur ve bulut
  kabul/mutabakat el sıkışması (`begin_call`/`end_call`, header + 429-gövde
  ayrıştırma) onun tiplerine göre biçimlenmiştir — bu, genel bir arayüz değil,
  bilinçli bir dikiş yeridir.
- **Dejenere-çıktı dedektörü** de baskın yolda sıkıdır: streaming biriktirici,
  tekrarlı üretimi anında durdurmak için onun stream callback'ini sarmalamadan
  import eder. (Bir çağrıdan *sonra* çalışan streaming-dışı kalite kontrolü ise,
  aksine, en-iyi-çaba esaslıdır.)
- **Geri kalan her şey yumuşaktır / host tarafından sağlanır.** Sampling
  profilleri, streaming-guard hattı, yerel-model yöneticisi, registry, loglama,
  metrikler, audit, heartbeat ve DB telemetrisi hepsi `try/except` içinde (ya da
  no-op'a düşen lazy erişimcilerle) import edilir; böylece host bağlantıları
  eksikken bile paket import olur ve bir çağrı çalışır — birim testlerin
  izolasyonda çalışabilmesini sağlayan da budur.

### Tuzaklar

- **İki-değer sözleşmesi, exception değil.** `call`, başarısız bir LLM çağrısı
  için asla exception fırlatmaz — bir `CallError` döndürür. Sonucu her zaman
  `isinstance` ile dallandır.
- **`timeout <= 0` bir sentinel'dir, hata değil.** Yerel modeller için çağıran
  taraf "duvar-saati sınırı yok" demek için `0.0` geçer; takılmaları bunun
  yerine stream-durağanlık gözcüsü (ilk-chunk 180 sn, chunk-arası 20 sn) yönetir.
  Pozitif bir timeout, HTTP süresi eksi 5 sn pay (en az 10 sn) olur.
- **Yerel her zaman stream eder; Ollama asla etmez.** Streaming-dışı + thinking
  + tools, yerel sunucunun istekleri sessizce düşürmesine yol açtı; bu yüzden
  yerel her zaman stream edilir ve biriktirici tool-call delta'larını yeniden
  birleştirir. Ollama streaming-dışı kalır (OpenAI-uyumlu katmanı stream'lenen
  tool çağrılarını yanlış işliyordu).
- **Bulut kapasite çağrıları dengelenmeli.** Bir bulut kabulü TPM ve bir RPM
  in-flight tutamacı rezerve eder; başarı yolu `post_call` ile mutabakat yapar,
  hata yolu rezervasyonu serbest bırakır ve tutamaç bir `finally`'de temizlenir.
  `call` etrafından kısa devre yapmak rezervasyonları ~60 sn sızdırır.
- **`GOOGLE_API_KEY` import anında çıkarılır.** Paketi import etmek
  `GOOGLE_API_KEY`'i ortamdan kaldırır (`GOOGLE_API_KEY_SAVED`'e kaydederek) ki
  litellm'in gemini auto-router'ı sessizce bir Vertex backend'ine geçmesin.
  Import sırası önemlidir: bunu, o env değişkenini okuyan her şeyden önce import et.
- **Durum kodu hata metninden üstündür.** Sınıflandırmayı ayarlarken önce
  durum-kodu haritasını değiştir; metin eşleştiricileri yalnızca HTTP durumu
  olmadan gelen hatalar için vardır ve birkaçı bilinçli olarak dardır (gevşek
  `404`/`"not found"` eşleşmesi bir zamanlar istek-yankısı gövdelerinden ötürü
  modelleri toplu hâlde ölü işaretliyordu).

### Testler

Repo kökünden çalıştır (testler host `src.*` modüllerini import eder, paketin
`conftest.py`'ı bunları stub'lar):

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\hallederiz_kadir\ -q
```
