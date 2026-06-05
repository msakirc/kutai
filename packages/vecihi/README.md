# Vecihi — Auto-escalating web fetcher

> Vecihi is the one who gets through the door when the polite knock fails. Hand it
> a URL; it keeps trying harder disguises until the page comes back — or tells you
> honestly that it couldn't.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** Real-world scraping breaks on anti-bot walls: a plain HTTP
request that works in dev gets a 403, a Cloudflare "Just a moment…" interstitial,
or a 429 in production. Vecihi removes that whole headache. The caller fetches a
URL *once* and Vecihi decides — per response — whether a cheap request was enough
or whether it needs to escalate to a heavier, more browser-like disguise. The
caller never reasons about TLS fingerprints, headless browsers, or which sites
need which trick.

**What it really does.** It owns a single decision: *given how blocked this
response looks, do I retry with a stronger tier?* Four tiers sit in a fixed ladder
— plain HTTP, TLS-fingerprint impersonation, a stealth browser, a full headless
browser — each more capable and more expensive than the last. Vecihi starts at the
cheapest, runs `detect_block` on the result, and climbs one rung only when the
page was *blocked* (not when it timed out or errored for another reason). It stops
at the first page that comes back clean, or at the `max_tier` ceiling the caller
set. Every result is the same flat `ScrapeResult` regardless of which tier won.

**It does NOT** parse, clean, or extract anything from the HTML — it returns the
raw page and the caller does the parsing. It does NOT crawl, follow links, retry
on its own schedule, rotate proxies, solve CAPTCHAs, or cache. The heavier tiers
are **optional dependencies**: if `curl_cffi` or `scrapling` isn't installed, that
tier is skipped with an honest `error` on the result — Vecihi never crashes the
caller because a browser engine is missing.

## Public API

Two entry points cover almost every use. Everything is async; nothing raises on a
failed fetch — failure is reported *in* the `ScrapeResult`.

```python
from vecihi import scrape_url, scrape_urls, ScrapeTier, ScrapeResult

# Single URL — escalates HTTP -> TLS by default (max_tier=TLS).
res = await scrape_url("https://example.com")                  # -> ScrapeResult
if res.ok:                                                     # 200 + no error + non-empty html
    parse(res.html)                                            # res.tier tells you which tier won
else:
    log(res.error)                                             # "blocked" / "timeout" / "<exc>" / "all tiers failed"

# Push harder for a known-tough site (each tier needs its optional dep installed):
res = await scrape_url("https://tough.example", max_tier=ScrapeTier.BROWSER)

# Many URLs, bounded concurrency. Non-http(s) URLs are silently dropped.
results = await scrape_urls(urls, max_tier=ScrapeTier.TLS, max_concurrent=5)  # -> dict[str, ScrapeResult]
```

Signatures (verified against source):

| symbol | signature | returns |
|---|---|---|
| `scrape_url` | `(url, max_tier=ScrapeTier.TLS, timeout=None)` | `ScrapeResult` |
| `scrape_urls` | `(urls, max_tier=ScrapeTier.TLS, max_concurrent=5)` | `dict[str, ScrapeResult]` |
| `ScrapeResult` | dataclass: `html, status, tier, url, error=None, headers={}` | `.ok` property: `status==200 and not error and bool(html)` |
| `ScrapeTier` | `IntEnum`: `HTTP=0 < TLS=1 < STEALTH=2 < BROWSER=3` | ordered, comparable |
| `detect_block` | `(status, html, headers) -> bool` | True on 403/429/402/451, CF-503, or a challenge marker in the first 2 KB of HTML |
| `install_browser_error_suppressor` | `()` | installs an asyncio exception handler that swallows orphaned playwright/patchright futures; no-op if no loop is running |

Per-tier fetchers — `fetch_http`, `fetch_tls`, `fetch_stealth`, `fetch_browser`
`(url, timeout=...) -> ScrapeResult` — and the constant `CHALLENGE_MARKERS` are
also exported, but call them directly only when you want a *specific* tier with no
escalation. Most callers use `scrape_url`.

## Architecture

The ladder. `scrape_url` walks tiers in order, stopping at the ceiling or the first
clean page. Escalation happens **only** on a block:

```
scrape_url(url, max_tier)
  │
  ├─ HTTP    (tier 0)  aiohttp                    ─┐
  ├─ TLS     (tier 1)  curl_cffi impersonate       │  on each result:
  ├─ STEALTH (tier 2)  scrapling StealthyFetcher    │   ok?     → return it
  └─ BROWSER (tier 3)  scrapling DynamicFetcher    ─┘   blocked? → climb one rung (if ≤ max_tier)
                                                         other error (timeout/...) → return as-is, do NOT climb
```

- **Stop on success.** `result.ok` ends the climb immediately.
- **Climb only on `error == "blocked"`.** A timeout, a missing optional dep, or any
  other error returns straight to the caller — Vecihi will not burn a browser tier
  on a site that was merely slow.
- **`max_tier` is the ceiling, not the floor.** Every call still starts at HTTP.
- **`scrape_urls`** fans `scrape_url` out under an `asyncio.Semaphore`; non-`http(s)`
  URLs are filtered before dispatch and absent from the returned dict.

## Dependencies

Vecihi imports **no other KutAI package** — it is a pure leaf and a host can drop it
into any asyncio program. Its dependencies are third-party HTTP/browser libraries,
tiered by need:

| tier | library | install | if missing |
|---|---|---|---|
| HTTP | `aiohttp` | **required** (`aiohttp>=3.9.0`) | — |
| TLS | `curl_cffi` | optional extra `tls` | tier returns `error="curl_cffi not installed"`, escalation continues |
| STEALTH / BROWSER | `scrapling` (Camoufox / Playwright) | optional extra `stealth` | tier returns `error="scrapling not installed"`, escalation continues |

Install heavier tiers with `pip install -e packages/vecihi[all]`. The stealth and
browser tiers also need their browser engines (Camoufox, Playwright Chromium)
provisioned by `scrapling` itself.

## Gotchas

- **Default ceiling is TLS, not BROWSER.** `scrape_url` stops escalating after the
  TLS tier unless you raise `max_tier`. The browser tiers are opt-in per call.
- **Nothing raises on failure.** A blocked, timed-out, or dependency-missing fetch
  is a normal `ScrapeResult` with `.ok == False` and a populated `.error`. Check
  `.ok`; don't wrap calls in `try/except` expecting exceptions.
- **`.ok` requires non-empty HTML.** A 200 with an empty body is `not ok` and will
  trigger escalation to the next tier.
- **Block detection only sniffs the first 2 KB.** `detect_block` lowercases the
  leading 2000 chars to spot challenge markers; a challenge phrase buried deeper in
  the page is missed by design (cheap, fast).
- **The error suppressor needs a running loop.** Call
  `install_browser_error_suppressor()` from inside the event loop (the stealth and
  browser fetchers self-install it); outside a loop it's a silent no-op. It only
  swallows playwright/patchright cancellation noise — never general exceptions.
- **`error` strings are truncated to 200 chars** — fine for logs, not for
  programmatic matching beyond the known sentinels (`"blocked"`, `"timeout"`,
  `"...not installed"`, `"all tiers failed"`).

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\vecihi\ -q
```

The suite mocks `aiohttp` and the tier fetchers, so it runs offline and needs none
of the optional browser dependencies.

---
<a id="türkçe"></a>

## Türkçe

> Vecihi, kibar kapı çalması işe yaramadığında içeri girmenin yolunu bulan adamdır.
> Ona bir URL ver; sayfa gelene kadar giderek daha güçlü kılıklarla denemeyi
> sürdürür — ya da geçemediğini sana dürüstçe söyler.

### Amaç

**Ne işe yarar.** Gerçek dünyada scraping, anti-bot duvarlarına toslar: geliştirme
ortamında çalışan düz bir HTTP isteği, canlıda 403, Cloudflare'in "Just a moment…"
ara sayfası ya da 429 alır. Vecihi bu derdi tümüyle ortadan kaldırır. Çağıran taraf
URL'yi *bir kez* ister; ucuz bir isteğin yetip yetmediğine ya da daha ağır,
tarayıcıya benzer bir kılığa yükselmek gerekip gerekmediğine — yanıt yanıt —
Vecihi karar verir. Çağıran taraf TLS parmak izlerini, başsız tarayıcıları ya da
hangi sitenin hangi numarayı gerektirdiğini hiç düşünmez.

**Gerçekte ne yapar.** Tek bir kararın sahibidir: *bu yanıt ne kadar engellenmiş
görünüyor, daha güçlü bir katmanla tekrar deneyeyim mi?* Dört katman sabit bir
merdivende durur — düz HTTP, TLS parmak izi taklidi, gizli (stealth) tarayıcı ve
tam başsız tarayıcı — her biri bir öncekinden daha yetenekli ve daha pahalı. Vecihi
en ucuzundan başlar, sonuca `detect_block` uygular ve yalnızca sayfa
*engellendiğinde* (zaman aşımı ya da başka bir hata değil) bir basamak çıkar. İlk
temiz dönen sayfada ya da çağıranın belirlediği `max_tier` tavanında durur. Hangi
katman kazanırsa kazansın sonuç hep aynı düz `ScrapeResult`'tır.

**Yapmadıkları**: HTML'i ayrıştırmaz, temizlemez ya da içinden veri çıkarmaz — ham
sayfayı döndürür, ayrıştırmayı çağıran taraf yapar. Tarama (crawl) yapmaz, bağlantı
izlemez, kendi takvimine göre yeniden denemez, proxy döndürmez, CAPTCHA çözmez ya
da önbelleğe almaz. Ağır katmanlar **opsiyonel bağımlılıktır**: `curl_cffi` ya da
`scrapling` kurulu değilse o katman atlanır ve sonuca dürüst bir `error` yazılır —
bir tarayıcı motoru eksik diye Vecihi çağıranı asla çökertmez.

### Genel API

İki giriş noktası hemen her kullanımı karşılar. Hepsi async'tir; başarısız bir
çağrı istisna fırlatmaz — başarısızlık `ScrapeResult`'ın *içinde* bildirilir.

```python
from vecihi import scrape_url, scrape_urls, ScrapeTier, ScrapeResult

# Tek URL — varsayılan olarak HTTP -> TLS yükselir (max_tier=TLS).
res = await scrape_url("https://example.com")                  # -> ScrapeResult
if res.ok:                                                     # 200 + hata yok + boş olmayan html
    parse(res.html)                                            # res.tier hangi katmanın kazandığını söyler
else:
    log(res.error)                                             # "blocked" / "timeout" / "<exc>" / "all tiers failed"

# Zoru bilinen bir site için daha sert bastır (her katman kendi opsiyonel bağımlılığını ister):
res = await scrape_url("https://tough.example", max_tier=ScrapeTier.BROWSER)

# Çok sayıda URL, sınırlı eşzamanlılık. http(s) olmayan URL'ler sessizce atılır.
results = await scrape_urls(urls, max_tier=ScrapeTier.TLS, max_concurrent=5)  # -> dict[str, ScrapeResult]
```

İmzalar (kaynağa karşı doğrulandı):

| sembol | imza | döner |
|---|---|---|
| `scrape_url` | `(url, max_tier=ScrapeTier.TLS, timeout=None)` | `ScrapeResult` |
| `scrape_urls` | `(urls, max_tier=ScrapeTier.TLS, max_concurrent=5)` | `dict[str, ScrapeResult]` |
| `ScrapeResult` | dataclass: `html, status, tier, url, error=None, headers={}` | `.ok` özelliği: `status==200 and not error and bool(html)` |
| `ScrapeTier` | `IntEnum`: `HTTP=0 < TLS=1 < STEALTH=2 < BROWSER=3` | sıralı, karşılaştırılabilir |
| `detect_block` | `(status, html, headers) -> bool` | 403/429/402/451'de, CF-503'te ya da HTML'in ilk 2 KB'ında bir challenge işareti olduğunda True |
| `install_browser_error_suppressor` | `()` | sahipsiz playwright/patchright future'larını yutan bir asyncio hata işleyicisi kurar; çalışan döngü yoksa no-op |

Katman bazlı fetcher'lar — `fetch_http`, `fetch_tls`, `fetch_stealth`,
`fetch_browser` `(url, timeout=...) -> ScrapeResult` — ve `CHALLENGE_MARKERS`
sabiti de export edilir; ama bunları doğrudan yalnızca yükselme olmadan *belirli*
bir katman istediğinde çağır. Çoğu çağıran taraf `scrape_url` kullanır.

### Mimari

Merdiven. `scrape_url` katmanları sırayla gezer; tavanda ya da ilk temiz sayfada
durur. Yükselme **yalnızca** engellemede olur:

```
scrape_url(url, max_tier)
  │
  ├─ HTTP    (katman 0)  aiohttp                    ─┐
  ├─ TLS     (katman 1)  curl_cffi impersonate       │  her sonuçta:
  ├─ STEALTH (katman 2)  scrapling StealthyFetcher    │   ok?       → döndür
  └─ BROWSER (katman 3)  scrapling DynamicFetcher    ─┘   blocked?  → bir basamak çık (≤ max_tier ise)
                                                          başka hata (timeout/...) → olduğu gibi döndür, ÇIKMA
```

- **Başarıda dur.** `result.ok` tırmanışı anında bitirir.
- **Yalnızca `error == "blocked"` ise çık.** Zaman aşımı, eksik opsiyonel bağımlılık
  ya da başka bir hata doğrudan çağırana döner — Vecihi yalnızca yavaş olan bir
  sitede tarayıcı katmanı yakmaz.
- **`max_tier` tavandır, taban değil.** Her çağrı yine HTTP'den başlar.
- **`scrape_urls`**, `scrape_url`'ı bir `asyncio.Semaphore` altında dağıtır;
  `http(s)` olmayan URL'ler dağıtımdan önce elenir ve dönen sözlükte yer almaz.

### Bağımlılıklar

Vecihi **başka hiçbir KutAI paketini** import etmez — saf bir yaprak pakettir ve
host onu herhangi bir asyncio programına bırakabilir. Bağımlılıkları, ihtiyaca göre
katmanlanmış üçüncü taraf HTTP/tarayıcı kütüphaneleridir:

| katman | kütüphane | kurulum | yoksa |
|---|---|---|---|
| HTTP | `aiohttp` | **zorunlu** (`aiohttp>=3.9.0`) | — |
| TLS | `curl_cffi` | opsiyonel extra `tls` | katman `error="curl_cffi not installed"` döner, yükselme sürer |
| STEALTH / BROWSER | `scrapling` (Camoufox / Playwright) | opsiyonel extra `stealth` | katman `error="scrapling not installed"` döner, yükselme sürer |

Ağır katmanları `pip install -e packages/vecihi[all]` ile kur. Stealth ve browser
katmanları ayrıca kendi tarayıcı motorlarını (Camoufox, Playwright Chromium)
gerektirir; bunları `scrapling` sağlar.

### Tuzaklar

- **Varsayılan tavan TLS'tir, BROWSER değil.** `max_tier` yükseltilmedikçe
  `scrape_url` TLS katmanından sonra yükselmeyi bırakır. Tarayıcı katmanları çağrı
  bazında isteğe bağlıdır.
- **Başarısızlıkta hiçbir şey fırlatmaz.** Engellenen, zaman aşımına uğrayan ya da
  bağımlılığı eksik bir çağrı, `.ok == False` ve dolu bir `.error` taşıyan normal
  bir `ScrapeResult`'tır. `.ok`'u kontrol et; istisna bekleyip çağrıları
  `try/except`'e sarma.
- **`.ok` boş olmayan HTML ister.** Gövdesi boş bir 200 `not ok`'tur ve bir sonraki
  katmana yükselmeyi tetikler.
- **Engel tespiti yalnızca ilk 2 KB'a bakar.** `detect_block` baştaki 2000 karakteri
  küçük harfe çevirip challenge işaretlerini arar; sayfanın derinine gömülü bir
  challenge ifadesi tasarım gereği kaçırılır (ucuz, hızlı).
- **Hata bastırıcı çalışan bir döngü ister.**
  `install_browser_error_suppressor()`'ı olay döngüsünün içinden çağır (stealth ve
  browser fetcher'ları onu kendileri kurar); döngü dışında sessiz bir no-op'tur.
  Yalnızca playwright/patchright iptal gürültüsünü yutar — genel istisnaları asla.
- **`error` metinleri 200 karaktere kısaltılır** — loglar için yeterli, ama bilinen
  işaretler (`"blocked"`, `"timeout"`, `"...not installed"`, `"all tiers failed"`)
  dışında programatik eşleştirme için değil.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\vecihi\ -q
```

Test paketi `aiohttp`'yi ve katman fetcher'larını mock'lar; bu yüzden çevrimdışı
çalışır ve opsiyonel tarayıcı bağımlılıklarının hiçbirine ihtiyaç duymaz.
