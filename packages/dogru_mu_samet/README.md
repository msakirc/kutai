# Doğru mu Samet — Degenerate-output detector

> Turkish for "is it right, Samet?" — the sceptical second pair of eyes that
> glances at a draft and says whether it's coherent or garbage. That's this
> package: a fast, opinion-free sanity check on whatever an LLM just produced.

[English](#english) · [Türkçe](#türkçe)

<a id="english"></a>

## Purpose

**What it's good for.** Local LLMs fail in ugly, recognisable ways: they loop a
paragraph forever, spit the same `## header` ten times, collapse into low-entropy
mush (`the the the …`), or run away to a multi-megabyte blob. Doğru mu Samet exists
so the caller can catch those failures *cheaply and without another model* —
no second LLM call, no network, no embeddings. Hand it the text, get back a verdict.

**What it really does.** It runs four pure-Python heuristics over a string —
size, duplicated markdown headers, duplicated paragraphs, and token entropy — and
folds them into one `is_degenerate` flag with human-readable `reasons`. It can also
hand back a stateful streaming callback that aborts a runaway generation mid-flight,
and a `salvage` pass that deduplicates repeated markdown sections so partially-good
output isn't thrown away.

**It does NOT** judge whether content is *correct*, *relevant*, or *well-written* —
it only catches mechanical degeneracy, not bad-but-coherent answers (that's the
caller's grader). It makes no LLM or network calls, has zero dependencies, decides
nothing about retries or model swaps, and `salvage` only dedups markdown sections —
it does not rewrite prose or fix non-markdown text.

## Public API

Four exports. All synchronous, all pure functions over a string.

```python
from dogru_mu_samet import assess, salvage, make_stream_callback, ContentQualityResult

# 1. Verdict on finished output.
result = assess(text, max_size=20_000)        # -> ContentQualityResult
if result.is_degenerate:
    print(result.summary)                     # e.g. "degenerate: low_entropy (1.84 bits)"
    print(result.reasons)                     # ["low_entropy", ...]

# 2. Recover partially-good markdown (drops duplicate / empty sections).
clean = salvage(text)                         # -> str ("" if nothing survives)

# 3. Abort a runaway stream early.
should_abort = make_stream_callback(max_size=200_000, check_interval=2000)
if should_abort(accumulated_so_far):          # -> bool
    cancel_generation()
```

`assess(text, max_size=20_000) -> ContentQualityResult` — `text` may be a `str`,
a `dict` (JSON-dumped first), or anything (`str()`-coerced). `max_size` is clamped
down to the absolute `HARD_CAP` (60 000 chars); you cannot raise the ceiling above it.

`ContentQualityResult` fields: `size: int`, `max_size: int`,
`repetition_ratio: float` (header dup ratio), `paragraph_repetition: float`,
`token_entropy: float`, `is_degenerate: bool`, `reasons: list[str]`, plus a
`.summary` property. Reason tags: `size_exceeded`, `header_repetition`,
`paragraph_repetition`, `low_entropy`.

`salvage(text) -> str` — keeps the first occurrence of each normalised `## header`,
drops empty-body sections; returns non-markdown text (no `## ` headers) unchanged,
and `""` if nothing salvageable survives.

`make_stream_callback(max_size=20_000, check_interval=4096) -> Callable[[str], bool]`
— returns a stateful closure; call it with the accumulated text, get `True` to abort.
Size is checked every call (cheap `len`); the full `assess` runs only every
`check_interval` chars.

## Gotchas

- **Size alone is not degenerate.** Large-but-unique output (e.g. a search result
  aggregating many sources) is legitimate, so `size_exceeded` can appear in
  `reasons` while `is_degenerate` stays `False`. The flag only flips on a *quality*
  signal (repetition / low entropy) **or** when size blows past the 60 000-char
  `HARD_CAP`. Don't treat `reasons` as a synonym for `is_degenerate`.
- **`max_size` is a floor-capped ceiling.** Passing `max_size=500_000` is silently
  clamped to `HARD_CAP`; there is no way to allow more than 60 000 chars.
- **Heuristics have minimum-input gates.** Header check needs 6+ `## ` sections,
  paragraph check needs 4+ blocks, entropy needs 20+ tokens — below those the
  detector returns "clean" rather than firing on tiny samples. Short degenerate
  snippets can slip through until enough text accumulates.
- **`salvage` is markdown-only.** Plain prose with no `## ` headers comes back
  byte-for-byte; it never edits sentences. It can also return `""` — guard for that.
- **The streaming callback is stateful.** Each `make_stream_callback(...)` call
  builds a fresh closure that remembers when it last ran `assess`; reuse one per
  generation, don't recreate it every chunk, and don't share it across streams.

## Dependencies

None. `pyproject.toml` declares an empty `dependencies` list and the source imports
only the standard library (`math`, `re`, `collections`, `json`). It has **no**
runtime coupling to any sibling package — the data flow is one-directional: the
host's LLM-call layer and streaming guard *import this package*, never the reverse.
That isolation is deliberate: a degeneracy check must stay trivially testable and
free of the rest of the system.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\dogru_mu_samet\ -q
```

---
<a id="türkçe"></a>

## Türkçe

> Türkçede "Doğru mu Samet?" — bir taslağa şüpheyle bakıp tutarlı mı yoksa çöp mü
> olduğunu söyleyen o ikinci çift göz. Bu paket de tam olarak odur: bir LLM'in az
> önce ürettiği şeye hızlı, yorumsuz bir akıl sağlığı kontrolü.

### Amaç

**Ne işe yarar.** Yerel LLM'ler çirkin ama tanıdık şekillerde bozulur: bir
paragrafı sonsuza dek tekrarlar, aynı `## başlık`'ı on kez basar, düşük entropili
bulamaca (`the the the …`) dönüşür ya da megabaytlık bir yığına kaçar. Doğru mu
Samet, çağıran tarafın bu hataları *ucuza ve başka bir model olmadan* yakalaması
için vardır — ikinci bir LLM çağrısı yok, ağ yok, embedding yok. Metni ver,
kararı al.

**Gerçekte ne yapar.** Bir string üzerinde dört saf-Python sezgiseli çalıştırır —
boyut, tekrarlanan markdown başlıkları, tekrarlanan paragraflar ve token entropisi —
ve bunları okunabilir `reasons` ile birlikte tek bir `is_degenerate` bayrağında
toplar. Ayrıca kaçan bir üretimi yarı yolda durduran durum-tutan bir streaming
callback'i ve kısmen iyi çıktının çöpe gitmemesi için tekrar eden markdown
bölümlerini ayıklayan bir `salvage` geçişi verebilir.

**Yapmadıkları**: içeriğin *doğru*, *ilgili* ya da *iyi yazılmış* olup olmadığına
karar vermez — yalnızca mekanik bozulmayı yakalar, kötü-ama-tutarlı yanıtları değil
(o, çağıran tarafın değerlendiricisinin işidir). LLM ya da ağ çağrısı yapmaz,
hiçbir bağımlılığı yoktur, tekrar denemeler veya model değişimleri hakkında bir
şey kararlaştırmaz ve `salvage` yalnızca markdown bölümlerini ayıklar — düzyazıyı
yeniden yazmaz, markdown olmayan metni düzeltmez.

### Genel API

Dört export. Hepsi senkron, hepsi bir string üzerinde saf fonksiyon.

```python
from dogru_mu_samet import assess, salvage, make_stream_callback, ContentQualityResult

# 1. Tamamlanmış çıktı üzerine karar.
result = assess(text, max_size=20_000)        # -> ContentQualityResult
if result.is_degenerate:
    print(result.summary)                     # ör. "degenerate: low_entropy (1.84 bits)"
    print(result.reasons)                     # ["low_entropy", ...]

# 2. Kısmen iyi markdown'ı kurtar (tekrarlı / boş bölümleri at).
clean = salvage(text)                         # -> str (hiçbir şey kalmazsa "")

# 3. Kaçan bir akışı erken durdur.
should_abort = make_stream_callback(max_size=200_000, check_interval=2000)
if should_abort(accumulated_so_far):          # -> bool
    cancel_generation()
```

`assess(text, max_size=20_000) -> ContentQualityResult` — `text` bir `str`, bir
`dict` (önce JSON'a dökülür) ya da herhangi bir şey (`str()`'e çevrilir) olabilir.
`max_size`, mutlak `HARD_CAP`'e (60 000 karakter) aşağı doğru kırpılır; tavanı
bunun üzerine çıkaramazsınız.

`ContentQualityResult` alanları: `size: int`, `max_size: int`,
`repetition_ratio: float` (başlık tekrar oranı), `paragraph_repetition: float`,
`token_entropy: float`, `is_degenerate: bool`, `reasons: list[str]` ve bir
`.summary` özelliği. Sebep etiketleri: `size_exceeded`, `header_repetition`,
`paragraph_repetition`, `low_entropy`.

`salvage(text) -> str` — her normalize edilmiş `## başlık`'ın ilk geçtiği yeri
tutar, boş-gövdeli bölümleri atar; markdown olmayan metni (hiç `## ` başlığı yoksa)
olduğu gibi döndürür, kurtarılacak hiçbir şey kalmazsa `""` döndürür.

`make_stream_callback(max_size=20_000, check_interval=4096) -> Callable[[str], bool]`
— durum-tutan bir closure döndürür; biriken metinle çağırın, durdurmak için `True`
alın. Boyut her çağrıda kontrol edilir (ucuz `len`); tam `assess` yalnızca her
`check_interval` karakterde bir çalışır.

### Tuzaklar

- **Tek başına boyut bozulma değildir.** Büyük-ama-özgün çıktı (ör. birçok kaynağı
  birleştiren bir arama sonucu) meşrudur; bu yüzden `size_exceeded`, `is_degenerate`
  `False` kalırken `reasons`'ta görünebilir. Bayrak yalnızca bir *kalite* sinyalinde
  (tekrar / düşük entropi) **ya da** boyut 60 000 karakterlik `HARD_CAP`'i aştığında
  döner. `reasons`'ı `is_degenerate` ile eşanlamlı sanmayın.
- **`max_size` tavanı tabanla kırpılır.** `max_size=500_000` vermek sessizce
  `HARD_CAP`'e kırpılır; 60 000 karakterden fazlasına izin vermenin bir yolu yoktur.
- **Sezgisellerin asgari-girdi kapıları var.** Başlık kontrolü 6+ `## ` bölüm,
  paragraf kontrolü 4+ blok, entropi 20+ token ister — bunların altında detektör
  küçük örnekler üzerinde tetiklenmek yerine "temiz" döndürür. Kısa bozuk parçalar
  yeterli metin birikene dek elden kaçabilir.
- **`salvage` yalnızca markdown'dır.** `## ` başlığı olmayan düz metin bayt-bayt
  geri gelir; cümleleri asla düzenlemez. Ayrıca `""` döndürebilir — buna karşı
  önlem alın.
- **Streaming callback durum tutar.** Her `make_stream_callback(...)` çağrısı,
  `assess`'i en son ne zaman çalıştırdığını hatırlayan yeni bir closure kurar;
  üretim başına bir tane kullanın, her parçada yeniden yaratmayın ve akışlar arası
  paylaşmayın.

### Bağımlılıklar

Yok. `pyproject.toml` boş bir `dependencies` listesi bildirir ve kaynak yalnızca
standart kütüphaneyi (`math`, `re`, `collections`, `json`) import eder. Hiçbir
kardeş pakete çalışma-zamanı bağı **yoktur** — veri akışı tek yönlüdür: host'un
LLM-çağrı katmanı ve streaming koruması *bu paketi import eder*, tersi değil. Bu
yalıtım bilinçlidir: bir bozulma kontrolü kolayca test edilebilir ve sistemin geri
kalanından bağımsız kalmalıdır.

### Testler

```powershell
& .\.venv\Scripts\python.exe -m pytest packages\dogru_mu_samet\ -q
```
