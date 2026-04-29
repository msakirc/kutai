"""Shopping pipeline v2 — trust site ordering, LLM-based grouping + review synthesis.

See docs/superpowers/specs/2026-04-21-shopping-trust-sites-synthesize-reviews-design.md
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field

from src.infra.logging_config import get_logger

logger = get_logger("workflows.shopping.pipeline_v2")


@dataclass
class Candidate:
    """One search result from one site, in that site's original order."""
    title: str
    site: str
    site_rank: int                 # 1-based position in site's result list
    price: float | None
    original_price: float | None
    url: str
    rating: float | None
    review_count: int | None
    review_snippets: list[str] = field(default_factory=list)
    sku: str | None = None
    category_path: str | None = None


@dataclass
class ProductGroup:
    """A cluster of Candidates judged to refer to the same product."""
    representative_title: str
    member_indices: list[int]      # indices into the Candidate list
    is_accessory_or_part: bool
    prominence: float              # sum(1 / site_rank) across members
    product_type: str = "unknown"           # authentic_product | accessory | replacement_part | knockoff | refurbished | unknown
    base_model: str = ""
    variant: str | None = None
    authenticity_confidence: float = 0.5
    matches_user_intent: bool = True
    line_id: str = ""           # LLM-emitted canonical slug; primary bucket key


@dataclass
class AspectInsight:
    """Per-aspect (camera, battery, screen, perf, build, value, software, seller, shipping)
    sentiment + mention count + verbatim quote."""
    aspect: str
    sentiment: float            # -1.0 (neg) … +1.0 (pos)
    mention_count: int
    summary: str                # one-line synthesis
    quote: str = ""             # verbatim snippet (best representative)


@dataclass
class ReviewSynthesis:
    """Synthesised pros/cons/red-flags + aspect-level insights for one ProductGroup."""
    praise: list[str]
    complaints: list[str]
    red_flags: list[str]
    insufficient_data: bool
    aspects: list[AspectInsight] = field(default_factory=list)
    overall_sentiment: float = 0.0       # -1..1
    review_volume: int = 0               # snippet count fed to LLM
    notable_quote: str = ""              # single most informative verbatim
    comparative_mentions: list[str] = field(default_factory=list)  # quotes referencing rival products


async def _fetch_products(query: str) -> list:
    """Thin wrapper around the shopping scraper fleet — mocked in tests."""
    from src.shopping.resilience.fallback_chain import get_product_with_fallback
    return await asyncio.wait_for(get_product_with_fallback(query), timeout=45)


def _attr(obj, name: str, default=None):
    """Read attribute from dataclass OR dict — scrapers mix both."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


_REVIEW_FETCH_TIMEOUT_S = 45
_MAX_SNIPPETS_PER_PRODUCT = 80
_REVIEW_PAGES = 8
_MIN_SNIPPET_CHARS = 20

# Community / forum sources — tapped when commerce listings yield thin reviews
# or when a deep-scrape pass is requested (post-pick).
_COMMUNITY_SOURCES = ("eksisozluk", "sikayetvar", "technopat", "donanimhaber")
_COMMUNITY_TOPICS_PER_SOURCE = 3
_COMMUNITY_PAGES_PER_TOPIC = 2
_COMMUNITY_TIMEOUT_S = 30
_THIN_REVIEW_THRESHOLD = 25       # below this → trigger community augment

# Boilerplate fragments common in TR e-commerce reviews — low signal.
_BOILERPLATE_RE = re.compile(
    r"^(çok\s+(iyi|güzel|hızlı|memnunum)|teşekkür(ler)?|tavsiye\s+ederim|"
    r"güvenilir\s+satıcı|kargo\s+(hızlı|özen|güzel)|paketleme\s+güzel|"
    r"sorunsuz|harika|mükemmel|süper|10\s+numara|5\s+yıldız)\.?\s*$",
    re.IGNORECASE,
)


def _filter_snippets(snippets: list[str]) -> list[str]:
    """Drop too-short / boilerplate / near-duplicate snippets. Preserves order."""
    seen_norm: set[str] = set()
    out: list[str] = []
    for s in snippets:
        s = (s or "").strip()
        if len(s) < _MIN_SNIPPET_CHARS:
            continue
        if _BOILERPLATE_RE.match(s):
            continue
        # near-dup detection: first 60 chars normalized
        norm = re.sub(r"\s+", " ", s.lower())[:60]
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        out.append(s)
    return out


async def _fetch_community_reviews(
    query: str,
    *,
    sources: tuple[str, ...] = _COMMUNITY_SOURCES,
    topics_per_source: int = _COMMUNITY_TOPICS_PER_SOURCE,
    pages_per_topic: int = _COMMUNITY_PAGES_PER_TOPIC,
) -> list[str]:
    """Search forum/community scrapers for *query*, harvest review-style snippets.

    Used to augment thin product-page review piles (e.g. S25 Ultra had only 5
    snippets from commerce listings). Returns deduped, filtered snippets.
    Failures per source/topic swallowed — best-effort.
    """
    if not query:
        return []
    import logging as _logging
    from src.shopping.scrapers import get_scraper

    # Quiet scraper-level ERROR logs during best-effort community fetch.
    # Search failures are expected (rate limits, 404s, budget) and shouldn't
    # surface as Telegram alerts via the global error handler.
    _scraper_loggers = [
        _logging.getLogger(f"shopping.scrapers.{s}") for s in sources
    ]
    _orig_levels = [(lg, lg.level) for lg in _scraper_loggers]
    for lg in _scraper_loggers:
        if lg.level < _logging.CRITICAL:
            lg.setLevel(_logging.CRITICAL)

    async def _harvest_one_source(name: str) -> list[str]:
        cls = get_scraper(name)
        if cls is None:
            return []
        try:
            scraper = cls()
        except Exception:
            return []
        try:
            topics = await asyncio.wait_for(
                scraper.search(query, max_results=topics_per_source),
                timeout=_COMMUNITY_TIMEOUT_S,
            )
        except Exception as exc:
            logger.debug("community search failed", source=name, err=str(exc))
            return []

        async def _harvest_topic(topic) -> list[str]:
            url = _attr(topic, "url") or ""
            if not url:
                return []
            try:
                reviews = await asyncio.wait_for(
                    scraper.get_reviews(url, max_pages=pages_per_topic),
                    timeout=_COMMUNITY_TIMEOUT_S,
                )
            except Exception:
                return []
            out: list[str] = []
            for r in reviews or []:
                if not isinstance(r, dict):
                    continue
                txt = str(r.get("text") or r.get("content") or r.get("comment") or "").strip()
                if txt:
                    out.append(txt)
            return out

        results = await asyncio.gather(
            *[_harvest_topic(t) for t in (topics or [])],
            return_exceptions=True,
        )
        flat: list[str] = []
        for r in results:
            if isinstance(r, list):
                flat.extend(r)
        return flat

    try:
        per_source = await asyncio.gather(
            *[_harvest_one_source(s) for s in sources],
            return_exceptions=True,
        )
    finally:
        for lg, lvl in _orig_levels:
            lg.setLevel(lvl)
    pile: list[str] = []
    for r in per_source:
        if isinstance(r, list):
            pile.extend(r)
    return _filter_snippets(pile)


async def _fetch_reviews(products: list) -> dict[str, list[str]]:
    """For each product, fetch review snippets via its scraper. Keyed by URL.

    Concurrent; individual failures/timeouts yield empty list, never raise.
    Separate from ``_fetch_products`` so tests can mock independently.
    """
    if not products:
        return {}
    from src.shopping.scrapers import get_scraper

    async def _one(p) -> tuple[str, list[str]]:
        url = str(_attr(p, "url") or "")
        source = str(_attr(p, "source") or _attr(p, "site") or "")
        if not url or not source:
            return url, []
        scraper_cls = get_scraper(source)
        if scraper_cls is None:
            return url, []
        try:
            scraper = scraper_cls()
            reviews = await asyncio.wait_for(
                scraper.get_reviews(url, max_pages=_REVIEW_PAGES),
                timeout=_REVIEW_FETCH_TIMEOUT_S,
            )
        except Exception as exc:
            logger.debug("review fetch failed", source=source, url=url, err=str(exc))
            return url, []
        snippets: list[str] = []
        for r in reviews or []:
            if not isinstance(r, dict):
                continue
            text = r.get("text") or r.get("content") or r.get("comment") or ""
            text = str(text).strip()
            if text:
                snippets.append(text)
        snippets = _filter_snippets(snippets)
        return url, snippets[:_MAX_SNIPPETS_PER_PRODUCT]

    tasks = [asyncio.create_task(_one(p)) for p in products]
    out: dict[str, list[str]] = {}
    for coro in asyncio.as_completed(tasks):
        try:
            url, snips = await coro
        except Exception:
            continue
        if url and snips:
            out[url] = snips
    return out


async def step_resolve(query: str, per_site_n: int) -> list[Candidate]:
    """Fetch scraper results, keep top-N per site in site order, no filtering."""
    logger.info("step_resolve start", query=query[:80], per_site_n=per_site_n)
    try:
        raw = await _fetch_products(query)
    except Exception as exc:
        logger.warning("resolve fetch failed: %s", exc)
        raw = []

    # Group by site preserving order of first appearance
    by_site: "OrderedDict[str, list]" = OrderedDict()
    for p in raw or []:
        site = _attr(p, "source") or _attr(p, "site") or "unknown"
        by_site.setdefault(site, []).append(p)

    # Keep only top-N per site before fetching reviews (reviews are expensive)
    kept_products: list = []
    site_rank_map: list[tuple[str, int]] = []  # parallel to kept_products
    for site, products in by_site.items():
        for rank, p in enumerate(products[:per_site_n], start=1):
            kept_products.append(p)
            site_rank_map.append((site, rank))

    # Fetch review snippets only for products that don't already carry them
    # (tests inject them inline via the fixture products).
    needs_fetch = [p for p in kept_products if not _attr(p, "review_snippets")]
    try:
        reviews_map = await _fetch_reviews(needs_fetch) if needs_fetch else {}
    except Exception as exc:
        logger.warning("review fetch stage failed, continuing without snippets: %s", exc)
        reviews_map = {}

    cands: list[Candidate] = []
    for p, (site, rank) in zip(kept_products, site_rank_map):
        url = str(_attr(p, "url") or "")
        inline = list(_attr(p, "review_snippets") or [])
        snippets = inline or reviews_map.get(url, [])
        cands.append(
            Candidate(
                title=str(_attr(p, "name") or ""),
                site=site,
                site_rank=rank,
                price=(_attr(p, "discounted_price") or _attr(p, "original_price") or _attr(p, "price")),
                original_price=_attr(p, "original_price"),
                url=url,
                rating=_attr(p, "rating"),
                review_count=_attr(p, "review_count"),
                review_snippets=list(snippets),
                sku=_attr(p, "sku"),
                category_path=_attr(p, "category_path"),
            )
        )
    logger.info(
        "step_resolve done",
        candidate_count=len(cands),
        with_snippets=sum(1 for c in cands if c.review_snippets),
    )
    return cands


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` fences some models emit."""
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    return m.group(1) if m else text


def _per_site_top1_fallback(candidates: list[Candidate]) -> list[ProductGroup]:
    """Grouping fallback: one group per site's rank-1 candidate."""
    seen_sites: set[str] = set()
    groups: list[ProductGroup] = []
    for idx, c in enumerate(candidates):
        if c.site_rank != 1:
            continue
        if c.site in seen_sites:
            continue
        seen_sites.add(c.site)
        groups.append(
            ProductGroup(
                representative_title=c.title,
                member_indices=[idx],
                is_accessory_or_part=False,
                prominence=1.0,
            )
        )
    return groups


async def _grouping_llm_call(prompt: str) -> dict:
    """Dispatch the grouping prompt. Returns the dispatcher response dict.

    Split out so tests can patch this one function instead of the dispatcher.
    """
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    dispatcher = get_dispatcher()
    return await dispatcher.request(
        category=CallCategory.MAIN_WORK,
        task="shopping_grouper",
        agent_type="shopping_pipeline_v2",
        difficulty=3,
        # Grouping is structured-JSON transformation, not a reasoning task.
        # Thinking-on wastes thousands of invisible reasoning tokens per call
        # and bursts past the dispatch timeout on small local models.
        needs_thinking=False,
        messages=[
            {"role": "system", "content": "You output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )


async def _llm_group_residuals(candidates: list[Candidate], query: str) -> list[ProductGroup]:
    """LLM-based grouping for sku-less residual candidates.

    Falls back to one group per site's rank-1 candidate on any LLM or parse error.
    """
    from src.workflows.shopping.prompts_v2 import GROUPING_PROMPT

    # Compact JSON view for the LLM — include sku and category_path for context
    view = [
        {"index": i, "title": c.title, "site": c.site, "price": c.price,
         "sku": c.sku, "category_path": c.category_path}
        for i, c in enumerate(candidates)
    ]
    prompt = GROUPING_PROMPT.format(candidates_json=json.dumps(view, ensure_ascii=False))

    try:
        resp = await _grouping_llm_call(prompt)
    except Exception as exc:
        logger.warning("grouping LLM failed, using per-site fallback: %s", exc)
        return _per_site_top1_fallback(candidates)

    content = _strip_json_fences(str(resp.get("content", "")).strip())
    try:
        parsed = json.loads(content)
        raw_groups = parsed.get("groups", [])
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning("grouping LLM output not parseable, using fallback: %s", exc)
        return _per_site_top1_fallback(candidates)

    groups: list[ProductGroup] = []
    n = len(candidates)
    for g in raw_groups:
        members = [i for i in (g.get("member_indices") or []) if isinstance(i, int) and 0 <= i < n]
        if not members:
            continue
        title = str(g.get("representative_title") or candidates[members[0]].title)
        is_acc = bool(g.get("is_accessory_or_part"))
        prominence = sum(1.0 / candidates[i].site_rank for i in members)
        groups.append(
            ProductGroup(
                representative_title=title,
                member_indices=members,
                is_accessory_or_part=is_acc,
                prominence=prominence,
            )
        )

    if not groups:
        logger.warning("grouping returned no valid groups, using fallback")
        return _per_site_top1_fallback(candidates)

    logger.info(
        "_llm_group_residuals done",
        group_count=len(groups),
        accessory_drop_count=sum(1 for g in groups if g.is_accessory_or_part),
    )
    return groups


async def step_group(candidates: list[Candidate], query: str = "") -> list[ProductGroup]:
    """SKU-first deterministic bucket, then LLM-group the residuals."""
    if not candidates:
        return []

    sku_buckets: dict[str, list[int]] = {}
    unbucketed: list[int] = []
    for i, c in enumerate(candidates):
        if c.sku:
            sku_buckets.setdefault(c.sku, []).append(i)
        else:
            unbucketed.append(i)

    groups: list[ProductGroup] = []
    for _sku, indices in sku_buckets.items():
        first = candidates[indices[0]]
        prominence = sum(1.0 / candidates[i].site_rank for i in indices)
        groups.append(ProductGroup(
            representative_title=first.title,
            member_indices=indices,
            is_accessory_or_part=False,
            prominence=prominence,
        ))

    if unbucketed:
        residual_cands = [candidates[i] for i in unbucketed]
        residual_groups = await _llm_group_residuals(residual_cands, query)
        for g in residual_groups:
            g.member_indices = [unbucketed[j] for j in g.member_indices]
            groups.append(g)

    logger.info(
        "step_group done",
        group_count=len(groups),
        sku_bucket_count=len(sku_buckets),
        residual_count=len(unbucketed),
    )
    return groups


async def _synthesis_llm_call(prompt: str) -> dict:
    """Dispatch the synthesis prompt. Returns the dispatcher response dict."""
    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    dispatcher = get_dispatcher()
    # Estimate input size — review prompts can balloon to ~25K tokens at
    # 80-snippet cap × 5 listings. Floor the load to avoid getting routed
    # onto an 8K-ctx model that would truncate the snippet pile.
    char_count = len(prompt)
    est_tokens = char_count // 3
    return await dispatcher.request(
        category=CallCategory.MAIN_WORK,
        task="shopping_review_synthesizer",
        agent_type="shopping_pipeline_v2",
        difficulty=6,
        # Synthesis is structured JSON extraction over review snippets —
        # no chain-of-thought needed. Suppress reasoning to stay under the
        # dispatch timeout when a thinking model happens to be resident.
        needs_thinking=False,
        estimated_output_tokens=1200,
        min_context=max(8192, est_tokens + 2048),
        messages=[
            {"role": "system", "content": "You output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )


def _insufficient() -> ReviewSynthesis:
    return ReviewSynthesis(
        praise=[], complaints=[], red_flags=[], insufficient_data=True,
        aspects=[], overall_sentiment=0.0, review_volume=0,
        notable_quote="", comparative_mentions=[],
    )


_ASPECT_ICONS = {
    "kamera": "📷", "pil": "🔋", "ekran": "📱", "performans": "⚡",
    "yapım_kalitesi": "🛠", "yazılım": "💾", "fiyat": "💰", "satıcı": "🏬",
    "kargo": "📦", "ses": "🔊", "şarj": "🔌", "güncellemeler": "🔄",
    "oyun": "🎮", "boyut": "📐", "ergonomi": "🤲", "ısınma": "🌡",
}


def _sentiment_bar(s: float, width: int = 6) -> str:
    """Render sentiment as a small bar. Negative = ░, neutral = ▒, positive = █."""
    s = max(-1.0, min(1.0, s))
    if s >= 0.6:
        return "█" * width
    if s >= 0.2:
        return "█" * (width - 1) + "▒"
    if s >= -0.2:
        return "▒" * width
    if s >= -0.6:
        return "░" * (width - 1) + "▒"
    return "░" * width


def _sentiment_label(s: float) -> str:
    if s >= 0.6:
        return "çok pozitif"
    if s >= 0.2:
        return "pozitif"
    if s >= -0.2:
        return "karışık"
    if s >= -0.6:
        return "negatif"
    return "çok negatif"


async def step_synthesize_reviews(
    group: ProductGroup,
    candidates: list[Candidate],
    *,
    deep_scrape: bool = False,
    community_query: str | None = None,
) -> ReviewSynthesis:
    """LLM-based review synthesis for one group.

    *deep_scrape=True* always taps community sources (eksisozluk / sikayetvar /
    forums) for the line — used after user picks a variant. When False, taps
    community only as fallback when commerce snippets are thin.

    *community_query* defaults to group.representative_title; pass a cleaner
    query (e.g. base_model only) for better forum search hits.
    """
    from src.workflows.shopping.prompts_v2 import SYNTHESIS_PROMPT

    # Gather snippets from all group members. Commerce snippets already filtered
    # in _fetch_reviews; don't re-filter (would drop short test fixtures).
    snippets: list[str] = []
    for idx in group.member_indices:
        if 0 <= idx < len(candidates):
            snippets.extend(s for s in candidates[idx].review_snippets if s and s.strip())

    # Augment with community/forum sources only on explicit deep_scrape (post-pick).
    # Auto-augment on thin commerce piles disabled — community scrapers hit strict
    # rate limits / daily budgets and surfaced ERROR-level alerts during compare-all.
    # Deep_scrape runs after user commits to a line, where extra latency + risk is OK.
    if deep_scrape:
        cq = (community_query or group.base_model or group.representative_title or "").strip()
        if cq:
            try:
                community = await _fetch_community_reviews(cq)
            except Exception as exc:
                logger.warning("community augment failed for %s: %s", cq, exc)
                community = []
            if community:
                # Merge while keeping order + dedup against existing snippets
                seen = {re.sub(r"\s+", " ", s.lower())[:60] for s in snippets}
                for s in community:
                    norm = re.sub(r"\s+", " ", s.lower())[:60]
                    if norm in seen:
                        continue
                    seen.add(norm)
                    snippets.append(s)
                logger.info(
                    "synthesize community-augmented",
                    line=cq, before=len(snippets) - len(community), added=len(community),
                )

    if not snippets:
        logger.info(
            "synthesize short-circuit (no snippets)",
            representative_title=group.representative_title,
        )
        return _insufficient()

    # Cap total to avoid context blow-up when community pile is huge
    cap = _MAX_SNIPPETS_PER_PRODUCT * (3 if deep_scrape else 2)
    if len(snippets) > cap:
        snippets = snippets[:cap]

    prompt = SYNTHESIS_PROMPT.format(
        representative_title=group.representative_title,
        review_snippets_json=json.dumps(snippets, ensure_ascii=False),
    )

    try:
        resp = await _synthesis_llm_call(prompt)
    except Exception as exc:
        logger.warning("synthesis LLM failed: %s", exc)
        return _insufficient()

    content = _strip_json_fences(str(resp.get("content", "")).strip())
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("synthesis LLM output not parseable: %s", exc)
        return _insufficient()

    def _take_list(key: str, n: int = 5) -> list[str]:
        v = parsed.get(key) or []
        return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:n]

    aspects: list[AspectInsight] = []
    for a in (parsed.get("aspects") or [])[:8]:
        if not isinstance(a, dict):
            continue
        try:
            aspects.append(AspectInsight(
                aspect=str(a.get("aspect", "")).strip().lower(),
                sentiment=max(-1.0, min(1.0, float(a.get("sentiment", 0.0)))),
                mention_count=int(a.get("mention_count", 0)),
                summary=str(a.get("summary", "")).strip(),
                quote=str(a.get("quote", "")).strip(),
            ))
        except (TypeError, ValueError):
            continue
    aspects.sort(key=lambda a: a.mention_count, reverse=True)

    try:
        overall = max(-1.0, min(1.0, float(parsed.get("overall_sentiment", 0.0))))
    except (TypeError, ValueError):
        overall = 0.0

    # Dedup comparative_mentions — LLM occasionally repeats the same snippet
    raw_comp = _take_list("comparative_mentions", n=8)
    seen_comp: set[str] = set()
    deduped_comp: list[str] = []
    for q in raw_comp:
        norm = re.sub(r"\s+", " ", q.lower())[:80]
        if norm in seen_comp:
            continue
        seen_comp.add(norm)
        deduped_comp.append(q)
        if len(deduped_comp) >= 3:
            break

    syn = ReviewSynthesis(
        praise=_take_list("praise"),
        complaints=_take_list("complaints"),
        red_flags=_take_list("red_flags"),
        insufficient_data=bool(parsed.get("insufficient_data", False)),
        aspects=aspects,
        overall_sentiment=overall,
        review_volume=len(snippets),
        notable_quote=str(parsed.get("notable_quote", "")).strip()[:240],
        comparative_mentions=deduped_comp,
    )
    logger.info(
        "step_synthesize done",
        representative_title=group.representative_title,
        snippet_count=len(snippets),
        insufficient=syn.insufficient_data,
    )
    return syn


# Variant suffixes that disqualify a result unless the user explicitly asked
# for them. Matched case-insensitively against the representative title.
_VARIANT_SUFFIXES = {"fe", "plus", "ultra", "pro", "mini", "lite", "edge"}


def _query_match_score(title: str, query: str) -> float:
    """Jaccard-based query match: how many query tokens appear in the title.

    Also applies a penalty for variant tokens (FE, Plus, Ultra …) present in
    the title but absent from the query — prevents budget/premium variants
    from ranking above the exact requested model.

    Returns a float in (0, 1].  Never 0 so pure-prominence still breaks ties.
    """
    def _tokens(s: str) -> set[str]:
        return {t.lower() for t in re.split(r"[\s\-_/]+", s) if t}

    q_tok = _tokens(query)
    t_tok = _tokens(title)

    if not q_tok:
        return 1.0

    # Fraction of query tokens present in the title
    hit = len(q_tok & t_tok) / len(q_tok)

    # Penalty: variant tokens in title that aren't in query → multiply down
    unsolicited_variants = (t_tok & _VARIANT_SUFFIXES) - q_tok
    penalty = 0.5 * len(unsolicited_variants)    # 0.5 per unsolicited suffix
    score = max(0.05, hit - penalty)
    return score


def select_groups(
    groups: list[ProductGroup], max_groups: int, query: str = "",
) -> list[ProductGroup]:
    """Filter accessories, rank by prominence × query-match, apply 50%-of-top rule."""
    non_acc = [g for g in groups if not g.is_accessory_or_part]
    if not non_acc:
        return []

    def _score(g: ProductGroup) -> float:
        qm = _query_match_score(g.representative_title, query) if query else 1.0
        return g.prominence * qm

    non_acc.sort(key=_score, reverse=True)
    top_score = _score(non_acc[0])
    kept: list[ProductGroup] = [non_acc[0]]
    for g in non_acc[1:max_groups]:
        if _score(g) >= top_score * 0.5:
            kept.append(g)
        else:
            break
    return kept


def _fmt_price_tr(value: float | None) -> str:
    if value is None:
        return "—"
    s = f"{value:,.0f}"
    return s.replace(",", ".")


def _site_label(site: str) -> str:
    mapping = {
        "trendyol": "Trendyol", "hepsiburada": "Hepsiburada",
        "amazon_tr": "Amazon.tr", "akakce": "Akakçe", "n11": "n11",
        "gittigidiyor": "GittiGidiyor", "epey": "Epey",
        "teknosa": "Teknosa", "vatan": "Vatan",
    }
    return mapping.get(site, site.title())


def format_group_card(
    group: ProductGroup,
    synthesis: ReviewSynthesis,
    candidates: list[Candidate],
    community_counts: dict[str, int] | None = None,
) -> str:
    """Reviews-first Telegram markdown for one product group."""
    members = [candidates[i] for i in group.member_indices if 0 <= i < len(candidates)]

    # Aggregate ratings — only count members with actual review backing.
    # Default-5.0/0-review listings (price aggregators, never-rated SKUs) pollute
    # the weighted average and surface as misleading "⭐ 5.0 (0)" in the UI.
    rated = [m for m in members if m.rating is not None and (m.review_count or 0) > 0]
    total_reviews = sum(m.review_count or 0 for m in members)
    sources_with_reviews = sum(1 for m in members if (m.review_count or 0) > 0)
    if rated:
        weights = [m.review_count for m in rated]
        wsum = sum(weights)
        agg_rating = sum(m.rating * w for m, w in zip(rated, weights)) / wsum if wsum else None
    else:
        agg_rating = None

    lines: list[str] = []
    head = f"*{group.representative_title}*"
    if agg_rating is not None:
        rc_parts = []
        if total_reviews:
            rc_parts.append(f"{total_reviews:,}".replace(",", ".") + " değerlendirme")
        if sources_with_reviews >= 2:
            rc_parts.append(f"{sources_with_reviews} kaynak")
        rc = f" ({', '.join(rc_parts)})" if rc_parts else ""
        head += f" ⭐ {agg_rating:.1f}/5{rc}"
    elif total_reviews:
        head += f" ({total_reviews:,} değerlendirme)".replace(",", ".")
    lines.append(head)
    lines.append("")

    if not synthesis.insufficient_data:
        # Aspect-rich block (preferred when LLM emitted aspects)
        if synthesis.aspects:
            vol = synthesis.review_volume or 0
            vol_note = f" ({vol} yorum analiz edildi)" if vol else ""
            lines.append(f"📊 *İnceleme analizi*{vol_note}")
            for a in synthesis.aspects:
                if not a.aspect or a.mention_count <= 0:
                    continue
                icon = _ASPECT_ICONS.get(a.aspect, "•")
                bar = _sentiment_bar(a.sentiment)
                label = _sentiment_label(a.sentiment)
                aspect_title = a.aspect.replace("_", " ").capitalize()
                pct = f" ({a.mention_count / vol * 100:.0f}%)" if vol else ""
                line = (
                    f"{icon} *{aspect_title}* `{bar}` {label} · "
                    f"{a.mention_count}×{pct} — {a.summary}"
                ).rstrip(" —")
                lines.append(line)
                if a.quote:
                    lines.append(f"   _\"{a.quote}\"_")
            lines.append("")
        # Fall back / supplement with terse praise/complaints if aspects sparse
        elif synthesis.praise or synthesis.complaints:
            if synthesis.praise:
                lines.append("👍 Kullanıcılar beğeniyor:")
                lines.extend(f"• {p}" for p in synthesis.praise)
                lines.append("")
            if synthesis.complaints:
                lines.append("👎 Şikayetler:")
                lines.extend(f"• {c}" for c in synthesis.complaints)
                lines.append("")

        if synthesis.red_flags:
            lines.append("⚠️ *Dikkat:*")
            lines.extend(f"• {r}" for r in synthesis.red_flags)
            lines.append("")

        if synthesis.comparative_mentions:
            lines.append("⚔️ *Rakiplerle karşılaştırma:*")
            for q in synthesis.comparative_mentions:
                lines.append(f"   _\"{q}\"_")
            lines.append("")

        if synthesis.notable_quote:
            lines.append(f"💬 _\"{synthesis.notable_quote}\"_")
            lines.append("")

    # Per-site row: price + rating + review_count (transparency on source weight)
    seen_sites: set[str] = set()
    price_rows: list[tuple[str, float | None, float | None, int, str]] = []
    for m in members:
        if m.site in seen_sites:
            continue
        seen_sites.add(m.site)
        price_rows.append((
            _site_label(m.site), m.price, m.rating, m.review_count or 0, m.url,
        ))
    price_rows.sort(key=lambda r: (r[1] is None, r[1] or 0))
    if price_rows:
        # Highlight site with most review weight (most credible source)
        top_site = max(price_rows, key=lambda r: r[3])[0] if any(r[3] for r in price_rows) else None
        lines.append("💰 *Fiyatlar:*")
        for label, price, m_rating, m_reviews, _url in price_rows:
            # Only show rating when it's review-backed; bare 5.0/0 is noise.
            star_part = ""
            if m_rating is not None and m_reviews:
                rc = f"/{m_reviews:,}".replace(",", ".")
                star_part = f"  ⭐ {m_rating:.1f}{rc}"
            elif m_reviews:
                star_part = f"  ({m_reviews:,} değerl.)".replace(",", ".")
            badge = " 🏆" if label == top_site else ""
            if price is None:
                lines.append(f"• {label}{badge} — stokta yok{star_part}")
            else:
                lines.append(f"• {label}{badge} — {_fmt_price_tr(price)} TL{star_part}")
        lines.append("")

    community_counts = community_counts or {}
    if community_counts:
        bits = ", ".join(f"{k} ({v} konu)" for k, v in community_counts.items())
        lines.append(f"💬 Topluluk: {bits}")
        lines.append("")

    if synthesis.insufficient_data:
        lines.append("⚠️ Yeterli inceleme verisi yok")

    return "\n".join(lines).rstrip() + "\n"


def format_response(cards: list[str]) -> str:
    """Join per-group cards with a blank-line separator."""
    return "\n".join(c.rstrip() for c in cards if c).strip() + "\n"


def step_compare_all(
    groups: list[ProductGroup],
    candidates: list[Candidate],
    base_label: str,
) -> str:
    """Render a compact variant-comparison markdown table."""
    lines: list[str] = [f"*{base_label} — Karşılaştırma*", "─" * 20]
    # Show full representative_title per row when base_model differs across groups
    # (e.g. "S25" query returns both Galaxy S25 and Galaxy S25 FE) — otherwise the
    # variant suffix alone ("256GB Siyah") is ambiguous across product lines.
    base_models = {(g.base_model or "").lower() for g in groups}
    disambiguate = len(base_models - {""}) > 1
    for g in groups:
        members = [candidates[i] for i in g.member_indices if 0 <= i < len(candidates)]
        prices = [m.price for m in members if m.price is not None]
        pmin = min(prices) if prices else None
        pmax = max(prices) if prices else None
        # Weighted rating, review-backed only — drops misleading "⭐ 5.0 (0)" rows
        rated = [m for m in members if m.rating is not None and (m.review_count or 0) > 0]
        review_total = sum(m.review_count or 0 for m in members)
        if rated:
            wsum = sum(m.review_count for m in rated)
            agg_rating = sum(m.rating * m.review_count for m in rated) / wsum if wsum else None
        else:
            agg_rating = None

        if disambiguate:
            variant_label = g.representative_title or (
                f"{g.base_model} {g.variant}" if g.variant else g.base_model or "Vanilla"
            )
        else:
            variant_label = g.variant or "Vanilla"

        if pmin is not None and pmax is not None and pmin == pmax:
            price_str = f"{_fmt_price_tr(pmin)} TL"
        elif pmin is not None:
            price_str = f"{_fmt_price_tr(pmin)}–{_fmt_price_tr(pmax)} TL"
        else:
            price_str = "fiyat yok"

        if agg_rating is not None and review_total:
            rt = f"{review_total:,}".replace(",", ".")
            rating_str = f" ⭐ {agg_rating:.1f} ({rt})"
        else:
            rating_str = ""
        lines.append(f"• *{variant_label}* — {price_str}{rating_str}")
    lines.append("─" * 20)
    lines.append("Seçmek için sorunuzu daraltın.")
    return "\n".join(lines) + "\n"


# ── Workflow step handlers (task-shaped I/O) ────────────────────────────────

def _parse_context(task: dict) -> dict:
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            return json.loads(ctx)
        except (json.JSONDecodeError, ValueError):
            return {}
    return ctx or {}


def _candidates_to_json(cands: list[Candidate]) -> list[dict]:
    return [
        {
            "title": c.title, "site": c.site, "site_rank": c.site_rank,
            "price": c.price, "original_price": c.original_price,
            "url": c.url, "rating": c.rating, "review_count": c.review_count,
            "review_snippets": c.review_snippets,
            "sku": c.sku, "category_path": c.category_path,
        }
        for c in cands
    ]


def _candidates_from_json(items: list[dict]) -> list[Candidate]:
    return [
        Candidate(
            title=i.get("title", ""), site=i.get("site", ""),
            site_rank=int(i.get("site_rank", 1)),
            price=i.get("price"), original_price=i.get("original_price"),
            url=i.get("url", ""), rating=i.get("rating"),
            review_count=i.get("review_count"),
            review_snippets=list(i.get("review_snippets") or []),
            sku=i.get("sku"), category_path=i.get("category_path"),
        )
        for i in items
    ]


async def _read_artifacts(mission_id: int, keys: list[str]) -> dict:
    """Reuse v1's artifact reader — same table, same semantics."""
    from src.workflows.shopping.pipeline import _read_artifacts as _v1_read
    return await _v1_read(mission_id, keys)


async def _handler_resolve_candidates(task: dict, artifacts: dict, ctx: dict) -> dict:
    query = ""
    for key in ("clarified_query", "user_query"):
        raw = artifacts.get(key, "")
        if not raw:
            continue
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
                query = parsed.get("clarified_query") or parsed.get("query") or parsed.get("user_query", "")
                if query:
                    break
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(raw, str) and raw.strip():
            query = raw.strip()
            break
    if not query:
        query = task.get("description", "")
    per_site_n = int(ctx.get("per_site_n", 3))
    cands = await step_resolve(query, per_site_n=per_site_n)
    return {
        "query": query,
        "candidates": _candidates_to_json(cands),
        "escalation_needed": len(cands) == 0,
    }


async def _handler_group_and_synthesize(task: dict, artifacts: dict, ctx: dict) -> dict:
    payload_raw = artifacts.get("search_results", "{}")
    payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
    cands = _candidates_from_json(payload.get("candidates", []))
    if not cands:
        return {"cards": [], "escalation_needed": True}
    query = payload.get("query", "")
    groups = await step_group(cands)
    max_groups = int(ctx.get("max_groups", 2))
    kept = select_groups(groups, max_groups=max_groups, query=query)
    community_counts = payload.get("community_counts") or {}
    cards: list[str] = []
    for g in kept:
        syn = await step_synthesize_reviews(g, cands)
        cards.append(format_group_card(g, syn, cands, community_counts=community_counts))
    return {"cards": cards, "escalation_needed": False}


async def _handler_format_response(task: dict, artifacts: dict, ctx: dict) -> dict:
    # Workflow emits the synth output as `synth_result`; legacy code looked for
    # `grouped_synth` and silently produced "Sonuç bulunamadı". Read both.
    raw = artifacts.get("synth_result") or artifacts.get("grouped_synth") or "{}"
    payload = json.loads(raw) if isinstance(raw, str) else raw
    cards = payload.get("cards", [])
    if not cards:
        return {"formatted_text": "🔍 Sonuç bulunamadı.", "escalation": True}
    return {"formatted_text": format_response(cards), "escalation": False}


def _group_to_dict(g: ProductGroup) -> dict:
    return {
        "representative_title": g.representative_title,
        "member_indices": g.member_indices,
        "is_accessory_or_part": g.is_accessory_or_part,
        "prominence": g.prominence,
        "product_type": g.product_type,
        "base_model": g.base_model,
        "variant": g.variant,
        "authenticity_confidence": g.authenticity_confidence,
        "matches_user_intent": g.matches_user_intent,
        "line_id": g.line_id,
    }


def _group_from_dict(d: dict) -> ProductGroup:
    return ProductGroup(
        representative_title=d["representative_title"],
        member_indices=list(d["member_indices"]),
        is_accessory_or_part=bool(d.get("is_accessory_or_part", False)),
        prominence=float(d.get("prominence", 0.0)),
        product_type=str(d.get("product_type", "unknown")),
        base_model=str(d.get("base_model", "")),
        variant=d.get("variant"),
        authenticity_confidence=float(d.get("authenticity_confidence", 0.5)),
        matches_user_intent=bool(d.get("matches_user_intent", True)),
        line_id=str(d.get("line_id", "")),
    )


async def _handler_group_label_filter_gate(
    task: dict, artifacts: dict, ctx: dict,
) -> dict:
    from src.workflows.shopping.labels import step_label
    from src.workflows.shopping.variant_gate import step_filter, step_variant_gate

    raw = artifacts.get("search_results", "{}")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    cands = _candidates_from_json(payload.get("candidates", []))
    query = payload.get("query", "")
    if not cands:
        return {"gate": {"kind": "escalation", "reason": "no_candidates"},
                "candidates": [], "query": query}

    groups = await step_group(cands, query=query)
    groups = await step_label(groups, cands, query=query)
    survivors = step_filter(groups)
    gate = step_variant_gate(survivors, groups, query=query)

    out: dict = {
        "gate": {"kind": gate["kind"]},
        "candidates": _candidates_to_json(cands),
        "query": query,
    }
    if gate["kind"] == "chosen":
        out["chosen_group"] = _group_to_dict(gate["group"])
    elif gate["kind"] == "clarify":
        out["clarify_options"] = gate["options"]
        out["clarify_payloads"] = {
            str(gid): _group_to_dict(g) for gid, g in gate["payloads"].items()
        }
        bases = {s.base_model for s in survivors if s.base_model}
        if len(bases) == 1:
            out["base_label"] = next(iter(bases))
        else:
            out["base_label"] = query.strip().title() or (survivors[0].base_model if survivors else "")
    elif gate["kind"] == "escalation":
        out["gate"]["reason"] = gate.get("reason", "unknown")
    return out


async def _handler_synth_one(task: dict, artifacts: dict, ctx: dict) -> dict:
    raw = artifacts.get("gate_result", "{}")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    cands = _candidates_from_json(payload.get("candidates", []))

    gate_kind = payload.get("gate", {}).get("kind")
    group: ProductGroup | None = None
    deep = False  # default: only synth pre-fetched commerce reviews

    if gate_kind == "chosen":
        group = _group_from_dict(payload["chosen_group"])
    else:
        # Post-clarify path: user picked a variant, look up that group in payloads.
        # Trigger deep_scrape — pick = user committed, worth the extra latency to
        # tap eksisozluk/sikayetvar/forums for richer review pile.
        choice_raw = artifacts.get("clarify_choice", "{}")
        choice = json.loads(choice_raw) if isinstance(choice_raw, str) else (choice_raw or {})
        if choice.get("kind") == "variant":
            gid = choice.get("group_id")
            payloads = payload.get("clarify_payloads", {}) or {}
            picked = payloads.get(str(gid)) if gid is not None else None
            if picked:
                group = _group_from_dict(picked)
                deep = True

    if group is None:
        logger.warning(
            "synth_one: no group resolved | gate_kind=%s clarify=%s",
            gate_kind, artifacts.get("clarify_choice", "")[:120],
        )
        return {"cards": [], "escalation_needed": True}

    syn = await step_synthesize_reviews(group, cands, deep_scrape=deep)
    cards = [format_group_card(group, syn, cands)]
    return {"cards": cards, "escalation_needed": False}


async def _handler_format_compare(task: dict, artifacts: dict, ctx: dict) -> dict:
    """Category-style compare: review-synth every line, stack full cards + price summary header.

    Treats the clarified lines as a small category and presents each with full
    pros/cons/red-flags/prices so the user can compare across products, not just
    read a terse price table.
    """
    raw = artifacts.get("gate_result", "{}")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    payloads = payload.get("clarify_payloads", {}) or {}
    base_label = payload.get("base_label") or "Ürün"
    cands = _candidates_from_json(payload.get("candidates", []))
    groups = [_group_from_dict(v) for v in payloads.values()]

    if not groups:
        return {"formatted_text": f"*{base_label} — Karşılaştırma*\n\nVeri yok.\n", "escalation": True}

    header = step_compare_all(groups, cands, base_label=base_label)

    # Synthesize per line concurrently — one LLM call per group
    synths = await asyncio.gather(
        *[step_synthesize_reviews(g, cands) for g in groups],
        return_exceptions=True,
    )
    cards: list[str] = []
    for g, syn in zip(groups, synths):
        if isinstance(syn, BaseException):
            logger.warning("synth failed for %s: %s", g.representative_title, syn)
            syn = _insufficient()
        cards.append(format_group_card(g, syn, cands))

    separator = "\n" + ("─" * 20) + "\n"
    body = separator.join(cards)
    text = f"{header}\n{body}"
    return {"formatted_text": text, "escalation": False}


_STEP_HANDLERS_V2 = {
    "resolve_candidates": _handler_resolve_candidates,
    "group_label_filter_gate": _handler_group_label_filter_gate,
    "group_and_synthesize": _handler_group_and_synthesize,
    "format_response": _handler_format_response,
}

_STEP_HANDLERS_V2.update({
    "synth_one": _handler_synth_one,
    "format_compare": _handler_format_compare,
})


class ShoppingPipelineV2:
    """Dispatch class — same contract as v1 ShoppingPipeline."""

    async def run(self, task: dict) -> dict:
        ctx = _parse_context(task)
        step_name = ctx.get("step_name", "")
        if not step_name:
            title = task.get("title", "")
            if "] " in title:
                step_name = title.split("] ", 1)[1]
        if not step_name:
            step_name = ctx.get("workflow_step_id", "")

        logger.info("pipeline_v2 dispatch", step_name=step_name, task_id=task.get("id"))

        handler = _STEP_HANDLERS_V2.get(step_name)
        if not handler:
            return {
                "status": "failed",
                "result": f"Unknown step: {step_name!r}",
                "model": "shopping_pipeline_v2",
                "cost": 0.0,
                "iterations": 1,
            }

        mission_id = task.get("mission_id")
        input_artifacts = ctx.get("input_artifacts", [])
        artifacts = (
            await _read_artifacts(mission_id, input_artifacts)
            if mission_id
            else {}
        )

        try:
            result = await handler(task, artifacts, ctx)
            if isinstance(result, dict) and result.get("_needs_clarification"):
                return {
                    "status": "needs_clarification",
                    "clarification": result.get("clarification", "More info needed"),
                    "result": json.dumps(result, ensure_ascii=False, default=str),
                    "model": "shopping_pipeline_v2",
                    "cost": 0.0,
                    "iterations": 1,
                }
            return {
                "status": "completed",
                "result": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str),
                "model": "shopping_pipeline_v2",
                "cost": 0.0,
                "iterations": 1,
            }
        except Exception as exc:
            logger.exception("pipeline_v2 step %r failed", step_name)
            return {
                "status": "failed",
                "result": f"Pipeline v2 error: {exc}",
                "model": "shopping_pipeline_v2",
                "cost": 0.0,
                "iterations": 1,
            }
