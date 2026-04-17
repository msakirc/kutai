"""ShoppingPipeline — mechanical executor for shopping workflow steps.

Mirrors CodingPipeline's interface but handles shopping-specific steps
without any LLM calls. Steps map to Python functions that use existing
shopping modules.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("workflows.shopping.pipeline")


# ── Relevance filtering ────────────────────────────────────────────────────

def _relevance_score(product_name: str, query: str) -> float:
    """Score 0.0–1.0 how relevant *product_name* is to the search *query*.

    Tokenizes the query, normalises both sides (Turkish-aware lowercase,
    strip punctuation), and counts what fraction of query tokens appear
    in the product name.  A collapsed (no-space) check catches model
    numbers like "EQ.s100" matching token "s100".
    """
    from src.shopping.text_utils import normalize_turkish

    def _norm(s: str) -> str:
        s = normalize_turkish(s)
        return re.sub(r"[^a-z0-9ğüşıöç\s]", "", s)

    name_norm = _norm(product_name)
    query_norm = _norm(query)
    name_collapsed = name_norm.replace(" ", "")

    tokens = [t for t in query_norm.split() if len(t) >= 2]
    if not tokens:
        return 1.0

    matched = sum(1 for t in tokens if t in name_norm or t in name_collapsed)
    return matched / len(tokens)


def _filter_relevant(products: list, query: str, strict: bool = False) -> list:
    """Keep only products whose names are relevant to *query*.

    Strategy: score every product, then keep those within 20% of the best
    score, with a hard floor of 0.5 (at least half the query tokens must
    appear).  If nothing passes the floor:
      - strict=False (default): return the original list so the user still
        gets *something* (safe for product results).
      - strict=True: return an empty list (used for community results
        where irrelevant complaints are worse than no data).
    """
    if not products:
        return products
    if not query:
        return [] if strict else products

    scored = []
    for p in products:
        name = p.name if hasattr(p, "name") else (p.get("name", "") if isinstance(p, dict) else "")
        s = _relevance_score(name, query)
        scored.append((p, s))

    max_score = max(s for _, s in scored)
    threshold = max(max_score * 0.8, 0.5)

    filtered = [p for p, s in scored if s >= threshold]

    dropped = len(products) - len(filtered)
    if dropped:
        logger.info(
            "relevance filter: %d/%d kept (threshold=%.2f, strict=%s)",
            len(filtered), len(products), threshold, strict,
        )

    if filtered:
        return filtered
    return [] if strict else products


def _annotate_fake_discounts(groups: list[dict]) -> dict[tuple, dict]:
    """Flag stores whose discount ratio is way off from the group median.

    For each matched-product group, compute ``original_price /
    discounted_price`` per store. If the group has ≥2 entries with both
    fields populated, flag any entry whose ratio is >1.5× the median AND
    >1.2 absolute (i.e. claiming at least 20% off when peers aren't).

    Returns a dict keyed by ``(name, source, url)`` tuple → flag payload.
    Empty dict when no flags apply.
    """
    import statistics

    flags: dict[tuple, dict] = {}
    for group in groups:
        entries = group.get("products", [])
        if len(entries) < 2:
            continue

        pairs: list[tuple[dict, float]] = []
        for e in entries:
            orig = e.get("original_price")
            disc = e.get("discounted_price")
            if orig and disc and disc > 0 and orig > disc:
                pairs.append((e, orig / disc))

        if len(pairs) < 2:
            continue

        ratios = [r for _, r in pairs]
        median = statistics.median(ratios)
        for entry, ratio in pairs:
            if ratio > median * 1.5 and ratio > 1.2:
                key = (entry.get("name", ""), entry.get("source", ""), entry.get("url", ""))
                flags[key] = {
                    "is_suspicious_discount": True,
                    "discount_flag_reason": (
                        f"Bu mağazada indirim oranı ({(ratio - 1) * 100:.0f}%) "
                        f"diğer mağazalardaki medyana ({(median - 1) * 100:.0f}%) "
                        f"göre çok yüksek — 'orijinal fiyat' şişirilmiş olabilir"
                    ),
                }
    return flags


async def _match_and_flatten(products: list) -> list[dict]:
    """Run product matching, then flatten groups back to a sorted list.

    Each match group keeps all its source entries (for cross-source price
    comparison).  Groups are ordered by match count desc, then cheapest
    price asc — so multi-source matches with good prices come first.
    The original Product fields are preserved (not the matcher's stripped
    version) so the format step has access to prices, ratings, etc.
    """
    # Separate Product dataclass instances from plain dicts
    product_objs: list = []
    plain_dicts: list[dict] = []
    for p in products:
        if dataclasses.is_dataclass(p):
            product_objs.append(p)
        elif isinstance(p, dict):
            plain_dicts.append(p)

    if not product_objs:
        return plain_dicts

    # Build lookup: (name, source, url) -> original full dict
    orig_lookup: dict[tuple, dict] = {}
    for p in product_objs:
        d = dataclasses.asdict(p)
        key = (d.get("name", ""), d.get("source", ""), d.get("url", ""))
        orig_lookup[key] = d

    try:
        from src.shopping.intelligence.product_matcher import match_products
        groups = await match_products(product_objs)
    except Exception as exc:
        logger.warning("product_matcher failed, returning unmatched: %s", exc)
        return list(orig_lookup.values()) + plain_dicts

    # ── Fake-discount flags from cross-store ratio analysis ──
    flags = _annotate_fake_discounts(groups)

    # Flatten: for each group, look up the original full dict for every
    # product entry.  Fall back to the matcher's stripped dict if lookup
    # misses (shouldn't happen, but be safe).
    flat: list[dict] = []
    for group in groups:
        for prod in group.get("products", []):
            key = (prod.get("name", ""), prod.get("source", ""), prod.get("url", ""))
            full = orig_lookup.get(key, prod)
            flag = flags.get(key)
            if flag:
                full = {**full, **flag}
            flat.append(full)

    flat.extend(plain_dicts)
    return flat


# ── Artifact helper ─────────────────────────────────────────────────────────

async def _read_artifacts(mission_id: int | str, artifact_names: list[str]) -> dict:
    """Read artifacts from the mission blackboard."""
    from src.workflows.engine.artifacts import ArtifactStore
    store = ArtifactStore()
    result: dict[str, Any] = {}
    for name in artifact_names:
        value = await store.retrieve(mission_id, name)
        if value is not None:
            result[name] = value
    return result


def _extract_query(artifacts: dict, task: dict) -> str:
    """Extract the actual search query string from artifacts.

    Artifacts may be raw strings ("siemens s100") or JSON-encoded dicts
    (``{"clarified_query": "siemens s100", "skipped": true}``).
    """
    # Try clarified_query first (from step 1.1), then user_query (initial input)
    for key in ("clarified_query", "user_query"):
        raw = artifacts.get(key, "")
        if not raw:
            continue
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
                # The artifact might wrap the query in a field
                q = parsed.get("clarified_query") or parsed.get("query") or parsed.get("user_query", "")
                if q:
                    return q
            except (json.JSONDecodeError, ValueError):
                pass
        # Raw string — use as-is
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    # Last resort: task description
    return task.get("description", "")


# ── Step handlers ────────────────────────────────────────────────────────────

async def _step_search(task: dict, artifacts: dict) -> str:
    """Search products and community data. Returns JSON."""
    query = _extract_query(artifacts, task)
    logger.info("search step starting", query=query[:100])

    from src.shopping.resilience.fallback_chain import (
        get_product_with_fallback,
        get_community_data,
    )

    product_task = asyncio.ensure_future(
        asyncio.wait_for(get_product_with_fallback(query), timeout=45)
    )
    community_task = asyncio.ensure_future(
        asyncio.wait_for(get_community_data(query), timeout=20)
    )

    products: list = []
    community: list = []
    try:
        products = await product_task
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("product search failed for _step_search", error=str(exc))
    try:
        community = await community_task
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("community search failed for _step_search", error=str(exc))

    logger.info("search step done", product_count=len(products or []), community_count=len(community or []))

    if not isinstance(products, list):
        products = []
    if not isinstance(community, list):
        community = []

    # ── Relevance filtering ──
    products = _filter_relevant(products, query)
    community = _filter_relevant(community, query, strict=True)

    # ── Value scoring (pure Python, no LLM) ──
    score_lookup: dict[str, dict] = {}
    product_objs = [p for p in products if dataclasses.is_dataclass(p)]
    if product_objs:
        try:
            from src.shopping.intelligence.value_scorer import score_products
            scores = await score_products(product_objs)
            for s in scores:
                score_lookup[s["product_name"]] = s
        except Exception as exc:
            logger.warning("value_scorer failed, continuing without scores: %s", exc)

    # ── Product matching / deduplication ──
    products_dicts = await _match_and_flatten(products)

    # ── Merge value scores into product dicts ──
    for p in products_dicts:
        name = p.get("name", "")
        if name in score_lookup:
            p["value_score"] = score_lookup[name]["value_score"]
            p["value_rank"] = score_lookup[name]["rank"]
            p["score_breakdown"] = score_lookup[name]["breakdown"]

    community_dicts = [
        dataclasses.asdict(c) if dataclasses.is_dataclass(c) else c
        for c in community
    ]

    # Cap results to prevent oversized artifacts (50K+ JSON from 16 scrapers)
    _MAX_PRODUCTS = 20
    _MAX_COMMUNITY = 10
    result = {
        "formatted_text": "",
        "products": products_dicts[:_MAX_PRODUCTS],
        "community": community_dicts[:_MAX_COMMUNITY],
        "product_count": len(products_dicts),
        "community_count": len(community_dicts),
        "escalation_needed": len(products_dicts) == 0,
    }
    return json.dumps(result, default=str)


async def _step_search_and_reviews(task: dict, artifacts: dict) -> str:
    """Search products, community, and fetch reviews for top products."""
    query = _extract_query(artifacts, task)

    from src.shopping.resilience.fallback_chain import (
        get_product_with_fallback,
        get_community_data,
    )

    product_task = asyncio.ensure_future(
        asyncio.wait_for(get_product_with_fallback(query), timeout=45)
    )
    community_task = asyncio.ensure_future(
        asyncio.wait_for(get_community_data(query), timeout=20)
    )

    products: list = []
    community: list = []
    try:
        products = await product_task
    except (asyncio.TimeoutError, Exception):
        pass
    try:
        community = await community_task
    except (asyncio.TimeoutError, Exception):
        pass

    if not isinstance(products, list):
        products = []
    if not isinstance(community, list):
        community = []

    # ── Relevance filtering ──
    products = _filter_relevant(products, query)
    community = _filter_relevant(community, query, strict=True)

    # ── Value scoring (pure Python, no LLM) ──
    score_lookup: dict[str, dict] = {}
    product_objs = [p for p in products if dataclasses.is_dataclass(p)]
    if product_objs:
        try:
            from src.shopping.intelligence.value_scorer import score_products
            scores = await score_products(product_objs)
            for s in scores:
                score_lookup[s["product_name"]] = s
        except Exception as exc:
            logger.warning("value_scorer failed, continuing without scores: %s", exc)

    # ── Product matching / deduplication ──
    products_dicts = await _match_and_flatten(products)

    # ── Merge value scores into product dicts ──
    for p in products_dicts:
        name = p.get("name", "")
        if name in score_lookup:
            p["value_score"] = score_lookup[name]["value_score"]
            p["value_rank"] = score_lookup[name]["rank"]
            p["score_breakdown"] = score_lookup[name]["breakdown"]

    community_dicts = [
        dataclasses.asdict(c) if dataclasses.is_dataclass(c) else c
        for c in community
    ]

    # Fetch reviews for top 3 products with URLs
    from src.shopping.scrapers import get_scraper

    reviews: list = []
    for p in products_dicts[:3]:
        url = p.get("url", "")
        if not url:
            continue
        source = None
        for domain, key in [
            ("trendyol", "trendyol"),
            ("hepsiburada", "hepsiburada"),
            ("amazon.com.tr", "amazon_tr"),
        ]:
            if domain in url:
                source = key
                break
        if source:
            try:
                scraper_cls = get_scraper(source)
                if scraper_cls:
                    scraper = scraper_cls()
                    r = await asyncio.wait_for(scraper.get_reviews(url), timeout=15)
                    if r:
                        reviews.extend(r[:10])
            except Exception:
                pass

    review_dicts = [
        dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r
        for r in reviews
    ]

    result = {
        "formatted_text": "",
        "products": products_dicts,
        "community": community_dicts,
        "reviews": review_dicts,
        "product_count": len(products_dicts),
        "community_count": len(community_dicts),
        "escalation_needed": len(products_dicts) == 0,
    }
    return json.dumps(result, default=str)


async def _step_search_for_product(task: dict, artifacts: dict) -> str:
    """Search step for product_research workflow.

    Same search/filter/score/match logic as _step_search_and_reviews but
    named distinctly so the workflow JSON can be read for its intent.
    Reviews are fetched when a matching scraper exists; if the scraper
    doesn't expose detailed reviews yet, the reviews field is an empty
    list — the enrich step treats that as 'no data' gracefully.
    """
    return await _step_search_and_reviews(task, artifacts)


async def _step_format(task: dict, artifacts: dict) -> str:
    """Format search results into a Telegram-ready message.

    If a 'recommendation' artifact exists (from the LLM synthesis step in
    the full shopping workflow), use it directly — it's already a rich
    text recommendation.  Otherwise fall back to formatting raw products.
    """
    from src.shopping.output.summary import format_recommendation_summary
    from src.shopping.output.formatters import format_price

    # Prefer the LLM-synthesized recommendation if available
    recommendation = artifacts.get("recommendation", "")
    if isinstance(recommendation, str) and len(recommendation.strip()) > 20:
        return recommendation.strip()

    raw = artifacts.get("search_results", "{}")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            data = {}
    else:
        data = raw if isinstance(raw, dict) else {}

    # Also read review_data artifact if available (from step 2.1 in full workflow)
    review_raw = artifacts.get("review_data", "{}")
    if isinstance(review_raw, str):
        try:
            review_data = json.loads(review_raw)
        except (json.JSONDecodeError, ValueError):
            review_data = {}
    else:
        review_data = review_raw if isinstance(review_raw, dict) else {}

    products = data.get("products", [])
    community = data.get("community", [])
    reviews = review_data.get("reviews", []) if isinstance(review_data, dict) else []

    if not products:
        if community:
            text = (
                f"No product listings found, but found {len(community)} "
                f"community discussions:\n\n"
            )
            for c in community[:5]:
                text += f"• {c.get('name', 'Discussion')}\n  {c.get('url', '')}\n"
            return text
        return "No results found. Try a more specific search."

    # ── Build formatted output directly ──
    lines: list[str] = []

    # ── Pick best product per source, rank by value_score ──
    # Each site returns its most relevant result first.  Take the #1
    # result from each source, then rank by value_score (composite of
    # price, rating, seller, shipping, warranty).  Falls back to
    # highest-price heuristic when scores are unavailable.
    best_per_source: dict[str, dict] = {}
    for p in products:
        src = p.get("source", "")
        if src and src not in best_per_source:
            best_per_source[src] = p

    top_products = list(best_per_source.values())

    has_scores = any(p.get("value_score") for p in top_products)
    priced_tops = [p for p in top_products if p.get("original_price") or p.get("discounted_price")]

    if has_scores:
        # Rank by value_score descending (best value first)
        priced_tops.sort(key=lambda p: p.get("value_score", 0), reverse=True)
    else:
        # Fallback: highest price first (main product vs spare parts)
        priced_tops.sort(
            key=lambda p: p.get("discounted_price") or p.get("original_price") or 0,
            reverse=True,
        )

    if priced_tops:
        best = priced_tops[0]
        best_price = best.get("discounted_price") or best.get("original_price")
        score = best.get("value_score")
        score_str = f"  ({score:.0f}/100)" if score else ""
        lines.append(
            f"🏆 *{best.get('name', '')}*{score_str}\n"
            f"   💰 {format_price(best_price)} — {best.get('source', '')}"
        )
        if best.get("url"):
            lines.append(f"   🔗 {best['url']}")

        # Rating if available
        if best.get("rating"):
            stars = "⭐" * int(best["rating"])
            review_count = best.get("review_count", "")
            rc_str = f" ({review_count} değerlendirme)" if review_count else ""
            lines.append(f"   {stars} {best['rating']}/5{rc_str}")

        # Installment info for winner
        try:
            from src.shopping.intelligence.installment_calculator import calculate_installments
            best_source = best.get("source", "")
            if best_price and best_source:
                installments = await calculate_installments(best_price, best_source)
                # Show best faizsiz option (longest interest-free term)
                faizsiz = [i for i in installments if i.get("is_faizsiz") and i["tier"] > 1]
                if faizsiz:
                    best_inst = max(faizsiz, key=lambda i: i["tier"])
                    lines.append(
                        f"   💳 {best_inst['tier']} ay x {format_price(best_inst['monthly_payment'])} (faizsiz)"
                    )
                else:
                    # Show cheapest non-peşin option
                    non_pesin = [i for i in installments if i["tier"] > 1]
                    if non_pesin:
                        cheapest = non_pesin[0]
                        lines.append(
                            f"   💳 {cheapest['tier']} ay x {format_price(cheapest['monthly_payment'])}"
                        )
        except Exception as exc:
            logger.debug("installment calc failed for winner: %s", exc)

        # Other sources — show their #1 result price + score
        others = [p for p in priced_tops[1:] if p.get("source") != best.get("source")]
        if others:
            lines.append("\n📍 *Diğer Fiyatlar:*")
            for p in others[:5]:
                price = p.get("discounted_price") or p.get("original_price")
                orig = p.get("original_price")
                disc = p.get("discounted_price")
                price_str = format_price(price)
                if disc and orig and disc < orig:
                    price_str += f" ~~{format_price(orig)}~~"
                src = p.get("source", "?")
                url = p.get("url", "")
                url_str = f"  {url}" if url else ""
                p_score = p.get("value_score")
                score_tag = f" [{p_score:.0f}]" if p_score else ""
                lines.append(f"  • {src} — {price_str}{score_tag}{url_str}")
    else:
        # Products found but no prices
        top = products[0]
        lines.append(f"🔍 *{top.get('name', '')}*")
        lines.append(f"   {top.get('source', '')} — fiyat bilgisi yok")
        if top.get("url"):
            lines.append(f"   🔗 {top['url']}")

    # ── Community data ──
    if community:
        by_source: dict[str, list] = {}
        for c in community:
            src = c.get("source", "other")
            by_source.setdefault(src, []).append(c)

        lines.append("\n💬 *Topluluk:*")
        for src, items in by_source.items():
            src_label = {
                "teknopat": "Technopat",
                "sikayetvar": "Şikayetvar",
                "donanimhaber": "DonanımHaber",
                "eksisozluk": "Ekşi Sözlük",
            }.get(src, src)

            if src == "sikayetvar":
                lines.append(f"  ⚠️ {len(items)} şikayet — {src_label}")
            else:
                lines.append(f"  💭 {len(items)} tartışma — {src_label}")

            # Show top 2 thread titles
            for item in items[:2]:
                name = item.get("name", "")[:60]
                url = item.get("url", "")
                if name:
                    lines.append(f"     • {name}")

    # ── Reviews ──
    if reviews:
        lines.append("\n📝 *Kullanıcı Yorumları:*")
        # Group by positive/negative sentiment if rating available
        positive = [r for r in reviews if (r.get("rating") or 0) >= 4]
        negative = [r for r in reviews if (r.get("rating") or 5) <= 2]
        neutral = [r for r in reviews if r not in positive and r not in negative]

        if positive:
            lines.append(f"  👍 {len(positive)} olumlu")
            for r in positive[:2]:
                text = r.get("text", "")[:80]
                if text:
                    lines.append(f"     \"{text}\"")
        if negative:
            lines.append(f"  👎 {len(negative)} olumsuz")
            for r in negative[:2]:
                text = r.get("text", "")[:80]
                if text:
                    lines.append(f"     \"{text}\"")
        if not positive and not negative and neutral:
            for r in neutral[:2]:
                text = r.get("text", "")[:80]
                if text:
                    lines.append(f"  💬 \"{text}\"")

    # ── Source count ──
    sources = set(p.get("source", "") for p in products)
    sources.discard("")
    if len(sources) >= 3:
        lines.append(f"\n✅ {len(sources)} kaynaktan karşılaştırma")
    elif len(sources) == 2:
        lines.append(f"\n🟡 {len(sources)} kaynak")
    elif len(sources) == 1:
        lines.append(f"\n🟠 Tek kaynak ({list(sources)[0]})")

    return "\n".join(lines)


async def _step_analyze_query(task: dict, artifacts: dict) -> str:
    """Analyze the shopping query — intent, category, constraints, vagueness."""
    from src.shopping.intelligence.query_analyzer import _fallback_analyze

    query = artifacts.get("user_query", "") or task.get("description", "")
    if isinstance(query, str) and query.startswith("{"):
        try:
            query = json.loads(query).get("user_query", query)
        except (json.JSONDecodeError, ValueError):
            pass

    analysis = _fallback_analyze(query)
    # Output must match artifact_schema: requires "query" and "needs_clarification"
    analysis["query"] = query
    return json.dumps(analysis, ensure_ascii=False, default=str)


# ── Step registry ────────────────────────────────────────────────────────────

async def _step_clarify(task: dict, artifacts: dict) -> str | dict:
    """Check if clarification is needed. If not, pass the query through.

    Returns str (artifact) for pass-through, or dict with
    status="needs_clarification" to trigger the Telegram pause mechanism.
    """
    parsed = artifacts.get("parsed_intent", "{}")
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (json.JSONDecodeError, ValueError):
            parsed = {}

    needs_clarification = parsed.get("needs_clarification", False)
    query = _extract_query(artifacts, task)

    if not needs_clarification:
        return json.dumps({
            "clarified_query": query,
            "skipped": True,
        }, ensure_ascii=False)

    # Vague query — trigger the Telegram clarification mechanism.
    # Return a special dict (not str) that run() will detect and
    # pass through as the result, with status="needs_clarification".
    category = parsed.get("category", "")
    cat_hint = f" ({category})" if category else ""
    return {
        "_needs_clarification": True,
        "clarification": (
            f"'{query}' biraz geniş bir arama{cat_hint}. "
            f"Aradığınız ürünü daraltabilir misiniz?\n\n"
            f"Örneğin:\n"
            f"• Marka veya model (ör: Nike Air Max)\n"
            f"• Bütçe (ör: 2000 TL altı)\n"
            f"• Kullanım amacı (ör: koşu ayakkabısı)"
        ),
        "query": query,
    }


async def _step_enrich_product(task: dict, artifacts: dict) -> str:
    """Deterministic enrichment for specific-product research.

    Reads ``search_results``, attaches a ``cross_store_summary`` section
    (store count, how many flagged as suspicious discounts, price spread),
    and passes the product list through unchanged.  No LLM. No review
    synthesis, delivery calculation, or timing — those are stubs until
    scrapers provide the underlying data.
    """
    from src.shopping.intelligence.special.fake_discount_detector import (
        check_cross_store_consistency,
    )

    raw = artifacts.get("search_results", "{}")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            data = {}
    else:
        data = raw if isinstance(raw, dict) else {}

    products = data.get("products", [])
    community = data.get("community", [])
    reviews = data.get("reviews", [])

    prices_by_source: dict[str, float] = {}
    for p in products:
        src = p.get("source", "")
        price = p.get("discounted_price") or p.get("original_price")
        if src and price and src not in prices_by_source:
            prices_by_source[src] = float(price)

    consistency = check_cross_store_consistency(prices_by_source)
    suspicious_count = sum(
        1 for p in products if p.get("is_suspicious_discount")
    )

    cross_store_summary = {
        "store_count": len(prices_by_source),
        "suspicious_discount_count": suspicious_count,
        "price_spread_pct": consistency.get("spread_pct", 0.0),
        "cheapest_store": consistency.get("cheapest"),
        "most_expensive_store": consistency.get("most_expensive"),
        "notes": consistency.get("notes", []),
    }

    enriched = {
        "products": products,
        "community": community,
        "reviews": reviews,
        "cross_store_summary": cross_store_summary,
        "product_count": data.get("product_count", len(products)),
        "community_count": data.get("community_count", len(community)),
    }
    return json.dumps(enriched, default=str, ensure_ascii=False)


async def _step_deliver_product_research(task: dict, artifacts: dict) -> str:
    """Format enriched product research data for Telegram delivery.

    Reuses _step_format but passes the enriched artifact in under the
    ``search_results`` key so the existing formatter logic (winner,
    others, community, reviews) runs unchanged. Adds a "fake discount"
    callout when any products carry is_suspicious_discount=True.
    """
    enriched_raw = artifacts.get("enriched_product_data", "{}")
    adapted = {
        "search_results": enriched_raw,
        "user_query": artifacts.get("user_query", ""),
    }
    base = await _step_format(task, adapted)

    try:
        data = json.loads(enriched_raw) if isinstance(enriched_raw, str) else enriched_raw
    except (json.JSONDecodeError, ValueError):
        data = {}
    suspicious = [
        p for p in data.get("products", [])
        if p.get("is_suspicious_discount")
    ]
    if suspicious:
        base += "\n\n⚠️ *Şüpheli İndirim:*\n"
        for p in suspicious[:3]:
            src = p.get("source", "?")
            reason = p.get("discount_flag_reason", "")
            base += f"  • {src}: {reason}\n"

    return base


async def _step_stub_disabled(task: dict, artifacts: dict) -> str:
    """Placeholder for steps that depend on data scrapers don't provide yet.

    Returns a neutral status=disabled artifact so downstream steps can
    check and skip. When scrapers improve (review text, price history,
    shipping fields), replace this handler with the real implementation.
    """
    context = task.get("context", {}) if isinstance(task, dict) else {}
    step_name = context.get("step_name", "unknown") if isinstance(context, dict) else "unknown"
    return json.dumps({
        "status": "disabled",
        "step": step_name,
        "reason": "Scraper data insufficient — this module activates when "
                  "scrapers populate review text / shipping fields / price history.",
    }, ensure_ascii=False)


_STEP_HANDLERS = {
    # quick_search
    "execute_product_search": _step_search,
    "format_and_deliver": _step_format,
    # shopping (full category/discovery workflow)
    "search_and_collect_reviews": _step_search_and_reviews,
    "understand_query_check_clarity": _step_analyze_query,
    # product_research (specific product workflow — NEW)
    "search_for_product": _step_search_for_product,
    "enrich_product_results": _step_enrich_product,
    "deliver_product_research": _step_deliver_product_research,
    # product_research stubs — scaffolded, activate when scrapers improve
    "synthesize_product_reviews": _step_stub_disabled,
    "compare_delivery_options": _step_stub_disabled,
    "advise_buy_timing": _step_stub_disabled,
}


# ── Pipeline class ───────────────────────────────────────────────────────────

class ShoppingPipeline:
    """Mechanical executor for shopping workflow steps.

    Runs Python functions for data-fetch/format steps — no LLM involved.
    Intelligence steps (scoring, alternatives, budget analysis) are handled
    by separate LLM agent tasks in the workflow.
    """

    async def run(self, task: dict) -> dict:
        """Execute a mechanical shopping workflow step.

        Returns dict with:
            status: "completed" or "failed"
            result: str (the output text/data)
            model: "shopping_pipeline" (no LLM used)
            cost: 0.0
            iterations: 1
        """
        context = task.get("context", {})
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (json.JSONDecodeError, ValueError):
                context = {}

        step_name = context.get("step_name", "")
        if not step_name:
            # Fallback: parse from title "[0.1] step_name_here"
            title = task.get("title", "")
            if "] " in title:
                step_name = title.split("] ", 1)[1]
        if not step_name:
            step_name = context.get("workflow_step_id", "")

        logger.info("step dispatch", step_name=step_name, task_id=task.get("id"))

        mission_id = task.get("mission_id")
        input_artifacts = context.get("input_artifacts", [])

        artifacts = (
            await _read_artifacts(mission_id, input_artifacts)
            if mission_id
            else {}
        )

        logger.info(
            "artifacts loaded",
            step_name=step_name,
            artifact_keys=list(artifacts.keys()),
            mission_id=mission_id,
        )

        handler = _STEP_HANDLERS.get(step_name)
        if not handler:
            return {
                "status": "failed",
                "result": f"Unknown step: {step_name!r}",
                "model": "shopping_pipeline",
                "cost": 0.0,
                "iterations": 1,
            }

        try:
            result = await handler(task, artifacts)

            # Handler can return a dict to signal special statuses
            # (e.g. needs_clarification from _step_clarify)
            if isinstance(result, dict) and result.get("_needs_clarification"):
                return {
                    "status": "needs_clarification",
                    "clarification": result.get("clarification", "More info needed"),
                    "result": json.dumps(result, ensure_ascii=False, default=str),
                    "model": "shopping_pipeline",
                    "cost": 0.0,
                    "iterations": 1,
                }

            return {
                "status": "completed",
                "result": result if isinstance(result, str) else json.dumps(result, default=str),
                "model": "shopping_pipeline",
                "cost": 0.0,
                "iterations": 1,
            }
        except Exception as exc:
            logger.exception("ShoppingPipeline step %r failed", step_name)
            return {
                "status": "failed",
                "result": f"Pipeline error: {exc}",
                "model": "shopping_pipeline",
                "cost": 0.0,
                "iterations": 1,
            }
