# shopping/intelligence/vector_bridge.py
"""
Phase C — Shopping RAG Vector Bridge

Connects the shopping subsystem to the main vector store, enabling
semantic search over products, reviews, and shopping sessions.

Public API:
    await embed_product(product_dict, source)
    await embed_products_batch(products, source)
    await embed_review_synthesis(synthesis, product_name, product_url)
    await embed_shopping_session(session_dict)
    await embed_purchase(user_id, product_name, price, category)
    await embed_user_shopping_profile(user_id)
    await search_shopping_knowledge(query, top_k=5, data_type=None, user_id=None)
"""
import hashlib
import time
from typing import Optional

from src.infra.logging_config import get_logger
from src.memory.vector_store import embed_and_store, query, is_ready

logger = get_logger("shopping.intelligence.vector_bridge")


# ─── Product Embedding ──────────────────────────────────────────────────────

async def embed_product(
    product: dict,
    source: str = "",
) -> Optional[str]:
    """
    Embed a product into the shopping vector collection after scraping.

    Composes an embedding text from name, brand, key specs, and price info.

    Args:
        product: Product dict with keys like name, brand, price, specs, url, category.
        source:  Scraper source name (e.g. 'trendyol', 'hepsiburada').

    Returns:
        Document ID if stored, None otherwise.
    """
    if not is_ready():
        return None

    name = product.get("name", "")
    if not name:
        return None

    brand = product.get("brand", "")
    price = product.get("price")
    category = product.get("category", "")
    url = product.get("url", "")

    # Build embedding text: name + brand + key specs + price
    parts = [name]
    if brand and brand.lower() not in name.lower():
        parts.append(brand)
    if category:
        parts.append(f"category: {category}")

    # Include specs (top features)
    specs = product.get("specs", product.get("specifications", {}))
    if isinstance(specs, dict):
        spec_lines = [f"{k}: {v}" for k, v in list(specs.items())[:8]]
        if spec_lines:
            parts.append(" | ".join(spec_lines))
    elif isinstance(specs, list):
        parts.append(" | ".join(str(s) for s in specs[:8]))

    if price is not None:
        parts.append(f"price: {price} TL")

    text = " — ".join(parts)

    # Generate stable doc_id from URL or name+source
    key = url or f"{name}:{source}"
    doc_id = f"shop-prod-{hashlib.sha256(key.encode()).hexdigest()[:16]}"

    metadata = {
        "data_type": "product",
        "source": source,
        "product_name": name[:200],
        "brand": brand[:100] if brand else "",
        "category": category[:100] if category else "",
        "url": url[:500] if url else "",
        "timestamp": time.time(),
    }
    if price is not None:
        metadata["price"] = float(price)

    result = await embed_and_store(
        text=text,
        metadata=metadata,
        collection="shopping",
        doc_id=doc_id,
    )

    if result:
        logger.debug("Embedded product: %s (source=%s)", name[:60], source)
    return result


# ─── Review Synthesis Embedding ──────────────────────────────────────────────

async def embed_review_synthesis(
    synthesis: dict,
    product_name: str,
    product_url: str = "",
) -> Optional[str]:
    """
    Embed a synthesized review summary into the shopping collection.

    Args:
        synthesis:    Result dict from review_synthesizer.synthesize_reviews().
        product_name: Canonical product name.
        product_url:  Product URL (for linking).

    Returns:
        Document ID if stored, None otherwise.
    """
    if not is_ready():
        return None

    sentiment = synthesis.get("overall_sentiment", "unknown")
    rating = synthesis.get("confidence_adjusted_rating")
    positives = synthesis.get("positive_themes", [])
    negatives = synthesis.get("negative_themes", [])
    defects = synthesis.get("defect_patterns", [])
    warnings = synthesis.get("warnings", [])

    # Build embedding text
    parts = [f"Review summary for {product_name}"]
    parts.append(f"Overall sentiment: {sentiment}")
    if rating is not None:
        parts.append(f"Adjusted rating: {rating}/5")
    if positives:
        parts.append(f"Pros: {', '.join(positives[:5])}")
    if negatives:
        parts.append(f"Cons: {', '.join(negatives[:5])}")
    if defects:
        parts.append(f"Known defects: {', '.join(defects[:3])}")
    if warnings:
        parts.append(f"Warnings: {', '.join(warnings[:3])}")

    text = ". ".join(parts)

    key = product_url or product_name
    doc_id = f"shop-rev-{hashlib.sha256(key.encode()).hexdigest()[:16]}"

    metadata = {
        "data_type": "review",
        "product_name": product_name[:200],
        "url": product_url[:500] if product_url else "",
        "sentiment": sentiment,
        "timestamp": time.time(),
    }
    if rating is not None:
        metadata["rating"] = float(rating)

    result = await embed_and_store(
        text=text,
        metadata=metadata,
        collection="shopping",
        doc_id=doc_id,
    )

    if result:
        logger.debug("Embedded review synthesis for: %s", product_name[:60])
    return result


# ─── Shopping Session Embedding ──────────────────────────────────────────────

async def embed_shopping_session(session: dict) -> Optional[str]:
    """
    Embed a shopping session summary into the shopping collection.

    Called when a session is completed (status='completed').

    Args:
        session: Session dict with keys: session_id, user_id, topic,
                 summary, products, questions.

    Returns:
        Document ID if stored, None otherwise.
    """
    if not is_ready():
        return None

    session_id = session.get("session_id", "")
    topic = session.get("topic", "")
    summary = session.get("summary", "")
    user_id = session.get("user_id", "")

    if not topic:
        return None

    # Build embedding text from session context
    parts = [f"Shopping session: {topic}"]

    if summary:
        parts.append(summary)

    # Include products discussed
    products = session.get("products", [])
    if products:
        product_names = []
        for p in products[:10]:
            pname = p.get("name", p.get("product_name", ""))
            if pname:
                product_names.append(pname)
        if product_names:
            parts.append(f"Products compared: {', '.join(product_names)}")

    # Include key questions
    questions = session.get("questions", [])
    if questions:
        q_texts = [q.get("question", "") for q in questions[:5] if q.get("question")]
        if q_texts:
            parts.append(f"Questions: {'; '.join(q_texts)}")

    text = ". ".join(parts)

    doc_id = f"shop-sess-{hashlib.sha256(session_id.encode()).hexdigest()[:16]}" if session_id else None

    metadata = {
        "data_type": "shopping_session",
        "session_id": session_id,
        "user_id": str(user_id),
        "topic": topic[:200],
        "product_count": len(products),
        "timestamp": time.time(),
    }

    result = await embed_and_store(
        text=text,
        metadata=metadata,
        collection="shopping",
        doc_id=doc_id,
    )

    if result:
        logger.debug("Embedded session: %s (topic=%s)", session_id[:8] if session_id else "?", topic[:40])
    return result


# ─── Purchase Embedding ──────────────────────────────────────────────────────

async def embed_purchase(
    user_id: int,
    product_name: str,
    price: float = None,
    category: str = "",
    store: str = "",
) -> Optional[str]:
    """
    Embed a purchase record into the shopping collection.

    Args:
        user_id:      User who made the purchase.
        product_name: What was purchased.
        price:        Purchase price.
        category:     Product category.
        store:        Store where purchased.

    Returns:
        Document ID if stored, None otherwise.
    """
    if not is_ready() or not product_name:
        return None

    parts = [f"Purchased: {product_name}"]
    if category:
        parts.append(f"category: {category}")
    if store:
        parts.append(f"from: {store}")
    if price is not None:
        parts.append(f"price: {price} TL")

    text = " — ".join(parts)

    doc_id = f"shop-purch-{hashlib.sha256(f'{user_id}:{product_name}:{time.time()}'.encode()).hexdigest()[:16]}"

    metadata = {
        "data_type": "purchase",
        "user_id": str(user_id),
        "product_name": product_name[:200],
        "category": category[:100] if category else "",
        "store": store[:100] if store else "",
        "timestamp": time.time(),
    }
    if price is not None:
        metadata["price"] = float(price)

    return await embed_and_store(
        text=text,
        metadata=metadata,
        collection="shopping",
        doc_id=doc_id,
    )


# ─── User Shopping Profile Embedding ────────────────────────────────────────

async def embed_user_shopping_profile(user_id: int) -> int:
    """
    Bridge shopping_memory.db entities to the vector store.

    Embeds owned items, preferences, and behaviors from the user profile
    into the shopping collection for semantic retrieval.

    Returns:
        Number of embeddings stored.
    """
    if not is_ready():
        return 0

    try:
        from src.shopping.memory.user_profile import get_user_profile
    except ImportError:
        logger.debug("Cannot import user_profile — shopping memory not available")
        return 0

    profile = await get_user_profile(user_id)
    count = 0

    # Embed owned items individually
    for item in profile.get("owned_items", []):
        item_name = item.get("name", "")
        if not item_name:
            continue

        text = f"User owns: {item_name}"
        extras = {k: v for k, v in item.items()
                  if k not in ("_added_at", "name") and v}
        if extras:
            specs_str = ", ".join(f"{k}: {v}" for k, v in list(extras.items())[:6])
            text += f" ({specs_str})"

        doc_id = f"shop-own-{user_id}-{hashlib.sha256(item_name.encode()).hexdigest()[:12]}"

        result = await embed_and_store(
            text=text,
            metadata={
                "data_type": "owned_item",
                "user_id": str(user_id),
                "item_name": item_name[:200],
                "timestamp": item.get("_added_at", time.time()),
            },
            collection="shopping",
            doc_id=doc_id,
        )
        if result:
            count += 1

    # Embed all preferences as a single document
    prefs = profile.get("preferences", {})
    inferred = profile.get("inferred_preferences", {})
    all_prefs = {**inferred, **prefs}

    if all_prefs:
        pref_lines = [f"{k}: {v}" for k, v in all_prefs.items()]
        text = f"User shopping preferences: {'; '.join(pref_lines)}"

        dietary = profile.get("dietary_restrictions", [])
        if dietary:
            text += f". Dietary restrictions: {', '.join(dietary)}"

        location = profile.get("location", "")
        if location:
            text += f". Location: {location}"

        result = await embed_and_store(
            text=text,
            metadata={
                "data_type": "user_preferences",
                "user_id": str(user_id),
                "timestamp": time.time(),
            },
            collection="shopping",
            doc_id=f"shop-pref-{user_id}",
        )
        if result:
            count += 1

    # Embed behaviors
    behaviors = profile.get("behaviors", [])
    if behaviors:
        text = f"User shopping behaviors: {', '.join(behaviors[:10])}"
        result = await embed_and_store(
            text=text,
            metadata={
                "data_type": "user_behavior",
                "user_id": str(user_id),
                "timestamp": time.time(),
            },
            collection="shopping",
            doc_id=f"shop-beh-{user_id}",
        )
        if result:
            count += 1

    if count:
        logger.info("Embedded %d shopping profile items for user %d", count, user_id)
    return count


async def embed_products_batch(products: list[dict], source: str = "") -> int:
    """Embed multiple products. Returns count of successfully embedded."""
    count = 0
    for product in products:
        result = await embed_product(product, source)
        if result:
            count += 1
    return count


# ─── Search ──────────────────────────────────────────────────────────────────

async def search_shopping_knowledge(
    query_text: str,
    top_k: int = 5,
    data_type: str = None,
    user_id: int = None,
) -> list[dict]:
    """
    Search the shopping vector collection for relevant knowledge.

    Args:
        query_text: Natural language query.
        top_k:      Number of results.
        data_type:  Optional filter: 'product', 'review', 'shopping_session', 'purchase'.
        user_id:    Optional filter by user.

    Returns:
        List of result dicts with keys: text, metadata, distance.
    """
    if not is_ready():
        return []

    where = {}
    if data_type:
        where["data_type"] = data_type
    if user_id is not None:
        where["user_id"] = str(user_id)

    return await query(
        text=query_text,
        collection="shopping",
        top_k=top_k,
        where=where if where else None,
    )
