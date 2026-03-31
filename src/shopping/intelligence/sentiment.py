"""Fast keyword-based sentiment analysis for product reviews.

No LLM needed -- uses Turkish + English keyword dictionaries.
Returns sentiment score (-1 to +1) and extracted key phrases.

Designed for high-throughput batch analysis of scraped reviews.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

# Positive Turkish keywords -> weight
_POS_TR: dict[str, float] = {
    "mükemmel": 1.0,
    "harika": 0.9,
    "çok iyi": 0.8,
    "güzel": 0.7,
    "memnun": 0.8,
    "tavsiye": 0.7,
    "kaliteli": 0.8,
    "hızlı": 0.6,
    "başarılı": 0.7,
    "beğendim": 0.7,
    "süper": 0.8,
    "kusursuz": 0.9,
    "sorunsuz": 0.7,
    "sağlam": 0.7,
    "dayanıklı": 0.7,
    "değer": 0.6,
    "fiyat performans": 0.8,
    "teşekkür": 0.5,
    "memnunum": 0.8,
    "sevdim": 0.7,
    "tam istediğim": 0.8,
    "iyi geldi": 0.6,
    "hızlı geldi": 0.6,
    "sağlıklı": 0.6,
    "orijinal": 0.6,
    "gerçek": 0.5,
    "güvenilir": 0.7,
    "şık": 0.6,
    "kullanışlı": 0.6,
    "kolay": 0.5,
}

# Negative Turkish keywords -> weight (already negative)
_NEG_TR: dict[str, float] = {
    "kötü": -0.8,
    "berbat": -0.9,
    "bozuk": -0.8,
    "sorunlu": -0.7,
    "pişman": -0.8,
    "iade": -0.6,
    "arızalı": -0.8,
    "yavaş": -0.5,
    "pahalı": -0.4,
    "kalitesiz": -0.8,
    "sahte": -0.9,
    "çöp": -0.9,
    "hayal kırıklığı": -0.8,
    "tavsiye etmem": -0.9,
    "almayın": -0.9,
    "şikayet": -0.6,
    "geç geldi": -0.5,
    "eksik": -0.6,
    "kırık": -0.7,
    "çalışmıyor": -0.8,
    "işe yaramıyor": -0.8,
    "rezalet": -0.9,
    "aldatıcı": -0.8,
    "yanıltıcı": -0.7,
    "hata": -0.5,
    "problem": -0.5,
    "sorun": -0.5,
    "beklentimi karşılamadı": -0.7,
    "vasat": -0.5,
    "berbat geldi": -0.9,
    "ambalaj hasarlı": -0.6,
    "hasarlı geldi": -0.7,
}

# Positive English keywords -> weight
_POS_EN: dict[str, float] = {
    "excellent": 1.0,
    "great": 0.8,
    "good": 0.6,
    "amazing": 0.9,
    "perfect": 1.0,
    "love": 0.8,
    "recommend": 0.7,
    "quality": 0.6,
    "fast": 0.5,
    "worth": 0.6,
    "satisfied": 0.7,
    "best": 0.8,
    "awesome": 0.9,
    "happy": 0.7,
    "nice": 0.6,
    "works great": 0.8,
    "highly recommend": 0.9,
    "as described": 0.6,
}

# Negative English keywords -> weight (already negative)
_NEG_EN: dict[str, float] = {
    "terrible": -0.9,
    "awful": -0.9,
    "bad": -0.7,
    "worst": -1.0,
    "broken": -0.8,
    "defective": -0.8,
    "slow": -0.5,
    "expensive": -0.4,
    "return": -0.5,
    "refund": -0.6,
    "fake": -0.9,
    "waste": -0.8,
    "disappointed": -0.7,
    "regret": -0.8,
    "avoid": -0.9,
    "not working": -0.8,
    "stopped working": -0.8,
    "poor quality": -0.8,
    "do not buy": -0.9,
    "dont buy": -0.9,
    "misleading": -0.7,
}

# Negation words that flip sentiment of the following match
_NEGATION_TR = {"değil", "yok", "hiç", "olmadı", "olmaz", "olmayan"}
_NEGATION_EN = {"not", "no", "never", "don't", "doesn't", "isn't", "wasn't"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Analyze sentiment of a single review text.

    Args:
        text: Review text (Turkish or English or mixed).

    Returns:
        dict with keys:
        - ``score``: float in [-1, 1] (negative to positive)
        - ``positive_words``: list of matched positive keywords
        - ``negative_words``: list of matched negative keywords
        - ``label``: one of ``"positive"``, ``"negative"``, ``"neutral"``
    """
    if not text or not text.strip():
        return {
            "score": 0.0,
            "positive_words": [],
            "negative_words": [],
            "label": "neutral",
        }

    text_lower = text.lower()
    words = set(text_lower.split())

    pos_words: list[str] = []
    neg_words: list[str] = []
    total_score = 0.0
    count = 0

    # Check for negation context around each keyword
    def _has_negation(keyword: str) -> bool:
        idx = text_lower.find(keyword)
        if idx == -1:
            return False
        prefix = text_lower[max(0, idx - 30) : idx]
        prefix_words = set(prefix.split())
        if prefix_words & _NEGATION_TR:
            return True
        if prefix_words & _NEGATION_EN:
            return True
        return False

    for word_dict, is_positive in [
        (_POS_TR, True),
        (_POS_EN, True),
        (_NEG_TR, False),
        (_NEG_EN, False),
    ]:
        for keyword, weight in word_dict.items():
            if keyword not in text_lower:
                continue

            negated = _has_negation(keyword)
            effective_weight = -weight if negated else weight

            if effective_weight > 0:
                pos_words.append(keyword)
            else:
                neg_words.append(keyword)

            total_score += effective_weight
            count += 1

    avg_score = total_score / count if count > 0 else 0.0
    avg_score = max(-1.0, min(1.0, avg_score))

    if avg_score > 0.2:
        label = "positive"
    elif avg_score < -0.2:
        label = "negative"
    else:
        label = "neutral"

    return {
        "score": round(avg_score, 2),
        "positive_words": pos_words,
        "negative_words": neg_words,
        "label": label,
    }


def analyze_reviews_batch(reviews: list[dict]) -> dict[str, Any]:
    """Analyze sentiment for a batch of reviews.

    Args:
        reviews: List of review dicts, each with at least a ``text`` key.
                 Also accepts ``content`` key as fallback.

    Returns:
        dict with keys:
        - ``avg_sentiment``: float in [-1, 1]
        - ``positive_pct``: % of reviews labeled positive
        - ``negative_pct``: % of reviews labeled negative
        - ``neutral_pct``: % of reviews labeled neutral
        - ``top_positive_words``: top 5 most frequent positive keywords
        - ``top_negative_words``: top 5 most frequent negative keywords
        - ``review_count``: total reviews analyzed
        - ``star_distribution``: dict {1..5 -> count} if ratings available
    """
    if not reviews:
        return {
            "avg_sentiment": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "top_positive_words": [],
            "top_negative_words": [],
            "review_count": 0,
            "star_distribution": {},
        }

    sentiments: list[dict] = []
    all_pos: list[str] = []
    all_neg: list[str] = []
    star_counts: dict[int, int] = {}

    for review in reviews:
        text = review.get("text") or review.get("content") or review.get("comment", "")
        result = analyze_sentiment(str(text))
        sentiments.append(result)
        all_pos.extend(result["positive_words"])
        all_neg.extend(result["negative_words"])

        # Collect star distribution
        rating = review.get("rating") or review.get("star") or review.get("rate")
        if rating is not None:
            try:
                star = int(float(rating))
                if 1 <= star <= 5:
                    star_counts[star] = star_counts.get(star, 0) + 1
            except (ValueError, TypeError):
                pass

    scores = [s["score"] for s in sentiments]
    labels = [s["label"] for s in sentiments]
    total = len(sentiments)

    pos_freq = Counter(all_pos).most_common(5)
    neg_freq = Counter(all_neg).most_common(5)

    return {
        "avg_sentiment": round(sum(scores) / total, 2),
        "positive_pct": round(labels.count("positive") / total * 100, 1),
        "negative_pct": round(labels.count("negative") / total * 100, 1),
        "neutral_pct": round(labels.count("neutral") / total * 100, 1),
        "top_positive_words": [w for w, _ in pos_freq],
        "top_negative_words": [w for w, _ in neg_freq],
        "review_count": total,
        "star_distribution": star_counts,
    }
