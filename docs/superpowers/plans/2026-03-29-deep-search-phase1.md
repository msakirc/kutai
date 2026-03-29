# Deep Search Phase 1: Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Trafilatura content extraction, BM25 relevance scoring with adaptive budget allocation, search intent inference from task context, and classifier `search_depth` field — so `web_search` automatically delivers depth-appropriate results without agent-side changes.

**Architecture:** `web_search()` infers intent from agent_type + task context + classifier hint, then for standard/deep intents runs: ddgs → page_fetch → Trafilatura extraction → BM25 scoring → budget allocation → formatted output. Quick intent uses existing fast path unchanged.

**Tech Stack:** `trafilatura` (content extraction), `bm25s` (relevance scoring), existing `aiohttp`/`beautifulsoup4`/`ddgs`/`chromadb`

---

## File Structure

```
src/tools/
├── web_search.py         # MODIFY: orchestrate pipeline, infer intent, route quick vs deep
├── page_fetch.py         # EXISTING: unchanged (Tier 0 quick fetch)
├── content_extract.py    # CREATE: Trafilatura extraction + content type detection
├── relevance.py          # CREATE: BM25 scoring + budget allocation
├── __init__.py           # MODIFY: pass task_hints through execute_tool

src/core/
├── task_classifier.py    # MODIFY: add search_depth to classification output

tests/
├── test_content_extract.py  # CREATE
├── test_relevance.py        # CREATE
├── test_deep_search_integration.py  # CREATE
```

---

### Task 1: Add dependencies (trafilatura, bm25s)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add trafilatura and bm25s to requirements.txt**

Add after the `lxml>=5.0.0` line in `requirements.txt`:

```
trafilatura>=2.0.0
bm25s>=0.2.0
```

- [ ] **Step 2: Install and verify**

Run: `pip install trafilatura bm25s`
Expected: Successful install. Verify: `python -c "import trafilatura; import bm25s; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add trafilatura and bm25s dependencies"
```

---

### Task 2: Content extraction module

**Files:**
- Create: `src/tools/content_extract.py`
- Create: `tests/test_content_extract.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_content_extract.py`:

```python
# tests/test_content_extract.py
"""Tests for content_extract module — Trafilatura-based content extraction."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.content_extract import extract_content, ExtractedContent


class TestExtractContent(unittest.TestCase):
    """Test Trafilatura-based content extraction."""

    def test_returns_extracted_content_dataclass(self):
        html = """
        <html><body>
        <article>
        <h1>Best Coffee Machines 2026</h1>
        <p>We tested 15 coffee machines to find the best options for every budget.
        The DeLonghi Dinamica costs 28000 TL and scored highest in our tests.
        The Philips 3200 at 18000 TL offers the best value for money.</p>
        </article>
        </body></html>
        """
        result = extract_content(html, url="https://example.com/coffee")
        self.assertIsInstance(result, ExtractedContent)
        self.assertIn("coffee", result.text.lower())
        self.assertEqual(result.url, "https://example.com/coffee")
        self.assertGreater(result.word_count, 10)

    def test_detects_prices(self):
        html = """
        <html><body><article>
        <p>The iPhone 15 costs $799 in the US market. In Turkey, it retails
        for 45000 TL. The Samsung Galaxy S24 is priced at 35000₺.</p>
        </article></body></html>
        """
        result = extract_content(html)
        self.assertTrue(result.has_prices)

    def test_no_prices_detected(self):
        html = """
        <html><body><article>
        <p>Transformers use attention mechanisms to process sequential data.
        The key innovation is the self-attention layer.</p>
        </article></body></html>
        """
        result = extract_content(html)
        self.assertFalse(result.has_prices)

    def test_detects_reviews(self):
        html = """
        <html><body><article>
        <p>User rating: 4.5 out of 5 stars. Based on 230 reviews.</p>
        <p>Pros: Great battery life. Cons: Heavy weight.</p>
        <div class="review">Amazing product, highly recommend!</div>
        </article></body></html>
        """
        result = extract_content(html)
        self.assertTrue(result.has_reviews)

    def test_no_reviews_detected(self):
        html = """
        <html><body><article>
        <p>Python 3.12 was released on October 2, 2023.</p>
        </article></body></html>
        """
        result = extract_content(html)
        self.assertFalse(result.has_reviews)

    def test_empty_html_returns_empty_content(self):
        result = extract_content("")
        self.assertEqual(result.text, "")
        self.assertEqual(result.word_count, 0)

    def test_extracts_title(self):
        html = """
        <html><head><title>Product Review Page</title></head>
        <body><article><p>Content here with enough words to be meaningful
        for the extraction to actually work properly.</p></article></body></html>
        """
        result = extract_content(html)
        self.assertEqual(result.title, "Product Review Page")

    def test_fallback_to_beautifulsoup(self):
        """When Trafilatura returns nothing, fall back to BeautifulSoup."""
        # Minimal HTML that Trafilatura may skip but BS4 can handle
        html = "<html><body><p>Short but valid content that should be extracted by the fallback parser at minimum.</p></body></html>"
        result = extract_content(html)
        # Should get something, even if from fallback
        self.assertIsInstance(result, ExtractedContent)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_content_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.content_extract'`

- [ ] **Step 3: Write the content_extract module**

Create `src/tools/content_extract.py`:

```python
# src/tools/content_extract.py
"""Content extraction using Trafilatura with BeautifulSoup fallback.

Standalone module — usable by web_search pipeline or directly by agents.
"""

import re
from dataclasses import dataclass, field

from src.infra.logging_config import get_logger

logger = get_logger("tools.content_extract")

# Price patterns for multiple currencies
_PRICE_PATTERNS = [
    re.compile(r"\d[\d.,]+\s*(?:TL|₺|USD|\$|EUR|€|GBP|£)", re.IGNORECASE),
    re.compile(r"(?:TL|₺|\$|€|£)\s*\d[\d.,]+", re.IGNORECASE),
    re.compile(r"\d[\d.,]+\s*(?:lira|dolar|euro|pound)", re.IGNORECASE),
]

# Review/rating patterns
_REVIEW_PATTERNS = [
    re.compile(r"\d+\.?\d*\s*(?:out of|/)\s*[5-9]\d*\s*(?:stars?)?", re.IGNORECASE),
    re.compile(r"(?:user\s*)?(?:rating|review|score)\s*[:=]\s*\d", re.IGNORECASE),
    re.compile(r"\b(?:pros?|cons?)\s*:", re.IGNORECASE),
    re.compile(r"(?:highly\s+)?recommend", re.IGNORECASE),
    re.compile(r"\d+\s*reviews?\b", re.IGNORECASE),
]


@dataclass
class ExtractedContent:
    """Result of content extraction from a single page."""
    text: str = ""
    title: str = ""
    url: str = ""
    word_count: int = 0
    has_prices: bool = False
    has_reviews: bool = False


def extract_content(html: str, url: str = "") -> ExtractedContent:
    """Extract main content from HTML using Trafilatura, with BS4 fallback.

    Args:
        html: Raw HTML string.
        url: Source URL (for metadata).

    Returns:
        ExtractedContent with extracted text and metadata.
    """
    if not html or not html.strip():
        return ExtractedContent(url=url)

    text = ""
    title = ""

    # Primary: Trafilatura
    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            include_tables=True,
            include_comments=True,
            include_links=False,
            favor_recall=True,
            url=url or None,
        ) or ""

        # Extract title via Trafilatura metadata
        meta = trafilatura.extract_metadata(html, default_url=url or None)
        if meta and meta.title:
            title = meta.title
    except Exception as e:
        logger.debug("trafilatura extraction failed, trying fallback", error=str(e)[:100])

    # Fallback: BeautifulSoup (same logic as page_fetch.py)
    if not text or len(text) < 50:
        try:
            from src.tools.page_fetch import extract_main_text
            text = extract_main_text(html, max_chars=30000)  # no truncation here — budget allocator handles it
        except Exception as e:
            logger.debug("beautifulsoup fallback also failed", error=str(e)[:100])

    # Extract title from HTML if Trafilatura didn't find one
    if not title:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
        except Exception:
            pass

    word_count = len(text.split()) if text else 0
    has_prices = any(p.search(text) for p in _PRICE_PATTERNS) if text else False
    has_reviews = any(p.search(text) for p in _REVIEW_PATTERNS) if text else False

    logger.debug(
        "content extracted",
        url=url[:80] if url else "",
        word_count=word_count,
        has_prices=has_prices,
        has_reviews=has_reviews,
    )

    return ExtractedContent(
        text=text,
        title=title,
        url=url,
        word_count=word_count,
        has_prices=has_prices,
        has_reviews=has_reviews,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_content_extract.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/content_extract.py tests/test_content_extract.py
git commit -m "feat(search): add Trafilatura content extraction module"
```

---

### Task 3: Relevance scoring and budget allocation module

**Files:**
- Create: `src/tools/relevance.py`
- Create: `tests/test_relevance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_relevance.py`:

```python
# tests/test_relevance.py
"""Tests for relevance scoring and budget allocation."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.content_extract import ExtractedContent
from src.tools.relevance import score_and_budget, BudgetedContent


def _make_content(text: str, url: str = "", has_prices: bool = False, has_reviews: bool = False) -> ExtractedContent:
    return ExtractedContent(
        text=text,
        title="",
        url=url,
        word_count=len(text.split()),
        has_prices=has_prices,
        has_reviews=has_reviews,
    )


class TestScoreAndBudget(unittest.TestCase):
    """Test BM25 scoring and budget allocation."""

    def test_returns_budgeted_content_list(self):
        contents = [
            _make_content("Coffee machines are great for morning routines and productivity"),
            _make_content("Python programming tutorial for beginners with examples"),
        ]
        result = score_and_budget(contents, query="coffee machines", total_budget=5000)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], BudgetedContent)

    def test_higher_relevance_gets_more_budget(self):
        contents = [
            _make_content("The best coffee machines of 2026 include DeLonghi and Philips models with automatic brewing"),
            _make_content("Weather forecast for Istanbul shows sunny skies and mild temperatures this week"),
        ]
        result = score_and_budget(contents, query="best coffee machines 2026", total_budget=6000)
        # First content is more relevant to the query
        coffee = next(b for b in result if "coffee" in b.content.text.lower())
        weather = next(b for b in result if "weather" in b.content.text.lower())
        self.assertGreater(coffee.allocated_chars, weather.allocated_chars)

    def test_total_budget_respected(self):
        contents = [
            _make_content("Word " * 500),
            _make_content("Text " * 500),
            _make_content("Data " * 500),
        ]
        total = 3000
        result = score_and_budget(contents, query="word", total_budget=total)
        total_allocated = sum(b.allocated_chars for b in result)
        self.assertLessEqual(total_allocated, total)

    def test_minimum_budget_per_page(self):
        contents = [
            _make_content("Highly relevant coffee machine review " * 20),
            _make_content("Completely irrelevant text " * 5),
        ]
        result = score_and_budget(contents, query="coffee machine review", total_budget=5000)
        # Even the irrelevant page should get at least 200 chars
        min_budget = min(b.allocated_chars for b in result)
        self.assertGreaterEqual(min_budget, 200)

    def test_max_budget_cap_per_page(self):
        contents = [
            _make_content("Coffee " * 100),
            _make_content("Tea " * 10),
        ]
        total = 10000
        result = score_and_budget(contents, query="coffee", total_budget=total)
        max_budget = max(b.allocated_chars for b in result)
        self.assertLessEqual(max_budget, total * 0.4 + 1)  # 40% cap

    def test_truncated_text_within_budget(self):
        long_text = "This is a sentence about coffee machines. " * 100
        contents = [_make_content(long_text)]
        result = score_and_budget(contents, query="coffee", total_budget=500)
        self.assertLessEqual(len(result[0].truncated_text), 520)  # small margin for sentence boundary

    def test_product_intent_boosts_price_pages(self):
        contents = [
            _make_content("The iPhone costs $799 and Samsung costs $699", has_prices=True),
            _make_content("Smartphones use lithium ion batteries and ARM processors"),
        ]
        result = score_and_budget(contents, query="smartphone", total_budget=5000, intent="product")
        price_page = next(b for b in result if b.content.has_prices)
        no_price = next(b for b in result if not b.content.has_prices)
        self.assertGreater(price_page.relevance_score, no_price.relevance_score)

    def test_reviews_intent_boosts_review_pages(self):
        contents = [
            _make_content("User rating: 4.5 out of 5 stars. 230 reviews.", has_reviews=True),
            _make_content("The product specifications include 8GB RAM and 256GB storage"),
        ]
        result = score_and_budget(contents, query="product", total_budget=5000, intent="reviews")
        review_page = next(b for b in result if b.content.has_reviews)
        spec_page = next(b for b in result if not b.content.has_reviews)
        self.assertGreater(review_page.relevance_score, spec_page.relevance_score)

    def test_empty_contents_returns_empty(self):
        result = score_and_budget([], query="anything", total_budget=5000)
        self.assertEqual(result, [])

    def test_single_content(self):
        contents = [_make_content("Single page about coffee machines and their features")]
        result = score_and_budget(contents, query="coffee", total_budget=5000)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].allocated_chars, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_relevance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.relevance'`

- [ ] **Step 3: Write the relevance module**

Create `src/tools/relevance.py`:

```python
# src/tools/relevance.py
"""BM25 relevance scoring and adaptive context budget allocation.

Standalone module — scores documents against a query and allocates
character budgets proportional to relevance, respecting per-page
min/max caps.
"""

from dataclasses import dataclass

from src.infra.logging_config import get_logger
from src.tools.content_extract import ExtractedContent

logger = get_logger("tools.relevance")

# Budget allocation constants
_MIN_BUDGET_PER_PAGE = 200    # chars — even irrelevant pages get this
_MAX_BUDGET_RATIO = 0.4       # no single page gets >40% of total budget

# Intent bias boosts
_INTENT_BOOSTS = {
    "product": {"prices": 0.2, "reviews": 0.0},
    "reviews": {"prices": 0.0, "reviews": 0.2},
    "market":  {"prices": 0.1, "reviews": 0.1},
    "research": {"prices": 0.0, "reviews": 0.0},
    "factual": {"prices": 0.0, "reviews": 0.0},
}


@dataclass
class BudgetedContent:
    """A page with relevance score, allocated budget, and truncated text."""
    content: ExtractedContent
    relevance_score: float
    allocated_chars: int
    truncated_text: str


def _bm25_score(contents: list[ExtractedContent], query: str) -> list[float]:
    """Score documents against query using BM25.

    Falls back to simple term-frequency if bm25s is unavailable.
    """
    if not contents:
        return []

    docs = [c.text for c in contents]

    try:
        import bm25s

        # Tokenize
        query_tokens = bm25s.tokenize([query])
        doc_tokens = bm25s.tokenize(docs)

        # Build index and score
        retriever = bm25s.BM25()
        retriever.index(doc_tokens)
        scores_array, _ = retriever.retrieve(query_tokens, corpus=docs, k=len(docs))

        # scores_array is shape (1, k) — extract the single query's scores
        # Map back to original document order
        score_map = {}
        results_docs, results_scores = retriever.retrieve(query_tokens, corpus=list(range(len(docs))), k=len(docs))
        for idx, score in zip(results_docs[0], results_scores[0]):
            score_map[int(idx)] = float(score)

        return [score_map.get(i, 0.0) for i in range(len(docs))]

    except Exception as e:
        logger.debug("bm25s scoring failed, using term frequency fallback", error=str(e)[:100])

        # Simple TF fallback
        query_terms = set(query.lower().split())
        scores = []
        for doc in docs:
            doc_lower = doc.lower()
            matches = sum(1 for t in query_terms if t in doc_lower)
            scores.append(matches / max(len(query_terms), 1))
        return scores


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at a sentence boundary, not mid-word."""
    if len(text) <= max_chars:
        return text

    # Find last sentence-ending punctuation before max_chars
    truncated = text[:max_chars]
    for sep in [". ", ".\n", "! ", "? "]:
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars * 0.5:  # don't cut too aggressively
            return truncated[:last_sep + 1]

    # Fallback: cut at last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return truncated[:last_space] + "..."
    return truncated + "..."


def score_and_budget(
    contents: list[ExtractedContent],
    query: str,
    total_budget: int = 12000,
    intent: str = "factual",
) -> list[BudgetedContent]:
    """Score pages by relevance and allocate character budgets.

    Args:
        contents: List of extracted page contents.
        query: The search query for relevance scoring.
        total_budget: Total character budget across all pages.
        intent: Search intent for bias adjustments.

    Returns:
        List of BudgetedContent, sorted by relevance (highest first).
    """
    if not contents:
        return []

    # 1. BM25 scoring
    raw_scores = _bm25_score(contents, query)

    # 2. Apply intent biases
    boosts = _INTENT_BOOSTS.get(intent, {"prices": 0.0, "reviews": 0.0})
    adjusted_scores = []
    for score, content in zip(raw_scores, contents):
        adj = score
        if content.has_prices:
            adj += boosts["prices"]
        if content.has_reviews:
            adj += boosts["reviews"]
        adjusted_scores.append(max(adj, 0.01))  # floor to avoid zero

    # 3. Allocate budgets proportional to scores
    total_score = sum(adjusted_scores)
    max_per_page = int(total_budget * _MAX_BUDGET_RATIO)

    budgets = []
    for score in adjusted_scores:
        raw_budget = int((score / total_score) * total_budget) if total_score > 0 else total_budget // len(contents)
        clamped = max(_MIN_BUDGET_PER_PAGE, min(raw_budget, max_per_page))
        budgets.append(clamped)

    # 4. Scale down if total exceeds budget
    budget_sum = sum(budgets)
    if budget_sum > total_budget:
        scale = total_budget / budget_sum
        budgets = [max(_MIN_BUDGET_PER_PAGE, int(b * scale)) for b in budgets]

    # 5. Build results, sorted by relevance
    results = []
    for content, score, budget in zip(contents, adjusted_scores, budgets):
        truncated = _truncate_at_sentence(content.text, budget) if content.text else ""
        results.append(BudgetedContent(
            content=content,
            relevance_score=round(score, 4),
            allocated_chars=budget,
            truncated_text=truncated,
        ))

    results.sort(key=lambda b: b.relevance_score, reverse=True)

    logger.debug(
        "budget allocation complete",
        pages=len(results),
        total_budget=total_budget,
        used=sum(len(b.truncated_text) for b in results),
        intent=intent,
    )

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_relevance.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/relevance.py tests/test_relevance.py
git commit -m "feat(search): add BM25 relevance scoring and budget allocation"
```

---

### Task 4: Task classifier — add search_depth field

**Files:**
- Modify: `src/core/task_classifier.py`
- Test: `tests/test_content_extract.py` (extend existing classifier tests if they exist, or add inline)

- [ ] **Step 1: Write a test for search_depth in classification output**

Create or add to a test file. First check if classifier tests exist:

Run: `ls tests/test_task_classifier* tests/test_classifier* 2>/dev/null`

If no existing file, create `tests/test_search_depth.py`:

```python
# tests/test_search_depth.py
"""Tests for search_depth classification field."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.task_classifier import _classify_search_depth


class TestSearchDepthClassification(unittest.TestCase):
    """Test keyword-based search_depth classification."""

    def test_deep_for_analysis_keywords(self):
        self.assertEqual(_classify_search_depth("Analyze the market for robot vacuums"), "deep")
        self.assertEqual(_classify_search_depth("Research the best laptops in detail"), "deep")

    def test_standard_for_comparison_keywords(self):
        self.assertEqual(_classify_search_depth("Compare iPhone vs Samsung"), "standard")
        self.assertEqual(_classify_search_depth("What is the price of RTX 4070"), "standard")
        self.assertEqual(_classify_search_depth("DeLonghi kahve makinesi fiyat"), "standard")
        self.assertEqual(_classify_search_depth("Sony WH-1000XM5 review"), "standard")

    def test_quick_for_simple_queries(self):
        self.assertEqual(_classify_search_depth("What is Python"), "quick")
        self.assertEqual(_classify_search_depth("Hello"), "quick")

    def test_none_for_non_search_tasks(self):
        self.assertEqual(_classify_search_depth("Write a function to sort a list"), "none")
        self.assertEqual(_classify_search_depth("Fix the bug in main.py"), "none")

    def test_case_insensitive(self):
        self.assertEqual(_classify_search_depth("ANALYZE market trends"), "deep")
        self.assertEqual(_classify_search_depth("COMPARE products"), "standard")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search_depth.py -v`
Expected: FAIL — `ImportError: cannot import name '_classify_search_depth'`

- [ ] **Step 3: Add search_depth classification to task_classifier.py**

In `src/core/task_classifier.py`, add the following after the `_classify_shopping_sub_intent` function (after line 121):

```python
# ─── Search Depth Detection ──────────────────────────────────────────────

_SEARCH_DEPTH_RULES: list[tuple[str, list[str]]] = [
    ("deep", [
        "analyze", "analyse", "research in detail", "market analysis",
        "competitor analysis", "in-depth", "comprehensive",
        "multi-source", "synthesize", "synthesis",
    ]),
    ("standard", [
        "price", "fiyat", "fiyatı", "ne kadar", "how much",
        "compare", "karşılaştır", "vs ", " vs ",
        "review", "inceleme", "best ", "en iyi",
        "recommend", "tavsiye", "öneri",
    ]),
    ("none", [
        "write ", "fix ", "implement", "refactor", "debug",
        "create ", "build ", "deploy", "test ",
    ]),
]


def _classify_search_depth(text: str) -> str:
    """Classify how much web search depth a task needs.

    Returns: "deep", "standard", "quick", or "none".
    """
    text_lower = text.lower()
    for depth, keywords in _SEARCH_DEPTH_RULES:
        if any(kw in text_lower for kw in keywords):
            return depth
    return "quick"  # default for general questions
```

Also modify the LLM classifier prompt (around line 80-92) to add the `search_depth` field. Find the line:

```python
- priority: "critical" | "high" | "normal" | "low" | "background"
```

Add after it:

```python
- search_depth: how much web research does this need?
  "deep" — market analysis, multi-source research, review synthesis
  "standard" — product info, comparison, how-to with examples
  "quick" — simple fact, definition, date, status
  "none" — no web search needed (code tasks, file operations)
```

And update the example JSON response (around line 92) from:

```python
Respond as: {{"agent_type": "coder", "difficulty": 6, "needs_tools": true, "needs_vision": false, "needs_thinking": false, "local_only": false, "priority": "normal"}}"""
```

To:

```python
Respond as: {{"agent_type": "coder", "difficulty": 6, "needs_tools": true, "needs_vision": false, "needs_thinking": false, "local_only": false, "priority": "normal", "search_depth": "none"}}"""
```

Finally, update the `classify_task` function to extract `search_depth` from the LLM response and fall back to keyword classification. Find where the LLM result is parsed and add:

```python
search_depth = result.get("search_depth") or _classify_search_depth(title + " " + description)
```

Store it in the returned `TaskClassification` — check the dataclass/dict structure and add `search_depth` to it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search_depth.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/task_classifier.py tests/test_search_depth.py
git commit -m "feat(classifier): add search_depth field for web search intent routing"
```

---

### Task 5: Pass task hints through execute_tool to web_search

**Files:**
- Modify: `src/tools/__init__.py:980` (execute_tool function)
- Modify: `src/agents/base.py:1510` (execute_tool call site)

- [ ] **Step 1: Modify execute_tool to accept and forward task_hints**

In `src/tools/__init__.py`, find `execute_tool` (line 980):

```python
async def execute_tool(tool_name: str, agent_type: str | None = None, **kwargs: Any) -> str:
```

Add `task_hints` parameter:

```python
async def execute_tool(tool_name: str, agent_type: str | None = None, task_hints: dict | None = None, **kwargs: Any) -> str:
```

Inside the function, before the tool function is called, inject `task_hints` for `web_search`:

Find where kwargs are filtered and the tool function is called. Add this logic:

```python
    # Inject task_hints for web_search (search_depth, agent_type, shopping_sub_intent)
    if tool_name == "web_search" and task_hints:
        kwargs["_task_hints"] = task_hints
```

- [ ] **Step 2: Modify base.py to pass task_hints when calling execute_tool**

In `src/agents/base.py`, find the `execute_tool` call (around line 1510):

```python
tool_output = await asyncio.wait_for(
    execute_tool(
        tool_name, agent_type=self.name, **tool_args
    ),
    timeout=_tool_timeout,
)
```

Change to:

```python
# Build task hints for tools that need context
_hints = {
    "agent_type": self.name,
    "search_depth": task.get("search_depth") or task.get("context", {}).get("search_depth"),
    "shopping_sub_intent": task.get("shopping_sub_intent"),
}

tool_output = await asyncio.wait_for(
    execute_tool(
        tool_name, agent_type=self.name, task_hints=_hints, **tool_args
    ),
    timeout=_tool_timeout,
)
```

Note: `task` is already available in scope — it's the task dict passed to `execute()`. Verify the variable name by checking the method signature.

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `pytest tests/test_web_search_integration.py tests/test_page_fetch.py -v`
Expected: All pass (task_hints defaults to None, no behavior change for existing calls).

- [ ] **Step 4: Commit**

```bash
git add src/tools/__init__.py src/agents/base.py
git commit -m "feat(tools): pass task_hints through execute_tool for context-aware web search"
```

---

### Task 6: Wire up the deep search pipeline in web_search.py

**Files:**
- Modify: `src/tools/web_search.py`
- Create: `tests/test_deep_search_integration.py`

- [ ] **Step 1: Write integration tests**

Create `tests/test_deep_search_integration.py`:

```python
# tests/test_deep_search_integration.py
"""Integration tests for the deep search pipeline."""

import asyncio
import importlib
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ws_mod = importlib.import_module("src.tools.web_search")


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestIntentInference(unittest.TestCase):
    """Test _infer_search_intent logic."""

    def test_search_depth_takes_priority(self):
        intent, params = _ws_mod._infer_search_intent({"search_depth": "deep", "agent_type": "assistant"})
        self.assertEqual(intent, "research")

    def test_search_depth_quick(self):
        intent, params = _ws_mod._infer_search_intent({"search_depth": "quick"})
        self.assertEqual(intent, "factual")

    def test_search_depth_standard(self):
        intent, params = _ws_mod._infer_search_intent({"search_depth": "standard"})
        self.assertEqual(intent, "product")

    def test_shopping_sub_intent_compare(self):
        intent, _ = _ws_mod._infer_search_intent({"shopping_sub_intent": "compare", "agent_type": "shopping_advisor"})
        self.assertEqual(intent, "product")

    def test_shopping_sub_intent_research(self):
        intent, _ = _ws_mod._infer_search_intent({"shopping_sub_intent": "research", "agent_type": "product_researcher"})
        self.assertEqual(intent, "market")

    def test_shopping_sub_intent_purchase_advice(self):
        intent, _ = _ws_mod._infer_search_intent({"shopping_sub_intent": "purchase_advice"})
        self.assertEqual(intent, "reviews")

    def test_agent_type_researcher(self):
        intent, _ = _ws_mod._infer_search_intent({"agent_type": "researcher"})
        self.assertEqual(intent, "research")

    def test_agent_type_analyst(self):
        intent, _ = _ws_mod._infer_search_intent({"agent_type": "analyst"})
        self.assertEqual(intent, "research")

    def test_agent_type_deal_analyst(self):
        intent, _ = _ws_mod._infer_search_intent({"agent_type": "deal_analyst"})
        self.assertEqual(intent, "market")

    def test_agent_type_assistant_defaults_factual(self):
        intent, _ = _ws_mod._infer_search_intent({"agent_type": "assistant"})
        self.assertEqual(intent, "factual")

    def test_no_hints_defaults_factual(self):
        intent, _ = _ws_mod._infer_search_intent({})
        self.assertEqual(intent, "factual")

    def test_params_have_required_fields(self):
        _, params = _ws_mod._infer_search_intent({"search_depth": "deep"})
        self.assertIn("max_results", params)
        self.assertIn("max_chars_per_page", params)
        self.assertIn("total_budget", params)


class TestDeepSearchPipeline(unittest.TestCase):
    """Test that deep intents trigger extraction + budgeting pipeline."""

    def test_deep_intent_uses_extraction_pipeline(self):
        """When intent is deep, content_extract and relevance modules are used."""
        mock_ddgs_results = [
            {"title": "Coffee Review", "body": "Best machines reviewed", "href": "https://example.com/coffee"},
            {"title": "Tech Blog", "body": "Latest gadgets", "href": "https://example.com/tech"},
        ]

        mock_html_pages = {
            "https://example.com/coffee": "<html><body><article><p>Detailed coffee machine review with DeLonghi at 28000 TL and Philips at 18000 TL. User rating: 4.5 out of 5 stars based on 200 reviews. We tested extensively.</p></article></body></html>",
            "https://example.com/tech": "<html><body><article><p>Latest tech gadgets roundup covering smartphones, laptops, and accessories for the modern professional user.</p></article></body></html>",
        }

        with patch.object(_ws_mod, "_DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = mock_ddgs_results
            mock_ddgs_cls.return_value = mock_instance

            with patch("src.tools.page_fetch.fetch_pages", new_callable=AsyncMock, return_value=mock_html_pages):
                with patch.object(_ws_mod, "_task_hints", {"search_depth": "deep"}):
                    result = run_async(_ws_mod.web_search("coffee machine reviews", max_results=5))

        # Deep pipeline should produce structured output with relevance-ordered content
        self.assertIn("coffee", result.lower())

    def test_quick_intent_uses_existing_fast_path(self):
        """When intent is factual/quick, the existing fast path is used (no Trafilatura)."""
        mock_ddgs_results = [
            {"title": "Python Docs", "body": "Official documentation", "href": "https://docs.python.org"},
        ]

        with patch.object(_ws_mod, "_DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = mock_ddgs_results
            mock_ddgs_cls.return_value = mock_instance

            with patch("src.tools.page_fetch.fetch_pages", new_callable=AsyncMock, return_value={}):
                with patch.object(_ws_mod, "_task_hints", {}):
                    result = run_async(_ws_mod.web_search("what is python"))

        self.assertIn("Python Docs", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_deep_search_integration.py -v`
Expected: FAIL — `_infer_search_intent` doesn't exist yet.

- [ ] **Step 3: Add intent inference and deep pipeline to web_search.py**

In `src/tools/web_search.py`, add the following after the imports (after line 16):

```python
from dataclasses import dataclass

# Thread-local task hints — set by execute_tool before calling web_search
_task_hints: dict = {}

# Intent parameters
@dataclass
class _SearchParams:
    max_results: int
    max_chars_per_page: int
    total_budget: int
    use_deep_pipeline: bool

_INTENT_PARAMS = {
    "factual":  _SearchParams(max_results=5,  max_chars_per_page=1500, total_budget=5000,  use_deep_pipeline=False),
    "product":  _SearchParams(max_results=7,  max_chars_per_page=2000, total_budget=10000, use_deep_pipeline=True),
    "reviews":  _SearchParams(max_results=8,  max_chars_per_page=2500, total_budget=15000, use_deep_pipeline=True),
    "market":   _SearchParams(max_results=10, max_chars_per_page=3000, total_budget=20000, use_deep_pipeline=True),
    "research": _SearchParams(max_results=10, max_chars_per_page=3000, total_budget=20000, use_deep_pipeline=True),
}


def _infer_search_intent(hints: dict) -> tuple[str, _SearchParams]:
    """Infer search intent from task context. Returns (intent_name, params)."""
    # 1. Classifier search_depth takes priority
    depth = hints.get("search_depth")
    if depth == "deep":
        return "research", _INTENT_PARAMS["research"]
    if depth == "standard":
        return "product", _INTENT_PARAMS["product"]
    if depth == "quick":
        return "factual", _INTENT_PARAMS["factual"]

    # 2. Shopping sub-intent
    sub = hints.get("shopping_sub_intent")
    if sub in ("research", "exploration"):
        return "market", _INTENT_PARAMS["market"]
    if sub in ("compare", "price_check", "deal_hunt", "upgrade"):
        return "product", _INTENT_PARAMS["product"]
    if sub in ("purchase_advice", "complaint_return_help"):
        return "reviews", _INTENT_PARAMS["reviews"]

    # 3. Agent type defaults
    agent = hints.get("agent_type", "")
    agent_map = {
        "researcher": "research",
        "analyst": "research",
        "deal_analyst": "market",
        "shopping_advisor": "product",
        "product_researcher": "product",
    }
    intent = agent_map.get(agent, "factual")
    return intent, _INTENT_PARAMS[intent]
```

Then modify the `web_search` function signature and the Method 1 block. Replace the current `web_search` function (line 377 onwards) with:

```python
async def web_search(query: str, max_results: int = 5, search_type: str = "web", _task_hints: dict | None = None) -> str:
    """
    Search the web with auto-scaling depth based on task context.

    Quick searches (factual intent): ddgs + page_fetch (fast, ~3-5s).
    Deep searches (product/reviews/market/research): ddgs + Trafilatura + BM25 budgeting (~10-20s).

    Fallback chain: ddgs+pages → Perplexica/Vane → SearXNG → curl.
    """
    global _task_hints as _current_hints
    hints = _task_hints or _current_hints or {}
    intent, params = _infer_search_intent(hints)

    effective_max = max(max_results, params.max_results)

    logger.info("web search query", query=query, max_results=effective_max, intent=intent, search_type=search_type)

    # Phase D: Check existing web knowledge before live search
    try:
        from src.memory.vector_store import query as vquery, is_ready as vs_ready
        import time as _t
        if vs_ready():
            cached = await vquery(text=query, collection="web_knowledge", top_k=3)
            fresh_results = [
                r for r in cached
                if r.get("distance", 1.0) < 0.5
                and (_t.time() - r.get("metadata", {}).get("timestamp", 0)) < 43200
            ]
            if fresh_results:
                lines = [f"**Cached web knowledge for '{query}':**\n"]
                for r in fresh_results:
                    text = r.get("text", "")[:500]
                    lines.append(f"- {text}")
                lines.append(
                    "\n[Retrieved from cached web knowledge. "
                    "The information may be up to 12 hours old.]"
                )
                logger.debug("Returning cached web knowledge for: %s", query[:60])
                return "\n".join(lines)
    except Exception:
        pass

    # Check for degraded capability
    try:
        from src.infra.runtime_state import runtime_state
        is_degraded = "web_search" in runtime_state.get("degraded_capabilities", [])
    except Exception:
        is_degraded = False

    # Method 1 (primary): DuckDuckGo + page fetch (quick or deep)
    if _DDGS is not None:
        try:
            results = _DDGS().text(query, max_results=effective_max)
            if results:
                logger.debug("ddgs search ok", count=len(results), intent=intent)

                urls = [r.get("href", "") for r in results if r.get("href")]

                if params.use_deep_pipeline and urls:
                    # Deep path: fetch pages → Trafilatura → BM25 → budget
                    result_text = await _deep_search_pipeline(
                        query, results, urls, intent, params
                    )
                else:
                    # Quick path: existing page_fetch + simple format
                    result_text = await _quick_search_pipeline(query, results, urls)

                await _embed_web_results(query, result_text)
                return result_text
        except Exception as e:
            logger.warning("ddgs primary search failed", error=str(e))

    # Method 2-4 (fallbacks): Perplexica → SearXNG → curl (unchanged)
    if not is_degraded:
        perplexica_result = await _search_perplexica(query, effective_max, search_type)
        if perplexica_result:
            logger.debug("using perplexica fallback for web search")
            lines = [
                "## AI-Synthesized Answer (from Perplexica)\n",
                perplexica_result["answer"],
            ]
            if perplexica_result["sources"]:
                lines.append("\n### Sources")
                for i, src in enumerate(perplexica_result["sources"], 1):
                    title = src.get("title", "Untitled")
                    url = src.get("url", "")
                    lines.append(f"- [{title}]({url})")
            lines.append(
                "\n**Note: This answer is already synthesized from multiple "
                "sources. Use it as your final answer unless something "
                "specific is missing.**"
            )
            result_text = "\n".join(lines)
            await _embed_web_results(query, result_text)
            return result_text

    searxng_result = await _search_searxng_direct(query, effective_max)
    if searxng_result and searxng_result.count("**") >= 6:
        asyncio.ensure_future(_embed_web_results(query, searxng_result))
        return searxng_result

    try:
        safe_query = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={safe_query}&format=json&no_html=1&no_redirect=1"
        result = await run_shell(f'curl -s --max-time 10 "{url}"', timeout=15)
        if result.startswith("\u2705"):
            result = result[1:].strip()
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return f"Search returned non-JSON response for '{query}':\n{result[:1000]}"

        lines = []
        if data.get("Abstract"):
            lines.append(f"**Summary:** {data['Abstract']}")
            if data.get("AbstractURL"):
                lines.append(f"Source: {data['AbstractURL']}")
        for topic in data.get("RelatedTopics", [])[:effective_max]:
            if isinstance(topic, dict) and "Text" in topic:
                text = topic["Text"][:200]
                url = topic.get("FirstURL", "")
                lines.append(f"- {text}\n  {url}")
        if lines:
            return f"Search results for '{query}':\n\n" + "\n\n".join(lines)

        scrape_url = f"https://html.duckduckgo.com/html/?q={safe_query}"
        scrape_result = await run_shell(
            f'curl -s --max-time 10 "{scrape_url}" | grep -oP \'<a rel="nofollow" class="result__a" href="[^"]*">[^<]*</a>\' | head -5',
            timeout=15,
        )
        if scrape_result.startswith("\u2705"):
            scrape_result = scrape_result[1:].strip()
        if scrape_result and "\u274c" not in scrape_result:
            return f"Search results for '{query}':\n\n{scrape_result}"
        return f"No results found for '{query}'. All search backends failed."

    except Exception as e:
        logger.exception("web search all backends failed", error=str(e))
        return f"Search error: {e}"
```

Now add the two pipeline functions before `web_search`:

```python
async def _quick_search_pipeline(query: str, ddgs_results: list, urls: list) -> str:
    """Existing fast path: page_fetch + simple format."""
    page_contents = {}
    if urls:
        try:
            from src.tools.page_fetch import fetch_pages
            page_contents = await fetch_pages(urls, max_pages=3, max_chars=1500)
            logger.debug("page_fetch: fetched pages", count=len(page_contents))
        except Exception as e:
            logger.debug("page_fetch: skipped", error=str(e)[:100])

    lines = []
    for i, r in enumerate(ddgs_results, 1):
        title = r.get("title", "No title")
        body = r.get("body", "")[:200]
        href = r.get("href", "")
        parts = [f"{i}. **{title}**\n   {body}\n   {href}"]
        if href in page_contents:
            parts.append(f"   ---\n   {page_contents[href]}")
        lines.append("\n".join(parts))

    return f"Search results for '{query}':\n\n" + "\n\n".join(lines)


async def _deep_search_pipeline(
    query: str, ddgs_results: list, urls: list, intent: str, params: _SearchParams
) -> str:
    """Deep path: fetch pages → Trafilatura → BM25 → budget allocation."""
    from src.tools.page_fetch import fetch_pages
    from src.tools.content_extract import extract_content
    from src.tools.relevance import score_and_budget

    # Fetch pages (use existing page_fetch, but more pages and more content)
    page_htmls = await fetch_pages(urls, max_pages=params.max_results, max_chars=50000)
    logger.debug("deep pipeline: fetched pages", count=len(page_htmls))

    if not page_htmls:
        # Fall back to quick pipeline if no pages fetched
        logger.debug("deep pipeline: no pages fetched, falling back to quick")
        return await _quick_search_pipeline(query, ddgs_results, urls)

    # Extract content with Trafilatura
    contents = []
    for url, html in page_htmls.items():
        extracted = extract_content(html, url=url)
        if extracted.text and extracted.word_count > 10:
            contents.append(extracted)

    if not contents:
        logger.debug("deep pipeline: no content extracted, falling back to quick")
        return await _quick_search_pipeline(query, ddgs_results, urls)

    # Score relevance and allocate budgets
    budgeted = score_and_budget(contents, query, total_budget=params.total_budget, intent=intent)

    # Format output: snippets from ddgs + budgeted page content
    lines = []
    # First: ddgs snippet summary (always included)
    for i, r in enumerate(ddgs_results, 1):
        title = r.get("title", "No title")
        body = r.get("body", "")[:150]
        href = r.get("href", "")
        lines.append(f"{i}. **{title}** — {body} ({href})")

    # Then: detailed content from top pages (ordered by relevance)
    lines.append("\n---\n**Detailed content (by relevance):**\n")
    for b in budgeted:
        if not b.truncated_text:
            continue
        title = b.content.title or b.content.url.split("/")[-1] or "Untitled"
        tags = []
        if b.content.has_prices:
            tags.append("prices")
        if b.content.has_reviews:
            tags.append("reviews")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"### {title}{tag_str}")
        lines.append(f"Source: {b.content.url}")
        lines.append(b.truncated_text)
        lines.append("")

    return "\n".join(lines)
```

Also update the `_task_hints` handling — in `src/tools/__init__.py`, where `web_search` is called via `execute_tool`, the `_task_hints` kwarg needs to be forwarded. Since `execute_tool` already filters kwargs to match function parameters, and we added `_task_hints` to `web_search`'s signature, it should flow through. But we also need to set the module-level `_task_hints` as a fallback. In `execute_tool`, before calling the tool function, add:

```python
    if tool_name == "web_search" and task_hints:
        import src.tools.web_search as _ws
        _ws._task_hints = task_hints
```

- [ ] **Step 4: Fix the global alias syntax error**

The line `global _task_hints as _current_hints` is invalid Python. Replace the first 3 lines of `web_search` with:

```python
    hints = _task_hints or {}
    if _task_hints:
        # Reset after reading (one-shot, set by execute_tool)
        pass
    intent, params = _infer_search_intent(hints)
```

Actually simpler — just use the `_task_hints` param directly:

```python
    hints = _task_hints or _task_hints_module_var_if_set or {}
```

The cleanest approach: `web_search` accepts `_task_hints` as a kwarg (set by execute_tool's filtered kwargs), and also reads the module-level `_task_hints` as fallback:

```python
    hints = _task_hints if _task_hints else globals().get("_task_hints", {}) or {}
```

Wait — this gets confusing. Let me simplify. The `_task_hints` parameter to `web_search` IS the hints. The module-level `_task_hints` variable is set by `execute_tool` as a secondary mechanism. Use whichever is available:

```python
async def web_search(query: str, max_results: int = 5, search_type: str = "web", _task_hints: dict | None = None) -> str:
    hints = _task_hints or {}
    intent, params = _infer_search_intent(hints)
```

This is clean. The parameter `_task_hints` is passed by `execute_tool` via the kwargs mechanism.

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_deep_search_integration.py tests/test_web_search_integration.py tests/test_content_extract.py tests/test_relevance.py tests/test_page_fetch.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/tools/web_search.py src/tools/__init__.py tests/test_deep_search_integration.py
git commit -m "feat(search): wire up deep search pipeline with intent inference and budget allocation"
```

---

### Task 7: End-to-end smoke tests

**Files:**
- No code changes — verification only.

- [ ] **Step 1: Test quick search (factual intent)**

Run:
```bash
python -c "
import asyncio, logging
logging.disable(logging.DEBUG)
from src.tools.web_search import web_search
result = asyncio.get_event_loop().run_until_complete(web_search('capital of Australia'))
print(f'Length: {len(result)} chars')
print(result[:500])
" 2>/dev/null
```

Expected: Quick results (~3-5s), mentions Canberra, no "Detailed content" section.

- [ ] **Step 2: Test deep search (research intent via _task_hints)**

Run:
```bash
python -c "
import asyncio, logging
logging.disable(logging.DEBUG)
from src.tools.web_search import web_search
result = asyncio.get_event_loop().run_until_complete(
    web_search('best coffee machines 2026 comparison', _task_hints={'search_depth': 'deep'})
)
print(f'Length: {len(result)} chars')
print(result[:3000])
" 2>/dev/null
```

Expected: Deep results (~10-20s), contains "Detailed content (by relevance)" section, multiple pages with content, possibly `[prices]` tags.

- [ ] **Step 3: Test product intent**

Run:
```bash
python -c "
import asyncio, logging
logging.disable(logging.DEBUG)
from src.tools.web_search import web_search
result = asyncio.get_event_loop().run_until_complete(
    web_search('iPhone 15 price Turkey', _task_hints={'shopping_sub_intent': 'price_check', 'agent_type': 'shopping_advisor'})
)
print(f'Length: {len(result)} chars')
print(result[:2000])
" 2>/dev/null
```

Expected: Product-focused results, price data prioritized.

- [ ] **Step 4: Test import from agent context**

Run:
```bash
python -c "from src.tools import web_search; print('ok:', callable(web_search))"
```

Expected: `ok: True`

- [ ] **Step 5: Commit if any fixes were needed**

```bash
git add -u
git commit -m "fix(search): address issues found in deep search e2e testing"
```

Only commit if changes were made.
