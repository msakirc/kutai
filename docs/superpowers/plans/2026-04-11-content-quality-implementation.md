# Content Quality Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone content quality assessment package and wire it into KutAI at 13 integration points, replacing scattered inline checks with a unified assess/salvage API.

**Architecture:** Standalone `packages/content_quality/` package with zero external dependencies (pure stdlib). Four heuristic detectors (size, header repetition, paragraph repetition, token entropy) compose into a single `assess()` call returning a structured result. KutAI integration is surgical — each callsite replaces one inline check with one `assess()` call. Streaming abort via callback pattern keeps the router decoupled.

**Tech Stack:** Python 3.10, stdlib only (`re`, `collections`, `math`, `dataclasses`). pytest for testing.

**Design spec:** `docs/superpowers/specs/2026-04-11-content-quality-design.md`

---

## File Structure

### New files (package)

| File | Responsibility |
|------|---------------|
| `packages/content_quality/pyproject.toml` | Package metadata, zero deps |
| `packages/content_quality/src/content_quality/__init__.py` | Public API re-exports: assess, salvage, make_stream_callback, ContentQualityResult |
| `packages/content_quality/src/content_quality/assessor.py` | ContentQualityResult dataclass + assess() orchestrator |
| `packages/content_quality/src/content_quality/detectors.py` | Four pure detection functions |
| `packages/content_quality/src/content_quality/salvager.py` | Section deduplication |
| `packages/content_quality/src/content_quality/streaming.py` | Streaming callback factory |

### New files (tests)

| File | Responsibility |
|------|---------------|
| `packages/content_quality/tests/__init__.py` | Test package marker |
| `packages/content_quality/tests/test_detectors.py` | Unit tests for each detector |
| `packages/content_quality/tests/test_assessor.py` | Unit tests for assess() |
| `packages/content_quality/tests/test_salvager.py` | Unit tests for salvage() |
| `packages/content_quality/tests/test_streaming.py` | Unit tests for streaming callback |
| `tests/test_content_quality_integration.py` | Integration tests for KutAI wiring |

### Modified files (KutAI integration)

| File | Lines | Change |
|------|-------|--------|
| `src/workflows/engine/hooks.py` | 563-577, 611-622, 862-876, 905-917, ~434, ~1194 | Replace 4 inline `_detect_repetition_ratio` calls + add 2 summary validations |
| `src/core/grading.py` | 24-38, 41-51, 91-145, 188-190 | Add WELL_FORMED/COHERENT to prompt, parse them, pre-grading assess() |
| `src/core/router.py` | 43-86, 988-993 | Add on_chunk callback to streaming + call_model |
| `src/core/llm_dispatcher.py` | 116-147 | Thread on_chunk through request() |
| `src/agents/base.py` | 856-862, 1649-1659, 2828-2881 | Dependency validation, streaming callback, self-reflect validation |
| `src/memory/episodic.py` | 82-89 | Gate storage on assess() |

---

## Task 1: Package Skeleton

**Files:**
- Create: `packages/content_quality/pyproject.toml`
- Create: `packages/content_quality/src/content_quality/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "content-quality"
version = "0.1.0"
description = "Heuristic content quality assessment — detect degenerate, repetitive, or oversized LLM output"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create __init__.py with public API stubs**

```python
"""Content quality assessment — detect degenerate LLM output."""

from .assessor import ContentQualityResult, assess
from .salvager import salvage
from .streaming import make_stream_callback

__all__ = ["assess", "salvage", "make_stream_callback", "ContentQualityResult"]
```

This will fail to import until we create the modules — that's expected.

- [ ] **Step 3: Commit**

```bash
git add packages/content_quality/pyproject.toml packages/content_quality/src/content_quality/__init__.py
git commit -m "feat(content-quality): package skeleton with public API stubs"
```

---

## Task 2: Detectors — Size and Header Repetition

**Files:**
- Create: `packages/content_quality/src/content_quality/detectors.py`
- Create: `packages/content_quality/tests/__init__.py`
- Create: `packages/content_quality/tests/test_detectors.py`

- [ ] **Step 1: Write failing tests for check_size**

```python
"""Tests for content_quality.detectors."""

from content_quality.detectors import (
    check_size,
    check_header_repetition,
)


class TestCheckSize:
    def test_under_limit(self):
        score, breached, reason = check_size("hello world", max_size=20_000)
        assert score == 11
        assert breached is False
        assert reason is None

    def test_at_limit(self):
        text = "x" * 20_000
        score, breached, reason = check_size(text, max_size=20_000)
        assert breached is False

    def test_over_limit(self):
        text = "x" * 20_001
        score, breached, reason = check_size(text, max_size=20_000)
        assert breached is True
        assert reason == "size_exceeded"

    def test_hard_cap(self):
        text = "x" * 50_001
        score, breached, reason = check_size(text, max_size=999_999)
        assert breached is True
        assert reason == "size_exceeded"

    def test_empty(self):
        score, breached, reason = check_size("", max_size=20_000)
        assert score == 0
        assert breached is False
```

- [ ] **Step 2: Write failing tests for check_header_repetition**

Add to the same test file:

```python
class TestCheckHeaderRepetition:
    def test_no_headers(self):
        text = "Just plain text without any markdown headers."
        ratio, breached, reason = check_header_repetition(text)
        assert ratio == 0.0
        assert breached is False

    def test_few_sections_skipped(self):
        text = "intro\n## A\ncontent\n## A\ncontent\n## B\ncontent"
        ratio, breached, reason = check_header_repetition(text)
        # Only 3 sections — below minimum of 5, should return 0.0
        assert ratio == 0.0
        assert breached is False

    def test_unique_sections(self):
        sections = [f"## Section {i}\nContent for section {i}" for i in range(6)]
        text = "Intro\n" + "\n".join(sections)
        ratio, breached, reason = check_header_repetition(text)
        assert ratio == 0.0
        assert breached is False

    def test_degenerate_repetition(self):
        # 3 unique headers, each repeated 2x = 6 sections total, 3 duplicated
        sections = []
        for name in ["Component Usage", "Component Usage Summary", "API Reference"]:
            for _ in range(2):
                sections.append(f"## {name}\nSome content here")
        text = "Intro\n" + "\n".join(sections)
        ratio, breached, reason = check_header_repetition(text)
        # "Component Usage" and "Component Usage Summary" normalize to same thing
        # So we have: 4x "component usage" + 2x "api reference" = 6 sections
        # duplicated = (4-1) + (2-1) = 4 out of 6 => 0.67
        assert ratio > 0.4
        assert breached is True
        assert reason == "header_repetition"

    def test_suffix_normalization(self):
        sections = [
            "## Component Usage\ncontent",
            "## Component Usage Summary\ncontent",
            "## Component Usage Examples\ncontent",
            "## Component Usage Notes\ncontent",
            "## Component Usage Details\ncontent",
            "## API Reference\ncontent",
        ]
        text = "Intro\n" + "\n".join(sections)
        ratio, breached, reason = check_header_repetition(text)
        # First 5 all normalize to "component usage" => 4 duplicated out of 6
        assert ratio > 0.4
        assert breached is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd packages/content_quality && python -m pytest tests/test_detectors.py -v`
Expected: ImportError — detectors module doesn't exist yet.

- [ ] **Step 4: Implement check_size and check_header_repetition**

```python
"""Heuristic content quality detectors.

Each function returns (score, breached, reason_tag).
- score: numeric measurement (size in chars, ratio 0.0-1.0, entropy in bits)
- breached: True if threshold exceeded
- reason_tag: short string for ContentQualityResult.reasons, or None if not breached
"""

from __future__ import annotations

import re
from collections import Counter

# Hard ceiling — no single output should ever exceed this regardless of config
HARD_CAP = 50_000

# Minimum number of ## sections required before header repetition check fires.
# Below this count, short documents with a few repeated headers get false positives.
MIN_SECTIONS_FOR_HEADER_CHECK = 5

# Suffixes stripped during header normalization — these are the patterns
# 9B models append when repeating sections ("Usage", "Usage Summary", "Usage Examples")
_HEADER_SUFFIX_RE = re.compile(
    r'\s+(summary|examples?|notes|details)\s*$', re.IGNORECASE,
)


def check_size(
    text: str, max_size: int = 20_000,
) -> tuple[int, bool, str | None]:
    """Check if text exceeds size limit.

    Args:
        text: Input text.
        max_size: Soft ceiling (caller-configurable). Clamped to HARD_CAP.

    Returns:
        (size_chars, breached, "size_exceeded" | None)
    """
    effective_max = min(max_size, HARD_CAP)
    size = len(text)
    if size > effective_max:
        return size, True, "size_exceeded"
    return size, False, None


def check_header_repetition(text: str) -> tuple[float, bool, str | None]:
    """Detect duplicate markdown ## section headers.

    Splits text by ``\\n## ``, normalizes headers (strips trailing
    summary/examples/notes/details suffixes, lowercases), and counts
    how many sections share a normalized header with at least one other.

    Requires >= MIN_SECTIONS_FOR_HEADER_CHECK sections to trigger
    (avoids false positives on short documents).

    Returns:
        (repetition_ratio 0.0-1.0, breached, "header_repetition" | None)
    """
    sections = text.split("\n## ")
    if len(sections) < MIN_SECTIONS_FOR_HEADER_CHECK + 1:  # +1 for content before first ##
        return 0.0, False, None

    norm_headers: list[str] = []
    for sec in sections[1:]:  # skip content before first ##
        header = sec.split("\n", 1)[0].strip()
        norm = _HEADER_SUFFIX_RE.sub("", header.lower()).strip()
        norm_headers.append(norm)

    if not norm_headers:
        return 0.0, False, None

    counts = Counter(norm_headers)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    ratio = duplicated / len(norm_headers)

    if ratio > 0.4:
        return ratio, True, "header_repetition"
    return ratio, False, None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/content_quality && python -m pytest tests/test_detectors.py::TestCheckSize tests/test_detectors.py::TestCheckHeaderRepetition -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/content_quality/src/content_quality/detectors.py packages/content_quality/tests/
git commit -m "feat(content-quality): size and header repetition detectors with tests"
```

---

## Task 3: Detectors — Paragraph Repetition and Token Entropy

**Files:**
- Modify: `packages/content_quality/src/content_quality/detectors.py`
- Modify: `packages/content_quality/tests/test_detectors.py`

- [ ] **Step 1: Write failing tests for check_paragraph_repetition**

Add to `test_detectors.py`:

```python
from content_quality.detectors import (
    check_size,
    check_header_repetition,
    check_paragraph_repetition,
    check_token_entropy,
)


class TestCheckParagraphRepetition:
    def test_no_repetition(self):
        text = "First paragraph about apples.\n\nSecond paragraph about oranges.\n\nThird paragraph about bananas."
        ratio, breached, reason = check_paragraph_repetition(text)
        assert ratio == 0.0
        assert breached is False

    def test_short_text_skipped(self):
        text = "Short.\n\nAlso short."
        ratio, breached, reason = check_paragraph_repetition(text)
        assert ratio == 0.0
        assert breached is False

    def test_repeated_paragraphs(self):
        block = "This is a detailed paragraph about component usage patterns in React applications."
        unique = "This paragraph is unique and different from the others."
        # 4 copies of same block + 1 unique = 3 duplicated out of 5
        text = "\n\n".join([block, block, unique, block, block])
        ratio, breached, reason = check_paragraph_repetition(text)
        assert ratio > 0.3
        assert breached is True
        assert reason == "paragraph_repetition"

    def test_whitespace_normalization(self):
        block1 = "Same content   with   extra   spaces."
        block2 = "Same content with extra spaces."
        unique1 = "Unique paragraph one."
        unique2 = "Unique paragraph two."
        unique3 = "Unique paragraph three."
        text = "\n\n".join([block1, block2, unique1, unique2, unique3])
        ratio, breached, reason = check_paragraph_repetition(text)
        # block1 and block2 normalize to same thing => 1 dup out of 5 = 0.2
        assert breached is False
```

- [ ] **Step 2: Write failing tests for check_token_entropy**

Add to `test_detectors.py`:

```python
class TestCheckTokenEntropy:
    def test_high_entropy_natural_text(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Meanwhile, a curious cat watches from the windowsill. "
            "Birds sing melodiously in the ancient oak tree nearby."
        )
        entropy, breached, reason = check_token_entropy(text)
        assert entropy > 3.0
        assert breached is False

    def test_low_entropy_garbage(self):
        text = " ".join(["the"] * 200)
        entropy, breached, reason = check_token_entropy(text)
        assert entropy < 1.0
        assert breached is True
        assert reason == "low_entropy"

    def test_moderate_repetition(self):
        # Mix of repeated and unique — should be above threshold
        words = ["hello", "world"] * 20 + ["unique", "words", "here", "today", "test"]
        text = " ".join(words)
        entropy, breached, reason = check_token_entropy(text)
        assert entropy > 1.5
        # This is borderline — just verify it returns a valid result
        assert isinstance(breached, bool)

    def test_empty_text(self):
        entropy, breached, reason = check_token_entropy("")
        assert entropy == 0.0
        assert breached is False

    def test_single_token(self):
        entropy, breached, reason = check_token_entropy("hello")
        assert entropy == 0.0
        assert breached is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd packages/content_quality && python -m pytest tests/test_detectors.py::TestCheckParagraphRepetition tests/test_detectors.py::TestCheckTokenEntropy -v`
Expected: ImportError — functions not defined yet.

- [ ] **Step 4: Implement check_paragraph_repetition and check_token_entropy**

Add to `detectors.py`:

```python
import math

# Minimum paragraph blocks before paragraph repetition check fires
MIN_PARAGRAPHS_FOR_CHECK = 4

# Minimum word count before entropy check fires
MIN_WORDS_FOR_ENTROPY = 20


def check_paragraph_repetition(text: str) -> tuple[float, bool, str | None]:
    """Detect repeated paragraph blocks.

    Splits text by double-newlines, normalizes whitespace in each block,
    hashes them, and counts blocks that share a hash with 2+ others.

    Returns:
        (repetition_ratio 0.0-1.0, breached, "paragraph_repetition" | None)
    """
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if len(blocks) < MIN_PARAGRAPHS_FOR_CHECK:
        return 0.0, False, None

    # Normalize: collapse whitespace, lowercase
    normalized = [re.sub(r'\s+', ' ', b).lower() for b in blocks]
    counts = Counter(normalized)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    ratio = duplicated / len(normalized)

    if ratio > 0.3:
        return ratio, True, "paragraph_repetition"
    return ratio, False, None


def check_token_entropy(text: str) -> tuple[float, bool, str | None]:
    """Measure Shannon entropy of whitespace-split tokens.

    Natural English text has ~9-10 bits of entropy per token.
    Repetitive garbage (\"the the the...\") has < 3 bits.

    Returns:
        (entropy_bits, breached, "low_entropy" | None)
    """
    tokens = text.split()
    if len(tokens) < MIN_WORDS_FOR_ENTROPY:
        return 0.0, False, None

    total = len(tokens)
    counts = Counter(tokens)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )

    if entropy < 3.0:
        return entropy, True, "low_entropy"
    return entropy, False, None
```

- [ ] **Step 5: Run all detector tests**

Run: `cd packages/content_quality && python -m pytest tests/test_detectors.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/content_quality/src/content_quality/detectors.py packages/content_quality/tests/test_detectors.py
git commit -m "feat(content-quality): paragraph repetition and token entropy detectors"
```

---

## Task 4: Assessor — ContentQualityResult and assess()

**Files:**
- Create: `packages/content_quality/src/content_quality/assessor.py`
- Create: `packages/content_quality/tests/test_assessor.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for content_quality.assessor."""

from content_quality.assessor import ContentQualityResult, assess


class TestContentQualityResult:
    def test_summary_clean(self):
        r = ContentQualityResult(
            size=500, max_size=20_000,
            repetition_ratio=0.0, paragraph_repetition=0.0,
            token_entropy=8.5, is_degenerate=False, reasons=[],
        )
        assert "degenerate" not in r.summary.lower()

    def test_summary_degenerate(self):
        r = ContentQualityResult(
            size=30_000, max_size=20_000,
            repetition_ratio=0.6, paragraph_repetition=0.0,
            token_entropy=8.5, is_degenerate=True,
            reasons=["size_exceeded", "header_repetition"],
        )
        s = r.summary
        assert "size_exceeded" in s
        assert "header_repetition" in s


class TestAssess:
    def test_clean_text(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a normal paragraph with varied vocabulary and structure. "
            "It discusses multiple topics including weather, technology, and food."
        )
        result = assess(text)
        assert result.is_degenerate is False
        assert result.reasons == []
        assert result.size == len(text)

    def test_oversized(self):
        text = "word " * 5000  # 25K chars
        result = assess(text, max_size=20_000)
        assert result.is_degenerate is True
        assert "size_exceeded" in result.reasons

    def test_header_repetition_detected(self):
        sections = []
        for _ in range(3):
            sections.append("## Component Usage\nSome content about usage")
            sections.append("## Component Usage Summary\nMore content")
        sections.append("## API Reference\nAPI docs here")
        text = "Intro\n" + "\n".join(sections)
        result = assess(text)
        assert result.is_degenerate is True
        assert "header_repetition" in result.reasons

    def test_low_entropy_detected(self):
        text = " ".join(["the"] * 200)
        result = assess(text)
        assert result.is_degenerate is True
        assert "low_entropy" in result.reasons

    def test_paragraph_repetition_detected(self):
        block = "This is a detailed paragraph about component patterns in modern web applications."
        unique = "This paragraph is entirely unique and different."
        text = "\n\n".join([block, block, unique, block, block])
        result = assess(text)
        assert result.is_degenerate is True
        assert "paragraph_repetition" in result.reasons

    def test_max_size_clamped_to_hard_cap(self):
        text = "x" * 50_001
        result = assess(text, max_size=999_999)
        assert result.is_degenerate is True
        assert result.max_size == 50_000

    def test_multiple_reasons(self):
        # Oversized + low entropy
        text = " ".join(["the"] * 10_000)  # ~40K chars of garbage
        result = assess(text, max_size=20_000)
        assert result.is_degenerate is True
        assert "size_exceeded" in result.reasons
        assert "low_entropy" in result.reasons
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/content_quality && python -m pytest tests/test_assessor.py -v`
Expected: ImportError — assessor module doesn't exist yet.

- [ ] **Step 3: Implement assessor.py**

```python
"""Content quality assessment orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .detectors import (
    HARD_CAP,
    check_header_repetition,
    check_paragraph_repetition,
    check_size,
    check_token_entropy,
)


@dataclass
class ContentQualityResult:
    """Structured result from assess()."""

    size: int
    max_size: int
    repetition_ratio: float
    paragraph_repetition: float
    token_entropy: float
    is_degenerate: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One-line human-readable summary for logs and error messages."""
        if not self.is_degenerate:
            return f"ok ({self.size} chars)"
        parts = []
        for reason in self.reasons:
            if reason == "size_exceeded":
                parts.append(f"size_exceeded ({self.size} > {self.max_size})")
            elif reason == "header_repetition":
                parts.append(f"header_repetition ({self.repetition_ratio:.2f})")
            elif reason == "paragraph_repetition":
                parts.append(f"paragraph_repetition ({self.paragraph_repetition:.2f})")
            elif reason == "low_entropy":
                parts.append(f"low_entropy ({self.token_entropy:.2f} bits)")
            else:
                parts.append(reason)
        return "degenerate: " + ", ".join(parts)


def assess(text: str, max_size: int = 20_000) -> ContentQualityResult:
    """Run all heuristic checks on any text.

    No side effects. No LLM calls. Pure string analysis.

    Args:
        text: Input text to assess.
        max_size: Soft ceiling for size check. Clamped to HARD_CAP (50,000).

    Returns:
        ContentQualityResult with is_degenerate=True if ANY check fails.
    """
    effective_max = min(max_size, HARD_CAP)
    reasons: list[str] = []

    size_val, size_breached, size_reason = check_size(text, effective_max)
    if size_breached and size_reason:
        reasons.append(size_reason)

    header_ratio, header_breached, header_reason = check_header_repetition(text)
    if header_breached and header_reason:
        reasons.append(header_reason)

    para_ratio, para_breached, para_reason = check_paragraph_repetition(text)
    if para_breached and para_reason:
        reasons.append(para_reason)

    entropy_val, entropy_breached, entropy_reason = check_token_entropy(text)
    if entropy_breached and entropy_reason:
        reasons.append(entropy_reason)

    return ContentQualityResult(
        size=size_val,
        max_size=effective_max,
        repetition_ratio=header_ratio,
        paragraph_repetition=para_ratio,
        token_entropy=entropy_val,
        is_degenerate=bool(reasons),
        reasons=reasons,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/content_quality && python -m pytest tests/test_assessor.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/content_quality/src/content_quality/assessor.py packages/content_quality/tests/test_assessor.py
git commit -m "feat(content-quality): assess() orchestrator with ContentQualityResult"
```

---

## Task 5: Salvager

**Files:**
- Create: `packages/content_quality/src/content_quality/salvager.py`
- Create: `packages/content_quality/tests/test_salvager.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for content_quality.salvager."""

from content_quality.salvager import salvage


class TestSalvage:
    def test_no_headers_returns_original(self):
        text = "Just plain text without any markdown structure."
        assert salvage(text) == text

    def test_unique_sections_unchanged(self):
        text = (
            "# Intro\n\nSome intro text.\n\n"
            "## Section A\nContent for A.\n\n"
            "## Section B\nContent for B.\n\n"
            "## Section C\nContent for C."
        )
        assert salvage(text) == text

    def test_deduplicates_repeated_sections(self):
        text = (
            "# Doc\n\n"
            "## Component Usage\nReal content about components.\n\n"
            "## API Reference\nAPI docs here.\n\n"
            "## Component Usage Summary\nRepeated content about components.\n\n"
            "## Component Usage Examples\nMore repeated content.\n\n"
            "## Component Usage Notes\nEven more repeated content.\n\n"
            "## Component Usage Details\nStill more repeated content."
        )
        result = salvage(text)
        # Should keep first "Component Usage" and "API Reference", drop the rest
        assert "## Component Usage\n" in result
        assert "## API Reference\n" in result
        assert "## Component Usage Summary" not in result
        assert "## Component Usage Examples" not in result
        assert "## Component Usage Notes" not in result
        assert "## Component Usage Details" not in result

    def test_keeps_first_occurrence(self):
        text = (
            "## Setup\nFirst setup content — the real one.\n\n"
            "## Usage\nUsage info.\n\n"
            "## Setup\nDuplicate setup — should be dropped.\n\n"
            "## Setup Summary\nAnother duplicate.\n\n"
            "## Testing\nTest info.\n\n"
            "## Deployment\nDeploy info."
        )
        result = salvage(text)
        assert "First setup content" in result
        assert "Duplicate setup" not in result
        assert "Another duplicate" not in result
        assert "## Usage\n" in result
        assert "## Testing\n" in result

    def test_empty_sections_dropped(self):
        text = (
            "## Section A\n\n\n"
            "## Section B\nReal content here.\n\n"
            "## Section C\nMore content.\n\n"
            "## Section D\nContent.\n\n"
            "## Section E\nContent."
        )
        result = salvage(text)
        # Section A has no content — should be dropped
        assert "## Section A" not in result
        assert "## Section B\n" in result

    def test_returns_empty_when_nothing_salvageable(self):
        # All sections are empty or whitespace-only
        text = "## A\n\n## B\n  \n## C\n\n## D\n\n## E\n"
        result = salvage(text)
        assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/content_quality && python -m pytest tests/test_salvager.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement salvager.py**

```python
"""Section deduplication for degenerate markdown output."""

from __future__ import annotations

import re

_HEADER_SUFFIX_RE = re.compile(
    r'\s+(summary|examples?|notes|details)\s*$', re.IGNORECASE,
)


def salvage(text: str) -> str:
    """Deduplicate repeated markdown sections.

    1. Split text by ``## `` headers
    2. Normalize each header (strip trailing suffixes, lowercase)
    3. Keep first occurrence of each normalized header
    4. Drop sections with no content (empty body)
    5. Reassemble

    Returns:
        Cleaned text, or empty string if nothing salvageable survives.
        Non-markdown text (no ## headers) is returned unchanged.
    """
    # Split into sections. First element is content before any ## header.
    parts = text.split("\n## ")

    if len(parts) <= 1:
        # No ## headers — not markdown-structured, return as-is
        return text

    preamble = parts[0]
    sections = parts[1:]

    seen_normalized: set[str] = set()
    kept: list[str] = []

    for sec in sections:
        header, _, body = sec.partition("\n")
        header = header.strip()
        body = body.strip()

        # Skip sections with no content
        if not body:
            continue

        # Normalize header for dedup comparison
        norm = _HEADER_SUFFIX_RE.sub("", header.lower()).strip()

        if norm in seen_normalized:
            continue  # duplicate — drop it

        seen_normalized.add(norm)
        kept.append(f"## {header}\n{body}")

    if not kept:
        return ""

    result_parts = []
    preamble_stripped = preamble.strip()
    if preamble_stripped:
        result_parts.append(preamble_stripped)
    result_parts.extend(kept)

    return "\n\n".join(result_parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/content_quality && python -m pytest tests/test_salvager.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/content_quality/src/content_quality/salvager.py packages/content_quality/tests/test_salvager.py
git commit -m "feat(content-quality): salvage() section deduplication"
```

---

## Task 6: Streaming Callback

**Files:**
- Create: `packages/content_quality/src/content_quality/streaming.py`
- Create: `packages/content_quality/tests/test_streaming.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for content_quality.streaming."""

from content_quality.streaming import make_stream_callback


class TestMakeStreamCallback:
    def test_clean_text_no_abort(self):
        cb = make_stream_callback(max_size=20_000, check_interval=100)
        # Simulate streaming clean text
        accumulated = ""
        for word in ["Hello ", "world, ", "this ", "is ", "clean ", "text. "] * 20:
            accumulated += word
            assert cb(accumulated) is False

    def test_size_abort_immediate(self):
        cb = make_stream_callback(max_size=100, check_interval=50)
        # Size check runs every call — should abort as soon as we exceed
        text = "x" * 101
        assert cb(text) is True

    def test_repetition_abort_at_interval(self):
        cb = make_stream_callback(max_size=50_000, check_interval=200)
        # Build degenerate text with repeated sections
        sections = []
        for _ in range(4):
            sections.append("## Component Usage\nSome content about usage patterns.")
            sections.append("## Component Usage Summary\nMore content.")
        sections.append("## API Reference\nUnique content here.")
        degenerate = "Intro\n" + "\n".join(sections)
        # Feed it all at once — should abort on next interval check
        assert cb(degenerate) is True

    def test_no_check_before_interval(self):
        cb = make_stream_callback(max_size=50_000, check_interval=1000)
        # Build text that IS degenerate but under check_interval length
        # Should NOT abort because we haven't hit the interval yet
        short_degenerate = " ".join(["the"] * 50)  # ~200 chars, low entropy
        assert cb(short_degenerate) is False

    def test_stateful_tracks_last_check(self):
        cb = make_stream_callback(max_size=50_000, check_interval=100)
        # First call at 150 chars — triggers check at interval
        text = "Hello world. " * 12  # ~156 chars
        cb(text)
        # Extend by small amount — should NOT re-check (haven't passed next interval)
        text += "tiny"
        result = cb(text)
        assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/content_quality && python -m pytest tests/test_streaming.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement streaming.py**

```python
"""Streaming abort callback factory."""

from __future__ import annotations

from typing import Callable

from .detectors import HARD_CAP, check_size


def make_stream_callback(
    max_size: int = 20_000,
    check_interval: int = 4096,
) -> Callable[[str], bool]:
    """Create a stateful callback for streaming quality checks.

    Returns a function ``callback(accumulated_text) -> should_abort``.

    - Size check runs on every call (cheap len() comparison).
    - Full quality assessment runs every ``check_interval`` chars.
    - Returns True (abort) when content is degenerate.

    Args:
        max_size: Maximum output size. Clamped to HARD_CAP.
        check_interval: Run full assessment every N chars of new content.
    """
    effective_max = min(max_size, HARD_CAP)
    last_checked_len = 0

    def callback(accumulated: str) -> bool:
        nonlocal last_checked_len

        # Size check — always runs (cheap)
        if len(accumulated) > effective_max:
            return True

        # Full assessment — only at intervals
        if len(accumulated) - last_checked_len >= check_interval:
            last_checked_len = len(accumulated)
            # Import here to avoid circular import at module level
            from .assessor import assess
            result = assess(accumulated, max_size=effective_max)
            return result.is_degenerate

        return False

    return callback
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/content_quality && python -m pytest tests/test_streaming.py -v`
Expected: All PASS.

- [ ] **Step 5: Run full package test suite**

Run: `cd packages/content_quality && python -m pytest tests/ -v`
Expected: All tests PASS. Package is complete.

- [ ] **Step 6: Commit**

```bash
git add packages/content_quality/src/content_quality/streaming.py packages/content_quality/tests/test_streaming.py
git commit -m "feat(content-quality): streaming abort callback with interval checks"
```

---

## Task 7: Integrate into hooks.py — Final Quality Gate + Workspace Recovery

**Files:**
- Modify: `src/workflows/engine/hooks.py:862-876,905-917`
- Modify: `tests/test_content_quality_integration.py` (create)

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_content_quality_integration.py`:

```python
"""Integration tests for content_quality wiring in KutAI."""

import pytest
from content_quality import assess, salvage


class TestHooksIntegration:
    """Test the assess/salvage patterns used in hooks.py."""

    def test_final_quality_gate_rejects_degenerate(self):
        """Simulates hooks.py line 905-917 replacement."""
        sections = []
        for _ in range(3):
            sections.append("## Component Usage\nContent about usage")
            sections.append("## Component Usage Summary\nMore content")
        sections.append("## API Ref\nUnique")
        output_value = "Intro\n" + "\n".join(sections)

        step_max = 20_000
        cq = assess(output_value, max_size=step_max)
        assert cq.is_degenerate is True
        assert "header_repetition" in cq.reasons

    def test_final_quality_gate_passes_clean(self):
        output_value = "## Overview\nGood content.\n\n## Details\nMore good content."
        cq = assess(output_value, max_size=20_000)
        assert cq.is_degenerate is False

    def test_workspace_salvage_keeps_good_content(self):
        """Simulates hooks.py line 862-876 replacement."""
        text = (
            "# Doc\n\n"
            "## Component Usage\nReal useful content.\n\n"
            "## API Reference\nAPI docs.\n\n"
            "## Component Usage Summary\nRepeated.\n\n"
            "## Component Usage Examples\nRepeated again.\n\n"
            "## Component Usage Notes\nRepeated more.\n\n"
            "## Component Usage Details\nRepeated yet again."
        )
        cq = assess(text)
        assert cq.is_degenerate is True

        cleaned = salvage(text)
        assert cleaned  # not empty — salvageable
        assert "## Component Usage\n" in cleaned
        assert "## API Reference\n" in cleaned
        assert "## Component Usage Summary" not in cleaned

    def test_workspace_salvage_returns_empty_for_pure_garbage(self):
        text = "## A\n\n## B\n\n## C\n\n## D\n\n## E\n"
        cq = assess(text)
        cleaned = salvage(text)
        assert cleaned == ""

    def test_oversized_output_rejected(self):
        text = "word " * 5000  # ~25K chars
        cq = assess(text, max_size=20_000)
        assert cq.is_degenerate is True
        assert "size_exceeded" in cq.reasons

    def test_max_output_chars_from_schema(self):
        """Artifact schema can override default max_size."""
        text = "x" * 35_000
        # Default 20K would reject this
        cq_default = assess(text, max_size=20_000)
        assert cq_default.is_degenerate is True

        # But schema says 40K is OK
        cq_custom = assess(text, max_size=40_000)
        assert cq_custom.is_degenerate is False
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_content_quality_integration.py -v`
Expected: All PASS (these test the package API, not the hooks.py wiring yet).

- [ ] **Step 3: Replace final quality gate in hooks.py (lines 905-917)**

In `src/workflows/engine/hooks.py`, replace the final quality gate block:

Old code (lines 905-917):
```python
    # ── Final quality gate before storing ──
    if output_value:
        rep = _detect_repetition_ratio(output_value)
        if rep > 0.4:
            result["status"] = "failed"
            result["error"] = (
                f"Output is {rep:.0%} repetitive — degenerate content rejected"
            )
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' output rejected: "
                f"{rep:.0%} repetition ({len(output_value)} chars)"
            )
            return
```

New code:
```python
    # ── Final quality gate before storing ──
    if output_value:
        from content_quality import assess as cq_assess
        _artifact_schema = ctx.get("artifact_schema", {})
        _step_max = _artifact_schema.get("max_output_chars", 20_000)
        cq = cq_assess(output_value, max_size=_step_max)
        if cq.is_degenerate:
            result["status"] = "failed"
            result["error"] = f"Degenerate content rejected: {cq.summary}"
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' output rejected: "
                f"{cq.summary} ({len(output_value)} chars)"
            )
            return
```

- [ ] **Step 4: Replace workspace file recovery in hooks.py (lines 862-876)**

Old code (lines 862-876):
```python
                        # Quality gate: delete garbage files so next
                        # attempt starts clean instead of building on them
                        rep_ratio = _detect_repetition_ratio(file_content)
                        if rep_ratio > 0.4:
                            logger.warning(
                                f"[Workflow Hook] Deleting degenerate "
                                f"workspace file '{name}{ext}' "
                                f"({len(file_content)} chars, "
                                f"{rep_ratio:.0%} repetition)"
                            )
                            try:
                                os.remove(fpath)
                            except OSError:
                                pass
                            break
```

New code:
```python
                        from content_quality import assess as cq_assess, salvage as cq_salvage
                        cq = cq_assess(file_content)
                        if cq.is_degenerate:
                            cleaned = cq_salvage(file_content)
                            if cleaned:
                                logger.info(
                                    f"[Workflow Hook] Salvaged degenerate "
                                    f"workspace file '{name}{ext}' "
                                    f"({len(file_content)} -> {len(cleaned)} chars)"
                                )
                                try:
                                    with open(fpath, "w", encoding="utf-8") as wf:
                                        wf.write(cleaned)
                                except OSError:
                                    pass
                                file_parts.append(cleaned)
                            else:
                                logger.warning(
                                    f"[Workflow Hook] Deleting unsalvageable "
                                    f"workspace file '{name}{ext}' "
                                    f"({len(file_content)} chars, {cq.summary})"
                                )
                                try:
                                    os.remove(fpath)
                                except OSError:
                                    pass
                            break
```

- [ ] **Step 5: Verify hooks.py imports work**

Run: `python -c "from src.workflows.engine.hooks import post_execute_workflow_step; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/workflows/engine/hooks.py tests/test_content_quality_integration.py
git commit -m "feat(hooks): replace inline repetition checks with content_quality.assess/salvage"
```

---

## Task 8: Integrate into hooks.py — _prev_output + Context Injection + Summaries

**Files:**
- Modify: `src/workflows/engine/hooks.py:563-577,611-622,~434,~1194`

- [ ] **Step 1: Replace _prev_output sanitization (lines 563-577)**

Old code:
```python
    # ── Sanitize _prev_output before injection ──
    # If previous output is degenerate (>40% repeated sections), discard
    # it entirely — building on garbage just produces more garbage.
    _prev = ctx.get("_prev_output")
    if _prev:
        _prev = _unwrap_envelope(_prev)
        if _detect_repetition_ratio(_prev) > 0.4:
            logger.warning(
                f"[Workflow Hook] Discarding degenerate _prev_output "
                f"({len(_prev)} chars, "
                f"{_detect_repetition_ratio(_prev):.0%} repetition)"
            )
            _prev = None
        else:
            ctx["_prev_output"] = _prev  # store cleaned version
```

New code:
```python
    # ── Sanitize _prev_output before injection ──
    _prev = ctx.get("_prev_output")
    if _prev:
        _prev = _unwrap_envelope(_prev)
        from content_quality import assess as cq_assess, salvage as cq_salvage
        _prev_cq = cq_assess(_prev)
        if _prev_cq.is_degenerate:
            cleaned = cq_salvage(_prev)
            if cleaned:
                logger.info(
                    f"[Workflow Hook] Salvaged degenerate _prev_output "
                    f"({len(_prev)} -> {len(cleaned)} chars)"
                )
                _prev = cleaned
            else:
                logger.warning(
                    f"[Workflow Hook] Discarding unsalvageable _prev_output "
                    f"({len(_prev)} chars, {_prev_cq.summary})"
                )
                _prev = None
        else:
            ctx["_prev_output"] = _prev  # store cleaned version
```

- [ ] **Step 2: Replace context injection filter (lines 611-622)**

Old code:
```python
                            content = _unwrap_envelope(content)
                            if content and len(content) > 100:
                                if _detect_repetition_ratio(content) > 0.4:
                                    logger.warning(
                                        f"[Workflow Hook] Skipping degenerate "
                                        f"workspace file {name}{ext}"
                                    )
                                else:
                                    parts.append(
                                        f"\n\n## Already Written: {name}{ext}\n"
                                        f"```\n{content[:3000]}\n```"
                                    )
```

New code:
```python
                            content = _unwrap_envelope(content)
                            if content and len(content) > 100:
                                from content_quality import assess as cq_assess
                                _file_cq = cq_assess(content)
                                if _file_cq.is_degenerate:
                                    logger.warning(
                                        f"[Workflow Hook] Skipping degenerate "
                                        f"workspace file {name}{ext} ({_file_cq.summary})"
                                    )
                                else:
                                    parts.append(
                                        f"\n\n## Already Written: {name}{ext}\n"
                                        f"```\n{content[:3000]}\n```"
                                    )
```

- [ ] **Step 3: Add assessment to _llm_summarize output (around line 432-434)**

After line 433 (`if summary and len(summary) > 50:`), wrap the return with a quality check:

Old code:
```python
        summary = response.get("content", "").strip()
        if summary and len(summary) > 50:
            return summary
```

New code:
```python
        summary = response.get("content", "").strip()
        if summary and len(summary) > 50:
            from content_quality import assess as cq_assess
            _sum_cq = cq_assess(summary)
            if _sum_cq.is_degenerate:
                logger.warning(
                    f"[Workflow Hook] LLM summary degenerate ({_sum_cq.summary}), "
                    f"falling back to structural summary"
                )
            else:
                return summary
```

If degenerate, it falls through to the existing `return _structural_summary(text)` on line 440.

- [ ] **Step 4: Add assessment to _generate_phase_summary output (around line 1194-1197)**

After `summary_text = "\n".join(lines).rstrip()` (line 1194), before storing:

Old code:
```python
    summary_text = "\n".join(lines).rstrip()

    summary_artifact_name = f"phase_{phase_num}_summary"
    await store.store(mission_id, summary_artifact_name, summary_text)
```

New code:
```python
    summary_text = "\n".join(lines).rstrip()

    from content_quality import assess as cq_assess
    _phase_cq = cq_assess(summary_text)
    if _phase_cq.is_degenerate:
        logger.warning(
            f"[Workflow Hook] Phase summary degenerate ({_phase_cq.summary}), "
            f"using structural fallback"
        )
        summary_text = _structural_summary(summary_text)

    summary_artifact_name = f"phase_{phase_num}_summary"
    await store.store(mission_id, summary_artifact_name, summary_text)
```

- [ ] **Step 5: Add deprecation comment to _detect_repetition_ratio**

At the top of `_detect_repetition_ratio` (line 87):

```python
def _detect_repetition_ratio(text: str) -> float:
    """Return 0.0-1.0 indicating how much of the text is repetitive.

    .. deprecated::
        Use ``content_quality.assess()`` instead. This function is kept
        for reference but all callsites now use the content_quality package.
    ...
```

- [ ] **Step 6: Verify hooks.py imports work**

Run: `python -c "from src.workflows.engine.hooks import enrich_task_description, post_execute_workflow_step; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/workflows/engine/hooks.py
git commit -m "feat(hooks): complete content_quality integration — prev_output, injection, summaries"
```

---

## Task 9: Integrate into grading.py — Pre-grading + Prompt + Parsing

**Files:**
- Modify: `src/core/grading.py:24-38,41-51,91-145,188-190`

- [ ] **Step 1: Add pre-grading content quality check (after line 190)**

Old code (lines 188-190):
```python
    result_text = task.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return GradeResult(passed=False, raw="auto-fail: trivial/empty output")
```

New code:
```python
    result_text = task.get("result", "")
    if not result_text or len(str(result_text).strip()) < 10:
        return GradeResult(passed=False, raw="auto-fail: trivial/empty output")

    from content_quality import assess as cq_assess
    _grade_cq = cq_assess(str(result_text))
    if _grade_cq.is_degenerate:
        return GradeResult(passed=False, raw=f"auto-fail: {_grade_cq.summary}")
```

- [ ] **Step 2: Add WELL_FORMED and COHERENT to GRADING_PROMPT**

Old code (lines 24-38):
```python
GRADING_PROMPT = """Evaluate this task result.

Task: {title}
Description: {description}
Result: {response}

Reply with EXACTLY these fields, one per line:
RELEVANT: YES or NO
COMPLETE: YES or NO
VERDICT: PASS or FAIL
SITUATION: one line, what type of problem was solved
STRATEGY: one line, what approach worked
TOOLS: comma-separated list of tools used effectively
PREFERENCE: one-line user preference signal observed in this task, or NONE
INSIGHT: one-line reusable learning from this task, or NONE"""
```

New code:
```python
GRADING_PROMPT = """Evaluate this task result.

Task: {title}
Description: {description}
Result: {response}

Reply with EXACTLY these fields, one per line:
RELEVANT: YES or NO
COMPLETE: YES or NO
VERDICT: PASS or FAIL
WELL_FORMED: PASS or FAIL (no repeated sections, no garbage, structurally sound)
COHERENT: PASS or FAIL (output makes logical sense end-to-end)
SITUATION: one line, what type of problem was solved
STRATEGY: one line, what approach worked
TOOLS: comma-separated list of tools used effectively
PREFERENCE: one-line user preference signal observed in this task, or NONE
INSIGHT: one-line reusable learning from this task, or NONE"""
```

- [ ] **Step 3: Add well_formed and coherent to GradeResult dataclass**

Old code (lines 41-51):
```python
@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    situation: str = ""
    strategy: str = ""
    tools: list[str] = field(default_factory=list)
    preference: str = ""
    insight: str = ""
    raw: str = ""
```

New code:
```python
@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    well_formed: Optional[bool] = None
    coherent: Optional[bool] = None
    situation: str = ""
    strategy: str = ""
    tools: list[str] = field(default_factory=list)
    preference: str = ""
    insight: str = ""
    raw: str = ""
```

- [ ] **Step 4: Parse WELL_FORMED and COHERENT in parse_grade_response**

In the field extraction block (around lines 101-117), after `insight` parsing, add:

```python
    well_formed = _parse_yes_no(raw, "WELL_FORMED")
    coherent = _parse_yes_no(raw, "COHERENT")
```

Then in each GradeResult constructor in the cascade (lines 121-143), add `well_formed=well_formed, coherent=coherent` to the kwargs.

Also add the override rule: if `well_formed` explicitly parsed as `False`, force `passed=False`:

After the existing cascade level 1 (line 119-125), modify:

Old cascade 1:
```python
    # Cascade 1: VERDICT present
    if verdict is not None:
        return GradeResult(
            passed=verdict, relevant=relevant, complete=complete,
            situation=situation, strategy=strategy, tools=tools,
            preference=preference, insight=insight, raw=raw,
        )
```

New cascade 1:
```python
    # Cascade 1: VERDICT present
    if verdict is not None:
        # WELL_FORMED: FAIL overrides VERDICT — garbage output can't pass
        effective_passed = verdict if well_formed is not False else False
        return GradeResult(
            passed=effective_passed, relevant=relevant, complete=complete,
            well_formed=well_formed, coherent=coherent,
            situation=situation, strategy=strategy, tools=tools,
            preference=preference, insight=insight, raw=raw,
        )
```

Apply the same `well_formed` override and field additions to cascade 2 and cascade 3 GradeResult constructors.

Cascade 2 (lines 128-133):
```python
    if relevant is not None and complete is not None:
        derived = relevant and complete
        effective_passed = derived if well_formed is not False else False
        return GradeResult(
            passed=effective_passed, relevant=relevant, complete=complete,
            well_formed=well_formed, coherent=coherent,
            situation=situation, strategy=strategy, tools=tools,
            preference=preference, insight=insight, raw=raw,
        )
```

Cascade 3 (lines 135-143):
```python
    bare = re.search(r'\bPASS\b', raw, re.IGNORECASE)
    if bare:
        effective_passed = True if well_formed is not False else False
        return GradeResult(passed=effective_passed, well_formed=well_formed, coherent=coherent,
                           situation=situation, strategy=strategy, tools=tools,
                           preference=preference, insight=insight, raw=raw)
    bare_fail = re.search(r'\bFAIL\b', raw, re.IGNORECASE)
    if bare_fail:
        return GradeResult(passed=False, well_formed=well_formed, coherent=coherent,
                           situation=situation, strategy=strategy, tools=tools,
                           preference=preference, insight=insight, raw=raw)
```

- [ ] **Step 5: Update grading tests**

Run: `python -m pytest tests/test_grading.py -v`

If existing tests fail due to GradeResult field changes, update them to include the new optional fields. The new fields default to `None`, so existing constructors should work. If any test asserts on exact GradeResult equality, add `well_formed=None, coherent=None`.

- [ ] **Step 6: Verify grading.py imports work**

Run: `python -c "from src.core.grading import grade_task, parse_grade_response, GradeResult; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/core/grading.py tests/test_grading.py
git commit -m "feat(grading): add WELL_FORMED/COHERENT fields and content_quality pre-check"
```

---

## Task 10: Integrate into router.py — Streaming Callback

**Files:**
- Modify: `src/core/router.py:43-86,988-993`
- Modify: `src/core/llm_dispatcher.py:116-147`

- [ ] **Step 1: Add on_chunk parameter to _stream_with_accumulator (lines 43-46)**

Old signature:
```python
async def _stream_with_accumulator(
    completion_kwargs: dict,
    partial_buf: object,
) -> "litellm.ModelResponse":
```

New signature:
```python
async def _stream_with_accumulator(
    completion_kwargs: dict,
    partial_buf: object,
    on_chunk: Callable[[str], bool] | None = None,
) -> "litellm.ModelResponse":
```

Add import at top of function or at file level:
```python
from typing import Callable
```

- [ ] **Step 2: Add callback invocation in streaming loop (line 63-64)**

Old loop body:
```python
          if delta:
              if delta.content:
                  accumulated += delta.content
                  partial_buf._partial_content = accumulated
```

New loop body:
```python
          if delta:
              if delta.content:
                  accumulated += delta.content
                  partial_buf._partial_content = accumulated
                  if on_chunk and on_chunk(accumulated):
                      logger.info(
                          "[Router] Streaming aborted by quality callback "
                          f"at {len(accumulated)} chars"
                      )
                      break
```

- [ ] **Step 3: Thread on_chunk through call_model (line 988-993)**

Old signature:
```python
async def call_model(
    reqs: ModelRequirements,
    messages: list[dict],
    tools: list[dict] | None = None,
    timeout_override: float | None = None,
    partial_buf: object | None = None,
) -> dict:
```

New signature:
```python
async def call_model(
    reqs: ModelRequirements,
    messages: list[dict],
    tools: list[dict] | None = None,
    timeout_override: float | None = None,
    partial_buf: object | None = None,
    on_chunk: Callable[[str], bool] | None = None,
) -> dict:
```

At the streaming invocation site (lines 1339-1342):

Old:
```python
                        response = await _stream_with_accumulator(
                            completion_kwargs, partial_buf,
                        )
```

New:
```python
                        response = await _stream_with_accumulator(
                            completion_kwargs, partial_buf, on_chunk=on_chunk,
                        )
```

- [ ] **Step 4: Thread on_chunk through LLMDispatcher.request (llm_dispatcher.py lines 116-147)**

Old signature:
```python
    async def request(
        self,
        category: CallCategory,
        reqs: "ModelRequirements",
        messages: list[dict],
        tools: list[dict] | None = None,
        partial_buf: object | None = None,
    ) -> dict:
```

New signature:
```python
    async def request(
        self,
        category: CallCategory,
        reqs: "ModelRequirements",
        messages: list[dict],
        tools: list[dict] | None = None,
        partial_buf: object | None = None,
        on_chunk: "Callable[[str], bool] | None" = None,
    ) -> dict:
```

Add `on_chunk` to `_route_main_work` signature (line 209) and thread to all 3 `call_model` sites:

```python
    async def _route_main_work(
        self,
        reqs: "ModelRequirements",
        messages: list[dict],
        tools: list[dict] | None,
        partial_buf: object | None = None,
        on_chunk: "Callable[[str], bool] | None" = None,
    ) -> dict:
```

Update the call at line 147 to forward `on_chunk`:
```python
            return await self._route_main_work(reqs, messages, tools,
                                                partial_buf=partial_buf, on_chunk=on_chunk)
```

Update all 3 `call_model` invocations inside `_route_main_work` (lines 242, 258, and any others with `partial_buf`):
```python
        result = await call_model(reqs_copy, messages, tools,
                                  timeout_override=timeout,
                                  partial_buf=partial_buf, on_chunk=on_chunk)
```
```python
        result = await call_model(reqs, messages, tools, timeout_override=timeout,
                                 partial_buf=partial_buf, on_chunk=on_chunk)
```

OVERHEAD calls (`_route_overhead`, line 320) do NOT get `on_chunk` — they're short, cheap calls that don't need streaming quality checks.

- [ ] **Step 5: Wire callback creation in base.py (line 1649-1659)**

In the agent's LLM call site where `partial_buf=self` is passed:

Old:
```python
                    response = await get_dispatcher().request(
                        CallCategory.MAIN_WORK,
                        reqs,
                        messages,
                        tools=litellm_tools,
                        partial_buf=self,
                    )
```

New:
```python
                    _on_chunk = None
                    if _task_ctx.get("is_workflow_step"):
                        from content_quality import make_stream_callback
                        _step_max = _task_ctx.get("artifact_schema", {}).get(
                            "max_output_chars", 20_000
                        )
                        _on_chunk = make_stream_callback(max_size=_step_max)

                    response = await get_dispatcher().request(
                        CallCategory.MAIN_WORK,
                        reqs,
                        messages,
                        tools=litellm_tools,
                        partial_buf=self,
                        on_chunk=_on_chunk,
                    )
```

Only workflow steps get the streaming callback — regular tasks don't need it (they have grading as backstop).

- [ ] **Step 6: Verify imports work**

Run: `python -c "from src.core.router import call_model; from src.core.llm_dispatcher import LLMDispatcher; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/core/router.py src/core/llm_dispatcher.py src/agents/base.py
git commit -m "feat(router): streaming quality callback threaded through dispatcher"
```

---

## Task 11: Integrate into base.py — Dependency, Checkpoint, Self-Reflect

**Files:**
- Modify: `src/agents/base.py:856-862,1443-1478,2828-2881`

- [ ] **Step 1: Add content quality check to dependency result injection (lines 856-862)**

Old code:
```python
            text = dep.get("result") or "(no result)"
            if len(text) > per_dep:
                text = text[:per_dep] + "\n... (truncated)"
            parts.append(
                f"### Step #{dep_id}: {dep.get('title', 'Unknown')}\n{text}"
            )
```

New code:
```python
            text = dep.get("result") or "(no result)"
            from content_quality import assess as cq_assess, salvage as cq_salvage
            _dep_cq = cq_assess(text)
            if _dep_cq.is_degenerate:
                cleaned = cq_salvage(text)
                text = cleaned if cleaned else "(dependency output was degenerate — skipped)"
            if len(text) > per_dep:
                text = text[:per_dep] + "\n... (truncated)"
            parts.append(
                f"### Step #{dep_id}: {dep.get('title', 'Unknown')}\n{text}"
            )
```

- [ ] **Step 2: Add content quality check to checkpoint recovery (after line 1478)**

After the existing checkpoint restoration block (line 1478), add validation of recovered messages. The key concern is `_prev_output` stored in task context — the orchestrator already sets this before checkpoint. Add after line 1478:

```python
            # Validate recovered _prev_output from checkpoint context
            _recovered_prev = _task_ctx.get("_prev_output")
            if _recovered_prev:
                from content_quality import assess as cq_assess, salvage as cq_salvage
                _rec_cq = cq_assess(_recovered_prev)
                if _rec_cq.is_degenerate:
                    cleaned = cq_salvage(_recovered_prev)
                    if cleaned:
                        _task_ctx["_prev_output"] = cleaned
                    else:
                        _task_ctx.pop("_prev_output", None)
                    logger.info(
                        f"[Task #{task_id}] Checkpoint _prev_output was degenerate, "
                        f"{'salvaged' if cleaned else 'discarded'}"
                    )
```

- [ ] **Step 3: Add content quality check to self-reflection corrected_result (lines 2876-2878)**

Old code:
```python
            parsed = self._try_parse_json(raw)
            if parsed and parsed.get("verdict") == "fix":
                return parsed
```

New code:
```python
            parsed = self._try_parse_json(raw)
            if parsed and parsed.get("verdict") == "fix":
                corrected = parsed.get("corrected_result")
                if corrected:
                    from content_quality import assess as cq_assess
                    _reflect_cq = cq_assess(corrected)
                    if _reflect_cq.is_degenerate:
                        logger.warning(
                            f"Self-reflection produced degenerate corrected_result "
                            f"({_reflect_cq.summary}), keeping original"
                        )
                        return None
                return parsed
```

- [ ] **Step 4: Verify base.py imports work**

Run: `python -c "from src.agents.base import BaseAgent; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/agents/base.py
git commit -m "feat(base): content_quality checks for deps, checkpoint, self-reflect"
```

---

## Task 12: Integrate into episodic.py — Memory Storage Gate

**Files:**
- Modify: `src/memory/episodic.py:56-63`

- [ ] **Step 1: Add content quality gate before embedding**

In `store_task_result()`, after building the `result_summary` (line 56) and before building the `text` variable:

Old code (lines 55-63):
```python
    # Text to embed — combines title + description + result summary
    result_summary = (result or "")[:500]
    text = (
        f"Task: {title}\n"
        f"Description: {description[:300]}\n"
        f"Agent: {agent_type}\n"
        f"Outcome: {'success' if success else 'failure'}\n"
        f"Result: {result_summary}"
    )
```

New code:
```python
    # Text to embed — combines title + description + result summary
    result_summary = (result or "")[:500]

    # Don't memorize degenerate output — it poisons future RAG retrieval
    if result_summary and success:
        from content_quality import assess as cq_assess
        _ep_cq = cq_assess(result_summary)
        if _ep_cq.is_degenerate:
            logger.info(
                f"[Episodic] Skipping storage for task #{task_id}: "
                f"degenerate result ({_ep_cq.summary})"
            )
            return None

    text = (
        f"Task: {title}\n"
        f"Description: {description[:300]}\n"
        f"Agent: {agent_type}\n"
        f"Outcome: {'success' if success else 'failure'}\n"
        f"Result: {result_summary}"
    )
```

Note: only gate on `success=True` — failed task results should still be stored for error pattern analysis.

- [ ] **Step 2: Add logger import if not present**

Check if `episodic.py` already imports a logger. If not, add at top:
```python
from ..infra.logging_config import get_logger
logger = get_logger("memory.episodic")
```

- [ ] **Step 3: Verify episodic.py imports work**

Run: `python -c "from src.memory.episodic import store_task_result; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/memory/episodic.py
git commit -m "feat(episodic): gate memory storage on content quality assessment"
```

---

## Task 13: Final Verification

**Files:** None (testing only)

- [ ] **Step 1: Run full package test suite**

Run: `cd packages/content_quality && python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_content_quality_integration.py -v`
Expected: All PASS.

- [ ] **Step 3: Run existing grading tests**

Run: `python -m pytest tests/test_grading.py -v`
Expected: All PASS (new fields default to None, no breakage).

- [ ] **Step 4: Run existing workflow/hook tests if they exist**

Run: `python -m pytest tests/ -k "hook or workflow or skill" -v`
Expected: All PASS.

- [ ] **Step 5: Smoke test key imports**

Run:
```bash
python -c "
from content_quality import assess, salvage, make_stream_callback
from src.workflows.engine.hooks import post_execute_workflow_step, enrich_task_description
from src.core.grading import grade_task, parse_grade_response, GradeResult
from src.core.router import call_model
from src.core.llm_dispatcher import LLMDispatcher
from src.agents.base import BaseAgent
from src.memory.episodic import store_task_result
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 6: Commit final state**

```bash
git add -A
git commit -m "test: verify content_quality integration across all 13 points"
```
