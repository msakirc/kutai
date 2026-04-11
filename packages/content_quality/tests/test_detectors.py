"""Tests for content_quality.detectors."""

from content_quality.detectors import (
    check_size,
    check_header_repetition,
    check_paragraph_repetition,
    check_token_entropy,
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


class TestCheckHeaderRepetition:
    def test_no_headers(self):
        text = "Just plain text without any markdown headers."
        ratio, breached, reason = check_header_repetition(text)
        assert ratio == 0.0
        assert breached is False

    def test_few_sections_skipped(self):
        text = "intro\n## A\ncontent\n## A\ncontent\n## B\ncontent"
        ratio, breached, reason = check_header_repetition(text)
        assert ratio == 0.0
        assert breached is False

    def test_unique_sections(self):
        sections = [f"## Section {i}\nContent for section {i}" for i in range(6)]
        text = "Intro\n" + "\n".join(sections)
        ratio, breached, reason = check_header_repetition(text)
        assert ratio == 0.0
        assert breached is False

    def test_degenerate_repetition(self):
        sections = []
        for name in ["Component Usage", "Component Usage Summary", "API Reference"]:
            for _ in range(2):
                sections.append(f"## {name}\nSome content here")
        text = "Intro\n" + "\n".join(sections)
        ratio, breached, reason = check_header_repetition(text)
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
        assert ratio > 0.4
        assert breached is True


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
        assert breached is False


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
        words = ["hello", "world"] * 20 + ["unique", "words", "here", "today", "test"]
        text = " ".join(words)
        entropy, breached, reason = check_token_entropy(text)
        assert entropy > 1.5
        assert isinstance(breached, bool)

    def test_empty_text(self):
        entropy, breached, reason = check_token_entropy("")
        assert entropy == 0.0
        assert breached is False

    def test_single_token(self):
        entropy, breached, reason = check_token_entropy("hello")
        assert entropy == 0.0
        assert breached is False
