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
