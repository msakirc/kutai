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
        text = " ".join(["the"] * 10_000)
        result = assess(text, max_size=20_000)
        assert result.is_degenerate is True
        assert "size_exceeded" in result.reasons
        assert "low_entropy" in result.reasons
