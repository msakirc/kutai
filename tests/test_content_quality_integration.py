"""Integration tests for content_quality wiring in KutAI."""

from content_quality import assess, salvage


class TestHooksIntegration:
    def test_final_quality_gate_rejects_degenerate(self):
        sections = []
        for _ in range(3):
            sections.append("## Component Usage\nContent about usage")
            sections.append("## Component Usage Summary\nMore content")
        sections.append("## API Ref\nUnique")
        output_value = "Intro\n" + "\n".join(sections)
        cq = assess(output_value, max_size=20_000)
        assert cq.is_degenerate is True
        assert "header_repetition" in cq.reasons

    def test_final_quality_gate_passes_clean(self):
        output_value = "## Overview\nGood content.\n\n## Details\nMore good content."
        cq = assess(output_value, max_size=20_000)
        assert cq.is_degenerate is False

    def test_workspace_salvage_keeps_good_content(self):
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
        assert cleaned
        assert "## Component Usage\n" in cleaned
        assert "## API Reference\n" in cleaned
        assert "## Component Usage Summary" not in cleaned

    def test_workspace_salvage_returns_empty_for_pure_garbage(self):
        text = "## A\n\n## B\n\n## C\n\n## D\n\n## E\n"
        cleaned = salvage(text)
        assert cleaned == ""

    def test_oversized_output_rejected(self):
        text = "word " * 5000
        cq = assess(text, max_size=20_000)
        assert cq.is_degenerate is True
        assert "size_exceeded" in cq.reasons

    def test_max_output_chars_from_schema(self):
        text = "x" * 35_000
        cq_default = assess(text, max_size=20_000)
        assert cq_default.is_degenerate is True
        cq_custom = assess(text, max_size=40_000)
        assert cq_custom.is_degenerate is False
