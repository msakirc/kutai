"""Tests for dogru_mu_samet.salvager."""

from dogru_mu_samet.salvager import salvage


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
        assert "## Section A" not in result
        assert "## Section B\n" in result

    def test_returns_empty_when_nothing_salvageable(self):
        text = "## A\n\n## B\n  \n## C\n\n## D\n\n## E\n"
        result = salvage(text)
        assert result == ""
