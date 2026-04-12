"""Tests for dogru_mu_samet.streaming."""

from dogru_mu_samet.streaming import make_stream_callback


class TestMakeStreamCallback:
    def test_clean_text_no_abort(self):
        cb = make_stream_callback(max_size=20_000, check_interval=100)
        sentences = [
            "The quick brown fox jumps over the lazy dog. ",
            "Python is a versatile programming language used widely. ",
            "Machine learning models require careful tuning and evaluation. ",
            "Data structures are fundamental to computer science education. ",
            "Asynchronous programming improves application responsiveness significantly. ",
            "Testing ensures code reliability and catches regressions early. ",
            "Documentation helps future developers understand complex codebases. ",
            "Version control tracks changes and enables team collaboration. ",
            "APIs provide clean interfaces between different software components. ",
            "Databases store and retrieve structured information efficiently. ",
        ]
        accumulated = ""
        for sentence in sentences * 3:
            accumulated += sentence
            assert cb(accumulated) is False

    def test_size_abort_immediate(self):
        cb = make_stream_callback(max_size=100, check_interval=50)
        text = "x" * 101
        assert cb(text) is True

    def test_repetition_abort_at_interval(self):
        cb = make_stream_callback(max_size=50_000, check_interval=200)
        sections = []
        for _ in range(4):
            sections.append("## Component Usage\nSome content about usage patterns.")
            sections.append("## Component Usage Summary\nMore content.")
        sections.append("## API Reference\nUnique content here.")
        degenerate = "Intro\n" + "\n".join(sections)
        assert cb(degenerate) is True

    def test_no_check_before_interval(self):
        cb = make_stream_callback(max_size=50_000, check_interval=1000)
        short_degenerate = " ".join(["the"] * 50)
        assert cb(short_degenerate) is False

    def test_stateful_tracks_last_check(self):
        cb = make_stream_callback(max_size=50_000, check_interval=100)
        text = "Hello world. " * 12  # ~156 chars
        cb(text)
        text += "tiny"
        result = cb(text)
        assert result is False
