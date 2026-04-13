# tests/test_exposition.py
from nerd_herd.exposition import build_metrics_text
from nerd_herd.registry import CollectorRegistry


def test_build_metrics_text_empty():
    reg = CollectorRegistry()
    text = build_metrics_text(reg)
    assert isinstance(text, str)


def test_build_metrics_text_with_collector():
    from prometheus_client import Gauge

    class StubCollector:
        name = "stub"
        def collect(self):
            return {"val": 42}
        def prometheus_metrics(self):
            return [_g]

    # Use a unique gauge name to avoid conflicts with other tests
    _g = Gauge("stub_exposition_test_val", "test value for exposition")
    _g.set(42)

    reg = CollectorRegistry()
    reg.register("stub", StubCollector())
    text = build_metrics_text(reg)
    assert isinstance(text, str)
    assert "stub_exposition_test_val" in text
