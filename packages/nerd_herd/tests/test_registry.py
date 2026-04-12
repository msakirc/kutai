import pytest
from nerd_herd.registry import CollectorRegistry, Collector


class DummyCollector:
    name = "dummy"

    def collect(self) -> dict:
        return {"dummy_value": 42}

    def prometheus_metrics(self) -> list:
        return []


def test_register_and_get():
    reg = CollectorRegistry()
    c = DummyCollector()
    reg.register("dummy", c)
    assert reg.get("dummy") is c


def test_get_unknown_raises():
    reg = CollectorRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_unregister():
    reg = CollectorRegistry()
    reg.register("dummy", DummyCollector())
    reg.unregister("dummy")
    with pytest.raises(KeyError):
        reg.get("dummy")


def test_collect_all():
    reg = CollectorRegistry()
    reg.register("a", DummyCollector())
    result = reg.collect_all()
    assert "a" in result
    assert result["a"]["dummy_value"] == 42


def test_names():
    reg = CollectorRegistry()
    reg.register("a", DummyCollector())
    reg.register("b", DummyCollector())
    assert set(reg.names()) == {"a", "b"}
