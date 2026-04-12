"""Collector registry — register metric sources, collect from all."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from prometheus_client.metrics import MetricWrapperBase


@runtime_checkable
class Collector(Protocol):
    name: str

    def collect(self) -> dict[str, float | int | str]:
        ...

    def prometheus_metrics(self) -> list[MetricWrapperBase]:
        ...


class CollectorRegistry:
    def __init__(self) -> None:
        self._collectors: dict[str, Collector] = {}

    def register(self, name: str, collector: Collector) -> None:
        self._collectors[name] = collector

    def unregister(self, name: str) -> None:
        del self._collectors[name]

    def get(self, name: str) -> Collector:
        return self._collectors[name]

    def names(self) -> list[str]:
        return list(self._collectors.keys())

    def collect_all(self) -> dict[str, dict[str, Any]]:
        result = {}
        for name, collector in self._collectors.items():
            try:
                result[name] = collector.collect()
            except Exception:
                result[name] = {}
        return result

    def all_prometheus_metrics(self) -> list[MetricWrapperBase]:
        metrics = []
        for collector in self._collectors.values():
            try:
                metrics.extend(collector.prometheus_metrics())
            except Exception:
                pass
        return metrics
