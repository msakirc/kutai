"""Guard for the 2026-07-27 false-hung fix.

ChromaDB lazily imports its heavyweight OpenTelemetry OTLP-gRPC exporter on the
first real query; cold (post-restart, .pyc evicted) that import runs ~260s of
disk-read + compile that GIL-starves the event-loop heartbeat, so Yaşar Usta
false-kills at the 300s startup-grace boundary. run.main() pre-warms it at boot
before the orchestrator heartbeat exists. If a ChromaDB upgrade moves/renames
the module, the pre-warm would silently become a no-op and the bug would return
— this test fails loud in that case.
"""
import sys


def test_chromadb_otel_module_path_is_pinned():
    from src.app.run import _CHROMADB_OTEL_MODULE
    assert _CHROMADB_OTEL_MODULE == "chromadb.telemetry.opentelemetry"


async def test_prewarm_actually_loads_the_heavy_module():
    from src.app.run import _prewarm_chromadb_otel, _CHROMADB_OTEL_MODULE

    ok = await _prewarm_chromadb_otel()
    assert ok is True, "pre-warm must import the module (else it's a silent no-op)"
    assert _CHROMADB_OTEL_MODULE in sys.modules
    # The heavy leaf that actually cost the 260s cold-import must be resident too.
    assert "opentelemetry.exporter.otlp.proto.grpc.trace_exporter" in sys.modules
