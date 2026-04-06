"""
CLI tool for benchmark operations and registry diagnostics.

Usage:
    python benchmark_cli.py scan              # Rescan model directory
    python benchmark_cli.py benchmarks        # Fetch/refresh benchmarks
    python benchmark_cli.py enrich            # Enrich registry with benchmarks
    python benchmark_cli.py show              # Show full registry
    python benchmark_cli.py score <task>      # Show model ranking for a task
    python benchmark_cli.py model <name>      # Show single model details
    python benchmark_cli.py compare <m1> <m2> # Side-by-side comparison
"""

import logging
import sys

from src.infra.logging_config import get_logger
from src.models.benchmark.benchmark_fetcher import \
    enrich_registry_with_benchmarks, BenchmarkFetcher
from ..model_registry import reload_registry, get_registry
from ..capabilities import TASK_PROFILES, Cap

logger = get_logger("models.benchmark.cli")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


def cmd_scan():
    logger.info("alo")
    """Rescan model directory and show results."""

    result = reload_registry()
    print(f"\n✅ Scan complete:")
    print(f"   Added:   {result['added'] or '(none)'}")
    print(f"   Removed: {result['removed'] or '(none)'}")
    print(f"   Total:   {result['total']} models")


def cmd_benchmarks():
    """Fetch/refresh benchmark data from all sources."""
    fetcher = BenchmarkFetcher()
    fetcher.refresh_cache()

    bulk = fetcher.fetch_all_bulk()
    print(f"\n✅ Fetched benchmark data for {len(bulk)} models")

    # Show coverage per source
    for f in fetcher.fetchers:
        cached = fetcher.cache.get_all_models(f.source_name)
        count = len(cached.get("models", {})) if cached else 0
        print(f"   {f.source_name:25s}: {count:>4} models")


def cmd_enrich(force_refresh: bool = False):
    """Enrich registry with benchmark data."""
    if force_refresh:
        print("🔄 Force-refreshing benchmark cache...")
        fetcher = BenchmarkFetcher()
        fetcher.refresh_cache()

    registry = get_registry()
    enriched = enrich_registry_with_benchmarks(registry)

    print(f"\n✅ Enriched {len(enriched)} models with benchmark data")
    for name, caps in enriched.items():
        model = registry.get(name)
        variant_tag = ""
        if model and getattr(model, 'is_variant', False):
            variant_tag = f" [{','.join(sorted(model.variant_flags))}]"
        top = sorted(caps.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}={v:.1f}" for k, v in top)
        print(f"   {name:45s}{variant_tag:20s}: {top_str}")

    fallback = [n for n in registry.models if n not in enriched]
    if fallback:
        print(f"\n⚠️  {len(fallback)} models fell back to family profiles:")
        for name in sorted(fallback):
            model = registry.get(name)
            print(f"   {name:45s} (family={model.family if model else '?'})")


def cmd_show():
    """Show full registry summary."""
    registry = get_registry()
    registry.print_summary()


def cmd_score(task_name: str):
    """Show model ranking for a task."""

    if task_name not in TASK_PROFILES:
        print(f"Unknown task: {task_name}")
        print(f"Available: {', '.join(sorted(TASK_PROFILES.keys()))}")
        return

    registry = get_registry()
    ranked = registry.best_for_task(task_name, top_k=20)

    print(f"\n🎯 Best models for task: {task_name}")
    print(f"{'─' * 60}")

    profile = TASK_PROFILES[task_name]
    # Show which capabilities matter most
    top_weights = sorted(
        [(k.value if hasattr(k, 'value') else k, v) for k, v in profile.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    weight_str = ", ".join(f"{k}({v:.1f})" for k, v in top_weights)
    print(f"Key capabilities: {weight_str}\n")

    for i, (name, score) in enumerate(ranked, 1):
        model = registry.get(name)
        loc = {"local": "💻", "ollama": "🦙", "cloud": "☁️"}.get(model.location, "?")
        flags = ""
        if model.has_vision:
            flags += "👁️"
        if model.thinking_model:
            flags += "🧠"
        print(f"  {i:2d}. {loc} {name:40s} score={score:5.2f} {flags}")


def cmd_model(name: str):
    """Show detailed info for a single model."""

    registry = get_registry()
    model = registry.get(name)

    if not model:
        # Try fuzzy match
        candidates = [m for m in registry.models if name.lower() in m.lower()]
        if candidates:
            print(f"Model '{name}' not found. Did you mean:")
            for c in candidates:
                print(f"  - {c}")
        else:
            print(f"Model '{name}' not found.")
        return

    print(f"\n📦 {model.name}")
    print(f"{'═' * 60}")
    print(f"  Location:    {model.location} ({model.provider})")
    print(f"  LiteLLM:     {model.litellm_name}")
    print(f"  Family:      {model.family}")
    if model.path:
        print(f"  Path:        {model.path}")
    print(f"  Params:      {model.total_params_b:.1f}B", end="")
    if model.active_params_b:
        print(f" (active: {model.active_params_b:.1f}B, {model.model_type})", end="")
    print()
    if model.quantization:
        print(f"  Quant:       {model.quantization}")
    print(f"  Context:     {model.context_length:,}")
    print(f"  Max Output:  {model.max_tokens:,}")
    print(f"  GPU Layers:  {model.gpu_layers}/{model.total_layers}")

    flags = []
    if model.supports_function_calling:
        flags.append("function_calling")
    if model.supports_json_mode:
        flags.append("json_mode")
    if model.thinking_model:
        flags.append("thinking")
    if model.has_vision:
        flags.append("vision")
    print(f"  Features:    {', '.join(flags) or '(none)'}")
    print(f"  Specialty:   {model.specialty or '(general)'}")
    print(f"  Priority:    {model.priority_class}")
    if model.cost_per_1k_output > 0:
        print(f"  Cost:        ${model.cost_per_1k_input:.4f}/1k in, ${model.cost_per_1k_output:.4f}/1k out")
    if model.rate_limit_rpm < 999:
        print(f"  Rate Limit:  {model.rate_limit_rpm} RPM")
    print(f"  Loaded:      {'✅' if model.is_loaded else '❌'}")
    if model.tokens_per_second > 0:
        print(f"  Speed:       {model.tokens_per_second:.1f} tok/s")

    print(f"\n  📊 Capabilities (15 dimensions):")
    print(f"  {'─' * 50}")
    sorted_caps = sorted(model.capabilities.items(), key=lambda x: x[1], reverse=True)
    for cap_name, score in sorted_caps:
        bar_len = int(score * 3)  # 30 chars max for score=10
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {cap_name:25s} {score:4.1f} {bar}")

    # Show task fitness
    print(f"\n  🎯 Task Fitness:")
    print(f"  {'─' * 50}")

    registry = get_registry()
    task_scores = []
    for task_name in sorted(TASK_PROFILES.keys()):
        ranked = registry.best_for_task(task_name, top_k=len(registry.models))
        for model_name, score in ranked:
            if model_name == model.name:
                task_scores.append((task_name, score))
                break

    task_scores.sort(key=lambda x: x[1], reverse=True)
    for task_name, score in task_scores:
        bar_len = int(score * 3)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {task_name:25s} {score:4.1f} {bar}")


def cmd_compare(name1: str, name2: str):
    """Side-by-side comparison of two models."""

    registry = get_registry()
    m1 = registry.get(name1)
    m2 = registry.get(name2)

    if not m1:
        candidates = [m for m in registry.models if name1.lower() in m.lower()]
        print(f"Model '{name1}' not found.", end="")
        if candidates:
            print(f" Did you mean: {', '.join(candidates[:3])}")
        else:
            print()
        return
    if not m2:
        candidates = [m for m in registry.models if name2.lower() in m.lower()]
        print(f"Model '{name2}' not found.", end="")
        if candidates:
            print(f" Did you mean: {', '.join(candidates[:3])}")
        else:
            print()
        return

    col_w = 25
    print(f"\n{'':30s} {'Model A':>{col_w}s}  {'Model B':>{col_w}s}")
    print(f"{'':30s} {m1.name:>{col_w}s}  {m2.name:>{col_w}s}")
    print(f"{'═' * (30 + col_w * 2 + 4)}")

    # Meta comparison
    def row(label, v1, v2):
        print(f"  {label:28s} {str(v1):>{col_w}s}  {str(v2):>{col_w}s}")

    row("Location", m1.location, m2.location)
    row("Provider", m1.provider, m2.provider)
    row("Family", m1.family, m2.family)
    row("Params", f"{m1.total_params_b:.1f}B", f"{m2.total_params_b:.1f}B")
    if m1.active_params_b or m2.active_params_b:
        row("Active Params",
            f"{m1.active_params_b:.1f}B" if m1.active_params_b else "—",
            f"{m2.active_params_b:.1f}B" if m2.active_params_b else "—")
    row("Quantization", m1.quantization or "—", m2.quantization or "—")
    row("Context", f"{m1.context_length:,}", f"{m2.context_length:,}")
    row("Vision", "✅" if m1.has_vision else "❌", "✅" if m2.has_vision else "❌")
    row("Thinking", "✅" if m1.thinking_model else "❌", "✅" if m2.thinking_model else "❌")
    row("Tool Calling", "✅" if m1.supports_function_calling else "❌",
        "✅" if m2.supports_function_calling else "❌")

    # Capability comparison
    print(f"\n  📊 Capabilities:")
    print(f"  {'─' * (28 + col_w * 2 + 4)}")

    all_caps = sorted(
        set(list(m1.capabilities.keys()) + list(m2.capabilities.keys()))
    )
    for cap in all_caps:
        s1 = m1.capabilities.get(cap, 0.0)
        s2 = m2.capabilities.get(cap, 0.0)
        diff = s2 - s1
        if abs(diff) < 0.1:
            indicator = "  "
        elif diff > 0:
            indicator = " ▶" if diff > 1.0 else " ›"
        else:
            indicator = "◀ " if diff < -1.0 else "‹ "
        print(f"  {cap:28s} {s1:>{col_w-1}.1f}  {s2:>{col_w-1}.1f} {indicator}")

    # Task fitness comparison
    print(f"\n  🎯 Task Fitness:")
    print(f"  {'─' * (28 + col_w * 2 + 4)}")

    for task_name in sorted(TASK_PROFILES.keys()):
        ranked = registry.best_for_task(task_name, top_k=len(registry.models))
        score_map = {n: s for n, s in ranked}
        s1 = score_map.get(m1.name, 0.0)
        s2 = score_map.get(m2.name, 0.0)
        diff = s2 - s1
        if abs(diff) < 0.1:
            indicator = "  "
        elif diff > 0:
            indicator = " ▶" if diff > 0.5 else " ›"
        else:
            indicator = "◀ " if diff < -0.5 else "‹ "
        winner = ""
        if abs(diff) >= 0.3:
            winner = " ★" if diff > 0 else "★ "
        print(f"  {task_name:28s} {s1:>{col_w-1}.1f}  {s2:>{col_w-1}.1f} {indicator}{winner}")


def cmd_variants():
    """Show all model variants and their mode flags."""
    registry = get_registry()

    print(f"\n🔀 Model Variants ({len(registry.models)} total entries)")
    print(f"{'═' * 70}")

    bases = {}
    for name, model in registry.models.items():
        base = getattr(model, 'base_model_name', '') or name
        if base not in bases:
            bases[base] = []
        bases[base].append(model)

    for base_name, variants in sorted(bases.items()):
        for v in sorted(variants, key=lambda m: m.name):
            flags = getattr(v, 'variant_flags', set())
            mode = "base" if not getattr(v, 'is_variant', False) else ",".join(sorted(flags))
            thinking = "🧠" if v.thinking_model else "  "
            vision = "👁️" if v.has_vision else "  "
            print(f"  {thinking} {vision} {v.name:45s} [{mode:20s}] best={v.best_score():.1f}")
        print()


def cmd_tasks():
    """List all available tasks and their key capabilities."""

    print(f"\n📋 Available Task Profiles ({len(TASK_PROFILES)})")
    print(f"{'═' * 70}")

    for task_name in sorted(TASK_PROFILES.keys()):
        profile = TASK_PROFILES[task_name]
        # Get top 4 weighted capabilities
        top = sorted(
            [(k.value if isinstance(k, Cap) else k, v) for k, v in profile.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:4]
        top_str = ", ".join(f"{k}({v:.1f})" for k, v in top)
        print(f"  {task_name:20s} → {top_str}")


def cmd_export(path: str = "registry_export.json"):
    """Export full registry to JSON for debugging / external tools."""
    import json

    registry = get_registry()
    export = {}

    for name, m in registry.models.items():
        export[name] = {
            "location": m.location,
            "provider": m.provider,
            "litellm_name": m.litellm_name,
            "family": m.family,
            "total_params_b": m.total_params_b,
            "active_params_b": m.active_params_b,
            "quantization": m.quantization,
            "context_length": m.context_length,
            "max_tokens": m.max_tokens,
            "has_vision": m.has_vision,
            "thinking_model": m.thinking_model,
            "supports_function_calling": m.supports_function_calling,
            "specialty": m.specialty,
            "capabilities": m.capabilities,
            "best_score": m.best_score(),
        }

    with open(path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"✅ Exported {len(export)} models to {path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "scan":
        cmd_scan()
    elif cmd == "benchmarks":
        cmd_benchmarks()
    elif cmd == "enrich":
        force = "--force-refresh" in sys.argv or "-f" in sys.argv
        cmd_enrich(force_refresh=force)
    elif cmd == "show":
        cmd_show()
    elif cmd == "score":
        if len(sys.argv) < 3:
            print("Usage: benchmark_cli.py score <task_name>")
            cmd_tasks()
            return
        cmd_score(sys.argv[2])
    elif cmd == "model":
        if len(sys.argv) < 3:
            print("Usage: benchmark_cli.py model <model_name>")
            return
        cmd_model(sys.argv[2])
    elif cmd == "compare":
        if len(sys.argv) < 4:
            print("Usage: benchmark_cli.py compare <model1> <model2>")
            return
        cmd_compare(sys.argv[2], sys.argv[3])
    elif cmd == "variants":
        cmd_variants()
    elif cmd == "tasks":
        cmd_tasks()
    elif cmd == "export":
        path = sys.argv[2] if len(sys.argv) > 2 else "registry_export.json"
        cmd_export(path)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # loads .env into environment
    main()
