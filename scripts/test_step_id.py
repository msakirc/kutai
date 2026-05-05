"""Test step-id fast path."""
import asyncio
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def main() -> None:
    from src.memory.skills import _step_id_fast_path

    cases = [
        "[7.6] test_infrastructure",
        "[5.7] component_specs",
        "[2.7] mvp_scope_definition",
        "[1.10] technology_trend_research",
        "[3.1] analyze_value_and_risks",
        "no step id here",
        "[99.99] non_existent",
    ]
    for c in cases:
        r = await _step_id_fast_path(c)
        if not r:
            print(f"{c!r}: (vector fallback)")
        else:
            print(f"{c!r}:")
            for s in r:
                print(f"  -> {s['name']} ({s['injection_count']}/{s['injection_success']})")


asyncio.run(main())
