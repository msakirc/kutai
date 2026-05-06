"""Roundtrip smoke test for workflow_exemplars."""
import asyncio
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def main() -> None:
    from src.memory.workflow_exemplars import (
        capture_exemplar, lookup_exemplars, format_exemplars_for_prompt,
    )

    # Capture three exemplars under same key with varying quality
    for q, tid in [(0.7, 9991), (0.95, 9992), (0.5, 9993), (0.85, 9994)]:
        await capture_exemplar(
            workflow="i2p",
            step_id="3.10a",
            agent_type="writer",
            result=f"# Requirements (quality {q})\n\nFunctional spec produced by task#{tid}.\n"
                   f"1. Auth\n2. CRUD\n3. Reports\n",
            quality_score=q,
            task_id=tid,
            mission_id=42,
        )

    # Lookup — should return top 3 by quality
    rows = await lookup_exemplars(workflow="i2p", step_id="3.10a", agent_type="writer")
    print(f"got {len(rows)} exemplars (expect 3)")
    for r in rows:
        print(f"  q={r['quality_score']} task#{r['task_id']} result_chars={len(r['result'])}")

    print()
    print("=== format ===")
    block = format_exemplars_for_prompt(rows, step_id="3.10a", max_chars=600)
    print(block)


asyncio.run(main())
