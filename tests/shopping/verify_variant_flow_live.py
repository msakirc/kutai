"""Live smoke for variant disambiguation — NOT a pytest.

Run manually:
    timeout 240 .venv/Scripts/python.exe -m tests.shopping.verify_variant_flow_live
"""
from __future__ import annotations
import asyncio
import json
import sys
from dotenv import load_dotenv
load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


async def _bootstrap_fatih_hoca() -> None:
    """Match src/app/run.py init so the dispatcher can select models."""
    import os
    from pathlib import Path
    import fatih_hoca
    from src.app.config import AVAILABLE_KEYS

    catalog = str(Path("src/models/models.yaml").resolve())
    models_dir = os.getenv("MODEL_DIR") or None
    providers = {p for p, ok in AVAILABLE_KEYS.items() if ok}
    fatih_hoca.init(catalog_path=catalog, models_dir=models_dir,
                    available_providers=providers)


async def main():
    await _bootstrap_fatih_hoca()
    from src.workflows.shopping.pipeline_v2 import (
        _handler_resolve_candidates, _handler_group_label_filter_gate,
    )
    query = "samsung s25"
    print(f"Query: {query!r}", flush=True)
    r1 = await _handler_resolve_candidates(
        task={"id": 1}, artifacts={"user_query": query}, ctx={"per_site_n": 3},
    )
    print(f"candidates={len(r1['candidates'])}", flush=True)
    r2 = await _handler_group_label_filter_gate(
        task={"id": 2}, artifacts={"search_results": json.dumps(r1)}, ctx={},
    )
    print(f"gate={r2['gate']['kind']}", flush=True)
    if r2["gate"]["kind"] == "clarify":
        for opt in r2.get("clarify_options", []):
            print(f"  option: {opt['label']} (prominence {opt['prominence']:.2f})", flush=True)
    elif r2["gate"]["kind"] == "chosen":
        print(f"  chosen: {r2['chosen_group']['representative_title']}", flush=True)
    else:
        print(f"  reason: {r2['gate'].get('reason')}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
