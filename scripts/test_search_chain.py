"""Quick test of the web_search fallback chain."""
import asyncio, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PERPLEXICA_URL"] = "http://localhost:3000"
os.environ["SEARXNG_URL"] = "http://localhost:3002"

async def main():
    from importlib import import_module
    ws = import_module("src.tools.web_search")
    ws._perplexica_models = None
    ws._perplexica_fail_count = 5  # skip perplexica (no llama for it)
    ws._perplexica_disabled_at = 0.0

    query = "Turkey vs Kosovo predicted lineup March 31 2026"
    print(f"Query: {query}")
    t0 = time.time()
    result = await ws.web_search(query)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s | Length: {len(result) if result else 0}")
    print(f"Result:\n{(result or 'NONE')[:600]}")

asyncio.run(main())
