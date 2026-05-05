"""One-shot: count auto skill strategies whose tools_used are real registered tools."""
from __future__ import annotations
import json
import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.tools import TOOL_REGISTRY

real_tools = set(TOOL_REGISTRY.keys())
print(f"real tools in registry: {len(real_tools)}")
print()

c = sqlite3.connect(r"C:\Users\sakir\ai\kutai\kutai.db")
c.row_factory = sqlite3.Row

junk_count = 0
mixed = 0
clean = 0
junk_examples = []

for r in c.execute("SELECT name, strategies FROM skills WHERE skill_type='auto'"):
    try:
        strats = json.loads(r["strategies"])
    except Exception:
        continue
    all_tools: list[str] = []
    for s in strats:
        tu = s.get("tools_used") or []
        if isinstance(tu, list):
            all_tools.extend(tu)
    if not all_tools:
        continue
    real_hits = sum(1 for t in all_tools if t in real_tools)
    if real_hits == 0:
        junk_count += 1
        if len(junk_examples) < 8:
            junk_examples.append((r["name"], all_tools[:5]))
    elif real_hits == len(all_tools):
        clean += 1
    else:
        mixed += 1

print("examples (all-junk tools_used):")
for n, tools in junk_examples:
    print(f"  {n}: {tools}")
print()
print(f"summary: clean={clean} mixed={mixed} all-junk={junk_count}")
