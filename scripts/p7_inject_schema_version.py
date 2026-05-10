"""Add _schema_version: "1" to every artifact schema in phase 0-6 of i2p_v3.json.

Preserves the source file's existing whitespace by editing inline rather than
re-serializing the JSON. Idempotent: artifacts already carrying
`_schema_version` are skipped.

Run with:
    python scripts/p7_inject_schema_version.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / "src" / "workflows" / "i2p" / "i2p_v3.json"
PHASES = ("phase_0", "phase_1", "phase_2", "phase_3", "phase_4", "phase_5", "phase_6")


def find_step_block_range(text: str, step_id: str):
    pat = re.compile(r'"id"\s*:\s*"' + re.escape(step_id) + r'"\s*,')
    m = pat.search(text)
    if not m:
        return None
    pos = m.start()
    depth = 0
    i = pos
    start = None
    while i > 0:
        i -= 1
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            if depth == 0:
                start = i
                break
            depth -= 1
    if start is None:
        return None
    depth = 0
    j = start
    while j < len(text):
        c = text[j]
        if c == '"':
            j += 1
            while j < len(text) and text[j] != '"':
                if text[j] == "\\":
                    j += 2
                    continue
                j += 1
            j += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (start, j + 1)
        j += 1
    return None


def patch_step_block(block_text: str):
    m = re.search(r'"artifact_schema"\s*:\s*\{', block_text)
    if not m:
        return block_text, 0
    schema_open = m.end() - 1
    j = schema_open
    depth = 0
    schema_close = None
    while j < len(block_text):
        c = block_text[j]
        if c == '"':
            j += 1
            while j < len(block_text) and block_text[j] != '"':
                if block_text[j] == "\\":
                    j += 2
                    continue
                j += 1
            j += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                schema_close = j
                break
        j += 1
    if schema_close is None:
        return block_text, 0

    schema_body = block_text[schema_open + 1 : schema_close]
    edits = []
    k = 0
    while k < len(schema_body):
        while k < len(schema_body) and schema_body[k] in " \t\n\r,":
            k += 1
        if k >= len(schema_body):
            break
        if schema_body[k] != '"':
            k += 1
            continue
        # parse string key
        k += 1
        while k < len(schema_body) and schema_body[k] != '"':
            if schema_body[k] == "\\":
                k += 2
                continue
            k += 1
        k += 1
        while k < len(schema_body) and schema_body[k] in " \t\n\r:":
            k += 1
        if k >= len(schema_body) or schema_body[k] != "{":
            depth_v = 0
            in_str = False
            while k < len(schema_body):
                c = schema_body[k]
                if in_str:
                    if c == "\\":
                        k += 2
                        continue
                    if c == '"':
                        in_str = False
                    k += 1
                    continue
                if c == '"':
                    in_str = True
                    k += 1
                    continue
                if c in "{[":
                    depth_v += 1
                elif c in "}]":
                    depth_v -= 1
                if depth_v == 0 and c == ",":
                    break
                k += 1
            continue
        obj_start = k
        depth_v = 0
        in_str = False
        obj_end = None
        while k < len(schema_body):
            c = schema_body[k]
            if in_str:
                if c == "\\":
                    k += 2
                    continue
                if c == '"':
                    in_str = False
                k += 1
                continue
            if c == '"':
                in_str = True
                k += 1
                continue
            if c == "{":
                depth_v += 1
            elif c == "}":
                depth_v -= 1
                if depth_v == 0:
                    obj_end = k
                    break
            k += 1
        if obj_end is None:
            return block_text, 0
        artifact_text = schema_body[obj_start : obj_end + 1]
        if '"_schema_version"' in artifact_text:
            k = obj_end + 1
            continue
        # Detect inline (single-line) vs multi-line object body.
        body_inner = schema_body[obj_start + 1 : obj_end]
        is_inline = "\n" not in body_inner
        i_pre = obj_end - 1
        while i_pre > obj_start and schema_body[i_pre] in " \t\n\r":
            i_pre -= 1
        prev_char = schema_body[i_pre] if i_pre > obj_start else "{"
        if is_inline:
            # `{...}` on one line. Insert before closing `}` with leading
            # comma when there is content; no comma for empty `{}`.
            insert_pos = obj_end
            if prev_char in "{":
                insert_text = '"_schema_version": "1"'
            else:
                insert_text = ', "_schema_version": "1"'
        else:
            line_start = schema_body.rfind("\n", 0, obj_end) + 1
            closing_indent = schema_body[line_start:obj_end]
            inner_indent = closing_indent + "  "
            if prev_char not in "{,":
                insert_pos = i_pre + 1
                insert_text = ",\n" + inner_indent + '"_schema_version": "1"'
            else:
                insert_pos = line_start
                insert_text = inner_indent + '"_schema_version": "1"\n'
        edits.append((insert_pos, insert_text))
        k = obj_end + 1

    if not edits:
        return block_text, 0
    new_schema_body = schema_body
    for pos, ins in sorted(edits, key=lambda x: -x[0]):
        new_schema_body = new_schema_body[:pos] + ins + new_schema_body[pos:]
    new_block = block_text[: schema_open + 1] + new_schema_body + block_text[schema_close:]
    return new_block, len(edits)


def main():
    text = WORKFLOW.read_text(encoding="utf-8")
    wf = json.loads(text)
    target_ids = [
        s["id"]
        for s in wf.get("steps", [])
        if s.get("phase") in PHASES and isinstance(s.get("artifact_schema"), dict)
    ]
    print(f"targets: {len(target_ids)}")
    total_edits = 0
    patched_steps = 0
    for sid in reversed(target_ids):
        rng = find_step_block_range(text, sid)
        if rng is None:
            print(f"WARN: step {sid} not located")
            continue
        start, end = rng
        block = text[start:end]
        new_block, n = patch_step_block(block)
        if n:
            text = text[:start] + new_block + text[end:]
            patched_steps += 1
            total_edits += n
    print(f"patched_steps={patched_steps} total_edits={total_edits}")
    wf2 = json.loads(text)  # validate
    missing = []
    for s in wf2.get("steps", []):
        if s.get("phase") not in PHASES:
            continue
        for art_name, art_def in (s.get("artifact_schema") or {}).items():
            if isinstance(art_def, dict) and "_schema_version" not in art_def:
                missing.append((s["id"], art_name))
    if missing:
        raise SystemExit(f"MISSING (count={len(missing)}): {missing[:10]}")
    print("all phase 0-6 artifacts now carry _schema_version")
    WORKFLOW.write_text(text, encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
