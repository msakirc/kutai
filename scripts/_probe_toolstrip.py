"""Pin the divergence: why does reverse_pitch (0.0z) keep write_file but
intake (0.0a.draft) end up with Allowed:[]? Runs the REAL tool-resolution
chain on both steps' contexts and prints allowed_tools at each stage."""
import io, json, types, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coulson import _apply_tools_hint, _apply_auto_strip, _apply_hint_from_targets_runtime

WF = r"C:\Users\sakir\Dropbox\Workspaces\kutay\src\workflows\i2p\i2p_v3.json"
wf = json.load(io.open(WF, encoding="utf-8"))
steps = {s["id"]: s for s in wf["steps"]}

# Writer base toolset (representative — tools_hint overrides it anyway)
BASE = ["read_file", "write_file", "file_tree", "web_search", "shell", "final_answer"]

def build_ctx(step):
    """Mirror what the expander puts into task_ctx for these fields."""
    ctx = {
        "workflow_step_id": step["id"],
        "is_workflow_step": True,
        "mission_id": 73,
        "tools_hint": step.get("tools_hint"),
        "artifact_schema": step.get("artifact_schema"),
        "produces": step.get("produces"),
        "output_artifacts": step.get("output_artifacts"),
        "input_artifacts": step.get("input_artifacts"),
    }
    # carry any explicit strip/allow flags from step context
    for k, v in (step.get("context") or {}).items():
        ctx[k] = v
    return ctx

for sid in ("0.0z", "0.0a.draft"):
    step = steps[sid]
    ctx = build_ctx(step)
    prof = types.SimpleNamespace(name="writer", allowed_tools=list(BASE))
    print(f"\n===== {sid}  (schema_type={ (step.get('artifact_schema') or {}) and list((step.get('artifact_schema') or {}).values())[0].get('type') if step.get('artifact_schema') else None }) =====")
    print("  produces:", step.get("produces"))
    print("  tools_hint(step):", step.get("tools_hint"))
    print("  start allowed:", prof.allowed_tools)
    try:
        _apply_hint_from_targets_runtime(ctx)
        print("  after hint_from_targets: ctx.tools_hint=", ctx.get("tools_hint"), "_allow_write_tools=", ctx.get("_allow_write_tools"))
    except Exception as e:
        print("  hint_from_targets ERR:", e)
    _apply_tools_hint(prof, ctx)
    print("  after tools_hint -> allowed:", prof.allowed_tools)
    _apply_auto_strip(prof, ctx)
    print("  after auto_strip -> allowed:", prof.allowed_tools, "  <<< FINAL")
