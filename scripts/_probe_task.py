"""Read-only forensics for one DLQ'd task. Compact output."""
import json, sqlite3, sys

TID = int(sys.argv[1]) if len(sys.argv) > 1 else 165064
con = sqlite3.connect("file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro", uri=True)
con.row_factory = sqlite3.Row

t = con.execute("SELECT * FROM tasks WHERE id=?", (TID,)).fetchone()
if not t:
    print(f"task {TID} not found"); sys.exit(0)
print(f"=== task {TID} ===")
for k in ("agent_type", "runner", "status", "tier", "worker_attempts",
          "max_worker_attempts", "grade_attempts", "error"):
    print(f"  {k}: {t[k]}")
ctx = {}
try:
    ctx = json.loads(t["context"]) if t["context"] else {}
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
except Exception as e:
    print(f"  ctx parse err: {e}")
print(f"  ctx keys: {sorted(ctx)[:25]}")
for k in ("workflow_step_id", "workflow_phase", "step_name", "produces",
          "difficulty", "tools_hint", "agent", "max_iterations", "is_workflow_step"):
    if k in ctx:
        v = ctx[k]
        print(f"  ctx.{k}: {str(v)[:160]}")
res = t["result"]
print(f"  result[:300]: {str(res)[:300]}")

print(f"\n=== model_pick_log (task_id={TID}) ===")
for r in con.execute(
    "SELECT timestamp,picked_model,picked_score,call_category,urgency,success,error_category "
    "FROM model_pick_log WHERE task_id=? ORDER BY id", (TID,)):
    print(f"  {r['timestamp']} {r['picked_model']} score={r['picked_score']:.2f} "
          f"urg={r['urgency']} ok={r['success']} err={r['error_category']}")

print(f"\n=== model_call_tokens (task_id={TID}) ===")
rows = con.execute(
    "SELECT iteration_n,model,prompt_tokens,completion_tokens,reasoning_tokens,"
    "total_tokens,duration_ms,success FROM model_call_tokens WHERE task_id=? ORDER BY id",
    (TID,)).fetchall()
for r in rows:
    print(f"  it={r['iteration_n']} {r['model']} in={r['prompt_tokens']} "
          f"out={r['completion_tokens']} reason={r['reasoning_tokens']} "
          f"dur={r['duration_ms']}ms ok={r['success']}")
print(f"  total calls: {len(rows)}")
con.close()
