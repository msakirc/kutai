"""RC-A change-B validation: btable (estimate_for) vs char-based (requirements_for)
token estimate accuracy, measured against ground-truth model_call_tokens.

Read-only against the live KutAI DB. Prints a compact summary only.
"""
from __future__ import annotations

import json
import sqlite3
import statistics as st

from fatih_hoca.estimates import estimate_for
from fatih_hoca.requirements import AGENT_REQUIREMENTS

DB = "file:C:/Users/sakir/ai/kutai/kutai.db?mode=ro"
con = sqlite3.connect(DB, uri=True)
con.row_factory = sqlite3.Row

# ── btable (step_token_stats) ──
btable: dict = {}
for r in con.execute("SELECT * FROM step_token_stats"):
    btable[(r["agent_type"], r["workflow_step_id"] or "", r["workflow_phase"] or "")] = dict(r)

# ── ground-truth successful main_work calls + their task text ──
rows = con.execute(
    """
    SELECT mct.prompt_tokens AS ain, mct.completion_tokens AS aout,
           mct.agent_type AS agent, mct.provider AS provider,
           t.description AS descr, t.context AS ctx
    FROM model_call_tokens mct
    LEFT JOIN tasks t ON t.id = mct.task_id
    WHERE mct.success = 1 AND mct.call_category = 'main_work'
          AND mct.prompt_tokens IS NOT NULL AND mct.prompt_tokens > 0
    """
).fetchall()


class _Shim:
    __slots__ = ("agent_type", "context")


def char_in(descr, ctx_str) -> int:
    # mirrors requirements_for: max((len(desc)+len(ctx_json))//4, 1000)
    return max((len(descr or "") + len(ctx_str or "{}")) // 4, 1000)


def tmpl_out(agent: str) -> int:
    reqs = AGENT_REQUIREMENTS.get(agent) or AGENT_REQUIREMENTS["assistant"]
    return reqs.estimated_output_tokens


recs = []
fired = 0
for r in rows:
    agent = r["agent"] or "assistant"
    try:
        ctxd = json.loads(r["ctx"]) if r["ctx"] else {}
    except Exception:
        ctxd = {}
    if not isinstance(ctxd, dict):
        ctxd = {}
    sh = _Shim(); sh.agent_type = agent; sh.context = ctxd
    est = estimate_for(sh, btable=btable)
    key = (agent, ctxd.get("workflow_step_id") or "", ctxd.get("workflow_phase") or "")
    brow = btable.get(key)
    lvl1 = bool(brow and (brow.get("samples_n") or 0) >= 5)
    fired += lvl1
    recs.append({
        "ain": r["ain"], "aout": r["aout"] or 0,
        "bt_in": est.in_tokens, "bt_tot": est.in_tokens + est.out_tokens,
        "ch_in": char_in(r["descr"], r["ctx"]),
        "ch_tot": char_in(r["descr"], r["ctx"]) + tmpl_out(agent),
        "lvl1": lvl1, "cloud": (r["provider"] or "") not in ("local", "llama_cpp", ""),
    })

N = len(recs)


def pct(xs, p):
    s = sorted(xs); return s[min(len(s) - 1, int(p * len(s)))]


def block(label, sub):
    n = len(sub)
    if not n:
        print(f"\n[{label}] n=0"); return
    a = [x["ain"] for x in sub]
    print(f"\n[{label}] n={n}")
    print(f"  actual_in : med={st.median(a):.0f} mean={st.mean(a):.0f} p90={pct(a,.9):.0f} max={max(a)}")
    for key, name in (("bt_in", "btable_in"), ("ch_in", "char_in")):
        mae = st.mean([abs(x[key] - x["ain"]) for x in sub])
        bias = st.mean([x[key] - x["ain"] for x in sub])
        cov = sum(1 for x in sub if x[key] >= x["ain"]) / n
        print(f"  {name:9s}: med={st.median([x[key] for x in sub]):.0f} "
              f"MAE={mae:.0f} bias={bias:+.0f} covers_actual={cov:.0%}")


print(f"DB rows (successful main_work calls w/ task text): {N}")
print(f"btable level-1 fired (>=5 samples): {fired}/{N} = {fired/max(N,1):.0%}")
print(f"step_token_stats rows: {len(btable)} "
      f"(>=5 samples: {sum(1 for v in btable.values() if (v.get('samples_n') or 0) >= 5)})")

block("ALL", recs)
block("btable FIRED (>=5 samples)", [x for x in recs if x["lvl1"]])
block("btable FELL BACK to agent default", [x for x in recs if not x["lvl1"]])

# ── practical impact: per_call_too_large at a representative free-tier TPM ──
for cap in (6600, 8800):  # groq 6k & 8k tpm * 1.1 slack
    cloud = [x for x in recs if x["cloud"]]
    if not cloud:
        break
    real = sum(1 for x in cloud if x["ain"] + x["aout"] > cap)
    bt_fp = sum(1 for x in cloud if x["bt_tot"] > cap and x["ain"] + x["aout"] <= cap)
    ch_fp = sum(1 for x in cloud if x["ch_tot"] > cap and x["ain"] + x["aout"] <= cap)
    bt_fn = sum(1 for x in cloud if x["bt_tot"] <= cap and x["ain"] + x["aout"] > cap)
    ch_fn = sum(1 for x in cloud if x["ch_tot"] <= cap and x["ain"] + x["aout"] > cap)
    print(f"\n[per_call_too_large @cap={cap}] cloud_calls={len(cloud)} actually_over={real}")
    print(f"  btable: false_reject={bt_fp} missed_overflow={bt_fn}")
    print(f"  char  : false_reject={ch_fp} missed_overflow={ch_fn}")

con.close()
