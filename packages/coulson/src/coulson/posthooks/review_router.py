"""Route a reviewer's fail issues to the producer step(s) to re-pend.

Tag path (deterministic): issue.target_artifact -> producer via the
artifact->producer index. Untagged/unmappable issues are returned for the
LLM fallback (the async driver that calls build_router_prompt /
parse_router_assignment lives in general_beckman.review_routing)."""
from __future__ import annotations


def map_tagged_issues(
    issues: list[dict], index: dict[str, list[str]]
) -> tuple[dict[str, list[dict]], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    unresolved: list[dict] = []
    for issue in issues:
        art = issue.get("target_artifact")
        producers = index.get(art) if art else None
        if not producers:
            unresolved.append(issue)
            continue
        grouped.setdefault(producers[0], []).append(issue)
    return grouped, unresolved


def build_router_prompt(issue: dict, candidates: list[tuple[str, str]]) -> str:
    lines = [f"- {sid}: produces {art}" for sid, art in candidates]
    return (
        "A reviewer flagged a problem. Pick the single producer step whose "
        "output most likely caused it, or 'unknown' if it cannot be attributed.\n\n"
        f"Problem (severity {issue.get('severity')}): {issue.get('problem')}\n\n"
        "Candidate producer steps:\n" + "\n".join(lines) + "\n\n"
        "Reply with exactly one line: 'STEP: <step_id>' or 'STEP: unknown'."
    )


def parse_router_assignment(raw: str, candidate_ids: list[str]) -> str | None:
    import re
    m = re.search(r"STEP:\s*([^\s]+)", raw or "", re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).strip()
    return val if val in candidate_ids else None
