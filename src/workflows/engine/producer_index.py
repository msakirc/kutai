"""Map workflow artifacts to the step(s) that produce them, and resolve the
producer set a reviewer step reviews (its input_artifacts -> producers)."""
from __future__ import annotations


def build_producer_index(workflow: dict) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for step in workflow.get("steps", []):
        for art in step.get("output_artifacts") or []:
            index.setdefault(art, []).append(step["id"])
    return index


def producers_for_reviewer(
    workflow: dict, reviewer_id: str, index: dict[str, list[str]] | None = None
) -> list[str]:
    index = index or build_producer_index(workflow)
    reviewer = next(
        (s for s in workflow.get("steps", []) if s.get("id") == reviewer_id), None
    )
    if reviewer is None:
        return []
    out: list[str] = []
    for art in reviewer.get("input_artifacts") or []:
        for pid in index.get(art, []):
            if pid not in out and pid != reviewer_id:
                out.append(pid)
    return out


def producer_for_artifact(artifact: str, index: dict[str, list[str]]) -> str | None:
    producers = index.get(artifact) or []
    return producers[0] if producers else None
