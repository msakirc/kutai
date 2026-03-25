# pipeline_context.py
"""
Accumulates results across pipeline stages so each stage
has awareness of what prior stages produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("workflows.pipeline.pipeline_context")


@dataclass
class StageResult:
    """Output from one pipeline stage."""
    stage_name: str
    agent_type: str
    model_used: str
    result_text: str
    files_touched: list[str] = field(default_factory=list)
    cost: float = 0.0
    success: bool = True


@dataclass
class PipelineContext:
    """
    Accumulates context across pipeline stages.

    Each stage appends its result. Subsequent stages receive
    a formatted summary of everything that happened before them.
    """
    mission_title: str = ""
    mission_description: str = ""
    mission_id: int | None = None
    complexity: str = "feature"

    architecture_plan: str = ""
    stages: list[StageResult] = field(default_factory=list)
    files_implemented: list[str] = field(default_factory=list)
    test_results: str = ""
    review_feedback: list[str] = field(default_factory=list)
    total_cost: float = 0.0

    def record_stage(self, result: StageResult) -> None:
        """Record a completed stage's output."""
        self.stages.append(result)
        self.total_cost += result.cost

        if result.stage_name == "architect":
            self.architecture_plan = result.result_text
        elif result.stage_name == "test":
            self.test_results = result.result_text
        elif result.stage_name == "review":
            self.review_feedback.append(result.result_text)

        if result.files_touched:
            for f in result.files_touched:
                if f not in self.files_implemented:
                    self.files_implemented.append(f)

    def format_for_stage(
        self,
        target_stage: str,
        target_file: str | None = None,
        max_chars: int = 12000,
    ) -> str:
        """
        Build a context string tailored for the next stage.

        Different stages need different context:
        - implementer: architecture plan + prior implementations
        - test_generator: architecture + all implementations
        - reviewer: architecture + implementations + test results
        - fixer: all of the above + review feedback
        """
        parts: list[str] = []

        # ── Architecture plan (always useful) ──
        if self.architecture_plan:
            plan_text = self.architecture_plan
            if target_stage == "implement" and len(plan_text) > 3000:
                # Implementer needs the full plan for interface contracts
                plan_text = plan_text[:3000] + "\n... [see ARCHITECTURE.md for full plan]"
            elif len(plan_text) > 2000:
                plan_text = plan_text[:2000] + "\n... [truncated]"
            parts.append(f"## Architecture Plan\n{plan_text}")

        # ── Prior implementations ──
        if target_stage in ("implement", "test", "review", "fix"):
            impl_stages = [
                s for s in self.stages
                if s.stage_name == "implement" and s.success
            ]

            if impl_stages:
                # For implementer: show what was already built
                # so it knows interfaces and can avoid duplication
                if target_stage == "implement":
                    parts.append("## Already Implemented Files")
                    budget = max_chars // 2
                    used = 0
                    for s in impl_stages:
                        # Skip the current file if re-implementing
                        if target_file and any(
                            target_file in f for f in s.files_touched
                        ):
                            continue
                        entry = (
                            f"### {', '.join(s.files_touched) or s.stage_name}\n"
                            f"{s.result_text}"
                        )
                        if used + len(entry) > budget:
                            # Truncate older implementations more aggressively
                            entry = (
                                f"### {', '.join(s.files_touched) or s.stage_name}\n"
                                f"{s.result_text[:500]}\n... [truncated]"
                            )
                        parts.append(entry)
                        used += len(entry)

                # For reviewer/test: just list files and brief summaries
                elif target_stage in ("test", "review"):
                    parts.append("## Implemented Files Summary")
                    for s in impl_stages:
                        files = ", ".join(s.files_touched) or "unknown"
                        summary = s.result_text[:300]
                        parts.append(f"- **{files}**: {summary}")

                # For fixer: brief list + the specific review feedback
                elif target_stage == "fix":
                    parts.append(
                        f"## Files Changed\n"
                        + "\n".join(
                            f"- {f}" for f in self.files_implemented
                        )
                    )

        # ── Test results ──
        if target_stage in ("review", "fix") and self.test_results:
            test_text = self.test_results
            if len(test_text) > 2000:
                test_text = test_text[:2000] + "\n... [truncated]"
            parts.append(f"## Test Results\n{test_text}")

        # ── Review feedback (for fixer) ──
        if target_stage == "fix" and self.review_feedback:
            latest = self.review_feedback[-1]
            if len(latest) > 2000:
                latest = latest[:2000] + "\n... [truncated]"
            parts.append(f"## Review Feedback (Latest)\n{latest}")

            if len(self.review_feedback) > 1:
                parts.append(
                    f"_This is review iteration {len(self.review_feedback)}. "
                    f"Prior reviews also found issues._"
                )

        # ── Cost tracking ──
        parts.append(
            f"\n---\n_Pipeline cost so far: ${self.total_cost:.4f} | "
            f"Stages completed: {len(self.stages)}_"
        )

        combined = "\n\n".join(parts)

        # Final length check
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n\n... [context truncated to fit]"

        return combined
