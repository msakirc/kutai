# pipeline.py
"""
Orchestrates the multi-agent Coding Pipeline for complex coding tasks.

Phase 8 enhancements:
  - Adaptive pipeline stages (classify complexity → skip unnecessary stages)
  - Test-driven mode (tests first, then implement)
  - PR workflow (diff summary on completion)
  - Incremental implementation (resume from last incomplete file)
  - Context-aware code generation (inject conventions)
  - Codebase indexing (agents query index instead of browsing)

Original Stages:
1. Architect -> structured plan (ARCHITECTURE.md)
2. Implementer x N -> one file per call
3. verify_dependencies -> auto-install missing packages
4. TestGenerator -> writes + runs tests
5. Reviewer -> structured code review
6. Fixer -> apply review/test fixes
   (Loops 5 & 6 until review passes)
7. Committer -> git_commit
"""
import json
import os
from typing import Any, Dict

from src.infra.logging_config import get_logger
from src.agents import get_agent
from src.tools.workspace import read_file
from src.tools.deps import verify_dependencies
from .pipeline_context import PipelineContext, StageResult
from .pipeline_utils import (
    classify_complexity,
    get_stages_for_complexity,
    generate_pr_summary,
    _load_progress,
    _save_progress,
    _cleanup_progress,
    _get_convention_context,
    _get_codebase_map_context,
)
from src.tools.git_ops import git_commit

logger = get_logger("workflows.pipeline.pipeline")


# ─── Pipeline ───────────────────────────────────────────────────────────────

class CodingPipeline:
    """Multi-stage orchestration for complex coding features."""

    def __init__(self):
        self.architect = get_agent("architect")
        self.implementer = get_agent("implementer")
        self.test_generator = get_agent("test_generator")
        self.reviewer = get_agent("reviewer")
        self.fixer = get_agent("fixer")
        self._implementer_model: str = ""
# pipeline.py — replace the entire run() method

    async def run(self, task: dict) -> dict:
        """
        Run the coding pipeline with adaptive stage selection
        and accumulated context between stages.
        """
        original_goal = task.get("goal_id")
        base_title = task.get("title", "Complex Coding Task")
        base_desc = task.get("description", "")
        context = task.get("context", {})
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (json.JSONDecodeError, TypeError):
                context = {}

        explicit_mode = context.get("pipeline_mode")

        result_log: list[str] = []
        stages_run: list[str] = []

        # ── Classify complexity ──
        complexity = explicit_mode or classify_complexity(base_title, base_desc)
        stages = get_stages_for_complexity(complexity)

        logger.info(
            f"🚀 Pipeline: {base_title} "
            f"(complexity: {complexity}, stages: {stages})"
        )

        # ── Initialize pipeline context ──
        pipe_ctx = PipelineContext(
            goal_title=base_title,
            goal_description=base_desc,
            goal_id=original_goal,
            complexity=complexity,
        )

        # ── Load incremental progress ──
        progress = _load_progress(original_goal)
        completed_files = set(progress.get("completed_files", []))

        # ── Convention + codebase context (static, computed once) ──
        extra_context = _get_convention_context() + _get_codebase_map_context()

        # Track which model implemented code (for review diversity)
        implementer_model: str = ""

        # ─────────────────────────────────────────────────────────
        #  Stage: ARCHITECT
        # ─────────────────────────────────────────────────────────
        if "architect" in stages:
            logger.info("   [Architect] Planning architecture...")
            arch_task = {
                "title": f"Plan Architecture: {base_title}",
                "description": base_desc + extra_context,
                "goal_id": original_goal,
                "priority": 8,
                "context": json.dumps({"prefer_quality": True}),
            }
            arch_result = await self.architect.execute(arch_task)
            arch_text = arch_result.get("result", "")

            pipe_ctx.record_stage(StageResult(
                stage_name="architect",
                agent_type="architect",
                model_used=arch_result.get("model", ""),
                result_text=arch_text,
                cost=arch_result.get("cost", 0),
            ))
            result_log.append(f"Architect: {arch_text}")
            stages_run.append("architect")

        # ── Parse files to implement ──
        files_to_implement: list[str] = []
        try:
            arch_md = await read_file("ARCHITECTURE.md", max_lines=1000)
            for line in arch_md.split("\n"):
                line = line.strip()
                if line.startswith("### ") and "`" in line:
                    parts = line.split("`")
                    if len(parts) >= 3:
                        files_to_implement.append(parts[1])
        except Exception:
            pass

        if not files_to_implement:
            files_to_implement = ["All files from plan"]

        logger.info(f"   Files to build: {files_to_implement}")

        # ─────────────────────────────────────────────────────────
        #  Stage: TEST (TDD mode — tests before implementation)
        # ─────────────────────────────────────────────────────────
        if "test" in stages and complexity == "tdd":
            logger.info("   [TDD] Writing tests first...")

            # Build context: architect output + file list
            tdd_context = pipe_ctx.format_for_stage("test")

            test_task = {
                "title": f"Write Tests (TDD): {base_title}",
                "description": (
                    f"Goal: {base_desc}\n\n"
                    f"TDD MODE: Write comprehensive tests FIRST based on the "
                    f"architecture plan. The implementation will follow.\n"
                    f"Files to be implemented: {', '.join(files_to_implement)}\n"
                    f"Write tests that will pass once implementation is complete.\n\n"
                    f"{tdd_context}"
                    + extra_context
                ),
                "goal_id": original_goal,
            }
            test_result = await self.test_generator.execute(test_task)
            test_text = test_result.get("result", "")

            pipe_ctx.record_stage(StageResult(
                stage_name="test",
                agent_type="test_generator",
                model_used=test_result.get("model", ""),
                result_text=test_text,
                cost=test_result.get("cost", 0),
            ))
            result_log.append(f"TDD Tests: {test_text}")
            stages_run.append("test_tdd")

        # ─────────────────────────────────────────────────────────
        #  Stage: IMPLEMENT
        # ─────────────────────────────────────────────────────────
        if "implement" in stages:
            for idx, f in enumerate(files_to_implement):
                if f in completed_files:
                    logger.info(f"   [Implement] Skipping (done): {f}")
                    result_log.append(f"Implementer ({f}): skipped (already completed)")
                    continue

                logger.info(f"   [Implement] Implementing: {f}")

                # Build context WITH prior stages and prior files
                impl_stage_ctx = pipe_ctx.format_for_stage(
                    "implement", target_file=f,
                )

                impl_task = {
                    "title": f"Implement {f}",
                    "description": (
                        f"Overall Goal: {base_desc}\n\n"
                        f"Your specific assignment is to implement: {f}\n"
                        f"Strictly follow the interfaces designed in ARCHITECTURE.md.\n\n"
                        f"{impl_stage_ctx}"
                        + extra_context
                    ),
                    "goal_id": original_goal,
                }
                impl_result = await self.implementer.execute(impl_task)
                impl_text = impl_result.get("result", "")
                implementer_model = impl_result.get("model", "")

                pipe_ctx.record_stage(StageResult(
                    stage_name="implement",
                    agent_type="implementer",
                    model_used=implementer_model,
                    result_text=impl_text,
                    files_touched=[f],
                    cost=impl_result.get("cost", 0),
                ))
                result_log.append(f"Implementer ({f}): {impl_text}")

                completed_files.add(f)
                _save_progress(original_goal, {
                    "completed_files": list(completed_files),
                    "stage": "implement",
                })

            pipe_ctx.files_implemented = list(completed_files)
            stages_run.append("implement")

        # ─────────────────────────────────────────────────────────
        #  Stage: FIX (bugfix mode — direct fixer entry point)
        # ─────────────────────────────────────────────────────────
        if "fix" in stages:
            logger.info("   [Fix] Running fixer for bug...")

            fix_stage_ctx = pipe_ctx.format_for_stage("fix")

            fix_task = {
                "title": f"Fix Bug: {base_title}",
                "description": (
                    f"Bug report: {base_desc}\n\n"
                    f"Investigate and fix this bug. Read relevant files, "
                    f"identify the root cause, and apply a fix.\n\n"
                    f"{fix_stage_ctx}"
                    + extra_context
                ),
                "goal_id": original_goal,
            }
            fix_result = await self.fixer.execute(fix_task)
            fix_text = fix_result.get("result", "")

            pipe_ctx.record_stage(StageResult(
                stage_name="fix",
                agent_type="fixer",
                model_used=fix_result.get("model", ""),
                result_text=fix_text,
                cost=fix_result.get("cost", 0),
            ))
            result_log.append(f"Fixer: {fix_text}")
            stages_run.append("fix")

        # ─────────────────────────────────────────────────────────
        #  Stage: DEPENDENCIES
        # ─────────────────────────────────────────────────────────
        if "deps" in stages:
            logger.info("   [Deps] Verifying dependencies...")
            dep_res = await verify_dependencies()
            result_log.append(f"Deps: {dep_res}")
            stages_run.append("deps")

        # ─────────────────────────────────────────────────────────
        #  Stage: TEST (normal — tests after implementation)
        # ─────────────────────────────────────────────────────────
        if "test" in stages and complexity != "tdd":
            logger.info("   [Test] Generating and running tests...")

            test_stage_ctx = pipe_ctx.format_for_stage("test")

            test_task = {
                "title": f"Write Tests: {base_title}",
                "description": (
                    f"Goal: {base_desc}\n\n"
                    f"We just implemented {len(files_to_implement)} files. "
                    f"Write tests for them and run pytest.\n\n"
                    f"{test_stage_ctx}"
                    + extra_context
                ),
                "goal_id": original_goal,
            }
            test_result = await self.test_generator.execute(test_task)
            test_text = test_result.get("result", "")

            pipe_ctx.record_stage(StageResult(
                stage_name="test",
                agent_type="test_generator",
                model_used=test_result.get("model", ""),
                result_text=test_text,
                cost=test_result.get("cost", 0),
            ))
            result_log.append(f"Tester: {test_text}")
            stages_run.append("test")

        # ─────────────────────────────────────────────────────────
        #  Stage: COVERAGE (quality gate after tests)
        # ─────────────────────────────────────────────────────────
        if "test" in stages_run:
            try:
                from src.tools.coverage import get_coverage_summary
                from src.languages import detect_language
                # Detect project language from implementation files
                impl_exts = [
                    os.path.splitext(f)[1]
                    for f in files_to_implement if os.path.splitext(f)[1]
                ]
                proj_lang = detect_language(impl_exts) or "python"
                cov_report = await get_coverage_summary(
                    project_root=pipe_ctx.workspace or ".",
                    language=proj_lang,
                )
                if cov_report:
                    result_log.append(f"Coverage: {cov_report[:200]}")
                    pipe_ctx.record_stage(StageResult(
                        stage_name="coverage",
                        agent_type="tool",
                        model_used="",
                        result_text=cov_report,
                        cost=0,
                    ))
                    stages_run.append("coverage")
                    logger.info(f"   [Coverage] {cov_report[:100]}")
            except Exception as cov_err:
                logger.debug(f"Coverage check skipped: {cov_err}")

        # ─────────────────────────────────────────────────────────
        #  Stage: REVIEW + FIX loop (with diversity + escalation)
        # ─────────────────────────────────────────────────────────
        qa_iteration = 1
        max_qa = 5
        review_passed = False

        if "review_fix" in stages or "review" in stages:
            while not review_passed and qa_iteration <= max_qa:
                logger.info(f"   [Review] Iteration {qa_iteration}...")

                review_stage_ctx = pipe_ctx.format_for_stage("review")

                # Enforce model diversity: reviewer ≠ implementer
                review_ctx_dict: dict = {}
                if implementer_model:
                    review_ctx_dict["exclude_models"] = [implementer_model]
                review_ctx_dict["prefer_quality"] = True

                review_task = {
                    "title": f"Review Code: {base_title}",
                    "description": (
                        f"Review the newly implemented files for {base_title}.\n"
                        f"Look for logic bugs, security issues, "
                        f"missing error handling, and test failures.\n\n"
                        f"{review_stage_ctx}"
                    ),
                    "goal_id": original_goal,
                    "context": json.dumps(review_ctx_dict),
                }
                review_result = await self.reviewer.execute(review_task)
                review_summary = review_result.get("result", "")

                pipe_ctx.record_stage(StageResult(
                    stage_name="review",
                    agent_type="reviewer",
                    model_used=review_result.get("model", ""),
                    result_text=review_summary,
                    cost=review_result.get("cost", 0),
                ))
                result_log.append(
                    f"Review {qa_iteration}: {review_summary[:100]}..."
                )

                # Phase 10.3: Parse structured reviewer JSON
                review_verdict = ""
                review_issues: list[dict] = []
                try:
                    # Reviewer should return JSON; try to parse
                    _raw = review_summary.strip()
                    if _raw.startswith("```"):
                        _raw = _raw.split("\n", 1)[1] if "\n" in _raw else _raw[3:]
                        _raw = _raw.rsplit("```", 1)[0]
                    parsed_review = json.loads(_raw)
                    review_verdict = parsed_review.get("verdict", "")
                    review_issues = parsed_review.get("issues", [])
                except (json.JSONDecodeError, AttributeError):
                    # Fallback: treat as prose
                    pass

                # Determine pass/fail from structured verdict or string heuristics
                if review_verdict == "pass" or (
                    not review_verdict and (
                        "✅ Good" in review_summary
                        or "passed" in review_summary.lower()
                    )
                ):
                    logger.info("   [QA Passed] Implementation is solid.")
                    review_passed = True

                elif "review_fix" in stages:
                    logger.info(
                        f"   [Fix] Fixing issues from review "
                        f"{qa_iteration}..."
                    )

                    fix_stage_ctx = pipe_ctx.format_for_stage("fix")

                    # Escalate quality after repeated failures
                    fix_ctx_dict: dict = {}
                    if qa_iteration >= 3:
                        fix_ctx_dict["prefer_quality"] = True

                    # Build structured issue description for fixer
                    if review_issues:
                        issue_lines = ["## Issues to Fix\n"]
                        for i, iss in enumerate(review_issues, 1):
                            sev = iss.get("severity", "medium").upper()
                            fname = iss.get("file", "unknown")
                            line = iss.get("line", "")
                            desc = iss.get("description", "")
                            fix_hint = iss.get("suggested_fix", "")
                            loc = f"{fname}:{line}" if line else fname
                            issue_lines.append(
                                f"{i}. [{sev}] {loc}: {desc}"
                            )
                            if fix_hint:
                                issue_lines.append(f"   Suggested: {fix_hint}")
                        structured_issues = "\n".join(issue_lines)
                    else:
                        structured_issues = review_summary

                    fix_task = {
                        "title": f"Fix QA Issues: {base_title}",
                        "description": (
                            f"The code reviewer found issues. "
                            f"Fix them all.\n\n"
                            f"{structured_issues}\n\n"
                            f"{fix_stage_ctx}"
                        ),
                        "goal_id": original_goal,
                        "context": json.dumps(fix_ctx_dict),
                    }
                    fix_result = await self.fixer.execute(fix_task)
                    fix_text = fix_result.get("result", "")

                    pipe_ctx.record_stage(StageResult(
                        stage_name="fix",
                        agent_type="fixer",
                        model_used=fix_result.get("model", ""),
                        result_text=fix_text,
                        cost=fix_result.get("cost", 0),
                    ))
                    result_log.append(
                        f"Fixer {qa_iteration}: {fix_text}"
                    )
                    qa_iteration += 1
                else:
                    break

            stages_run.append("review")
            if not review_passed and "review_fix" in stages:
                result_log.append(
                    "QA Notes: Exited review loop without full pass."
                )

        # ─────────────────────────────────────────────────────────
        #  Stage: COMMIT
        # ─────────────────────────────────────────────────────────
        if "commit" in stages:
            logger.info("   [Commit] Committing changes...")
            commit_msg = (
                f"feat: completed pipeline for - {base_title[:50]}"
            )
            try:
                commit_res = await git_commit(
                    message=commit_msg, add_all=True,
                )
                result_log.append(f"Committer: {commit_res}")
                stages_run.append("commit")
            except Exception as e:
                result_log.append(f"Committer Error: {e}")

        # ── PR summary ──
        pr_summary = await generate_pr_summary(
            title=base_title,
            files_changed=pipe_ctx.files_implemented,
            stages_run=stages_run,
            review_iterations=qa_iteration - 1,
            complexity=complexity,
        )
        result_log.append(f"\n{pr_summary}")

        # ── Phase 10.6: Create PR if on a goal branch ──
        try:
            from src.tools.git_ops import get_current_branch
            from src.tools.shell import run_shell
            branch = await get_current_branch(pipe_ctx.workspace or "")
            if branch and branch.startswith("goal/") and branch != "main":
                # Push branch and create PR via gh CLI
                push_out = await run_shell(
                    f"git push -u origin {branch} 2>&1 || true",
                    timeout=60,
                )
                pr_body = (
                    f"## Summary\n"
                    f"Pipeline completed for: **{base_title}**\n\n"
                    f"- Complexity: {complexity}\n"
                    f"- Stages: {', '.join(stages_run)}\n"
                    f"- Review iterations: {qa_iteration - 1}\n"
                    f"- Files changed: {len(pipe_ctx.files_implemented)}\n"
                    f"- Cost: ${pipe_ctx.total_cost:.4f}\n\n"
                    f"Generated by kutay coding pipeline."
                )
                pr_out = await run_shell(
                    f'gh pr create --title "{base_title[:70]}" '
                    f'--body "{pr_body.replace(chr(34), chr(39))}" '
                    f"--base main 2>&1 || true",
                    timeout=30,
                )
                if pr_out and "http" in pr_out:
                    result_log.append(f"PR created: {pr_out.strip()}")
                    logger.info(f"   [PR] Created: {pr_out.strip()}")
                elif "already exists" in (pr_out or ""):
                    result_log.append("PR already exists for this branch.")
                else:
                    logger.debug(f"PR creation skipped: {pr_out}")
        except Exception as pr_err:
            logger.debug(f"PR creation failed (non-critical): {pr_err}")

        # ── Cleanup ──
        _cleanup_progress(original_goal)

        logger.info(
            f"🏁 Pipeline Complete: {base_title} "
            f"(complexity: {complexity}, "
            f"cost: ${pipe_ctx.total_cost:.4f})"
        )

        return {
            "status": "completed",
            "result": "\n\n".join(result_log),
            "cost": pipe_ctx.total_cost,
            "memories": {
                "pipeline_stages_run": len(stages_run),
                "review_iterations": qa_iteration - 1,
                "complexity": complexity,
                "files_implemented": pipe_ctx.files_implemented,
                "total_cost": pipe_ctx.total_cost,
            },
        }
