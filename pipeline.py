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

import logging
from typing import Any, Dict

from agents import get_agent
from tools.workspace import read_file
from tools.deps import verify_dependencies
from pipeline_utils import (
    classify_complexity,
    get_stages_for_complexity,
    generate_pr_summary,
    _load_progress,
    _save_progress,
    _cleanup_progress,
    _get_convention_context,
    _get_codebase_map_context,
)
from tools.git_ops import git_commit

logger = logging.getLogger(__name__)


# ─── Pipeline ───────────────────────────────────────────────────────────────

class CodingPipeline:
    """Multi-stage orchestration for complex coding features."""

    def __init__(self):
        self.architect = get_agent("architect")
        self.implementer = get_agent("implementer")
        self.test_generator = get_agent("test_generator")
        self.reviewer = get_agent("reviewer")
        self.fixer = get_agent("fixer")

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the coding pipeline with adaptive stage selection.

        Classifies task complexity and runs only the necessary stages.
        Supports incremental implementation (resume from failure).
        """
        original_goal = task.get("goal_id")
        base_title = task.get("title", "Complex Coding Task")
        base_desc = task.get("description", "")
        context = task.get("context", {})

        # Check for explicit mode override in context
        explicit_mode = context.get("pipeline_mode")

        result_log = []
        stages_run = []
        files_implemented = []

        # ── Phase 8: Classify complexity ──
        complexity = explicit_mode or classify_complexity(base_title, base_desc)
        stages = get_stages_for_complexity(complexity)

        logger.info(
            f"🚀 Starting Coding Pipeline for: {base_title} "
            f"(complexity: {complexity}, stages: {stages})"
        )

        # ── Phase 8: Load incremental progress ──
        progress = _load_progress(original_goal)
        completed_files = set(progress.get("completed_files", []))

        # ── Phase 8: Build convention + codebase context ──
        convention_ctx = _get_convention_context()
        codebase_map_ctx = _get_codebase_map_context()
        extra_context = convention_ctx + codebase_map_ctx

        # ── Stage: Architect ──
        if "architect" in stages:
            logger.info("   [Architect] Planning architecture...")
            arch_task = {
                "title": f"Plan Architecture: {base_title}",
                "description": base_desc + extra_context,
                "goal_id": original_goal,
            }
            arch_result = await self.architect.execute(arch_task)
            result_log.append(f"Architect: {arch_result.get('result', '')}")
            stages_run.append("architect")

        # ── Parse files to implement ──
        files_to_implement = []
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

        # ── Stage: Test (TDD mode — tests before implementation) ──
        if "test" in stages and complexity == "tdd":
            logger.info("   [TDD] Writing tests first...")
            test_task = {
                "title": f"Write Tests (TDD): {base_title}",
                "description": (
                    f"Goal: {base_desc}\n\n"
                    f"TDD MODE: Write comprehensive tests FIRST based on the "
                    f"architecture plan. The implementation will follow.\n"
                    f"Files to be implemented: {', '.join(files_to_implement)}\n"
                    f"Write tests that will pass once implementation is complete."
                    + extra_context
                ),
                "goal_id": original_goal,
            }
            test_result = await self.test_generator.execute(test_task)
            result_log.append(f"TDD Tests: {test_result.get('result', '')}")
            stages_run.append("test_tdd")

        # ── Stage: Implement ──
        if "implement" in stages:
            for f in files_to_implement:
                # Phase 8: Skip already-completed files (incremental)
                if f in completed_files:
                    logger.info(f"   [Implement] Skipping (already done): {f}")
                    result_log.append(f"Implementer ({f}): skipped (already completed)")
                    continue

                logger.info(f"   [Implement] Implementing: {f}")
                impl_task = {
                    "title": f"Implement {f}",
                    "description": (
                        f"Overall Goal: {base_desc}\n\n"
                        f"Your specific assignment is to implement: {f}\n"
                        f"Strictly follow the interfaces designed in ARCHITECTURE.md."
                        + extra_context
                    ),
                    "goal_id": original_goal,
                }
                impl_result = await self.implementer.execute(impl_task)
                result_log.append(f"Implementer ({f}): {impl_result.get('result', '')}")
                files_implemented.append(f)

                # Track progress incrementally
                completed_files.add(f)
                _save_progress(original_goal, {
                    "completed_files": list(completed_files),
                    "stage": "implement",
                })

            stages_run.append("implement")

        # ── Stage: Fix (bugfix mode — run fixer directly) ──
        if "fix" in stages:
            logger.info("   [Fix] Running fixer for bug...")
            fix_task = {
                "title": f"Fix Bug: {base_title}",
                "description": (
                    f"Bug report: {base_desc}\n\n"
                    f"Investigate and fix this bug. Read relevant files, "
                    f"identify the root cause, and apply a fix."
                    + extra_context
                ),
                "goal_id": original_goal,
            }
            fix_result = await self.fixer.execute(fix_task)
            result_log.append(f"Fixer: {fix_result.get('result', '')}")
            stages_run.append("fix")

        # ── Stage: Dependencies ──
        if "deps" in stages:
            logger.info("   [Deps] Verifying dependencies...")
            dep_res = await verify_dependencies()
            result_log.append(f"Deps: {dep_res}")
            stages_run.append("deps")

        # ── Stage: Test (normal mode — tests after implementation) ──
        if "test" in stages and complexity != "tdd":
            logger.info("   [Test] Generating and running tests...")
            test_task = {
                "title": f"Write Tests: {base_title}",
                "description": (
                    f"Goal: {base_desc}\n\n"
                    f"We just implemented {len(files_to_implement)} files. "
                    f"Write tests for them and run pytest."
                    + extra_context
                ),
                "goal_id": original_goal,
            }
            test_result = await self.test_generator.execute(test_task)
            result_log.append(f"Tester: {test_result.get('result', '')}")
            stages_run.append("test")

        # ── Stage: Review + Fix loop ──
        qa_iteration = 1
        max_qa = 5
        review_passed = False

        if "review_fix" in stages or "review" in stages:
            while not review_passed and qa_iteration <= max_qa:
                logger.info(f"   [Review] Iteration {qa_iteration}...")

                review_task = {
                    "title": f"Review Code: {base_title}",
                    "description": (
                        f"Review the newly implemented files for {base_title}.\n"
                        f"Look for logic bugs, security issues, "
                        f"missing error handling, and test failures."
                    ),
                    "goal_id": original_goal,
                }
                review_result = await self.reviewer.execute(review_task)
                review_summary = review_result.get("result", "")
                result_log.append(f"Review {qa_iteration}: {review_summary[:100]}...")

                if "✅ Good" in review_summary or "passed" in review_summary.lower():
                    logger.info("   [QA Passed] Implementation is solid.")
                    review_passed = True
                elif "review_fix" in stages:
                    logger.info(f"   [Fix] Fixing issues from review {qa_iteration}...")
                    fix_task = {
                        "title": f"Fix QA Issues: {base_title}",
                        "description": (
                            f"The code reviewer found issues. Fix them all.\n\n"
                            f"Review Feedback:\n{review_summary}"
                        ),
                        "goal_id": original_goal,
                    }
                    fix_result = await self.fixer.execute(fix_task)
                    result_log.append(f"Fixer {qa_iteration}: {fix_result.get('result', '')}")
                    qa_iteration += 1
                else:
                    # Review-only mode (oneliner) — no fix loop
                    break

            stages_run.append("review")
            if not review_passed and "review_fix" in stages:
                result_log.append("QA Notes: Exited review loop without full pass.")

        # ── Stage: Commit ──
        if "commit" in stages:
            logger.info("   [Commit] Committing changes...")
            commit_msg = f"feat: completed pipeline for - {base_title[:50]}"
            try:
                commit_res = await git_commit(message=commit_msg, add_all=True)
                result_log.append(f"Committer: {commit_res}")
                stages_run.append("commit")
            except Exception as e:
                result_log.append(f"Committer Error: {e}")

        # ── Phase 8: Generate PR summary ──
        pr_summary = await generate_pr_summary(
            title=base_title,
            files_changed=files_implemented or files_to_implement,
            stages_run=stages_run,
            review_iterations=qa_iteration - 1,
            complexity=complexity,
        )
        result_log.append(f"\n{pr_summary}")

        # ── Clean up progress on success ──
        _cleanup_progress(original_goal)

        logger.info(f"🏁 Pipeline Complete: {base_title} (complexity: {complexity})")

        return {
            "status": "completed",
            "result": "\n\n".join(result_log),
            "memories": {
                "pipeline_stages_run": len(stages_run),
                "review_iterations": qa_iteration - 1,
                "complexity": complexity,
                "files_implemented": files_implemented,
            },
        }
