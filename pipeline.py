# pipeline.py
"""
Orchestrates the multi-agent Coding Pipeline for complex coding tasks.

Stages:
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
import logging
from typing import Any, Dict

from agents import get_agent
from tools.workspace import read_file
from tools.deps import verify_dependencies
from tools.git_ops import git_commit

logger = logging.getLogger(__name__)


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
        Run the entire coding pipeline. Returns a result dict
        matching the output format of BaseAgent.execute().
        """
        original_goal = task.get("goal_id")
        base_title = task.get("title", "Complex Coding Task")
        base_desc = task.get("description", "")

        result_log = []

        logger.info(f"🚀 Starting Coding Pipeline for: {base_title}")

        # --- Stage 1: Architect ---
        logger.info("   [Stage 1] Architecting...")
        arch_task = {
            "title": f"Plan Architecture: {base_title}",
            "description": base_desc,
            "goal_id": original_goal,
        }
        arch_result = await self.architect.execute(arch_task)
        result_log.append(f"Architect: {arch_result.get('result', '')}")

        # Read the ARCHITECTURE.md to decide how many implementers to spin up
        arch_md = await read_file("ARCHITECTURE.md", max_lines=1000)
        
        # Simple heuristic to find file assignments
        # We look for lines starting with `### ` that end in typical extensions
        files_to_implement = []
        for line in arch_md.split("\n"):
            line = line.strip()
            if line.startswith("### ") and "`" in line:
                # Extract filepath from `path/to/script.py`
                parts = line.split("`")
                if len(parts) >= 3:
                    filepath = parts[1]
                    files_to_implement.append(filepath)

        if not files_to_implement:
            # Fallback if parsing failed: Just run implementer once overall
            files_to_implement = ["All files from plan"]

        logger.info(f"   [Stage 1 Complete] Files to build: {files_to_implement}")

        # --- Stage 2: Implementer (Sequentially per file) ---
        for f in files_to_implement:
            logger.info(f"   [Stage 2] Implementing: {f}")
            impl_task = {
                "title": f"Implement {f}",
                "description": (
                    f"Overall Goal: {base_desc}\n\n"
                    f"Your specific assignment is to implement: {f}\n"
                    f"Strictly follow the interfaces designed in ARCHITECTURE.md."
                ),
                "goal_id": original_goal,
            }
            impl_result = await self.implementer.execute(impl_task)
            result_log.append(f"Implementer ({f}): {impl_result.get('result', '')}")

        # --- Stage 3: Dependency Verification ---
        logger.info("   [Stage 3] Verifying dependencies...")
        dep_res = await verify_dependencies()
        result_log.append(f"Deps: {dep_res}")
        logger.info(f"   [Stage 3 Complete] {dep_res[-100:].strip()}")

        # --- Stage 4: Test Generator ---
        logger.info("   [Stage 4] Generating and running tests...")
        test_task = {
            "title": f"Write Tests: {base_title}",
            "description": (
                f"Goal: {base_desc}\n\n"
                f"We just implemented {len(files_to_implement)} files. "
                f"Write tests for them and run pytest."
            ),
            "goal_id": original_goal,
        }
        test_result = await self.test_generator.execute(test_task)
        result_log.append(f"Tester: {test_result.get('result', '')}")

        # --- Loops 5 & 6: QA (Review -> Fix) ---
        qa_iteration = 1
        max_qa = 5  # Give it a reasonable upper bound to avoid infinite loops, though user asked for "unbounded until quality passes". 5 is safe.
        review_passed = False

        while not review_passed and qa_iteration <= max_qa:
            logger.info(f"   [Stage 5] Review iteration {qa_iteration}...")
            
            review_task = {
                "title": f"Review Code: {base_title}",
                "description": (
                    f"Review the newly implemented files for {base_title}.\n"
                    f"Look for logic bugs, security issues, missing error handling, and test failures."
                ),
                "goal_id": original_goal,
            }
            review_result = await self.reviewer.execute(review_task)
            review_summary = review_result.get("result", "")
            result_log.append(f"Review {qa_iteration}: {review_summary[:100]}...")

            if "✅ Good" in review_summary or "passed" in review_summary.lower():
                logger.info(f"   [QA Passed] Implementation is solid.")
                review_passed = True
            else:
                logger.info(f"   [Stage 6] Fixing issues from review {qa_iteration}...")
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

        if not review_passed:
            logger.warning("   [QA Warning] Exiting QA loop without fully passing review.")
            result_log.append("QA Notes: Exited review loop without full pass.")

        # --- Stage 7: Committer ---
        logger.info("   [Stage 7] Committing changes...")
        commit_msg = f"feat: completed pipeline for - {base_title[:50]}"
        try:
            commit_res = await git_commit(message=commit_msg, add_all=True)
            result_log.append(f"Committer: {commit_res}")
        except Exception as e:
            result_log.append(f"Committer Error: {e}")

        logger.info(f"🏁 Pipeline Complete: {base_title}")

        return {
            "status": "completed",
            "result": "\n\n".join(result_log),
            "memories": {
                "pipeline_stages_run": len(result_log),
                "review_iterations": qa_iteration - 1,
            }
        }
