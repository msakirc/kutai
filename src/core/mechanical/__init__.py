"""In-tree home for non-LLM task executors.

Phase 1: absorbs _auto_commit, workspace snapshots, and other mechanical work
currently scattered across the orchestrator.

Phase 2a: promote this whole directory to packages/mechanical_dispatcher/.
At that point it becomes a sibling to the LLM dispatcher and is selected by
orchestrator based on task.executor tag.
"""
