# Context Layers Reference

Each layer in `_build_context()` and what it provides to the agent prompt.

## Ungatable (always injected when present)

| Layer | ID | Source | What it provides |
|-------|----|--------|-----------------|
| Task core | `task` | `task.title` + `task.description` | The actual instruction — what the agent must do |
| Task context | `task_ctx` | `task.context` dict | Inline data: workspace_snapshot, tool_result, user_clarification, extra fields |

## Gatable (controlled by context policy)

| Layer | ID | Source | What it provides |
|-------|----|--------|-----------------|
| Dependencies | `deps` | `get_completed_dependency_results()` | Results from prerequisite tasks (e.g., step 1 output feeds step 2) |
| Prior steps | `prior` | `task.context.prior_steps` | Inline workflow step results (orchestrator-injected fallback when deps aren't in DB) |
| Conversation | `convo` | `task.context.recent_conversation` | Last few user-AI exchanges — needed for follow-up references like "list them", "do it again" |
| Ambient | `ambient` | `assembler.assemble_ambient_context()` | Mission-wide contextual info (cross-task state, mission goals) |
| Project profile | `profile` | `get_project_profile_for_task()` | Repository structure, language, framework — useful for code tasks |
| Blackboard | `board` | `get_or_create_blackboard()` | Shared mission data updated in real-time by agents (key-value scratchpad) |
| Skills | `skills` | `find_relevant_skills()` via ChromaDB | Execution recipes: "we've seen a similar task, here's what worked" |
| API enrichment | `api` | `task.context.api_enrichment` | Free API suggestions from smart resource integration |
| RAG | `rag` | `retrieve_context()` via ChromaDB | Vector-retrieved knowledge: episodic memory, error patterns, web knowledge, semantic facts |
| Preferences | `prefs` | `get_user_preferences()` via ChromaDB | Detected user preferences (language, style, tools). Currently noisy — see memory-subsystem-findings.md #3 |
| Memory | `memory` | `recall_memory()` via SQLite | Mission-scoped key-value facts stored during execution |

## Token Cost Estimates (current, ungated)

| Layer | Typical tokens | Can exceed |
|-------|---------------|-----------|
| task | 200-400 | Rarely |
| task_ctx | 0-2000 | Yes, workspace snapshots can be large |
| deps | 0-8000+ | Unbounded per dependency (4000 chars each, no dep count limit) |
| prior | 0-4000 | 2000 chars per step |
| convo | 0-1500 | 3 entries x 600 chars |
| ambient | 0-400 | Hard-capped at 400 tokens |
| profile | 0-500 | Unbounded |
| board | 0-1000 | Unbounded |
| skills | 0-1200 | 1 verbose ~1000, 3 compact ~450 |
| api | 0-300 | Small |
| rag | 2000-12000 | Budget-capped but minimum 2000 |
| prefs | 0-300 | Unbounded |
| memory | 0-3000 | 10 entries x 300 chars |

**Worst case total**: 15K-30K+ tokens. On 8K context model, this is 2-4x the entire context window.
