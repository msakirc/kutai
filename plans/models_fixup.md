1.1 — Fix _inference_semaphore crash in local_model_manager.py

Remove self._inference_semaphore.locked() from get_status(). Replace with self._scheduler.get_status()["active"] or a simple bool tracking whether a slot is held.


1.2 — Fix dot normalization destroying family matching

In scan_model_directory() and _load_single_local(): remove .replace(".", "-") from detect_name normalization.
Instead, normalize FAMILY_PATTERNS to also match with dots intact, OR do matching on the raw name before normalization.
Test: "Qwen2.5-Coder-7B-Q4_K_M" must match qwen25_coder.


1.3 — Fix cloud profile substring ordering (gpt-4o-mini gets gpt-4o)

In detect_cloud_model(), sort CLOUD_PROFILES iteration by key length descending (longest match first).
OR switch to exact-prefix matching: split litellm_name, match against profile keys with a specificity score.
Phase 2: Broken Routing Logic


2.1 — Wire classify_task into process_task

In orchestrator.py:process_task, after _inject_chain_context and before agent execution, call classify_task(title, description) if agent_type == "auto" or task has no min_score set.
Store the resulting ModelRequirements fields (min_score, needs_thinking, local_only, etc.) into task context so base.py:_build_model_requirements can read them.


2.2 — Flow task difficulty into model selection

Add difficulty: int = 5 field to ModelRequirements.
In select_model(), when difficulty <= 3, boost cost weight to 50, drop capability weight to 20. When difficulty >= 8, do the inverse.
classify_task already returns min_score — map it directly to difficulty.


2.3 — Fix checkpoint serialization losing ModelRequirements

In _save_checkpoint, serialize ModelRequirements as dict via dataclasses.asdict(reqs).
In _execute_react_loop checkpoint restoration, deserialize back to ModelRequirements(**saved_dict).
Handle legacy string checkpoints with existing from_tier fallback.


2.4 — Fix grading feeding back wrong capability

In grade_response(), accept task_name parameter (the original task, not the grading task).
Pass it through from base.py:_execute_react_loop where the task profile is known.
Use that task's dominant capability for update_quality_from_grading.


2.5 — Remove legacy tier validation in _handle_subtasks

Remove the MODEL_TIERS membership check block. Just pass tier through as-is or set to "auto" unconditionally since ModelRequirements handles everything now.
Phase 3: Model Management Gaps


3.1 — Add missing family patterns and profiles

Add to FAMILY_PATTERNS: GLM4/GLM-4 variants, DeepSeek-V3, Qwen3.5, Gemma3n, Phi-4-reasoning, Command-A, Mistral-Small-3.2.
Add corresponding FAMILY_PROFILES entries with capability scores.
Add to _THINKING_FAMILIES: "glm4" (for flash-thinking variants).
Add to CLOUD_PROFILES: missing providers/models.


3.2 — Make profiles data-driven and overridable

Move FAMILY_PROFILES and CLOUD_PROFILES to a YAML file (model_families.yaml).
Load at registry init, merge with hardcoded defaults.
Users can add new families (e.g., Qwen3.5) by editing YAML without touching code.
Include a last_updated field per family so staleness is visible.


3.3 — Fix Cerebras litellm name matching

cerebras/llama3.3-70b doesn't match profile key "llama-3.3-70b". In detect_cloud_model, strip provider prefix before matching: name_lower = litellm_name.split("/", 1)[-1].lower().


3.4 — Fix VRAM reading total instead of free

In _get_available_vram(), change nvidia-smi query to memory.free.
In gpu_monitor path, use vram_free_mb instead of vram_total_mb.


3.5 — Fix Claude provider detection

In _load_cloud_models, for litellm names without / prefix (e.g., "claude-sonnet-4-20250514"), detect provider from name patterns: if starts with "claude" → "anthropic", if starts with "gpt" or "o1"/"o3"/"o4" → "openai".
Phase 4: Cost & Concurrency


4.1 — Make cost a first-class routing dimension

Add a budget_tier field to ModelRequirements: "free", "cheap", "moderate", "unlimited".
In select_model, for budget_tier="free": hard-filter to is_free models only. For "cheap": hard-filter to cost < \$0.01/call. etc.
Low-difficulty tasks get budget_tier="free" by default.


4.2 — Coordinate concurrent tasks on local model needs

In orchestrator.py:run_loop, when picking multiple tasks, group by required model (same agent_type/capability → same local model likely).
If two tasks need different local models, run one local + one cloud, or serialize.


4.3 — Limit grading to non-trivial tasks

Skip grade_response() when reqs.min_quality <= 3 or reqs.priority <= 2. Saves one LLM call per trivial task.
Phase 5: Dead Code & Validation Cleanup


5.1 — Integrate or remove task_classifier.py

Option A: Wire classify_task_semantic into router.classify_task as the primary classifier (before LLM call). Use embedding result if confidence > 0.7, else fall through to LLM.
Option B: Remove the file entirely and consolidate into router's keyword fallback.


5.2 — Complete _AGENT_TYPE_CATEGORY in models.py

Add: "reviewer": "code", "writer": "writing" (new category), "executor": "execution" (new category), "visual_reviewer": "code", "assistant": "writing", "summarizer": "writing", "error_recovery": "code".
Add validators for new categories.


5.3 — Fix shared validation_retried flag

Split into custom_validation_retried and task_type_validation_retried so each validator gets one retry independently.


5.4 — Tighten research output validator

Remove "from " from the has_source keywords list. Require at least a URL or the word "source"/"reference"/"according to"/"documentation".
Phase 6: Robustness


6.1 — Add model health tracking to select_model

Track last-N call success rate per model in-memory. In select_model, penalize availability score for models with >30% recent failure rate.


6.2 — Add thread safety to registry queries

Wrap best_for_task, get, iteration methods with self._lock (read lock ideally, or snapshot self.models reference).


6.3 — Add stale clarification timeout

In watchdog, find tasks in needs_clarification for >24h. Auto-cancel or auto-answer with "User did not respond, proceeding with best guess."


6.4 — Use MoE dimension sets in scoring

In score_model_for_task, if model_operational.get("model_type") == "moe", apply per-dimension weighting using KNOWLEDGE_DIMENSIONS / REASONING_DIMENSIONS / EXECUTION_DIMENSIONS. (This needs model_type added to operational_dict().)


6.5 — Protect quality EMA from single-sample noise

In update_quality_from_grading, only apply EMA update after total_calls >= 5 for that model+capability. Before that, use simple average.
Phase 7: Future-Proofing (the staleness problem)


7.1 — Auto-profile unknown models from benchmarks

When a GGUF or cloud model has no family match, and enrich_with_benchmarks is enabled, try to fetch benchmark scores and build a profile automatically.
Store auto-generated profiles in a cache file so they persist across restarts.


7.2 — Community profile format

Define a simple JSON/YAML schema for model family profiles.
Document it so users can submit profiles for new models (Qwen3.5, etc.) as a single file drop.
On startup, scan a profiles.d/ directory for user-contributed profiles.
