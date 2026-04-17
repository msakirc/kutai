# Degenerate Workspace File Handling — Design Decision Needed

## Context

During i2p v3 workflow execution, local LLMs (9B models) sometimes produce degenerate output — files where the same content sections are repeated 3-5x with slightly different headers ("## Component Usage", "## Component Usage Summary", "## Component Example Usage"). A 3K file of real content becomes 20-27K of garbage.

These files persist on disk in `workspace/mission_N/` and poison downstream tasks when injected into context.

## Current State (as of 2026-04-11)

### Detection
- `_detect_repetition_ratio(text)` in `src/workflows/engine/hooks.py` returns 0.0-1.0
- Normalizes headers (strips "Summary", "Examples", "Notes" suffixes), counts how many sections share a normalized header
- Threshold: >40% = degenerate

### Where detection runs
1. **`_prev_output` injection** (hooks.py `enrich_task_description`): If previous output is >40% repetitive, discard it entirely — don't feed garbage to next attempt
2. **Workspace file injection** (hooks.py `enrich_task_description`): Skip degenerate workspace files when building prompt context
3. **Post-hook workspace recovery** (hooks.py `post_execute_workflow_step`): When reading workspace files to combine with result — currently **deletes** degenerate files
4. **Post-hook final gate** (hooks.py `post_execute_workflow_step`): Rejects degenerate output before storing to blackboard, triggers retry

### The Question

**Point 3 above: should degenerate workspace files be deleted, skipped, or truncated?**

## Options

### A: Delete (current implementation)
```python
if rep_ratio > 0.4:
    os.remove(fpath)  # delete garbage file
```
- **Pro**: Clean slate, next attempt can't be influenced by garbage
- **Con**: Throws away 3-4K of real content buried in 20K of repetition. The model spent iterations and inference time producing that core content.

### B: Skip (don't use, don't delete)
```python
if rep_ratio > 0.4:
    break  # skip this file, don't include in output_value
```
- **Pro**: No data loss, file stays as a record. Next attempt overwrites if successful.
- **Con**: Failed attempts accumulate garbage files. If the model reads the workspace (via `file_tree` + `read_file`), it may find and re-read the garbage anyway.
- **Note**: The `_prev_output` injection and workspace file injection in `enrich_task_description` already skip degenerate files, so the prompt won't include them. But the agent could manually `read_file` the garbage.

### C: Truncate to unique sections
```python
if rep_ratio > 0.4:
    clean = deduplicate_sections(file_content)
    write(fpath, clean)  # overwrite with clean version
```
- **Pro**: Preserves the 3-4K of real content. Next attempt sees clean partial work, can complete it.
- **Con**: Dedup is fragile — sections with legitimately different content but similar headers could be falsely removed. We discussed this risk earlier and agreed the dedup is better as a detection signal than an automatic fix.

### D: Rename/quarantine
```python
if rep_ratio > 0.4:
    os.rename(fpath, fpath + '.degenerate')
```
- **Pro**: Preserved for debugging, won't be found by the post-hook (wrong extension), won't confuse `file_tree`
- **Con**: Accumulates quarantine files

## Key Files
- `src/workflows/engine/hooks.py` — `_detect_repetition_ratio()`, `_unwrap_envelope()`, `post_execute_workflow_step()`, `enrich_task_description()`
- `src/agents/base.py` — checkpoint handling, format correction, tool call interception

## Recommendation Request
Evaluate these options considering:
1. How often does the real content in a degenerate file actually help the next attempt vs. confuse it?
2. Is there a risk of the agent finding and re-reading a skipped file via `read_file`?
3. Is the false-positive risk of truncation acceptable for workspace files (not stored artifacts)?
4. What's the operational cost of accumulating quarantine files?

Pick one and implement it, or propose a hybrid.
