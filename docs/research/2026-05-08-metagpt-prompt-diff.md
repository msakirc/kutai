# MetaGPT vs KutAI Prompt Diff — 2026-05-08

## Fetch results

All seven URLs resolved successfully (HTTP 200). No 404s, no redirects.

| File fetched | Lines | Notes |
|---|---|---|
| `metagpt/actions/write_prd.py` | 325 | Orchestration logic only. Prompt nodes live in `write_prd_an.py`. Fetched both. |
| `metagpt/actions/design_api.py` | 279 | Same pattern — prompt nodes in `design_api_an.py`. Fetched both. |
| `metagpt/actions/write_code.py` | 228 | Contains `PROMPT_TEMPLATE` directly — 7 numbered coding rules inline. |
| `metagpt/actions/write_test.py` | 70 | Contains `PROMPT_TEMPLATE` directly — 7 numbered QA rules + task framing. |
| `metagpt/actions/run_code.py` | 173 | Contains `PROMPT_TEMPLATE` for code-run summarization + structured verdict section. |
| `metagpt/actions/research.py` | 343 | Rich: `RESEARCH_BASE_SYSTEM`, `RESEARCH_TOPIC_SYSTEM`, `CONDUCT_RESEARCH_PROMPT`, `WEB_BROWSE_AND_SUMMARIZE_PROMPT`, `COLLECT_AND_RANKURLS_PROMPT`. |
| `metagpt/actions/debug_error.py` | 77 | Contains `PROMPT_TEMPLATE` directly — role-from-context + legacy/test/log sections. |

---

## Per-pair analysis

---

### 1. planner ↔ MetaGPT ProductManager (write_prd_an.py)

**MetaGPT prompt excerpt** (ActionNode instructions — these are injected as structured fields):
```python
PRODUCT_GOALS = ActionNode(
    key="Product Goals",
    instruction="Provide up to three clear, orthogonal product goals.",
    example=["Create an engaging user experience", "Improve accessibility, be responsive", "More beautiful UI"],
)
USER_STORIES = ActionNode(
    key="User Stories",
    instruction="Provide up to 3 to 5 scenario-based user stories.",
    example=["As a player, I want to be able to choose difficulty levels", ...]
)
REQUIREMENT_POOL = ActionNode(
    key="Requirement Pool",
    instruction="List down the top-5 requirements with their priority (P0, P1, P2).",
    example=[["P0", "The main code ..."], ["P0", "The game algorithm ..."]],
)
ANYTHING_UNCLEAR = ActionNode(
    key="Anything UNCLEAR",
    instruction="Mention any aspects of the project that are unclear and try to clarify them.",
    example="Currently, all aspects of the project are clear.",
)
```

**KutAI prompt excerpt** (`src/agents/planner.py`):
```python
"You are a senior technical project planner.\n"
"\n"
"Your job is to take a mission and decompose it into concrete, executable subtasks.\n"
"\n"
"## Your Process\n"
"1. FIRST, always inspect the workspace (use `file_tree` and `project_info` tools)\n"
"   to understand what already exists.\n"
"2. Understand the mission thoroughly.\n"
"3. Create a plan with specific subtasks.\n"
```

**3-5 wins to lift:**

1. **Explicit clarification gate before planning.** MetaGPT's `ANYTHING_UNCLEAR` node forces the model to surface ambiguities before committing. KutAI planner has no equivalent — add a mandatory step: "Before producing subtasks, state any unclear requirements and your resolution assumption for each." This prevents silent misinterpretation.

2. **Priority taxonomy (P0/P1/P2) on subtasks.** MetaGPT's `REQUIREMENT_POOL` tags every requirement with P0/P1/P2. KutAI has a `priority` numeric field (0-10) but no label vocabulary, so downstream agents can't quickly distinguish must-have from nice-to-have. Add `"priority_label": "P0 | P1 | P2"` to the subtask schema alongside numeric priority.

3. **User-story framing for the mission.** MetaGPT rewrites requirements as "As a X, I want Y" stories before decomposing. This isn't directly applicable to KutAI's Telegram-driven tasks, but the principle — restate what the *end-user outcome* should be before cutting tasks — is missing. Add: "Before listing subtasks, write one sentence stating the intended user outcome of this mission."

4. **Competitive/context analysis step.** MetaGPT's `COMPETITIVE_ANALYSIS` node asks for 5-7 comparable existing solutions. For software planning, the equivalent is: "Check if the workspace already has code/files that partially address this mission before decomposing." KutAI does this (step 1 — file_tree), but the instruction doesn't say *what to look for* (partial implementations, relevant libraries, existing configs). Make it explicit.

5. **Role primer with domain expertise claim.** MetaGPT's ProductManager role is declared with full experience context in the broader role system prompt (not in these action files, but the pattern is consistent across MetaGPT). KutAI's opener is "You are a senior technical project planner." — functional, but no domain-depth signal. Strengthen to: "You are a senior technical project planner with 10+ years decomposing software missions into executable subtasks. You have shipped multi-team projects and know how dependencies, parallelism, and scope-creep manifest in task graphs."

---

### 2. architect ↔ MetaGPT Architect (design_api_an.py)

**MetaGPT prompt excerpt** (ActionNode instructions):
```python
IMPLEMENTATION_APPROACH = ActionNode(
    key="Implementation approach",
    instruction="Analyze the difficult points of the requirements, select the appropriate open-source framework.",
    example="We will ...",
)
DATA_STRUCTURES_AND_INTERFACES = ActionNode(
    key="Data structures and interfaces",
    instruction="Use mermaid classDiagram code syntax, including classes, method(__init__ etc.) and functions "
                "with type annotations, CLEARLY MARK the RELATIONSHIPS between classes, and comply with PEP8 "
                "standards. The data structures SHOULD BE VERY DETAILED and the API should be comprehensive "
                "with a complete design.",
    example=MMC1,  # full mermaid class diagram example
)
PROGRAM_CALL_FLOW = ActionNode(
    key="Program call flow",
    instruction="Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED "
                "ABOVE accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.",
    example=MMC2,  # full mermaid sequence diagram example
)
ANYTHING_UNCLEAR = ActionNode(
    key="Anything UNCLEAR",
    instruction="Mention unclear project aspects, then try to clarify it.",
)
```

**KutAI prompt excerpt** (`src/agents/architect.py`):
```python
"You are a Principal Software Architect.\n"
"\n"
"Your job is to analyze the user's task and the current project workspace, "
"then create a structured architectural plan for the implementation.\n"
"\n"
"## Your Workflow\n"
"1. **Explore** — Use `file_tree` and `project_info` to understand the project structure.\n"
"2. **Read** — Use `read_file` on existing files if you need to understand current interfaces.\n"
"3. **Research** — Use `web_search` if you need to check documentation for specific libraries.\n"
"4. **Design** — Create a clear, file-by-file implementation plan.\n"
"5. **Document** — Write your plan to `ARCHITECTURE.md` using `write_file`.\n"
"6. **Finish** — Return `final_answer` summarizing the architecture.\n"
```

**3-5 wins to lift:**

1. **Explicit "difficult points first" framing.** MetaGPT's `IMPLEMENTATION_APPROACH` demands that the architect call out *difficult points* before selecting frameworks. KutAI goes straight to file listing. Add a `## Hard Problems` section to `ARCHITECTURE.md`: "Before listing files, state the 1-3 technically difficult aspects of this task and your chosen approach for each." This catches risky assumptions early.

2. **Interface contracts with type annotations required.** MetaGPT explicitly requires "type annotations, CLEARLY MARK the RELATIONSHIPS between classes, comply with PEP8 standards." KutAI's format asks for `def function_name(args) -> ReturnType` but doesn't demand relationship markers or type strictness. Strengthen: "Every interface definition MUST include full type annotations. Mark cross-file dependencies explicitly with `# depends on: path/to/file.py`."

3. **Program call flow / sequence diagram.** MetaGPT always produces a `sequenceDiagram` for the key initialization and CRUD paths. KutAI produces no equivalent. Even a text-based call flow ("Request arrives → validator.check() → db.save() → notifier.send()") would catch integration mismatches before implementers start. Add an optional `## 5. Call Flow` section to the ARCHITECTURE.md template.

4. **Clarification gate on ambiguity.** MetaGPT includes `ANYTHING_UNCLEAR` as a mandatory output section. KutAI architect has no analogous field — unclear specs get silently absorbed into design choices that later agents inherit. Add to the `final_answer` schema: `"open_questions": ["..."]` so the orchestrator can surface them.

5. **Role primer with breadth signal.** KutAI: "You are a Principal Software Architect." MetaGPT's broader role context establishes the model as someone who has designed APIs, data stores, and call flows for production systems. Add: "You have designed production-grade APIs, data models, and service architectures across multiple languages and frameworks."

---

### 3. coder ↔ MetaGPT Engineer (write_code.py)

**MetaGPT prompt excerpt** (`PROMPT_TEMPLATE` in `write_code.py`):
```
Role: You are a professional engineer; the main goal is to write google-style, elegant, modular,
easy to read and maintain code

1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete,
   reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE
   AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN.
   Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.
```

**KutAI prompt excerpt** (`src/agents/coder.py`):
```python
"You are an expert software engineer. You BUILD working software.\n"
"\n"
"## Critical Rules\n"
"- ALWAYS check existing files before creating new ones.\n"
"- ALWAYS run code after writing it. Never assume it works.\n"
"- When you see an error, FIX it — don't just report it.\n"
"- Write complete, runnable code — not snippets or pseudocode.\n"
"- Include error handling in your code.\n"
...
"## IMPORTANT — Do NOT give a final_answer until you have:\n"
"1. Written all files\n"
"2. Installed any dependencies\n"
"3. Run the code successfully\n"
"4. Committed with `git_commit`\n"
```

**3-5 wins to lift:**

1. **Code quality standard named explicitly.** MetaGPT opens with "write google-style, elegant, modular, easy to read and maintain code." KutAI says "expert software engineer" but gives no style target. Add to the role opener: "You write clean, modular, well-named code. Functions do one thing. Files have one responsibility. Variables have explicit types where the language supports it."

2. **ALWAYS SET A DEFAULT VALUE / STRONG TYPE.** MetaGPT's rule 3 is missing from KutAI. KutAI mentions "strong type" nowhere in the coder prompt. Add to Critical Rules: "For any configurable value (timeouts, limits, paths), set an explicit default. Use type annotations in Python. Avoid `Any` unless necessary."

3. **Don't leave TODO rule.** MetaGPT's rule 7 ("Write out EVERY CODE DETAIL, DON'T LEAVE TODO") is explicit. KutAI says "Write complete, runnable code — not snippets or pseudocode" which partially covers it but doesn't call out TODOs by name. Add: "Do NOT leave `TODO`, `pass`, or placeholder comments in your code. If you can't implement something, say so in `final_answer`."

4. **Import before use — explicit check.** MetaGPT rule 6: "Before using an external variable/module, make sure you import it first." KutAI has no equivalent. Small but catches a frequent LLM failure mode. Add to Critical Rules: "Double-check that every import is at the top of the file before returning final_answer."

5. **Pre-submission self-check checklist.** MetaGPT's rule 5 ("CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION") is an explicit cognitive step. KutAI's "Do NOT give a final_answer until you have:" list is close but tool-execution focused. Add a mental-check line: "Before final_answer, re-read the task and verify every requested function/class/endpoint exists in your code."

---

### 4. implementer ↔ MetaGPT Engineer (write_code.py)

**MetaGPT prompt excerpt:** Same `PROMPT_TEMPLATE` as coder pair above, particularly:
```
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN.
   Do not use public member functions that do not exist in your design.
```

**KutAI prompt excerpt** (`src/agents/implementer.py`):
```python
"You are an expert Software Engineer and Implementer.\n"
"\n"
"Your job is to implement EXACTLY ONE FILE according to the "
"architectural plan provided in the task description.\n"
"\n"
"## Critical Rules\n"
"- Implement ONLY your assigned file. Do not wander off and modify other things.\n"
"- Ensure your code perfectly matches the interfaces designing in the `ARCHITECTURE.md`.\n"
"- Use absolute imports where appropriate based on the workspace root.\n"
"- Write robust code with error handling.\n"
"- You do not need to write tests. Another agent will handle testing.\n"
"- You do not need to commit. The orchestrator will handle commits.\n"
```

**3-5 wins to lift:**

1. **Design-lock enforcement with explicit consequence.** MetaGPT: "DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design." KutAI: "matches the interfaces designing in `ARCHITECTURE.md`." The KutAI wording is softer — it says "matches" not "MUST NOT deviate." Strengthen: "You MUST NOT add, remove, or rename any class, method, or function that appears in `ARCHITECTURE.md`. Deviations break other implementers. If the design has an error, flag it in `final_answer` rather than silently patching it."

2. **Complete-code mandate.** MetaGPT: "COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets." KutAI: "Do NOT leave placeholders like `pass` or `# TODO: implement`." The KutAI wording is good but frames it as a prohibition. Add the positive framing: "Your file is one piece of a larger system — write it as if it will be imported by five other files immediately."

3. **Circular import guard.** MetaGPT: "AVOID circular import." KutAI has no mention. Given KutAI's own codebase explicitly uses lazy imports to avoid this (per CLAUDE.md), adding this is directly relevant. Add to Critical Rules: "Avoid circular imports. If you need something from a file that might import yours, use a local import inside the function."

4. **Default value / strong type rule.** Same as coder pair — MetaGPT's rule 3 is absent in implementer too. Add: "Set explicit default values for all configuration or optional parameters. Use type annotations."

5. **"One file only" pre-submission audit.** KutAI has "Implement ONLY your assigned file" but doesn't trigger a final check. Add to workflow step 7: "Before `final_answer`, use `git_diff` or `file_tree` to confirm you only modified your assigned file."

---

### 5. fixer ↔ MetaGPT DebugError (debug_error.py)

**MetaGPT prompt excerpt** (`PROMPT_TEMPLATE` in `debug_error.py`):
```
1. Role: You are a Development Engineer or QA engineer;
2. Task: You received this message from another Development Engineer or QA engineer who ran or
   tested your code. Based on the message, first, figure out your own role, i.e. Engineer or
   QaEngineer, then rewrite the development code or the test code based on your role, the error,
   and the summary, such that all bugs are fixed and the code performs well.

# Legacy Code
```python
{code}
```
---
# Unit Test Code
```python
{test_code}
```
---
# Console logs
```text
{logs}
```
---
Now you should start rewriting the code:
## file name of the code to rewrite: Write code with triple quote.
   Do your best to implement THIS IN ONLY ONE FILE.
```

**KutAI prompt excerpt** (`src/agents/fixer.py`):
```python
"You are an expert Software Debugger and Code Fixer.\n"
"\n"
"Your job is to read review feedback and test failures, then "
"directly edit the source code to fix all identified issues.\n"
"\n"
"## Critical Rules\n"
"- Address EVERY point of feedback. Do not ignore any warnings or errors.\n"
"- Use `edit_file` where possible so you don't inadvertently delete other logic.\n"
"- ALWAYS run tests via `shell` after making modifications, if a test suite exists.\n"
```

**3-5 wins to lift:**

1. **Role disambiguation: dev-fix vs test-fix.** MetaGPT explicitly tells the fixer to determine whether it's fixing *implementation code* or *test code*. KutAI's fixer receives review feedback which may reference either, but the prompt doesn't mention this distinction. Add: "First determine: is the root cause in the implementation code or the test code? Fix only the correct file — do not reflexively blame implementation when the test may be wrong."

2. **Three-section context requirement.** MetaGPT structures the prompt as three named sections: legacy code, unit test code, and console logs. KutAI's fixer gets feedback as free-form text. The fixer prompt should instruct: "Your task context MUST include: (a) the failing code, (b) the test code if applicable, (c) the exact error logs. If any of these are missing, state what you're assuming."

3. **"Perform well" success criterion.** MetaGPT ends with: "all bugs are fixed and the code performs well." KutAI ends the workflow with "Return `final_answer` summarizing what you fixed." Add an explicit success gate: "Do not return `final_answer` until `pytest` (or equivalent) shows no new failures introduced by your fix."

4. **Scope lock — one file only.** MetaGPT: "Do your best to implement THIS IN ONLY ONE FILE." KutAI's fixer can touch multiple files (it has `write_file` access). This is intentionally broader, but the prompt should warn: "Prefer fixing the minimum number of files. If you need to touch more than two files, explain why in `final_answer` rather than silently expanding scope."

5. **Root-cause statement before fixing.** MetaGPT says "first, figure out your own role ... based on the error." KutAI jumps straight to "Read Code → Apply Fixes." Add step 1.5: "**State root cause** — Before touching any file, write one sentence in your reasoning stating the exact root cause. This prevents fixing symptoms instead of causes."

---

### 6. test_generator ↔ MetaGPT QA Engineer (write_test.py)

**MetaGPT prompt excerpt** (`PROMPT_TEMPLATE` in `write_test.py`):
```
1. Role: You are a QA engineer; the main goal is to design, develop, and execute PEP8 compliant,
   well-structured, maintainable test cases and scripts for Python 3.9. Your focus should be on
   ensuring the product quality of the entire project through systematic testing.
2. Requirement: Based on the context, develop a comprehensive test suite that adequately covers
   all relevant aspects of the code file under review. Your test suite will be part of the overall
   project QA, so please develop complete, robust, and reusable test cases.
3. Attention1: Use '##' to split sections...
4. Attention2: ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
5. Attention3: YOU MUST FOLLOW "Data structures and interfaces". DO NOT CHANGE ANY DESIGN.
   Make sure your tests respect the existing design and ensure its validity.
6. Think before writing: What should be tested and validated in this document?
   What edge cases could exist? What might fail?
7. CAREFULLY CHECK THAT YOU DON'T MISS ANY NECESSARY TEST CASES/SCRIPTS IN THIS FILE.
```

**KutAI prompt excerpt** (`src/agents/test_generator.py`):
```python
"You are an expert Software Developer in Test (SDET).\n"
"\n"
"## Testing Guidelines\n"
"- Use `pytest` for all Python testing.\n"
"- Use `unittest.mock` (Standard Library) or `pytest-mock` for mocking external calls or DBs.\n"
"- Test edge cases, not just the happy path.\n"
"- Make sure tests are isolated and don't depend on each other.\n"
"\n"
"## Critical Rules\n"
"- You MUST use the `shell` tool to execute `pytest` at least once.\n"
"- It is OK if tests ultimately fail. If you cannot fix the issue within your iterations,\n"
"  report the failing tests so the review process catches them.\n"
```

**3-5 wins to lift:**

1. **Pre-write thinking prompt.** MetaGPT rule 6: "Think before writing: What should be tested and validated in this document? What edge cases could exist? What might fail?" KutAI has "Test edge cases, not just the happy path" but no explicit cognitive pause before writing. Add to the workflow between steps 1 and 2: "**Plan** — Before writing any test code, list in your reasoning: the 3-5 most important behaviors to test, the likely edge cases, and the failure modes. Only then start writing."

2. **"Comprehensive suite for the entire file" scope framing.** MetaGPT: "develop a comprehensive test suite that adequately covers all relevant aspects of the code file under review." KutAI: "write comprehensive test files ... Focus on testing the core logic and critical paths." KutAI's "focus on core logic" implicitly permits leaving non-core paths untested. Strengthen: "Your test suite should cover: (a) every public function's happy path, (b) error/exception paths, (c) boundary values. Do not stop after testing only the most obvious path."

3. **CAREFULLY CHECK — nothing missed rule.** MetaGPT rule 7: "CAREFULLY CHECK THAT YOU DON'T MISS ANY NECESSARY TEST CASES." This explicit audit is absent in KutAI. Add as the last workflow step before `final_answer`: "Re-read the source file's public interface. Verify every public function or class has at least one test. If something is untested, add it or explain why it's excluded."

4. **Role primer with quality mandate.** MetaGPT: "ensuring the product quality of the entire project through systematic testing." KutAI: "expert SDET" with no quality-ownership framing. Strengthen: "You are the last line of defence before code ships. If your tests don't catch a bug, it reaches production. Write tests as if you own the quality outcome."

5. **Design-conformance note.** MetaGPT rule 5: "YOU MUST FOLLOW 'Data structures and interfaces'. DO NOT CHANGE ANY DESIGN. Make sure your tests respect the existing design." KutAI has no equivalent — the test generator could import things that don't exist yet. Add: "Import only modules and functions that exist in the workspace. Check with `file_tree` or `read_file` before writing import statements."

---

### 7. reviewer ↔ MetaGPT RunCode review section (run_code.py)

**MetaGPT prompt excerpt** (`PROMPT_TEMPLATE` in `run_code.py`):
```
Role: You are a senior development and qa engineer, your role is summarize the code running result.
If the running result does not include an error, you should explicitly approve the result.
On the other hand, if the running result indicates some error, you should point out which part,
the development code or the test code, produces the error, and give specific instructions on
fixing the errors.

## instruction:
Please summarize the cause of the errors and give correction instruction
## File To Rewrite:
Determine the ONE file to rewrite in order to fix the error, for example, xyz.py, or test_xyz.py
## Status:
Determine if all of the code works fine, if so write PASS, else FAIL,
WRITE ONLY ONE WORD, PASS OR FAIL, IN THIS SECTION
## Send To:
Please write NoOne if there are no errors, Engineer if the errors are due to problematic
development codes, else QaEngineer,
WRITE ONLY ONE WORD, NoOne OR Engineer OR QaEngineer, IN THIS SECTION.
```

**KutAI prompt excerpt** (`src/agents/reviewer.py`):
```python
"You are a senior code reviewer and quality checker.\n"
"\n"
"## What to Check\n"
"- **Bugs** — logic errors, off-by-one, null/undefined handling\n"
"- **Security** — injection, hardcoded secrets, unsafe input handling\n"
"- **Error handling** — missing try/except, unhandled edge cases\n"
"- **Completeness** — missing features, TODO/placeholder code\n"
"- **Code style** — readability, naming, structure\n"
"- **Tests** — do they exist? do they pass? adequate coverage?\n"
"\n"
"## Output Format (REQUIRED)\n"
"Your final_answer MUST be a JSON string with this structure:\n"
'  "verdict": "pass" | "fail" | "needs_minor_fixes",\n'
'  "issues": [{"severity": "critical|high|medium|low", "file": ..., "line": ..., ...}]\n'
```

**3-5 wins to lift:**

1. **Explicit pass/fail binary + routing.** MetaGPT's review always produces a single-word `Status: PASS | FAIL` and a `Send To: NoOne | Engineer | QaEngineer` routing decision. KutAI has `"verdict": "pass" | "fail" | "needs_minor_fixes"` which is richer, but lacks explicit routing. The fixer needs to know *who* should fix it. Add `"route_to": "fixer" | "test_generator" | "none"` to the verdict schema, with instructions: "If the bug is in implementation code → `fixer`. If tests are wrong → `test_generator`. If nothing needs fixing → `none`."

2. **"Approve explicitly if no errors" rule.** MetaGPT: "If the running result does not include an error, you should **explicitly approve** the result." KutAI says "If no issues, set verdict='pass' and issues=[]" but doesn't prompt for an affirmative approval statement. Add: "When verdict is 'pass', your `summary` MUST include a positive confirmation sentence (e.g., 'All critical paths verified. Code is production-ready for its scope.')."

3. **File-To-Rewrite single-target discipline.** MetaGPT: "Determine the ONE file to rewrite in order to fix the error." KutAI issues can reference multiple files. When a downstream fixer receives issues across 5 files, it's likely to lose focus. Add to reviewer instructions: "For each critical/high issue, state the single most important file to fix first. If issues span many files, rank by blast radius."

4. **Correction instruction quality gate.** MetaGPT: "give specific instructions on fixing the errors." KutAI: `"suggested_fix": "How to fix it"` is the field but it's just a description slot with no quality gate. Add to the format instructions: "Each `suggested_fix` MUST be actionable — it should name the specific function, line range, or pattern to change, not just describe the problem in different words."

5. **Role primer spans both dev and QA.** MetaGPT: "You are a senior development and qa engineer." KutAI: "senior code reviewer and quality checker." Both are fine, but MetaGPT's dual-hat framing ("dev AND qa") signals that the reviewer should think like someone who wrote the code AND someone testing it. Add: "Review from both perspectives: as the engineer who wrote it (what did I intend?) and as the QA engineer who will test it (what could go wrong?)."

---

### 8. researcher ↔ MetaGPT Research (research.py)

**MetaGPT prompt excerpts** (`research.py`):
```python
RESEARCH_BASE_SYSTEM = """You are an AI critical thinker research assistant. Your sole purpose is
to write well written, critically acclaimed, objective and structured reports on the given text."""

RESEARCH_TOPIC_SYSTEM = "You are an AI researcher assistant, and your research topic is:\n#TOPIC#\n{topic}"

CONDUCT_RESEARCH_PROMPT = """### Reference Information
{content}

### Requirements
Please provide a detailed research report in response to the following topic: "{topic}", using
the information provided above. The report must meet the following requirements:
- Focus on directly addressing the chosen topic.
- Ensure a well-structured and in-depth presentation, incorporating relevant facts and figures.
- Present data and findings in an intuitive manner, utilizing feature comparative tables.
- The report should have a minimum word count of 2,000 and be formatted with Markdown syntax
  following APA style guidelines.
- Include all source URLs in APA format at the end of the report.
"""

WEB_BROWSE_AND_SUMMARIZE_PROMPT = """### Requirements
1. Utilize the text in the "Reference Information" section to respond to the question "{query}".
2. If the question cannot be directly answered, but the text is related to the research topic,
   provide a comprehensive summary of the text.
3. If the text is entirely unrelated to the research topic, reply with "Not relevant."
4. Include all relevant factual information, numbers, statistics, etc., if available.
"""
```

**KutAI prompt excerpt** (`src/agents/researcher.py`):
```python
"You are a research specialist. You find accurate, useful "
"information and present it clearly.\n"
"\n"
"## IMPORTANT: Be efficient\n"
"- ONE search is usually enough. Do NOT search multiple times "
"unless the first result genuinely lacks the answer.\n"
"- After getting search results, respond with `final_answer` "
"immediately. Do not search again with rephrased queries.\n"
"\n"
"## Rules\n"
"- Cite sources when possible (include URLs).\n"
"- If you can't find reliable information, say so honestly.\n"
"- Keep summaries focused and actionable — no filler.\n"
```

**3-5 wins to lift:**

1. **"Critical thinker" framing vs "specialist" framing.** MetaGPT: "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports." KutAI: "You are a research specialist. You find accurate, useful information and present it clearly." MetaGPT's framing sets a much higher bar — *critically acclaimed, objective, structured*. Add to the role opener: "You think critically about sources — you flag conflicting claims, note when evidence is thin, and distinguish opinion from fact."

2. **"Not relevant" explicit rejection path.** MetaGPT's `WEB_BROWSE_AND_SUMMARIZE_PROMPT` includes: "If the text is entirely unrelated to the research topic, reply with 'Not relevant.'" This teaches the model to discard bad sources explicitly. KutAI's researcher has no equivalent — it might synthesize irrelevant content. Add to rules: "If a source is off-topic, state that explicitly and do not include it in your findings. Better to cite fewer good sources than many weak ones."

3. **Factual density requirement.** MetaGPT: "Include all relevant factual information, numbers, statistics, etc., if available." KutAI: "Keep summaries focused and actionable." KutAI's rule pushes toward brevity but could sacrifice data richness. Reconcile: "Include specific facts, numbers, and dates when available. Summaries should be concise but data-dense — a concrete number is worth ten adjectives."

4. **Structured report format with APA citations.** MetaGPT's `CONDUCT_RESEARCH_PROMPT` mandates a structured Markdown report with APA citations, minimum word count, and comparative tables where relevant. KutAI has a `final_answer` format with `## Research: [Topic]\n### Key Findings` but no citation format standard or depth requirement. Add to the format: "Structure findings as: (1) Summary (2) Key facts with sources (3) Caveats / conflicting info (4) Sources list. For each source cited, include the URL."

5. **Topic-scoping system prompt injection.** MetaGPT uses `RESEARCH_TOPIC_SYSTEM = "You are an AI researcher assistant, and your research topic is:\n#TOPIC#\n{topic}"` as a *system* message, not a user message. This keeps the topic anchored across all tool calls. KutAI passes the topic only in the task description which can get buried after several tool calls. Consider: "At the start of your reasoning, restate the research question in one sentence. Check every result against this question before including it."

---

## Summary table

| Pair | Top win | Second win |
|---|---|---|
| planner ↔ ProductManager | Clarification gate before generating subtasks | P0/P1/P2 priority labels on subtasks |
| architect ↔ Architect | "Difficult points first" analysis before file listing | Interface contracts require full type annotations |
| coder ↔ Engineer | Explicit code quality standard (google-style / modular) | "Don't leave TODO" named prohibition |
| implementer ↔ Engineer | Design-lock with explicit consequence ("flag, don't patch") | Circular import guard |
| fixer ↔ DebugError | Role disambiguation: fix implementation vs fix test | State root cause before touching any file |
| test_generator ↔ QA | Pre-write thinking pause (list what to test before coding) | "Comprehensive suite" scope = every public function |
| reviewer ↔ RunCode | Add `route_to` field to route issues to fixer vs test_generator | Explicit approval statement when verdict is pass |
| researcher ↔ Research | "Critical thinker" framing — flag conflicting claims, thin evidence | "Not relevant" explicit rejection path for off-topic sources |

---

## Notes on MetaGPT content quality

- **`write_code.py` and `write_test.py`**: Substantial — concrete numbered rules that transfer directly.
- **`debug_error.py`**: Thin on content (77 lines) but the three-section structure (code / test / logs) and role-disambiguation pattern are genuinely useful.
- **`run_code.py`**: The reviewer pair is the weakest MetaGPT side. `run_code.py` is a code-*runner* that does a lightweight review as a side effect. KutAI's reviewer is richer in scope. The main win is the PASS/FAIL/routing discipline, not the review criteria.
- **`write_prd_an.py` and `design_api_an.py`**: These use ActionNode structured output (field-by-field JSON schemas). There is no equivalent in KutAI — prompts are free-form system strings. The wins extracted above are the *reasoning patterns* MetaGPT enforces via schema, translated into explicit instruction language for KutAI.
- **`research.py`**: Richest MetaGPT file. The `RESEARCH_BASE_SYSTEM` "critical thinker" framing and the multi-prompt architecture (collect links → browse → summarize → conduct report) shows more sophistication than KutAI's single-search researcher. Several wins here are high-leverage.
