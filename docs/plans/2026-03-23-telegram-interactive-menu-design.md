# Telegram Interactive Menu + Natural Input Understanding

**Date**: 2026-03-23
**Status**: Approved

## Problem

39 slash commands are hard to discover and remember. Users must type exact syntax. Free-text input works via LLM classifier but users don't know what's possible.

## Design

### 1. Interactive Menu (Edit-in-Place)

`/start` shows 6 category buttons as an InlineKeyboard. Tapping a category **edits the same message** to show that group's commands + ⬅️ Back button. No new messages, no chat clutter.

**Categories:**

| Category | Button Label | Commands |
|----------|-------------|----------|
| Goals & Tasks | 🎯 Goals & Tasks | goal, goalforce, task, goals, queue, cancel, priority, pause, resume, graph |
| Status & Monitoring | 📊 Monitoring | status, digest, progress, metrics, audit, replay, cost, modelstats, budget |
| Projects & Workflows | 🏗️ Projects | project, projects, workspace, product, preview, wfstatus |
| Memory & Knowledge | 🧠 Knowledge | remember, recall, ingest, feedback |
| System & Config | ⚙️ System | autonomy, credential, tune, load, improve, dlq |
| Danger Zone | 🔴 Danger | debug, reset, resetall |

### 2. Callback Data Convention

- `menu_cat:<key>` — show category commands
- `menu_back` — return to category list
- `menu_cmd:<command>` — execute no-arg command immediately
- `menu_ask:<command>` — enter conversation flow for arg-needing command

### 3. Conversation Flow for Argument Commands

When a user taps a button for a command that needs arguments:

1. Bot sends a contextual prompt ("What's your goal?", "Which task ID?", etc.)
2. Sets `_pending_action[chat_id] = {"command": "goal"}`
3. Next user message → routed to the `cmd_*` handler with text as argument
4. State cleared after execution
5. Takes priority over LLM classification in `handle_message()`

### 4. Argument Prompts

| Command | Prompt |
|---------|--------|
| goal | "What's your goal? Describe it:" |
| goalforce | "Describe the goal (skipping refinement):" |
| task | "What task should I do?" |
| cancel | "Which task/goal ID to cancel?" |
| priority | "Enter: <task_id> <priority 1-10>" |
| graph | "Which goal ID to graph?" |
| progress | "Which goal ID? (or leave empty for all)" |
| audit | "Which task ID to audit?" |
| replay | "Which task ID to replay?" |
| cost | "Which goal ID for cost breakdown?" |
| budget | "Enter daily budget limit (or empty to view):" |
| ingest | "Send a URL or file path to ingest:" |
| product | "Describe your product idea:" |
| preview | "Describe the idea to preview:" |
| wfstatus | "Which goal ID for workflow status?" |
| resume | "Which goal ID to resume?" |
| pause | "Which goal ID to pause?" |
| feedback | "Enter: <task_id> <good|bad|partial> [reason]" |
| remember | "What should I remember?" |
| recall | "What do you want to recall?" |
| credential | "Which credential to manage? (key=value or key to view)" |
| tune | "What setting to tune?" |
| load | "Set load mode (minimal/normal/auto):" |
| autonomy | "Set level: low, medium, high, or paranoid" |

### 5. No-Arg Commands (Execute Immediately)

goals, queue, status, digest, metrics, modelstats, project, projects, workspace, debug, improve, dlq, reset, resetall
