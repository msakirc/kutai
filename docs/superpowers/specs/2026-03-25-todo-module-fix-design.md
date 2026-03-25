# Todo Module Fix — Design Spec

**Date:** 2026-03-25
**Status:** Approved

## Overview

Fix the todo reminder module so it actually works end-to-end: reminders fire on schedule, include AI-generated actionable suggestions, show item age, and provide compact inline buttons for marking done or accepting help.

## Requirements

1. **Collect todo items** from user via commands and implicit natural language detection (already works)
2. **Remind every 2 hours** during Turkey daytime: 9,11,13,15,17,19,21 (cron already seeded, scheduler already runs)
3. **AI suggestions before reminder** — for each pending todo, generate a concrete actionable suggestion if KutAI can help. Low bar but must be concrete, not generic. Don't do any actual work — just suggest.
4. **Easy mark-done** — compact inline keyboard buttons in the reminder message
5. **Close button** — dismiss the reminder message

## Reminder Message Format

```
📝 Pending Todos

🟡 #1 — Buy milk (2d ago)
   💡 I can compare prices across online markets

🔴 #3 — Schedule dentist (5d ago)
   💡 I can search for nearby dentists with availability

🟡 #7 — Fix garden fence (1h ago)

[✅ #1] [✅ #3] [✅ #7]
[🤖 #1 Help] [🤖 #3 Help]
[🔙 Close]
```

- Each item: priority icon, ID, title, age suffix
- Suggestion indented below item (only if AI has one)
- Age: compact relative time — `1h ago`, `2d ago`, `3w ago`
- Done buttons: one row, multiple per row, compact labels
- Help buttons: separate row, only for items with suggestions
- Close button: bottom row, deletes the reminder message

## AI Suggestion Generation

Before sending the reminder, create one task per pending todo item: "Suggest action for: {todo title}". These are normal tasks — same queue, same routing, same failure handling as anything else.

Task context should include `local_only: true` and `prefer_quality: true` since todos may contain sensitive personal info.

Once all suggestion tasks complete (or fail), collect the results and attach them to the reminder. Items whose task failed or produced nothing get no suggestion.

## "Help" Button Flow — Two-Step Confirmation

When user taps a Help button:
1. Send a new message with the suggestion prefilled, using Telegram's `ForceReply` to prefill the user's input bar with the suggestion text. This lets the user edit the action before sending.
2. Message text: "🤖 *Help with: {todo title}*\nSuggested action: _{suggestion}_\n\nEdit below and send, or tap Cancel."
3. Include a `[❌ Cancel]` inline button to dismiss.
4. When the user sends their (possibly edited) reply, create the task with their text as the description.

This gives the user control over what KutAI actually does — the suggestion is a starting point, not a commitment.

## Close Button

`callback_data = "todo_close"` — handler calls `query.delete_message()` to remove the reminder.

## Changes Required

### `src/app/reminders.py`
- Add `_generate_suggestions(todos)` — batched LLM call, returns `{todo_id: suggestion_str|None}`
- Add `_format_age(created_at)` — returns compact relative time string
- Update `send_todo_reminder()` to call suggestions, format new message layout, build compact keyboard

### `src/app/telegram_bot.py`
- Add `todo_close` callback handler — deletes message
- Add `todo_help:<id>` callback handler — sends prefilled help message with ForceReply
- Add `todo_help_cancel` callback handler — dismisses help message
- Handle reply to help message — create task with user's edited text

### No changes needed
- DB schema (unchanged)
- Orchestrator scheduler (already works, was blocked by crash)
- Todo CRUD functions
- Classification/detection
- Commands (`/todo`, `/todos`, `/cleartodos`)

## Cleanup

- Delete test data from production `todo_items` table (IDs 1-8 are test artifacts)
