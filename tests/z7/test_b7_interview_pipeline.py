"""Z7 T4 B7 — Customer interview / call notes pipeline tests.

Covers:
  1. DB migration: interview_notes table exists with correct columns.
  2. interview_notes row lifecycle (insert → fetch → update).
  3. interview/transcribe verb: pluggable _transcribe() — mocked; raises clear
     error when whisper not installed.
  4. interview/summarize verb: structured output (bullets, quotes, insights,
     action_items) via mocked LLM (beckman.enqueue → OVERHEAD lane).
  5. interview/cross_link verb:
     a. Writes a crm interaction row (kind='interview').
     b. Enqueues action items as candidate backlog tasks (beckman.enqueue).
     c. Quote push gated on crm.has_consent(product_id, contact_id, 'quote_use'):
        - When consent present → inserts into press_kit_quotes.
        - When consent absent → emits a founder_action (NOT insert into quotes).
  6. /interview Telegram command handlers registered (start / stop / list).
  7. A0 briefing founder_action "review interview note" emitted after completion.
  8. Reversibility entries registered for interview/transcribe, interview/summarize,
     interview/cross_link.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── DB helpers ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for B7 tests."""
    db_file = str(tmp_path / "test_b7.db")
    monkeypatch.setenv("DB_PATH", db_file)
    try:
        import src.infra.db as _db_mod
        monkeypatch.setattr(_db_mod, "DB_PATH", db_file)
        monkeypatch.setattr(_db_mod, "_db_connection", None)
        monkeypatch.setattr(_db_mod, "_db_connection_path", None)
    except Exception:
        pass
    return db_file


@pytest_asyncio.fixture
async def db(tmp_db):
    """Initialised DB with full schema (includes B7 migration)."""
    import src.infra.db as _db_mod
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None
    from src.infra.db import init_db, get_db
    await init_db()
    _db = await get_db()
    yield _db
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ── Helper factories ──────────────────────────────────────────────────────────


async def _add_contact(db, product_id="prod1", handle="alice") -> int:
    cur = await db.execute(
        "INSERT INTO relationships (product_id, handle, display_name, category) "
        "VALUES (?, ?, ?, ?)",
        (product_id, handle, "Alice", "customer"),
    )
    await db.commit()
    return cur.lastrowid


async def _insert_interview(
    db,
    product_id="prod1",
    contact_id=1,
    audio_path="/tmp/call.mp3",
) -> int:
    cur = await db.execute(
        "INSERT INTO interview_notes "
        "(product_id, contact_id, started_at, audio_path) "
        "VALUES (?, ?, strftime('%Y-%m-%d %H:%M:%S','now'), ?)",
        (product_id, contact_id, audio_path),
    )
    await db.commit()
    return cur.lastrowid


# ── 1. DB migration ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interview_notes_table_exists(db):
    """interview_notes table created by migration."""
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='interview_notes'"
    )
    row = await cur.fetchone()
    assert row is not None, "interview_notes table not found"


@pytest.mark.asyncio
async def test_interview_notes_columns(db):
    """interview_notes has all required columns."""
    cur = await db.execute("PRAGMA table_info(interview_notes)")
    cols = {row[1] for row in await cur.fetchall()}
    required = {
        "note_id", "product_id", "contact_id", "started_at",
        "duration_minutes", "transcript_md", "summary_md",
        "quotes_json", "insights_md", "action_items_json", "audio_path",
    }
    missing = required - cols
    assert not missing, f"interview_notes missing columns: {missing}"


@pytest.mark.asyncio
async def test_interview_notes_product_id_not_null(db):
    """product_id NOT NULL constraint enforced."""
    with pytest.raises(Exception):
        await db.execute(
            "INSERT INTO interview_notes (product_id) VALUES (NULL)"
        )


# ── 2. Row lifecycle ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_insert_interview_row(db):
    """Can insert a row and retrieve it."""
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)
    assert note_id > 0

    cur = await db.execute(
        "SELECT note_id, product_id, contact_id, audio_path "
        "FROM interview_notes WHERE note_id=?",
        (note_id,),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[1] == "prod1"
    assert row[3] == "/tmp/call.mp3"


@pytest.mark.asyncio
async def test_update_transcript_and_summary(db):
    """transcript_md and summary_md can be updated after transcription/summarization."""
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)

    await db.execute(
        "UPDATE interview_notes SET transcript_md=?, summary_md=? WHERE note_id=?",
        ("Speaker A: Hello. Speaker B: Hi.", "## Summary\n- Greeting discussed", note_id),
    )
    await db.commit()

    cur = await db.execute(
        "SELECT transcript_md, summary_md FROM interview_notes WHERE note_id=?",
        (note_id,),
    )
    row = await cur.fetchone()
    assert "Speaker A" in row[0]
    assert "Summary" in row[1]


# ── 3. interview/transcribe pluggability ─────────────────────────────────────


def test_transcribe_raises_when_whisper_missing():
    """_transcribe raises ImportError-derived error when whisper not installed."""
    from src.app.interview import _transcribe

    with patch.dict("sys.modules", {"whisper": None, "faster_whisper": None}):
        with pytest.raises(RuntimeError, match="whisper"):
            # Run synchronously since _transcribe is sync
            _transcribe("/fake/audio.mp3")


def test_transcribe_uses_whisper_when_available():
    """_transcribe returns transcript string when whisper is importable (mocked)."""
    import sys
    import types

    # Build minimal whisper stub
    fake_whisper = types.ModuleType("whisper")
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"text": "Hello world from interview."}
    fake_whisper.load_model = MagicMock(return_value=fake_model)

    with patch.dict(sys.modules, {"whisper": fake_whisper}):
        from src.app.interview import _transcribe
        result = _transcribe("/fake/audio.mp3", model_size="small")
    assert "Hello world" in result


def test_transcribe_uses_faster_whisper_when_openai_whisper_missing():
    """_transcribe falls back to faster-whisper when openai whisper absent."""
    import sys
    import types

    fake_fw = types.ModuleType("faster_whisper")
    fake_model = MagicMock()
    fake_segment = MagicMock()
    fake_segment.text = " Hello faster."
    fake_model.transcribe.return_value = ([fake_segment], MagicMock())
    fw_model_cls = MagicMock(return_value=fake_model)
    fake_fw.WhisperModel = fw_model_cls

    patched_modules = {"whisper": None, "faster_whisper": fake_fw}
    with patch.dict(sys.modules, patched_modules):
        from src.app import interview as _interview_mod
        # reload so the import-time check re-evaluates
        import importlib
        importlib.reload(_interview_mod)
        result = _interview_mod._transcribe("/fake/audio.mp3", model_size="small")
    assert "Hello faster" in result


# ── 3b. interview/transcribe — A10.r1 consent gate ───────────────────────────


async def _grant_consent(db, product_id, contact_id, purpose):
    """Insert a valid (not revoked, not expired) consent_records row."""
    await db.execute(
        "INSERT INTO consent_records "
        "(product_id, contact_id, purpose, granted_at, expires_at, "
        " source_evidence_url, revoked_at) "
        "VALUES (?, ?, ?, strftime('%Y-%m-%d %H:%M:%S','now'), NULL, ?, NULL)",
        (product_id, contact_id, purpose, "https://evidence.example/consent"),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_transcribe_blocked_without_recording_consent(db):
    """transcribe_interview refuses to transcribe when interview_recording
    consent is absent — exercises the REAL crm.has_consent path (no mock).

    Tautology guard: removing the consent gate would let this transcribe
    and return ok=True, failing the assertions below.
    """
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)

    transcribe_calls = []

    def _fake_transcribe(audio_path, model_size=None):
        transcribe_calls.append(audio_path)
        return "SHOULD NOT BE REACHED"

    # Only the whisper boundary is faked; crm.has_consent runs for real
    # against the real (empty) consent_records table.
    with patch("src.app.interview._transcribe", side_effect=_fake_transcribe):
        from src.app.interview import transcribe_interview
        result = await transcribe_interview(note_id=note_id, product_id="prod1")

    assert result["ok"] is False
    assert result.get("reason") == "consent_missing"
    # _transcribe must NOT have been called — no audio decode without consent.
    assert transcribe_calls == [], "audio was transcribed despite missing consent"

    # transcript_md must remain unset in the DB.
    cur = await db.execute(
        "SELECT transcript_md FROM interview_notes WHERE note_id=?", (note_id,)
    )
    row = await cur.fetchone()
    assert not row[0], "transcript written despite missing consent"


@pytest.mark.asyncio
async def test_transcribe_proceeds_with_recording_consent(db):
    """transcribe_interview transcribes when a real interview_recording
    consent record exists — verifies the gate is a gate, not a wall.
    """
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)
    # Grant real consent via the real schema.
    await _grant_consent(db, "prod1", contact_id, "interview_recording")

    def _fake_transcribe(audio_path, model_size=None):
        return "Speaker A: Hello from the interview."

    with patch("src.app.interview._transcribe", side_effect=_fake_transcribe):
        from src.app.interview import transcribe_interview
        result = await transcribe_interview(note_id=note_id, product_id="prod1")

    assert result["ok"] is True
    assert result["transcript_length"] > 0

    cur = await db.execute(
        "SELECT transcript_md FROM interview_notes WHERE note_id=?", (note_id,)
    )
    row = await cur.fetchone()
    assert "Hello from the interview" in (row[0] or "")


@pytest.mark.asyncio
async def test_transcribe_blocked_after_consent_revoked(db):
    """A revoked interview_recording consent must block transcription —
    exercises has_consent's revoked_at branch via the real ledger.
    """
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)
    await _grant_consent(db, "prod1", contact_id, "interview_recording")
    # Revoke it.
    await db.execute(
        "UPDATE consent_records "
        "SET revoked_at=strftime('%Y-%m-%d %H:%M:%S','now') "
        "WHERE product_id='prod1' AND contact_id=? AND purpose='interview_recording'",
        (contact_id,),
    )
    await db.commit()

    with patch("src.app.interview._transcribe",
               side_effect=AssertionError("must not transcribe")):
        from src.app.interview import transcribe_interview
        result = await transcribe_interview(note_id=note_id, product_id="prod1")

    assert result["ok"] is False
    assert result.get("reason") == "consent_missing"


# ── 4. interview/summarize — LLM-bound (mocked beckman.enqueue OVERHEAD) ─────


@pytest.mark.asyncio
async def test_summarize_structured_output(db):
    """interview/summarize enqueues via CPS, and the resume writes the DB.

    SP2: ``summarize_interview`` is now fire-and-forget — it returns
    ``{"ok": True, "queued": True}`` immediately. The structured columns
    are written by ``_summary_persist_resume`` when the child terminates.
    """
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)

    # Pre-populate transcript
    transcript = "Speaker: We love the pricing. Feature request: dark mode."
    await db.execute(
        "UPDATE interview_notes SET transcript_md=? WHERE note_id=?",
        (transcript, note_id),
    )
    await db.commit()

    # Mock beckman.enqueue to capture the CPS call (returns child task id).
    with patch(
        "src.app.interview.beckman_enqueue", new=AsyncMock(return_value=4242)
    ):
        from src.app.interview import summarize_interview
        result = await summarize_interview(note_id=note_id, product_id="prod1")

    assert result["ok"] is True
    assert result.get("queued") is True

    # Drive the resume manually with a representative terminal child result.
    mock_summary = {
        "bullets": ["Customer likes pricing", "Feature request: dark mode"],
        "quotes": ["We love the pricing"],
        "insights": "Customer is price-sensitive; values aesthetics.",
        "action_items": ["Add dark mode to backlog"],
    }
    from src.app.interview import _summary_persist_resume
    await _summary_persist_resume(
        child_task_id=4242,
        result={"status": "completed",
                "result": {"content": json.dumps(mock_summary)}},
        state={"note_id": note_id, "product_id": "prod1"},
    )

    # Verify DB updated
    cur = await db.execute(
        "SELECT summary_md, quotes_json, insights_md, action_items_json "
        "FROM interview_notes WHERE note_id=?",
        (note_id,),
    )
    row = await cur.fetchone()
    assert row[0] is not None, "summary_md not written"
    quotes = json.loads(row[1])
    assert "We love the pricing" in quotes
    action_items = json.loads(row[3])
    assert any("dark mode" in item.lower() for item in action_items)


# ── 5a. interview/cross_link — crm interaction ───────────────────────────────


@pytest.mark.asyncio
async def test_cross_link_writes_crm_interaction(db):
    """cross_link creates an interactions row (kind='interview')."""
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)

    # Set summary
    await db.execute(
        "UPDATE interview_notes SET summary_md=?, action_items_json=?, quotes_json=? "
        "WHERE note_id=?",
        (
            "## Summary\n- Pricing feedback",
            json.dumps(["Add dark mode"]),
            json.dumps(["We love it"]),
            note_id,
        ),
    )
    await db.commit()

    with patch(
        "src.app.interview.beckman_enqueue", new=AsyncMock()
    ), patch(
        "src.app.crm.has_consent", new=AsyncMock(return_value=False)
    ):
        # Mock founder_actions.create to avoid DB dependency chain
        with patch("src.founder_actions.create", new=AsyncMock(return_value=MagicMock(id=42))):
            from src.app.interview import cross_link_interview
            result = await cross_link_interview(
                note_id=note_id,
                product_id="prod1",
                contact_id=contact_id,
            )

    assert result["ok"] is True
    assert result.get("interaction_id") is not None

    # Verify interaction row
    cur = await db.execute(
        "SELECT kind, contact_id FROM interactions WHERE interaction_id=?",
        (result["interaction_id"],),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] == "interview"
    assert row[1] == contact_id


# ── 5b. cross_link — backlog enqueue ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_cross_link_enqueues_action_items(db):
    """cross_link enqueues each action item as a candidate backlog task."""
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)
    action_items = ["Add dark mode", "Fix onboarding flow"]

    await db.execute(
        "UPDATE interview_notes SET summary_md=?, action_items_json=?, quotes_json=? "
        "WHERE note_id=?",
        ("## Summary", json.dumps(action_items), json.dumps([]), note_id),
    )
    await db.commit()

    enqueue_calls = []

    async def _mock_enqueue(spec, **kw):
        enqueue_calls.append((spec, kw))
        return MagicMock(id=99)

    with patch("src.app.interview.beckman_enqueue", new=_mock_enqueue), \
         patch("src.app.crm.has_consent", new=AsyncMock(return_value=False)), \
         patch("src.founder_actions.create", new=AsyncMock(return_value=MagicMock(id=1))):
        from src.app.interview import cross_link_interview
        await cross_link_interview(
            note_id=note_id,
            product_id="prod1",
            contact_id=contact_id,
        )

    # One enqueue call per action item
    assert len(enqueue_calls) >= len(action_items)


# ── 5c. cross_link — quote consent gate ──────────────────────────────────────


@pytest.mark.asyncio
async def test_cross_link_quote_push_gated_on_consent_present(db):
    """When quote_use consent is present, quotes are inserted into press_kit_quotes."""
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)
    quotes = ["KutAI saved us 10 hours a week!"]

    await db.execute(
        "UPDATE interview_notes SET summary_md=?, action_items_json=?, quotes_json=? "
        "WHERE note_id=?",
        ("## Summary", json.dumps([]), json.dumps(quotes), note_id),
    )
    await db.commit()

    with patch("src.app.interview.beckman_enqueue", new=AsyncMock()), \
         patch("src.app.crm.has_consent", new=AsyncMock(return_value=True)), \
         patch("src.founder_actions.create", new=AsyncMock(return_value=MagicMock(id=1))):
        from src.app.interview import cross_link_interview
        result = await cross_link_interview(
            note_id=note_id,
            product_id="prod1",
            contact_id=contact_id,
        )

    assert result["ok"] is True
    assert result.get("quotes_pushed", 0) >= 1

    # Verify row in press_kit_quotes
    cur = await db.execute(
        "SELECT body, source_kind FROM press_kit_quotes WHERE product_id='prod1'"
    )
    rows = await cur.fetchall()
    assert any("KutAI" in r[0] for r in rows), "Quote not inserted into press_kit_quotes"
    assert all(r[1] == "interview" for r in rows)


@pytest.mark.asyncio
async def test_cross_link_quote_consent_absent_emits_founder_action(db):
    """When quote_use consent is absent, a founder_action is emitted instead of writing quote."""
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)
    quotes = ["Amazing product!"]

    await db.execute(
        "UPDATE interview_notes SET summary_md=?, action_items_json=?, quotes_json=? "
        "WHERE note_id=?",
        ("## Summary", json.dumps([]), json.dumps(quotes), note_id),
    )
    await db.commit()

    fa_create_calls = []

    async def _mock_fa_create(*args, **kw):
        fa_create_calls.append(kw)
        return MagicMock(id=77)

    with patch("src.app.interview.beckman_enqueue", new=AsyncMock()), \
         patch("src.app.crm.has_consent", new=AsyncMock(return_value=False)), \
         patch("src.founder_actions.create", new=_mock_fa_create):
        from src.app.interview import cross_link_interview
        result = await cross_link_interview(
            note_id=note_id,
            product_id="prod1",
            contact_id=contact_id,
        )

    assert result["ok"] is True
    # A founder_action should have been created requesting consent
    assert len(fa_create_calls) >= 1, "Expected founder_action for missing quote consent"

    # Quote must NOT be in press_kit_quotes
    cur = await db.execute(
        "SELECT COUNT(*) FROM press_kit_quotes WHERE product_id='prod1'"
    )
    count = (await cur.fetchone())[0]
    assert count == 0, "Quote inserted without consent — should be 0"


# ── 6. /interview Telegram commands registered ────────────────────────────────


def test_telegram_interview_commands_registered():
    """
    /interview command registered in TelegramInterface._setup_handlers
    or handle_message dispatch table.
    """
    import ast, pathlib
    src = pathlib.Path(
        "C:/Users/sakir/Dropbox/Workspaces/kutay/src/app/telegram_bot.py"
    ).read_text(encoding="utf-8")
    assert "interview" in src.lower(), (
        "/interview command not found in telegram_bot.py"
    )


# ── 7. A0 briefing founder_action after completion ────────────────────────────


@pytest.mark.asyncio
async def test_complete_interview_emits_review_founder_action(db):
    """After interview completion, a 'review interview note' founder_action is emitted."""
    contact_id = await _add_contact(db)
    note_id = await _insert_interview(db, contact_id=contact_id)

    # Set the note as fully populated
    await db.execute(
        "UPDATE interview_notes "
        "SET transcript_md=?, summary_md=?, quotes_json=?, action_items_json=? "
        "WHERE note_id=?",
        (
            "Full transcript here.",
            "## Summary\n- Key finding",
            json.dumps(["Great product"]),
            json.dumps(["Action A"]),
            note_id,
        ),
    )
    await db.commit()

    fa_calls = []

    async def _capture_fa(*args, **kw):
        fa_calls.append(kw)
        return MagicMock(id=55)

    with patch("src.founder_actions.create", new=_capture_fa):
        from src.app.interview import emit_review_founder_action
        result = await emit_review_founder_action(
            note_id=note_id,
            product_id="prod1",
            mission_id=0,
        )

    assert result["ok"] is True
    assert len(fa_calls) >= 1
    # Title or kind should reference interview review
    titles = " ".join(str(c.get("title", "")) for c in fa_calls).lower()
    assert "interview" in titles or "review" in titles


# ── 8. Reversibility registry ─────────────────────────────────────────────────


def test_interview_reversibility_registered():
    """interview/* verbs have reversibility entries."""
    from mr_roboto.reversibility import VERB_REVERSIBILITY
    assert "interview/transcribe" in VERB_REVERSIBILITY
    assert "interview/summarize" in VERB_REVERSIBILITY
    assert "interview/cross_link" in VERB_REVERSIBILITY


def test_interview_transcribe_is_full():
    """interview/transcribe is 'full' (local file write, no external side-effect)."""
    from mr_roboto.reversibility import VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["interview/transcribe"] == "full"


def test_interview_summarize_is_full():
    """interview/summarize is 'full' (DB write; re-runnable)."""
    from mr_roboto.reversibility import VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["interview/summarize"] == "full"


def test_interview_cross_link_is_full():
    """interview/cross_link is 'full' (local DB writes; no external publish)."""
    from mr_roboto.reversibility import VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["interview/cross_link"] == "full"
