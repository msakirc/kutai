"""Z7 T4 A8 — FAQ flywheel + A8.r1 multilingual regen tests.

Covers:
  1. DB migrations: docs_gap_log table exists with correct columns.
  2. faq_regen: pulls low-confidence + escalated tickets from last 7 days.
  3. faq_regen: clusters within-language only (en + tr never cross-cluster).
  4. faq_regen: cluster > 3 tickets drafts an FAQ entry and emits a founder_action.
  5. faq_regen: on approve, appends to faq_{lang}.md via lang_artifact_path convention.
  6. faq_regen: on approve, re-indexes per-language Chroma collection via lang_collection_name.
  7. quote_harvest: scans positive-resolution tickets; emits founder_action for consent.
  8. quote_harvest: on consent, inserts into press_kit_quotes.
  9. documentation_gap_detect posthook: semantic search miss → writes docs_gap_log row.
  10. documentation_gap_detect: semantic search hit → no gap row written.
  11. documentation_gap_detect posthook: registered in POST_HOOK_REGISTRY.
  12. documentation_gap_detect posthook: routed in mr_roboto action dispatch.
  13. faq_regen + quote_harvest cron entries registered in INTERNAL_CADENCES.
"""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── DB fixture ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for A8 tests."""
    db_file = str(tmp_path / "test_a8.db")
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
async def init_db(tmp_db):
    """Run init_db against the tmp DB so all migrations are applied."""
    from src.infra.db import init_db as _init_db
    await _init_db()
    return tmp_db


# ── Migration tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_docs_gap_log_table_exists(init_db):
    """docs_gap_log table must exist after init_db()."""
    import aiosqlite
    async with aiosqlite.connect(init_db) as db:
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_gap_log'"
        )
        row = await cur.fetchone()
    assert row is not None, "docs_gap_log table not created by migration"


@pytest.mark.asyncio
async def test_docs_gap_log_columns(init_db):
    """docs_gap_log must have gap_id, product_id (NOT NULL), question, matched_doc_id, logged_at."""
    import aiosqlite
    async with aiosqlite.connect(init_db) as db:
        cur = await db.execute("PRAGMA table_info(docs_gap_log)")
        cols = {row[1]: row for row in await cur.fetchall()}
    assert "gap_id" in cols
    assert "product_id" in cols
    assert "question" in cols
    assert "matched_doc_id" in cols
    assert "logged_at" in cols
    # product_id must be NOT NULL (notnull=1 in PRAGMA)
    # cols[name] = (cid, name, type, notnull, dflt_value, pk)
    assert cols["product_id"][3] == 1, "product_id should be NOT NULL"


# ── faq_regen tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_faq_regen_pulls_low_confidence_and_escalated(init_db):
    """faq_regen fetches tickets with confidence < threshold OR escalated_to_founder."""
    import aiosqlite
    async with aiosqlite.connect(init_db) as db:
        # Insert qualifying tickets
        await db.execute(
            "INSERT INTO tickets (user_id, question, answer, confidence, status, "
            "escalated_to_founder, sentiment, created_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            ("u1", "How to reset password?", "Go to settings.", 0.5, "closed", 0, "neutral"),
        )
        await db.execute(
            "INSERT INTO tickets (user_id, question, answer, confidence, status, "
            "escalated_to_founder, sentiment, created_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            ("u2", "Why is billing broken?", "We're investigating.", 0.9, "escalated", 1, "angry"),
        )
        # Insert non-qualifying ticket (high confidence + not escalated)
        await db.execute(
            "INSERT INTO tickets (user_id, question, answer, confidence, status, "
            "escalated_to_founder, sentiment, created_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            ("u3", "What is the pricing?", "See /pricing.", 0.95, "closed", 0, "neutral"),
        )
        await db.commit()

    from src.app.jobs.faq_regen import _fetch_candidate_tickets
    tickets = await _fetch_candidate_tickets(confidence_threshold=0.7, days=7)

    questions = [t["question"] for t in tickets]
    assert "How to reset password?" in questions
    assert "Why is billing broken?" in questions
    assert "What is the pricing?" not in questions


@pytest.mark.asyncio
async def test_faq_regen_clusters_within_language_only(init_db):
    """Tickets should be grouped by language before clustering; en and tr never cross-cluster."""
    import aiosqlite
    async with aiosqlite.connect(init_db) as db:
        en_question = "How do I reset my password in the system?"
        tr_question = "Şifremi nasıl sıfırlayabilirim bu sistemde?"
        for q in [en_question, tr_question]:
            await db.execute(
                "INSERT INTO tickets (user_id, question, answer, confidence, status, "
                "escalated_to_founder, sentiment, created_at) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                ("u1", q, "answer", 0.5, "closed", 0, "neutral"),
            )
        await db.commit()

    from src.app.jobs.faq_regen import _group_tickets_by_language
    from src.util.lang import detect_language

    tickets = [
        {"question": "How do I reset my password in the system?", "id": 1},
        {"question": "Şifremi nasıl sıfırlayabilirim bu sistemde?", "id": 2},
        {"question": "How can I change my account email address?", "id": 3},
    ]
    grouped = _group_tickets_by_language(tickets)

    assert "en" in grouped
    assert "tr" in grouped
    # tr ticket must not appear in en group
    en_ids = [t["id"] for t in grouped["en"]]
    tr_ids = [t["id"] for t in grouped["tr"]]
    assert 2 not in en_ids
    assert 1 not in tr_ids


@pytest.mark.asyncio
async def test_faq_regen_cluster_gt3_drafts_entry():
    """A cluster with > 3 tickets should produce a draft FAQ entry via beckman enqueue."""
    import general_beckman
    from general_beckman import TaskResult
    from src.app.jobs.faq_regen import _draft_faq_entry

    cluster = [
        {"question": "How to reset password?", "answer": "Go to settings."},
        {"question": "I forgot my password, help!", "answer": "Go to settings."},
        {"question": "Password reset not working?", "answer": "Try the email link."},
        {"question": "Can I change my password?", "answer": "Yes, in settings."},
    ]

    # Mock beckman.enqueue at the LLM boundary — the real API is called,
    # only the actual LLM execution is faked.
    mock_result = TaskResult(
        status="completed",
        result={"content": '{"question": "How do I reset or change my password?", "answer": "Go to Settings > Security > Reset Password."}'},
        error=None,
    )
    with patch.object(general_beckman, "enqueue", new_callable=AsyncMock) as m:
        m.return_value = mock_result
        result = await _draft_faq_entry(cluster, lang="en")

    # enqueue must have been called with await_inline=True
    m.assert_called_once()
    call_kwargs = m.call_args[1]
    assert call_kwargs.get("await_inline") is True, "enqueue must be called with await_inline=True"

    assert result is not None
    assert "question" in result
    assert "answer" in result


@pytest.mark.asyncio
async def test_faq_regen_small_cluster_skipped():
    """A cluster with <= 3 tickets should NOT call beckman enqueue."""
    import general_beckman
    from src.app.jobs.faq_regen import _draft_faq_entry

    cluster = [
        {"question": "How to reset password?", "answer": "Go to settings."},
        {"question": "I forgot my password, help!", "answer": "Go to settings."},
    ]

    with patch.object(general_beckman, "enqueue", new_callable=AsyncMock) as m:
        result = await _draft_faq_entry(cluster, lang="en")

    m.assert_not_called()
    assert result is None


@pytest.mark.asyncio
async def test_faq_regen_approve_appends_to_faq_file(tmp_path, monkeypatch):
    """On approve, the FAQ entry is appended to the per-language faq_{lang}.md file."""
    from src.app.jobs.faq_regen import _apply_faq_approval
    from src.util.lang import lang_artifact_path

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    monkeypatch.setattr("src.app.jobs.faq_regen.FAQ_ARTIFACTS_DIR", str(artifacts_dir))

    from src.infra.db import get_db
    _db = await get_db()
    _c = await _db.execute(
        "INSERT INTO missions (title, status, created_at) "
        "VALUES ('m', 'active', datetime('now'))")
    await _db.commit()
    _mid = _c.lastrowid

    entry = {
        "question": "How do I reset my password?",
        "answer": "Go to Settings > Security > Reset.",
        "lang": "en",
    }

    await _apply_faq_approval(entry)

    faq_path = artifacts_dir / lang_artifact_path("faq", "en")  # → faq.md
    assert faq_path.exists()
    content = faq_path.read_text(encoding="utf-8")
    assert "How do I reset my password?" in content
    assert "Settings > Security > Reset." in content


@pytest.mark.asyncio
async def test_faq_regen_approve_appends_turkish_faq(tmp_path, monkeypatch):
    """On approve for Turkish, the FAQ entry is appended to faq_tr.md."""
    from src.app.jobs.faq_regen import _apply_faq_approval
    from src.util.lang import lang_artifact_path

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    monkeypatch.setattr("src.app.jobs.faq_regen.FAQ_ARTIFACTS_DIR", str(artifacts_dir))

    entry = {
        "question": "Şifremi nasıl sıfırlayabilirim?",
        "answer": "Ayarlar > Güvenlik > Şifreyi Sıfırla bölümüne gidin.",
        "lang": "tr",
    }

    await _apply_faq_approval(entry)

    faq_path = artifacts_dir / lang_artifact_path("faq", "tr")  # → faq_tr.md
    assert faq_path.exists()
    content = faq_path.read_text(encoding="utf-8")
    assert "Şifremi nasıl sıfırlayabilirim?" in content


@pytest.mark.asyncio
async def test_faq_regen_approve_reindexes_chroma_collection(tmp_path, monkeypatch):
    """On approve, re-index the per-language Chroma collection via the real embed_and_store path.

    _reindex_collection is NOT mocked — the real function is exercised.
    Only embed_and_store (the ChromaDB/GPU boundary) is faked.
    """
    from src.util.lang import lang_collection_name
    import src.memory.vector_store as vs_mod

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    monkeypatch.setattr("src.app.jobs.faq_regen.FAQ_ARTIFACTS_DIR", str(artifacts_dir))

    stored_calls: list[dict] = []

    async def mock_embed_and_store(text, metadata, collection="semantic", doc_id=None):
        stored_calls.append({"text": text, "metadata": metadata, "collection": collection})
        return "fake-doc-id"

    monkeypatch.setattr(vs_mod, "embed_and_store", mock_embed_and_store)

    from src.app.jobs.faq_regen import _apply_faq_approval
    entry = {
        "question": "How to reset password?",
        "answer": "Via settings.",
        "lang": "en",
    }
    await _apply_faq_approval(entry)

    # embed_and_store must have been called with the per-language collection
    expected_collection = lang_collection_name("support_docs", "en")
    assert len(stored_calls) == 1, "embed_and_store must be called exactly once"
    assert stored_calls[0]["collection"] == expected_collection, (
        f"embed_and_store called with collection {stored_calls[0]['collection']!r}, "
        f"expected {expected_collection!r}"
    )


@pytest.mark.asyncio
async def test_faq_emit_founder_action_uses_real_create(init_db):
    """_emit_faq_founder_action must call founder_actions.create with valid params (no context_json).

    Previously this call passed context_json= which raises TypeError at runtime.
    This test exercises the REAL create() path against SQLite to catch any re-regression.
    """
    import src.founder_actions as fa_mod
    from src.app.jobs.faq_regen import _emit_faq_founder_action

    entry = {
        "question": "How do I reset my password?",
        "answer": "Go to Settings > Reset.",
        "lang": "en",
    }

    # Suppress telegram notification only
    with patch.object(fa_mod, "_notify_telegram", new_callable=AsyncMock):
        result = await _emit_faq_founder_action(
            mission_id=0,
            entry=entry,
            cluster_size=5,
        )

    assert result is not None, "_emit_faq_founder_action must return a FounderAction"
    assert hasattr(result, "id"), "result must be a persisted FounderAction with id"
    # Payload must be accessible from expected_output_schema
    assert result.expected_output_schema is not None
    assert "_faq_approval_pending" in result.expected_output_schema


# ── quote_harvest tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quote_harvest_scans_positive_tickets(init_db):
    """quote_harvest fetches tickets resolved with positive sentiment."""
    import aiosqlite
    async with aiosqlite.connect(init_db) as db:
        await db.execute(
            "INSERT INTO tickets (user_id, question, answer, confidence, status, "
            "escalated_to_founder, sentiment, created_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            ("u1", "Great product!", "Thank you!", 0.95, "closed", 0, "positive"),
        )
        await db.execute(
            "INSERT INTO tickets (user_id, question, answer, confidence, status, "
            "escalated_to_founder, sentiment, created_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            ("u2", "This is terrible!", "Sorry to hear that.", 0.8, "closed", 0, "angry"),
        )
        await db.commit()

    from src.app.jobs.quote_harvest import _fetch_positive_tickets
    tickets = await _fetch_positive_tickets(days=30)

    sentiments = [t["sentiment"] for t in tickets]
    assert "positive" in sentiments
    assert "angry" not in sentiments


@pytest.mark.asyncio
async def test_quote_harvest_emits_founder_action_for_consent(init_db):
    """quote_harvest emits a founder_action(kind='generic') via the REAL founder_actions.create.

    _create_founder_action is NOT mocked — the real create() is called against real SQLite,
    exercising the actual parameter contract. Only the Telegram notifier (network boundary)
    is suppressed.
    """
    import src.founder_actions as fa_mod
    from src.app.jobs.quote_harvest import _emit_consent_request

    # Suppress telegram notification (network boundary only)
    with patch.object(fa_mod, "_notify_telegram", new_callable=AsyncMock):
        result = await _emit_consent_request(
            ticket={
                "id": 1,
                "user_id": "u1",
                "question": "Great product!",
                "answer": "Thank you!",
            },
            product_id="prod-1",
            mission_id=0,
        )

    # The call must succeed and return a persisted FounderAction
    assert result is not None, "founder_actions.create must return a FounderAction"
    assert hasattr(result, "id"), "result must have an id (persisted row)"
    assert result.kind in ("generic", "consent_request", "quote_consent")
    # Payload must be recoverable from expected_output_schema (not lost via context_json)
    assert result.expected_output_schema is not None, (
        "payload must be stored in expected_output_schema"
    )
    assert "_quote_consent_pending" in result.expected_output_schema


@pytest.mark.asyncio
async def test_quote_harvest_consent_inserts_press_kit_quote(init_db):
    """On consent, quote_harvest inserts a row into press_kit_quotes."""
    import aiosqlite
    from src.app.jobs.quote_harvest import _on_consent_approved

    await _on_consent_approved(
        product_id="prod-1",
        ticket_id=1,
        speaker="u1",
        body="This product changed our workflow completely.",
    )

    async with aiosqlite.connect(init_db) as db:
        cur = await db.execute(
            "SELECT body, source_kind FROM press_kit_quotes WHERE product_id = 'prod-1'"
        )
        rows = await cur.fetchall()

    assert len(rows) == 1
    assert "changed our workflow" in rows[0][0]
    assert rows[0][1] == "ticket"


# ── documentation_gap_detect tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_gap_detect_writes_row_when_no_match(init_db):
    """documentation_gap_detect writes a docs_gap_log row when semantic search returns no match."""
    import aiosqlite

    # Mock the support_rag retrieve_docs to return empty list (no match)
    with patch(
        "packages.general_beckman.src.general_beckman.posthook_handlers.documentation_gap_detect.retrieve_docs",
        new_callable=AsyncMock,
    ) as mock_retrieve:
        mock_retrieve.return_value = []  # no matching docs

        from packages.general_beckman.src.general_beckman.posthook_handlers.documentation_gap_detect import handle

        task = {
            "id": 99,
            "mission_id": 10,
            "context": json.dumps({
                "product_id": "prod-1",
                "question": "How do I enable two-factor authentication?",
            }),
        }
        result = await handle(task, {})

    assert result.get("status") in ("gap_logged", "ok")

    async with aiosqlite.connect(init_db) as db:
        cur = await db.execute("SELECT * FROM docs_gap_log WHERE product_id='prod-1'")
        rows = await cur.fetchall()
    assert len(rows) >= 1


@pytest.mark.asyncio
async def test_gap_detect_no_row_when_match_found(init_db):
    """documentation_gap_detect does NOT write a gap row when semantic search returns a hit."""
    import aiosqlite

    with patch(
        "packages.general_beckman.src.general_beckman.posthook_handlers.documentation_gap_detect.retrieve_docs",
        new_callable=AsyncMock,
    ) as mock_retrieve:
        mock_retrieve.return_value = [{"id": "doc-123", "score": 0.9}]

        from packages.general_beckman.src.general_beckman.posthook_handlers.documentation_gap_detect import handle

        task = {
            "id": 100,
            "mission_id": 10,
            "context": json.dumps({
                "product_id": "prod-2",
                "question": "What are the pricing plans?",
            }),
        }
        result = await handle(task, {})

    assert result.get("status") in ("ok", "skip", "covered")

    async with aiosqlite.connect(init_db) as db:
        cur = await db.execute("SELECT * FROM docs_gap_log WHERE product_id='prod-2'")
        rows = await cur.fetchall()
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_gap_detect_retrieve_docs_passes_collection_to_vector_store():
    """retrieve_docs must pass collection_name to vector_store.query, not support_rag.

    Previously, retrieve_docs delegated to support_rag.retrieve_docs which has NO
    collection_name parameter — the multilingual collection was silently ignored.
    This test verifies the fix: vector_store.query is called with the correct collection.
    """
    import src.memory.vector_store as vs_mod
    from packages.general_beckman.src.general_beckman.posthook_handlers.documentation_gap_detect import (
        retrieve_docs,
    )

    queried_collections: list[str] = []

    async def mock_vs_query(text, collection="semantic", top_k=5, where=None):
        queried_collections.append(collection)
        return []

    with patch.object(vs_mod, "query", mock_vs_query):
        await retrieve_docs("Şifre nasıl sıfırlanır?", collection_name="support_docs_tr", top_k=1)

    assert queried_collections == ["support_docs_tr"], (
        f"Expected vector_store.query to be called with 'support_docs_tr', "
        f"got {queried_collections}"
    )


# ── Registry + cron tests ─────────────────────────────────────────────────────


def test_documentation_gap_detect_in_posthook_registry():
    """documentation_gap_detect must be in POST_HOOK_REGISTRY."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "documentation_gap_detect" in POST_HOOK_REGISTRY


def test_faq_regen_in_cron_cadences():
    """faq_regen must be in INTERNAL_CADENCES."""
    from general_beckman.cron_seed import INTERNAL_CADENCES
    titles = [c["title"] for c in INTERNAL_CADENCES]
    assert "faq_regen" in titles


def test_quote_harvest_in_cron_cadences():
    """quote_harvest must be in INTERNAL_CADENCES."""
    from general_beckman.cron_seed import INTERNAL_CADENCES
    titles = [c["title"] for c in INTERNAL_CADENCES]
    assert "quote_harvest" in titles


def test_faq_regen_weekly_interval():
    """faq_regen cron entry must be weekly (7 days = 604800 seconds)."""
    from general_beckman.cron_seed import INTERNAL_CADENCES
    entry = next((c for c in INTERNAL_CADENCES if c["title"] == "faq_regen"), None)
    assert entry is not None
    # Either interval_seconds=604800 or cron_expression with weekly pattern
    interval = entry.get("interval_seconds")
    cron = entry.get("cron_expression", "")
    assert interval == 604800 or "7" in cron or "week" in cron.lower()


def test_quote_harvest_monthly_interval():
    """quote_harvest cron entry must be monthly (~2592000 seconds)."""
    from general_beckman.cron_seed import INTERNAL_CADENCES
    entry = next((c for c in INTERNAL_CADENCES if c["title"] == "quote_harvest"), None)
    assert entry is not None
    interval = entry.get("interval_seconds")
    cron = entry.get("cron_expression", "")
    assert interval == 2592000 or "month" in cron.lower() or (cron and cron.count("*") >= 2)


def test_documentation_gap_detect_routed_in_mr_roboto():
    """documentation_gap_detect must be handled in mr_roboto.__init__ run()."""
    import mr_roboto
    source = inspect_run_fn_for_action("documentation_gap_detect")
    assert source, "documentation_gap_detect not found in mr_roboto run() dispatch"


def test_faq_regen_routed_in_mr_roboto():
    """faq_regen must be handled in mr_roboto.__init__ run()."""
    source = inspect_run_fn_for_action("faq_regen")
    assert source, "faq_regen not found in mr_roboto run() dispatch"


def test_quote_harvest_routed_in_mr_roboto():
    """quote_harvest must be handled in mr_roboto.__init__ run()."""
    source = inspect_run_fn_for_action("quote_harvest")
    assert source, "quote_harvest not found in mr_roboto run() dispatch"


def inspect_run_fn_for_action(action_name: str) -> bool:
    """Check that the mr_roboto _run_dispatch() function source references action_name."""
    import inspect
    import mr_roboto
    try:
        # The actual action routing lives in _run_dispatch, not run().
        src = inspect.getsource(mr_roboto._run_dispatch)
    except Exception:
        return False
    return action_name in src


# ── lang_artifact_path convention tests ──────────────────────────────────────


def test_lang_artifact_path_en_returns_bare():
    """English FAQ → faq.md (no language suffix)."""
    from src.util.lang import lang_artifact_path
    assert lang_artifact_path("faq", "en") == "faq.md"


def test_lang_artifact_path_tr_returns_suffixed():
    """Turkish FAQ → faq_tr.md."""
    from src.util.lang import lang_artifact_path
    assert lang_artifact_path("faq", "tr") == "faq_tr.md"


def test_lang_collection_name_en():
    """English support_docs collection → support_docs_en."""
    from src.util.lang import lang_collection_name
    assert lang_collection_name("support_docs", "en") == "support_docs_en"


def test_lang_collection_name_tr():
    """Turkish support_docs collection → support_docs_tr."""
    from src.util.lang import lang_collection_name
    assert lang_collection_name("support_docs", "tr") == "support_docs_tr"


@pytest.mark.asyncio
async def test_support_docs_en_round_trip(tmp_path, monkeypatch):
    """support_docs_en is registered in COLLECTIONS and accepts real embed_and_store + query.

    This test exercises the REAL embed_and_store and query paths (no mock at the
    collection guard layer) so any regression in COLLECTIONS registration is caught
    immediately.  The embedder itself runs locally on CPU (multilingual-e5-base) — no
    network call.  ChromaDB is pointed at a tmp_path so there is no shared-state
    pollution.
    """
    import src.memory.vector_store as vs_mod

    # Redirect Chroma to an isolated tmp dir so the test never touches production data.
    monkeypatch.setattr(vs_mod, "_DB_DIR", str(tmp_path / "chroma"))
    # Reset module-level state so init_store() rebuilds with the tmp dir.
    monkeypatch.setattr(vs_mod, "_initialized", False)
    monkeypatch.setattr(vs_mod, "_collections", {})

    from src.memory.vector_store import embed_and_store, query, init_store

    # Initialise — this is what creates the Chroma collections from COLLECTIONS.
    ok = await init_store()
    assert ok, "init_store must succeed with tmp Chroma dir"

    # Write a document to the per-language English collection.
    doc_id = await embed_and_store(
        "How to reset your password: visit Settings and click Reset.",
        metadata={"lang": "en", "source": "faq_regen_test"},
        collection="support_docs_en",
    )
    assert doc_id is not None, (
        "embed_and_store returned None for 'support_docs_en' — "
        "collection is likely not registered in COLLECTIONS"
    )

    # Read it back via query.
    results = await query(
        "password reset",
        collection="support_docs_en",
        top_k=1,
    )
    assert len(results) == 1, (
        f"query on 'support_docs_en' returned {len(results)} results, expected 1 — "
        "collection guard is rejecting the collection or embedding failed"
    )
    assert "password" in results[0]["text"].lower() or "reset" in results[0]["text"].lower(), (
        f"Unexpected round-trip text: {results[0]['text']!r}"
    )


# ── #2 wiring sweep — founder-action approval routes into _apply_faq_approval ──


class _FakeMsg:
    def __init__(self):
        self.replies = []

        class _Chat:
            id = 4242
        self.chat = _Chat()

    async def reply_text(self, text, **kw):
        self.replies.append(text)
        return self


class _FakeUpdate:
    def __init__(self):
        self.message = _FakeMsg()

    @property
    def effective_chat(self):
        return self.message.chat


class _FakeCtx:
    def __init__(self, args=None):
        self.args = args or []


def _make_tg():
    from src.app.telegram_bot import TelegramInterface
    tg = TelegramInterface.__new__(TelegramInterface)
    tg._kb_state = {}
    return tg


@pytest.mark.asyncio
async def test_action_done_routes_faq_card_into_support_docs(init_db, tmp_path, monkeypatch):
    """Resolving a _faq_approval_pending founder_action via /action_done must
    append the drafted FAQ entry to the support docs — the only writer of the
    support_docs_* collections previously had zero callers."""
    from src.util.lang import lang_artifact_path
    import src.founder_actions as fa

    artifacts_dir = tmp_path / "faq_artifacts"
    artifacts_dir.mkdir()
    monkeypatch.setattr("src.app.jobs.faq_regen.FAQ_ARTIFACTS_DIR", str(artifacts_dir))

    from src.infra.db import get_db
    _db = await get_db()
    _c = await _db.execute(
        "INSERT INTO missions (title, status, created_at) "
        "VALUES ('m', 'active', datetime('now'))")
    await _db.commit()
    _mid = _c.lastrowid

    entry = {
        "question": "How do I export my data?",
        "answer": "Settings > Data > Export.",
        "lang": "en",
    }
    action = await fa.create(
        mission_id=_mid,
        kind="generic",
        title="Approve FAQ entry [en]",
        why="clustered tickets",
        instructions=["review"],
        expected_output_kind="ack_only",
        expected_output_schema={"faq_entry": entry, "_faq_approval_pending": True},
    )

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_action_done(update, _FakeCtx(args=[str(action.id)]))

    faq_path = artifacts_dir / lang_artifact_path("faq", "en")
    assert faq_path.exists(), "FAQ approval did not write the support-docs file"
    assert "How do I export my data?" in faq_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_action_done_faq_reject_does_not_index(init_db, tmp_path, monkeypatch):
    """A reject payload discards the draft — nothing is indexed."""
    from src.util.lang import lang_artifact_path
    import src.founder_actions as fa

    artifacts_dir = tmp_path / "faq_artifacts_rej"
    artifacts_dir.mkdir()
    monkeypatch.setattr("src.app.jobs.faq_regen.FAQ_ARTIFACTS_DIR", str(artifacts_dir))

    from src.infra.db import get_db
    _db = await get_db()
    _c = await _db.execute(
        "INSERT INTO missions (title, status, created_at) "
        "VALUES ('m', 'active', datetime('now'))")
    await _db.commit()
    _mid = _c.lastrowid

    action = await fa.create(
        mission_id=_mid,
        kind="generic",
        title="Approve FAQ entry [en]",
        why="clustered tickets",
        instructions=["review"],
        expected_output_kind="ack_only",
        expected_output_schema={
            "faq_entry": {"question": "Q?", "answer": "A.", "lang": "en"},
            "_faq_approval_pending": True,
        },
    )

    tg = _make_tg()
    update = _FakeUpdate()
    await tg.cmd_action_done(update, _FakeCtx(args=[str(action.id), '{"reject": true}']))

    faq_path = artifacts_dir / lang_artifact_path("faq", "en")
    assert not faq_path.exists(), "rejected FAQ draft must not be indexed"
