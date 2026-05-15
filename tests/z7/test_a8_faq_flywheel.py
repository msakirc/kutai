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
    """A cluster with > 3 tickets should produce a draft FAQ entry."""
    from src.app.jobs.faq_regen import _draft_faq_entry

    cluster = [
        {"question": "How to reset password?", "answer": "Go to settings."},
        {"question": "I forgot my password, help!", "answer": "Go to settings."},
        {"question": "Password reset not working?", "answer": "Try the email link."},
        {"question": "Can I change my password?", "answer": "Yes, in settings."},
    ]

    # Mock the LLM clustering call
    mock_result = {
        "question": "How do I reset or change my password?",
        "answer": "Go to Settings > Security > Reset Password. An email link will be sent.",
    }
    with patch("src.app.jobs.faq_regen._llm_cluster_draft", new_callable=AsyncMock) as m:
        m.return_value = mock_result
        result = await _draft_faq_entry(cluster, lang="en")

    assert result is not None
    assert "question" in result
    assert "answer" in result


@pytest.mark.asyncio
async def test_faq_regen_small_cluster_skipped():
    """A cluster with <= 3 tickets should NOT draft an FAQ entry."""
    from src.app.jobs.faq_regen import _draft_faq_entry

    cluster = [
        {"question": "How to reset password?", "answer": "Go to settings."},
        {"question": "I forgot my password, help!", "answer": "Go to settings."},
    ]

    with patch("src.app.jobs.faq_regen._llm_cluster_draft", new_callable=AsyncMock) as m:
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
    """On approve, re-index the per-language Chroma collection via lang_collection_name."""
    from src.util.lang import lang_collection_name

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    monkeypatch.setattr("src.app.jobs.faq_regen.FAQ_ARTIFACTS_DIR", str(artifacts_dir))

    reindexed_collections: list[str] = []

    async def mock_reindex(collection_name: str, text: str) -> None:
        reindexed_collections.append(collection_name)

    monkeypatch.setattr("src.app.jobs.faq_regen._reindex_collection", mock_reindex)

    from src.app.jobs.faq_regen import _apply_faq_approval
    entry = {
        "question": "How to reset password?",
        "answer": "Via settings.",
        "lang": "en",
    }
    await _apply_faq_approval(entry)

    expected_collection = lang_collection_name("support_docs", "en")
    assert expected_collection in reindexed_collections


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
async def test_quote_harvest_emits_founder_action_for_consent():
    """quote_harvest emits a founder_action(kind='generic') for each quote candidate."""
    from src.app.jobs.quote_harvest import _emit_consent_request

    created_actions: list[dict] = []

    async def mock_create(**kwargs):
        created_actions.append(kwargs)
        return MagicMock(id=42)

    with patch("src.app.jobs.quote_harvest._create_founder_action", mock_create):
        result = await _emit_consent_request(
            ticket={
                "id": 1,
                "user_id": "u1",
                "question": "Great product!",
                "answer": "Thank you!",
            },
            product_id="prod-1",
            mission_id=10,
        )

    assert len(created_actions) == 1
    action = created_actions[0]
    assert action.get("kind") in ("generic", "consent_request", "quote_consent")


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

    mock_hit = MagicMock()
    mock_hit.id = "doc-123"

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
