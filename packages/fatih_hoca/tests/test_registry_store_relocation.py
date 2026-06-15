def test_relocated_module_importable():
    from fatih_hoca import registry_store as rs
    for fn in ("register_model", "mark_dead", "is_dead", "revive", "list_dead",
               "register_provider", "mark_provider_dead", "is_provider_dead",
               "list_dead_providers", "revive_provider", "get_provider_key_hash",
               "recent_events", "get_model_cause"):
        assert hasattr(rs, fn)
    assert hasattr(rs, "CAUSE_POLICY")


def test_legacy_import_path_still_works():
    from src.infra import registry_store as legacy
    from fatih_hoca import registry_store as new
    assert legacy.register_model is new.register_model
    assert legacy.CAUSE_POLICY is new.CAUSE_POLICY


def test_ensure_schema_uses_shared_ddl(tmp_path):
    from fatih_hoca import registry_store as rs
    rs.set_db_path(str(tmp_path / "rs.db"))
    try:
        conn = rs._get_conn()  # triggers _ensure_schema
        cols = {r[1] for r in conn.execute("PRAGMA table_info(registry_events)").fetchall()}
        assert {"mission_id", "task_id", "verb", "reversibility"}.issubset(cols)
    finally:
        rs.close()
