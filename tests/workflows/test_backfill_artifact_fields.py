"""Evidence-backfill for the coder grade-path schema gate.

A weak model often DOES the work (write_file / migrate — files on disk) but
returns a malformed final artifact and loops on the identical artifact_schema
reject. `backfill_artifact_fields` fills REQUIRED, NON-must_be_true, path-shaped
fields from concrete evidence so the loop converges — while NEVER fabricating a
must_be_true verification claim and NEVER filling with no evidence.
"""
from src.workflows.engine.hooks import (
    backfill_artifact_fields,
    validate_artifact_schema,
)


def _wf(path):
    return {"name": "write_file", "ok": True, "args": {"filepath": path}}


def _sh(cmd, ok=True):
    return {"name": "shell", "ok": ok, "args": {"command": cmd}}


SCHEMA_A = {"schema_authored": {"type": "object",
            "required_fields": ["schema_path", "datasource"]}}
SCHEMA_B = {"db_client": {"type": "object",
            "required_fields": ["client_path", "connection_verified"],
            "must_be_true": ["connection_verified"]}}
SCHEMA_C = {"migration_system": {"type": "object",
            "required_fields": ["tool", "initial_migration"]}}


def test_path_field_backfilled_from_write_file():
    draft = '{"schema_authored": {"datasource": "sqlite: file:./dev.db"}}'
    tc = [_wf("mission_90/backend/prisma/schema.prisma")]
    enriched, changed = backfill_artifact_fields(
        draft, SCHEMA_A, tc, ["mission_90/backend/"])
    assert changed
    ok, err = validate_artifact_schema(enriched, SCHEMA_A)
    assert ok, err
    assert "schema.prisma" in enriched


def test_raw_content_result_gets_path_filled():
    # coder returned the schema text itself, not the JSON
    draft = "generator client {\n provider = \"prisma-client-js\"\n}"
    tc = [_wf("mission_90/backend/prisma/schema.prisma")]
    enriched, changed = backfill_artifact_fields(
        draft, SCHEMA_A, tc, ["mission_90/backend/"])
    assert changed and "schema_path" in enriched and "schema.prisma" in enriched


def test_must_be_true_never_fabricated():
    # coder wrote the client but omitted BOTH client_path and connection_verified
    draft = '{"db_client": {"orm": "Prisma"}}'
    tc = [_wf("mission_90/backend/src/lib/prisma.ts")]
    enriched, changed = backfill_artifact_fields(
        draft, SCHEMA_B, tc, ["mission_90/backend/"])
    assert changed
    assert "prisma.ts" in enriched                 # client_path filled
    assert "connection_verified" not in enriched   # verification NEVER derived
    ok, _ = validate_artifact_schema(enriched, SCHEMA_B)
    assert not ok  # still fails — genuine verification remains the coder's job


def test_shell_tool_and_disk_migration_backfill():
    draft = "# Lock file"
    tc = [_sh("cd mission_90/backend && npx prisma migrate dev --name init")]
    disk = ["mission_90/backend/prisma/migrations/20260801_init/migration.sql"]
    enriched, changed = backfill_artifact_fields(
        draft, SCHEMA_C, tc,
        ["mission_90/backend/prisma/migrations/migration_lock.toml"],
        disk_paths=disk)
    assert changed
    ok, err = validate_artifact_schema(enriched, SCHEMA_C)
    assert ok, err
    assert '"tool": "prisma"' in enriched


def test_no_evidence_no_change():
    enriched, changed = backfill_artifact_fields(
        '{"schema_authored": {}}', SCHEMA_A, [], ["mission_90/backend/"])
    assert not changed


def test_valid_draft_is_noop():
    draft = ('{"schema_authored": {"schema_path": "mission_90/backend/prisma/'
             'schema.prisma", "datasource": "sqlite"}}')
    tc = [_wf("mission_90/backend/prisma/schema.prisma")]
    enriched, changed = backfill_artifact_fields(
        draft, SCHEMA_A, tc, ["mission_90/backend/"])
    assert not changed  # already-valid output is never mutated


def test_count_field_not_treated_as_path():
    # 13.6 migrations_run is a count/list, NOT a path — must not be filled.
    schema = {"result": {"type": "object",
              "required_fields": ["migrations_run", "success"],
              "must_be_true": ["success"]}}
    tc = [_sh("npx prisma migrate deploy")]
    enriched, changed = backfill_artifact_fields(
        '{"result": {}}', schema, tc, ["mission_90/backend/"],
        disk_paths=["mission_90/backend/prisma/migrations/x/migration.sql"])
    assert "migrations_run" not in enriched  # not path-matched
