-- Default smoke SELECTs after pg_restore. Replace with mission-specific checks.
SELECT count(*) AS table_count FROM information_schema.tables WHERE table_schema='public';
