-- Default smoke SELECTs run after sqlite backup restore.
-- The executor will substitute these with EXPECT_TABLES if provided.
SELECT count(*) AS table_count FROM sqlite_master WHERE type='table';
