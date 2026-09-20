CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_chunks (
    collection_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document TEXT NOT NULL,
    version TEXT NOT NULL,
    section TEXT NOT NULL,
    section_title TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR NOT NULL,
    PRIMARY KEY (collection_name, chunk_id)
);
