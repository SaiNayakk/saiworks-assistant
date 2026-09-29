"""SQLite storage: one file, no server — suits a phone.

Keyword search uses the built-in FTS5 index; embeddings live in the chunks
table as packed float32 blobs and are searched in Python (fine for the few
hundred chunks a small business has).
"""

from __future__ import annotations

import sqlite3
from array import array
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    tenant_id    TEXT NOT NULL REFERENCES tenants(id),
    source       TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (tenant_id, source)
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL,
    ord         INTEGER NOT NULL,
    heading     TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_tenant ON chunks(tenant_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading, text, tenant_id UNINDEXED, tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS queries (
    id            INTEGER PRIMARY KEY,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    tenant_id     TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    answerable    INTEGER NOT NULL,
    abstain_reason TEXT,
    retrieved_ids TEXT NOT NULL,
    cited_ids     TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    retrieve_ms   REAL NOT NULL,
    generate_ms   REAL NOT NULL,
    total_ms      REAL NOT NULL
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def pack(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def unpack(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a
