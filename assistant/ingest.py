"""Load each tenant's markdown knowledge base into SQLite.

Layout:  seed/<tenant_id>/_tenant.json   {"name": "Glow Salon"}
         seed/<tenant_id>/*.md

Re-running is cheap: documents whose content hash hasn't changed are skipped,
so only edited files are re-embedded (embedding calls are the slow part).

    python -m assistant.ingest [seed_dir]
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from .db import pack

MAX_CHUNK_CHARS = 900


@dataclass
class Chunk:
    heading: str  # "Services > Hair" — the section path within the document
    text: str


def chunk_markdown(md: str, title: str) -> list[Chunk]:
    """Split on headings so each chunk is one coherent section; split long
    sections on paragraph boundaries with one paragraph of overlap."""
    path: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    body: list[str] = []

    def flush():
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(path) or title, text))
        body.clear()

    for line in md.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            body.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, text in sections:
        if len(text) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(heading, text))
            continue
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        current: list[str] = []
        for p in paras:
            if current and len("\n\n".join(current + [p])) > MAX_CHUNK_CHARS:
                chunks.append(Chunk(heading, "\n\n".join(current)))
                current = current[-1:]  # overlap keeps context across the split
            current.append(p)
        if current:
            chunks.append(Chunk(heading, "\n\n".join(current)))
    return chunks


def embedding_text(business: str, heading: str, text: str) -> str:
    """Contextual header: a chunk like '₹500' is meaningless alone; prefixing the
    business and section makes it retrievable for 'how much is a haircut at Glow'."""
    return f"Business: {business}\nSection: {heading}\n\n{text}"


def upsert_tenant(conn: sqlite3.Connection, tenant_id: str, name: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO tenants(id, name) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name",
            (tenant_id, name),
        )


def ingest_document(conn: sqlite3.Connection, provider, tenant_id: str, source: str, title: str, content: str) -> int | None:
    """Index one document. Returns the chunk count, or None if the content is unchanged
    (so re-syncing an unedited document costs no embedding calls)."""
    business = conn.execute("SELECT name FROM tenants WHERE id = ?", (tenant_id,)).fetchone()["name"]
    digest = hashlib.sha256(f"{title}\n{content}".encode()).hexdigest()
    row = conn.execute(
        "SELECT id, content_hash FROM documents WHERE tenant_id = ? AND source = ?", (tenant_id, source)
    ).fetchone()
    if row and row["content_hash"] == digest:
        return None

    chunks = chunk_markdown(content, title=title)
    vectors = provider.embed([embedding_text(business, c.heading, c.text) for c in chunks], "document") if chunks else []

    with conn:  # one transaction per document: never leave half-indexed docs
        if row:
            _delete_document(conn, row["id"])
        doc_id = conn.execute(
            "INSERT INTO documents(tenant_id, source, content_hash) VALUES (?, ?, ?)",
            (tenant_id, source, digest),
        ).lastrowid
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            chunk_id = conn.execute(
                "INSERT INTO chunks(document_id, tenant_id, ord, heading, text, embedding) VALUES (?,?,?,?,?,?)",
                (doc_id, tenant_id, i, c.heading, c.text, pack(v)),
            ).lastrowid
            conn.execute(
                "INSERT INTO chunks_fts(rowid, heading, text, tenant_id) VALUES (?,?,?,?)",
                (chunk_id, c.heading, c.text, tenant_id),
            )
    return len(chunks)


def delete_document(conn: sqlite3.Connection, tenant_id: str, source: str) -> bool:
    row = conn.execute("SELECT id FROM documents WHERE tenant_id = ? AND source = ?", (tenant_id, source)).fetchone()
    if not row:
        return False
    with conn:
        _delete_document(conn, row["id"])
    return True


def delete_tenant(conn: sqlite3.Connection, tenant_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM chunks_fts WHERE tenant_id = ?", (tenant_id,))
        conn.execute("DELETE FROM chunks WHERE tenant_id = ?", (tenant_id,))
        conn.execute("DELETE FROM documents WHERE tenant_id = ?", (tenant_id,))
        conn.execute("DELETE FROM queries WHERE tenant_id = ?", (tenant_id,))
        conn.execute("DELETE FROM tenants WHERE id = ?", (tenant_id,))


def clone_tenant(conn: sqlite3.Connection, source_id: str, target_id: str, name: str, rename: dict | None = None) -> int:
    """Copy a tenant's documents, chunks and embeddings — no embedding calls.
    Used to give every demo sandbox a ready-made knowledge base. `rename` maps
    source document keys to the caller's own ids, so later edits replace them."""
    rename = rename or {}
    delete_tenant(conn, target_id)
    upsert_tenant(conn, target_id, name)
    copied = 0
    with conn:
        for doc in conn.execute("SELECT id, source, content_hash FROM documents WHERE tenant_id = ?", (source_id,)).fetchall():
            new_doc = conn.execute(
                "INSERT INTO documents(tenant_id, source, content_hash) VALUES (?, ?, ?)",
                (target_id, rename.get(doc["source"], doc["source"]), doc["content_hash"]),
            ).lastrowid
            for c in conn.execute("SELECT ord, heading, text, embedding FROM chunks WHERE document_id = ?", (doc["id"],)).fetchall():
                cid = conn.execute(
                    "INSERT INTO chunks(document_id, tenant_id, ord, heading, text, embedding) VALUES (?,?,?,?,?,?)",
                    (new_doc, target_id, c["ord"], c["heading"], c["text"], c["embedding"]),
                ).lastrowid
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, heading, text, tenant_id) VALUES (?,?,?,?)",
                    (cid, c["heading"], c["text"], target_id),
                )
                copied += 1
    return copied


def ingest_tenant(conn: sqlite3.Connection, provider, tenant_dir: Path) -> dict:
    tenant_id = tenant_dir.name
    meta = json.loads((tenant_dir / "_tenant.json").read_text(encoding="utf-8"))
    upsert_tenant(conn, tenant_id, meta["name"])
    stats = {"tenant": tenant_id, "updated": 0, "skipped": 0, "removed": 0, "chunks": 0}
    seen = set()

    for md_path in sorted(tenant_dir.glob("*.md")):
        seen.add(md_path.name)
        n = ingest_document(
            conn, provider, tenant_id, md_path.name,
            title=md_path.stem.replace("-", " ").title(), content=md_path.read_text(encoding="utf-8"),
        )
        if n is None:
            stats["skipped"] += 1
        else:
            stats["updated"] += 1
            stats["chunks"] += n

    # Files deleted from disk should disappear from the index too.
    for row in conn.execute("SELECT source FROM documents WHERE tenant_id = ?", (tenant_id,)).fetchall():
        if row["source"] not in seen:
            delete_document(conn, tenant_id, row["source"])
            stats["removed"] += 1
    return stats


def _delete_document(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id = ?)", (doc_id,))
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def ingest_all(conn: sqlite3.Connection, provider, seed_dir: Path) -> list[dict]:
    return [ingest_tenant(conn, provider, d) for d in sorted(seed_dir.iterdir()) if (d / "_tenant.json").exists()]


if __name__ == "__main__":
    from .config import settings
    from .db import connect
    from .llm import GeminiProvider

    seed = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.seed_dir
    for s in ingest_all(connect(settings.db_path), GeminiProvider(settings), seed):
        print(f"{s['tenant']}: {s['updated']} updated ({s['chunks']} chunks), {s['skipped']} unchanged, {s['removed']} removed")
