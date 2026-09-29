"""Hybrid retrieval: BM25 keyword search (FTS5) + embedding similarity,
merged with Reciprocal Rank Fusion.

Why both: embeddings miss exact tokens ("GSTIN", "Keratin", phone numbers),
keyword search misses paraphrases ("how late are you open" vs "Hours").
RRF merges the two rankings without having to calibrate their scores.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass, field

from .db import unpack

STOPWORDS = set(
    "a an and are at be can do does for from have how i in is it me my of on or our please the to "
    "we what when where which who will with you your".split()
)


@dataclass
class Hit:
    chunk_id: int
    source: str
    heading: str
    text: str
    score: float = 0.0  # fused RRF score
    vector_sim: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None


@dataclass
class RetrievalResult:
    hits: list[Hit]
    best_similarity: float
    keyword_matches: int
    debug: dict = field(default_factory=dict)


class VectorIndex:
    """In-memory copy of a tenant's embeddings, reloaded only when the tenant's
    chunks change. Avoids unpacking every blob on every request."""

    def __init__(self):
        self._cache: dict[str, tuple[tuple, list[tuple[int, object]]]] = {}
        self._lock = threading.Lock()

    def get(self, conn: sqlite3.Connection, tenant_id: str):
        version = tuple(conn.execute("SELECT COUNT(*), MAX(id) FROM chunks WHERE tenant_id = ?", (tenant_id,)).fetchone())
        with self._lock:
            cached = self._cache.get(tenant_id)
            if cached and cached[0] == version:
                return cached[1]
        rows = conn.execute("SELECT id, embedding FROM chunks WHERE tenant_id = ?", (tenant_id,)).fetchall()
        vectors = [(r["id"], unpack(r["embedding"])) for r in rows]
        with self._lock:
            self._cache[tenant_id] = (version, vectors)
        return vectors


def fts_query(question: str) -> str | None:
    """Turn free text into a safe FTS5 query: quoted terms OR'd together.
    Quoting stops user input from being parsed as FTS syntax (AND, NEAR, *, ...)."""
    terms = [t for t in re.findall(r"\w+", question.lower()) if t not in STOPWORDS and len(t) > 1]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


def keyword_search(conn: sqlite3.Connection, tenant_id: str, question: str, k: int) -> list[int]:
    q = fts_query(question)
    if not q:
        return []
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? AND tenant_id = ? ORDER BY bm25(chunks_fts, 2.0, 1.0) LIMIT ?",
        (q, tenant_id, k),
    ).fetchall()
    return [r[0] for r in rows]


def vector_search(vectors, query_vec: list[float], k: int) -> list[tuple[int, float]]:
    # Vectors are unit length, so dot product == cosine similarity.
    scored = [(cid, sum(a * b for a, b in zip(vec, query_vec))) for cid, vec in vectors]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def rrf(rankings: list[list[int]], k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve(
    conn: sqlite3.Connection,
    index: VectorIndex,
    provider,
    tenant_id: str,
    question: str,
    top_k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
    mode: str = "hybrid",  # "hybrid" | "vector" | "keyword" — lets evals compare strategies
) -> RetrievalResult:
    vec_ranked: list[tuple[int, float]] = []
    kw_ranked: list[int] = []

    if mode in ("hybrid", "vector"):
        query_vec = provider.embed([question], "query")[0]
        vec_ranked = vector_search(index.get(conn, tenant_id), query_vec, candidate_k)
    if mode in ("hybrid", "keyword"):
        kw_ranked = keyword_search(conn, tenant_id, question, candidate_k)

    fused = rrf([[cid for cid, _ in vec_ranked], kw_ranked], rrf_k)
    top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]

    sims = dict(vec_ranked)
    vec_pos = {cid: i + 1 for i, (cid, _) in enumerate(vec_ranked)}
    kw_pos = {cid: i + 1 for i, cid in enumerate(kw_ranked)}

    hits: list[Hit] = []
    if top_ids:
        placeholders = ",".join("?" * len(top_ids))
        rows = conn.execute(
            f"""SELECT c.id, c.heading, c.text, d.source FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders}) AND c.tenant_id = ?""",
            (*top_ids, tenant_id),
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        for cid in top_ids:
            r = by_id.get(cid)
            if r is None:
                continue
            hits.append(
                Hit(cid, r["source"], r["heading"], r["text"], fused[cid], sims.get(cid), vec_pos.get(cid), kw_pos.get(cid))
            )

    return RetrievalResult(
        hits=hits,
        best_similarity=max(sims.values(), default=0.0),
        keyword_matches=len(kw_ranked),
    )
