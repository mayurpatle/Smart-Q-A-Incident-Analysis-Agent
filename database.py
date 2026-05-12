"""
database.py
-----------
SQLite-backed persistent memory for Admin-fed incident/resolution pairs.
This is the "past experience" store that the MemoryLookup node queries.

Uses sentence-transformer embeddings stored as BLOBs alongside the rows
so we can compute cosine similarity for fuzzy matching (not just keyword).
"""
from __future__ import annotations

import sqlite3
import json
import io
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

from config import SQLITE_DB_PATH


def _np_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32), allow_pickle=False)
    return buf.getvalue()


def _blob_to_np(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the admin_memory table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                resolution TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                embedding BLOB,
                created_at TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
            """
        )
        conn.commit()


def insert_memory(
    title: str,
    resolution: str,
    category: Optional[str],
    tags: Optional[List[str]],
    embedding: np.ndarray,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO admin_memory (title, resolution, category, tags, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                title.strip(),
                resolution.strip(),
                (category or "").strip() or None,
                json.dumps(tags or []),
                _np_to_blob(embedding),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid


def list_all_memories() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, resolution, category, tags, created_at, hit_count "
            "FROM admin_memory ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "resolution": r["resolution"],
            "category": r["category"],
            "tags": json.loads(r["tags"]) if r["tags"] else [],
            "created_at": r["created_at"],
            "hit_count": r["hit_count"],
        }
        for r in rows
    ]


def delete_memory(memory_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM admin_memory WHERE id = ?", (memory_id,))
        conn.commit()


def search_memory(
    query_embedding: np.ndarray, top_k: int = 3
) -> List[Dict]:
    """
    Cosine-similarity search over admin memory embeddings.
    Returns rows sorted by similarity DESC with a 'similarity' field [0,1].
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, resolution, category, tags, embedding FROM admin_memory"
        ).fetchall()

    if not rows:
        return []

    q = query_embedding.astype(np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)

    scored = []
    for r in rows:
        emb = _blob_to_np(r["embedding"])
        e_norm = emb / (np.linalg.norm(emb) + 1e-9)
        sim = float(np.dot(q_norm, e_norm))
        scored.append(
            {
                "id": r["id"],
                "title": r["title"],
                "resolution": r["resolution"],
                "category": r["category"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "similarity": sim,
            }
        )

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def increment_hit(memory_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE admin_memory SET hit_count = hit_count + 1 WHERE id = ?",
            (memory_id,),
        )
        conn.commit()
