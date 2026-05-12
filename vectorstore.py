"""
vectorstore.py
--------------
ChromaDB-backed knowledge article store. Handles PDF/Markdown ingestion,
chunking, embedding, and similarity search.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_KB,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_DOCS,
)
from llm_factory import get_embeddings


def get_vectorstore() -> Chroma:
    """Return (and lazily create) the persistent ChromaDB collection."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_KB,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DB_PATH),
    )


def _load_file(file_path: Path) -> List[Document]:
    """Dispatch to the correct loader based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if suffix in {".md", ".markdown"}:
        try:
            return UnstructuredMarkdownLoader(str(file_path)).load()
        except Exception:
            # Fallback: load as plain text if unstructured deps are missing
            return TextLoader(str(file_path), encoding="utf-8").load()
    if suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(str(file_path)).load()
    if suffix in {".txt"}:
        return TextLoader(str(file_path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}")


def ingest_file(file_path: Path, source_label: str | None = None) -> Dict:
    """
    Load -> chunk -> embed -> persist.
    Returns ingestion stats.
    """
    docs = _load_file(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Annotate with source + a stable doc_id per upload
    doc_id = str(uuid.uuid4())[:8]
    label = source_label or file_path.name
    for i, ch in enumerate(chunks):
        ch.metadata = {
            **(ch.metadata or {}),
            "source": label,
            "doc_id": doc_id,
            "chunk_index": i,
        }

    vs = get_vectorstore()
    vs.add_documents(chunks)
    return {
        "source": label,
        "doc_id": doc_id,
        "chunks": len(chunks),
        "pages": len(docs),
    }


def search_knowledge(query: str, k: int = TOP_K_DOCS) -> List[Dict]:
    """
    Similarity search against the knowledge base.
    Returns list of {content, metadata, score} dicts.
    Score is similarity in [0,1] (higher = more similar).
    """
    vs = get_vectorstore()
    # Chroma returns "distance" (lower = better). Convert to similarity.
    results = vs.similarity_search_with_relevance_scores(query, k=k)
    out = []
    for doc, score in results:
        out.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
        )
    return out


def list_sources() -> List[Dict]:
    """Group the KB by source file and return counts."""
    vs = get_vectorstore()
    # Pull all metadata; for a small/local KB this is fine.
    try:
        data = vs.get()
    except Exception:
        return []
    metas = data.get("metadatas") or []
    agg: Dict[str, Dict] = {}
    for m in metas:
        src = (m or {}).get("source", "unknown")
        doc_id = (m or {}).get("doc_id", "")
        key = f"{src}::{doc_id}"
        if key not in agg:
            agg[key] = {"source": src, "doc_id": doc_id, "chunks": 0}
        agg[key]["chunks"] += 1
    return sorted(agg.values(), key=lambda x: x["source"])


def delete_source(doc_id: str) -> int:
    """Delete all chunks belonging to a given doc_id. Returns count deleted."""
    vs = get_vectorstore()
    data = vs.get(where={"doc_id": doc_id})
    ids = data.get("ids") or []
    if ids:
        vs.delete(ids=ids)
    return len(ids)
