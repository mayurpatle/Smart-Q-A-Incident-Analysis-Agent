"""
config.py
---------
Centralized configuration loader. All environment-driven settings live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- LLM ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---- Embeddings ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", DATA_DIR / "chroma_db"))
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", DATA_DIR / "incidents.db"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))

# Ensure directories exist
for p in (DATA_DIR, CHROMA_DB_PATH, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---- Retrieval params ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_DOCS = 4
TOP_K_MEMORY = 3
SIMILARITY_THRESHOLD = 0.75  # for memory "direct match" prioritization

# ---- Collection name ----
CHROMA_COLLECTION_KB = "knowledge_articles"
