# 🛠️ Incident Intelligence Agent

A stateful LangGraph agent that automates incident triage and resolution by combining:

- **Admin memory** (SQLite) — your manually curated past incident/resolution pairs (**top priority**)
- **Knowledge base** (ChromaDB) — uploaded runbooks, post-mortems, vendor docs
- **LLM general reasoning** (fallback)

## Architecture

```
              ┌──────────────┐
   Incident ─▶│  Categorize  │  ─ extract category, severity, symptoms, keywords
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │   Retrieve   │  ─ ChromaDB similarity search over uploaded KB
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ MemoryLookup │  ─ cosine search over admin's SQLite memory
              └──────┬───────┘     (sim ≥ 0.75 = direct match → priority)
                     ▼
              ┌──────────────┐
              │  Synthesize  │  ─ LLM composes final resolution +
              └──────┬───────┘     confidence score + primary-source label
                     ▼
                  Resolution
```

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | **LangGraph** (`StateGraph` with 4 nodes) |
| LLM | **Groq** (free, Llama 3.3 70B) — or **Ollama** local fallback |
| Embeddings | **HuggingFace** `all-MiniLM-L6-v2` (local, free, no API key) |
| Vector store | **ChromaDB** (persistent, local) |
| Memory store | **SQLite** with embedded vectors as BLOBs |
| UI | **Streamlit** (3-tab admin interface) |

## Setup

### 1. Clone & create venv

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and pick **one** of:

**Option A — Groq (recommended, free, fast):**
1. Sign up at https://console.groq.com
2. Create an API key
3. Set `LLM_PROVIDER=groq` and `GROQ_API_KEY=gsk_...`

**Option B — Ollama (fully local, no cloud):**
1. Install Ollama: https://ollama.com/download
2. `ollama pull llama3.1:8b`
3. `ollama serve` (in a separate terminal)
4. Set `LLM_PROVIDER=ollama` in `.env`

### 3. (Optional) Smoke-test the graph without UI

```bash
python test_agent.py
```

This seeds one demo memory and runs a sample incident through the full graph.
First run downloads the embedding model (~80MB) — be patient.

### 4. Launch the UI

```bash
streamlit run app.py
```

Open http://localhost:8501

## Using the UI

### Tab 1 — Training / Feeding
Add incident/resolution pairs from past on-call work. Title + resolution are required.
These are embedded immediately and become the **highest-priority** retrieval source.

### Tab 2 — Knowledge Upload
Drop in PDFs, Markdown, or TXT files. They're chunked (800 chars, 100 overlap),
embedded with sentence-transformers, and persisted to ChromaDB.

### Tab 3 — Incident Solver
Paste a raw incident log/alert. The agent:
1. Categorizes it
2. Retrieves top-k KB chunks
3. Looks up similar admin memories (and flags direct matches ≥ 0.75 similarity)
4. Synthesizes a structured resolution

You get back:
- **Confidence score** (HIGH/MEDIUM/LOW) + percentage
- **Primary source** label (🥇 admin memory / 🥈 KB / 🥉 LLM)
- **Resolution path** (diagnosis, numbered steps, validation, rollback)
- **Diagnostic panel** showing all retrieval hits and similarity scores

## File layout

```
incident_agent/
├── app.py              # Streamlit UI (3 tabs)
├── agent_graph.py      # LangGraph: state + 4 nodes + compile
├── llm_factory.py      # LLM & embedding singletons
├── vectorstore.py      # ChromaDB wrapper (ingest, search, list, delete)
├── database.py         # SQLite admin memory (insert, search, list, delete)
├── config.py           # All env-driven settings
├── test_agent.py       # End-to-end smoke test (no UI)
├── requirements.txt
├── .env.example
└── data/               # auto-created
    ├── chroma_db/      # vector store
    ├── incidents.db    # SQLite memory
    └── uploads/        # raw uploaded files
```

## How the priority rule works

The `Synthesize` node receives:
- A formatted `admin_memory_block` that flags any similarity ≥ 0.75 as **DIRECT MATCH FOUND — FOLLOW THIS**
- The system prompt explicitly orders: Admin Memory > KB > General reasoning

Confidence is computed by `_compute_confidence`:
- Direct admin match → 0.80–0.99 (HIGH)
- Strong KB hit (≥ 0.7) → 0.60–0.90 (MEDIUM/HIGH)
- Weak memory + weak KB → 0.30–0.45 (LOW)

Tune `SIMILARITY_THRESHOLD` in `config.py` to make direct matches stricter or looser.

## Troubleshooting

**`GROQ_API_KEY is not set`** — fill it in `.env` or switch to `LLM_PROVIDER=ollama`.

**First run is slow** — the embedding model downloads on first use. Subsequent runs are cached.

**`unstructured` errors on `.md` files** — already handled; falls back to plain `TextLoader`.

**Ollama connection refused** — make sure `ollama serve` is running and the model is pulled.

**ChromaDB locking issues** — close other processes touching `data/chroma_db/`.
