"""
app.py
------
Streamlit multi-tab UI for the Incident Intelligence Agent.

Tabs:
  1. Training / Feeding  — admin enters incident -> resolution pairs
  2. Knowledge Upload    — upload PDFs / Markdown into ChromaDB
  3. Incident Solver     — paste a raw incident log; get the resolution

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st

from config import UPLOAD_DIR, LLM_PROVIDER, GROQ_MODEL, OLLAMA_MODEL
from database import (
    init_db,
    insert_memory,
    list_all_memories,
    delete_memory,
)
from vectorstore import ingest_file, list_sources, delete_source
from llm_factory import get_embeddings
from agent_graph import resolve_incident


# ============================================================
# Page config & one-time init
# ============================================================
st.set_page_config(
    page_title="Incident Intelligence Agent",
    page_icon="🛠️",
    layout="wide",
)

init_db()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🛠️ Incident Agent")
    st.caption("LangGraph + RAG + Admin Memory")

    st.divider()
    st.subheader("⚙️ Active Config")
    st.write(f"**LLM provider:** `{LLM_PROVIDER}`")
    if LLM_PROVIDER == "groq":
        st.write(f"**Model:** `{GROQ_MODEL}`")
    else:
        st.write(f"**Model:** `{OLLAMA_MODEL}`")
    st.write("**Embeddings:** `all-MiniLM-L6-v2` (local)")

    st.divider()
    st.subheader("📊 Stats")
    memories = list_all_memories()
    sources = list_sources()
    st.metric("Admin Memories", len(memories))
    st.metric("KB Sources", len(sources))
    st.metric("KB Chunks", sum(s["chunks"] for s in sources))

    st.divider()
    st.caption(
        "Priority order:\n"
        "1. 🥇 Admin Memory (direct match)\n"
        "2. 🥈 Knowledge Base\n"
        "3. 🥉 LLM general reasoning"
    )


# ============================================================
# Tabs
# ============================================================
tab_train, tab_upload, tab_solve = st.tabs(
    ["📝 1. Training / Feeding", "📚 2. Knowledge Upload", "🔍 3. Incident Solver"]
)


# ------------------------------------------------------------
# Tab 1: Training / Feeding
# ------------------------------------------------------------
with tab_train:
    st.header("Feed the Agent's Memory")
    st.write(
        "Add incident → resolution pairs from your past on-call experience. "
        "These take **priority** over knowledge-base articles when a direct match is found."
    )

    with st.form("memory_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            title = st.text_input(
                "Incident Title *",
                placeholder="e.g., Kafka consumer lag spike on order-events topic",
            )
        with col2:
            category = st.selectbox(
                "Category",
                ["", "database", "network", "application", "infrastructure",
                 "security", "deployment", "performance", "other"],
                index=0,
            )

        tags_str = st.text_input(
            "Tags (comma-separated)",
            placeholder="kafka, consumer-lag, partition-rebalance",
        )

        resolution = st.text_area(
            "Resolution Steps *",
            height=220,
            placeholder=(
                "1. Check consumer group state: `kafka-consumer-groups.sh --bootstrap-server ... --describe`\n"
                "2. Identify partitions with growing lag.\n"
                "3. Scale consumer pods: `kubectl scale deploy/order-consumer --replicas=6`\n"
                "4. If rebalance storm: increase `session.timeout.ms` to 30s.\n"
                "5. Validate: lag should drop below 1000 within 5 min."
            ),
        )

        submitted = st.form_submit_button("💾 Save to Memory", type="primary")

        if submitted:
            if not title.strip() or not resolution.strip():
                st.error("Title and Resolution Steps are required.")
            else:
                with st.spinner("Embedding and saving..."):
                    embeddings = get_embeddings()
                    # Embed title + resolution together for better recall
                    embed_text = f"{title}\n\n{resolution}"
                    vec = np.array(embeddings.embed_query(embed_text), dtype=np.float32)
                    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                    new_id = insert_memory(
                        title=title,
                        resolution=resolution,
                        category=category or None,
                        tags=tags,
                        embedding=vec,
                    )
                st.success(f"✅ Saved as memory #{new_id}")
                time.sleep(0.5)
                st.rerun()

    st.divider()
    st.subheader(f"📋 Saved Memories ({len(memories)})")

    if not memories:
        st.info("No memories yet. Add your first one above.")
    else:
        for m in memories:
            with st.expander(
                f"#{m['id']} • {m['title']}  "
                f"·  {m['category'] or 'uncategorized'}  ·  hits: {m['hit_count']}"
            ):
                st.markdown(f"**Created:** {m['created_at']}")
                if m["tags"]:
                    st.markdown("**Tags:** " + " ".join(f"`{t}`" for t in m["tags"]))
                st.markdown("**Resolution:**")
                st.code(m["resolution"], language="text")
                if st.button("🗑️ Delete", key=f"del_mem_{m['id']}"):
                    delete_memory(m["id"])
                    st.rerun()


# ------------------------------------------------------------
# Tab 2: Knowledge Upload
# ------------------------------------------------------------
with tab_upload:
    st.header("Upload Knowledge Articles")
    st.write(
        "Drop in runbooks, post-mortems, or vendor docs (PDF / Markdown / TXT). "
        "They get chunked, embedded, and stored in ChromaDB for the **Retrieve** node."
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 Ingest All", type="primary"):
            progress = st.progress(0.0)
            status = st.empty()
            results = []
            for i, uf in enumerate(uploaded_files):
                status.info(f"Processing **{uf.name}** ...")
                # Save to disk first (loaders expect a path)
                dest = UPLOAD_DIR / uf.name
                dest.write_bytes(uf.getbuffer())
                try:
                    stats = ingest_file(dest, source_label=uf.name)
                    results.append(("✅", uf.name, stats))
                except Exception as e:
                    results.append(("❌", uf.name, {"error": str(e)}))
                progress.progress((i + 1) / len(uploaded_files))

            status.empty()
            progress.empty()

            for icon, name, stats in results:
                if icon == "✅":
                    st.success(
                        f"{icon} **{name}** — {stats['chunks']} chunks from {stats['pages']} pages "
                        f"(doc_id: `{stats['doc_id']}`)"
                    )
                else:
                    st.error(f"{icon} **{name}** — {stats.get('error', 'failed')}")
            time.sleep(0.5)
            st.rerun()

    st.divider()
    st.subheader(f"📂 Ingested Sources ({len(sources)})")

    if not sources:
        st.info("No knowledge articles yet. Upload above to get started.")
    else:
        for s in sources:
            cols = st.columns([4, 1, 1])
            with cols[0]:
                st.markdown(f"**{s['source']}**  ·  `{s['doc_id']}`")
            with cols[1]:
                st.write(f"{s['chunks']} chunks")
            with cols[2]:
                if st.button("🗑️", key=f"del_src_{s['doc_id']}"):
                    n = delete_source(s["doc_id"])
                    st.toast(f"Deleted {n} chunks", icon="🗑️")
                    st.rerun()


# ------------------------------------------------------------
# Tab 3: Incident Solver
# ------------------------------------------------------------
with tab_solve:
    st.header("Resolve an Incident")
    st.write("Paste the raw incident log, alert payload, or symptom description.")

    # Chat-style history (per session)
    if "solver_history" not in st.session_state:
        st.session_state.solver_history = []

    incident_text = st.text_area(
        "Incident description / log",
        height=200,
        placeholder=(
            "2026-05-12 09:14 UTC ALERT: PagerDuty P2 — payments-api error rate 14% (>5% SLO).\n"
            "Symptoms: HTTP 503 from /v1/charge, upstream 'database connection pool exhausted'.\n"
            "Last deploy: 90 min ago. RDS CPU 92%, active connections 198/200."
        ),
    )

    col_a, col_b = st.columns([1, 5])
    with col_a:
        run = st.button("🚀 Resolve", type="primary", disabled=not incident_text.strip())
    with col_b:
        if st.button("🧹 Clear history"):
            st.session_state.solver_history = []
            st.rerun()

    if run and incident_text.strip():
        with st.spinner("🧠 Running graph: Categorize → Retrieve → MemoryLookup → Synthesize..."):
            try:
                result = resolve_incident(incident_text.strip())
                st.session_state.solver_history.insert(0, {
                    "incident": incident_text.strip(),
                    "result": result,
                })
            except Exception as e:
                st.error(f"Agent failed: {e}")
                st.exception(e)

    # Render history
    for i, entry in enumerate(st.session_state.solver_history):
        r = entry["result"]
        confidence = r.get("confidence", 0.0)
        label = r.get("confidence_label", "LOW")
        primary = r.get("primary_source", "llm_general")

        st.divider()

        # Header row: confidence badge + primary source
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(label, "⚪")
            st.metric("Confidence", f"{confidence:.0%}", help=f"{color} {label}")
        with c2:
            src_emoji = {
                "admin_memory": "🥇 Admin Memory",
                "knowledge_base": "🥈 Knowledge Base",
                "llm_general": "🥉 LLM General",
            }
            st.metric("Primary Source", src_emoji.get(primary, primary))
        with c3:
            st.metric("Category", r.get("category", "—"))
        with c4:
            st.metric("Severity", r.get("severity", "—"))

        # The actual resolution
        st.markdown("### 🧩 Resolution Path")
        st.markdown(r.get("resolution", "_no resolution returned_"))

        # Diagnostic detail (collapsed)
        with st.expander("🔬 Diagnostic detail (triage, retrieval hits, memory matches)"):
            st.markdown("**Triage**")
            st.json({
                "category": r.get("category"),
                "severity": r.get("severity"),
                "symptoms": r.get("symptoms", []),
                "keywords": r.get("keywords", []),
            })

            st.markdown("**Admin memory matches**")
            mm = r.get("memory_matches", []) or []
            if mm:
                for m in mm:
                    st.write(
                        f"- [{m['similarity']:.2f}] #{m['id']} {m['title']}"
                    )
            else:
                st.write("_none_")

            direct = r.get("direct_match")
            if direct:
                st.success(
                    f"🎯 Direct match used: #{direct['id']} "
                    f"({direct['similarity']:.2f} ≥ threshold)"
                )

            st.markdown("**Knowledge base hits**")
            kb = r.get("kb_results", []) or []
            if kb:
                for j, k in enumerate(kb, 1):
                    st.write(
                        f"- [{k.get('score', 0):.2f}] {k['metadata'].get('source', '?')} "
                        f"(chunk {k['metadata'].get('chunk_index', '?')})"
                    )
            else:
                st.write("_none_")

            st.markdown("**Sources used in synthesis**")
            for s in r.get("sources_used", []):
                st.write(f"- {s}")

        with st.expander("📝 Original incident"):
            st.code(entry["incident"], language="text")
