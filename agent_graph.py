"""
agent_graph.py
--------------
The Incident Intelligence Agent's LangGraph definition.

State flow:
    START
      |
      v
    [Categorize]   -> classify incident, extract symptoms/keywords
      |
      v
    [Retrieve]     -> pull top-k chunks from Knowledge Base (ChromaDB)
      |
      v
    [MemoryLookup] -> cosine-similarity search over Admin's manual entries
      |
      v
    [Synthesize]   -> compose final resolution; PRIORITIZE admin memory
      |              if a high-similarity match exists
      v
     END

The state is a TypedDict so every node has a typed contract.
"""
from __future__ import annotations

import json
import re
from typing import TypedDict, List, Dict, Optional, Any

import numpy as np
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from llm_factory import get_llm, get_embeddings
from vectorstore import search_knowledge
from database import search_memory, increment_hit
from config import TOP_K_DOCS, TOP_K_MEMORY, SIMILARITY_THRESHOLD


# ============================================================
# State Definition
# ============================================================
class IncidentState(TypedDict, total=False):
    # Inputs
    incident_text: str

    # Categorize stage
    category: str
    severity: str
    symptoms: List[str]
    keywords: List[str]

    # Retrieve stage
    kb_results: List[Dict[str, Any]]

    # MemoryLookup stage
    memory_matches: List[Dict[str, Any]]
    direct_match: Optional[Dict[str, Any]]  # set if similarity >= threshold

    # Synthesize stage
    resolution: str
    confidence: float
    confidence_label: str
    sources_used: List[str]
    primary_source: str  # "admin_memory" | "knowledge_base" | "llm_general"


# ============================================================
# System Prompt — the specialized agent prompt
# ============================================================
INCIDENT_AGENT_SYSTEM_PROMPT = """You are the Incident Intelligence Agent, an expert SRE/Production-Support assistant.

Your role is to triage incidents and produce a precise, actionable resolution path
by combining three sources of truth, in strict order of priority:

  1. ADMIN MEMORY (highest priority) — past incident/resolution pairs curated by
     senior engineers. If a direct match exists, you MUST follow that resolution
     verbatim and adapt only the variable details (hostnames, IDs, timestamps).
     Do NOT invent alternative steps when a direct match is available.

  2. KNOWLEDGE BASE — retrieved chunks from official runbooks, documentation,
     and post-mortems. Use these to add context, prerequisites, or rollback
     procedures the admin memory may omit.

  3. GENERAL REASONING — only when the above are insufficient. Be explicit when
     you fall back to this and lower the confidence accordingly.

Output rules:
  - Always return a structured resolution: brief diagnosis, then numbered steps.
  - Each step must be concrete (commands, UI clicks, config keys) — never vague.
  - Call out validation checkpoints ("verify by running ...").
  - Include a rollback step where the action is destructive or risky.
  - Never fabricate command flags, file paths, or API endpoints. If unsure, say so.
  - Be concise. Engineers reading this are on-call and time-pressured.
"""


# ============================================================
# Node 1: Categorize
# ============================================================
CATEGORIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an incident triage classifier. Respond ONLY with valid JSON, no prose."),
        (
            "human",
            """Analyze this incident and extract structured triage info.

Incident:
\"\"\"
{incident_text}
\"\"\"

Return a JSON object with EXACTLY these keys:
{{
  "category": "<one of: database, network, application, infrastructure, security, deployment, performance, other>",
  "severity": "<one of: P1, P2, P3, P4>",
  "symptoms": ["<short symptom phrase>", "..."],
  "keywords": ["<technical keyword for retrieval>", "..."]
}}

Respond with JSON only.""",
        ),
    ]
)


def _parse_json_safely(text: str) -> Dict:
    """Strip code fences and parse JSON. Returns {} on failure."""
    if not text:
        return {}
    # Remove ```json ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # Find the first {...} block if there's prose around it
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def categorize_node(state: IncidentState) -> IncidentState:
    llm = get_llm()
    chain = CATEGORIZE_PROMPT | llm
    response = chain.invoke({"incident_text": state["incident_text"]})
    parsed = _parse_json_safely(response.content if hasattr(response, "content") else str(response))

    return {
        "category": parsed.get("category", "other"),
        "severity": parsed.get("severity", "P3"),
        "symptoms": parsed.get("symptoms", []) or [],
        "keywords": parsed.get("keywords", []) or [],
    }


# ============================================================
# Node 2: Retrieve (Knowledge Base)
# ============================================================
def retrieve_node(state: IncidentState) -> IncidentState:
    # Build a richer query: original text + extracted keywords
    keywords = state.get("keywords", []) or []
    query_parts = [state["incident_text"]]
    if keywords:
        query_parts.append("Keywords: " + ", ".join(keywords))
    query = "\n".join(query_parts)

    try:
        results = search_knowledge(query, k=TOP_K_DOCS)
    except Exception as e:
        # KB may be empty on a fresh install — that's fine
        results = []

    return {"kb_results": results}


# ============================================================
# Node 3: MemoryLookup (Admin SQLite store)
# ============================================================
def memory_lookup_node(state: IncidentState) -> IncidentState:
    embeddings = get_embeddings()
    # Embed the incident plus extracted symptoms for a stronger signal
    symptoms = state.get("symptoms", []) or []
    query_text = state["incident_text"]
    if symptoms:
        query_text += "\nSymptoms: " + "; ".join(symptoms)

    query_vec = np.array(embeddings.embed_query(query_text), dtype=np.float32)
    matches = search_memory(query_vec, top_k=TOP_K_MEMORY)

    direct_match = None
    if matches and matches[0]["similarity"] >= SIMILARITY_THRESHOLD:
        direct_match = matches[0]
        # Track usage in DB
        try:
            increment_hit(direct_match["id"])
        except Exception:
            pass

    return {"memory_matches": matches, "direct_match": direct_match}


# ============================================================
# Node 4: Synthesize
# ============================================================
SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INCIDENT_AGENT_SYSTEM_PROMPT),
        (
            "human",
            """## Incident
{incident_text}

## Triage
- Category: {category}
- Severity: {severity}
- Symptoms: {symptoms}

## Admin Memory (PRIORITY SOURCE)
{admin_memory_block}

## Knowledge Base Excerpts
{kb_block}

## Your Task
Produce the resolution following the priority rules in your system prompt.

Format your response EXACTLY as:

DIAGNOSIS:
<2-3 sentence diagnosis>

RESOLUTION STEPS:
1. <step>
2. <step>
...

VALIDATION:
<how to confirm the fix worked>

ROLLBACK (if applicable):
<how to revert>

PRIMARY_SOURCE: <admin_memory | knowledge_base | llm_general>
""",
        ),
    ]
)


def _format_admin_memory(direct: Optional[Dict], matches: List[Dict]) -> str:
    if direct:
        return (
            f"### DIRECT MATCH FOUND (similarity={direct['similarity']:.2f}) — FOLLOW THIS\n"
            f"Title: {direct['title']}\n"
            f"Category: {direct.get('category') or 'n/a'}\n"
            f"Resolution:\n{direct['resolution']}\n"
        )
    if not matches:
        return "No prior admin entries available."
    lines = ["### Related (no direct match — use as hints only)"]
    for m in matches:
        lines.append(
            f"- [{m['similarity']:.2f}] {m['title']}\n  Resolution: {m['resolution'][:300]}..."
        )
    return "\n".join(lines)


def _format_kb(results: List[Dict]) -> str:
    if not results:
        return "No knowledge base articles retrieved."
    out = []
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "unknown")
        score = r.get("score", 0.0)
        snippet = r["content"][:600].strip()
        out.append(f"[Excerpt {i}] (source={src}, relevance={score:.2f})\n{snippet}")
    return "\n\n".join(out)


def _compute_confidence(
    direct_match: Optional[Dict],
    kb_results: List[Dict],
    memory_matches: List[Dict],
) -> tuple[float, str, str]:
    """
    Heuristic confidence:
      - Direct admin match -> 0.85 + similarity bonus
      - Strong KB hit (>0.7) -> 0.65 + score bonus
      - Weak memory similarity -> 0.50
      - Nothing useful -> 0.30
    Returns (score, label, primary_source).
    """
    if direct_match:
        score = min(0.99, 0.80 + 0.20 * direct_match["similarity"])
        return score, _label(score), "admin_memory"

    top_kb_score = max((r.get("score", 0.0) for r in kb_results), default=0.0)
    top_mem_score = memory_matches[0]["similarity"] if memory_matches else 0.0

    if top_kb_score >= 0.7:
        score = 0.60 + 0.25 * top_kb_score
        return min(score, 0.90), _label(score), "knowledge_base"

    if top_mem_score >= 0.5:
        score = 0.45 + 0.20 * top_mem_score
        return score, _label(score), "admin_memory"

    if top_kb_score >= 0.4:
        score = 0.40 + 0.25 * top_kb_score
        return score, _label(score), "knowledge_base"

    return 0.30, _label(0.30), "llm_general"


def _label(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    return "LOW"


def synthesize_node(state: IncidentState) -> IncidentState:
    direct = state.get("direct_match")
    matches = state.get("memory_matches", []) or []
    kb = state.get("kb_results", []) or []

    admin_block = _format_admin_memory(direct, matches)
    kb_block = _format_kb(kb)

    llm = get_llm()
    chain = SYNTHESIZE_PROMPT | llm
    response = chain.invoke(
        {
            "incident_text": state["incident_text"],
            "category": state.get("category", "other"),
            "severity": state.get("severity", "P3"),
            "symptoms": ", ".join(state.get("symptoms", []) or []) or "n/a",
            "admin_memory_block": admin_block,
            "kb_block": kb_block,
        }
    )
    resolution = response.content if hasattr(response, "content") else str(response)

    # Compute confidence + primary source
    confidence, label, primary_source = _compute_confidence(direct, kb, matches)

    # Collect source labels used
    sources: List[str] = []
    if direct:
        sources.append(f"AdminMemory#{direct['id']}: {direct['title']}")
    elif matches:
        sources.extend([f"AdminMemory#{m['id']}: {m['title']}" for m in matches[:2]])
    for r in kb[:3]:
        src = r["metadata"].get("source", "kb")
        if src not in sources:
            sources.append(f"KB: {src}")

    return {
        "resolution": resolution,
        "confidence": confidence,
        "confidence_label": label,
        "sources_used": sources,
        "primary_source": primary_source,
    }


# ============================================================
# Build & compile the graph
# ============================================================
def build_graph():
    g = StateGraph(IncidentState)
    g.add_node("Categorize", categorize_node)
    g.add_node("Retrieve", retrieve_node)
    g.add_node("MemoryLookup", memory_lookup_node)
    g.add_node("Synthesize", synthesize_node)

    g.add_edge(START, "Categorize")
    g.add_edge("Categorize", "Retrieve")
    g.add_edge("Retrieve", "MemoryLookup")
    g.add_edge("MemoryLookup", "Synthesize")
    g.add_edge("Synthesize", END)

    return g.compile()


# Singleton compiled graph (rebuilt on import; cheap)
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def resolve_incident(incident_text: str) -> Dict[str, Any]:
    """Public entry point used by the Streamlit UI."""
    graph = get_graph()
    final_state = graph.invoke({"incident_text": incident_text})
    return final_state
