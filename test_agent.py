"""
test_agent.py
-------------
Quick smoke test to validate the LangGraph pipeline end-to-end
WITHOUT spinning up Streamlit. Run after `pip install -r requirements.txt`
and configuring .env.

    python test_agent.py
"""
import numpy as np

from database import init_db, insert_memory, list_all_memories
from llm_factory import get_embeddings
from agent_graph import resolve_incident


def seed_demo_memory():
    """Insert one admin memory so MemoryLookup has something to match."""
    if list_all_memories():
        return  # already seeded
    embeddings = get_embeddings()
    title = "Payments API 503s with DB connection pool exhausted"
    resolution = (
        "1. Check RDS active connections: `SELECT count(*) FROM pg_stat_activity;`\n"
        "2. If > 90% of max_connections, scale read replicas or bump pool_max.\n"
        "3. Roll back the last deploy if it changed pool settings.\n"
        "4. Add HikariCP leak detection: `spring.datasource.hikari.leak-detection-threshold=30000`.\n"
        "5. Validate by tailing /v1/charge — 5xx should drop within 3 min."
    )
    vec = np.array(embeddings.embed_query(f"{title}\n\n{resolution}"), dtype=np.float32)
    insert_memory(
        title=title,
        resolution=resolution,
        category="database",
        tags=["postgres", "hikari", "connection-pool"],
        embedding=vec,
    )
    print("✅ Seeded one demo admin memory.")


def main():
    init_db()
    seed_demo_memory()

    incident = (
        "2026-05-12 09:14 UTC ALERT: PagerDuty P2 — payments-api error rate 14% (>5% SLO).\n"
        "Symptoms: HTTP 503 from /v1/charge, upstream error 'database connection pool exhausted'.\n"
        "Last deploy: 90 min ago. RDS CPU 92%, active connections 198/200."
    )

    print("\n" + "=" * 70)
    print("INCIDENT:")
    print(incident)
    print("=" * 70 + "\n")

    print("🧠 Running graph...\n")
    result = resolve_incident(incident)

    print(f"Category:        {result.get('category')}")
    print(f"Severity:        {result.get('severity')}")
    print(f"Symptoms:        {result.get('symptoms')}")
    print(f"Keywords:        {result.get('keywords')}")
    print(f"Confidence:      {result.get('confidence'):.0%} ({result.get('confidence_label')})")
    print(f"Primary Source:  {result.get('primary_source')}")
    direct = result.get("direct_match")
    if direct:
        print(f"Direct Match:    #{direct['id']} '{direct['title']}' (sim={direct['similarity']:.2f})")
    print("\n--- RESOLUTION ---")
    print(result.get("resolution"))
    print("\n--- SOURCES USED ---")
    for s in result.get("sources_used", []):
        print(f"  - {s}")


if __name__ == "__main__":
    main()
