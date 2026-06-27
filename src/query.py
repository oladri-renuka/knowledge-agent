"""Query engine: answer questions from the knowledge graph with source citations."""

import logging
from difflib import SequenceMatcher
from src.llm_client import chat_text

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledge graph query engine. You answer questions ONLY from the provided graph context — entities, claims, and relationships.

Rules:
1. Every factual statement in your answer MUST cite the source document in brackets, e.g. [source.pdf].
2. If the graph has NO relevant claims for the question, say exactly: "I don't know — the knowledge graph has no relevant coverage for this question."
3. If there are contradictions or refinements noted, mention them explicitly: "Note: there is a conflict between sources — [doc1] states X while [doc2] states Y."
4. Never infer or hallucinate beyond what the graph claims say. If the graph only partially answers the question, answer what you can and state what's missing.
5. Keep answers concise and factual."""


def _find_relevant_entities(question: str, graph) -> list[str]:
    """Return entities that are likely relevant to the question, checking names and aliases."""
    q_lower = question.lower()
    q_words = set(q_lower.split())

    relevant = []
    for entity in graph.get_all_entities():
        node_data = graph.graph.nodes.get(entity, {})
        aliases = node_data.get("aliases", [])
        all_names = [entity] + aliases

        for name in all_names:
            n_lower = name.lower()
            if n_lower in q_lower:
                relevant.append(entity)
                break
            n_words = set(n_lower.split())
            if q_words & n_words:
                relevant.append(entity)
                break
            if any(SequenceMatcher(None, w, n_lower).ratio() > 0.8 for w in q_words if len(w) > 3):
                relevant.append(entity)
                break

    if not relevant:
        return graph.get_all_entities()

    return relevant


def query(question: str, graph) -> str:
    """Answer a question from the knowledge graph."""
    all_claims = graph.claims
    contradictions = graph.contradictions

    relevant_entities = _find_relevant_entities(question, graph)
    relevant_entity_set = set(relevant_entities)

    relevant_claims = [c for c in all_claims if c["entity"] in relevant_entity_set]
    if not relevant_claims:
        relevant_claims = all_claims

    context = "=== KNOWLEDGE GRAPH CONTEXT ===\n\n"
    context += f"Entities ({len(relevant_entities)}): {', '.join(relevant_entities)}\n\n"

    context += "Claims:\n"
    for i, c in enumerate(relevant_claims):
        context += f"  {i+1}. [{c.get('source_doc', '?')}] {c['entity']}: {c['claim']} (confidence: {c.get('confidence', '?')})\n"

    relevant_contradictions = [
        con for con in contradictions
        if con["existing_claim"].get("entity") in relevant_entity_set
        or con["new_claim"].get("entity") in relevant_entity_set
    ]
    if not relevant_contradictions:
        relevant_contradictions = contradictions

    if relevant_contradictions:
        context += "\nDetected Conflicts:\n"
        for i, con in enumerate(relevant_contradictions):
            context += (
                f"  {i+1}. {con['relation'].upper()}: "
                f"[{con['existing_claim'].get('source_doc', '?')}] \"{con['existing_claim']['claim']}\" vs "
                f"[{con['new_claim'].get('source_doc', '?')}] \"{con['new_claim']['claim']}\" — {con['explanation']}\n"
            )

    edges = []
    for u, v, d in graph.graph.edges(data=True):
        if u in relevant_entity_set or v in relevant_entity_set:
            edges.append(f"  {u} --[{d.get('relation', '?')}]--> {v}")
    if edges:
        context += "\nRelationships:\n" + "\n".join(edges) + "\n"

    log.info("Query '%s' — %d relevant entities, %d relevant claims", question[:60], len(relevant_entities), len(relevant_claims))

    answer = chat_text(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\n=== QUESTION ===\n{question}"}
        ]
    )
    return answer


def query_with_contexts(question: str, graph) -> tuple[str, list[str]]:
    """Answer a question and return the retrieved claim texts used as context."""
    all_claims = graph.claims
    contradictions = graph.contradictions

    relevant_entities = _find_relevant_entities(question, graph)
    relevant_entity_set = set(relevant_entities)

    relevant_claims = [c for c in all_claims if c["entity"] in relevant_entity_set]
    if not relevant_claims:
        relevant_claims = all_claims

    contexts = [f"[{c.get('source_doc', '?')}] {c['entity']}: {c['claim']}" for c in relevant_claims]

    for con in contradictions:
        if (con["existing_claim"].get("entity") in relevant_entity_set
                or con["new_claim"].get("entity") in relevant_entity_set):
            contexts.append(
                f"CONFLICT: [{con['existing_claim'].get('source_doc', '?')}] \"{con['existing_claim']['claim']}\" "
                f"vs [{con['new_claim'].get('source_doc', '?')}] \"{con['new_claim']['claim']}\""
            )

    answer = query(question, graph)
    return answer, contexts
