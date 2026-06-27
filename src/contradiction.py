"""Cross-entity contradiction detection via ChromaDB similarity search.

Embeds new claims in one batch call, queries ChromaDB for top-k similar existing
claims (O(log n) via HNSW index), then sends pairs to LLM for classification.
"""

import logging
from src.llm_client import chat_json, embed

log = logging.getLogger(__name__)

MAX_SIMILAR_CLAIMS = 10
MAX_NEW_CLAIMS_PER_BATCH = 5

BATCH_CONTRADICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "new_claim_index": {"type": "integer", "description": "1-based index of the new claim"},
                    "existing_claim_index": {"type": "integer", "description": "1-based index of the existing claim"},
                    "relation": {"type": "string", "enum": ["confirms", "refines", "contradicts", "unrelated"]},
                    "explanation": {"type": "string"}
                },
                "required": ["new_claim_index", "existing_claim_index", "relation", "explanation"],
                "additionalProperties": False
            }
        }
    },
    "required": ["results"],
    "additionalProperties": False
}

SYSTEM_PROMPT = """You compare NEW claims from a document against EXISTING claims in a knowledge graph.
The existing claims may be about DIFFERENT entities — cross-entity contradictions are important to catch.

For each new claim, check it against the existing claims and report ONLY contradictions and refinements.

Classification:
- **refines**: The new claim adds nuance, qualifies, or partially updates an existing claim without flatly contradicting it.
- **contradicts**: The new claim directly conflicts with an existing claim — they cannot both be fully true. This includes cross-entity conflicts (e.g., "X outperforms Y" vs "Y is the best approach").

Be conservative with "contradicts" — genuine contradiction means the claims are logically incompatible.

Examples:
- "DPO eliminates the reward model" vs "RLHF requires a separate reward model" → refines (different methods)
- "DPO achieves 61% win rate against PPO" vs "PPO achieves 58% win rate against DPO" → contradicts
- "RL is not suitable for NLP" vs "RL is used for fine-tuning language models" → contradicts
- "Transformers use self-attention" vs "CNNs use convolution" → unrelated

Use new_claim_index and existing_claim_index (both 1-based).
Return an EMPTY results array if there are no contradictions or refinements."""


def check_contradictions(new_claims: list[dict], graph) -> tuple[list[dict], list[list[float]]]:
    """Check new claims against ChromaDB-indexed existing claims.

    Embedding cost: 1 API call (batch embed new claims).
    Similarity search: 0 API calls (ChromaDB HNSW, O(log n)).
    LLM calls: ~N/5 for contradiction classification.

    Returns (conflicts, new_claim_embeddings).
    """
    if not new_claims:
        return [], []

    new_texts = [c["claim"] for c in new_claims]
    try:
        new_embeddings = embed(new_texts)
    except Exception:
        log.exception("Failed to embed new claims — skipping contradiction check")
        return [], []

    if not graph.claims:
        return [], new_embeddings

    # Query ChromaDB for similar claims — O(log n) per query, zero API calls
    claim_groups: dict[int, list[dict]] = {}
    source_doc = new_claims[0].get("source_doc", "")

    for i, nc_emb in enumerate(new_embeddings):
        similar = graph.find_similar_claims(
            nc_emb, top_k=MAX_SIMILAR_CLAIMS, exclude_source=source_doc
        )
        if similar:
            claim_groups[i] = similar

    if not claim_groups:
        return [], new_embeddings

    # Batch LLM calls
    conflicts = []
    total_calls = 0
    group_indices = sorted(claim_groups.keys())

    for batch_start in range(0, len(group_indices), MAX_NEW_CLAIMS_PER_BATCH):
        batch_indices = group_indices[batch_start:batch_start + MAX_NEW_CLAIMS_PER_BATCH]
        batch_claims = [new_claims[i] for i in batch_indices]

        all_existing = {}
        for idx in batch_indices:
            for ec in claim_groups[idx]:
                key = (ec.get("entity", ""), ec.get("claim", ""))
                if key not in all_existing:
                    all_existing[key] = ec
        existing_list = list(all_existing.values())

        if not existing_list:
            continue

        prompt = "NEW claims (from the document being ingested):\n"
        for i, nc in enumerate(batch_claims):
            prompt += f'  N{i+1}. [{nc.get("entity", "?")}] "{nc["claim"]}" (from {nc.get("source_doc", "unknown")})\n'

        prompt += "\nEXISTING claims in the knowledge graph:\n"
        for i, ec in enumerate(existing_list):
            prompt += f'  E{i+1}. [{ec.get("entity", "?")}] "{ec["claim"]}" (from {ec.get("source_doc", "unknown")})\n'

        prompt += "\nReport only contradictions and refinements, including cross-entity conflicts. Return empty results array if none."

        try:
            result = chat_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                schema=BATCH_CONTRADICTION_SCHEMA,
            )
            total_calls += 1
        except Exception:
            log.exception("Contradiction check failed for batch")
            continue

        for item in result.get("results", []):
            if item.get("relation") not in ("contradicts", "refines"):
                continue

            new_idx = item.get("new_claim_index")
            existing_idx = item.get("existing_claim_index")

            if not isinstance(new_idx, int) or not isinstance(existing_idx, int):
                continue
            if not (1 <= new_idx <= len(batch_claims)):
                continue
            if not (1 <= existing_idx <= len(existing_list)):
                continue

            conflicts.append({
                "existing_claim": existing_list[existing_idx - 1],
                "new_claim": batch_claims[new_idx - 1],
                "relation": item["relation"],
                "explanation": item["explanation"],
            })

    log.info(
        "Contradiction check: %d new claims, %d with similar existing, %d LLM calls, %d conflicts",
        len(new_claims), len(claim_groups), total_calls, len(conflicts)
    )
    return conflicts, new_embeddings
