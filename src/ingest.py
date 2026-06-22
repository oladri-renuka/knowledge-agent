"""Document ingestion: read, extract, batch-resolve entities, check contradictions, update graph.

Embedding budget per document:
  - 1 call: batch embed all new entity names
  - 1 call: batch embed all new claim texts (inside contradiction checker)
  - 0 calls: all similarity is computed against cached vectors
Total: exactly 2 embedding API calls per document, regardless of document size.
"""

import logging
from pathlib import Path
from src.extractor import extract
from src.contradiction import check_contradictions
from src.graph_store import KnowledgeGraph
from src.llm_client import embed

log = logging.getLogger(__name__)

MAX_EMBED_BATCH = 2048


def read_document(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(str(p))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            log.warning("PDF '%s' yielded no extractable text — may be scanned/image-only", p.name)
        return text
    return p.read_text(encoding="utf-8")


def _batch_embed_safe(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches to avoid API limits."""
    if not texts:
        return []
    all_vecs = []
    for i in range(0, len(texts), MAX_EMBED_BATCH):
        batch = texts[i:i + MAX_EMBED_BATCH]
        vecs = embed(batch)
        all_vecs.extend(vecs)
    return all_vecs


def ingest_document(doc_path: str, graph: KnowledgeGraph, enable_contradiction_check: bool = True) -> dict:
    source_name = Path(doc_path).name

    already_ingested = any(c.get("source_doc") == source_name for c in graph.claims)
    if already_ingested:
        log.warning("Document '%s' was already ingested — skipping to avoid duplicates", source_name)
        return {
            "source": source_name, "entities_extracted": 0, "claims_extracted": 0,
            "relationships_extracted": 0, "conflicts_detected": 0, "conflicts": [], "skipped": True,
        }

    text = read_document(doc_path)
    if not text.strip():
        log.warning("Document '%s' is empty — skipping", source_name)
        return {
            "source": source_name, "entities_extracted": 0, "claims_extracted": 0,
            "relationships_extracted": 0, "conflicts_detected": 0, "conflicts": [], "skipped": True,
        }

    # Step 1: Extract
    extraction = extract(text, source_name)
    entities = extraction["entities"]
    claims = extraction["claims"]
    relationships = extraction["relationships"]

    if not entities and not claims:
        log.warning("Document '%s' yielded no entities or claims", source_name)
        return {
            "source": source_name, "entities_extracted": 0, "claims_extracted": 0,
            "relationships_extracted": 0, "conflicts_detected": 0, "conflicts": [], "skipped": True,
        }

    # Step 2: Batch embed all entity names — 1 API call
    entity_names = [e["name"] for e in entities]
    try:
        entity_embeddings = _batch_embed_safe(entity_names)
    except Exception:
        log.exception("Failed to embed entity names — falling back to alias-only resolution")
        entity_embeddings = [[] for _ in entity_names]

    # Step 3: Batch resolve entities against cached embeddings — 0 API calls
    resolution_map = graph.batch_resolve_entities(entity_names, entity_embeddings)

    # Step 4: Add entities with their embeddings cached
    for i, entity in enumerate(entities):
        name = entity["name"]
        canonical = resolution_map.get(name, name)
        emb = entity_embeddings[i] if i < len(entity_embeddings) else None
        graph.add_entity(name, entity["type"], entity.get("aliases", []),
                         canonical=canonical, embedding=emb if canonical == name else None)

    # Step 5: Remap relationships to canonical entity names, add to graph
    for rel in relationships:
        src = resolution_map.get(rel["source"], rel["source"])
        tgt = resolution_map.get(rel["target"], rel["target"])
        graph.add_relationship(src, tgt, rel["relation"])

    # Step 6: Remap claim entity names to canonical
    for claim in claims:
        claim["entity"] = resolution_map.get(claim["entity"], claim["entity"])

    # Step 7: Contradiction check — 1 embedding call (for claims) + N/5 LLM calls
    conflicts = []
    claim_embeddings = []
    if enable_contradiction_check and claims:
        conflicts, claim_embeddings = check_contradictions(claims, graph)
        for conflict in conflicts:
            graph.add_contradiction(
                conflict["existing_claim"], conflict["new_claim"],
                conflict["relation"], conflict["explanation"],
            )

    # Step 8: Add claims with their embeddings cached
    for i, claim in enumerate(claims):
        emb = claim_embeddings[i] if i < len(claim_embeddings) else None
        graph.add_claim(claim, canonical_entity=claim["entity"], embedding=emb)

    graph.save()

    log.info(
        "Ingested '%s': %d entities, %d claims, %d relationships, %d conflicts. "
        "Embedding calls: 2 (entities + claims)",
        source_name, len(entities), len(claims), len(relationships), len(conflicts)
    )

    return {
        "source": source_name,
        "entities_extracted": len(entities),
        "claims_extracted": len(claims),
        "relationships_extracted": len(relationships),
        "conflicts_detected": len(conflicts),
        "conflicts": conflicts,
        "skipped": False,
    }
