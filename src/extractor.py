"""Entity/claim/relationship extraction from unstructured text via LLM structured output."""

import logging
from src.llm_client import chat_json

log = logging.getLogger(__name__)

CHUNK_SIZE = 10000
CHUNK_OVERLAP = 500

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Canonical name of the entity"},
                    "type": {"type": "string", "enum": ["concept", "method", "person", "organization", "dataset", "metric"]},
                    "aliases": {"type": "array", "items": {"type": "string"}, "description": "Alternative names or abbreviations"}
                },
                "required": ["name", "type", "aliases"],
                "additionalProperties": False
            }
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The entity this claim is about — must match an entity name above"},
                    "claim": {"type": "string", "description": "A specific, falsifiable factual assertion"},
                    "confidence": {"type": "string", "enum": ["stated_directly", "implied", "speculative"]}
                },
                "required": ["entity", "claim", "confidence"],
                "additionalProperties": False
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string", "enum": ["improves_upon", "is_variant_of", "uses", "contradicts", "evaluates_on"]}
                },
                "required": ["source", "target", "relation"],
                "additionalProperties": False
            }
        }
    },
    "required": ["entities", "claims", "relationships"],
    "additionalProperties": False
}

SYSTEM_PROMPT = """You are a precise knowledge extractor. Given a document chunk, extract:

1. **Entities**: Key concepts, methods, people, organizations, datasets, or metrics mentioned. Use canonical names. Include aliases (e.g., "DPO" for "Direct Preference Optimization"). Always provide the aliases array, even if empty.

2. **Claims**: Specific, falsifiable factual assertions made about entities. Each claim should be atomic — one fact per claim. The entity field MUST exactly match one of the entity names you extracted above. Prefer direct quotes or close paraphrases over vague summaries. Mark confidence:
   - stated_directly: the document explicitly says this
   - implied: strongly suggested but not stated verbatim
   - speculative: the document speculates or hedges

3. **Relationships**: How entities relate to each other. Source and target MUST exactly match entity names above. Use only these relation types:
   - improves_upon: X is presented as better than Y
   - is_variant_of: X is a version/extension of Y
   - uses: X depends on or incorporates Y
   - contradicts: X's claims conflict with Y's claims
   - evaluates_on: X is tested/measured using Y

Extract 2-3 entity types max. Focus on depth and correctness over breadth. Only extract what the document actually says — never infer beyond the text.
If this chunk contains mostly references, bibliographies, or boilerplate, return empty arrays."""


def _split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks, breaking at paragraph boundaries when possible."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE

        if end < len(text):
            # Try to break at a paragraph boundary (double newline) within the last 20% of the chunk
            search_start = start + int(CHUNK_SIZE * 0.8)
            break_point = text.rfind("\n\n", search_start, end)
            if break_point != -1:
                end = break_point

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break

    return chunks


def _extract_chunk(chunk: str, source_doc: str, chunk_idx: int, total_chunks: int) -> dict:
    """Extract from a single chunk."""
    result = chat_json(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract entities, claims, and relationships from this document chunk ({chunk_idx+1}/{total_chunks}):\n\n{chunk}"}
        ],
        schema=EXTRACTION_SCHEMA,
    )

    for key in ("entities", "claims", "relationships"):
        if key not in result:
            log.error("Extraction missing '%s' field in chunk %d — LLM returned malformed response", key, chunk_idx + 1)
            result[key] = []

    return result


def _deduplicate_claims(claims: list[dict]) -> list[dict]:
    """Remove near-duplicate claims from merged chunk results."""
    seen = set()
    unique = []
    for claim in claims:
        key = (claim["entity"].lower(), claim["claim"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(claim)
    return unique


def extract(text: str, source_doc: str) -> dict:
    if not text.strip():
        log.warning("Document '%s' is empty, returning empty extraction", source_doc)
        return {"entities": [], "claims": [], "relationships": []}

    chunks = _split_into_chunks(text)
    log.info("Document '%s': %d chars → %d chunk(s)", source_doc, len(text), len(chunks))

    all_entities = []
    all_claims = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        log.info("Extracting chunk %d/%d from '%s' (%d chars)", i + 1, len(chunks), source_doc, len(chunk))
        result = _extract_chunk(chunk, source_doc, i, len(chunks))

        all_entities.extend(result["entities"])
        all_claims.extend(result["claims"])
        all_relationships.extend(result["relationships"])

    for claim in all_claims:
        claim["source_doc"] = source_doc

    all_claims = _deduplicate_claims(all_claims)

    # Deduplicate entities by lowercase name
    seen_entities = {}
    for entity in all_entities:
        key = entity["name"].lower()
        if key in seen_entities:
            existing = seen_entities[key]
            merged_aliases = set(existing.get("aliases", []) + entity.get("aliases", []))
            existing["aliases"] = list(merged_aliases)
        else:
            seen_entities[key] = entity
    all_entities = list(seen_entities.values())

    # Deduplicate relationships
    seen_rels = set()
    unique_rels = []
    for rel in all_relationships:
        key = (rel["source"].lower(), rel["target"].lower(), rel["relation"])
        if key not in seen_rels:
            seen_rels.add(key)
            unique_rels.append(rel)
    all_relationships = unique_rels

    log.info(
        "Extracted from '%s': %d entities, %d claims, %d relationships (from %d chunks)",
        source_doc, len(all_entities), len(all_claims), len(all_relationships), len(chunks)
    )

    return {"entities": all_entities, "claims": all_claims, "relationships": all_relationships}
