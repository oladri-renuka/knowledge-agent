"""Persistent knowledge graph with ChromaDB-backed embedding storage."""

import json
import copy
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
import networkx as nx
import chromadb

log = logging.getLogger(__name__)

SEMANTIC_THRESHOLD = 0.75


class KnowledgeGraph:
    def __init__(self, path: str = "results/graph.json", chroma_path: str = "results/chroma_db"):
        self.path = Path(path)
        self.graph = nx.DiGraph()
        self._claims: list[dict] = []
        self._contradictions: list[dict] = []

        self._chroma = chromadb.PersistentClient(path=chroma_path)
        self._entity_col = self._chroma.get_or_create_collection(
            name="entities", metadata={"hnsw:space": "cosine"}
        )
        self._claim_col = self._chroma.get_or_create_collection(
            name="claims", metadata={"hnsw:space": "cosine"}
        )

        if self.path.exists():
            self._load()

    def _load(self):
        raw = self.path.read_text()
        if not raw.strip():
            log.warning("Graph file exists but is empty: %s", self.path)
            return
        data = json.loads(raw)
        for node in data.get("nodes", []):
            self.graph.add_node(node["name"], **{k: v for k, v in node.items() if k != "name"})
        for edge in data.get("edges", []):
            self.graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
        self._claims = data.get("claims", [])
        self._contradictions = data.get("contradictions", [])
        log.info("Loaded graph: %d nodes, %d edges, %d claims, chroma entities: %d, chroma claims: %d",
                 self.graph.number_of_nodes(), self.graph.number_of_edges(), len(self._claims),
                 self._entity_col.count(), self._claim_col.count())

    def _serialize(self) -> dict:
        return {
            "nodes": [{"name": n, **d} for n, d in self.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.graph.edges(data=True)],
            "claims": self._claims,
            "contradictions": self._contradictions,
        }

    def _atomic_write(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(tmp_fd, "w") as f:
                json.dump(data, f)
            Path(tmp_path).replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def save(self):
        self._atomic_write(self.path, self._serialize())

    def save_snapshot(self, snapshot_dir: str, label: str):
        snap_path = Path(snapshot_dir) / f"snapshot_{label}.json"
        data = self._serialize()
        data["snapshot_label"] = label
        data["snapshot_time"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write(snap_path, data)

    # --- Entity resolution ---

    def resolve_entity_by_alias(self, name: str, aliases: list[str] | None = None) -> str | None:
        check_names = [name.lower()] + [a.lower() for a in (aliases or [])]
        for node, data in self.graph.nodes(data=True):
            node_aliases = data.get("aliases", [])
            all_names = [node.lower()] + [a.lower() for a in node_aliases]
            for cn in check_names:
                if cn in all_names:
                    return node
        return None

    def batch_resolve_entities(self, names: list[str], embeddings: list[list[float]]) -> dict[str, str]:
        """Resolve entity names against ChromaDB. Zero API calls — uses cached vectors."""
        resolution_map = {}

        for name, emb in zip(names, embeddings):
            alias_match = self.resolve_entity_by_alias(name)
            if alias_match:
                resolution_map[name] = alias_match
                continue

            if self._entity_col.count() > 0 and emb:
                results = self._entity_col.query(query_embeddings=[emb], n_results=1)
                if results["distances"] and results["distances"][0]:
                    cosine_dist = results["distances"][0][0]
                    similarity = 1 - cosine_dist
                    if similarity >= SEMANTIC_THRESHOLD:
                        matched_name = results["metadatas"][0][0]["name"]
                        log.debug("Semantic entity match: '%s' → '%s' (%.3f)", name, matched_name, similarity)
                        resolution_map[name] = matched_name
                        continue

            resolution_map[name] = name

        return resolution_map

    def add_entity(self, name: str, entity_type: str, aliases: list[str] | None = None,
                   canonical: str | None = None, embedding: list[float] | None = None):
        target = canonical or name
        if target in self.graph:
            existing_aliases = set(self.graph.nodes[target].get("aliases", []))
            existing_aliases.update(aliases or [])
            if name != target:
                existing_aliases.add(name)
            self.graph.nodes[target]["aliases"] = list(existing_aliases)
        else:
            self.graph.add_node(target, type=entity_type, aliases=aliases or [])
            if embedding:
                self._entity_col.add(
                    ids=[str(uuid.uuid4())],
                    embeddings=[embedding],
                    metadatas=[{"name": target, "type": entity_type}],
                )

    def add_relationship(self, source: str, target: str, relation: str):
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, relation=relation)
        else:
            src = self.resolve_entity_by_alias(source) or source
            tgt = self.resolve_entity_by_alias(target) or target
            self.graph.add_edge(src, tgt, relation=relation)

    def add_claim(self, claim: dict, canonical_entity: str | None = None,
                  embedding: list[float] | None = None) -> dict:
        claim = copy.deepcopy(claim)
        claim["ingested_at"] = datetime.now(timezone.utc).isoformat()
        if canonical_entity:
            claim["entity"] = canonical_entity
        claim_id = str(uuid.uuid4())
        claim["id"] = claim_id
        self._claims.append(claim)

        if embedding:
            self._claim_col.add(
                ids=[claim_id],
                embeddings=[embedding],
                metadatas=[{
                    "entity": claim.get("entity", ""),
                    "source_doc": claim.get("source_doc", ""),
                    "claim_text": claim.get("claim", "")[:500],
                }],
            )
        return claim

    def add_contradiction(self, existing_claim: dict, new_claim: dict, relation_type: str, explanation: str):
        self._contradictions.append({
            "existing_claim": existing_claim,
            "new_claim": new_claim,
            "relation": relation_type,
            "explanation": explanation,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        })

    def get_claims_for_entity(self, entity: str) -> list[dict]:
        return [c for c in self._claims if c["entity"] == entity]

    def find_similar_claims(self, query_embedding: list[float], top_k: int = 10,
                            exclude_source: str | None = None) -> list[dict]:
        """Find top-k similar claims via ChromaDB HNSW index. O(log n), zero API calls."""
        if self._claim_col.count() == 0:
            return []

        where = {"source_doc": {"$ne": exclude_source}} if exclude_source else None
        try:
            results = self._claim_col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._claim_col.count()),
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception:
            log.warning("ChromaDB query failed, falling back to empty results")
            return []

        matched_claims = []
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                claim_text = meta.get("claim_text", "")
                entity = meta.get("entity", "")
                for c in self._claims:
                    if c.get("claim", "")[:500] == claim_text and c.get("entity", "") == entity:
                        matched_claims.append(c)
                        break

        return matched_claims

    def get_all_entities(self) -> list[str]:
        return list(self.graph.nodes)

    @property
    def claims(self) -> list[dict]:
        return self._claims

    @property
    def contradictions(self) -> list[dict]:
        return self._contradictions

    def stats(self) -> dict:
        return {
            "entities": self.graph.number_of_nodes(),
            "relationships": self.graph.number_of_edges(),
            "claims": len(self._claims),
            "contradictions": len(self._contradictions),
        }

    def reset(self):
        self.graph.clear()
        self._claims.clear()
        self._contradictions.clear()
        self._chroma.delete_collection("entities")
        self._chroma.delete_collection("claims")
        self._entity_col = self._chroma.get_or_create_collection(
            name="entities", metadata={"hnsw:space": "cosine"}
        )
        self._claim_col = self._chroma.get_or_create_collection(
            name="claims", metadata={"hnsw:space": "cosine"}
        )
        if self.path.exists():
            self.path.unlink()
