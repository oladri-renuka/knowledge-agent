"""Persistent knowledge graph with source attribution and cached semantic embeddings."""

import json
import copy
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import networkx as nx

log = logging.getLogger(__name__)

SEMANTIC_THRESHOLD = 0.75


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class KnowledgeGraph:
    def __init__(self, path: str = "results/graph.json"):
        self.path = Path(path)
        self.graph = nx.DiGraph()
        self._claims: list[dict] = []
        self._contradictions: list[dict] = []
        self._entity_embeddings: dict[str, list[float]] = {}
        self._claim_embeddings: list[list[float]] = []
        if self.path.exists():
            self._load()

    def _load(self):
        raw = self.path.read_text()
        if not raw.strip():
            log.warning("Graph file exists but is empty: %s", self.path)
            return
        data = json.loads(raw)
        for node in data.get("nodes", []):
            attrs = {k: v for k, v in node.items() if k not in ("name", "embedding")}
            self.graph.add_node(node["name"], **attrs)
            if "embedding" in node:
                self._entity_embeddings[node["name"]] = node["embedding"]
        for edge in data.get("edges", []):
            self.graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
        self._claims = data.get("claims", [])
        self._contradictions = data.get("contradictions", [])
        self._claim_embeddings = data.get("claim_embeddings", [])
        log.info("Loaded graph: %d nodes, %d edges, %d claims, %d entity embeddings cached",
                 self.graph.number_of_nodes(), self.graph.number_of_edges(),
                 len(self._claims), len(self._entity_embeddings))

    def _serialize(self) -> dict:
        nodes = []
        for n, d in self.graph.nodes(data=True):
            node = {"name": n, **d}
            if n in self._entity_embeddings:
                node["embedding"] = self._entity_embeddings[n]
            nodes.append(node)
        return {
            "nodes": nodes,
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.graph.edges(data=True)],
            "claims": self._claims,
            "contradictions": self._contradictions,
            "claim_embeddings": self._claim_embeddings,
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
        data.pop("claim_embeddings", None)
        for node in data["nodes"]:
            node.pop("embedding", None)
        self._atomic_write(snap_path, data)

    # --- Entity resolution: alias-only (fast), or batch semantic (see batch_resolve) ---

    def resolve_entity_by_alias(self, name: str, aliases: list[str] | None = None) -> str | None:
        """Fast alias-only resolution. No API calls. Returns None if no match."""
        check_names = [name.lower()] + [a.lower() for a in (aliases or [])]
        for node, data in self.graph.nodes(data=True):
            node_aliases = data.get("aliases", [])
            all_names = [node.lower()] + [a.lower() for a in node_aliases]
            for cn in check_names:
                if cn in all_names:
                    return node
        return None

    def batch_resolve_entities(self, names: list[str], embeddings: list[list[float]]) -> dict[str, str]:
        """Resolve a batch of entity names against cached entity embeddings.

        Returns a mapping {input_name: canonical_name}.
        Uses alias matching first, then semantic similarity against cached embeddings.
        Zero API calls — all similarity is computed against pre-cached vectors.
        """
        resolution_map = {}
        cached_entities = list(self._entity_embeddings.keys())
        cached_vecs = [self._entity_embeddings[e] for e in cached_entities]

        for name, emb in zip(names, embeddings):
            # Pass 1: alias match
            alias_match = self.resolve_entity_by_alias(name)
            if alias_match:
                resolution_map[name] = alias_match
                continue

            # Pass 2: semantic similarity against cached entity embeddings
            if cached_vecs:
                best_score = 0.0
                best_entity = None
                for i, cv in enumerate(cached_vecs):
                    score = _cosine_similarity(emb, cv)
                    if score > best_score:
                        best_score = score
                        best_entity = cached_entities[i]

                if best_score >= SEMANTIC_THRESHOLD and best_entity:
                    log.debug("Semantic entity match: '%s' → '%s' (%.3f)", name, best_entity, best_score)
                    resolution_map[name] = best_entity
                    continue

            resolution_map[name] = name

        return resolution_map

    def add_entity(self, name: str, entity_type: str, aliases: list[str] | None = None,
                   canonical: str | None = None, embedding: list[float] | None = None):
        """Add or merge an entity. If canonical is provided, skip resolution."""
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
                self._entity_embeddings[target] = embedding

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
        self._claims.append(claim)
        if embedding:
            self._claim_embeddings.append(embedding)
        else:
            self._claim_embeddings.append([])
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

    def find_similar_claims_by_embedding(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """Find top-k similar claims using cached embeddings only. Zero API calls."""
        if not self._claims or not self._claim_embeddings:
            return []

        scored = []
        for i, cached_emb in enumerate(self._claim_embeddings):
            if not cached_emb:
                continue
            score = _cosine_similarity(query_embedding, cached_emb)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._claims[i] for _, i in scored[:top_k]]

    def get_all_entities(self) -> list[str]:
        return list(self.graph.nodes)

    @property
    def claims(self) -> list[dict]:
        return self._claims

    @property
    def claim_embeddings(self) -> list[list[float]]:
        return self._claim_embeddings

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
        self._entity_embeddings.clear()
        self._claim_embeddings.clear()
        if self.path.exists():
            self.path.unlink()
