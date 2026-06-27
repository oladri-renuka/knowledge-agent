"""MCP server exposing the knowledge graph as queryable tools."""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.graph_store import KnowledgeGraph
from src.query import query as query_graph

app = Server("knowledge-agent")

GRAPH_PATH = os.environ.get("GRAPH_PATH", "results/graph.json")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "results/chroma_db")

_graph: KnowledgeGraph | None = None


def get_graph() -> KnowledgeGraph:
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph(GRAPH_PATH, CHROMA_PATH)
    return _graph


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_knowledge_graph",
            description="Query the belief graph built from ingested research papers. Returns answer with source citations. Says 'I don't know' when the graph has no relevant coverage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question to answer from the knowledge graph"}
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="get_contradictions",
            description="Returns all detected contradictions and refinements in the knowledge graph with source documents cited.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_entity_claims",
            description="Returns all claims about a specific entity with source attribution and confidence levels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Entity name to look up (e.g., 'DPO', 'Transformer', 'RLHF')"}
                },
                "required": ["entity_name"],
            },
        ),
        types.Tool(
            name="get_graph_stats",
            description="Returns summary statistics of the knowledge graph: entity count, claim count, relationship count, contradiction count.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="list_entities",
            description="Returns all entity names in the knowledge graph.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    graph = get_graph()

    if name == "query_knowledge_graph":
        result = query_graph(arguments["question"], graph)
        return [types.TextContent(type="text", text=result)]

    elif name == "get_contradictions":
        contradictions = graph.contradictions
        if not contradictions:
            return [types.TextContent(type="text", text="No contradictions detected in the knowledge graph.")]
        output = []
        for i, con in enumerate(contradictions):
            ec = con["existing_claim"]
            nc = con["new_claim"]
            output.append(
                f"{i+1}. [{con['relation'].upper()}]\n"
                f"   Existing [{ec.get('source_doc', '?')}]: {ec['claim']}\n"
                f"   New [{nc.get('source_doc', '?')}]: {nc['claim']}\n"
                f"   Explanation: {con['explanation']}\n"
            )
        return [types.TextContent(type="text", text="\n".join(output))]

    elif name == "get_entity_claims":
        entity = arguments["entity_name"]
        claims = graph.get_claims_for_entity(entity)
        if not claims:
            all_entities = graph.get_all_entities()
            matches = [e for e in all_entities if entity.lower() in e.lower()]
            if matches:
                for m in matches:
                    claims.extend(graph.get_claims_for_entity(m))
            if not claims:
                return [types.TextContent(type="text",
                    text=f"No claims found for entity '{entity}'. Available entities with similar names: {matches[:10] if matches else 'none'}.")]
        output = []
        for c in claims:
            output.append(f"[{c.get('source_doc', '?')}] ({c.get('confidence', '?')}): {c['claim']}")
        return [types.TextContent(type="text", text=f"Claims about '{entity}' ({len(claims)} total):\n" + "\n".join(output))]

    elif name == "get_graph_stats":
        stats = graph.stats()
        return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]

    elif name == "list_entities":
        entities = graph.get_all_entities()
        return [types.TextContent(type="text", text=f"{len(entities)} entities:\n" + "\n".join(sorted(entities)))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
