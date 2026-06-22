"""Demo entry point: ingest documents in order, save snapshots, then run sample queries."""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler

from src.graph_store import KnowledgeGraph
from src.ingest import ingest_document
from src.query import query

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger(__name__)
console = Console()

DOCS_DIR = "documents"
GRAPH_PATH = "results/graph.json"
SNAPSHOT_DIR = "results/graph_snapshots"

DEMO_QUERIES = [
    "What is DPO and how does it relate to RLHF?",
    "Are there any contradictions in the knowledge graph?",
    "What methods improve upon RLHF?",
    "What is the capital of France?",
]


def run_ingestion():
    graph = KnowledgeGraph(GRAPH_PATH)
    docs = sorted(Path(DOCS_DIR).glob("*"))
    docs = [d for d in docs if d.suffix.lower() in (".txt", ".pdf", ".md")]

    if not docs:
        console.print(f"[red]No documents found in {DOCS_DIR}/. Add .txt, .md, or .pdf files and re-run.[/red]")
        sys.exit(1)

    console.print(Panel(f"[bold]Personal Knowledge Agent — Demo Run[/bold]\n\nDocuments to ingest: {len(docs)}", style="blue"))

    for i, doc in enumerate(docs):
        console.print(f"\n[bold cyan]--- Ingesting [{i+1}/{len(docs)}]: {doc.name} ---[/bold cyan]")
        result = ingest_document(str(doc), graph)

        if result.get("skipped"):
            console.print("[dim]Skipped (already ingested or empty)[/dim]")
            continue

        graph.save_snapshot(SNAPSHOT_DIR, f"{i+1:02d}_{doc.stem}")

        table = Table(show_header=False, box=None)
        table.add_row("Entities extracted", str(result["entities_extracted"]))
        table.add_row("Claims extracted", str(result["claims_extracted"]))
        table.add_row("Relationships extracted", str(result["relationships_extracted"]))
        table.add_row("Conflicts detected", str(result["conflicts_detected"]))
        console.print(table)

        for conflict in result["conflicts"]:
            console.print(Panel(
                f"[bold red]{conflict['relation'].upper()}[/bold red]\n\n"
                f"[yellow]Existing:[/yellow] \"{conflict['existing_claim']['claim']}\"\n"
                f"  Source: {conflict['existing_claim'].get('source_doc', '?')}\n\n"
                f"[yellow]New:[/yellow] \"{conflict['new_claim']['claim']}\"\n"
                f"  Source: {conflict['new_claim'].get('source_doc', '?')}\n\n"
                f"[dim]{conflict['explanation']}[/dim]",
                title="Conflict Detected",
                border_style="red"
            ))

        stats = graph.stats()
        console.print(f"[dim]Graph now: {stats['entities']} entities, {stats['claims']} claims, "
                      f"{stats['relationships']} relationships, {stats['contradictions']} contradictions[/dim]")

    return graph


def run_queries(graph):
    console.print(f"\n[bold cyan]--- Query Demo ---[/bold cyan]\n")

    for q in DEMO_QUERIES:
        console.print(f"[bold]Q: {q}[/bold]")
        answer = query(q, graph)
        console.print(Panel(answer, border_style="green"))
        console.print()


def interactive_mode(graph):
    console.print("\n[bold]Interactive mode — ask questions (type 'quit' to exit):[/bold]\n")
    while True:
        try:
            q = console.input("[bold cyan]> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in ("quit", "exit", "q"):
            break
        if not q.strip():
            continue
        answer = query(q, graph)
        console.print(Panel(answer, border_style="green"))
        console.print()


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print("[red]Set OPENROUTER_API_KEY in .env or environment.[/red]")
        sys.exit(1)

    graph = run_ingestion()
    run_queries(graph)

    if "--interactive" in sys.argv:
        interactive_mode(graph)

    from visualize import generate
    generate(GRAPH_PATH, "results/graph.html")

    console.print("\n[bold green]Done.[/bold green] Graph saved to results/graph.json")
    console.print(f"Snapshots saved to {SNAPSHOT_DIR}/")
    console.print("Visualization: results/graph.html")


if __name__ == "__main__":
    main()
