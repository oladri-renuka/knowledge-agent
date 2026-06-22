"""Generate an interactive D3 force-directed graph visualization from the knowledge graph."""

import json
import sys
from pathlib import Path


def generate(graph_path: str = "results/graph.json", output_path: str = "results/graph.html"):
    data = json.loads(Path(graph_path).read_text())

    nodes = []
    node_ids = set()
    for node in data.get("nodes", []):
        name = node["name"]
        node_ids.add(name)
        claim_count = sum(1 for c in data.get("claims", []) if c.get("entity") == name)
        contradiction_count = sum(
            1 for con in data.get("contradictions", [])
            if con.get("existing_claim", {}).get("entity") == name
            or con.get("new_claim", {}).get("entity") == name
        )
        nodes.append({
            "id": name,
            "type": node.get("type", "unknown"),
            "aliases": node.get("aliases", []),
            "claims": claim_count,
            "contradictions": contradiction_count,
        })

    links = []
    for edge in data.get("edges", []):
        if edge["source"] in node_ids and edge["target"] in node_ids:
            links.append({
                "source": edge["source"],
                "target": edge["target"],
                "relation": edge.get("relation", "related"),
            })

    contradictions = []
    for con in data.get("contradictions", []):
        contradictions.append({
            "existing": con.get("existing_claim", {}).get("claim", ""),
            "existing_source": con.get("existing_claim", {}).get("source_doc", "?"),
            "new": con.get("new_claim", {}).get("claim", ""),
            "new_source": con.get("new_claim", {}).get("source_doc", "?"),
            "relation": con.get("relation", ""),
            "explanation": con.get("explanation", ""),
        })

    graph_json = json.dumps({"nodes": nodes, "links": links, "contradictions": contradictions})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Knowledge Agent — Belief Graph</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; overflow: hidden; }}
svg {{ display: block; }}
.link {{ stroke-opacity: 0.4; }}
.node circle {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; transition: r 0.3s; }}
.node circle:hover {{ stroke: #7eb8ff; stroke-width: 2.5px; }}
.node text {{ font-size: 11px; fill: #ccc; pointer-events: none; }}
#tooltip {{ position: absolute; background: #1a1a2e; border: 1px solid #444; border-radius: 6px; padding: 10px 14px; font-size: 13px; max-width: 350px; pointer-events: none; display: none; z-index: 10; }}
#tooltip h3 {{ color: #7eb8ff; margin-bottom: 4px; font-size: 14px; }}
#tooltip .type {{ color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
#tooltip .stat {{ margin-top: 4px; }}

/* Sidebar — slides in/out */
#sidebar {{ position: absolute; top: 0; right: 0; width: 340px; height: 100vh; background: #111122; border-left: 1px solid #333; overflow-y: auto; padding: 20px; transition: transform 0.35s ease; z-index: 5; }}
#sidebar.collapsed {{ transform: translateX(100%); }}
#sidebar h2 {{ color: #ff6b6b; font-size: 16px; margin-bottom: 12px; }}
.conflict {{ background: #1a1020; border: 1px solid #442233; border-radius: 6px; padding: 10px; margin-bottom: 10px; font-size: 12px; }}
.conflict .label {{ color: #ff6b6b; font-weight: bold; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
.conflict .claim {{ margin: 4px 0; color: #ccc; }}
.conflict .source {{ color: #666; font-size: 11px; }}
.conflict .explanation {{ color: #888; font-style: italic; margin-top: 4px; font-size: 11px; }}

/* Toggle button */
#sidebar-toggle {{ position: absolute; top: 50%; right: 340px; transform: translateY(-50%); z-index: 6; background: #111122; border: 1px solid #333; border-right: none; border-radius: 6px 0 0 6px; color: #7eb8ff; cursor: pointer; padding: 12px 6px; font-size: 18px; transition: right 0.35s ease; line-height: 1; }}
#sidebar-toggle.shifted {{ right: 0; }}

#legend {{ position: absolute; bottom: 20px; left: 20px; background: #111122; border: 1px solid #333; border-radius: 6px; padding: 12px 16px; font-size: 12px; }}
#legend div {{ margin: 3px 0; display: flex; align-items: center; gap: 8px; }}
#legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
#stats {{ position: absolute; top: 20px; left: 20px; background: #111122; border: 1px solid #333; border-radius: 6px; padding: 12px 16px; font-size: 13px; }}
#stats div {{ margin: 2px 0; }}
#stats span {{ color: #7eb8ff; font-weight: bold; }}
</style>
</head>
<body>
<div id="tooltip"></div>
<div id="stats"></div>
<div id="legend"></div>
<button id="sidebar-toggle" title="Toggle conflicts panel"></button>
<div id="sidebar"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {graph_json};

const typeColors = {{
  method: '#4ecdc4',
  concept: '#7eb8ff',
  person: '#ffd166',
  organization: '#ff6b6b',
  dataset: '#a78bfa',
  metric: '#f093fb',
  unknown: '#888',
}};

const relationColors = {{
  improves_upon: '#4ecdc4',
  is_variant_of: '#a78bfa',
  uses: '#7eb8ff',
  contradicts: '#ff6b6b',
  evaluates_on: '#ffd166',
}};

// --- Sidebar toggle ---
const sidebar = document.getElementById('sidebar');
const toggleBtn = document.getElementById('sidebar-toggle');
let sidebarOpen = true;

function updateToggle() {{
  toggleBtn.textContent = sidebarOpen ? '▶' : '◀';
  sidebar.classList.toggle('collapsed', !sidebarOpen);
  toggleBtn.classList.toggle('shifted', !sidebarOpen);
  resizeSVG();
}}

toggleBtn.addEventListener('click', () => {{
  sidebarOpen = !sidebarOpen;
  updateToggle();
}});

// Stats
const statsDiv = document.getElementById('stats');
statsDiv.innerHTML = `
  <div>Entities: <span>${{data.nodes.length}}</span></div>
  <div>Relationships: <span>${{data.links.length}}</span></div>
  <div>Contradictions: <span style="color:#ff6b6b">${{data.contradictions.length}}</span></div>
`;

// Legend
const legendDiv = document.getElementById('legend');
legendDiv.innerHTML = Object.entries(typeColors).map(([t, c]) =>
  `<div><span class="dot" style="background:${{c}}"></span>${{t}}</div>`
).join('');

// Sidebar content
sidebar.innerHTML = `<h2>Contradictions & Refinements (${{data.contradictions.length}})</h2>` +
  data.contradictions.map(c => `
    <div class="conflict">
      <div class="label">${{c.relation}}</div>
      <div class="claim">"${{c.existing}}"</div>
      <div class="source">— ${{c.existing_source}}</div>
      <div class="claim" style="color:#ff9;">vs "${{c.new}}"</div>
      <div class="source">— ${{c.new_source}}</div>
      <div class="explanation">${{c.explanation}}</div>
    </div>
  `).join('');

const sidebarWidth = 340;
let width = window.innerWidth - sidebarWidth;
let height = window.innerHeight;

const svg = d3.select('body').insert('svg', '#sidebar-toggle')
  .attr('width', width).attr('height', height);

const g = svg.append('g');

const zoom = d3.zoom()
  .scaleExtent([0.05, 5])
  .on('zoom', (e) => {{ g.attr('transform', e.transform); }});

svg.call(zoom);

// Fit all nodes in view after initial layout settles
setTimeout(() => {{
  const bounds = g.node().getBBox();
  if (bounds.width === 0) return;
  const pad = 60;
  const scale = Math.min(
    width / (bounds.width + pad * 2),
    height / (bounds.height + pad * 2),
    1
  );
  const tx = width / 2 - (bounds.x + bounds.width / 2) * scale;
  const ty = height / 2 - (bounds.y + bounds.height / 2) * scale;
  svg.transition().duration(750).call(
    zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale)
  );
}}, 2000);

function resizeSVG() {{
  width = sidebarOpen ? window.innerWidth - sidebarWidth : window.innerWidth;
  height = window.innerHeight;
  svg.attr('width', width).attr('height', height);
  simulation.force('center', d3.forceCenter(width / 2, height / 2));
  simulation.alpha(0.15).restart();
}}

// --- Force simulation: never settles ---
const simulation = d3.forceSimulation(data.nodes)
  .force('link', d3.forceLink(data.links).id(d => d.id).distance(100).strength(0.3))
  .force('charge', d3.forceManyBody().strength(-200))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collision', d3.forceCollide().radius(25))
  .force('x', d3.forceX(width / 2).strength(0.01))
  .force('y', d3.forceY(height / 2).strength(0.01))
  .alphaDecay(0)
  .alphaTarget(0.02)
  .velocityDecay(0.4);

const link = g.append('g').selectAll('line')
  .data(data.links).join('line')
  .attr('class', 'link')
  .attr('stroke', d => relationColors[d.relation] || '#555')
  .attr('stroke-width', d => d.relation === 'contradicts' ? 2.5 : 1.5);

const node = g.append('g').selectAll('g')
  .data(data.nodes).join('g')
  .attr('class', 'node')
  .call(d3.drag()
    .on('start', (e, d) => {{ d.fx = d.x; d.fy = d.y; }})
    .on('drag', (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on('end', (e, d) => {{ d.fx = null; d.fy = null; }})
  );

node.append('circle')
  .attr('r', d => 6 + Math.sqrt(d.claims) * 3)
  .attr('fill', d => typeColors[d.type] || typeColors.unknown)
  .attr('opacity', d => d.contradictions > 0 ? 1 : 0.7);

node.append('text')
  .attr('dx', 14).attr('dy', 4)
  .text(d => d.id.length > 25 ? d.id.slice(0, 22) + '...' : d.id);

const tooltip = document.getElementById('tooltip');

node.on('mouseover', (e, d) => {{
  tooltip.style.display = 'block';
  tooltip.innerHTML = `
    <h3>${{d.id}}</h3>
    <div class="type">${{d.type}}</div>
    ${{d.aliases.length ? '<div class="stat">Aliases: ' + d.aliases.join(', ') + '</div>' : ''}}
    <div class="stat">Claims: ${{d.claims}}</div>
    ${{d.contradictions > 0 ? '<div class="stat" style="color:#ff6b6b">Contradictions: ' + d.contradictions + '</div>' : ''}}
  `;
}}).on('mousemove', e => {{
  tooltip.style.left = (e.pageX + 15) + 'px';
  tooltip.style.top = (e.pageY - 10) + 'px';
}}).on('mouseout', () => {{ tooltip.style.display = 'none'; }});

simulation.on('tick', () => {{
  link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});

updateToggle();
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    print(f"Visualization saved to {output_path}")
    print(f"Open in browser: file://{Path(output_path).resolve()}")


if __name__ == "__main__":
    graph_path = sys.argv[1] if len(sys.argv) > 1 else "results/graph.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/graph.html"
    generate(graph_path, output_path)
