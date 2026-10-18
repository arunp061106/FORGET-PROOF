"""
Consent lineage graph — shows the data flow from patient → shard → model → prediction.

When a forget request comes in, we highlight the affected path in red.
This gives regulators a visual audit trail of exactly what changes
when data is removed.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def build_lineage_graph(n_shards, forget_shard=None):
    """
    Build the consent lineage as a directed graph.

    Structure:
        Patients → Consent Batches → Shards → Sub-models → Ensemble

    If forget_shard is set, that path is marked as "affected".
    """
    G = nx.DiGraph()

    # layers
    G.add_node("Dataset", layer=0, type="data")
    for s in range(n_shards):
        batch = f"Batch {s}"
        shard = f"Shard {s}"
        model = f"Model {s}"
        G.add_node(batch, layer=1, type="batch")
        G.add_node(shard, layer=2, type="shard")
        G.add_node(model, layer=3, type="model")
        G.add_edge("Dataset", batch)
        G.add_edge(batch, shard)
        G.add_edge(shard, model)
        G.add_edge(model, "Ensemble")

    G.add_node("Ensemble", layer=4, type="ensemble")
    G.add_node("Prediction", layer=5, type="output")
    G.add_edge("Ensemble", "Prediction")

    return G


def draw_lineage(n_shards, forget_shard=None, output_dir="outputs"):
    """Draw the lineage graph, highlighting forgotten path in red."""
    G = build_lineage_graph(n_shards, forget_shard)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # position nodes in layers
    pos = {}
    layers = {}
    for node, data in G.nodes(data=True):
        layer = data.get("layer", 0)
        layers.setdefault(layer, []).append(node)

    for layer, nodes in layers.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = layer * 2.5
            y = (i - n/2) * 1.2
            pos[node] = (x, y)

    # figure out which edges are on the "forget path"
    forget_edges = set()
    forget_nodes = set()
    if forget_shard is not None:
        path_nodes = [
            "Dataset", f"Batch {forget_shard}", f"Shard {forget_shard}",
            f"Model {forget_shard}", "Ensemble", "Prediction"
        ]
        forget_nodes = set(path_nodes)
        for i in range(len(path_nodes) - 1):
            forget_edges.add((path_nodes[i], path_nodes[i+1]))

    # draw normal edges
    normal_edges = [(u, v) for u, v in G.edges() if (u, v) not in forget_edges]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges,
                           edge_color="#334155", arrows=True,
                           arrowsize=15, ax=ax, alpha=0.5)

    # draw forget-path edges (red, thicker)
    if forget_edges:
        nx.draw_networkx_edges(G, pos, edgelist=list(forget_edges),
                               edge_color="#ef4444", arrows=True,
                               arrowsize=20, width=3, ax=ax, alpha=0.9)

    # draw nodes
    for node in G.nodes():
        x, y = pos[node]
        color = "#ef4444" if node in forget_nodes else "#38bdf8"
        alpha = 1.0 if node in forget_nodes else 0.7
        ax.scatter(x, y, s=800, c=color, alpha=alpha, zorder=5,
                   edgecolors="#e2e8f0", linewidths=1.5)
        ax.text(x, y - 0.45, node, ha="center", va="top",
                fontsize=8, color="#e2e8f0", fontweight="bold")

    # title
    title = "Data Lineage: Patient → Consent Batch → Shard → Model → Prediction"
    if forget_shard is not None:
        title += f"\n🔴 Forget request affects Shard {forget_shard} path"
    ax.set_title(title, color="#e2e8f0", fontsize=13, fontweight="bold", pad=20)
    ax.axis("off")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "consent_lineage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"  Saved {path}")
    return path
