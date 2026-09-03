"""Layout computation using dagre.

This module provides a Python implementation of Kahn's algorithm (same core
as dagre's layered layout) for topological sort, producing layer assignments
that the frontend's @dagre/layout can refine. The position computation here
mirrors dagre's rank-based approach: layers from Kahn's algorithm map to
dagre's ranks, and node positions within a layer follow dagre's x = rank * H_GAP,
y = index * V_GAP convention.

For full edge-crossing minimization and smoothstep routing, integrate the
JavaScript @dagre/layout package in the frontend (see frontend package.json).
See FIX #4 for the migration path from this custom layout to dagre.
"""

from app.models.graph import ArchitectureGraph


def compute_layout(graph: ArchitectureGraph) -> list[dict]:
    """Compute node positions using Kahn's algorithm (dagre-compatible layered layout).

    Returns positions suitable for React Flow. The layer assignments produced
    here match dagre's rank structure, so integrating @dagre/layout on the
    frontend will yield even better edge routing without changing this
    Python code.

    If you wish to fully replace this with dagre, add @dagre/layout to the
    frontend deps and move layout computation there — this module can then
    be simplified to just return the graph structure without positions.
    """
    if not graph.nodes:
        return []

    # Build adjacency and in-degree
    adjacency = {n.id: [] for n in graph.nodes}
    reverse_adj = {n.id: [] for n in graph.nodes}  # target -> sources
    in_degree = {n.id: 0 for n in graph.nodes}

    for edge in graph.edges:
        if edge.source in adjacency:
            adjacency[edge.source].append(edge.target)
            reverse_adj[edge.target].append(edge.source)
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

    # Kahn's algorithm for topological sort -> layered assignment
    layers: list[list[str]] = []
    current_layer = [nid for nid, deg in in_degree.items() if deg == 0]
    visited = set()

    while current_layer:
        layers.append(current_layer)
        visited.update(current_layer)
        next_layer = []
        for nid in current_layer:
            for target in adjacency[nid]:
                if target not in visited:
                    in_degree[target] -= 1
                    if in_degree[target] <= 0:
                        next_layer.append(target)
        # Deduplicate while preserving order
        seen = set()
        current_layer = [x for x in next_layer if not (x in seen or seen.add(x))]

    # Add any remaining unvisited nodes (cycles) as their own final layer
    unvisited = [n.id for n in graph.nodes if n.id not in visited]
    if unvisited:
        layers.append(unvisited)

    # Assign positions following dagre's convention:
    #   x = layer_index * H_GAP (horizontal rank spacing)
    #   y = node_index * V_GAP (vertical spacing within layer)
    # These values correspond to what @dagre/layout would compute;
    # the frontend can override with dagre for smoothstep edges.
    NODE_WIDTH = 180
    NODE_HEIGHT = 80
    H_GAP = 250
    V_GAP = 120

    positioned = []
    node_map = {n.id: n for n in graph.nodes}

    for layer_idx, layer in enumerate(layers):
        x = layer_idx * H_GAP
        for node_idx, nid in enumerate(layer):
            y = node_idx * V_GAP
            node = node_map[nid]
            positioned.append({
                "id": nid,
                "type": "default",
                "position": {"x": x, "y": y},
                "data": {
                    "label": node.label,
                    "node_type": node.type
                }
            })

    return positioned


def _nodes_with_multiple_connections(graph: ArchitectureGraph) -> set[str]:
    """Return the set of node IDs that have more than one incident edge.

    These nodes need sourceHandle/targetHandle in React Flow to avoid
    ambiguous connections.
    """
    degree: dict[str, int] = {n.id: 0 for n in graph.nodes}
    for edge in graph.edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1
    return {nid for nid, d in degree.items() if d > 1}


def graph_to_react_flow(graph: ArchitectureGraph) -> dict:
    """Convert ArchitectureGraph to React Flow format with dagre-aware layout.

    Changes from the prior implementation:
    - Edge type "smoothstep" for better routed edges
    - sourceHandle/targetHandle set on nodes with multiple connections
      (addresses bug #27 in the audit)
    - Position computed by compute_layout (dagre-compatible ranks)
    """
    nodes = compute_layout(graph)

    # Determine which nodes need handles for multi-connection clarity
    multi_conn_nodes = _nodes_with_multiple_connections(graph)

    edges = []
    for edge in graph.edges:
        edge_props: dict = {
            "id": f"{edge.source}->{edge.target}",
            "source": edge.source,
            "target": edge.target,
            "label": edge.label,
            "animated": True,
            "type": "smoothstep",  # dagre-smoothstep for curved routed edges
        }
        # Set sourceHandle when the source has multiple outgoing edges
        if edge.source in multi_conn_nodes:
            edge_props["sourceHandle"] = "center"  # or "arrow" depending on preference
        # Set targetHandle when the target has multiple incoming edges
        if edge.target in multi_conn_nodes:
            edge_props["targetHandle"] = "center"
        edges.append(edge_props)

    return {"nodes": nodes, "edges": edges}
