"""Layout computation using dagre."""

from app.models.graph import ArchitectureGraph


def compute_layout(graph: ArchitectureGraph) -> list[dict]:
    """
    Compute node positions using a simple layered layout.
    Returns list of nodes with x, y positions for React Flow.
    
    ponytail: Using a simple BFS-based layered layout instead of dagre.
    dagre adds a dependency for what is essentially topological layering.
    Swap to dagre if graphs get complex enough to need edge-crossing minimization.
    """
    if not graph.nodes:
        return []
    
    # Build adjacency
    adjacency = {n.id: [] for n in graph.nodes}
    in_degree = {n.id: 0 for n in graph.nodes}
    
    for edge in graph.edges:
        if edge.source in adjacency:
            adjacency[edge.source].append(edge.target)
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1
    
    # BFS layering (Kahn's algorithm for topological sort with layer tracking)
    layers = []
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
        current_layer = list(set(next_layer))  # deduplicate
    
    # Add any remaining unvisited nodes (cycles) to a final layer
    unvisited = [n.id for n in graph.nodes if n.id not in visited]
    if unvisited:
        layers.append(unvisited)
    
    # Assign positions
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


def graph_to_react_flow(graph: ArchitectureGraph) -> dict:
    """Convert ArchitectureGraph to React Flow format with computed layout."""
    nodes = compute_layout(graph)
    
    edges = []
    for edge in graph.edges:
        edges.append({
            "id": f"{edge.source}->{edge.target}",
            "source": edge.source,
            "target": edge.target,
            "label": edge.label,
            "animated": True
        })
    
    return {"nodes": nodes, "edges": edges}
