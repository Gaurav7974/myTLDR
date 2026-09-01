"""Helpers for combining verified vision geometry with text-side semantics.

The trust-first rule is enforced at the boundary of this module:
* ``merge_graph_metadata`` may NEVER overwrite figure geometry (bbox,
  path_points, group bbox, color) with text-derived values.
* ``should_render_reconstruction`` is the single place that decides
  whether the frontend may draw a "figure-faithful" diagram. Anything
  that fails the gate is forced into ``layout_mode='auto'``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from app.models.graph import (
    ArchitectureGraph,
    GRAPH_PAYLOAD_VERSION,
    GraphProvenance,
)
from app.services.figure_extraction import (
    is_verified_vision_graph as _is_verified_vision_graph,  # re-export
)


# Re-export for callers that import the pipeline helpers.
__all__ = [
    "merge_graph_metadata",
    "should_render_reconstruction",
    "is_verified_vision_graph",
]


def _normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def _matching_text_node(
    vision_node, text_nodes: Iterable
):
    """Best-effort match: prefer id, then label, then paper_term."""
    for t in text_nodes:
        if t.id == vision_node.id:
            return t
    vision_label = _normalize_label(vision_node.label)
    for t in text_nodes:
        if _normalize_label(t.label) == vision_label and vision_label:
            return t
    vision_paper = _normalize_label(getattr(vision_node, "paper_term", "") or "")
    for t in text_nodes:
        t_paper = _normalize_label(getattr(t, "paper_term", "") or "")
        if t_paper and t_paper == vision_paper:
            return t
    return None


def merge_graph_metadata(
    vision_graph: ArchitectureGraph,
    text_graph: ArchitectureGraph,
    prefer: str = "vision",
) -> ArchitectureGraph:
    """Combine a verified vision graph with a verified text graph.

    The vision graph contributes geometry (bbox, path_points, group bbox,
    colors, shapes). The text graph contributes semantic enrichment
    (description, how_it_works, math_formula, paper_term, edge type/semantic).

    Visual fields are NEVER overwritten by text data. Provenance is merged
    so the sidebar can cite both sources.
    """
    assert prefer == "vision", "prefer must be 'vision' for figure-faithful output"
    text_nodes = list(text_graph.nodes)

    merged_nodes = []
    for vn in vision_graph.nodes:
        # Copy via model_dump then model_validate to keep the new instance
        # independent of the input.
        node_data = vn.model_dump(mode="json")
        tn = _matching_text_node(vn, text_nodes)
        if tn is not None:
            # Semantic enrichment: copy ONLY when the vision node lacks it.
            for field in ("description", "how_it_works", "math_formula",
                          "paper_term", "parameters", "flops"):
                v_val = node_data.get(field)
                t_val = getattr(tn, field, None)
                if not v_val and t_val:
                    node_data[field] = t_val
            # Provenance merge: keep figure provenance, add text citation if any.
            existing = node_data.get("provenance") or {}
            text_prov = tn.provenance.model_dump(mode="json") if tn.provenance else {}
            if text_prov.get("citation") and not existing.get("citation"):
                existing["citation"] = text_prov["citation"]
            if text_prov.get("section") and not existing.get("section"):
                existing["section"] = text_prov["section"]
            node_data["provenance"] = existing
        merged_nodes.append(node_data)

    # Edges: prefer vision edges, fall back to text-only edges for connections
    # that the vision model missed but the text describes.
    vision_edge_keys = {(e.source, e.target) for e in vision_graph.edges}
    merged_edges = [e.model_dump(mode="json") for e in vision_graph.edges]
    for te in text_graph.edges:
        if (te.source, te.target) not in vision_edge_keys:
            merged_edges.append(te.model_dump(mode="json"))
            vision_edge_keys.add((te.source, te.target))

    # Ensure all merged edges reference known node ids
    node_ids = {n["id"] for n in merged_nodes}
    merged_edges = [
        e for e in merged_edges
        if e.get("source") in node_ids and e.get("target") in node_ids
    ]

    # Groups come from the vision path only; text never overrides them.
    merged_groups = [g.model_dump(mode="json") for g in vision_graph.groups]

    merged = ArchitectureGraph(
        nodes=merged_nodes,
        edges=merged_edges,
        groups=merged_groups,
        version=GRAPH_PAYLOAD_VERSION,
        source="merged",
        layout_mode="figure",
        figure_id=vision_graph.figure_id,
        figure_page=vision_graph.figure_page,
        confidence=vision_graph.confidence,
        verification=vision_graph.verification,
        paper_title=vision_graph.paper_title or text_graph.paper_title,
        metadata={
            "merged_from": "vision+text",
            "vision_node_count": len(vision_graph.nodes),
            "text_node_count": len(text_graph.nodes),
        },
    )
    return merged


def should_render_reconstruction(graph: ArchitectureGraph | None) -> bool:
    """Single source of truth for whether a graph is renderable as a
    figure-faithful reconstruction.

    Returns True only when the graph is verified vision-derived with no
    hallucinations, source=vision/merged, and the confidence clears the
    threshold. Any other case is downgraded by the caller to auto-layout
    or removed entirely.
    """
    if graph is None:
        return False
    if graph.layout_mode != "figure":
        return False
    if graph.source not in ("vision", "merged"):
        return False
    return _is_verified_vision_graph(graph)


# Convenience alias for pipeline imports
def is_verified_vision_graph(graph: ArchitectureGraph | None) -> bool:
    return _is_verified_vision_graph(graph)
