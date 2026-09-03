"""Graph schema for architecture extraction.

The base schema is intentionally permissive so it can carry both the legacy
``{nodes, edges}`` payload and the rich figure-faithful payload produced by
the multimodal extraction pipeline.

Two layers of fields coexist:

* Required core fields (``id``, ``label``, ``type`` on nodes; ``source``,
  ``target`` on edges) keep the existing call sites and tests working.
* Optional figure-faithful fields (``bbox``, ``style``, ``group_id``,
  ``path_points``, ``provenance``, ``confidence``, ...) carry visual geometry
  when a verified figure is available, and gracefully degrade when not.

The whole module is import-safe in CI: there is no I/O and no LLM dependency.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Taxonomies ────────────────────────────────────────────────────────
NodeType = Literal[
    "input", "output", "embedding", "attention",
    "feedforward", "normalization", "block",
    "router", "expert", "ssm", "gate", "cache", "other",
]

NodeShape = Literal[
    "rect", "rounded", "ellipse", "diamond", "cylinder",
    "parallelogram", "trapezoid", "custom",
]

EdgeTypeGraph = Literal[
    "data_flow",
    "residual_skip",
    "cross_attention",
    "gated_modulation",
    "hierarchical_containment",
    "recurrent_state",
    "custom",
]

EdgeStyleName = Literal[
    "solid", "dashed", "dotted", "residual",
    "beam", "highlight", "custom", "default",
]

GraphSource = Literal["text", "vision", "merged", "auto", "legacy"]
LayoutMode = Literal["auto", "figure", "legacy"]


# ── Geometry primitives ──────────────────────────────────────────────
class Point(BaseModel):
    """2D point in normalized [0, 1] source-image coordinates."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in normalized [0, 1] source-image coordinates.

    The union of ``x + w`` and ``y + h`` may not exceed 1.0; otherwise the
    geometry is malformed and validation fails.
    """
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_in_canvas(self) -> "BoundingBox":
        # Allow a tiny epsilon to absorb floating-point quantization from
        # vision models, but reject clearly out-of-canvas boxes.
        if self.x + self.w > 1.0 + 1e-6 or self.y + self.h > 1.0 + 1e-6:
            raise ValueError(
                f"BoundingBox extends beyond normalized canvas: "
                f"x+w={self.x + self.w:.4f}, y+h={self.y + self.h:.4f}"
            )
        return self


# ── Provenance & verification ────────────────────────────────────────
class GraphProvenance(BaseModel):
    """Source provenance for a graph element."""
    source: str = Field(default="text", description="text|vision|merged")
    paper_title: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    figure_id: Optional[str] = None
    figure_page: Optional[int] = None
    citation: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


def _normalize_node_shape(val: Any) -> str:
    if val is None or val == "":
        return "rounded"
    if not isinstance(val, str):
        return "rounded"
    s = val.strip().lower()
    valid_shapes = {"rect", "rounded", "ellipse", "diamond", "cylinder", "parallelogram", "trapezoid", "custom"}
    if s in valid_shapes:
        return s
    if s in ("box", "square", "rectangle"):
        return "rect"
    if s in ("circle", "oval"):
        return "ellipse"
    return "rounded"


class GraphNodeStyle(BaseModel):
    """Visual style overrides for a node. All fields optional; renderer falls
    back to taxonomy colors and rounded-rect shape when missing."""
    fill: Optional[str] = None
    border: Optional[str] = None
    text: Optional[str] = None
    shape: NodeShape = "rounded"
    raw: Optional[dict[str, Any]] = None  # raw style data from extractor

    @field_validator("shape", mode="before")
    @classmethod
    def _coerce_shape(cls, value: Any) -> str:
        return _normalize_node_shape(value)


class GraphEdgeStyle(BaseModel):
    """Visual style overrides for an edge. ``raw`` preserves the extractor's
    original style string (e.g. ``"skip-connection"``) for debug display."""
    stroke: Optional[str] = None
    width: Optional[float] = Field(default=None, gt=0.0)
    dasharray: Optional[str] = None
    name: EdgeStyleName = "default"
    raw: Optional[dict[str, Any]] = None


class GraphGroup(BaseModel):
    """Visual grouping of nodes in a figure (e.g. an encoder-block boundary)."""
    id: str
    label: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    color: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphNode(BaseModel):
    """Architecture graph node.

    Required: ``id``, ``label``, ``type``.
    Everything else is optional; new fields are populated when the figure
    extractor or text enricher can supply them confidently.
    """
    id: str
    label: str
    type: NodeType  # validator normalizes unknown values to "other"

    # ── Semantic enrichment (text-derived) ──
    description: Optional[str] = None
    how_it_works: Optional[str] = None
    math_formula: Optional[str] = None
    paper_term: Optional[str] = None
    parameters: Optional[str] = None
    flops: Optional[str] = None

    # ── Visual extraction (figure-derived) ──
    group_id: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    shape: Optional[NodeShape] = None
    style: Optional[GraphNodeStyle] = None

    # ── Confidence & provenance ──
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Optional[GraphProvenance] = None

    @field_validator("shape", mode="before")
    @classmethod
    def _coerce_shape(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return _normalize_node_shape(value)

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_unknown_type(cls, value: Any) -> str:
        """Map known aliases to the canonical taxonomy; anything unknown
        becomes ``"other"`` so the renderer can still draw it safely while
        preserving ``paper_term`` for the sidebar."""
        if value is None or value == "":
            return "other"
        if not isinstance(value, str):
            return "other"
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")

        canonical = {
            "input", "output", "embedding", "attention",
            "feedforward", "normalization", "block",
            "router", "expert", "ssm", "gate", "cache", "other",
        }
        if normalized in canonical:
            return normalized

        alias = {
            "norm": "normalization",
            "layer_norm": "normalization",
            "rms_norm": "normalization",
            "layernorm": "normalization",
            "mlp": "feedforward",
            "ffn": "feedforward",
            "feed_forward": "feedforward",
            "linear": "other",
            "dense": "other",
            "projection": "other",
            "conv": "other",
            "conv1d": "other",
            "conv2d": "other",
            "encoder": "block",
            "decoder": "block",
            "transformer_block": "block",
            "ssm_block": "ssm",
            "selective_scan": "ssm",
            "cross_attention": "attention",
            "multi_head_attention": "attention",
            "mha": "attention",
            "self_attention": "attention",
            "residual": "other",
            "residual_add": "other",
            "add": "other",
            "embedding_layer": "embedding",
            "token_embedding": "embedding",
        }
        if normalized in alias:
            return alias[normalized]
        return "other"


class GraphEdge(BaseModel):
    """Architecture graph edge.

    Required: ``source``, ``target``. ``label`` is the existing optional
    human-readable description. The new ``type`` field uses the
    canonical taxonomy; ``semantic`` is an open string for paper-specific
    descriptions ("skip", "top-k", ...) that the renderer can surface in
    the sidebar without overloading ``type``.
    """
    source: str
    target: str
    label: Optional[str] = None

    # Semantic
    type: Optional[EdgeTypeGraph] = None
    semantic: Optional[str] = None

    # Visual (figure-derived)
    path_points: Optional[list[Point]] = None
    source_anchor: Optional[Point] = None
    target_anchor: Optional[Point] = None
    style: Optional[GraphEdgeStyle] = None
    raw_style: Optional[dict[str, Any]] = None


class GraphVerification(BaseModel):
    """Result of the figure-faithfulness verification pass on a vision graph.

    A graph is only marked ``layout_mode='figure'`` when ``valid`` is True
    AND ``confidence`` clears the configured threshold AND no hallucinations
    are listed.
    """
    valid: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucinations: list[str] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)
    node_verification: list[dict[str, Any]] = Field(default_factory=list)
    edge_verification: list[dict[str, Any]] = Field(default_factory=list)
    reason: Optional[str] = None


class ArchitectureGraph(BaseModel):
    """Top-level architecture graph payload.

    Backward compatible: a payload of the form
    ``{"nodes": [{"id":..., "label":..., "type":...}], "edges": [...]}``
    still validates. Rich payloads add ``groups``, ``bbox``, ``path_points``,
    ``provenance``, ``verification``, ``layout_mode``, etc.
    """
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    # Figure-faithful metadata (all optional)
    version: Optional[str] = None
    source: Optional[GraphSource] = None
    figure_id: Optional[str] = None
    figure_page: Optional[int] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    verification: Optional[GraphVerification] = None
    groups: list[GraphGroup] = Field(default_factory=list)
    layout_mode: Optional[LayoutMode] = None
    paper_title: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


# ── Public constants ─────────────────────────────────────────────────
GRAPH_PAYLOAD_VERSION = "2.0"

# Below this verification confidence, a vision-derived graph is NOT marked
# ``layout_mode='figure'`` even if extraction succeeded. Keep the
# threshold on the conservative side: trust-first beats polish.
VISION_CONFIDENCE_THRESHOLD = 0.65
