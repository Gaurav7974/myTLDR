"""Schema tests for ArchitectureGraph (Todo 1 + part of Todo 12).

Lock the rich-payload fields, backward compatibility, and validation
behavior so regressions are caught at the unit-test level.
"""
import pytest
from pydantic import ValidationError

from app.models.graph import (
    ArchitectureGraph, GraphNode, GraphEdge, BoundingBox, Point,
    GraphGroup, GraphNodeStyle, GraphEdgeStyle, GraphProvenance,
    GraphVerification,
    GRAPH_PAYLOAD_VERSION, VISION_CONFIDENCE_THRESHOLD,
)


def test_legacy_payload_still_validates():
    """The original {nodes, edges} payload must keep working."""
    g = ArchitectureGraph(
        nodes=[
            {"id": "in", "label": "Input", "type": "input"},
            {"id": "attn", "label": "Attention", "type": "attention"},
        ],
        edges=[{"source": "in", "target": "attn", "label": "tokens"}],
    )
    assert len(g.nodes) == 2
    assert len(g.edges) == 1


def test_shape_coercion_default_to_rounded():
    node = GraphNode(id="n1", label="N1", type="attention", style=GraphNodeStyle(shape="default"))
    assert node.style.shape == "rounded"
    node2 = GraphNode(id="n2", label="N2", type="attention", shape="box")
    assert node2.shape == "rect"
    node3 = GraphNode(id="n3", label="N3", type="attention", shape="circle")
    assert node3.shape == "ellipse"


def test_rich_payload_validates_with_all_fields():
    g = ArchitectureGraph(
        version=GRAPH_PAYLOAD_VERSION,
        source="vision",
        layout_mode="figure",
        figure_id="fig:1",
        figure_page=3,
        confidence=0.9,
        verification=GraphVerification(
            valid=True, confidence=0.9, hallucinations=[], missing_elements=[],
        ),
        groups=[GraphGroup(id="enc", label="Encoder",
                            bbox=BoundingBox(x=0.05, y=0.1, w=0.4, h=0.3))],
        nodes=[
            GraphNode(
                id="a", label="A", type="attention",
                description="desc", math_formula="x", paper_term="attention",
                bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.1),
                group_id="enc",
                style=GraphNodeStyle(fill="#fff", border="#000"),
                provenance=GraphProvenance(source="vision"),
            )
        ],
        edges=[GraphEdge(source="a", target="b",
                          path_points=[Point(x=0.1, y=0.1), Point(x=0.2, y=0.2)],
                          style=GraphEdgeStyle(name="residual"))],
    )
    assert g.layout_mode == "figure"
    assert g.groups[0].id == "enc"
    assert g.nodes[0].bbox.x == 0.1
    assert g.edges[0].path_points[0].x == 0.1


def test_unknown_type_normalized_to_other():
    n = GraphNode(id="x", label="X", type="some_new_thing")
    assert n.type == "other"


@pytest.mark.parametrize("alias,expected", [
    ("MLP", "feedforward"),
    ("LayerNorm", "normalization"),
    ("layer_norm", "normalization"),
    ("MHA", "attention"),
    ("multi_head_attention", "attention"),
    ("rms_norm", "normalization"),
    ("encoder", "block"),
])
def test_type_aliases(alias, expected):
    n = GraphNode(id="x", label="X", type=alias)
    assert n.type == expected


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        GraphNode(id="x", label="X", type="other", confidence=2.0)
    with pytest.raises(ValidationError):
        GraphNode(id="x", label="X", type="other", confidence=-0.1)


def test_bbox_out_of_canvas_rejected():
    with pytest.raises(ValidationError):
        BoundingBox(x=0.9, y=0.1, w=0.5, h=0.1)  # x+w > 1.0
    with pytest.raises(ValidationError):
        BoundingBox(x=0.1, y=0.1, w=0.0, h=0.1)  # zero width


def test_point_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Point(x=1.5, y=0.5)
    with pytest.raises(ValidationError):
        Point(x=0.1, y=-0.1)


def test_missing_required_id_rejected():
    with pytest.raises(ValidationError):
        GraphNode(label="X", type="other")  # type: ignore[call-arg]


def test_empty_graph_allowed():
    g = ArchitectureGraph(nodes=[], edges=[])
    assert len(g.nodes) == 0
    assert g.layout_mode is None


def test_round_trip_preserves_fields():
    g = ArchitectureGraph(
        version=GRAPH_PAYLOAD_VERSION,
        source="vision",
        layout_mode="figure",
        groups=[GraphGroup(id="g1")],
        nodes=[GraphNode(id="a", label="A", type="attention",
                          bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.1))],
        edges=[],
        verification=GraphVerification(valid=True, confidence=0.9),
    )
    restored = ArchitectureGraph.model_validate(g.model_dump(mode="json"))
    assert restored.layout_mode == "figure"
    assert restored.verification.valid
    assert restored.groups[0].id == "g1"
    assert restored.nodes[0].bbox.x == 0.1


def test_constants_exported():
    """The public version and threshold constants are required by callers."""
    assert isinstance(GRAPH_PAYLOAD_VERSION, str)
    assert GRAPH_PAYLOAD_VERSION == "2.0"
    assert isinstance(VISION_CONFIDENCE_THRESHOLD, float)
    assert 0.0 < VISION_CONFIDENCE_THRESHOLD < 1.0
