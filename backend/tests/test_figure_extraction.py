"""Vision + verification tests (Todos 2 + 3 + 12).

Verifies that:
* _parse_vision_payload / _parse_verification_payload produce well-typed
  ArchitectureGraph / GraphVerification values.
* is_verified_vision_graph correctly gates figure-faithful rendering.
* extract_architecture() and verify_figure_graph() never raise through
  the pipeline (network errors, malformed JSON, missing key).
"""
from unittest.mock import patch, MagicMock

import pytest

from app.models.graph import (
    ArchitectureGraph, GraphNode, GraphVerification,
    GRAPH_PAYLOAD_VERSION,
)
from app.services.figure_extraction import (
    FigureExtractor, _parse_vision_payload, _parse_verification_payload,
    is_verified_vision_graph,
)


class _FakeFigure:
    def __init__(self, fig_id="fig:1", page=2):
        self.id = fig_id
        self.page = page
        self.base64 = "data:image/png;base64,FAKE"
        self.caption = "Architecture diagram"


# ── parser tests ────────────────────────────────────────────────────
def test_rich_vision_payload_parses():
    payload = {
        "version": "2.0",
        "source": "vision",
        "layout_mode": "figure",
        "groups": [{"id": "g1", "label": "G", "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}],
        "nodes": [
            {"id": "a", "label": "A", "type": "attention",
             "bbox": {"x": 0.1, "y": 0.1, "w": 0.15, "h": 0.1}},
            {"id": "b", "label": "B", "type": "feedforward",
             "bbox": {"x": 0.3, "y": 0.1, "w": 0.15, "h": 0.1}},
        ],
        "edges": [{"source": "a", "target": "b"}],
    }
    g = _parse_vision_payload(payload, paper_title="Demo", figure_id="fig:1", figure_page=2)
    assert g.source == "vision"
    assert g.layout_mode == "figure"
    assert g.figure_id == "fig:1"
    assert g.figure_page == 2
    assert g.paper_title == "Demo"
    assert len(g.nodes) == 2
    assert len(g.groups) == 1
    # All nodes tagged with vision provenance
    assert all(n.provenance and n.provenance.source == "vision" for n in g.nodes)


def test_legacy_vision_payload_parses():
    payload = {"nodes": [{"id": "a", "label": "A", "type": "attention"}], "edges": []}
    g = _parse_vision_payload(payload)
    assert len(g.nodes) == 1
    assert g.source == "vision"  # default
    assert g.layout_mode == "figure"  # default


def test_malformed_bbox_rejected():
    payload = {
        "nodes": [{"id": "x", "label": "X", "type": "other",
                    "bbox": {"x": 0.9, "y": 0.1, "w": 0.5, "h": 0.1}}],  # out of canvas
        "edges": [],
    }
    with pytest.raises(Exception):
        _parse_vision_payload(payload)


# ── trust-first gate ────────────────────────────────────────────────
def test_is_verified_vision_graph_true_for_valid():
    g = ArchitectureGraph(
        source="vision", layout_mode="figure",
        version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="attention")],
        edges=[],
        verification=GraphVerification(valid=True, confidence=0.9),
    )
    assert is_verified_vision_graph(g) is True


@pytest.mark.parametrize("verification,label", [
    (None, "no verification"),
    (GraphVerification(valid=False, confidence=0.9), "valid=False"),
    (GraphVerification(valid=True, confidence=0.3), "low confidence"),
    (GraphVerification(valid=True, confidence=0.9, hallucinations=["x"]), "hallucination"),
])
def test_is_verified_vision_graph_rejects(verification, label):
    g = ArchitectureGraph(
        source="vision", layout_mode="figure", version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="other")],
        edges=[], verification=verification,
    )
    assert is_verified_vision_graph(g) is False, f"should reject: {label}"


def test_is_verified_vision_graph_rejects_non_vision_source():
    g = ArchitectureGraph(
        source="text", layout_mode="auto", version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="other")], edges=[],
        verification=GraphVerification(valid=True, confidence=0.95),
    )
    assert is_verified_vision_graph(g) is False


def test_is_verified_vision_graph_rejects_none():
    assert is_verified_vision_graph(None) is False


# ── extract_architecture (mocked) ──────────────────────────────────
def test_extract_architecture_happy():
    extractor = FigureExtractor()
    payload = {
        "version": "2.0", "source": "vision", "layout_mode": "figure",
        "nodes": [{"id": "in", "label": "In", "type": "input",
                    "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.05}},
                   {"id": "out", "label": "Out", "type": "output",
                    "bbox": {"x": 0.4, "y": 0.1, "w": 0.1, "h": 0.05}}],
        "edges": [{"source": "in", "target": "out"}],
    }
    with patch.object(extractor, "_call_and_parse", return_value=payload):
        result = extractor.extract_architecture(_FakeFigure(), "Demo")
    assert result is not None
    assert result.figure_id == "fig:1"
    assert result.figure_page == 2


def test_extract_architecture_returns_none_on_malformed():
    extractor = FigureExtractor()
    with patch.object(extractor, "_call_and_parse", return_value={"nodes": [{"label": "NoID", "type": "other"}]}):
        result = extractor.extract_architecture(_FakeFigure(), "Demo")
    assert result is None


def test_extract_architecture_returns_none_on_network_error():
    extractor = FigureExtractor()
    with patch.object(extractor, "_call_and_parse", side_effect=RuntimeError("network")):
        result = extractor.extract_architecture(_FakeFigure(), "Demo")
    assert result is None


def test_extract_architecture_returns_none_when_figure_missing():
    extractor = FigureExtractor()
    result = extractor.extract_architecture(None, "Demo")
    assert result is None


# ── verify_figure_graph (mocked) ───────────────────────────────────
def test_verify_happy():
    extractor = FigureExtractor()
    graph = ArchitectureGraph(
        source="vision", layout_mode="figure", version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="other")], edges=[],
    )
    v_payload = {"valid": True, "confidence": 0.9, "hallucinations": [], "missing_elements": []}
    with patch.object(extractor, "_call_and_parse", return_value=v_payload):
        v = extractor.verify_figure_graph(_FakeFigure(), graph, "Demo")
    assert v.valid is True
    assert v.confidence == 0.9


def test_verify_hallucinated():
    extractor = FigureExtractor()
    graph = ArchitectureGraph(
        source="vision", layout_mode="figure", version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="other")], edges=[],
    )
    v_payload = {"valid": True, "confidence": 0.9, "hallucinations": ["a"], "missing_elements": []}
    with patch.object(extractor, "_call_and_parse", return_value=v_payload):
        v = extractor.verify_figure_graph(_FakeFigure(), graph, "Demo")
    assert v.hallucinations == ["a"]


def test_verify_low_confidence():
    extractor = FigureExtractor()
    graph = ArchitectureGraph(
        source="vision", layout_mode="figure", version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="other")], edges=[],
    )
    v_payload = {"valid": True, "confidence": 0.3, "hallucinations": [], "missing_elements": []}
    with patch.object(extractor, "_call_and_parse", return_value=v_payload):
        v = extractor.verify_figure_graph(_FakeFigure(), graph, "Demo")
    assert v.confidence == 0.3
    assert v.valid is True  # valid=True but trust-first rejects low confidence downstream


def test_verify_network_error_returns_invalid():
    extractor = FigureExtractor()
    graph = ArchitectureGraph(
        source="vision", layout_mode="figure", version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="other")], edges=[],
    )
    with patch.object(extractor, "_call_and_parse", side_effect=RuntimeError("503")):
        v = extractor.verify_figure_graph(_FakeFigure(), graph, "Demo")
    assert v.valid is False
    assert v.confidence == 0.0


def test_verify_malformed_payload_returns_invalid():
    extractor = FigureExtractor()
    graph = ArchitectureGraph(
        source="vision", layout_mode="figure", version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="other")], edges=[],
    )
    with patch.object(extractor, "_call_and_parse", return_value={"valid": "yes"}):
        v = extractor.verify_figure_graph(_FakeFigure(), graph, "Demo")
    assert v.valid is False


def test_verify_missing_graph_returns_invalid():
    extractor = FigureExtractor()
    v = extractor.verify_figure_graph(_FakeFigure(), None, "Demo")
    assert v.valid is False


# ── regression: old unverified-vision acceptance is gone ──────────
def test_old_len_nodes_acceptance_is_gone():
    """The old pipeline acceptance was ``vision_graph and len(nodes) >= 2``
    with no verification. is_verified_vision_graph() rejects any graph
    that lacks valid verification metadata, regardless of node count."""
    g = ArchitectureGraph(
        source="vision", layout_mode="figure", version=GRAPH_PAYLOAD_VERSION,
        nodes=[GraphNode(id="a", label="A", type="other"),
                GraphNode(id="b", label="B", type="other")],  # 2 nodes
        edges=[], verification=None,  # no verification at all
    )
    # The trust-first rule: 2 nodes alone is not enough.
    assert is_verified_vision_graph(g) is False
