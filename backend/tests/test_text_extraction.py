"""Text graph extraction + verification tests (Todo 4 + 12)."""
from unittest.mock import patch, MagicMock

import pytest

from app.models.graph import (
    ArchitectureGraph, GraphNode,
    GRAPH_PAYLOAD_VERSION,
)
from app.services.graph_extraction import graph_extractor


def test_text_graph_extracts_rich_semantics():
    payload = {
        "version": GRAPH_PAYLOAD_VERSION,
        "source": "text",
        "layout_mode": "auto",
        "paper_title": "Demo",
        "nodes": [
            {"id": "attn", "label": "Multi-Head Attention", "type": "attention",
             "description": "Scaled dot-product attention.",
             "math_formula": "softmax(QK^T/sqrt(d_k))V",
             "paper_term": "multi-head attention",
             "provenance": {"source": "text", "section": "3.2"}},
        ],
        "edges": [{"source": "attn", "target": "ffn", "type": "data_flow"}],
    }
    mock_client = MagicMock()
    mock_client.extract_json.return_value = payload
    with patch("app.services.graph_extraction.groq_client", mock_client):
        g = graph_extractor.extract_graph(
            "The paper uses multi-head attention.", "Demo"
        )
    assert g is not None
    assert g.source == "text"
    assert g.layout_mode == "auto"
    attn = next(n for n in g.nodes if n.id == "attn")
    assert attn.description and "dot-product" in attn.description
    assert attn.math_formula and "softmax" in attn.math_formula
    assert attn.paper_term == "multi-head attention"


def test_text_graph_legacy_payload_parses():
    payload = {"nodes": [{"id": "a", "label": "A", "type": "attention"}], "edges": []}
    mock_client = MagicMock()
    mock_client.extract_json.return_value = payload
    with patch("app.services.graph_extraction.groq_client", mock_client):
        g = graph_extractor.extract_graph("Random text.", "Demo")
    assert g is not None
    assert g.source == "text"
    assert g.layout_mode == "auto"


def test_text_graph_returns_none_on_malformed_payload():
    payload = {"nodes": [{"label": "NoID", "type": "attention"}]}  # missing id
    mock_client = MagicMock()
    mock_client.extract_json.return_value = payload
    with patch("app.services.graph_extraction.groq_client", mock_client):
        g = graph_extractor.extract_graph("text", "Demo")
    assert g is None


def test_text_graph_returns_none_on_call_exception():
    mock_client = MagicMock()
    mock_client.extract_json.side_effect = RuntimeError("groq down")
    with patch("app.services.graph_extraction.groq_client", mock_client):
        g = graph_extractor.extract_graph("text", "Demo")
    assert g is None


def test_text_graph_returns_none_on_non_dict_reply():
    mock_client = MagicMock()
    mock_client.extract_json.return_value = "not a dict"
    with patch("app.services.graph_extraction.groq_client", mock_client):
        g = graph_extractor.extract_graph("text", "Demo")
    assert g is None


def test_text_graph_empty_is_safe():
    payload = {"nodes": [], "edges": []}
    mock_client = MagicMock()
    mock_client.extract_json.return_value = payload
    with patch("app.services.graph_extraction.groq_client", mock_client):
        g = graph_extractor.extract_graph("Unrelated text.", "Demo")
    assert g is not None
    assert len(g.nodes) == 0


def test_verify_returns_dict_with_required_keys():
    mock_client = MagicMock()
    mock_client.extract_json.return_value = {"valid": True, "reason": "ok"}
    with patch("app.services.graph_extraction.groq_client", mock_client):
        v = graph_extractor.verify_graph(
            "text", ArchitectureGraph(nodes=[], edges=[]), "Demo"
        )
    assert v.get("valid") is True
    assert v.get("reason") == "ok"


def test_verify_handles_call_exception():
    mock_client = MagicMock()
    mock_client.extract_json.side_effect = RuntimeError("503")
    with patch("app.services.graph_extraction.groq_client", mock_client):
        v = graph_extractor.verify_graph(
            "text", ArchitectureGraph(nodes=[], edges=[]), "Demo"
        )
    assert v.get("valid") is False
    assert "Verification error" in v.get("reason", "")


def test_verify_flags_unsupported_math():
    """A graph with an invented math_formula is flagged by the verifier."""
    mock_client = MagicMock()
    mock_client.extract_json.return_value = {
        "valid": False, "reason": "math is unsupported",
        "hallucinations": ["attn.math_formula"],
        "node_verification": [], "edge_verification": [],
        "missing_components": [],
    }
    with patch("app.services.graph_extraction.groq_client", mock_client):
        v = graph_extractor.verify_graph(
            "text", ArchitectureGraph(
                nodes=[GraphNode(id="attn", label="A", type="attention",
                                  math_formula="x = 42")],
                edges=[],
            ), "Demo"
        )
    assert v["valid"] is False
    assert any("math" in h for h in v.get("hallucinations", []))
