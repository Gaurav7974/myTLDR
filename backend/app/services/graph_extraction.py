"""Text graph extraction and verification service.

The text path adds cited SEMANTIC ENRICHMENT (description, math, paper_term)
to nodes and edges. It does NOT control visual geometry — the vision path
owns bbox, path_points, group bbox, and colors. The merger in
``pipeline.py`` is the only place the two are combined.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.models.graph import (
    ArchitectureGraph,
    GRAPH_PAYLOAD_VERSION,
    GraphProvenance,
)
from app.services.llm import groq_client


GRAPH_EXTRACTION_PROMPT = """You are an expert at extracting model architectures from research papers.

Given the architecture/method section of a paper, extract the model's
components and their connections as a graph. For each node and edge, only
include claims that you can CITE from the source text — the verification
step will reject anything you cannot justify.

CRITICAL Rules:
- Use ONLY the node types provided in the schema.
- Do NOT include components not described in the text.
- Do NOT invent connections.
- Do NOT add generic "Input" or "Output" nodes unless the paper explicitly
  describes them as architectural components.
- Only include nodes for components that the paper specifically names and
  describes.
- Edges should only represent data flow explicitly described in the text.
- If you extract 2 or more nodes, you MUST include edges showing how they
  connect.
- If the text doesn't describe a clear sequential/structural architecture,
  return an empty graph.
- For each node, fill in ``description`` (1-2 sentences paraphrasing the
  paper), ``how_it_works`` (short procedural note), and ``math_formula``
  ONLY when the text provides a real formula. Leave fields empty
  (do NOT guess) when the text is silent.
- For each node, set ``paper_term`` to the exact phrase the paper uses
  for the component (e.g. "selective state-space layer").
- For each edge, set ``type`` to one of: data_flow, residual_skip,
  cross_attention, gated_modulation, recurrent_state, hierarchical_containment,
  custom. Use ``semantic`` to preserve a paper-specific term like "skip"
  or "top-k routing" verbatim.

Output valid JSON matching this schema:
{{
  "version": "2.0",
  "source": "text",
  "layout_mode": "auto",
  "paper_title": "{paper_title}",
  "nodes": [
    {{
      "id": "unique_id",
      "label": "Human Readable Name",
      "type": "node_type",
      "description": "1-2 sentence paraphrase (cited)",
      "how_it_works": "short procedural note (cited)",
      "math_formula": "LaTeX or empty",
      "paper_term": "exact paper phrase",
      "provenance": {{"source": "text", "section": "section heading"}}
    }}
  ],
  "edges": [
    {{
      "source": "node_id", "target": "node_id",
      "label": "optional description",
      "type": "data_flow|residual_skip|cross_attention|gated_modulation|recurrent_state|hierarchical_containment|custom",
      "semantic": "paper-specific edge term"
    }}
  ]
}}

Node types: input, output, embedding, attention, feedforward, normalization,
block, router, expert, ssm, gate, cache, other.

Architecture text:
---
{text}
---"""


VERIFICATION_PROMPT = """You are an expert ML researcher verifying an architecture graph extracted from a paper.

For each node in the graph, find the exact sentence(s) in the source text
that justify its existence AND any enriched field (description, how_it_works,
math_formula, paper_term). If a field is filled in but you cannot find a
citation, flag it as a HALLUCINATION. If a node is real but its description
or formula is invented, flag the FIELD, not the node, as a hallucination.

For each edge, do the same. Reject edge types that are not supported by
the text (e.g. do not mark an edge ``residual_skip`` unless the text
explicitly describes a skip connection).

Then list any major components from the source text that are MISSING from
the graph.

Output valid JSON:
{{
  "node_verification": [
    {{"node_id": "id", "label": "label",
      "citation_node": "exact text excerpt for the node or 'NONE'",
      "citation_description": "exact text excerpt for description or 'NONE'",
      "citation_math": "exact text excerpt for math or 'NONE'",
      "status": "valid|hallucinated|partial"}}
  ],
  "edge_verification": [
    {{"edge": "source->target", "citation": "exact text excerpt or 'NONE'",
     "status": "valid|hallucinated"}}
  ],
  "missing_components": ["component described in text but not in graph"],
  "hallucinations": ["list of node ids, field names, or edges not supported by text"],
  "valid": true or false,
  "reason": "brief explanation"
}}

Source text:
---
{source_text}
---

Extracted graph:
{graph_json}
---"""


class GraphExtractor:
    """Extracts and verifies architecture graphs from paper text."""

    def extract_graph(
        self, architecture_text: str, paper_title: str = ""
    ) -> ArchitectureGraph | None:
        """Extract architecture graph from text.

        Returns a text-source graph (layout_mode='auto') carrying optional
        semantic enrichment. Returns None on any failure.
        """
        prompt = GRAPH_EXTRACTION_PROMPT.format(
            paper_title=paper_title or "Unknown",
            text=architecture_text[:settings.MAX_SECTION_CHARS],
        )

        try:
            data = groq_client.extract_json(
                messages=[{"role": "user", "content": prompt}],
                model=settings.MISTRAL_LLM_MODEL,
            )
        except Exception as e:
            print(f"[ERROR] Graph extraction call failed: {e}")
            return None

        if not isinstance(data, dict):
            return None
        # Ensure source/layout_mode/version defaults even if model omits them
        data.setdefault("version", GRAPH_PAYLOAD_VERSION)
        data.setdefault("source", "text")
        data.setdefault("layout_mode", "auto")
        if paper_title and not data.get("paper_title"):
            data["paper_title"] = paper_title

        try:
            graph = ArchitectureGraph(**data)
        except Exception as e:
            print(f"[WARN] Text graph payload failed validation: {e}")
            return None

        # Tag every node with text provenance if the model did not.
        for node in graph.nodes:
            if node.provenance is None:
                node.provenance = GraphProvenance(
                    source="text",
                    paper_title=paper_title or None,
                )
        return graph

    def verify_graph(
        self, source_text: str, graph: ArchitectureGraph, paper_title: str = ""
    ) -> dict[str, Any]:
        """Verify graph against source text.

        Returns the verifier's JSON dict (kept as a dict rather than a
        Pydantic model so the pipeline can log the raw reason). Always
        returns a dict with at least ``valid`` and ``reason`` keys.
        """
        graph_json = json.dumps(graph.model_dump(mode="json"), indent=2, default=str)
        prompt = VERIFICATION_PROMPT.format(
            source_text=source_text[:settings.MAX_SECTION_CHARS],
            graph_json=graph_json,
        )

        try:
            return groq_client.extract_json(
                messages=[{"role": "user", "content": prompt}],
                model=settings.MISTRAL_LLM_MODEL,
            )
        except Exception as e:
            print(f"[ERROR] Text verification failed: {e}")
            return {
                "valid": False,
                "reason": f"Verification error: {e}",
                "hallucinations": [],
                "missing_components": [],
            }


graph_extractor = GraphExtractor()
