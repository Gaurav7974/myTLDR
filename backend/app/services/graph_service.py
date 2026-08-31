"""Graph extraction and verification service."""

import json
from app.config import settings
from app.models.graph import ArchitectureGraph
from app.services.llm import groq_client


GRAPH_EXTRACTION_PROMPT = """You are an expert at extracting model architectures from research papers.

Given the architecture/method section of a paper, extract the model's components and their connections as a graph.

CRITICAL Rules:
- Use ONLY the node types provided in the schema
- Do NOT include components not described in the text
- Do NOT invent connections
- Do NOT add generic "Input" or "Output" nodes unless the paper explicitly describes them as architectural components
- Only include nodes for components that the paper specifically names and describes
- Edges should only represent data flow explicitly described in the text
- If you extract 2 or more nodes, you MUST include edges showing how they connect
- If the text doesn't describe a clear sequential/structural architecture, return an empty graph

Output valid JSON matching this schema:
{{
  "nodes": [{{"id": "unique_id", "label": "Human Readable Name", "type": "node_type"}}],
  "edges": [{{"source": "node_id", "target": "node_id", "label": "optional description"}}]
}}

Node types: input, output, embedding, attention, feedforward, normalization, block, router, expert, ssm, gate, cache, other

Architecture text:
---
{text}
---"""


VERIFICATION_PROMPT = """You are an expert ML researcher verifying an architecture graph extracted from a paper.

For each node in the graph, find the exact sentence(s) in the source text that justify it. If no citation exists, flag it as a HALLUCINATION.
For each edge, do the same.
Then list any major components from the source text that are MISSING from the graph.

Output valid JSON:
{{
  "nodes_verification": [
    {{"node_id": "id", "label": "label", "citation": "exact text excerpt or 'NONE'"}}
  ],
  "edges_verification": [
    {{"edge": "source->target", "citation": "exact text excerpt or 'NONE'"}}
  ],
  "missing_components": ["component described in text but not in graph"],
  "hallucinations": ["node/edge not supported by text"],
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


class GraphService:
    """Extracts and verifies architecture graphs."""
    
    def extract_graph(self, architecture_text: str) -> ArchitectureGraph | None:
        """Extract architecture graph from text."""
        prompt = GRAPH_EXTRACTION_PROMPT.format(text=architecture_text[:settings.MAX_SECTION_CHARS])
        
        try:
            data = groq_client.extract_json(
                messages=[{"role": "user", "content": prompt}],
                model=settings.MISTRAL_LLM_MODEL
            )
            return ArchitectureGraph(**data)
        except Exception as e:
            print(f"[ERROR] Graph extraction failed: {e}")
            return None
    
    def verify_graph(self, source_text: str, graph: ArchitectureGraph) -> dict:
        """Verify graph against source text."""
        graph_json = json.dumps(graph.model_dump(), indent=2)
        prompt = VERIFICATION_PROMPT.format(
            source_text=source_text[:settings.MAX_SECTION_CHARS],
            graph_json=graph_json
        )
        
        try:
            return groq_client.extract_json(
                messages=[{"role": "user", "content": prompt}],
                model=settings.MISTRAL_LLM_MODEL
            )
        except Exception as e:
            print(f"[ERROR] Verification failed: {e}")
            return {"valid": False, "reason": f"Verification error: {e}"}


graph_service = GraphService()
