"""Multimodal architecture extraction from paper figures via a vision LLM.

Groq is text-only, so the architecture diagram cannot be derived from images there.
This service sends the figure (base64) to a vision-capable model (Mistral by default,
reusing the OCR key) and asks for the same node/edge graph schema used by the text path.
"""

import json
import time
import requests
from typing import Optional

from app.core.config import settings
from app.models.graph import ArchitectureGraph


VISION_GRAPH_PROMPT = """You are an expert at reading neural-network architecture diagrams.

Extract the model's components and their connections from the provided figure.

CRITICAL Rules:
- Only include components that are visibly present in the figure.
- Edges represent arrows / data-flow lines drawn in the figure.
- Do NOT invent components or connections not shown.
- Keep labels short and human-readable.

Output valid JSON matching this schema:
{
  "nodes": [{"id": "unique_id", "label": "Human Readable Name", "type": "node_type"}],
  "edges": [{"source": "node_id", "target": "node_id", "label": "optional description"}]
}

Node types: input, output, embedding, attention, feedforward, normalization, block, router, expert, ssm, gate, cache, other"""


class FigureExtractor:
    """Extracts an ArchitectureGraph from a single paper figure."""

    def extract_architecture(
        self, figure, paper_title: str = ""
    ) -> Optional[ArchitectureGraph]:
        """Extract the architecture graph from a figure. Returns None on any failure."""
        if not settings.VISION_API_KEY:
            print("[WARN] Vision API key not configured; skipping image-based extraction")
            return None
        if not figure or not figure.base64:
            return None

        image_url = (
            figure.base64
            if figure.base64.startswith("data:")
            else f"data:image/jpeg;base64,{figure.base64}"
        )

        messages = [
            {"role": "system", "content": VISION_GRAPH_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Paper: {paper_title}\n"
                            "Extract the architecture graph shown in this figure as JSON."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        try:
            return self._call_and_parse(messages)
        except Exception as e:
            print(f"[ERROR] Vision architecture extraction failed: {e}")
            return None

    def _call_and_parse(self, messages: list) -> ArchitectureGraph:
        payload = {
            "model": settings.VISION_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(settings.LLM_MAX_RETRIES):
            try:
                resp = requests.post(
                    settings.VISION_API_URL,
                    headers={
                        "Authorization": f"Bearer {settings.VISION_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 429:
                    time.sleep(settings.LLM_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = json.loads(resp.json()["choices"][0]["message"]["content"])
                return ArchitectureGraph(**data)
            except requests.exceptions.RequestException as e:
                if attempt < settings.LLM_MAX_RETRIES - 1:
                    time.sleep(settings.LLM_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                raise
        raise Exception("Vision extraction failed after retries")


figure_extractor = FigureExtractor()
