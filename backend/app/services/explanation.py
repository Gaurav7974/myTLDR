"""Explanation generation service with structured pedagogical breakdowns."""

import json
import re
from typing import Dict, Any
from app.core.config import settings
from app.core.logger import get_logger
log = get_logger(__name__)
from app.services.llm import groq_client
from app.models.summary import StructuredPaperSummary

STRUCTURED_EXPLANATION_SYSTEM_PROMPT = """You are a research-paper summarization engine for an academic visualization platform.
Your task is NOT to write a general article or create arbitrary sections.
Your task is to populate a fixed UI schema with strict word limits, factual precision, and source provenance.

RULES:
1. Use only information supported by the provided paper text.
2. Never invent numerical values, architecture details, results, dates, datasets, or claims.
3. Every factual claim must include provenance (page, section, figure/table if available).
4. Distinguish novel contributions from standard/adopted architectural techniques.
5. Do not repeat the same idea across sections.
6. Do not create additional sections.
7. Prefer concrete facts, numbers, architecture details, and experimentally demonstrated findings over generic prose.
8. If information is unavailable, return null rather than guessing.

SECTION CONSTRAINTS:
- OVERVIEW:
  - one_sentence: 25–40 words explaining the central contribution.
  - problem: 30–50 words explaining the bottleneck/limitation addressed.
  - solution: 30–60 words explaining what the authors propose.
  - why_it_matters: 30–50 words explaining significance supported by the paper.
- KEY_SPECS: 4–8 items (e.g. layers, heads, d_model, inner dimension, parameters, training hours). Exact numbers only with source page.
- CONTRIBUTIONS: 3–5 items maximum (20–35 words each). Category MUST be one of: "novel", "architectural", "methodological", "experimental", "supporting".
- KEY_FINDINGS: 2–4 items prioritizing experimentally demonstrated results with exact metrics.
- LIMITATIONS: 2–4 items explicitly discussed by the authors or evident from results.

You must respond ONLY with a valid, clean JSON object matching this schema:
{
  "overview": {
    "one_sentence": "...",
    "problem": "...",
    "solution": "...",
    "why_it_matters": "..."
  },
  "key_specs": [
    {
      "label": "Encoder & Decoder Layers",
      "value": "N = 6",
      "source": { "page": 3, "section": "3.1" }
    }
  ],
  "contributions": [
    {
      "title": "Scaled Dot-Product & Multi-Head Attention",
      "summary": "Runs multiple attention operations in parallel, projecting queries, keys, and values into subspaces to attend across different relationships.",
      "category": "novel",
      "importance": "Replaces sequential recurrence with constant O(1) path length.",
      "source": { "page": 4, "section": "3.2" }
    }
  ],
  "key_findings": [
    {
      "title": "WMT 2014 English-to-German Translation",
      "summary": "Achieved new state-of-the-art score improving over existing best models including ensembles while training in 3.5 days on 8 GPUs.",
      "metric_value": "28.4 BLEU (+2.0 over SOTA)",
      "source": { "page": 8, "section": "5.1" }
    }
  ],
  "limitations": [
    {
      "title": "Quadratic Memory Complexity with Sequence Length",
      "summary": "Pairwise self-attention matrices require O(N^2) memory storage and computational bandwidth.",
      "details": "Requires sparse attention or memory optimizations for extremely long document contexts.",
      "source": { "page": 6, "section": "3.5" }
    }
  ]
}

DO NOT include markdown backticks or text outside the JSON. Return purely raw JSON."""


class ExplanationService:
    """Generates structured pedagogical explanations, LaTeX equations, and PyTorch code from research papers."""

    def generate(self, paper_text: str, difficulty: str = "intermediate") -> Dict[str, Any]:
        """Generate structured explanation adhering to pedagogical schema."""
        trimmed_text = paper_text[:settings.MAX_SECTION_CHARS]
        user_prompt = f"""Paper Content (Difficulty: {difficulty}):
---
{trimmed_text}
---

Generate the complete structured summary JSON according to the schema."""

        try:
            raw_response = groq_client.call(
                messages=[
                    {"role": "system", "content": STRUCTURED_EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=settings.MISTRAL_LLM_MODEL,
                temperature=0.1,
                # Force JSON mode so the model cannot drift into prose/markdown.
                response_format={"type": "json_object"},
            )

            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            # Validate through Pydantic
            validated = StructuredPaperSummary(**parsed)
            return validated.model_dump()

        except Exception as e:
            # Do NOT fabricate a plausible-looking summary on failure — return None so the
            # caller can show an honest "summary unavailable" state instead of invented facts.
            log.error(f"[ERROR] Structured explanation generation failed: {e}", exc_info=True)
            return None

    def extract_equations(self, paper_text: str) -> list:
        """Extract top LaTeX equations directly from the paper's method section."""
        trimmed_text = paper_text[:settings.MAX_SECTION_CHARS]
        prompt = f"""Extract the top 2 to 4 PRIMARY mathematical formulas from the methodology of this research paper.
---
{trimmed_text}
---

Respond ONLY with a valid JSON object matching:
{{
  "equations": [
    {{
      "id": "eq-1",
      "title": "Descriptive Formula Name",
      "latex": "y = \\\\sigma(W x + b)",
      "explanation": "Clear plain-English breakdown of what this formula computes.",
      "variables": [
        {{ "symbol": "W", "name": "Weight Matrix", "meaning": "Learned projection parameters" }},
        {{ "symbol": "x", "name": "Input Vector", "meaning": "Feature representation" }}
      ]
    }}
  ]
}}"""
        try:
            res = groq_client.extract_json(
                messages=[
                    {"role": "system", "content": "You are a mathematical AI researcher. Extract clean, valid LaTeX equations with detailed variable explanations."},
                    {"role": "user", "content": prompt}
                ]
            )
            return res.get("equations", [])
        except Exception as e:
            log.warning(f"[WARN] Equation extraction failed: {e}")
            return []

    def generate_pytorch_code(self, paper_text: str, paper_title: str) -> list:
        """Generate minimal core nn.Module (80-120 lines), no training loop."""
        trimmed_text = paper_text[:6000]  # smaller excerpt for concise code
        prompt = f"""Generate ONLY the core PyTorch nn.Module classes for "{paper_title}" — no training loop, no DataLoader, no Trainer.

Paper excerpt:
{trimmed_text}

Respond ONLY with JSON:
{{
  "snippets": [{{
    "title": "model.py",
    "language": "python",
    "code": "import torch\\nimport torch.nn as nn\\n# 80-120 lines max\\nclass Block(nn.Module): ..."
  }}]
}}
Rules: 80-120 lines max, only nn.Module + forward, well-commented, runnable."""
        try:
            res = groq_client.extract_json(
                messages=[
                    {"role": "system", "content": "You are an expert PyTorch engineer. Generate clean, modular, bug-free neural network modules."},
                    {"role": "user", "content": prompt}
                ]
            )
            return res.get("snippets", [])
        except Exception as e:
            log.warning(f"[WARN] PyTorch code generation failed: {e}")
            return []


explanation_service = ExplanationService()

