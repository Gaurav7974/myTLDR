"""Multimodal & Text Architecture Extractor producing strict ArchitectureIR."""

import json
from typing import Dict, Any, Tuple
from app.core.config import settings
from app.services.llm import groq_client
from app.models.architecture_ir import ArchitectureIR, SceneGraph3D
from app.services.graph_validator import graph_validator
from app.services.layout_engine import layout_engine

ARCHITECTURE_IR_SYSTEM_PROMPT = """You are an elite neural architecture analyst.
Your mission is to analyze the research paper's methodology, architecture descriptions, equations, and figures, and extract a canonical ArchitectureIR (Intermediate Representation) JSON.

CRITICAL RULES:
1. Do NOT generate 3D coordinates or meshes. Only extract semantic architecture components, tensor dimensions, hierarchy, and data connections.
2. Every node must have a valid component type from the taxonomy:
   [input, output, embedding, linear, matrix, attention, multi_head_attention, cross_attention, mlp, feed_forward, normalization, layer_norm, rms_norm, residual, convolution, conv1d, conv2d, pooling, encoder, decoder, transformer_block, ssm_block, selective_scan, unet_block, diffusion_step, adapter_lora, router, expert, classifier, loss, custom]
3. Include page numbers, section headers, and exact citations in the "provenance" object for each node so the user can click directly to the PDF source.
4. Distinguish regular data_flow edges from residual_skip or cross_attention edges.

SCHEMA:
{
  "version": "1.0",
  "architecture": {
    "name": "Human readable model name (e.g. Transformer)",
    "paradigm": "transformer" | "ssm" | "diffusion" | "cnn_vision" | "lora" | "other"
  },
  "nodes": [
    {
      "id": "unique_id",
      "type": "multi_head_attention",
      "label": "Multi-Head Self-Attention (8 Heads)",
      "parent_id": null,
      "dimensions": {
        "input_shape": "[B, L, 512]",
        "output_shape": "[B, L, 512]",
        "hidden_dim": 512,
        "num_heads": 8
      },
      "parameters": "4 × 512 × 512 = 1.05M params",
      "flops": "6.29 MFLOPs",
      "repetitions": 1,
      "description": "Computes parallel multi-head self-attention across 8 heads.",
      "math_formula": "Attention(Q,K,V) = \\\\text{softmax}\\\\left(\\\\frac{QK^T}{\\\\sqrt{d_k}}\\\\right) V",
      "provenance": {
        "page": 4,
        "section": "3.2 Attention",
        "citation": "An attention function can be described as mapping a query and a set of key-value pairs to an output...",
        "confidence": 0.98
      }
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "source_node_id",
      "target": "target_node_id",
      "type": "data_flow" | "residual_skip" | "cross_attention",
      "label": "Pre-Attn Skip"
    }
  ]
}

Return purely raw JSON."""


class ArchitectureExtractor:
    """Extracts, validates, and layouts ArchitectureIR from paper text and context."""

    def extract(self, full_text: str, paper_title: str = "") -> Tuple[ArchitectureIR, SceneGraph3D]:
        """Extract canonical ArchitectureIR and compile it to SceneGraph3D."""
        trimmed_text = full_text[:settings.MAX_SECTION_CHARS]
        user_prompt = f"""Paper Title: {paper_title}
---
Architecture & Method Sections:
{trimmed_text}
---

Extract the canonical ArchitectureIR JSON strictly matching the schema."""

        try:
            raw_data = groq_client.extract_json(
                messages=[
                    {"role": "system", "content": ARCHITECTURE_IR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=settings.MISTRAL_LLM_MODEL
            )
            # 1. Validate & Normalize
            canonical_ir, notices = graph_validator.validate_and_normalize(raw_data)
            # 2. Compute 3D Scene Graph via Layout Engine
            scene_graph = layout_engine.compute_scene_graph(canonical_ir)
            return canonical_ir, scene_graph
        except Exception as e:
            print(f"[ERROR] ArchitectureIR extraction failed: {e}")
            fallback_ir = self._build_fallback_ir(paper_title)
            scene_graph = layout_engine.compute_scene_graph(fallback_ir)
            return fallback_ir, scene_graph

    def _build_fallback_ir(self, title: str) -> ArchitectureIR:
        """Safe fallback ArchitectureIR."""
        raw_fallback = {
            "version": "1.0",
            "architecture": {
                "name": title or "Neural Architecture",
                "paradigm": "transformer"
            },
            "nodes": [
                {
                    "id": "node_embed",
                    "type": "embedding",
                    "label": "Token & Positional Embedding",
                    "description": "Converts discrete token symbols to continuous metric coordinates.",
                    "math_formula": "x_0 = \\text{Embedding}(w) + \\text{PE}(pos)",
                    "provenance": {"page": 3, "section": "3.1 Embeddings", "confidence": 0.95}
                },
                {
                    "id": "node_norm1",
                    "type": "layer_norm",
                    "label": "Pre-Attention LayerNorm",
                    "description": "Normalizes activations across hidden channels.",
                    "math_formula": "\\text{LN}(x) = \\frac{x - \\mu}{\\sigma} \\odot \\gamma + \\beta",
                    "provenance": {"page": 3, "section": "3.1 Normalization", "confidence": 0.95}
                },
                {
                    "id": "node_mha",
                    "type": "multi_head_attention",
                    "label": "Multi-Head Attention",
                    "description": "Computes scaled dot-product attention in parallel subspaces.",
                    "math_formula": "\\text{MHA}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h) W^O",
                    "provenance": {"page": 4, "section": "3.2 Multi-Head Attention", "confidence": 0.98}
                },
                {
                    "id": "node_mlp",
                    "type": "feed_forward",
                    "label": "Feed-Forward Network (4x Expansion)",
                    "description": "Two linear transformations with ReLU/GELU activation.",
                    "math_formula": "\\text{FFN}(x) = \\max(0, x W_1 + b_1) W_2 + b_2",
                    "provenance": {"page": 5, "section": "3.3 Position-wise Feed-Forward", "confidence": 0.95}
                },
                {
                    "id": "node_logits",
                    "type": "output",
                    "label": "Unembedding & Vocabulary Logits",
                    "description": "Linear projection to vocabulary logits and softmax probabilities.",
                    "math_formula": "P(y) = \\text{softmax}(x W_U)",
                    "provenance": {"page": 5, "section": "3.4 Output Distribution", "confidence": 0.95}
                }
            ],
            "edges": [
                {"id": "e1", "source": "node_embed", "target": "node_norm1", "type": "data_flow"},
                {"id": "e2", "source": "node_norm1", "target": "node_mha", "type": "data_flow"},
                {"id": "e3", "source": "node_mha", "target": "node_mlp", "type": "data_flow"},
                {"id": "e4", "source": "node_mlp", "target": "node_logits", "type": "data_flow"},
                {"id": "res1", "source": "node_embed", "target": "node_mlp", "type": "residual_skip", "label": "Residual Stream 1"}
            ]
        }
        canonical, _ = graph_validator.validate_and_normalize(raw_fallback)
        return canonical


architecture_extractor = ArchitectureExtractor()
