"""3D Architecture Extraction and Spatial Layout Service (Brendan Bycroft style)."""

import json
import re
from typing import Dict, Any, List
from app.config import settings
from app.services.llm import groq_client

BYCROFT_EXTRACTION_SYSTEM_PROMPT = """You are a world-class AI researcher and neural network architecture visualizer.
Your mission is to analyze the architecture and method sections of an AI research paper and extract its COMPLETE, sequential forward-pass computational pipeline for interactive 3D visualization (in the style of Brendan Bycroft's bbycroft.net/llm).

Deconstruct the architecture into 6 to 12 precise sequential stages from raw input to final output/loss.

For each stage, you must extract:
- id: Short unique kebab-case ID (e.g. "bycroft-embed", "bycroft-ssm-discretize", "bycroft-cross-attn")
- name: Clear human-readable name of the operation
- stepNumber: 1-indexed sequential step integer
- category: One of ["embedding", "projections", "attention_matrix", "multihead_norm", "mlp_ffn", "residual", "convolution", "ssm_state", "diffusion_denoise", "logits", "output"]
- description: Concise summary of what happens to the data at this step
- detailedWalkthrough: Granular, mechanically precise explanation of the exact matrix / vector calculations
- concreteExample: Numerical demonstration of the matrix multiplication or element-wise operation with concrete sample numbers
- mathFormula: Valid LaTeX formula representing this exact operation (without dollar signs)
- tensorInput: String representing input tensor dimensions with batch, sequence length, and channels (e.g. "[Batch: 1, Tokens: 6, Dim: 512]")
- tensorOutput: String representing output tensor dimensions (e.g. "[Batch: 1, Tokens: 6, Dim: 2048]")
- parameters: Estimated parameter count formula and total (e.g. "512 × 2048 = 1.05M params")
- flops: Estimated FLOPs formula (e.g. "6.29 MFLOPs")
- intuition: High-level intuitive explanation of why this step is necessary
- subLabels: Array of 2-3 short tags representing sub-operations inside this block (e.g. ["W_1 Proj", "GELU Activation"])

Also extract:
- paradigm: One of ["transformer", "diffusion", "ssm", "cnn_vision", "rl_policy", "adaptation_lora", "gnn", "other"]
- residualHighways: Array of skip connections leaping from one stepNumber to a later stepNumber:
  [{"startStep": 1, "endStep": 6, "label": "Residual Stream 1"}]
- heatmapTokens: 4-6 sample tokens or feature channels relevant to the paper
- heatmapMatrix: N x N normalized matrix (values 0.0 to 1.0) showing pairwise attention/affinity
- nextTokenPredictions: Array of 4-5 candidate predictions or classifications with probabilities and logits

You must respond ONLY with a valid, clean JSON object matching this schema. Return purely raw JSON."""


class BycroftService:
    """Extracts and computes 3D Bycroft architectures for arbitrary research papers."""

    def extract_3d_architecture(self, paper_text: str, model_title: str = "") -> Dict[str, Any]:
        """Extract sequential 3D stages and compute spatial 3D layout coordinates."""
        trimmed_text = paper_text[:settings.MAX_SECTION_CHARS]
        user_prompt = f"""Paper Title: {model_title}
---
Paper Architecture & Method Excerpt:
{trimmed_text}
---

Extract the full sequential 3D Bycroft architecture pipeline JSON adhering strictly to the schema."""

        try:
            raw_data = groq_client.extract_json(
                messages=[
                    {"role": "system", "content": BYCROFT_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=settings.MISTRAL_LLM_MODEL
            )
            
            # Compute 3D Spatial Layout
            return self._compute_spatial_layout(raw_data)
        except Exception as e:
            print(f"[ERROR] Bycroft 3D architecture extraction failed: {e}")
            return self._generate_fallback_3d_pipeline(model_title)

    def _compute_spatial_layout(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute deterministic 3D positions, slab sizes, and category colors."""
        stages = data.get("stages", [])
        num_stages = len(stages)
        if num_stages == 0:
            return data

        # Calculate horizontal spacing
        total_width = 18.0
        start_x = -8.0
        step_gap = total_width / max(1, num_stages - 1)

        category_colors = {
            "embedding": "#3B82F6",
            "projections": "#8B5CF6",
            "attention_matrix": "#F59E0B",
            "multihead_norm": "#64748B",
            "mlp_ffn": "#6366F1",
            "residual": "#EF4444",
            "convolution": "#0EA5E9",
            "ssm_state": "#14B8A6",
            "diffusion_denoise": "#EC4899",
            "logits": "#10B981",
            "output": "#059669"
        }

        for idx, stage in enumerate(stages):
            # Compute X position
            x = start_x + (idx * step_gap)
            
            # Elevation Y position (slight arch for attention/matrix blocks)
            cat = stage.get("category", "mlp_ffn")
            y = 0.3 if cat in ["attention_matrix", "diffusion_denoise"] else 0.0
            z = 0.0

            stage["position3D"] = [round(x, 2), y, z]

            # Compute Slab Size based on category / dimension
            if cat in ["mlp_ffn", "diffusion_denoise"]:
                stage["size3D"] = [2.0, 2.6, 0.55]
                stage["gridDimensions"] = {"rows": 6, "cols": 20}
            elif cat == "attention_matrix":
                stage["size3D"] = [2.0, 2.0, 0.3]
                stage["gridDimensions"] = {"rows": 6, "cols": 6}
            elif cat in ["embedding", "projections", "ssm_state"]:
                stage["size3D"] = [1.8, 2.2, 0.45]
                stage["gridDimensions"] = {"rows": 6, "cols": 12}
            elif cat in ["multihead_norm", "residual"]:
                stage["size3D"] = [1.4, 1.9, 0.35]
                stage["gridDimensions"] = {"rows": 6, "cols": 8}
            else:
                stage["size3D"] = [2.0, 2.4, 0.5]
                stage["gridDimensions"] = {"rows": 8, "cols": 16}

            stage["color"] = category_colors.get(cat, "#3B82F6")

        data["stages"] = stages
        return data

    def _generate_fallback_3d_pipeline(self, model_title: str) -> Dict[str, Any]:
        """Safe fallback 3D architecture if LLM extraction is unavailable."""
        return {
            "paradigm": "transformer",
            "stages": [
                {
                    "id": "stage-1-input",
                    "name": "Input Representation & Embedding",
                    "stepNumber": 1,
                    "category": "embedding",
                    "description": "Transforms discrete raw inputs into dense continuous feature coordinates.",
                    "detailedWalkthrough": "Looks up token coordinates from the embedding matrix W_E and superimposes positional state encodings.",
                    "concreteExample": "Input token -> W_E lookup -> [Batch: 1, Length: 6, Dim: 512].",
                    "mathFormula": "x_0 = \\text{Embedding}(x) + \\text{Position}(pos)",
                    "tensorInput": "[Batch: 1, Length: 6]",
                    "tensorOutput": "[Batch: 1, Length: 6, Dim: 512]",
                    "parameters": "Vocab × 512 = 18.9M params",
                    "flops": "12.3 KFLOPs",
                    "intuition": "Converts discrete symbols into dense geometric coordinate space.",
                    "position3D": [-7.5, 0.0, 0.0],
                    "size3D": [1.8, 2.2, 0.45],
                    "gridDimensions": {"rows": 6, "cols": 16},
                    "subLabels": ["Embedding Table", "Pos Encodings"],
                    "color": "#3B82F6"
                },
                {
                    "id": "stage-2-norm",
                    "name": "Layer Normalization",
                    "stepNumber": 2,
                    "category": "multihead_norm",
                    "description": "Standardizes feature distributions across channels to prevent exploding gradients.",
                    "detailedWalkthrough": "Subtracts channel mean μ and divides by standard deviation σ before affine scaling.",
                    "concreteExample": "Mean = 0.18, Var = 1.05. Normalized vector has zero mean and unit variance.",
                    "mathFormula": "x_{\\text{norm}} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta",
                    "tensorInput": "[Batch: 1, Length: 6, Dim: 512]",
                    "tensorOutput": "[Batch: 1, Length: 6, Dim: 512]",
                    "parameters": "1,024 params",
                    "flops": "3.1 KFLOPs",
                    "intuition": "Maintains well-behaved activation boundaries across deep layers.",
                    "position3D": [-4.5, 0.0, 0.0],
                    "size3D": [1.4, 1.9, 0.35],
                    "gridDimensions": {"rows": 6, "cols": 8},
                    "subLabels": ["Mean/Var", "Affine γ, β"],
                    "color": "#64748B"
                },
                {
                    "id": "stage-3-core-op",
                    "name": "Core Neural Transformation",
                    "stepNumber": 3,
                    "category": "attention_matrix",
                    "description": "Computes pairwise context interactions and cross-feature transformations.",
                    "detailedWalkthrough": "Applies non-linear routing and pairwise representation weighting across all positions.",
                    "concreteExample": "Computes similarity score matrix between all token pairs.",
                    "mathFormula": "A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V",
                    "tensorInput": "[Batch: 1, Heads: 8, Length: 6, d_k: 64]",
                    "tensorOutput": "[Batch: 1, Length: 6, Dim: 512]",
                    "parameters": "786,432 params",
                    "flops": "4.72 MFLOPs",
                    "intuition": "Allows each token to attend dynamically to the most relevant context words.",
                    "position3D": [-1.5, 0.3, 0.0],
                    "size3D": [2.0, 2.0, 0.3],
                    "gridDimensions": {"rows": 6, "cols": 6},
                    "subLabels": ["Similarity Q·K", "Softmax Weights"],
                    "color": "#F59E0B"
                },
                {
                    "id": "stage-4-skip",
                    "name": "Residual Skip Highway",
                    "stepNumber": 4,
                    "category": "residual",
                    "description": "Direct identity skip connection bypassing the core transformation.",
                    "detailedWalkthrough": "Adds the unmodified input representation directly to the sub-layer output.",
                    "concreteExample": "x_res = x_input + SubLayer(x_input).",
                    "mathFormula": "x_{\\text{res}} = x + \\text{SubLayer}(x)",
                    "tensorInput": "[Batch: 1, Length: 6, Dim: 512]",
                    "tensorOutput": "[Batch: 1, Length: 6, Dim: 512]",
                    "parameters": "0 params",
                    "flops": "3.07 KFLOPs",
                    "intuition": "Eliminates vanishing gradients and preserves low-level features.",
                    "position3D": [1.8, 0.0, 0.0],
                    "size3D": [1.4, 1.8, 0.3],
                    "gridDimensions": {"rows": 6, "cols": 8},
                    "subLabels": ["Identity Highway", "Element-wise Add"],
                    "color": "#EF4444"
                },
                {
                    "id": "stage-5-mlp",
                    "name": "Feed-Forward Expansion & Contraction",
                    "stepNumber": 5,
                    "category": "mlp_ffn",
                    "description": "High-capacity non-linear expansion (4x) for associative factual storage.",
                    "detailedWalkthrough": "Expands 512 -> 2048 dimensions with activation function, then contracts back to 512.",
                    "concreteExample": "Vector expands into 2048 factual neurons before contracting back.",
                    "mathFormula": "\\text{FFN}(x) = \\max(0, x W_1 + b_1) W_2 + b_2",
                    "tensorInput": "[Batch: 1, Length: 6, Dim: 512]",
                    "tensorOutput": "[Batch: 1, Length: 6, Dim: 512]",
                    "parameters": "2.09M params",
                    "flops": "12.58 MFLOPs",
                    "intuition": "Stores and recalls learned factual knowledge associations.",
                    "position3D": [5.5, 0.0, 0.0],
                    "size3D": [2.2, 2.6, 0.55],
                    "gridDimensions": {"rows": 6, "cols": 20},
                    "subLabels": ["W_1 (512->2048)", "W_2 (2048->512)"],
                    "color": "#6366F1"
                },
                {
                    "id": "stage-6-output",
                    "name": "Output Logits & Probability Distribution",
                    "stepNumber": 6,
                    "category": "logits",
                    "description": "Projects final latent representation to output space / vocabulary logits.",
                    "detailedWalkthrough": "Multiplies final vector against unembedding weights to generate prediction probabilities.",
                    "concreteExample": "Final vector -> W_U -> 37,000 logits -> Softmax probabilities.",
                    "mathFormula": "P(y) = \\text{softmax}(x_{\\text{final}} W_U + b)",
                    "tensorInput": "[Batch: 1, Length: 1, Dim: 512]",
                    "tensorOutput": "[Batch: 1, Vocab: 37,000]",
                    "parameters": "18.9M params",
                    "flops": "37.88 MFLOPs",
                    "intuition": "Converts mathematical vectors back into concrete predictions.",
                    "position3D": [9.5, 0.0, 0.0],
                    "size3D": [2.0, 2.4, 0.5],
                    "gridDimensions": {"rows": 8, "cols": 16},
                    "subLabels": ["Unembedding W_U", "Softmax Sampling"],
                    "color": "#10B981"
                }
            ],
            "residualHighways": [
                {"startStep": 1, "endStep": 4, "label": "Residual Stream 1"}
            ],
            "heatmapTokens": ["Input", "LayerNorm", "Attention", "FFN", "Logits"],
            "heatmapMatrix": [
                [0.85, 0.08, 0.04, 0.02, 0.01],
                [0.12, 0.65, 0.15, 0.05, 0.03],
                [0.05, 0.25, 0.55, 0.10, 0.05],
                [0.02, 0.10, 0.18, 0.60, 0.10],
                [0.01, 0.05, 0.12, 0.22, 0.60]
            ],
            "nextTokenPredictions": [
                {"token": "predicted_token_1", "probability": 0.45, "logit": 14.5},
                {"token": "predicted_token_2", "probability": 0.28, "logit": 13.8},
                {"token": "predicted_token_3", "probability": 0.15, "logit": 12.4},
                {"token": "predicted_token_4", "probability": 0.08, "logit": 11.2},
                {"token": "predicted_token_5", "probability": 0.04, "logit": 10.1}
            ]
        }


bycroft_service = BycroftService()
