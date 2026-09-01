"use client";

import React, { useState } from "react";
import ArchitectureDiagram, { DiagramNode, DiagramEdge } from "@/components/ArchitectureDiagram";

// ── Mock data simulating what the vision model / backend returns ──
// This is DYNAMIC — any paper would produce different nodes/edges

const MOCK_PAPERS: Record<string, { title: string; nodes: DiagramNode[]; edges: DiagramEdge[] }> = {
  transformer: {
    title: "Attention Is All You Need (Transformer)",
    nodes: [
      { id: "input", label: "Input Tokens", type: "input", description: "Raw token IDs from the vocabulary", math_formula: "x ∈ ℝ^{V}" },
      { id: "embed", label: "Token Embedding", type: "embedding", description: "Maps discrete tokens to continuous vectors", math_formula: "E = X · W_e", parameters: "19.2M" },
      { id: "pos_enc", label: "Positional Encoding", type: "embedding", description: "Sinusoidal position information", math_formula: "PE(pos, 2i) = sin(pos/10000^{2i/d})" },
      { id: "norm1", label: "Layer Norm", type: "normalization", description: "Normalize across hidden dimensions", math_formula: "LN(x) = (x - μ)/σ ⊙ γ + β", parameters: "1.02K" },
      { id: "qkv", label: "Q, K, V Projections", type: "linear", description: "Project to query, key, value spaces", math_formula: "Q = XW_Q, K = XW_K, V = XW_V", parameters: "786K" },
      { id: "attn", label: "Multi-Head Attention", type: "attention", description: "Scaled dot-product attention across 8 heads", math_formula: "Attention(Q,K,V) = softmax(QK^T/√d_k)V", flops: "0.52 GFLOPs" },
      { id: "add1", label: "Residual Add", type: "residual", description: "Skip connection around attention", math_formula: "output = x + Sublayer(x)" },
      { id: "norm2", label: "Layer Norm 2", type: "normalization", description: "Second normalization before FFN", math_formula: "LN(x) = (x - μ)/σ ⊙ γ + β" },
      { id: "ffn1", label: "FFN Expand (4x)", type: "feedforward", description: "First linear layer expanding to 2048", math_formula: "W_1 ∈ ℝ^{512×2048}", parameters: "1.05M" },
      { id: "ffn2", label: "FFN Contract", type: "feedforward", description: "Second linear layer contracting to 512", math_formula: "W_2 ∈ ℝ^{2048×512}", parameters: "1.05M" },
      { id: "add2", label: "Residual Add 2", type: "residual", description: "Skip connection around FFN", math_formula: "output = x + FFN(x)" },
      { id: "output", label: "Output Logits", type: "output", description: "Final projection to vocabulary", math_formula: "P(y) = softmax(xW_U)", parameters: "19.2M" },
    ],
    edges: [
      { source: "input", target: "embed" },
      { source: "embed", target: "pos_enc", label: "add" },
      { source: "pos_enc", target: "norm1" },
      { source: "norm1", target: "qkv" },
      { source: "qkv", target: "attn" },
      { source: "attn", target: "add1" },
      { source: "embed", target: "add1", label: "skip", type: "residual_skip" },
      { source: "add1", target: "norm2" },
      { source: "norm2", target: "ffn1" },
      { source: "ffn1", target: "ffn2" },
      { source: "ffn2", target: "add2" },
      { source: "add1", target: "add2", label: "skip", type: "residual_skip" },
      { source: "add2", target: "output" },
    ],
  },
  moe: {
    title: "Mixtral 8x7B (MoE)",
    nodes: [
      { id: "input", label: "Input Hidden State", type: "input", description: "Hidden state from previous layer" },
      { id: "norm", label: "RMS Normalization", type: "normalization", description: "Root Mean Square Layer Normalization", math_formula: "RMSNorm(x) = x / RMS(x) ⊙ γ" },
      { id: "router", label: "Router Gate", type: "router", description: "Decides which experts to activate", math_formula: "G(x) = Softmax(W_g · x)", parameters: "3.6M" },
      { id: "shared_exp", label: "Shared Expert", type: "expert", description: "Always-active shared feed-forward network", math_formula: "y_s = FFN_shared(x)", parameters: "14.3M" },
      { id: "exp1", label: "Expert 1", type: "expert", description: "Routed expert — activated selectively", math_formula: "y_1 = FFN_1(x)", parameters: "14.3M" },
      { id: "exp2", label: "Expert 2", type: "expert", description: "Routed expert — activated selectively", math_formula: "y_2 = FFN_2(x)", parameters: "14.3M" },
      { id: "exp3", label: "Expert N", type: "expert", description: "N-th routed expert", math_formula: "y_n = FFN_n(x)", parameters: "14.3M" },
      { id: "combine", label: "Weighted Combine", type: "block", description: "Combine expert outputs by router weights", math_formula: "y = Σ G_i(x) · E_i(x)" },
      { id: "add", label: "Residual Add", type: "residual", description: "Add MoE output to residual stream", math_formula: "h' = h + MoE(h)" },
      { id: "output", label: "Output", type: "output", description: "Output hidden state" },
    ],
    edges: [
      { source: "input", target: "norm" },
      { source: "norm", target: "router" },
      { source: "norm", target: "shared_exp" },
      { source: "router", target: "exp1", label: "top-k" },
      { source: "router", target: "exp2", label: "top-k" },
      { source: "router", target: "exp3", label: "top-k" },
      { source: "shared_exp", target: "combine" },
      { source: "exp1", target: "combine" },
      { source: "exp2", target: "combine" },
      { source: "exp3", target: "combine" },
      { source: "combine", target: "add" },
      { source: "input", target: "add", label: "skip", type: "residual_skip" },
      { source: "add", target: "output" },
    ],
  },
  ssm: {
    title: "Mamba (SSM)",
    nodes: [
      { id: "input", label: "Input Sequence", type: "input", description: "Token embeddings" },
      { id: "proj_in", label: "Input Projection", type: "linear", description: "Project input to inner dimension", math_formula: "x = W_x · h" },
      { id: "conv", label: "Causal Conv1D", type: "convolution", description: "Local context mixing before SSM", math_formula: "x' = Conv1D(x)" },
      { id: "ssm", label: "Selective SSM", type: "ssm", description: "State space model with input-dependent discretization", math_formula: "S_t = ᾱS_{t-1} + β̄x_t", parameters: "14.6M" },
      { id: "gate", label: "Gating (SiLU)", type: "gate", description: "SiLU activation and element-wise multiply", math_formula: "y = SiLU(x') ⊙ z" },
      { id: "proj_out", label: "Output Projection", type: "linear", description: "Project back to model dimension", math_formula: "h' = W_y · y" },
      { id: "add", label: "Residual Add", type: "residual", description: "Skip connection", math_formula: "output = h + h'" },
      { id: "output", label: "Output", type: "output", description: "Layer output" },
    ],
    edges: [
      { source: "input", target: "proj_in" },
      { source: "proj_in", target: "conv" },
      { source: "conv", target: "ssm" },
      { source: "ssm", target: "gate" },
      { source: "gate", target: "proj_out" },
      { source: "proj_out", target: "add" },
      { source: "input", target: "add", label: "skip", type: "residual_skip" },
      { source: "add", target: "output" },
    ],
  },
};

export default function TestPage() {
  const [selectedPaper, setSelectedPaper] = useState<string>("transformer");
  const paper = MOCK_PAPERS[selectedPaper];

  return (
    <div className="h-screen flex flex-col bg-zinc-50">
      {/* Header */}
      <div className="bg-white border-b border-zinc-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-zinc-900">Architecture Diagram — Dynamic Renderer</h1>
            <p className="text-xs text-zinc-500 mt-0.5">
              White theme · Auto-layout · Click nodes to inspect · Scroll to zoom · Drag to pan
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400 mr-2">Test paper:</span>
            {Object.entries(MOCK_PAPERS).map(([key, p]) => (
              <button
                key={key}
                onClick={() => setSelectedPaper(key)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md border transition-all ${
                  selectedPaper === key
                    ? "bg-zinc-900 text-white border-zinc-900"
                    : "bg-white text-zinc-600 border-zinc-200 hover:bg-zinc-50"
                }`}
              >
                {p.title.split("(")[0].trim()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Diagram */}
      <div className="flex-1 p-6">
        <div className="mb-3 flex items-center gap-4">
          <h2 className="text-sm font-semibold text-zinc-700">{paper.title}</h2>
          <span className="text-[10px] font-mono text-zinc-400 bg-zinc-100 px-2 py-0.5 rounded">
            {paper.nodes.length} nodes · {paper.edges.length} edges
          </span>
        </div>
        <ArchitectureDiagram
          nodes={paper.nodes}
          edges={paper.edges}
          height={620}
        />
      </div>
    </div>
  );
}
