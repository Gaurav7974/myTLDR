"use client";

import React, { useState } from "react";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import type { ArchitectureGraph } from "@/types/architecture_graph";
import { isVerifiedVisionGraph, shouldRenderReconstruction } from "@/types/architecture_graph";

// ── Fixtures ────────────────────────────────────────────────────────
// Each fixture is intentionally generic (no paper-specific hardcoding).
// Together they exercise every branch of the renderer:
//   1. figure-faithful verified (the headline case)
//   2. figure-faithful verified with multiple groups
//   3. text-only auto-layout
//   4. legacy react-flow shape (auto-layout fallback)
//   5. unverified / hallucinated (refused, shown as unavailable)
//   6. empty graph (empty state)

const figureFixture: ArchitectureGraph = {
  version: "2.0",
  source: "vision",
  layout_mode: "figure",
  figure_id: "fig:arch-1",
  figure_page: 3,
  confidence: 0.88,
  verification: {
    valid: true,
    confidence: 0.88,
    hallucinations: [],
    missing_elements: [],
    reason: "matches source figure",
  },
  groups: [
    { id: "encoder", label: "Encoder", bbox: { x: 0.05, y: 0.1, w: 0.4, h: 0.3 }, color: "#93c5fd", confidence: 0.9 },
    { id: "decoder", label: "Decoder", bbox: { x: 0.55, y: 0.1, w: 0.4, h: 0.3 }, color: "#fdba74", confidence: 0.9 },
  ],
  nodes: [
    {
      id: "in", label: "Input", type: "input",
      bbox: { x: 0.1, y: 0.2, w: 0.15, h: 0.1 },
      group_id: "encoder", shape: "rounded",
      description: "Token IDs from the vocabulary",
      math_formula: "x \\in \\mathbb{Z}^{V}",
      paper_term: "input embedding",
      provenance: { source: "vision", figure_id: "fig:arch-1", figure_page: 3, citation: "input tokens" },
      confidence: 0.95,
      style: { fill: "#eff6ff", border: "#93c5fd" },
    },
    {
      id: "attn", label: "Self-Attention", type: "attention",
      bbox: { x: 0.3, y: 0.2, w: 0.2, h: 0.1 },
      group_id: "encoder", shape: "rounded",
      description: "Scaled dot-product attention.",
      math_formula: "\\mathrm{softmax}(QK^\\top/\\sqrt{d_k}) V",
      paper_term: "self-attention",
      provenance: { source: "vision", figure_id: "fig:arch-1", figure_page: 3 },
      confidence: 0.9,
      style: { fill: "#fdf2f8", border: "#f9a8d4" },
    },
    {
      id: "ffn", label: "Feed-Forward", type: "feedforward",
      bbox: { x: 0.3, y: 0.3, w: 0.2, h: 0.08 },
      group_id: "encoder", shape: "rounded",
      description: "Two-layer MLP with ReLU.",
      math_formula: "\\mathrm{FFN}(x) = W_2 \\mathrm{ReLU}(W_1 x)",
      confidence: 0.85,
    },
    {
      id: "out", label: "Output", type: "output",
      bbox: { x: 0.7, y: 0.2, w: 0.15, h: 0.1 },
      group_id: "decoder", shape: "rounded",
      description: "Output projections.",
      confidence: 0.85,
    },
  ],
  edges: [
    { source: "in", target: "attn", type: "data_flow", label: "tokens",
      path_points: [{ x: 0.18, y: 0.25 }, { x: 0.3, y: 0.25 }],
      style: { stroke: "#3b82f6", name: "solid" } },
    { source: "attn", target: "ffn", type: "data_flow",
      path_points: [{ x: 0.4, y: 0.25 }, { x: 0.4, y: 0.34 }],
      style: { stroke: "#3b82f6", name: "solid" } },
    { source: "ffn", target: "out", type: "data_flow", label: "hidden",
      path_points: [{ x: 0.4, y: 0.34 }, { x: 0.4, y: 0.25 }, { x: 0.7, y: 0.25 }],
      style: { stroke: "#3b82f6", name: "solid" } },
    { source: "in", target: "out", type: "residual_skip", label: "skip",
      path_points: [{ x: 0.18, y: 0.27 }, { x: 0.18, y: 0.5 }, { x: 0.78, y: 0.5 }, { x: 0.78, y: 0.27 }],
      style: { stroke: "#ef4444", name: "residual" } },
  ],
};

const textAutoFixture: ArchitectureGraph = {
  version: "2.0",
  source: "text",
  layout_mode: "auto",
  paper_title: "Text-only paper",
  nodes: [
    { id: "a", label: "Layer A", type: "attention",
      description: "Generic attention layer.", math_formula: "softmax(x)" },
    { id: "b", label: "Layer B", type: "feedforward",
      description: "Generic feedforward." },
    { id: "c", label: "Output", type: "output" },
  ],
  edges: [
    { source: "a", target: "b", type: "data_flow" },
    { source: "b", target: "c", type: "data_flow" },
  ],
};

const legacyFixture: ArchitectureGraph = {
  version: "legacy",
  source: "legacy",
  layout_mode: "auto",
  nodes: [
    { id: "1", label: "Legacy 1", type: "input" },
    { id: "2", label: "Legacy 2", type: "output" },
  ],
  edges: [
    { source: "1", target: "2" },
  ],
};

const unverifiedFixture: ArchitectureGraph = {
  version: "2.0",
  source: "vision",
  layout_mode: "figure",
  // Note: this is figure-mode but the verification metadata is invalid;
  // shouldRenderReconstruction() will return false, so the renderer falls
  // back to auto-layout rather than claiming the geometry is trustworthy.
  nodes: [
    { id: "hallucinated", label: "Hallucinated Node", type: "other" },
  ],
  edges: [],
  verification: {
    valid: false,
    confidence: 0.4,
    hallucinations: ["hallucinated"],
    reason: "figure-faithfulness verifier rejected the candidate graph",
  },
};

const emptyFixture: ArchitectureGraph = {
  version: "2.0",
  source: "text",
  layout_mode: "auto",
  paper_title: "Empty",
  nodes: [],
  edges: [],
};

type FixtureKey = "figure" | "figure-multigroup" | "text-auto" | "legacy" | "unverified" | "empty";

const FIXTURES: Record<FixtureKey, { label: string; description: string; graph: ArchitectureGraph | null }> = {
  "figure": {
    label: "Figure-faithful",
    description: "Verified vision graph with bbox + groups + path_points + custom styles.",
    graph: figureFixture,
  },
  "figure-multigroup": {
    label: "Figure-faithful (multi-group)",
    description: "Same as above with two groups (Encoder + Decoder) and a residual skip.",
    graph: figureFixture,
  },
  "text-auto": {
    label: "Text-only (auto-layout)",
    description: "BFS auto-layout for text-extracted graphs that have no figure geometry.",
    graph: textAutoFixture,
  },
  "legacy": {
    label: "Legacy (auto-layout)",
    description: "Downgraded legacy payload from cache; same auto-layout, marked legacy.",
    graph: legacyFixture,
  },
  "unverified": {
    label: "Unverified figure (refused)",
    description: "Figure-mode graph with failing verification; renderer downgrades to auto-layout.",
    graph: unverifiedFixture,
  },
  "empty": {
    label: "Empty graph",
    description: "No architecture was extracted; honest empty state.",
    graph: emptyFixture,
  },
};

export default function TestPage() {
  const [selected, setSelected] = useState<FixtureKey>("figure");
  const fixture = FIXTURES[selected];
  const graph = fixture.graph;
  const isFigure = graph ? shouldRenderReconstruction(graph) : false;
  const isVerified = graph ? isVerifiedVisionGraph(graph) : false;

  return (
    <div className="h-screen flex flex-col bg-zinc-50">
      {/* Header */}
      <div className="bg-white border-b border-zinc-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-zinc-900">Architecture Diagram — Rich Fixture Exerciser</h1>
            <p className="text-xs text-zinc-500 mt-0.5">
              Figure-faithful rendering · Auto-layout fallback · Click nodes to inspect · Scroll to zoom · Drag to pan
            </p>
          </div>
          <div className="flex items-center gap-1.5 flex-wrap justify-end">
            {(Object.keys(FIXTURES) as FixtureKey[]).map((key) => (
              <button
                key={key}
                onClick={() => setSelected(key)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md border transition-all ${
                  selected === key
                    ? "bg-zinc-900 text-white border-zinc-900"
                    : "bg-white text-zinc-600 border-zinc-200 hover:bg-zinc-50"
                }`}
              >
                {FIXTURES[key].label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Diagram + meta */}
      <div className="flex-1 p-6 flex flex-col">
        <div className="mb-3 flex items-center gap-4">
          <h2 className="text-sm font-semibold text-zinc-700">{fixture.label}</h2>
          <span className="text-[10px] font-mono text-zinc-400 bg-zinc-100 px-2 py-0.5 rounded">
            {graph ? `${graph.nodes.length} nodes · ${graph.edges.length} edges` : "0 nodes"}
          </span>
          {graph && (
            <>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                isFigure ? "bg-emerald-100 text-emerald-800"
                         : graph.layout_mode === "figure" && !isFigure ? "bg-amber-100 text-amber-800"
                         : "bg-zinc-100 text-zinc-600"
              }`}>
                {isFigure ? "verified figure" : graph.layout_mode === "figure" ? "figure (unverified)" : graph.layout_mode}
              </span>
              <span className="text-[10px] font-mono text-zinc-400">
                isVerifiedVisionGraph = {String(isVerified)}
              </span>
            </>
          )}
        </div>
        <p className="text-xs text-zinc-500 mb-3">{fixture.description}</p>
        <div className="flex-1 min-h-0">
          {graph ? (
            <ArchitectureDiagram graph={graph} height={620} />
          ) : (
            <div className="h-full flex items-center justify-center text-zinc-400 text-sm">
              No graph supplied
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
