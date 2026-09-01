"use client";

import React, { useState, useMemo, useRef, useCallback, useEffect } from "react";
import {
  ArchitectureGraph,
  GraphBoundingBox,
  GraphGroup,
  GraphPoint,
  shouldRenderReconstruction,
} from "@/types/architecture_graph";
import {
  FIGURE_CANVAS_W,
  FIGURE_CANVAS_H,
  NODE_W_DEFAULT as NODE_W,
  NODE_H_DEFAULT as NODE_H,
  PADDING,
  figureNodePosition,
  computeAutoLayout,
  resolveEdgeStyle as resolveEdgeStyleImpl,
  type PositionedNode,
} from "./architectureDiagramLayout";

// ── Public types ────────────────────────────────────────────────────
// The simple `nodes`/`edges` props remain the primary API for callers
// that only have a flat graph. The rich `graph` prop activates the
// figure-faithful rendering path when present.

export interface DiagramNode {
  id: string;
  label: string;
  type: string;
  description?: string;
  how_it_works?: string;
  math_formula?: string;
  paper_term?: string;
  parameters?: string;
  flops?: string;
  group_id?: string;
  bbox?: GraphBoundingBox;
  shape?: "rect" | "rounded" | "ellipse" | "diamond" | "cylinder" | "parallelogram" | "trapezoid" | "custom";
  style?: {
    fill?: string;
    border?: string;
    text?: string;
    shape?: "rect" | "rounded" | "ellipse" | "diamond" | "cylinder" | "parallelogram" | "trapezoid" | "custom";
  };
  confidence?: number;
  provenance?: {
    source?: "text" | "vision" | "merged";
    paper_title?: string;
    page?: number;
    section?: string;
    figure_id?: string;
    figure_page?: number;
    citation?: string;
    confidence?: number;
  };
}

export interface DiagramEdge {
  source: string;
  target: string;
  label?: string;
  type?: "data_flow" | "residual_skip" | "cross_attention" | "gated_modulation" | "recurrent_state" | "custom";
  semantic?: string;
  path_points?: GraphPoint[];
  style?: {
    stroke?: string;
    width?: number;
    dasharray?: string;
    name?: "solid" | "dashed" | "dotted" | "residual" | "beam" | "highlight" | "custom" | "default";
  };
  raw_style?: Record<string, unknown>;
}

export interface ArchitectureDiagramProps {
  nodes?: DiagramNode[];
  edges?: DiagramEdge[];
  /** Optional rich graph. When `layout_mode === 'figure'` and the graph
   *  is verified, nodes/groups render at extracted normalized coordinates.
   *  When `layout_mode === 'auto'`, the BFS layout is used as a fallback. */
  graph?: ArchitectureGraph;
  height?: number;
  /** Optional callback when the user clicks a node — useful for tests. */
  onNodeClick?: (node: DiagramNode) => void;
}

// ── Constants (NODE_W, NODE_H, PADDING, FIGURE_CANVAS_W/H are imported
//    from ./architectureDiagramLayout; LAYER_GAP/NODE_GAP are auto-layout
//    constants and live with computeAutoLayout there.)

const TYPE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  input: { bg: "#f8fafc", border: "#94a3b8", text: "#334155" },
  output: { bg: "#f8fafc", border: "#94a3b8", text: "#334155" },
  embedding: { bg: "#eff6ff", border: "#93c5fd", text: "#1e40af" },
  attention: { bg: "#fdf2f8", border: "#f9a8d4", text: "#9d174d" },
  feedforward: { bg: "#f0fdf4", border: "#86efac", text: "#166534" },
  normalization: { bg: "#fafaf9", border: "#d6d3d1", text: "#44403c" },
  block: { bg: "#faf5ff", border: "#c4b5fd", text: "#5b21b6" },
  router: { bg: "#fff7ed", border: "#fdba74", text: "#9a3412" },
  expert: { bg: "#ecfdf5", border: "#6ee7b7", text: "#065f46" },
  ssm: { bg: "#f0f9ff", border: "#7dd3fc", text: "#075985" },
  gate: { bg: "#fefce8", border: "#fde047", text: "#854d0e" },
  cache: { bg: "#f5f3ff", border: "#a78bfa", text: "#4c1d95" },
  linear: { bg: "#f1f5f9", border: "#cbd5e1", text: "#334155" },
  convolution: { bg: "#ecfeff", border: "#67e8f9", text: "#155e75" },
  residual: { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b" },
  default: { bg: "#ffffff", border: "#d1d5db", text: "#374151" },
};

function getNodeColors(type: string) {
  return TYPE_COLORS[type?.toLowerCase()] || TYPE_COLORS.default;
}

// ── Edge style + layout helpers (imported) ─────────────────────────
// resolveEdgeStyle, computeAutoLayout, figureNodePosition,
// positionGroupsFigure, pathPointsToSvgPath, FIGURE_CANVAS_W, FIGURE_CANVAS_H,
// NODE_W, NODE_H, PADDING, and PositionedNode all come from
// ./architectureDiagramLayout so they can be unit-tested in isolation
// without React or a DOM.

// ── Math rendering (safe text + KaTeX for short LaTeX) ──────────────
function MathText({ value, maxLen = 240 }: { value: string; maxLen?: number }) {
  const safe = value.length > maxLen ? value.slice(0, maxLen) + "…" : value;
  // Render Unicode math directly; for LaTeX-style, fall back to monospace text
  // (KaTeX requires DOM measurement we don't have here; safe to keep as text
  // to avoid overflow). The sidebar always shows the full text.
  return <span style={{ fontFamily: "ui-monospace, monospace" }}>{safe}</span>;
}

// ── Component ──────────────────────────────────────────────────────
export default function ArchitectureDiagram({
  nodes: nodesProp,
  edges: edgesProp,
  graph,
  height = 600,
  onNodeClick,
}: ArchitectureDiagramProps) {
  // Unify the two APIs: a `graph` prop or flat `nodes`/`edges` props.
  const unified: { nodes: DiagramNode[]; edges: DiagramEdge[]; layoutMode: string; groups: GraphGroup[]; source: string } = useMemo(() => {
    if (graph && Array.isArray(graph.nodes) && graph.nodes.length > 0) {
      return {
        nodes: graph.nodes as unknown as DiagramNode[],
        edges: (graph.edges || []) as unknown as DiagramEdge[],
        layoutMode: graph.layout_mode || "auto",
        groups: graph.groups || [],
        source: graph.source || "auto",
      };
    }
    return {
      nodes: nodesProp || [],
      edges: edgesProp || [],
      layoutMode: "auto",
      groups: [],
      source: "auto",
    };
  }, [graph, nodesProp, edgesProp]);

  const { nodes, edges, layoutMode, groups, source } = unified;
  const isFigureMode = layoutMode === "figure" && shouldRenderReconstruction(graph as ArchitectureGraph | undefined);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });

  // Compute layout: figure-coordinates when verified, BFS fallback otherwise
  const positioned = useMemo(() => {
    if (isFigureMode) {
      const m = new Map<string, PositionedNode>();
      const layer = 0;
      nodes.forEach((n) => {
        const p = figureNodePosition(n);
        if (p) {
          m.set(n.id, {
            id: n.id, x: p.x, y: p.y, layer, index: 0, source: "figure",
          });
        }
      });
      if (m.size > 0) return m;
      // fall through to auto if figure positions were all missing
    }
    return computeAutoLayout(nodes, edges);
  }, [isFigureMode, nodes, edges]);

  // Mark nodes that have no layout as "missing" so the renderer can draw
  // them safely (or hide them).
  const nodesWithMissingLayout = useMemo(() => {
    return nodes.filter((n) => !positioned.has(n.id));
  }, [nodes, positioned]);

  // Canvas dimensions
  const { canvasW, canvasH } = useMemo(() => {
    if (positioned.size === 0) return { canvasW: 800, canvasH: 400 };
    let maxX = 0;
    let maxY = 0;
    positioned.forEach((p) => {
      maxX = Math.max(maxX, p.x + NODE_W);
      maxY = Math.max(maxY, p.y + NODE_H);
    });
    if (isFigureMode) {
      maxX = Math.max(maxX, FIGURE_CANVAS_W);
      maxY = Math.max(maxY, FIGURE_CANVAS_H);
    }
    return { canvasW: maxX + PADDING, canvasH: maxY + PADDING };
  }, [positioned, isFigureMode]);

  // Group rectangles
  const positionedGroups = useMemo(() => {
    if (!isFigureMode || groups.length === 0) return [];
    return groups.map((g) => {
      const b = g.bbox;
      if (!b) return null;
      return {
        id: g.id,
        label: g.label || g.id,
        x: b.x * FIGURE_CANVAS_W,
        y: b.y * FIGURE_CANVAS_H,
        w: b.w * FIGURE_CANVAS_W,
        h: b.h * FIGURE_CANVAS_H,
        color: g.color || "#e5e7eb",
      };
    }).filter(Boolean) as Array<{ id: string; label: string; x: number; y: number; w: number; h: number; color: string }>;
  }, [isFigureMode, groups]);

  // Edge paths
  const edgePaths = useMemo(() => {
    return edges
      .map((e) => {
        const src = positioned.get(e.source);
        const tgt = positioned.get(e.target);
        if (!src || !tgt) return null;
        // In figure mode with path_points, use them directly
        if (isFigureMode && e.path_points && e.path_points.length >= 2) {
          const points = e.path_points.map((p) => `${p.x * FIGURE_CANVAS_W},${p.y * FIGURE_CANVAS_H}`);
          const path = `M ${points.join(" L ")}`;
          const midX = ((e.path_points[0].x + e.path_points[e.path_points.length - 1].x) / 2) * FIGURE_CANVAS_W;
          const midY = ((e.path_points[0].y + e.path_points[e.path_points.length - 1].y) / 2) * FIGURE_CANVAS_H;
          return { ...e, path, midX, midY };
        }
        // Bezier fallback
        const sx = src.x + NODE_W;
        const sy = src.y + NODE_H / 2;
        const tx = tgt.x;
        const ty = tgt.y + NODE_H / 2;
        const dx = tx - sx;
        const controlOffset = Math.max(40, dx * 0.4);
        const path = `M ${sx} ${sy} C ${sx + controlOffset} ${sy}, ${tx - controlOffset} ${ty}, ${tx} ${ty}`;
        return { ...e, path, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
      })
      .filter(Boolean) as (DiagramEdge & { path: string; midX: number; midY: number })[];
  }, [edges, positioned, isFigureMode]);

  const selectedNode = selectedId ? nodes.find((n) => n.id === selectedId) : null;
  const selectedMeta = selectedId ? positioned.get(selectedId) : null;

  // Pan handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if ((e.target as HTMLElement).closest("[data-node]")) return;
      setIsPanning(true);
      panStart.current = { x: e.clientX - transform.x, y: e.clientY - transform.y };
    },
    [transform]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isPanning) return;
      setTransform((t) => ({
        ...t,
        x: e.clientX - panStart.current.x,
        y: e.clientY - panStart.current.y,
      }));
    },
    [isPanning]
  );

  const handleMouseUp = useCallback(() => setIsPanning(false), []);

  // Zoom via scroll
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.08 : 0.08;
      setTransform((t) => ({
        ...t,
        scale: Math.min(2, Math.max(0.3, t.scale + delta)),
      }));
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, []);

  const resetView = () => setTransform({ x: 0, y: 0, scale: 1 });

  const isAvailable = nodes.length > 0;

  return (
    <div className="flex h-full w-full bg-white rounded-xl border border-zinc-200 overflow-hidden">
      {/* ── Canvas ─────────────────────────────────────────────── */}
      <div
        ref={containerRef}
        className="flex-1 relative overflow-hidden cursor-grab active:cursor-grabbing"
        style={{ minHeight: height }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Grid pattern background */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-[0.03]">
          <defs>
            <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
              <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#000" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>

        {/* Zoom controls */}
        <div className="absolute top-3 right-3 z-20 flex flex-col gap-1">
          <button
            onClick={() => setTransform((t) => ({ ...t, scale: Math.min(2, t.scale + 0.15) }))}
            className="w-8 h-8 rounded-md bg-white border border-zinc-200 text-zinc-600 flex items-center justify-center text-sm font-medium hover:bg-zinc-50 shadow-sm"
          >
            +
          </button>
          <button
            onClick={() => setTransform((t) => ({ ...t, scale: Math.max(0.3, t.scale - 0.15) }))}
            className="w-8 h-8 rounded-md bg-white border border-zinc-200 text-zinc-600 flex items-center justify-center text-sm font-medium hover:bg-zinc-50 shadow-sm"
          >
            −
          </button>
          <button
            onClick={resetView}
            className="w-8 h-8 rounded-md bg-white border border-zinc-200 text-zinc-500 flex items-center justify-center text-[10px] hover:bg-zinc-50 shadow-sm"
            title="Reset view"
          >
            ⟲
          </button>
        </div>

        {/* Mode badge */}
        {isFigureMode && (
          <div className="absolute top-3 left-3 z-20 px-2 py-1 rounded bg-emerald-50 border border-emerald-200 text-emerald-800 text-[10px] font-semibold uppercase tracking-wider">
            Figure-faithful
          </div>
        )}
        {layoutMode === "auto" && !isFigureMode && source === "legacy" && isAvailable && (
          <div className="absolute top-3 left-3 z-20 px-2 py-1 rounded bg-amber-50 border border-amber-200 text-amber-800 text-[10px] font-semibold uppercase tracking-wider">
            Auto-layout (legacy graph)
          </div>
        )}

        {/* SVG Canvas */}
        <svg
          width="100%"
          height="100%"
          className="absolute inset-0"
          style={{ minHeight: height }}
        >
          <g
            transform={`translate(${transform.x}, ${transform.y}) scale(${transform.scale})`}
          >
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
              </marker>
              <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82f6" />
              </marker>
            </defs>

            {/* Group rectangles (figure-mode only) */}
            {positionedGroups.map((g) => (
              <g key={`group-${g.id}`}>
                <rect
                  x={g.x}
                  y={g.y}
                  width={g.w}
                  height={g.h}
                  fill={g.color}
                  fillOpacity={0.06}
                  stroke={g.color}
                  strokeOpacity={0.5}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  rx={6}
                />
                <text
                  x={g.x + 8}
                  y={g.y + 16}
                  className="text-[10px] font-semibold pointer-events-none select-none"
                  fill="#52525b"
                >
                  {g.label}
                </text>
              </g>
            ))}

            {/* Edges */}
            {edgePaths.map((e) => {
              const isActive =
                selectedId && (e.source === selectedId || e.target === selectedId);
              const { stroke, dasharray, width, marker } = resolveEdgeStyleImpl(e, !!isActive);
              return (
                <g key={`${e.source}-${e.target}`}>
                  <path
                    d={e.path}
                    fill="none"
                    stroke={stroke}
                    strokeWidth={width}
                    strokeDasharray={dasharray}
                    markerEnd={marker}
                    className="transition-all duration-200"
                  />
                  {e.label && (
                    <text
                      x={e.midX}
                      y={e.midY - 6}
                      textAnchor="middle"
                      className="text-[10px] fill-zinc-400 pointer-events-none select-none"
                    >
                      {e.label}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Nodes */}
            {nodes.map((node) => {
              const pos = positioned.get(node.id);
              if (!pos) return null;  // skip nodes without layout (figure mode missing bbox)
              // Style resolution: explicit fill/border > taxonomy > default
              const colors = getNodeColors(node.type);
              const fill = node.style?.fill || colors.bg;
              const border = node.style?.border || colors.border;
              const textColor = node.style?.text || colors.text;
              const isSelected = selectedId === node.id;
              const isHovered = hoveredId === node.id;
              const isConnected =
                selectedId &&
                edges.some(
                  (e) =>
                    (e.source === selectedId && e.target === node.id) ||
                    (e.target === selectedId && e.source === node.id)
                );

              return (
                <g
                  key={node.id}
                  data-node
                  data-node-id={node.id}
                  data-node-type={node.type}
                  data-source={pos.source}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className="cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedId(isSelected ? null : node.id);
                    onNodeClick?.(node);
                  }}
                  onMouseEnter={() => setHoveredId(node.id)}
                  onMouseLeave={() => setHoveredId(null)}
                >
                  {(isHovered || isSelected) && (
                    <rect
                      x={-2}
                      y={-2}
                      width={NODE_W + 4}
                      height={NODE_H + 4}
                      rx={10}
                      fill="none"
                      stroke={isSelected ? "#3b82f6" : "#94a3b8"}
                      strokeWidth={isSelected ? 2 : 1}
                      opacity={0.5}
                    />
                  )}
                  <rect
                    width={NODE_W}
                    height={NODE_H}
                    rx={8}
                    fill={fill}
                    stroke={isSelected ? "#3b82f6" : border}
                    strokeWidth={isSelected ? 2 : 1}
                    opacity={selectedId && !isSelected && !isConnected ? 0.35 : 1}
                    className="transition-all duration-150"
                  />
                  <rect
                    width={3}
                    height={NODE_H - 12}
                    x={6}
                    y={6}
                    rx={1.5}
                    fill={border}
                    opacity={0.6}
                  />
                  <text
                    x={NODE_W / 2 + 4}
                    y={NODE_H / 2 - 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="text-[12px] font-medium pointer-events-none select-none"
                    fill={textColor}
                  >
                    {node.label.length > 20
                      ? node.label.slice(0, 20) + "…"
                      : node.label}
                  </text>
                  <text
                    x={NODE_W / 2 + 4}
                    y={NODE_H / 2 + 14}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="text-[9px] pointer-events-none select-none uppercase tracking-wider"
                    fill={textColor}
                    opacity={0.5}
                  >
                    {node.type}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Unavailable state — only when no graph at all */}
        {!isAvailable && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-zinc-400">
              <svg className="w-12 h-12 mx-auto mb-3 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
              <p className="text-sm font-medium">Architecture diagram not available</p>
              <p className="text-xs mt-1 text-zinc-300 max-w-xs">
                The paper did not produce a figure-faithful architecture graph. Other tabs (3D scene, parsed text) still work.
              </p>
            </div>
          </div>
        )}

        {/* Figure-mode nodes with missing bbox warning */}
        {isFigureMode && nodesWithMissingLayout.length > 0 && (
          <div className="absolute bottom-3 left-3 z-20 px-2 py-1 rounded bg-amber-50 border border-amber-200 text-amber-800 text-[10px]">
            {nodesWithMissingLayout.length} node(s) without extracted bbox
          </div>
        )}
      </div>

      {/* ── Inspector Sidebar ──────────────────────────────────── */}
      <div
        className="w-[320px] border-l border-zinc-200 bg-white flex flex-col shrink-0 transition-all duration-300"
        style={{ minHeight: height }}
      >
        {selectedNode ? (
          <>
            {/* Header */}
            <div className="p-4 border-b border-zinc-100">
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded"
                  style={{
                    backgroundColor: getNodeColors(selectedNode.type).bg,
                    color: getNodeColors(selectedNode.type).text,
                    border: `1px solid ${getNodeColors(selectedNode.type).border}`,
                  }}
                >
                  {selectedNode.type}
                </span>
                {selectedNode.paper_term && (
                  <span className="text-[9px] italic text-zinc-500 truncate">
                    {selectedNode.paper_term}
                  </span>
                )}
              </div>
              <h3 className="text-base font-bold text-zinc-900 mt-2">
                {selectedNode.label}
              </h3>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* Description (What it does) */}
              {selectedNode.description && (
                <div>
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                    What it does
                  </h4>
                  <p className="text-sm text-zinc-700 leading-relaxed">
                    {selectedNode.description}
                  </p>
                </div>
              )}

              {/* How it works (paper-specific usage) */}
              {selectedNode.how_it_works && (
                <div>
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                    How it is used in this paper
                  </h4>
                  <p className="text-sm text-zinc-700 leading-relaxed">
                    {selectedNode.how_it_works}
                  </p>
                </div>
              )}

              {/* Math formula */}
              {selectedNode.math_formula && (
                <div>
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                    Mathematical model
                  </h4>
                  <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-100 text-sm text-zinc-800 overflow-x-auto">
                    <MathText value={selectedNode.math_formula} />
                  </div>
                </div>
              )}

              {/* Parameters & FLOPs */}
              {(selectedNode.parameters || selectedNode.flops) && (
                <div className="grid grid-cols-2 gap-2">
                  {selectedNode.parameters && (
                    <div className="p-2.5 bg-zinc-50 rounded-lg border border-zinc-100">
                      <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-400 block">
                        Parameters
                      </span>
                      <span className="text-xs font-mono font-bold text-zinc-800 mt-0.5 block">
                        {selectedNode.parameters}
                      </span>
                    </div>
                  )}
                  {selectedNode.flops && (
                    <div className="p-2.5 bg-zinc-50 rounded-lg border border-zinc-100">
                      <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-400 block">
                        FLOPs
                      </span>
                      <span className="text-xs font-mono font-bold text-zinc-800 mt-0.5 block">
                        {selectedNode.flops}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Connections */}
              <div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-2">
                  Connections
                </h4>
                <div className="space-y-1.5">
                  {edges
                    .filter((e) => e.source === selectedNode.id || e.target === selectedNode.id)
                    .map((e, i) => {
                      const isSource = e.source === selectedNode.id;
                      const otherId = isSource ? e.target : e.source;
                      const other = nodes.find((n) => n.id === otherId);
                      return (
                        <button
                          key={i}
                          onClick={() => setSelectedId(otherId)}
                          className="w-full text-left p-2 rounded-md bg-zinc-50 hover:bg-zinc-100 border border-zinc-100 flex items-center gap-2 transition-colors"
                        >
                          <span className="text-[10px] font-mono text-zinc-400">
                            {isSource ? "→ OUT" : "← IN"}
                          </span>
                          <span className="text-xs font-medium text-zinc-700 truncate">
                            {other?.label || otherId}
                          </span>
                          {e.label && (
                            <span className="text-[9px] text-zinc-400 ml-auto shrink-0">
                              {e.label}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  {edges.filter(
                    (e) => e.source === selectedNode.id || e.target === selectedNode.id
                  ).length === 0 && (
                    <p className="text-xs text-zinc-400 italic">No connections</p>
                  )}
                </div>
              </div>

              {/* Provenance (figure / page / section / citation / confidence) */}
              {selectedNode.provenance && (
                <div className="pt-2 border-t border-zinc-100">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                    Provenance
                  </h4>
                  <dl className="text-[11px] text-zinc-600 space-y-1">
                    {selectedNode.provenance.source && (
                      <div className="flex justify-between gap-2">
                        <dt className="text-zinc-400">Source</dt>
                        <dd className="font-mono">{selectedNode.provenance.source}</dd>
                      </div>
                    )}
                    {selectedNode.provenance.section && (
                      <div className="flex justify-between gap-2">
                        <dt className="text-zinc-400">Section</dt>
                        <dd className="font-mono truncate">{selectedNode.provenance.section}</dd>
                      </div>
                    )}
                    {selectedNode.provenance.page && (
                      <div className="flex justify-between gap-2">
                        <dt className="text-zinc-400">Text page</dt>
                        <dd className="font-mono">{selectedNode.provenance.page}</dd>
                      </div>
                    )}
                    {selectedNode.provenance.figure_id && (
                      <div className="flex justify-between gap-2">
                        <dt className="text-zinc-400">Figure</dt>
                        <dd className="font-mono">{selectedNode.provenance.figure_id}</dd>
                      </div>
                    )}
                    {selectedNode.provenance.figure_page && (
                      <div className="flex justify-between gap-2">
                        <dt className="text-zinc-400">Figure page</dt>
                        <dd className="font-mono">{selectedNode.provenance.figure_page}</dd>
                      </div>
                    )}
                    {selectedNode.confidence !== undefined && (
                      <div className="flex justify-between gap-2">
                        <dt className="text-zinc-400">Confidence</dt>
                        <dd className="font-mono">{(selectedNode.confidence * 100).toFixed(0)}%</dd>
                      </div>
                    )}
                    {selectedNode.provenance.citation && (
                      <div className="mt-2">
                        <dt className="text-zinc-400 mb-0.5">Citation</dt>
                        <dd className="text-zinc-700 italic leading-relaxed">
                          &ldquo;{selectedNode.provenance.citation}&rdquo;
                        </dd>
                      </div>
                    )}
                  </dl>
                </div>
              )}

              {/* Position */}
              {selectedMeta && (
                <div className="pt-2 border-t border-zinc-100">
                  <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-300">
                    {selectedMeta.source === "figure"
                      ? `Figure pos (${(selectedNode.bbox?.x ?? 0).toFixed(2)}, ${(selectedNode.bbox?.y ?? 0).toFixed(2)})`
                      : `Layer ${selectedMeta.layer + 1} · Position ${selectedMeta.index + 1}`}
                  </span>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center p-6">
            <div className="text-center">
              <div className="w-12 h-12 rounded-full bg-zinc-50 border border-zinc-100 flex items-center justify-center mx-auto mb-3">
                <svg className="w-5 h-5 text-zinc-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                </svg>
              </div>
              <p className="text-sm font-medium text-zinc-500">Select a component</p>
              <p className="text-xs text-zinc-300 mt-1 leading-relaxed">
                Click any block in the diagram to inspect its function, formulas, connections, and provenance.
              </p>
              {isFigureMode && (
                <p className="text-[10px] text-emerald-600 mt-2">
                  Figure-faithful mode active.
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
