"use client";

import React, { useState, useMemo, useRef, useCallback, useEffect } from "react";

// ── Types ──────────────────────────────────────────────────────────
export interface DiagramNode {
  id: string;
  label: string;
  type: string;
  description?: string;
  math_formula?: string;
  parameters?: string;
  flops?: string;
}

export interface DiagramEdge {
  source: string;
  target: string;
  label?: string;
  type?: "data_flow" | "residual_skip" | "cross_attention";
}

export interface ArchitectureDiagramProps {
  nodes: DiagramNode[];
  edges: DiagramEdge[];
  height?: number;
}

// ── Constants ──────────────────────────────────────────────────────
const NODE_W = 160;
const NODE_H = 44;
const LAYER_GAP = 200;
const NODE_GAP = 24;
const PADDING = 60;

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

// ── Layout Engine ──────────────────────────────────────────────────
interface PositionedNode {
  id: string;
  x: number;
  y: number;
  layer: number;
  index: number;
}

function computeLayout(nodes: DiagramNode[], edges: DiagramEdge[]) {
  if (nodes.length === 0) return new Map<string, PositionedNode>();

  // Build adjacency
  const adjacency = new Map<string, Set<string>>();
  const inDegree = new Map<string, number>();
  nodes.forEach((n) => {
    adjacency.set(n.id, new Set());
    inDegree.set(n.id, 0);
  });
  edges.forEach((e) => {
    if (adjacency.has(e.source) && adjacency.has(e.target)) {
      adjacency.get(e.source)!.add(e.target);
      inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
    }
  });

  // BFS layering (Kahn's algorithm)
  const layers: string[][] = [];
  let current = nodes.filter((n) => (inDegree.get(n.id) || 0) === 0).map((n) => n.id);
  const visited = new Set<string>();

  while (current.length > 0) {
    layers.push([...current]);
    current.forEach((id) => visited.add(id));
    const next: string[] = [];
    current.forEach((id) => {
      adjacency.get(id)?.forEach((target) => {
        if (!visited.has(target)) {
          const newDeg = (inDegree.get(target) || 0) - 1;
          inDegree.set(target, newDeg);
          if (newDeg <= 0 && !next.includes(target)) {
            next.push(target);
          }
        }
      });
    });
    current = next;
  }

  // Handle cycles — add remaining nodes to last layer
  const remaining = nodes.filter((n) => !visited.has(n.id)).map((n) => n.id);
  if (remaining.length > 0) {
    layers.push(remaining);
  }

  // Assign positions
  const positioned = new Map<string, PositionedNode>();
  layers.forEach((layer, layerIdx) => {
    layer.forEach((id, nodeIdx) => {
      positioned.set(id, {
        id,
        layer: layerIdx,
        index: nodeIdx,
        x: PADDING + layerIdx * LAYER_GAP,
        y: PADDING + nodeIdx * (NODE_H + NODE_GAP),
      });
    });
  });

  return positioned;
}

// ── Edge Path Calculator ───────────────────────────────────────────
function computeEdgePath(
  source: PositionedNode,
  target: PositionedNode
): { path: string; midX: number; midY: number } {
  const sx = source.x + NODE_W;
  const sy = source.y + NODE_H / 2;
  const tx = target.x;
  const ty = target.y + NODE_H / 2;

  const dx = tx - sx;
  const controlOffset = Math.max(40, dx * 0.4);
  const path = `M ${sx} ${sy} C ${sx + controlOffset} ${sy}, ${tx - controlOffset} ${ty}, ${tx} ${ty}`;
  const midX = (sx + tx) / 2;
  const midY = (sy + ty) / 2;

  return { path, midX, midY };
}

// ── Component ──────────────────────────────────────────────────────
export default function ArchitectureDiagram({
  nodes,
  edges,
  height = 600,
}: ArchitectureDiagramProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });

  // Compute layout
  const positioned = useMemo(() => computeLayout(nodes, edges), [nodes, edges]);

  // Canvas dimensions
  const { canvasW, canvasH } = useMemo(() => {
    if (positioned.size === 0) return { canvasW: 800, canvasH: 400 };
    let maxX = 0;
    let maxY = 0;
    positioned.forEach((p) => {
      maxX = Math.max(maxX, p.x + NODE_W);
      maxY = Math.max(maxY, p.y + NODE_H);
    });
    return { canvasW: maxX + PADDING, canvasH: maxY + PADDING };
  }, [positioned]);

  // Edge paths
  const edgePaths = useMemo(() => {
    return edges
      .map((e) => {
        const src = positioned.get(e.source);
        const tgt = positioned.get(e.target);
        if (!src || !tgt) return null;
        const { path, midX, midY } = computeEdgePath(src, tgt);
        return { ...e, path, midX, midY };
      })
      .filter(Boolean) as (DiagramEdge & { path: string; midX: number; midY: number })[];
  }, [edges, positioned]);

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

  // Reset view
  const resetView = () => setTransform({ x: 0, y: 0, scale: 1 });

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
            {/* Arrow marker */}
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
              </marker>
              <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82f6" />
              </marker>
            </defs>

            {/* Edges */}
            {edgePaths.map((e) => {
              const isActive =
                selectedId && (e.source === selectedId || e.target === selectedId);
              const isResidual = e.type === "residual_skip";
              return (
                <g key={`${e.source}-${e.target}`}>
                  <path
                    d={e.path}
                    fill="none"
                    stroke={isActive ? "#3b82f6" : isResidual ? "#f87171" : "#d1d5db"}
                    strokeWidth={isActive ? 2.5 : 1.5}
                    strokeDasharray={isResidual ? "6 4" : undefined}
                    markerEnd={isActive ? "url(#arrow-active)" : "url(#arrow)"}
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
              if (!pos) return null;
              const colors = getNodeColors(node.type);
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
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className="cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedId(isSelected ? null : node.id);
                  }}
                  onMouseEnter={() => setHoveredId(node.id)}
                  onMouseLeave={() => setHoveredId(null)}
                >
                  {/* Shadow on hover/select */}
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

                  {/* Main box */}
                  <rect
                    width={NODE_W}
                    height={NODE_H}
                    rx={8}
                    fill={colors.bg}
                    stroke={isSelected ? "#3b82f6" : colors.border}
                    strokeWidth={isSelected ? 2 : 1}
                    opacity={
                      selectedId && !isSelected && !isConnected ? 0.35 : 1}
                    className="transition-all duration-150"
                  />

                  {/* Type indicator bar */}
                  <rect
                    width={3}
                    height={NODE_H - 12}
                    x={6}
                    y={6}
                    rx={1.5}
                    fill={colors.border}
                    opacity={0.6}
                  />

                  {/* Label */}
                  <text
                    x={NODE_W / 2 + 4}
                    y={NODE_H / 2 - 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="text-[12px] font-medium pointer-events-none select-none"
                    fill={colors.text}
                  >
                    {node.label.length > 20
                      ? node.label.slice(0, 20) + "…"
                      : node.label}
                  </text>

                  {/* Type badge */}
                  <text
                    x={NODE_W / 2 + 4}
                    y={NODE_H / 2 + 14}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="text-[9px] pointer-events-none select-none uppercase tracking-wider"
                    fill={colors.text}
                    opacity={0.5}
                  >
                    {node.type}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Empty state */}
        {nodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-zinc-400">
              <svg className="w-12 h-12 mx-auto mb-3 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
              <p className="text-sm font-medium">No architecture data</p>
              <p className="text-xs mt-1 text-zinc-300">Upload a paper to extract its architecture</p>
            </div>
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
              </div>
              <h3 className="text-base font-bold text-zinc-900 mt-2">
                {selectedNode.label}
              </h3>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* Description */}
              {selectedNode.description && (
                <div>
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                    Description
                  </h4>
                  <p className="text-sm text-zinc-700 leading-relaxed">
                    {selectedNode.description}
                  </p>
                </div>
              )}

              {/* Math formula */}
              {selectedNode.math_formula && (
                <div>
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                    Mathematical Formula
                  </h4>
                  <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-100 font-mono text-sm text-zinc-800 overflow-x-auto">
                    {selectedNode.math_formula}
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

              {/* Position */}
              {selectedMeta && (
                <div className="pt-2 border-t border-zinc-100">
                  <span className="text-[9px] font-semibold uppercase tracking-wider text-zinc-300">
                    Layer {selectedMeta.layer + 1} · Position {selectedMeta.index + 1}
                  </span>
                </div>
              )}
            </div>
          </>
        ) : (
          /* Empty state */
          <div className="flex-1 flex items-center justify-center p-6">
            <div className="text-center">
              <div className="w-12 h-12 rounded-full bg-zinc-50 border border-zinc-100 flex items-center justify-center mx-auto mb-3">
                <svg className="w-5 h-5 text-zinc-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                </svg>
              </div>
              <p className="text-sm font-medium text-zinc-500">Select a component</p>
              <p className="text-xs text-zinc-300 mt-1 leading-relaxed">
                Click any block in the diagram to inspect its function, formulas, and connections.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
