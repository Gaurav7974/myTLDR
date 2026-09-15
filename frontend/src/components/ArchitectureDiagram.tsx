"use client";

import React, { useState, useMemo, useRef, useCallback, useEffect } from "react";
import katex from "katex";
import {
  Layers,
  Cpu,
  ArrowRight,
  Sparkles,
  X,
  ChevronRight,
  FileCode2,
  CheckCircle2,
  Box,
} from "lucide-react";
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
  PADDING,
  NODE_W_DEFAULT,
  NODE_H_DEFAULT,
  figureNodePosition,
  computeAutoLayout,
  computeElkLayout,
  positionNodesFigure,
  positionGroupsFigure,
  routeEdgeOrthogonal,
  resolveEdgeStyle as resolveEdgeStyleImpl,
  type PositionedNode,
  type PositionedGroup,
  type RoutedEdgePath,
  type NormalizedNode,
} from "./architectureDiagramLayout";
import { wrapToWidth } from "./diagramNormalizer";

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
  paperTitle?: string;
  isUnverified?: boolean;
  /** Optional callback when the user clicks a node — useful for tests. */
  onNodeClick?: (node: DiagramNode) => void;
}

// ── Constants (NODE_W_DEFAULT, NODE_H_DEFAULT, PADDING, FIGURE_CANVAS_W/H are imported
//    from ./architectureDiagramLayout; LAYER_GAP/NODE_GAP are auto-layout
//    constants and live with computeAutoLayout there.)

const TYPE_COLORS: Record<string, { bg: string; border: string; text: string; tagText?: string }> = {
  input: { bg: "#ffffff", border: "#475569", text: "#0f172a", tagText: "#475569" },
  output: { bg: "#ffffff", border: "#475569", text: "#0f172a", tagText: "#475569" },
  embedding: { bg: "#f0f9ff", border: "#0284c7", text: "#0369a1", tagText: "#0284c7" },
  encoder: { bg: "#eff6ff", border: "#2563eb", text: "#1d4ed8", tagText: "#1d4ed8" },
  decoder: { bg: "#faf5ff", border: "#9333ea", text: "#7e22ce", tagText: "#9333ea" },
  attention: { bg: "#fdf2f8", border: "#db2777", text: "#9d174d", tagText: "#be185d" },
  feedforward: { bg: "#f0fdf4", border: "#16a34a", text: "#15803d", tagText: "#16a34a" },
  normalization: { bg: "#fafaf9", border: "#78716c", text: "#44403c", tagText: "#78716c" },
  block: { bg: "#faf5ff", border: "#8b5cf6", text: "#6d28d9", tagText: "#8b5cf6" },
  router: { bg: "#fff7ed", border: "#ea580c", text: "#c2410c", tagText: "#ea580c" },
  expert: { bg: "#ecfdf5", border: "#059669", text: "#047857", tagText: "#059669" },
  ssm: { bg: "#ecfeff", border: "#0891b2", text: "#0e7490", tagText: "#0891b2" },
  gate: { bg: "#fefce8", border: "#ca8a04", text: "#a16207", tagText: "#ca8a04" },
  cache: { bg: "#f5f3ff", border: "#7c3aed", text: "#5b21b6", tagText: "#7c3aed" },
  linear: { bg: "#f8fafc", border: "#475569", text: "#1e293b", tagText: "#475569" },
  convolution: { bg: "#ecfeff", border: "#0284c7", text: "#0369a1", tagText: "#0284c7" },
  residual: { bg: "#fef2f2", border: "#dc2626", text: "#b91c1c", tagText: "#dc2626" },
  default: { bg: "#ffffff", border: "#64748b", text: "#0f172a", tagText: "#64748b" },
};

function getNodeColors(type: string) {
  const t = (type || "default").toLowerCase();
  return TYPE_COLORS[t] || TYPE_COLORS.default;
}

// ── Math rendering (Safe KaTeX rendering with fallback) ──────────────
function MathText({ value, displayMode = false }: { value: string; displayMode?: boolean }) {
  const rendered = useMemo(() => {
    if (!value) return null;
    try {
      return katex.renderToString(value, {
        displayMode,
        throwOnError: false,
      });
    } catch {
      return null;
    }
  }, [value, displayMode]);

  if (!rendered) {
    return <span className="font-mono text-xs text-zinc-700 break-all">{value}</span>;
  }
  return <span className="katex-rendered inline-block" dangerouslySetInnerHTML={{ __html: rendered }} />;
}

// ── Component ──────────────────────────────────────────────────────
export default function ArchitectureDiagram({
  nodes: nodesProp,
  edges: edgesProp,
  graph,
  height = 600,
  paperTitle,
  isUnverified,
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
  const hasFigureBboxes = useMemo(() => {
    return nodes.some(
      (n) => n.bbox && typeof n.bbox.x === "number" && typeof n.bbox.y === "number" && Number.isFinite(n.bbox.x) && Number.isFinite(n.bbox.y)
    );
  }, [nodes]);

  const isFigureAvailable = useMemo(() => {
    return hasFigureBboxes && shouldRenderReconstruction(graph as ArchitectureGraph | undefined);
  }, [hasFigureBboxes, graph]);

  // Default to Structured (ELK) for clean D2-style containerized layout, with paper figure toggle when available
  const [layoutView, setLayoutView] = useState<"elk" | "figure">("elk");
  const isEffectiveFigureMode = isFigureAvailable && layoutView === "figure";

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0 });

  const initialPositioned = useMemo(() => {
    if (isEffectiveFigureMode) {
      const fig = positionNodesFigure(nodes, true);
      if (fig.size > 0) return fig;
    }
    return computeAutoLayout(nodes, edges);
  }, [isEffectiveFigureMode, nodes, edges]);

  const [positioned, setPositioned] = useState<Map<string, PositionedNode>>(initialPositioned);
  const [elkRoutedEdges, setElkRoutedEdges] = useState<Map<string, RoutedEdgePath> | null>(null);
  const [elkGroups, setElkGroups] = useState<PositionedGroup[] | null>(null);

  useEffect(() => {
    if (isEffectiveFigureMode) {
      const fig = positionNodesFigure(nodes, true);
      setPositioned(fig.size > 0 ? fig : computeAutoLayout(nodes, edges));
      setElkRoutedEdges(null);
      setElkGroups(null);
      return;
    }

    let cancelled = false;
    if (nodes.length > 0) {
      computeElkLayout(nodes, edges, groups).then((elkResult) => {
        if (!cancelled && elkResult && elkResult.nodes && elkResult.nodes.size > 0) {
          setPositioned(elkResult.nodes);
          setElkRoutedEdges(elkResult.edges);
          if (elkResult.groups && elkResult.groups.length > 0) {
            setElkGroups(elkResult.groups);
          }
        }
      });
    }
    return () => {
      cancelled = true;
    };
  }, [isEffectiveFigureMode, nodes, edges, groups]);

  // Mark nodes that have no layout as "missing" so the renderer can draw
  // them safely (or hide them).
  const nodesWithMissingLayout = useMemo(() => {
    return nodes.filter((n) => !positioned.has(n.id));
  }, [nodes, positioned]);

  // Group rectangles — compute tight bounds around member nodes
  const positionedGroups = useMemo(() => {
    if (!isEffectiveFigureMode && elkGroups && elkGroups.length > 0) {
      return elkGroups;
    }
    if (groups.length === 0) return [];
    return positionGroupsFigure(groups, positioned, nodes as unknown as NormalizedNode[]);
  }, [groups, positioned, nodes, elkGroups, isEffectiveFigureMode]);

  // Canvas dimensions and bounding box
  const { minX, minY, maxX, maxY, canvasW, canvasH } = useMemo(() => {
    if (positioned.size === 0) return { minX: 0, minY: 0, maxX: 800, maxY: 400, canvasW: 800, canvasH: 400 };
    let minX = Infinity, minY = Infinity;
    let maxX = 0, maxY = 0;
    positioned.forEach((p: any) => {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + (p.w || NODE_W_DEFAULT));
      maxY = Math.max(maxY, p.y + (p.h || NODE_H_DEFAULT));
    });
    if (positionedGroups && positionedGroups.length > 0) {
      positionedGroups.forEach((g) => {
        minX = Math.min(minX, g.x);
        minY = Math.min(minY, g.y);
        maxX = Math.max(maxX, g.x + g.w);
        maxY = Math.max(maxY, g.y + g.h);
      });
    }
    const safeMinX = Number.isFinite(minX) ? minX : 0;
    const safeMinY = Number.isFinite(minY) ? minY : 0;
    const contentMaxX = Math.max(maxX, safeMinX + 100);
    const contentMaxY = Math.max(maxY, safeMinY + 100);
    return {
      minX: safeMinX,
      minY: safeMinY,
      maxX: contentMaxX,
      maxY: contentMaxY,
      canvasW: Math.max(800, contentMaxX + PADDING),
      canvasH: Math.max(600, contentMaxY + PADDING),
    };
  }, [positioned, positionedGroups, isEffectiveFigureMode]);

  // Edge paths — cleanly routed between node boundary ports
  const edgePaths = useMemo(() => {
    return edges
      .map((e, idx) => {
        const src = positioned.get(e.source);
        const tgt = positioned.get(e.target);
        if (!src || !tgt) return null;

        const elkKey = `${e.source}->${e.target}`;
        const elkEdge = elkRoutedEdges?.get(elkKey) || elkRoutedEdges?.get(`e${idx}`);
        if (elkEdge && !isEffectiveFigureMode) {
          return { ...e, ...elkEdge };
        }

        const routed = routeEdgeOrthogonal(src, tgt, e, canvasW);
        return { ...e, ...routed };
      })
      .filter(Boolean) as (DiagramEdge & { path: string; midX: number; midY: number })[];
  }, [edges, positioned, canvasW, elkRoutedEdges, isEffectiveFigureMode]);

  const selectedNode = selectedId ? nodes.find((n) => n.id === selectedId) : null;
  const selectedMeta = selectedId ? positioned.get(selectedId) : null;

  // Auto-fit diagram cleanly within viewport
  const fitToView = useCallback(() => {
    const el = containerRef.current;
    if (!el || positioned.size === 0) return;
    const containerW = el.clientWidth || 800;
    const containerH = el.clientHeight || height || 600;

    const diagW = maxX - minX + PADDING;
    const diagH = maxY - minY + PADDING;
    if (diagW <= 0 || diagH <= 0) return;

    const availW = Math.max(100, containerW - 48);
    const availH = Math.max(100, containerH - 48);

    const scaleX = availW / diagW;
    const scaleY = availH / diagH;
    const scale = Math.min(1.0, Math.max(0.3, Math.min(scaleX, scaleY)));

    const scaledDiagW = diagW * scale;
    const scaledDiagH = diagH * scale;

    const x = (containerW - scaledDiagW) / 2 - minX * scale;
    const y = Math.max(16, (containerH - scaledDiagH) / 2 - minY * scale);

    setTransform({ x, y, scale });
  }, [minX, minY, maxX, maxY, positioned.size, height]);

  // Auto-fit on layout calculation
  useEffect(() => {
    if (positioned.size > 0) {
      const raf = requestAnimationFrame(() => {
        fitToView();
      });
      return () => cancelAnimationFrame(raf);
    }
  }, [positioned, fitToView]);

  // Pan handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if ((e.target as HTMLElement).closest("[data-node]")) return;
      if ((e.target as HTMLElement).closest("[data-popup]")) return;
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

  const resetView = () => fitToView();

  const isAvailable = nodes.length > 0;

  return (
    <div className="relative flex h-full w-full bg-white rounded-xl border border-zinc-200 overflow-hidden">
      {/* ── Canvas ─────────────────────────────────────────────── */}
      <div
        ref={containerRef}
        className="flex-1 relative overflow-hidden cursor-grab active:cursor-grabbing w-full h-full"
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

        {/* Zoom controls (docked at bottom-right) */}
        <div className="absolute bottom-3 right-3 z-20 flex flex-col gap-1">
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

        {/* Layout Mode Toggle & Badges */}
        <div className="absolute top-3 left-3 z-20 flex items-center gap-1.5 p-1 bg-white/95 backdrop-blur-xs border border-zinc-200 rounded-lg shadow-2xs text-[11px]">
          <button
            onClick={() => setLayoutView("elk")}
            className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1.5 font-medium ${
              layoutView === "elk"
                ? "bg-blue-600 text-white font-semibold shadow-2xs"
                : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100"
            }`}
            title="Clean, hierarchical architecture layout with orthogonal arrows (D2 / ELK)"
          >
            <Layers className="w-3.5 h-3.5" />
            Structured Layout (ELK)
          </button>
          {isFigureAvailable && (
            <button
              onClick={() => setLayoutView("figure")}
              className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1.5 font-medium ${
                layoutView === "figure"
                  ? "bg-emerald-600 text-white font-semibold shadow-2xs"
                  : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100"
              }`}
              title="Spatial layout matching where bounding boxes were detected in the paper figure"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Paper Figure (Spatial)
            </button>
          )}
          {isUnverified && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-50 border border-amber-200 text-amber-800 text-[11px] font-mono font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
              <span>Unverified</span>
            </div>
          )}
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
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="4.5" markerWidth="8" markerHeight="8" orient="auto">
                <polygon points="0 1, 8 4.5, 0 8" fill="#64748b" />
              </marker>
              <marker id="arrow-active" viewBox="0 0 10 10" refX="8" refY="4.5" markerWidth="8" markerHeight="8" orient="auto">
                <polygon points="0 1, 8 4.5, 0 8" fill="#2563eb" />
              </marker>
              <marker id="arrow-residual" viewBox="0 0 10 10" refX="8" refY="4.5" markerWidth="8" markerHeight="8" orient="auto">
                <polygon points="0 1, 8 4.5, 0 8" fill="#ef4444" />
              </marker>
              <marker id="arrow-cross" viewBox="0 0 10 10" refX="8" refY="4.5" markerWidth="8" markerHeight="8" orient="auto">
                <polygon points="0 1, 8 4.5, 0 8" fill="#3b82f6" />
              </marker>
            </defs>

            {/* Group containers with floating header badge pill */}
            {positionedGroups.map((g) => {
              const labelW = Math.max(70, g.label.length * 6.5 + 24);
              return (
                <g key={`group-${g.id}`} className="group-container">
                  <rect
                    x={g.x}
                    y={g.y}
                    width={g.w}
                    height={g.h}
                    fill={g.color || "#f8fafc"}
                    fillOpacity={0.04}
                    stroke={g.color || "#cbd5e1"}
                    strokeOpacity={0.7}
                    strokeWidth={1.5}
                    strokeDasharray="5 4"
                    rx={10}
                  />
                  {/* Floating header pill badge on the top border */}
                  <g transform={`translate(${g.x + 14}, ${g.y - 10})`}>
                    <rect
                      x={0}
                      y={0}
                      width={labelW}
                      height={20}
                      rx={10}
                      fill="#ffffff"
                      stroke={g.color || "#94a3b8"}
                      strokeOpacity={0.8}
                      strokeWidth={1.2}
                    />
                    <text
                      x={labelW / 2}
                      y={14}
                      textAnchor="middle"
                      className="text-[10px] font-bold uppercase tracking-wider select-none pointer-events-none"
                      fill="#475569"
                    >
                      {g.label}
                    </text>
                  </g>
                </g>
              );
            })}

            {/* Edges */}
            {edgePaths.map((e) => {
              const isActive =
                selectedId && (e.source === selectedId || e.target === selectedId);
              const { stroke, dasharray, width, marker } = resolveEdgeStyleImpl(e, !!isActive);
              const labelW = e.label ? Math.max(50, e.label.length * 6.2 + 16) : 0;
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
                    <g transform={`translate(${e.midX}, ${e.midY})`}>
                      <rect
                        x={-labelW / 2}
                        y={-9}
                        width={labelW}
                        height={18}
                        rx={4}
                        fill="#ffffff"
                        stroke="#e2e8f0"
                        strokeWidth={1}
                        className="shadow-sm"
                      />
                      <text
                        x={0}
                        y={3.5}
                        textAnchor="middle"
                        className="text-[9.5px] font-medium fill-zinc-600 pointer-events-none select-none"
                      >
                        {e.label}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}

            {/* Nodes — shape-faithful: rect/ellipse/diamond/cylinder/parallelogram/trapezoid per original */}
            {nodes.map((node) => {
              const pos = positioned.get(node.id);
              if (!pos) return null;
              const colors = getNodeColors(node.type);
              const fill = node.style?.fill || colors.bg;
              const border = node.style?.border || colors.border;
              const textColor = node.style?.text || colors.text;
              const isSelected = selectedId === node.id;
              const isHovered = hoveredId === node.id;
              const isConnected = selectedId && edges.some((e) => (e.source === selectedId && e.target === node.id) || (e.target === selectedId && e.source === node.id));
              const w = (pos as any).w || NODE_W_DEFAULT, h = (pos as any).h || NODE_H_DEFAULT;
              const shape = (node.shape || node.style?.shape || "rounded") as string;
              const stroke = isSelected ? "#3b82f6" : border, sw = isSelected ? 2.5 : 1.5;
              const dim = selectedId && !isSelected && !isConnected ? 0.35 : 1;

              const shapeEl = (() => {
                const p = { fill, stroke, strokeWidth: sw, opacity: dim, className: "transition-all duration-150" } as any;
                if (shape === "ellipse") return <ellipse cx={w/2} cy={h/2} rx={w/2} ry={h/2} {...p} />;
                if (shape === "diamond") return <polygon points={`${w/2},0 ${w},${h/2} ${w/2},${h} 0,${h/2}`} {...p} />;
                if (shape === "parallelogram") { const o=w*0.15; return <polygon points={`${o},0 ${w},0 ${w-o},${h} 0,${h}`} {...p} />; }
                if (shape === "trapezoid") { const o=w*0.2; return <polygon points={`${o},0 ${w-o},0 ${w},${h} 0,${h}`} {...p} />; }
                if (shape === "cylinder") return <g opacity={dim}><rect width={w} height={h} rx={6} {...p} /><ellipse cx={w/2} cy={6} rx={w/2-1} ry={6} fill={fill} stroke={border} strokeWidth={1} opacity={0.6} /></g>;
                const rx = shape === "rect" ? 2 : 8;
                return <rect width={w} height={h} rx={rx} {...p} />;
              })();

              return (
                <g key={node.id} data-node data-node-id={node.id} transform={`translate(${pos.x}, ${pos.y})`} className="cursor-pointer"
                  onClick={(e) => { e.stopPropagation(); setSelectedId(isSelected ? null : node.id); onNodeClick?.(node); }}
                  onMouseEnter={() => setHoveredId(node.id)} onMouseLeave={() => setHoveredId(null)}>
                  {(isHovered || isSelected) && <rect x={-2} y={-2} width={w+4} height={h+4} rx={10} fill="none" stroke={isSelected ? "#3b82f6" : "#94a3b8"} strokeWidth={sw} opacity={0.5} />}
                  {shapeEl}
                  {(() => {
                    const labelText = node.label || node.id;
                    const maxTextW = Math.max(60, w - 20);
                    const lines = wrapToWidth(labelText, maxTextW, 11);
                    const lineH = 14;
                    const hasType = !!node.type && h >= 46;
                    const startY = hasType ? 16 : (h / 2 - ((lines.length - 1) * lineH) / 2);

                    return (
                      <g opacity={dim} className="pointer-events-none select-none">
                        {hasType && (
                          <text
                            x={w / 2}
                            y={14}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="text-[8.5px] font-bold uppercase tracking-wider"
                            fill={colors.tagText || textColor}
                            opacity={0.8}
                          >
                            {node.type}
                          </text>
                        )}
                        {lines.map((line, idx) => (
                          <text
                            key={idx}
                            x={w / 2}
                            y={(hasType ? 30 : startY) + idx * lineH}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="text-[11px] font-semibold"
                            fill={textColor}
                          >
                            {line}
                          </text>
                        ))}
                      </g>
                    );
                  })()}
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
        {isEffectiveFigureMode && nodesWithMissingLayout.length > 0 && (
          <div className="absolute bottom-3 left-3 z-20 px-2 py-1 rounded bg-amber-50 border border-amber-200 text-amber-800 text-[10px]">
            {nodesWithMissingLayout.length} node(s) without extracted bbox
          </div>
        )}
      </div>

        {/* Concise Block Info Pop-up Card (Only shown when a node/block is clicked) */}
        {selectedNode && (
          <div
            data-popup
            className="absolute top-3 right-3 z-30 w-80 sm:w-88 max-w-[calc(100%-1.5rem)] bg-white/95 backdrop-blur-md rounded-xl border border-zinc-200/90 shadow-xl overflow-hidden animate-fadeIn"
          >
            {/* Card Header */}
            <div className="px-4 py-3 border-b border-zinc-100 flex items-start justify-between gap-2 bg-zinc-50/60">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                  <span
                    className="text-[9px] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded border shadow-2xs"
                    style={{
                      backgroundColor: getNodeColors(selectedNode.type).bg,
                      color: getNodeColors(selectedNode.type).text,
                      borderColor: getNodeColors(selectedNode.type).border,
                    }}
                  >
                    {selectedNode.type.toUpperCase()}
                  </span>
                  {selectedNode.confidence !== undefined && (
                    <span className="text-[9px] font-mono font-medium px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                      {(selectedNode.confidence * 100).toFixed(0)}% verified
                    </span>
                  )}
                </div>
                <h4 className="text-sm font-bold text-zinc-950 leading-snug truncate" title={selectedNode.label}>
                  {selectedNode.label}
                </h4>
                {selectedNode.paper_term && (
                  <p className="text-[10.5px] font-mono text-zinc-500 mt-0.5 truncate">
                    Paper term: <span className="text-zinc-700 font-semibold">{selectedNode.paper_term}</span>
                  </p>
                )}
              </div>
              <button
                onClick={() => setSelectedId(null)}
                className="p-1 rounded-md text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors shrink-0"
                title="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Card Content - concise info only */}
            <div className="p-4 space-y-3 max-h-[60vh] overflow-y-auto text-left select-text">
              {/* Description / Role */}
              {(selectedNode.description || selectedNode.how_it_works) && (
                <p className="text-xs text-zinc-700 leading-relaxed">
                  {selectedNode.description || selectedNode.how_it_works}
                </p>
              )}

              {/* Mathematical Formula if present */}
              {selectedNode.math_formula && (
                <div className="p-2.5 bg-zinc-50 rounded-lg border border-zinc-200/80 text-center overflow-x-auto text-xs text-zinc-900 shadow-2xs">
                  <MathText value={selectedNode.math_formula} displayMode={true} />
                </div>
              )}

              {/* Parameters & FLOPs */}
              {(selectedNode.parameters || selectedNode.flops) && (
                <div className="grid grid-cols-2 gap-2 pt-1">
                  {selectedNode.parameters && (
                    <div className="p-2 rounded-lg bg-zinc-50 border border-zinc-200/70">
                      <span className="text-[9px] font-mono uppercase text-zinc-400 block">Params</span>
                      <span className="text-xs font-mono font-bold text-zinc-900 truncate block">
                        {selectedNode.parameters}
                      </span>
                    </div>
                  )}
                  {selectedNode.flops && (
                    <div className="p-2 rounded-lg bg-zinc-50 border border-zinc-200/70">
                      <span className="text-[9px] font-mono uppercase text-zinc-400 block">FLOPs</span>
                      <span className="text-xs font-mono font-bold text-zinc-900 truncate block">
                        {selectedNode.flops}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Source Provenance */}
              {(selectedNode.provenance?.page || selectedNode.provenance?.section || selectedNode.provenance?.figure_id) && (
                <div className="pt-2 border-t border-zinc-100 flex items-center justify-between text-[10px] font-mono text-zinc-500">
                  <span>Source:</span>
                  <span className="font-medium text-zinc-700 truncate max-w-[180px]">
                    {selectedNode.provenance?.section || selectedNode.provenance?.figure_id || 'Paper'}
                    {selectedNode.provenance?.page ? ` (p.${selectedNode.provenance.page})` : ''}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
    </div>
  );
}
