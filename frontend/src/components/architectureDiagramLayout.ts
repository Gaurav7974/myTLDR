/**
 * Pure layout and style helpers extracted from ArchitectureDiagram for
 * testability. These functions have no React or DOM dependencies and can
 * be exercised in plain Node tests.
 *
 * Mirrors the logic in `ArchitectureDiagram.tsx` — when updating the
 * component, update these helpers in lockstep (the QA fixtures assert
 * the matching behavior).
 */

import type { GraphBoundingBox, GraphPoint, GraphGroup } from "@/types/architecture_graph";

export const FIGURE_CANVAS_W = 1000;
export const FIGURE_CANVAS_H = 600;
export const NODE_W_DEFAULT = 160;
export const NODE_H_DEFAULT = 44;
export const LAYER_GAP = 200;
export const NODE_GAP = 24;
export const PADDING = 60;

export interface PositionedNode {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  layer: number;
  index: number;
  source: "auto" | "figure" | "missing";
}

export interface NormalizedNode {
  id: string;
  label: string;
  type: string;
  bbox?: GraphBoundingBox;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [k: string]: any;
}

export interface NormalizedEdge {
  source: string;
  target: string;
  path_points?: GraphPoint[] | null;
  type?: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [k: string]: any;
}

/** Convert a normalized bbox to figure-canvas pixel coordinates. */
export function figureNodePosition(node: NormalizedNode): {
  x: number; y: number; w: number; h: number;
} | null {
  const b = node.bbox;
  if (!b) return null;
  if (
    typeof b.x !== "number" || typeof b.y !== "number" ||
    typeof b.w !== "number" || typeof b.h !== "number"
  ) return null;
  // Reject NaN / Infinity explicitly — they pass numeric comparison checks.
  if (!Number.isFinite(b.x) || !Number.isFinite(b.y) ||
      !Number.isFinite(b.w) || !Number.isFinite(b.h)) return null;
  if (b.w <= 0 || b.h <= 0) return null;
  if (b.x < 0 || b.y < 0 || b.x > 1 || b.y > 1) return null;
  // Reject boxes that extend beyond the normalized canvas (with a tiny epsilon
  // for float quantization). This matches the backend BoundingBox validator
  // and prevents rendering off-canvas nodes.
  if (b.x + b.w > 1.0 + 1e-6) return null;
  if (b.y + b.h > 1.0 + 1e-6) return null;
  const w = Math.max(60, b.w * FIGURE_CANVAS_W);
  const h = Math.max(28, b.h * FIGURE_CANVAS_H);
  return {
    x: b.x * FIGURE_CANVAS_W,
    y: b.y * FIGURE_CANVAS_H,
    w,
    h,
  };
}

/** BFS layered auto-layout (Kahn's algorithm). */
export function computeAutoLayout(
  nodes: NormalizedNode[],
  edges: NormalizedEdge[]
): Map<string, PositionedNode> {
  if (nodes.length === 0) return new Map();
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
          if (newDeg <= 0 && !next.includes(target)) next.push(target);
        }
      });
    });
    current = next;
  }
  const remaining = nodes.filter((n) => !visited.has(n.id)).map((n) => n.id);
  if (remaining.length > 0) layers.push(remaining);
  // Vertical stack per original (like Kimi html right stack), not horizontal BFS
  const positioned = new Map<string, PositionedNode>();
  layers.forEach((layer, layerIdx) => {
    layer.forEach((id, nodeIdx) => {
      positioned.set(id, {
        id,
        layer: layerIdx,
        index: nodeIdx,
        x: PADDING + nodeIdx * (NODE_W_DEFAULT + NODE_GAP),
        y: PADDING + layerIdx * 88,
        w: NODE_W_DEFAULT,
        h: NODE_H_DEFAULT,
        source: "auto",
      });
    });
  });
  return positioned;
}

export function positionNodesFigure(
  nodes: NormalizedNode[],
  isFigureMode: boolean
): Map<string, PositionedNode> {
  if (isFigureMode) {
    const m = new Map<string, PositionedNode>();
    nodes.forEach((n) => {
      const p = figureNodePosition(n);
      if (p) m.set(n.id, { id: n.id, x: p.x, y: p.y, w: p.w, h: p.h, layer: 0, index: 0, source: "figure" });
    });
    if (m.size > 0) return m;
  }
  return computeAutoLayout(nodes, []);
}

export interface ResolvedEdgeStyle {
  stroke: string;
  dasharray?: string;
  width: number;
  marker: string;
}

/** Resolve the visual style of an edge given the edge and active flag. */
export function resolveEdgeStyle(
  edge: NormalizedEdge,
  isActive: boolean
): ResolvedEdgeStyle {
  const name = (edge.style?.name) ?? "default";
  // Defensive: an empty string is NOT a valid stroke — fall through to default.
  const rawStroke = edge.style?.stroke;
  let stroke: string | undefined = (rawStroke && rawStroke.length > 0) ? rawStroke : undefined;
  let dasharray: string | undefined;
  // Defensive: a non-positive width is invalid; fall back to default.
  const rawWidth = edge.style?.width;
  let width: number = (typeof rawWidth === "number" && rawWidth > 0)
    ? rawWidth
    : (isActive ? 2.5 : 1.5);

  if (name === "residual" || edge.type === "residual_skip") {
    stroke = isActive ? "#ef4444" : "#f87171";
    dasharray = "6 4";
  } else if (name === "dashed" || edge.type === "cross_attention") {
    stroke = stroke ?? (isActive ? "#3b82f6" : "#94a3b8");
    dasharray = dasharray ?? "5 3";
  } else if (name === "beam") {
    stroke = stroke ?? (isActive ? "#8b5cf6" : "#a78bfa");
    // Override width only when the user did not explicitly set one.
    if (!(typeof rawWidth === "number" && rawWidth > 0)) {
      width = isActive ? 3 : 2.4;
    }
  } else if (name === "highlight") {
    stroke = stroke ?? (isActive ? "#f59e0b" : "#fbbf24");
    if (!(typeof rawWidth === "number" && rawWidth > 0)) {
      width = isActive ? 2.6 : 1.8;
    }
  } else if (name === "dotted") {
    stroke = stroke ?? (isActive ? "#3b82f6" : "#94a3b8");
    dasharray = dasharray ?? "1 4";
  } else {
    stroke = isActive ? "#3b82f6" : stroke ?? "#d1d5db";
  }

  return {
    stroke,
    dasharray,
    width,
    marker: isActive ? "url(#arrow-active)" : "url(#arrow)",
  };
}

/** Group rectangles in figure-canvas pixel coordinates. */
export interface PositionedGroup {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
}

export function positionGroupsFigure(
  groups: GraphGroup[] | undefined
): PositionedGroup[] {
  if (!groups || groups.length === 0) return [];
  const out: PositionedGroup[] = [];
  for (const g of groups) {
    const b = g.bbox;
    if (!b) continue;
    if (
      typeof b.x !== "number" || typeof b.y !== "number" ||
      typeof b.w !== "number" || typeof b.h !== "number"
    ) continue;
    if (!Number.isFinite(b.x) || !Number.isFinite(b.y) ||
        !Number.isFinite(b.w) || !Number.isFinite(b.h)) continue;
    if (b.w <= 0 || b.h <= 0) continue;
    if (b.x < 0 || b.y < 0 || b.x > 1 || b.y > 1) continue;
    if (b.x + b.w > 1.0 + 1e-6) continue;
    if (b.y + b.h > 1.0 + 1e-6) continue;
    out.push({
      id: g.id,
      label: g.label || g.id,
      x: b.x * FIGURE_CANVAS_W,
      y: b.y * FIGURE_CANVAS_H,
      w: b.w * FIGURE_CANVAS_W,
      h: b.h * FIGURE_CANVAS_H,
      color: g.color || "#e5e7eb",
    });
  }
  return out;
}

/** Convert a list of normalized path points to an SVG path string. */
export function pathPointsToSvgPath(points: GraphPoint[]): string {
  if (points.length < 2) return "";
  const segments = points.map((p) => `${p.x * FIGURE_CANVAS_W},${p.y * FIGURE_CANVAS_H}`);
  return `M ${segments.join(" L ")}`;
}
