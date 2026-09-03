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
export const NODE_W_DEFAULT = 170;
export const NODE_H_DEFAULT = 52;
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

let elkInstance: any | null = null;
let elkPromise: Promise<any> | null = null;

async function getElk(): Promise<any | null> {
  if (elkInstance) return elkInstance;
  if (elkPromise) return elkPromise;
  elkPromise = import("elkjs/lib/elk.bundled.js").then((mod: any) => {
    elkInstance = new mod.default();
    return elkInstance;
  }).catch(() => null);
  return elkPromise;
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
  // Vertical stack with horizontal centering per layer
  const positioned = new Map<string, PositionedNode>();
  const maxNodesInLayer = Math.max(...layers.map((l) => l.length), 1);
  const maxWidth = maxNodesInLayer * (NODE_W_DEFAULT + NODE_GAP) - NODE_GAP;
  const startX = PADDING + Math.max(0, (FIGURE_CANVAS_W - maxWidth) / 2);
  const startY = PADDING + 20;

  layers.forEach((layer, layerIdx) => {
    const layerWidth = layer.length * (NODE_W_DEFAULT + NODE_GAP) - NODE_GAP;
    const xOffset = Math.max(0, (maxWidth - layerWidth) / 2);
    layer.forEach((id, nodeIdx) => {
      positioned.set(id, {
        id,
        layer: layerIdx,
        index: nodeIdx,
        x: startX + xOffset + nodeIdx * (NODE_W_DEFAULT + NODE_GAP),
        y: startY + layerIdx * 88,
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
    if (m.size > 0) return relaxNodePositions(m);
  }
  return computeAutoLayout(nodes, []);
}

/**
 * Nudge node positions to guarantee clean clearance between adjacent blocks.
 * Prevents horizontal or vertical overlapping of blocks in figure mode.
 */
export function relaxNodePositions(
  positioned: Map<string, PositionedNode>,
  minVerticalGap = 16,
  minHorizontalGap = 16
): Map<string, PositionedNode> {
  if (positioned.size <= 1) return positioned;
  const list = Array.from(positioned.values()).map((p) => ({ ...p }));

  for (let pass = 0; pass < 3; pass++) {
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const a = list[i];
        const b = list[j];

        const hOverlap = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
        const vOverlap = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);

        // Check if there is any collision or violation of minimum gaps
        const isCollidingY = hOverlap > 8 && vOverlap > -minVerticalGap;
        const isCollidingX = vOverlap > 8 && hOverlap > -minHorizontalGap;

        if (isCollidingY || isCollidingX) {
          const aCenterX = a.x + a.w / 2;
          const bCenterX = b.x + b.w / 2;
          const aCenterY = a.y + a.h / 2;
          const bCenterY = b.y + b.h / 2;

          const dx = Math.abs(aCenterX - bCenterX);
          const dy = Math.abs(aCenterY - bCenterY);

          // Dominant axis: separate along the axis where they are already furthest apart
          if (dy >= dx) {
            // Vertical separation
            if (aCenterY <= bCenterY) {
              const reqY = a.y + a.h + minVerticalGap;
              if (b.y < reqY) b.y = reqY;
            } else {
              const reqY = b.y + b.h + minVerticalGap;
              if (a.y < reqY) a.y = reqY;
            }
          } else {
            // Horizontal separation
            if (aCenterX <= bCenterX) {
              const reqX = a.x + a.w + minHorizontalGap;
              if (b.x < reqX) b.x = reqX;
            } else {
              const reqX = b.x + b.w + minHorizontalGap;
              if (a.x < reqX) a.x = reqX;
            }
          }
        }
      }
    }
  }

  const out = new Map<string, PositionedNode>();
  list.forEach((p) => out.set(p.id, p));
  return out;
}

export interface Point {
  x: number;
  y: number;
}

/** Generate an SVG path for an orthogonal polyline with rounded fillet corners. */
export function generateRoundedOrthogonalPath(points: Point[], radius = 8): string {
  if (points.length < 2) return "";
  if (points.length === 2) {
    return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`;
  }
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const pPrev = points[i - 1];
    const pCurr = points[i];
    const pNext = points[i + 1];

    const v1 = { x: pCurr.x - pPrev.x, y: pCurr.y - pPrev.y };
    const v2 = { x: pNext.x - pCurr.x, y: pNext.y - pCurr.y };
    const len1 = Math.hypot(v1.x, v1.y);
    const len2 = Math.hypot(v2.x, v2.y);
    if (len1 < 1 || len2 < 1) {
      d += ` L ${pCurr.x} ${pCurr.y}`;
      continue;
    }
    const r = Math.min(radius, len1 / 2, len2 / 2);
    const bx = pCurr.x - (v1.x / len1) * r;
    const by = pCurr.y - (v1.y / len1) * r;
    const ax = pCurr.x + (v2.x / len2) * r;
    const ay = pCurr.y + (v2.y / len2) * r;

    d += ` L ${bx} ${by} Q ${pCurr.x} ${pCurr.y} ${ax} ${ay}`;
  }
  d += ` L ${points[points.length - 1].x} ${points[points.length - 1].y}`;
  return d;
}

export interface RoutedEdgePath {
  path: string;
  midX: number;
  midY: number;
}

/**
 * Route an edge cleanly between node boundaries.
 * Handles residual gutters, cross-column corridors, and vertical flows with rounded fillets.
 */
export function routeEdgeOrthogonal(
  src: PositionedNode,
  tgt: PositionedNode,
  edge: NormalizedEdge,
  canvasW: number = FIGURE_CANVAS_W
): RoutedEdgePath {
  const srcW = src.w || NODE_W_DEFAULT;
  const srcH = src.h || NODE_H_DEFAULT;
  const tgtW = tgt.w || NODE_W_DEFAULT;
  const tgtH = tgt.h || NODE_H_DEFAULT;

  const srcCenterX = src.x + srcW / 2;
  const srcCenterY = src.y + srcH / 2;
  const tgtCenterX = tgt.x + tgtW / 2;
  const tgtCenterY = tgt.y + tgtH / 2;

  const dx = tgtCenterX - srcCenterX;
  const dy = tgtCenterY - srcCenterY;

  const isResidual =
    (edge.style as any)?.name === "residual" ||
    edge.type === "residual_skip" ||
    ((edge.raw_style as any)?.original || "").toLowerCase().includes("skip") ||
    ((edge.raw_style as any)?.original || "").toLowerCase().includes("residual");

  // 1. Residual / Skip connection: Route via outer gutter lane
  if (isResidual) {
    const isRightHalf = srcCenterX >= canvasW * 0.45;
    const gutterX = isRightHalf
      ? Math.max(src.x + srcW, tgt.x + tgtW) + 40
      : Math.min(src.x, tgt.x) - 40;
    const sx = isRightHalf ? src.x + srcW : src.x;
    const sy = srcCenterY;
    const tx = isRightHalf ? tgt.x + tgtW : tgt.x;
    const ty = tgtCenterY;

    const points: Point[] = [
      { x: sx, y: sy },
      { x: gutterX, y: sy },
      { x: gutterX, y: ty },
      { x: tx, y: ty },
    ];
    const path = generateRoundedOrthogonalPath(points, 8);
    return { path, midX: gutterX, midY: (sy + ty) / 2 };
  }

  // 2. Cross-Column Connection (e.g. Encoder to Decoder cross-attention)
  if (Math.abs(dx) > 120) {
    if (dx > 0) {
      const sx = src.x + srcW;
      const sy = srcCenterY;
      const tx = tgt.x;
      const ty = tgtCenterY;
      const midX = (sx + tx) / 2;
      const points: Point[] = [
        { x: sx, y: sy },
        { x: midX, y: sy },
        { x: midX, y: ty },
        { x: tx, y: ty },
      ];
      const path = generateRoundedOrthogonalPath(points, 8);
      return { path, midX, midY: (sy + ty) / 2 };
    } else {
      const sx = src.x;
      const sy = srcCenterY;
      const tx = tgt.x + tgtW;
      const ty = tgtCenterY;
      const midX = (sx + tx) / 2;
      const points: Point[] = [
        { x: sx, y: sy },
        { x: midX, y: sy },
        { x: midX, y: ty },
        { x: tx, y: ty },
      ];
      const path = generateRoundedOrthogonalPath(points, 8);
      return { path, midX, midY: (sy + ty) / 2 };
    }
  }

  // 3. Vertical flow (within the same column)
  if (Math.abs(dy) >= Math.abs(dx)) {
    if (dy >= 0) {
      // Downward flow: Source bottom -> Target top
      const sx = srcCenterX;
      const sy = src.y + srcH;
      const tx = tgtCenterX;
      const ty = tgt.y;

      if (Math.abs(dx) < 8) {
        return { path: `M ${sx} ${sy} L ${tx} ${ty}`, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
      }
      const vOffset = Math.max(16, (ty - sy) * 0.45);
      const path = `M ${sx} ${sy} C ${sx} ${sy + vOffset}, ${tx} ${ty - vOffset}, ${tx} ${ty}`;
      return { path, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
    } else {
      // Upward flow: Source top -> Target bottom
      const sx = srcCenterX;
      const sy = src.y;
      const tx = tgtCenterX;
      const ty = tgt.y + tgtH;

      if (Math.abs(dx) < 8) {
        return { path: `M ${sx} ${sy} L ${tx} ${ty}`, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
      }
      const vOffset = Math.max(16, (sy - ty) * 0.45);
      const path = `M ${sx} ${sy} C ${sx} ${sy - vOffset}, ${tx} ${ty + vOffset}, ${tx} ${ty}`;
      return { path, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
    }
  }

  // 4. Horizontal flow (e.g. side-by-side nodes like Positional Encoding -> Embedding)
  if (dx >= 0) {
    const sx = src.x + srcW;
    const sy = srcCenterY;
    const tx = tgt.x;
    const ty = tgtCenterY;
    if (Math.abs(dy) < 8) {
      return { path: `M ${sx} ${sy} L ${tx} ${ty}`, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
    }
    const hOffset = Math.max(16, (tx - sx) * 0.4);
    const path = `M ${sx} ${sy} C ${sx + hOffset} ${sy}, ${tx - hOffset} ${ty}, ${tx} ${ty}`;
    return { path, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
  } else {
    const sx = src.x;
    const sy = srcCenterY;
    const tx = tgt.x + tgtW;
    const ty = tgtCenterY;
    if (Math.abs(dy) < 8) {
      return { path: `M ${sx} ${sy} L ${tx} ${ty}`, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
    }
    const hOffset = Math.max(16, (sx - tx) * 0.4);
    const path = `M ${sx} ${sy} C ${sx - hOffset} ${sy}, ${tx + hOffset} ${ty}, ${tx} ${ty}`;
    return { path, midX: (sx + tx) / 2, midY: (sy + ty) / 2 };
  }
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

  let marker = isActive ? "url(#arrow-active)" : "url(#arrow)";

  if (name === "residual" || edge.type === "residual_skip") {
    stroke = isActive ? "#ef4444" : "#f87171";
    dasharray = "6 4";
    if (!isActive) marker = "url(#arrow-residual)";
  } else if (name === "dashed" || edge.type === "cross_attention") {
    stroke = stroke ?? (isActive ? "#3b82f6" : "#94a3b8");
    dasharray = dasharray ?? "5 3";
    if (!isActive) marker = "url(#arrow-cross)";
  } else if (name === "beam") {
    stroke = stroke ?? (isActive ? "#8b5cf6" : "#a78bfa");
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
    marker,
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
  groups: GraphGroup[] | undefined,
  positionedNodes?: Map<string, PositionedNode>,
  nodes?: NormalizedNode[]
): PositionedGroup[] {
  if (!groups || groups.length === 0) return [];
  const out: PositionedGroup[] = [];

  for (const g of groups) {
    // 1. Try to compute bounds from member nodes if available
    let memberBounds: { minX: number; minY: number; maxX: number; maxY: number } | null = null;

    if (positionedNodes && positionedNodes.size > 0) {
      const memberPos: PositionedNode[] = [];

      if (nodes && nodes.length > 0) {
        for (const n of nodes) {
          if (n.group_id === g.id && positionedNodes.has(n.id)) {
            memberPos.push(positionedNodes.get(n.id)!);
          }
        }
      }

      if (memberPos.length === 0 && g.bbox) {
        const rawGx = g.bbox.x * FIGURE_CANVAS_W;
        const rawGy = g.bbox.y * FIGURE_CANVAS_H;
        const rawGw = g.bbox.w * FIGURE_CANVAS_W;
        const rawGh = g.bbox.h * FIGURE_CANVAS_H;

        positionedNodes.forEach((p) => {
          const centerX = p.x + p.w / 2;
          const centerY = p.y + p.h / 2;
          if (
            centerX >= rawGx - 20 &&
            centerX <= rawGx + rawGw + 20 &&
            centerY >= rawGy - 20 &&
            centerY <= rawGy + rawGh + 20
          ) {
            memberPos.push(p);
          }
        });
      }

      if (memberPos.length > 0) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const p of memberPos) {
          minX = Math.min(minX, p.x);
          minY = Math.min(minY, p.y);
          maxX = Math.max(maxX, p.x + p.w);
          maxY = Math.max(maxY, p.y + p.h);
        }
        memberBounds = { minX, minY, maxX, maxY };
      }
    }

    if (memberBounds) {
      const PAD_X = 24;
      const PAD_TOP = 32;
      const PAD_BOTTOM = 20;
      out.push({
        id: g.id,
        label: g.label || g.id,
        x: memberBounds.minX - PAD_X,
        y: memberBounds.minY - PAD_TOP,
        w: (memberBounds.maxX - memberBounds.minX) + 2 * PAD_X,
        h: (memberBounds.maxY - memberBounds.minY) + PAD_TOP + PAD_BOTTOM,
        color: g.color || "#94a3b8",
      });
      continue;
    }

    // 2. Fallback to raw bbox
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

export interface ElkLayoutResult {
  nodes: Map<string, PositionedNode>;
  edges: Map<string, RoutedEdgePath>;
  groups: PositionedGroup[];
}

/** Run ELK layered layout on the graph, returning positioned nodes, routed edges, and groups. */
export async function computeElkLayout(
  nodes: NormalizedNode[],
  edges: NormalizedEdge[],
  groups?: GraphGroup[]
): Promise<ElkLayoutResult> {
  const elk = await getElk();
  if (!elk || nodes.length === 0) {
    return { nodes: new Map(), edges: new Map(), groups: [] };
  }

  // 1. Standardized, professional block dimensions matching D2 specifications
  const nodeDims = new Map<string, { w: number; h: number }>();
  nodes.forEach((n) => {
    const labelLen = (n.label || "").length;
    const w = Math.max(180, Math.min(240, Math.round(labelLen * 8.5 + 44)));
    const h = 48;
    nodeDims.set(n.id, { w, h });
  });

  // 2. Map nodes to groups if defined
  const nodeToGroup = new Map<string, string>();
  const groupDefs = new Map<string, GraphGroup>();
  if (Array.isArray(groups)) {
    groups.forEach((g) => {
      groupDefs.set(g.id, g);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const memberIds = (g as any).node_ids || (g as any).nodes || [];
      if (Array.isArray(memberIds)) {
        memberIds.forEach((id: string) => nodeToGroup.set(id, g.id));
      }
    });
  }
  nodes.forEach((n) => {
    if (n.group_id && !nodeToGroup.has(n.id)) {
      nodeToGroup.set(n.id, n.group_id);
    }
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const groupBuckets = new Map<string, any[]>();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const groupEdgeBuckets = new Map<string, any[]>();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const topLevelChildren: any[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const topLevelEdges: any[] = [];

  // 3. Place nodes into groups or top-level children
  nodes.forEach((n) => {
    const dims = nodeDims.get(n.id) || { w: NODE_W_DEFAULT, h: NODE_H_DEFAULT };
    const elkNode = {
      id: n.id,
      width: dims.w,
      height: dims.h,
    };
    const gId = nodeToGroup.get(n.id);
    if (gId) {
      if (!groupBuckets.has(gId)) groupBuckets.set(gId, []);
      groupBuckets.get(gId)!.push(elkNode);
    } else {
      topLevelChildren.push(elkNode);
    }
  });

  // 4. Lowest Common Ancestor (LCA) edge scoping
  edges.forEach((e, i) => {
    const srcG = nodeToGroup.get(e.source);
    const tgtG = nodeToGroup.get(e.target);
    const isIntraGroup = srcG && tgtG && srcG === tgtG;
    const elkEdge = {
      id: `e${i}`,
      sources: [e.source],
      targets: [e.target],
    };

    if (isIntraGroup && srcG) {
      if (!groupEdgeBuckets.has(srcG)) groupEdgeBuckets.set(srcG, []);
      groupEdgeBuckets.get(srcG)!.push(elkEdge);
    } else {
      topLevelEdges.push(elkEdge);
    }
  });

  // 5. Build container nodes
  groupBuckets.forEach((members, gId) => {
    topLevelChildren.push({
      id: gId,
      layoutOptions: {
        "elk.padding": "[top=40,left=24,bottom=24,right=24]",
        "elk.spacing.nodeNode": "24",
        "elk.layered.spacing.nodeNodeBetweenLayers": "48",
        "elk.direction": "DOWN",
      },
      children: members,
      edges: groupEdgeBuckets.get(gId) || [],
    });
  });

  const graph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.hierarchyHandling": "INCLUDE_CHILDREN",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.spacing.nodeNodeBetweenLayers": "56",
      "elk.spacing.nodeNode": "36",
      "elk.spacing.edgeNode": "24",
      "elk.spacing.edgeEdge": "18",
      "elk.layered.spacing.edgeNodeBetweenLayers": "24",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
    },
    children: topLevelChildren,
    edges: topLevelEdges,
  };

  try {
    const result = await elk.layout(graph);
    const containerOffsets = new Map<string, { x: number; y: number }>();
    const positioned = new Map<string, PositionedNode>();
    const positionedGroups: PositionedGroup[] = [];

    let minGlobalX = Infinity, maxGlobalX = -Infinity;
    let minGlobalY = Infinity, maxGlobalY = -Infinity;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function traverse(node: any, parentX = 0, parentY = 0) {
      const absX = parentX + (node.x ?? 0);
      const absY = parentY + (node.y ?? 0);
      containerOffsets.set(node.id, { x: absX, y: absY });

      if (groupBuckets.has(node.id) || groupDefs.has(node.id)) {
        const gDef = groupDefs.get(node.id);
        const w = node.width ?? 200;
        const h = node.height ?? 100;
        positionedGroups.push({
          id: node.id,
          label: gDef?.label || node.id,
          x: absX,
          y: absY,
          w,
          h,
          color: gDef?.color || "#94a3b8",
        });
        minGlobalX = Math.min(minGlobalX, absX);
        maxGlobalX = Math.max(maxGlobalX, absX + w);
        minGlobalY = Math.min(minGlobalY, absY);
        maxGlobalY = Math.max(maxGlobalY, absY + h);
      } else if (node.id !== "root") {
        const dims = nodeDims.get(node.id) ?? { w: NODE_W_DEFAULT, h: NODE_H_DEFAULT };
        positioned.set(node.id, {
          id: node.id,
          x: absX,
          y: absY,
          w: dims.w,
          h: dims.h,
          layer: 0,
          index: 0,
          source: "auto",
        });
        minGlobalX = Math.min(minGlobalX, absX);
        maxGlobalX = Math.max(maxGlobalX, absX + dims.w);
        minGlobalY = Math.min(minGlobalY, absY);
        maxGlobalY = Math.max(maxGlobalY, absY + dims.h);
      }

      if (Array.isArray(node.children)) {
        for (const child of node.children) {
          traverse(child, absX, absY);
        }
      }
    }

    traverse(result, 0, 0);

    const contentW = Number.isFinite(maxGlobalX - minGlobalX) ? maxGlobalX - minGlobalX : 500;
    const offsetX = PADDING + Math.max(0, (FIGURE_CANVAS_W - contentW) / 2) - (Number.isFinite(minGlobalX) ? minGlobalX : 0);
    const offsetY = PADDING + 20 - (Number.isFinite(minGlobalY) ? minGlobalY : 0);

    positioned.forEach((p) => {
      p.x += offsetX;
      p.y += offsetY;
    });

    positionedGroups.forEach((g) => {
      g.x += offsetX;
      g.y += offsetY;
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function collectAllEdges(n: any, currentContainerId = "root"): any[] {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const list: any[] = [];
      const thisContainerId = (n.id && n.id !== "root") ? n.id : currentContainerId;
      if (Array.isArray(n.edges)) {
        for (const e of n.edges) {
          e._owningContainer = thisContainerId;
          list.push(e);
        }
      }
      if (Array.isArray(n.children)) {
        for (const child of n.children) {
          list.push(...collectAllEdges(child, child.id));
        }
      }
      return list;
    }

    const allElkEdges = collectAllEdges(result);
    const edgeRoutes = new Map<string, RoutedEdgePath>();

    for (const e of allElkEdges) {
      const owningContainer = e._owningContainer || e.container || "root";
      const offset = containerOffsets.get(owningContainer) || { x: 0, y: 0 };
      const totalOffsetX = offset.x + offsetX;
      const totalOffsetY = offset.y + offsetY;

      const srcId = e.sources?.[0];
      const tgtId = e.targets?.[0];
      if (!srcId || !tgtId) continue;

      for (const section of (e.sections || [])) {
        if (!section.startPoint || !section.endPoint) continue;
        const points: Point[] = [
          { x: section.startPoint.x + totalOffsetX, y: section.startPoint.y + totalOffsetY },
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          ...(section.bendPoints || []).map((b: any) => ({
            x: b.x + totalOffsetX,
            y: b.y + totalOffsetY,
          })),
          { x: section.endPoint.x + totalOffsetX, y: section.endPoint.y + totalOffsetY },
        ];

        const path = generateRoundedOrthogonalPath(points, 8);

        let longestIdx = 0;
        let maxLen = -1;
        for (let i = 0; i < points.length - 1; i++) {
          const d = Math.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y);
          if (d > maxLen) {
            maxLen = d;
            longestIdx = i;
          }
        }
        const pA = points[longestIdx];
        const pB = points[longestIdx + 1] || pA;
        const midX = (pA.x + pB.x) / 2;
        const midY = (pA.y + pB.y) / 2;

        const routed = { path, midX, midY };
        edgeRoutes.set(`${srcId}->${tgtId}`, routed);
        if (e.id) edgeRoutes.set(e.id, routed);
      }
    }

    return { nodes: positioned, edges: edgeRoutes, groups: positionedGroups };
  } catch {
    return { nodes: new Map(), edges: new Map(), groups: [] };
  }
}
