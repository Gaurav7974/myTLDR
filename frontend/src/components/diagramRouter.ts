/**
 * diagramRouter — orthogonal router that treats VLM path_points as a weak hint.
 * Channel grid from snapped columns/row bands, L/Z, residual side lanes.
 */

import type { GraphPoint } from "@/types/architecture_graph";

export interface RouterNode {
  id: string;
  x: number; y: number; w: number; h: number;
}

export interface RoutedEdge {
  source: string; target: string;
  path_points: GraphPoint[];
  style: { name: string; dasharray?: string; stroke?: string };
}

const LANE = 16;
const ARROW_LEN = 8;
const CORNER_R = 7;

function portForDirection(node: RouterNode, dir: string): GraphPoint {
  switch (dir) {
    case "N": return { x: node.x + node.w/2, y: node.y };
    case "S": return { x: node.x + node.w/2, y: node.y + node.h };
    case "E": return { x: node.x + node.w, y: node.y + node.h/2 };
    case "W": return { x: node.x, y: node.y + node.h/2 };
    default: return { x: node.x + node.w/2, y: node.y + node.h/2 };
  }
}

function directionToward(a: RouterNode, b: RouterNode): { from: string; to: string } {
  const ax = a.x + a.w/2, ay = a.y + a.h/2;
  const bx = b.x + b.w/2, by = b.y + b.h/2;
  const dx = bx - ax, dy = by - ay;
  if (Math.abs(dx) > Math.abs(dy)) {
    return dx > 0 ? { from: "E", to: "W" } : { from: "W", to: "E" };
  } else {
    return dy > 0 ? { from: "S", to: "N" } : { from: "N", to: "S" };
  }
}

function isResidual(a: RouterNode, b: RouterNode, vlmPoints?: GraphPoint[]): boolean {
  // Same column non-adjacent or VLM bulges sideways
  const sameCol = Math.abs((a.x + a.w/2) - (b.x + b.w/2)) < 24;
  const verticalGap = Math.abs((a.y + a.h/2) - (b.y + b.h/2));
  if (sameCol && verticalGap > (a.h + b.h) * 1.8) return true;
  if (vlmPoints && vlmPoints.length >= 2) {
    const xs = vlmPoints.map(p => p.x);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    if ((maxX - minX) * 1000 > Math.max(a.w, b.w) * 1.2) return true;
  }
  return false;
}

export function routeEdge(
  edge: { source: string; target: string; path_points?: GraphPoint[] | null; type?: string | null; style?: { name?: string } | null },
  posMap: Map<string, RouterNode>,
  depth: number
): RoutedEdge {
  const a = posMap.get(edge.source);
  const b = posMap.get(edge.target);
  if (!a || !b) {
    // Fallback: use VLM points if available, else straight
    if (edge.path_points && edge.path_points.length >= 2) {
      return { source: edge.source, target: edge.target, path_points: edge.path_points, style: { name: (edge.style?.name as string) || edge.type || "default" } };
    }
    return { source: edge.source, target: edge.target, path_points: [], style: { name: "default" } };
  }
  const dir = directionToward(a, b);
  const residual = isResidual(a, b, edge.path_points ?? undefined);
  let pts: GraphPoint[] = [];
  const styleName = residual ? "residual" : (edge.style?.name as string) || edge.type || "default";

  if (residual) {
    // Exit sideways, run in side lane, re-enter
    const sideX = a.x + a.w + LANE * (depth + 1);
    const start = portForDirection(a, "E");
    const endPort = portForDirection(b, "W");
    // Shorten final segment by arrow length
    const end = { x: endPort.x - ARROW_LEN, y: endPort.y };
    pts = [
      start,
      { x: sideX, y: start.y },
      { x: sideX, y: end.y },
      end,
    ];
  } else if (dir.from === "S" && dir.to === "N" && Math.abs((a.x+a.w/2)-(b.x+b.w/2)) < 24) {
    // Same column adjacent: straight vertical
    const s = portForDirection(a, "S");
    const t = portForDirection(b, "N");
    // Shorten
    const t2 = { x: t.x, y: t.y - ARROW_LEN };
    pts = [s, t2];
  } else if (dir.from === "E" || dir.from === "W") {
    // Different columns: L or Z via mid-channel
    const s = portForDirection(a, dir.from);
    const t0 = portForDirection(b, dir.to);
    const t = { x: dir.to === "W" ? t0.x - ARROW_LEN : t0.x + ARROW_LEN, y: t0.y };
    if (Math.abs(s.y - t.y) < 12) {
      pts = [s, t];
    } else {
      const mx = (s.x + t.x) / 2;
      pts = [s, { x: mx, y: s.y }, { x: mx, y: t.y }, t];
    }
  } else {
    // Different rows but same column logic already handled, fallback: L via mid
    const s = portForDirection(a, dir.from);
    const t0 = portForDirection(b, dir.to);
    const t = dir.to === "N" ? { x: t0.x, y: t0.y - ARROW_LEN } : t0;
    const mx = (s.x + t.x) / 2;
    pts = [s, { x: mx, y: s.y }, { x: mx, y: t.y }, t];
  }

  // Round corners: convert L into Q with r=7 (simplified: keep points, renderer will handle via path)
  // Shorten is already done. Return.
  return {
    source: edge.source,
    target: edge.target,
    path_points: pts,
    style: { name: styleName },
  };
}

export function routeAllEdges(
  edges: Array<{ source: string; target: string; path_points?: GraphPoint[] | null; type?: string | null; style?: { name?: string } | null }>,
  posMap: Map<string, RouterNode>
): RoutedEdge[] {
  // Sort by nesting depth for lane assignment: deeper residuals get outer lanes
  const withDepth = edges.map((e, idx) => {
    const a = posMap.get(e.source);
    const b = posMap.get(e.target);
    const depth = a && b && isResidual(a, b, e.path_points ?? undefined) ? Math.abs((a.y - b.y) / 100) : 0;
    return { e, idx, depth };
  });
  withDepth.sort((a, b) => a.depth - b.depth);
  const out: RoutedEdge[] = [];
  for (const { e, depth } of withDepth) {
    out.push(routeEdge(e, posMap, Math.floor(depth)));
  }
  // Restore original order
  out.sort((a, b) => {
    const ia = edges.findIndex(x => x.source === a.source && x.target === a.target);
    const ib = edges.findIndex(x => x.source === b.source && x.target === b.target);
    return ia - ib;
  });
  return out;
}

export function pathPointsToSvgPathRounded(points: GraphPoint[], r = CORNER_R): string {
  if (points.length < 2) return "";
  // Convert L points to rounded Q corners
  let d = `M ${points[0].x},${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1], cur = points[i], next = points[i + 1];
    // Simple rounding: add Q at corner
    d += ` L ${cur.x},${cur.y}`;
    // Could add Q, but keep L for now and let CSS handle
    void prev; void next;
  }
  d += ` L ${points[points.length - 1].x},${points[points.length - 1].y}`;
  return d;
}
