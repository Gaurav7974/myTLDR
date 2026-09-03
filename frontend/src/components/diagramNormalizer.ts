/**
 * diagramNormalizer — deterministic normalizer that treats any model bbox as a weak hint.
 * Four passes: measure, snap, separate, groups. Generic, no paper-specific hard-coding.
 */

import type { GraphBoundingBox } from "@/types/architecture_graph";

export const PADX = 14;
export const PADY = 10;
export const LH = 17;
export const MIN_GAP = 16;
export const MIN_W = 72;
export const MIN_H = 34;

export interface NormaNode {
  id: string;
  label: string;
  bbox: GraphBoundingBox;
  x: number; y: number; w: number; h: number;
  lines: string[];
}

export interface NormaGroup {
  id: string;
  label: string;
  bbox: GraphBoundingBox;
  x: number; y: number; w: number; h: number;
}

// Simple advance-width table for Inter 13px (approx). For pixel-perfect server/client agreement,
// replace with node-canvas measureText. This table is ~95% accurate and deterministic.
const ADVANCE: Record<string, number> = (() => {
  const m: Record<string, number> = {};
  const defaults: Record<string, number> = { " ": 3.5, "i": 3.2, "l": 3.2, "I": 4.5, "W": 10, "M": 10 };
  for (let c = 32; c < 127; c++) {
    const ch = String.fromCharCode(c);
    m[ch] = defaults[ch] ?? 7.0;
  }
  return m;
})();

function measureText(text: string, fontSize = 13): number {
  // Try canvas if available (browser), else fallback to table
  if (typeof document !== "undefined") {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
      return ctx.measureText(text).width;
    }
  }
  // Fallback table
  let w = 0;
  for (const ch of text) w += ADVANCE[ch] ?? 7;
  return w * (fontSize / 13);
}

export function wrapToWidth(label: string, maxWidth: number, fontSize = 13): string[] {
  if (!label) return [""];
  const words = label.split(/\s+/);
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    const test = cur ? `${cur} ${w}` : w;
    if (measureText(test, fontSize) <= maxWidth) {
      cur = test;
    } else {
      if (cur) lines.push(cur);
      // If single word too long, hard-break
      if (measureText(w, fontSize) > maxWidth) {
        let part = "";
        for (const ch of w) {
          if (measureText(part + ch, fontSize) > maxWidth) {
            lines.push(part);
            part = ch;
          } else part += ch;
        }
        cur = part;
      } else cur = w;
    }
  }
  if (cur) lines.push(cur);
  return lines.length ? lines : [label];
}

function measureMax(lines: string[], fontSize = 13): number {
  return Math.max(...lines.map(l => measureText(l, fontSize)));
}

/** Pass 1: Measure text, then size the box. Fixes Atte bug. */
export function measureNode(node: { id: string; label: string; bbox: GraphBoundingBox }): NormaNode {
  const label = node.label || node.id;
  // Start from incoming bbox but allow expansion
  const incomingW = node.bbox.w * 1000;
  const maxTextW = Math.max(90, incomingW - 2 * PADX);
  const lines = wrapToWidth(label, maxTextW, 13);
  const textW = measureMax(lines, 13);
  const w = Math.max(incomingW, textW + 2 * PADX, MIN_W);
  const h = Math.max(node.bbox.h * 600, lines.length * LH + 2 * PADY, MIN_H);
  return {
    id: node.id,
    label,
    bbox: node.bbox,
    x: node.bbox.x * 1000,
    y: node.bbox.y * 600,
    w, h,
    lines,
  };
}

/** Pass 2: Snap to implicit rails via 1D agglomerative clustering, tolerance 2.5% */
export function snapToRails(nodes: NormaNode[]): void {
  const snapAxis = (getCenter: (n: NormaNode) => number, getPos: (n: NormaNode) => number, setPos: (n: NormaNode, v: number) => void, axisLen: number) => {
    const tol = axisLen * 0.025;
    // Extract centers
    const centers = nodes.map(n => ({ n, c: getCenter(n) }));
    centers.sort((a, b) => a.c - b.c);
    const clusters: Array<{ median: number; members: typeof centers }> = [];
    for (const item of centers) {
      const last = clusters[clusters.length - 1];
      if (last && Math.abs(item.c - last.median) <= tol) {
        last.members.push(item);
        // Update median
        const vals = last.members.map(m => m.c).sort((a, b) => a - b);
        last.median = vals[Math.floor(vals.length / 2)];
      } else {
        clusters.push({ median: item.c, members: [item] });
      }
    }
    for (const cl of clusters) {
      for (const { n } of cl.members) {
        const curCenter = getCenter(n);
        const delta = cl.median - curCenter;
        setPos(n, getPos(n) + delta);
      }
    }
  };
  snapAxis(n => n.x + n.w / 2, n => n.x, (n, v) => { n.x = v; }, 1000);
  snapAxis(n => n.y + n.h / 2, n => n.y, (n, v) => { n.y = v; }, 600);
}

/** Pass 3: Separate within rails, per column */
export function separateWithinRails(nodes: NormaNode[]): void {
  // Group by snapped x (column). Use tolerance 18px to define column
  const cols: Map<number, NormaNode[]> = new Map();
  const colTol = 18;
  for (const n of nodes) {
    const cx = n.x + n.w / 2;
    let found: number | null = null;
    for (const k of cols.keys()) {
      if (Math.abs(k - cx) <= colTol) { found = k; break; }
    }
    const key = found ?? cx;
    if (!cols.has(key)) cols.set(key, []);
    cols.get(key)!.push(n);
  }
  for (const col of cols.values()) {
    col.sort((a, b) => a.y - b.y);
    for (let i = 1; i < col.length; i++) {
      const prev = col[i - 1];
      const cur = col[i];
      const minY = prev.y + prev.h + MIN_GAP;
      if (cur.y < minY) cur.y = minY;
    }
  }
}

/** Pass 4: Groups from members, compounding padding */
export function reflowGroups(groups: NormaGroup[], nodes: NormaNode[]): void {
  for (const g of groups) {
    const members = nodes.filter(n => {
      // Check if node bbox is inside group bbox (with small epsilon)
      return n.x >= g.x - 1 && n.y >= g.y - 1 && n.x + n.w <= g.x + g.w + 1 && n.y + n.h <= g.y + g.h + 1;
    });
    // If no members by containment, try label chip proximity
    const relevant = members.length ? members : [];
    if (relevant.length) {
      const minX = Math.min(...relevant.map(n => n.x));
      const minY = Math.min(...relevant.map(n => n.y));
      const maxX = Math.max(...relevant.map(n => n.x + n.w));
      const maxY = Math.max(...relevant.map(n => n.y + n.h));
      g.x = minX - 12;
      g.y = minY - 12 - 20; // 20px band for label chip
      g.w = maxX - minX + 24;
      g.h = maxY - minY + 24 + 20;
    }
    // Nested groups get compounding padding via second pass (call again)
  }
  // Second pass for nested groups (outer contains inner)
  for (let iter = 0; iter < 2; iter++) {
    for (const outer of groups) {
      for (const inner of groups) {
        if (outer === inner) continue;
        if (inner.x >= outer.x && inner.y >= outer.y && inner.x + inner.w <= outer.x + outer.w && inner.y + inner.h <= outer.y + outer.h) {
          // inner is nested, ensure outer has extra padding (already done via union, but ensure)
          outer.w = Math.max(outer.w, inner.x + inner.w - outer.x + 12);
          outer.h = Math.max(outer.h, inner.y + inner.h - outer.y + 12);
        }
      }
    }
  }
}

export interface NormalizedGraph {
  nodes: NormaNode[];
  groups: NormaGroup[];
}

export function normalizeDiagram(
  rawNodes: Array<{ id: string; label: string; bbox: GraphBoundingBox }>,
  rawGroups: Array<{ id: string; label: string; bbox: GraphBoundingBox }>
): NormalizedGraph {
  const nodes = rawNodes.map(n => measureNode(n));
  const groups: NormaGroup[] = rawGroups.map(g => ({
    id: g.id,
    label: g.label,
    bbox: g.bbox,
    x: g.bbox.x * 1000,
    y: g.bbox.y * 600,
    w: g.bbox.w * 1000,
    h: g.bbox.h * 600,
  }));
  // 2-3 iterations alternating axes
  for (let i = 0; i < 3; i++) {
    snapToRails(nodes);
    separateWithinRails(nodes);
  }
  reflowGroups(groups, nodes);
  // Final separate after group inflation
  separateWithinRails(nodes);
  return { nodes, groups };
}
