/**
 * figureTransform — uniform figure fit that preserves aspect ratio.
 * Replaces independent axis scaling `x=bbox.x*1000, y=bbox.y*600` which
 * smears tall figures to 1.67:1.
 *
 * cropW/cropH are the figure crop's pixel dimensions (NOT 1000/600).
 * Normalized bbox coords [0,1] are relative to the crop, so we scale
 * uniformly then center in 1000×600.
 */

export interface Transform {
  s: number;
  ox: number;
  oy: number;
  cropW: number;
  cropH: number;
  margin: number;
}

export interface Point {
  x: number;
  y: number;
}

/**
 * Compute uniform transform that fits cropWxH into 1000×600 with margin M.
 * Returns null if cropW/H are zero, NaN, or non-finite.
 */
export function computeFigureTransform(
  cropW: number,
  cropH: number,
  opts?: { M?: number }
): Transform | null {
  const M = opts?.M ?? 24;
  if (
    !Number.isFinite(cropW) ||
    !Number.isFinite(cropH) ||
    !Number.isFinite(M) ||
    cropW <= 0 ||
    cropH <= 0
  ) {
    return null;
  }
  const availW = 1000 - 2 * M;
  const availH = 600 - 2 * M;
  if (availW <= 0 || availH <= 0) return null;
  const s = Math.min(availW / cropW, availH / cropH);
  if (!Number.isFinite(s) || s <= 0) return null;
  const ox = (1000 - cropW * s) / 2;
  const oy = (600 - cropH * s) / 2;
  return { s, ox, oy, cropW, cropH, margin: M };
}

/**
 * Transform a point in crop-pixel space [0,cropW]×[0,cropH] to canvas space [0,1000]×[0,600].
 */
export function toCanvasPixel(
  p: Point,
  t: Transform
): Point {
  return {
    x: t.ox + p.x * t.s,
    y: t.oy + p.y * t.s,
  };
}

/**
 * Transform a normalized point [0,1] within the crop to canvas space.
 * This is the common case for LLM bbox/path_points which are normalized to the figure crop.
 */
export function toCanvasNormalized(
  p: Point,
  t: Transform
): Point {
  return {
    x: t.ox + p.x * t.cropW * t.s,
    y: t.oy + p.y * t.cropH * t.s,
  };
}

/**
 * Compute union bbox of all content in normalized crop space [0,1].
 * Used to derive cropW/cropH when not known, and to ensure nothing is clipped
 * by fitting the union rather than the full crop.
 */
export function getFigureUnionBbox(
  nodes: Array<{ bbox?: { x: number; y: number; w: number; h: number } | null }>,
  groups: Array<{ bbox?: { x: number; y: number; w: number; h: number } | null }>,
  edges: Array<{ path_points?: Array<Point> | null }>
): { x: number; y: number; w: number; h: number } | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let has = false;
  const add = (x: number, y: number, w: number, h: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) return;
    if (w <= 0 || h <= 0) return;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + w);
    maxY = Math.max(maxY, y + h);
    has = true;
  };
  for (const n of nodes) if (n.bbox) add(n.bbox.x, n.bbox.y, n.bbox.w, n.bbox.h);
  for (const g of groups) if (g.bbox) add(g.bbox.x, g.bbox.y, g.bbox.w, g.bbox.h);
  for (const e of edges) for (const p of e.path_points ?? []) {
    if (p && Number.isFinite(p.x) && Number.isFinite(p.y)) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
      has = true;
    }
  }
  if (!has) return null;
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}
