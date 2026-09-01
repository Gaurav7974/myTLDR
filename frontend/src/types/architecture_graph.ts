/**
 * Frontend mirror of the backend `ArchitectureGraph` figure-faithful schema.
 *
 * The renderer is permissive: every field except `nodes` / `edges` is optional.
 * When fields are missing, the renderer falls back to a safe default
 * (taxonomy color, rounded rectangle, auto-layout, sidebar without
 * provenance, etc.). The `layoutMode` and `verification` fields are
 * the source of truth for whether a graph is renderable as a
 * figure-faithful reconstruction.
 */

export type ArchitectureNodeType =
  | 'input' | 'output' | 'embedding' | 'attention'
  | 'feedforward' | 'normalization' | 'block'
  | 'router' | 'expert' | 'ssm' | 'gate' | 'cache' | 'other';

export type NodeShape =
  | 'rect' | 'rounded' | 'ellipse' | 'diamond' | 'cylinder'
  | 'parallelogram' | 'trapezoid' | 'custom';

export type EdgeStyleName =
  | 'solid' | 'dashed' | 'dotted' | 'residual'
  | 'beam' | 'highlight' | 'custom' | 'default';

export type GraphEdgeType =
  | 'data_flow' | 'residual_skip' | 'cross_attention'
  | 'gated_modulation' | 'hierarchical_containment'
  | 'recurrent_state' | 'custom';

export type GraphSource = 'text' | 'vision' | 'merged' | 'auto' | 'legacy';
export type LayoutMode = 'auto' | 'figure' | 'legacy';

export interface GraphPoint {
  x: number; // normalized [0, 1]
  y: number;
}

export interface GraphBoundingBox {
  x: number; // normalized [0, 1]
  y: number;
  w: number; // > 0, <= 1
  h: number; // > 0, <= 1
}

export interface GraphProvenance {
  source: 'text' | 'vision' | 'merged';
  paper_title?: string | null;
  page?: number | null;
  section?: string | null;
  figure_id?: string | null;
  figure_page?: number | null;
  citation?: string | null;
  confidence?: number;
}

export interface GraphNodeStyle {
  fill?: string | null;
  border?: string | null;
  text?: string | null;
  shape?: NodeShape;
  raw?: Record<string, unknown> | null;
}

export interface GraphEdgeStyle {
  stroke?: string | null;
  width?: number | null;
  dasharray?: string | null;
  name?: EdgeStyleName;
  raw?: Record<string, unknown> | null;
}

export interface GraphGroup {
  id: string;
  label?: string | null;
  bbox?: GraphBoundingBox | null;
  color?: string | null;
  confidence?: number;
}

export interface ArchitectureGraphNode {
  id: string;
  label: string;
  type: ArchitectureNodeType | string;
  description?: string | null;
  how_it_works?: string | null;
  math_formula?: string | null;
  paper_term?: string | null;
  parameters?: string | null;
  flops?: string | null;
  group_id?: string | null;
  bbox?: GraphBoundingBox | null;
  shape?: NodeShape | null;
  style?: GraphNodeStyle | null;
  confidence?: number;
  provenance?: GraphProvenance | null;
}

export interface ArchitectureGraphEdge {
  source: string;
  target: string;
  label?: string | null;
  type?: GraphEdgeType | null;
  semantic?: string | null;
  path_points?: GraphPoint[] | null;
  source_anchor?: GraphPoint | null;
  target_anchor?: GraphPoint | null;
  style?: GraphEdgeStyle | null;
  raw_style?: Record<string, unknown> | null;
}

export interface GraphVerification {
  valid: boolean;
  confidence: number; // 0..1
  hallucinations?: string[];
  missing_elements?: string[];
  node_verification?: Array<Record<string, unknown>>;
  edge_verification?: Array<Record<string, unknown>>;
  reason?: string | null;
}

export interface ArchitectureGraph {
  nodes: ArchitectureGraphNode[];
  edges: ArchitectureGraphEdge[];
  version?: string | null;
  source?: GraphSource | null;
  figure_id?: string | null;
  figure_page?: number | null;
  confidence?: number | null;
  verification?: GraphVerification | null;
  groups?: GraphGroup[];
  layout_mode?: LayoutMode | null;
  paper_title?: string | null;
  metadata?: Record<string, unknown> | null;
}

/** Public constants mirrored from backend `app.models.graph`. */
export const GRAPH_PAYLOAD_VERSION = '2.0';
export const VISION_CONFIDENCE_THRESHOLD = 0.65;

/** Trust-first gate on the frontend. Mirrors `is_verified_vision_graph` in
 *  `backend/app/services/figure_extraction.py`. */
export function isVerifiedVisionGraph(graph: ArchitectureGraph | null | undefined): boolean {
  if (!graph) return false;
  if (graph.source !== 'vision' && graph.source !== 'merged') return false;
  if (graph.layout_mode !== 'figure') return false;
  const v = graph.verification;
  if (!v || !v.valid) return false;
  if (v.confidence < VISION_CONFIDENCE_THRESHOLD) return false;
  if (v.hallucinations && v.hallucinations.length > 0) return false;
  return true;
}

/** Single source of truth for "may this be drawn as a figure-faithful
 *  reconstruction?". Anything else is downgraded by the caller. */
export function shouldRenderReconstruction(
  graph: ArchitectureGraph | null | undefined
): boolean {
  return isVerifiedVisionGraph(graph);
}

/** True when the renderer should fall back to a BFS auto-layout. */
export function shouldUseAutoLayout(
  graph: ArchitectureGraph | null | undefined
): boolean {
  if (!graph) return false;
  return graph.layout_mode === 'auto' || graph.layout_mode === 'legacy' || !graph.layout_mode;
}
