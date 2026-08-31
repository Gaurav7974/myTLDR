/**
 * Frontend Layout Engine: Converts canonical ArchitectureIR into an interactive 3D SceneGraph.
 * Decouples graph topology from 3D spatial rendering.
 */

import {
  ArchitectureIR,
  ArchitectureNodeIR,
  SceneGraph3D,
  SceneNode3D,
  SceneEdge3D,
  LODLevel,
  NodeTypeTaxonomy
} from '@/types/architecture_ir';

export const TAXONOMY_COLORS: Record<NodeTypeTaxonomy, string> = {
  input: '#3B82F6',
  output: '#059669',
  embedding: '#3B82F6',
  linear: '#8B5CF6',
  matrix: '#8B5CF6',
  attention: '#F59E0B',
  multi_head_attention: '#F59E0B',
  cross_attention: '#D97706',
  mlp: '#6366F1',
  feed_forward: '#6366F1',
  normalization: '#64748B',
  layer_norm: '#64748B',
  rms_norm: '#64748B',
  residual: '#EF4444',
  convolution: '#0EA5E9',
  conv1d: '#0EA5E9',
  conv2d: '#0EA5E9',
  pooling: '#38BDF8',
  encoder: '#3B82F6',
  decoder: '#8B5CF6',
  transformer_block: '#6366F1',
  ssm_block: '#14B8A6',
  selective_scan: '#14B8A6',
  unet_block: '#EC4899',
  diffusion_step: '#EC4899',
  adapter_lora: '#F43F5E',
  router: '#A855F7',
  expert: '#8B5CF6',
  classifier: '#10B981',
  loss: '#DC2626',
  custom: '#6B7280'
};

export function computeClientSceneGraph(
  ir: ArchitectureIR,
  lod: LODLevel = 2,
  collapsedGroups: Set<string> = new Set()
): SceneGraph3D {
  const nodes = ir.nodes || [];
  const edges = ir.edges || [];

  if (nodes.length === 0) {
    return { nodes_3d: [], edges_3d: [] };
  }

  // Filter nodes according to LOD level or collapsed state
  const visibleNodes = nodes.filter((n) => {
    if (n.parent_id && collapsedGroups.has(n.parent_id)) return false;
    if (lod === 1 && n.parent_id) return false; // In LOD 1, hide nested internals
    return true;
  });

  const numNodes = visibleNodes.length;
  const totalWidth = Math.max(16.0, numNodes * 2.2);
  const startX = -(totalWidth / 2.0);
  const gap = totalWidth / Math.max(1, numNodes - 1);

  const nodePosMap = new Map<string, [number, number, number]>();
  const sceneNodes: SceneNode3D[] = [];

  visibleNodes.forEach((node, idx) => {
    const x = Number((startX + idx * gap).toFixed(2));
    const isAttn = ['attention', 'multi_head_attention', 'cross_attention', 'selective_scan'].includes(node.type);
    const y = isAttn ? 0.35 : 0.0;
    const z = 0.0;
    const pos: [number, number, number] = [x, y, z];
    nodePosMap.set(node.id, pos);

    const { size, grid } = getNodeSlabDimensions(node);
    const color = TAXONOMY_COLORS[node.type] || '#6B7280';

    const subLabels: string[] = [];
    if (node.repetitions && node.repetitions > 1) {
      subLabels.push(`× ${node.repetitions} Layers`);
    }
    if (node.dimensions?.hidden_dim) {
      subLabels.push(`Dim: ${node.dimensions.hidden_dim}`);
    }
    if (node.dimensions?.num_heads) {
      subLabels.push(`${node.dimensions.num_heads} Heads`);
    }

    sceneNodes.push({
      id: `scene_${node.id}`,
      ir_node_id: node.id,
      type: node.type,
      label: node.label,
      position_3d: pos,
      size_3d: size,
      grid_subdivisions: grid,
      color_hex: color,
      lod_level: lod,
      parent_group_id: node.parent_id,
      provenance: node.provenance || { page: 1, section: '3. Method', confidence: 0.95 },
      math_formula: node.math_formula,
      parameters: node.parameters,
      flops: node.flops,
      description: node.description,
      detailed_walkthrough: node.detailed_walkthrough,
      concrete_example: node.concrete_example,
      sub_labels: subLabels
    });
  });

  // Calculate 3D edges and bezier bypass curves
  const sceneEdges: SceneEdge3D[] = [];
  edges.forEach((edge) => {
    const srcPos = nodePosMap.get(edge.source);
    const tgtPos = nodePosMap.get(edge.target);
    if (!srcPos || !tgtPos) return;

    const startPt: [number, number, number] = [srcPos[0], srcPos[1] + 1.15, srcPos[2]];
    const endPt: [number, number, number] = [tgtPos[0], tgtPos[1] + 1.15, tgtPos[2]];

    let controlPts: [number, number, number][] | undefined = undefined;
    let edgeColor = '#3B82F6';

    if (edge.type === 'residual_skip' || edge.type === 'cross_attention') {
      const midX = (startPt[0] + endPt[0]) / 2.0;
      const distX = Math.abs(endPt[0] - startPt[0]);
      const arcHeight = Math.min(4.8, 1.4 + distX * 0.2);
      controlPts = [
        [startPt[0] + distX * 0.15, startPt[1] + arcHeight * 0.8, 0],
        [midX, arcHeight, 0],
        [endPt[0] - distX * 0.15, endPt[1] + arcHeight * 0.8, 0]
      ];
      edgeColor = edge.type === 'residual_skip' ? '#EF4444' : '#D97706';
    }

    sceneEdges.push({
      id: `scene_${edge.id}`,
      source_id: edge.source,
      target_id: edge.target,
      type: edge.type,
      start_pos_3d: startPt,
      end_pos_3d: endPt,
      control_points_3d: controlPts,
      color_hex: edgeColor,
      label: edge.label
    });
  });

  return {
    nodes_3d: sceneNodes,
    edges_3d: sceneEdges,
    camera_target: [0, 0, 0],
    bounds_3d: {
      min_x: startX - 2.5,
      max_x: Math.abs(startX) + 2.5,
      min_y: -2.0,
      max_y: 5.5
    }
  };
}

function getNodeSlabDimensions(
  node: ArchitectureNodeIR
): { size: [number, number, number]; grid: { rows: number; cols: number } } {
  const t = node.type;

  if (t === 'mlp' || t === 'feed_forward' || t === 'unet_block') {
    return { size: [2.0, 2.6, 0.55], grid: { rows: 6, cols: 20 } };
  }
  if (t === 'attention' || t === 'multi_head_attention' || t === 'cross_attention') {
    return { size: [2.0, 2.0, 0.35], grid: { rows: 6, cols: 6 } };
  }
  if (t === 'embedding' || t === 'linear' || t === 'matrix' || t === 'ssm_block' || t === 'selective_scan') {
    return { size: [1.8, 2.2, 0.45], grid: { rows: 6, cols: 12 } };
  }
  if (t === 'normalization' || t === 'layer_norm' || t === 'rms_norm' || t === 'residual') {
    return { size: [1.4, 1.9, 0.35], grid: { rows: 6, cols: 8 } };
  }
  if (t === 'encoder' || t === 'decoder' || t === 'transformer_block') {
    return { size: [2.4, 3.0, 0.6], grid: { rows: 8, cols: 16 } };
  }
  return { size: [1.8, 2.2, 0.4], grid: { rows: 6, cols: 10 } };
}
