/**
 * Canonical Architecture Intermediate Representation (ArchitectureIR) Schema
 * The strict contract between AI paper analysis and the 3D WebGL renderer.
 */

export type NodeTypeTaxonomy =
  | 'input'
  | 'output'
  | 'embedding'
  | 'linear'
  | 'matrix'
  | 'attention'
  | 'multi_head_attention'
  | 'cross_attention'
  | 'mlp'
  | 'feed_forward'
  | 'normalization'
  | 'layer_norm'
  | 'rms_norm'
  | 'residual'
  | 'convolution'
  | 'conv1d'
  | 'conv2d'
  | 'pooling'
  | 'encoder'
  | 'decoder'
  | 'transformer_block'
  | 'ssm_block'
  | 'selective_scan'
  | 'unet_block'
  | 'diffusion_step'
  | 'adapter_lora'
  | 'router'
  | 'expert'
  | 'classifier'
  | 'loss'
  | 'custom';

export type EdgeTypeTaxonomy =
  | 'data_flow'
  | 'residual_skip'
  | 'cross_attention'
  | 'hierarchical_containment'
  | 'gated_modulation'
  | 'recurrent_state';

export type LODLevel = 1 | 2 | 3 | 4; // 1=Overview, 2=Architecture Blocks, 3=Internals, 4=Mathematical

export interface NodeProvenance {
  page: number;
  section: string;
  figure_id?: string;
  citation?: string;
  confidence: number;
  is_inferred?: boolean;
}

export interface TensorDimensions {
  input_shape?: string;
  output_shape?: string;
  batch_size?: number;
  sequence_length?: number;
  hidden_dim?: number;
  num_heads?: number;
  expansion_factor?: number;
  kernel_size?: number;
}

export interface ArchitectureNodeIR {
  id: string;
  type: NodeTypeTaxonomy;
  label: string;
  parent_id?: string | null;
  children_ids?: string[];
  dimensions?: TensorDimensions;
  parameters?: string;
  flops?: string;
  repetitions?: number;
  description?: string;
  math_formula?: string;
  detailed_walkthrough?: string;
  concrete_example?: string;
  provenance: NodeProvenance;
  tags?: string[];
}

export interface ArchitectureEdgeIR {
  id: string;
  source: string;
  target: string;
  type: EdgeTypeTaxonomy;
  label?: string;
  tensor_shape?: string;
  provenance?: NodeProvenance;
}

export interface ArchitectureMetadataIR {
  name: string;
  paradigm: 'transformer' | 'ssm' | 'diffusion' | 'cnn_vision' | 'lora' | 'rl_policy' | 'other';
  primary_paper_id?: string;
  author_summary?: string;
}

export interface VisualizationHintsIR {
  recommended_lod?: LODLevel;
  flow_direction?: 'horizontal' | 'vertical' | 'bidirectional';
  highlight_nodes?: string[];
}

export interface ArchitectureIR {
  version: string;
  architecture: ArchitectureMetadataIR;
  nodes: ArchitectureNodeIR[];
  edges: ArchitectureEdgeIR[];
  visualization_hints?: VisualizationHintsIR;
}

export interface SceneNode3D {
  id: string;
  ir_node_id: string;
  type: NodeTypeTaxonomy;
  label: string;
  position_3d: [number, number, number];
  size_3d: [number, number, number];
  grid_subdivisions: { rows: number; cols: number };
  color_hex: string;
  lod_level: LODLevel;
  parent_group_id?: string | null;
  provenance: NodeProvenance;
  math_formula?: string;
  parameters?: string;
  flops?: string;
  description?: string;
  detailed_walkthrough?: string;
  concrete_example?: string;
  sub_labels?: string[];
}

export interface SceneEdge3D {
  id: string;
  source_id: string;
  target_id: string;
  type: EdgeTypeTaxonomy;
  start_pos_3d: [number, number, number];
  end_pos_3d: [number, number, number];
  control_points_3d?: [number, number, number][];
  color_hex: string;
  is_animated?: boolean;
  label?: string;
}

export interface SceneGraph3D {
  nodes_3d: SceneNode3D[];
  edges_3d: SceneEdge3D[];
  camera_target?: [number, number, number];
  bounds_3d?: { min_x: number; max_x: number; min_y: number; max_y: number };
}
