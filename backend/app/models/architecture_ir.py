"""Canonical Architecture Intermediate Representation (ArchitectureIR) Schema.

This is the strict contract between the multimodal/text AI extractor and the 3D renderer.
The AI never creates 3D coordinates or meshes; it produces semantic ArchitectureIR.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# =========================================================================
# 1. COMPONENT TAXONOMY (Controlled Vocabulary with Custom Fallback)
# =========================================================================
NodeTypeTaxonomy = Literal[
    "input",
    "output",
    "embedding",
    "linear",
    "matrix",
    "attention",
    "multi_head_attention",
    "cross_attention",
    "mlp",
    "feed_forward",
    "normalization",
    "layer_norm",
    "rms_norm",
    "residual",
    "convolution",
    "conv1d",
    "conv2d",
    "pooling",
    "encoder",
    "decoder",
    "transformer_block",
    "ssm_block",
    "selective_scan",
    "unet_block",
    "diffusion_step",
    "adapter_lora",
    "router",
    "expert",
    "classifier",
    "loss",
    "custom"  # Graceful degradation for novel research modules
]

EdgeTypeTaxonomy = Literal[
    "data_flow",
    "residual_skip",
    "cross_attention",
    "hierarchical_containment",
    "gated_modulation",
    "recurrent_state"
]


# =========================================================================
# 2. PROVENANCE & SOURCING (Links Visual Nodes Directly to PDF Source)
# =========================================================================
class NodeProvenance(BaseModel):
    page: int = Field(default=1, description="1-indexed PDF page number where this component is introduced/described")
    section: str = Field(default="3. Method", description="Section heading or number (e.g. '3.2 Multi-Head Attention')")
    figure_id: Optional[str] = Field(default=None, description="Figure identifier if extracted from a diagram (e.g. 'Figure 1')")
    citation: Optional[str] = Field(default=None, description="Exact sentence or equation snippet from paper text")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Extraction confidence score (0.0 - 1.0)")
    is_inferred: bool = Field(default=False, description="True if inferred from architecture context rather than explicitly named")


# =========================================================================
# 3. TENSOR DIMENSIONS & SPECIFICATION
# =========================================================================
class TensorDimensions(BaseModel):
    input_shape: Optional[str] = Field(default=None, description="Input tensor dimension string e.g. '[B, L, 512]'")
    output_shape: Optional[str] = Field(default=None, description="Output tensor dimension string e.g. '[B, L, 2048]'")
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_heads: Optional[int] = None
    expansion_factor: Optional[float] = None
    kernel_size: Optional[int] = None


# =========================================================================
# 4. ARCHITECTURE NODES (Hierarchical DAG Elements)
# =========================================================================
class ArchitectureNode(BaseModel):
    id: str = Field(description="Unique node identifier (e.g. 'enc_block_1_mha')")
    type: NodeTypeTaxonomy = Field(default="custom", description="Component taxonomy category")
    label: str = Field(description="Human-readable component title (e.g. 'Multi-Head Attention (12 Heads)')")
    parent_id: Optional[str] = Field(default=None, description="Parent container ID for hierarchical grouping (e.g. 'encoder_block_1')")
    children_ids: List[str] = Field(default_factory=list, description="Sub-component node IDs contained in this block")
    
    # Computational Metrics
    dimensions: TensorDimensions = Field(default_factory=TensorDimensions)
    parameters: Optional[str] = Field(default=None, description="Formula and total params (e.g. '4 × 512 × 512 = 1.05M')")
    flops: Optional[str] = Field(default=None, description="FLOP count estimate (e.g. '6.29 MFLOPs')")
    repetitions: int = Field(default=1, description="Repeated stack multiplier (e.g. 6 or 12 for stacked layers)")
    
    # Explanations & Math
    description: str = Field(default="", description="Concise explanation of the operation")
    math_formula: Optional[str] = Field(default=None, description="LaTeX math formula (e.g. 'Softmax(QK^T / \\sqrt{d_k}) V')")
    detailed_walkthrough: Optional[str] = Field(default=None, description="Step-by-step mathematical derivation")
    concrete_example: Optional[str] = Field(default=None, description="Concrete numerical tensor transformation")
    
    # Sourcing
    provenance: NodeProvenance = Field(default_factory=NodeProvenance)
    tags: List[str] = Field(default_factory=list)


# =========================================================================
# 5. ARCHITECTURE EDGES (Dataflow & Residual Connections)
# =========================================================================
class ArchitectureEdge(BaseModel):
    id: str = Field(description="Unique edge identifier (e.g. 'embed->attn_1')")
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    type: EdgeTypeTaxonomy = Field(default="data_flow", description="Type of data connection")
    label: Optional[str] = Field(default=None, description="Optional connection description")
    tensor_shape: Optional[str] = Field(default=None, description="Shape flowing along this edge (e.g. '[B, L, 512]')")
    provenance: Optional[NodeProvenance] = None


# =========================================================================
# 6. CANONICAL ARCHITECTURE IR (The Master Contract)
# =========================================================================
class ArchitectureMetadata(BaseModel):
    name: str = Field(description="Architecture name (e.g. 'Transformer', 'Mamba', 'Stable Diffusion U-Net')")
    paradigm: str = Field(description="Paradigm: 'transformer', 'ssm', 'diffusion', 'cnn_vision', 'lora', 'rl_policy', 'other'")
    primary_paper_id: Optional[str] = None
    author_summary: Optional[str] = None


class VisualizationHints(BaseModel):
    recommended_lod: int = Field(default=2, ge=1, le=4, description="Recommended default level of detail (1=Macro, 2=Blocks, 3=Internals, 4=Math)")
    flow_direction: Literal["horizontal", "vertical", "bidirectional"] = "horizontal"
    highlight_nodes: List[str] = Field(default_factory=list)


class ArchitectureIR(BaseModel):
    version: str = Field(default="1.0", description="IR schema version")
    architecture: ArchitectureMetadata
    nodes: List[ArchitectureNode]
    edges: List[ArchitectureEdge]
    visualization_hints: VisualizationHints = Field(default_factory=VisualizationHints)


# =========================================================================
# 7. 3D SCENE GRAPH (Produced by Layout Engine for Three.js/R3F)
# =========================================================================
class SceneNode3D(BaseModel):
    id: str
    ir_node_id: str
    type: NodeTypeTaxonomy
    label: str
    position_3d: List[float] = Field(default=[0.0, 0.0, 0.0])  # [x, y, z]
    size_3d: List[float] = Field(default=[1.8, 2.2, 0.45])     # [w, h, d]
    grid_subdivisions: Dict[str, int] = Field(default={"rows": 6, "cols": 12})
    color_hex: str = "#3B82F6"
    lod_level: int = 2
    parent_group_id: Optional[str] = None
    provenance: NodeProvenance
    math_formula: Optional[str] = None
    parameters: Optional[str] = None
    flops: Optional[str] = None
    description: str = ""
    sub_labels: List[str] = Field(default_factory=list)


class SceneEdge3D(BaseModel):
    id: str
    source_id: str
    target_id: str
    type: EdgeTypeTaxonomy
    start_pos_3d: List[float]
    end_pos_3d: List[float]
    control_points_3d: List[List[float]] = Field(default_factory=list)  # For curved bezier highways
    color_hex: str = "#EF4444"
    is_animated: bool = True
    label: Optional[str] = None


class SceneGraph3D(BaseModel):
    nodes_3d: List[SceneNode3D]
    edges_3d: List[SceneEdge3D]
    camera_target: List[float] = Field(default=[0.0, 0.0, 0.0])
    bounds_3d: Dict[str, float] = Field(default={"min_x": -10.0, "max_x": 10.0, "min_y": -2.0, "max_y": 4.0})
