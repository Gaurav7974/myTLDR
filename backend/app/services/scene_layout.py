"""3D Layout Engine: Converts semantic ArchitectureIR into a 3D SceneGraph."""

from typing import Dict, List, Tuple
from app.models.architecture_ir import (
    ArchitectureIR, ArchitectureNode, ArchitectureEdge,
    SceneGraph3D, SceneNode3D, SceneEdge3D
)


class LayoutEngine:
    """Computes deterministic 3D positions, bounding boxes, and bezier paths from ArchitectureIR."""

    CATEGORY_COLORS: Dict[str, str] = {
        "input": "#3B82F6",
        "output": "#059669",
        "embedding": "#3B82F6",
        "linear": "#8B5CF6",
        "matrix": "#8B5CF6",
        "attention": "#F59E0B",
        "multi_head_attention": "#F59E0B",
        "cross_attention": "#D97706",
        "mlp": "#6366F1",
        "feed_forward": "#6366F1",
        "normalization": "#64748B",
        "layer_norm": "#64748B",
        "rms_norm": "#64748B",
        "residual": "#EF4444",
        "convolution": "#0EA5E9",
        "conv1d": "#0EA5E9",
        "conv2d": "#0EA5E9",
        "pooling": "#38BDF8",
        "encoder": "#3B82F6",
        "decoder": "#8B5CF6",
        "transformer_block": "#6366F1",
        "ssm_block": "#14B8A6",
        "selective_scan": "#14B8A6",
        "unet_block": "#EC4899",
        "diffusion_step": "#EC4899",
        "adapter_lora": "#F43F5E",
        "router": "#A855F7",
        "expert": "#8B5CF6",
        "classifier": "#10B981",
        "loss": "#DC2626",
        "custom": "#6B7280"
    }

    def compute_scene_graph(self, ir: ArchitectureIR, target_lod: int = 2) -> SceneGraph3D:
        """Transform canonical ArchitectureIR into an interactive 3D SceneGraph."""
        nodes = ir.nodes
        edges = ir.edges
        num_nodes = len(nodes)

        if num_nodes == 0:
            return SceneGraph3D(nodes_3d=[], edges_3d=[])

        # 1. Compute Horizontal Staging
        total_width = 18.0
        start_x = - (total_width / 2.0)
        gap = total_width / max(1, num_nodes - 1)

        node_pos_map: Dict[str, Tuple[float, float, float]] = {}
        scene_nodes: List[SceneNode3D] = []

        for idx, node in enumerate(nodes):
            x = start_x + (idx * gap)
            # Arch attention or special modules upward
            y = 0.3 if node.type in ["attention", "multi_head_attention", "cross_attention", "selective_scan"] else 0.0
            z = 0.0

            pos = [round(x, 2), y, z]
            node_pos_map[node.id] = (pos[0], pos[1], pos[2])

            # Dimension-aware bounding box sizing
            size_3d, grid = self._calculate_dimensions(node)
            color_hex = self.CATEGORY_COLORS.get(node.type, "#6B7280")

            sub_labels = []
            if node.repetitions > 1:
                sub_labels.append(f"× {node.repetitions} Layers")
            if node.dimensions.hidden_dim:
                sub_labels.append(f"Dim: {node.dimensions.hidden_dim}")
            if node.dimensions.num_heads:
                sub_labels.append(f"{node.dimensions.num_heads} Heads")

            scene_node = SceneNode3D(
                id=f"scene_{node.id}",
                ir_node_id=node.id,
                type=node.type,
                label=node.label,
                position_3d=pos,
                size_3d=size_3d,
                grid_subdivisions=grid,
                color_hex=color_hex,
                lod_level=target_lod,
                parent_group_id=node.parent_id,
                provenance=node.provenance,
                math_formula=node.math_formula,
                parameters=node.parameters,
                flops=node.flops,
                description=node.description,
                sub_labels=sub_labels
            )
            scene_nodes.append(scene_node)

        # 2. Compute 3D Edge Paths and Bezier Highways
        scene_edges: List[SceneEdge3D] = []
        for edge in edges:
            src_pos = node_pos_map.get(edge.source)
            tgt_pos = node_pos_map.get(edge.target)

            if not src_pos or not tgt_pos:
                continue

            start_pt = [src_pos[0], src_pos[1] + 1.1, src_pos[2]]
            end_pt = [tgt_pos[0], tgt_pos[1] + 1.1, tgt_pos[2]]

            control_pts: List[List[float]] = []
            edge_color = "#3B82F6"

            # Curved Leap for Residual / Skip Highways
            if edge.type in ["residual_skip", "cross_attention"]:
                mid_x = (start_pt[0] + end_pt[0]) / 2.0
                dist_x = abs(end_pt[0] - start_pt[0])
                arc_height = min(4.5, 1.2 + (dist_x * 0.22))
                control_pts = [
                    [start_pt[0] + (dist_x * 0.15), start_pt[1] + (arc_height * 0.8), 0.0],
                    [mid_x, arc_height, 0.0],
                    [end_pt[0] - (dist_x * 0.15), end_pt[1] + (arc_height * 0.8), 0.0]
                ]
                edge_color = "#EF4444" if edge.type == "residual_skip" else "#D97706"

            scene_edge = SceneEdge3D(
                id=f"scene_{edge.id}",
                source_id=edge.source,
                target_id=edge.target,
                type=edge.type,
                start_pos_3d=start_pt,
                end_pos_3d=end_pt,
                control_points_3d=control_pts,
                color_hex=edge_color,
                label=edge.label
            )
            scene_edges.append(scene_edge)

        return SceneGraph3D(
            nodes_3d=scene_nodes,
            edges_3d=scene_edges,
            camera_target=[0.0, 0.0, 0.0],
            bounds_3d={"min_x": start_x - 2.0, "max_x": abs(start_x) + 2.0, "min_y": -2.0, "max_y": 5.0}
        )

    def _calculate_dimensions(self, node: ArchitectureNode) -> Tuple[List[float], Dict[str, int]]:
        """Calculate physical slab size [w, h, d] and lattice subdivisions."""
        t = node.type
        if t in ["mlp", "feed_forward", "unet_block"]:
            return [2.0, 2.6, 0.55], {"rows": 6, "cols": 20}
        elif t in ["attention", "multi_head_attention", "cross_attention"]:
            return [2.0, 2.0, 0.35], {"rows": 6, "cols": 6}
        elif t in ["embedding", "linear", "matrix", "ssm_block", "selective_scan"]:
            return [1.8, 2.2, 0.45], {"rows": 6, "cols": 12}
        elif t in ["normalization", "layer_norm", "rms_norm", "residual"]:
            return [1.4, 1.9, 0.35], {"rows": 6, "cols": 8}
        elif t in ["encoder", "decoder", "transformer_block"]:
            return [2.4, 3.0, 0.6], {"rows": 8, "cols": 16}
        else:
            return [1.8, 2.2, 0.4], {"rows": 6, "cols": 10}


layout_engine = LayoutEngine()
