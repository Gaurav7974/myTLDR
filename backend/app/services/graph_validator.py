"""ArchitectureIR Graph Validation and Integrity Checking Layer."""

from typing import Dict, Any, List, Set, Tuple
from app.models.architecture_ir import ArchitectureIR, ArchitectureNode, ArchitectureEdge, NodeTypeTaxonomy


class GraphValidator:
    """Validates raw LLM-generated ArchitectureIR against schema and graph rules."""

    KNOWN_TYPES: Set[str] = {
        "input", "output", "embedding", "linear", "matrix", "attention",
        "multi_head_attention", "cross_attention", "mlp", "feed_forward",
        "normalization", "layer_norm", "rms_norm", "residual", "convolution",
        "conv1d", "conv2d", "pooling", "encoder", "decoder", "transformer_block",
        "ssm_block", "selective_scan", "unet_block", "diffusion_step", "adapter_lora",
        "router", "expert", "classifier", "loss", "custom"
    }

    def validate_and_normalize(self, raw_data: Dict[str, Any]) -> Tuple[ArchitectureIR, List[str]]:
        """
        Validate, normalize and repair raw architecture data into canonical ArchitectureIR.
        Returns the sanitized ArchitectureIR and a list of validation notices / repairs.
        """
        notices: List[str] = []

        # 1. Architecture metadata validation
        arch_meta = raw_data.get("architecture", {})
        if not arch_meta.get("name"):
            arch_meta["name"] = "Neural Architecture"
            notices.append("Defaulted missing architecture name")
        if not arch_meta.get("paradigm"):
            arch_meta["paradigm"] = "transformer"
            notices.append("Defaulted paradigm to 'transformer'")

        # 2. Node validation & taxonomy enforcement
        raw_nodes = raw_data.get("nodes", [])
        valid_nodes: List[ArchitectureNode] = []
        node_ids: Set[str] = set()

        for idx, node_dict in enumerate(raw_nodes):
            nid = node_dict.get("id") or f"node_{idx+1}"
            
            # Ensure unique ID
            if nid in node_ids:
                nid = f"{nid}_{idx+1}"
                node_dict["id"] = nid
                notices.append(f"Deduplicated node ID to '{nid}'")
            node_ids.add(nid)

            # Taxonomy graceful degradation
            raw_type = node_dict.get("type", "custom")
            if raw_type not in self.KNOWN_TYPES:
                node_dict["type"] = "custom"
                notices.append(f"Degraded unknown component type '{raw_type}' on '{nid}' to 'custom'")

            try:
                node_obj = ArchitectureNode(**node_dict)
                valid_nodes.append(node_obj)
            except Exception as e:
                notices.append(f"Repaired node '{nid}' schema parsing: {e}")
                # Fallback minimal node
                valid_nodes.append(ArchitectureNode(
                    id=nid,
                    label=node_dict.get("label", nid),
                    type="custom",
                    description=node_dict.get("description", "")
                ))

        # 3. Edge validation & dangling connection pruning
        raw_edges = raw_data.get("edges", [])
        valid_edges: List[ArchitectureEdge] = []
        edge_ids: Set[str] = set()

        for idx, edge_dict in enumerate(raw_edges):
            eid = edge_dict.get("id") or f"edge_{idx+1}"
            src = edge_dict.get("source")
            tgt = edge_dict.get("target")

            # Prune dangling edges
            if not src or src not in node_ids:
                notices.append(f"Pruned edge '{eid}': source '{src}' does not exist")
                continue
            if not tgt or tgt not in node_ids:
                notices.append(f"Pruned edge '{eid}': target '{tgt}' does not exist")
                continue
            if src == tgt:
                notices.append(f"Pruned self-loop edge '{eid}' on '{src}'")
                continue

            try:
                edge_obj = ArchitectureEdge(**edge_dict)
                valid_edges.append(edge_obj)
                edge_ids.add(eid)
            except Exception as e:
                notices.append(f"Repaired edge '{eid}': {e}")
                valid_edges.append(ArchitectureEdge(
                    id=eid,
                    source=src,
                    target=tgt,
                    type="data_flow"
                ))

        # 4. If no edges exist between sequential nodes, automatically synthesize linear dataflow
        if len(valid_edges) == 0 and len(valid_nodes) > 1:
            notices.append("Synthesized sequential dataflow edges across nodes")
            for i in range(len(valid_nodes) - 1):
                valid_edges.append(ArchitectureEdge(
                    id=f"flow_{i+1}",
                    source=valid_nodes[i].id,
                    target=valid_nodes[i+1].id,
                    type="data_flow",
                    label=f"Step {i+1} → {i+2}"
                ))

        # 5. Construct canonical ArchitectureIR
        cleaned_payload = {
            "version": raw_data.get("version", "1.0"),
            "architecture": arch_meta,
            "nodes": [n.model_dump() for n in valid_nodes],
            "edges": [e.model_dump() for e in valid_edges],
            "visualization_hints": raw_data.get("visualization_hints", {"recommended_lod": 2, "flow_direction": "horizontal"})
        }

        canonical_ir = ArchitectureIR(**cleaned_payload)
        return canonical_ir, notices


graph_validator = GraphValidator()
