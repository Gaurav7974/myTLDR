"use client";

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

interface ArchitectureDiagramProps {
  nodes: Node[];
  edges: Edge[];
  nodeColors: Record<string, string>;
}

export default function ArchitectureDiagram({
  nodes,
  edges,
  nodeColors,
}: ArchitectureDiagramProps) {
  // Style nodes with type-based colors
  const styledNodes: Node[] = useMemo(
    () =>
      nodes.map((n) => ({
        ...n,
        style: {
          ...n.style,
          background: nodeColors[String(n.data?.node_type || '')] || "#94a3b8",
          color: "#fff",
          border: "none",
          borderRadius: 8,
          padding: 10,
          fontSize: 12,
          fontWeight: 500,
          minWidth: 120,
          textAlign: "center",
        },
      })),
    [nodes, nodeColors]
  );

  const styledEdges: Edge[] = useMemo(
    () =>
      edges.map((e) => ({
        ...e,
        style: { stroke: "#64748b", strokeWidth: 2 },
        labelStyle: { fontSize: 10, fill: "#64748b" },
      })),
    [edges]
  );

  const [rfNodes, , onNodesChange] = useNodesState(styledNodes);
  const [rfEdges, , onEdgesChange] = useEdgesState(styledEdges);

  const nodeColor = useCallback(
    (node: Node) => {
      const type = String(node.data?.node_type || '');
      return nodeColors[type] || "#94a3b8";
    },
    [nodeColors]
  );

  return (
    <div className="w-full h-full min-h-[400px]">
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        attributionPosition="bottom-left"
      >
        <Background gap={16} size={1} color="#e2e8f0" />
        <Controls />
        <MiniMap nodeColor={nodeColor} pannable zoomable />
      </ReactFlow>
    </div>
  );
}
