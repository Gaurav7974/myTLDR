"""Graph schema for architecture extraction."""

from pydantic import BaseModel
from typing import Literal

NodeType = Literal[
    "input", "output", "embedding", "attention",
    "feedforward", "normalization", "block",
    "router", "expert", "ssm", "gate", "cache", "other"
]


class GraphNode(BaseModel):
    id: str
    label: str
    type: NodeType


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str | None = None


class ArchitectureGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
