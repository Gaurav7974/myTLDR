"""Strict summary schema for academic visualization platform."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class ProvenanceSource(BaseModel):
    page: Optional[int] = Field(None, description="Page number in the original PDF")
    section: Optional[str] = Field(None, description="Section number or heading (e.g. '3.2')")
    figure: Optional[str] = Field(None, description="Figure or table identifier (e.g. 'Figure 2')")


class SummaryOverview(BaseModel):
    one_sentence: str = Field(..., description="25-40 words core takeaway in plain English")
    problem: str = Field(..., description="30-50 words explaining the bottleneck or limitation solved")
    solution: str = Field(..., description="30-60 words explaining the proposed methodology")
    why_it_matters: str = Field(..., description="30-50 words explaining academic/industrial significance")


class KeySpecification(BaseModel):
    label: str = Field(..., description="Name of hyperparameter or architectural metric (e.g. 'Attention Heads', 'd_model')")
    value: str = Field(..., description="Exact quantitative value (e.g. '8', '512', '2048')")
    source: Optional[ProvenanceSource] = None


class PaperContribution(BaseModel):
    title: str = Field(..., description="Short title of the contribution")
    summary: str = Field(..., description="20-35 words explaining what was done")
    category: Literal["novel", "architectural", "methodological", "experimental", "supporting"] = Field(
        "novel", description="Strict classification of contribution type"
    )
    importance: Optional[str] = Field(None, description="15-25 words explaining why this specific contribution matters")
    source: Optional[ProvenanceSource] = None


class KeyFinding(BaseModel):
    title: str = Field(..., description="Name of the empirical result or experiment")
    summary: str = Field(..., description="20-35 words describing the demonstrated finding")
    metric_value: Optional[str] = Field(None, description="Specific score/metric (e.g. '28.4 BLEU (+2.0 over SOTA)')")
    source: Optional[ProvenanceSource] = None


class PaperLimitation(BaseModel):
    title: str = Field(..., description="Name of the limitation or tradeoff")
    summary: str = Field(..., description="20-35 words describing the constraint or weakness")
    details: Optional[str] = Field(None, description="Brief technical explanation of the underlying cause")
    source: Optional[ProvenanceSource] = None


class StructuredPaperSummary(BaseModel):
    overview: SummaryOverview
    key_specs: List[KeySpecification] = Field(default_factory=list)
    contributions: List[PaperContribution] = Field(default_factory=list)
    key_findings: List[KeyFinding] = Field(default_factory=list)
    limitations: List[PaperLimitation] = Field(default_factory=list)
