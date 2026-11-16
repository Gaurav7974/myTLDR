"""Summary schema for academic visualization platform.

Tolerant by design: the LLM fills this schema, and it routinely drifts on
taxonomy fields (e.g. emitting "method" instead of "mechanism") or omits a
field. A strict schema turned one bad label into a completely missing summary,
so taxonomy values are normalized here and content fields default to empty
rather than failing the whole payload.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

KEYWORD_CATEGORIES = ("core", "architecture", "mechanism", "math", "optimization")
CONTRIBUTION_CATEGORIES = (
    "novel",
    "architectural",
    "methodological",
    "experimental",
    "supporting",
)

# Loose labels the model reaches for, mapped onto the fixed taxonomy.
_KEYWORD_CATEGORY_ALIASES = {
    "method": "mechanism",
    "methodology": "mechanism",
    "technique": "mechanism",
    "model": "architecture",
    "hardware": "architecture",
    "system": "architecture",
    "metric": "core",
    "dataset": "core",
    "concept": "core",
    "idea": "core",
}


def _normalize(
    value: object, allowed: tuple[str, ...], aliases: dict[str, str], fallback: str
) -> str:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    return aliases.get(text, fallback)


class ProvenanceSource(BaseModel):
    page: Optional[int] = Field(None, description="Page number in the original PDF")
    section: Optional[str] = Field(
        None, description="Section number or heading (e.g. '3.2')"
    )
    figure: Optional[str] = Field(
        None, description="Figure or table identifier (e.g. 'Figure 2')"
    )


class SummaryOverview(BaseModel):
    one_sentence: str = Field(
        default="", description="25-40 words core takeaway in plain English"
    )
    problem: str = Field(
        default="",
        description="30-50 words explaining the bottleneck or limitation solved",
    )
    solution: str = Field(
        default="", description="30-60 words explaining the proposed methodology"
    )
    why_it_matters: str = Field(
        default="",
        description="30-50 words explaining academic/industrial significance",
    )


class KeySpecification(BaseModel):
    label: str = Field(
        default="",
        description="Name of hyperparameter or architectural metric (e.g. 'Attention Heads', 'd_model')",
    )
    value: str = Field(
        default="", description="Exact quantitative value (e.g. '8', '512', '2048')"
    )
    source: Optional[ProvenanceSource] = None


class PaperContribution(BaseModel):
    title: str = Field(default="", description="Short title of the contribution")
    summary: str = Field(default="", description="20-35 words explaining what was done")
    category: str = Field(
        "supporting", description="Classification of contribution type"
    )
    importance: Optional[str] = Field(
        None,
        description="15-25 words explaining why this specific contribution matters",
    )
    source: Optional[ProvenanceSource] = None

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v: object) -> str:
        # Unknown categories fall back to "supporting" — the least claiming label.
        return _normalize(v, CONTRIBUTION_CATEGORIES, {}, "supporting")


class KeyFinding(BaseModel):
    title: str = Field(
        default="", description="Name of the empirical result or experiment"
    )
    summary: str = Field(
        default="", description="20-35 words describing the demonstrated finding"
    )
    metric_value: Optional[str] = Field(
        None, description="Specific score/metric (e.g. '28.4 BLEU (+2.0 over SOTA)')"
    )
    source: Optional[ProvenanceSource] = None


class PaperLimitation(BaseModel):
    title: str = Field(default="", description="Name of the limitation or tradeoff")
    summary: str = Field(
        default="", description="20-35 words describing the constraint or weakness"
    )
    details: Optional[str] = Field(
        None, description="Brief technical explanation of the underlying cause"
    )
    source: Optional[ProvenanceSource] = None


class KeywordDefinition(BaseModel):
    term: str = Field(
        default="",
        description="Exact technical term as it appears in the paper (no leading '#')",
    )
    category: str = Field("core", description="How the term is used in the paper")
    definition: str = Field(
        default="", description="One precise sentence defining the term"
    )
    simple_explanation: str = Field(
        default="", description="One plain-English sentence a non-expert understands"
    )
    formula: Optional[str] = Field(
        None, description="LaTeX formula from the paper, or null"
    )
    related_section: Optional[str] = Field(
        None, description="Where the term is defined, e.g. 'Page 4, Section 3.2'"
    )

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v: object) -> str:
        return _normalize(v, KEYWORD_CATEGORIES, _KEYWORD_CATEGORY_ALIASES, "core")


class StructuredPaperSummary(BaseModel):
    overview: SummaryOverview = Field(default_factory=SummaryOverview)
    key_specs: List[KeySpecification] = Field(default_factory=list)
    contributions: List[PaperContribution] = Field(default_factory=list)
    key_findings: List[KeyFinding] = Field(default_factory=list)
    limitations: List[PaperLimitation] = Field(default_factory=list)
    keywords: List[KeywordDefinition] = Field(default_factory=list)

    @field_validator("keywords", mode="before")
    @classmethod
    def _drop_blank_keywords(cls, v: object) -> list:
        if not isinstance(v, list):
            return []
        return [
            k for k in v if isinstance(k, dict) and str(k.get("term") or "").strip()
        ]
