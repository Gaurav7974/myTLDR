"""Multimodal architecture extraction from paper figures via a vision LLM.

The vision LLM is asked to produce a *visual* graph: nodes with normalized
bounding boxes, group containers, edge polylines, and any visual style cues
actually present in the figure. A separate verification pass then compares
that graph back to the source image so we never render a hallucinated
diagram in production.

The text path (``graph_extraction.GraphExtractor``) is unchanged for
semantic enrichment; the merger in ``pipeline.py`` is the only place where
visual geometry and text semantics are combined.
"""

from __future__ import annotations

import json
import time
import requests
from typing import Any, Optional

from pydantic import ValidationError

from app.core.config import settings
from app.core.logger import get_logger
log = get_logger(__name__)
from app.models.graph import (
    ArchitectureGraph,
    GRAPH_PAYLOAD_VERSION,
    GraphProvenance,
    GraphVerification,
    VISION_CONFIDENCE_THRESHOLD,
)


# ── Prompts ──────────────────────────────────────────────────────────
VISION_GRAPH_PROMPT = """You are an expert at reading neural-network architecture diagrams.

Context flag: {figure_context}
- If "CROPPED FIGURE" — image is tightly cropped to the diagram. Only include what's visibly present.
- If "WHOLE PAGE (fallback)" — image is a full PDF page render. Locate the architecture diagram within the page, ignore headers/footers/margins/text, and extract only the diagram's components. Do NOT hallucinate nodes from surrounding text.

Your job is to extract the figure-faithful visual structure of the diagram.

CRITICAL Visual Rules (these are your only sources of truth):
- Only include components that are VISIBLY PRESENT in the figure.
- For each node, give the NORMALIZED bounding box (x, y, w, h) where x,y is
  the top-left corner and w,h are width/height. All values in [0.0, 1.0]
  relative to the source image's width/height.
- For groups of nodes drawn inside a visible container (rectangle/dashed
  box, labelled e.g. "Encoder", "Block 1", "Decoder"), emit a "groups" entry
  with its own bounding box and a label.
- For each edge, give a "path_points" array of 2+ normalized (x, y) points
  tracing the visible arrow/line in the figure. Use straight segments — the
  renderer will smooth them. Only include an edge if you can see an actual
  drawn line connecting the source and target.
- Preserve the visual style if it is clearly drawn: solid / dashed / dotted
  stroke, residual-looking long skip, or beam-like grouped arrows.
- If a node box is filled with a clearly distinct color from neighbours,
  capture that color in "style.fill" as a hex string.
- Use the "raw_style" field to record any style string the figure literally
  shows (e.g. "skip-connection", "top-k routing") so we can surface it in
  the sidebar.

CRITICAL Anti-hallucination Rules:
- Do NOT invent components or connections not shown.
- If you cannot confidently locate a node, OMIT it instead of guessing.
- Do NOT generate descriptions, "how it works" prose, or math formulas.
  Visual extraction is geometry + labels only. Semantics are added by a
  separate text-based step.

Output valid JSON matching this schema:
{
  "version": "2.0",
  "source": "vision",
  "layout_mode": "figure",
  "figure_id": "optional figure identifier from the caption",
  "confidence": 0.0-1.0,
  "groups": [
    {"id": "group_id", "label": "Group Label", "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
     "color": "#hex", "confidence": 0.0-1.0}
  ],
  "nodes": [
    {
      "id": "unique_id",
      "label": "Human Readable Name (as written in figure)",
      "type": "node_type",
      "bbox": {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.05},
      "group_id": "optional_group_id",
      "shape": "rect|rounded|ellipse|diamond|cylinder|parallelogram|trapezoid|custom",
      "style": {"fill": "#hex", "border": "#hex", "text": "#hex", "shape": "rounded"},
      "confidence": 0.0-1.0
    }
  ],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id",
      "label": "optional visible edge label",
      "type": "data_flow|residual_skip|cross_attention|gated_modulation|recurrent_state|custom",
      "path_points": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
      "source_anchor": {"x": 0.1, "y": 0.1},
      "target_anchor": {"x": 0.2, "y": 0.2},
      "style": {"stroke": "#hex", "width": 1.5, "dasharray": "6 4", "name": "dashed|solid|residual|beam|highlight|default"},
      "raw_style": {"original": "whatever you literally saw"}
    }
  ]
}

Node types: input, output, embedding, attention, feedforward, normalization,
block, router, expert, ssm, gate, cache, other.
"""


VISION_VERIFICATION_PROMPT = """You are an expert ML researcher verifying whether an extracted
architecture graph matches the figure shown.

Compare each node, group, and edge in the proposed graph against what is
visibly present in the figure. Be honest: if a node was guessed or if a
major visible component was missed, flag it.

For each node and each edge, output whether it matches the figure, was
HALLUCINATED (not in figure), or was ambiguous. Also list any MAJOR visible
components (with their clear figure labels) that are MISSING from the graph.

Output valid JSON:
{
  "valid": true or false,
  "confidence": 0.0-1.0,
  "hallucinations": [
    {"element": "node or edge id", "severity": "structural|cosmetic", "reason": "why hallucinated"}
  ],
  "missing_elements": [
    {"element": "major visible component label", "severity": "structural|cosmetic", "reason": "why missing, e.g. style vs structure"}
  ],
  "node_verification": [
    {"node_id": "id", "label": "label", "status": "match|hallucinated|ambiguous", "note": "short reason"}
  ],
  "edge_verification": [
    {"edge": "source->target", "status": "match|hallucinated|ambiguous", "note": "short reason"}
  ],
  "reason": "one-sentence summary"
}
Severity rules:
- structural = missing/hallucinated node, missing edge between distinct components, wrong topology, extra block that changes dataflow. Must set valid=false.
- cosmetic = style mismatch (dashed vs solid, color, line width, arrowhead shape, font), label casing/punctuation, truncated text. Do NOT set valid=false for cosmetic alone; set valid=true and confidence 0.7-0.95.

Be CONSERVATIVE only for structural. The downstream renderer shows structural failures as invalid, cosmetic as amber "Unverified — style mismatches" but still renders.
"""


# ── Parsers (pure, testable) ─────────────────────────────────────────
def _parse_vision_payload(raw: dict[str, Any], paper_title: str = "",
                          figure_id: str | None = None,
                          figure_page: int | None = None) -> ArchitectureGraph:
    """Convert a vision-LLM JSON payload into an ArchitectureGraph.

    The payload may be the rich schema or a minimal ``{nodes, edges}``
    legacy payload — both are accepted, and missing fields are filled with
    safe defaults.
    """
    if not isinstance(raw, dict):
        raise ValueError("vision payload must be a JSON object")

    # Pop graph-level fields; everything else is nodes/edges/groups
    graph_kwargs: dict[str, Any] = {
        "version": raw.get("version") or GRAPH_PAYLOAD_VERSION,
        "source": raw.get("source") or "vision",
        "layout_mode": raw.get("layout_mode") or "figure",
        "figure_id": figure_id or raw.get("figure_id"),
        "figure_page": figure_page if figure_page is not None else raw.get("figure_page"),
        "confidence": raw.get("confidence"),
        "paper_title": paper_title or raw.get("paper_title"),
        "metadata": raw.get("metadata"),
    }
    # Groups
    if isinstance(raw.get("groups"), list):
        graph_kwargs["groups"] = raw["groups"]
    # Nodes
    if "nodes" in raw:
        graph_kwargs["nodes"] = raw["nodes"]
    if "edges" in raw:
        graph_kwargs["edges"] = raw["edges"]

    graph = ArchitectureGraph(**graph_kwargs)

    # Tag every node/edge with vision provenance if not already set
    for node in graph.nodes:
        if node.provenance is None:
            node.provenance = GraphProvenance(
                source="vision",
                paper_title=paper_title or None,
                figure_id=figure_id or graph.figure_id,
                figure_page=figure_page or graph.figure_page,
            )
    return graph


def _parse_verification_payload(raw: dict[str, Any]) -> GraphVerification:
    if not isinstance(raw, dict):
        return GraphVerification(valid=False, reason="verification payload was not a JSON object")
    raw_valid, valid = raw.get("valid", False), raw.get("valid", False) is True
    def _structural(items):
        out=[]
        for it in (items or []):
            if isinstance(it, dict):
                if it.get("severity") == "structural": out.append(str(it.get("element") or it.get("reason") or it))
                # cosmetic ignored for gate
            elif isinstance(it, str) and it.strip(): out.append(it)
        return out
    try:
        return GraphVerification(
            valid=valid,
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            hallucinations=_structural(raw.get("hallucinations")),
            missing_elements=_structural(raw.get("missing_elements")),
            node_verification=list(raw.get("node_verification") or []),
            edge_verification=list(raw.get("edge_verification") or []),
            reason=str(raw.get("reason") or "") or None,
        )
    except (ValidationError, ValueError, TypeError) as e:
        return GraphVerification(valid=False, confidence=0.0, reason=f"verification payload failed to parse: {e}")


def is_verified_vision_graph(graph: ArchitectureGraph | None) -> bool:
    """Trust-first gate: only an explicit verified vision graph may use
    ``layout_mode='figure'``. Anything else (text-only, partial, low
    confidence) must fall back to ``layout_mode='auto'`` or be hidden."""
    if graph is None:
        return False
    if graph.source != "vision" and graph.layout_mode != "figure":
        return False
    v = graph.verification
    if v is None or not v.valid:
        return False
    if v.confidence < VISION_CONFIDENCE_THRESHOLD:
        return False
    if v.hallucinations:
        return False
    return True


def classify_candidates(figures, paper_title: str = "") -> list[dict]:
    """Cheap per-candidate vision classifier: architecture_diagram | other + confidence. No new dep."""
    if not figures or not settings.VISION_API_KEY: return []
    out=[]
    for fig in figures:
        if not fig.base64: continue
        img_url = f"data:image/png;base64,{fig.base64}" if not fig.base64.startswith("data:") else fig.base64
        msgs=[{"role":"system","content":"You are an image classifier."},{"role":"user","content":[{"type":"text","text":"Is this image an architecture/model diagram, or something else (chart, plot, table, photo)? Respond ONLY with valid JSON: {\"label\": \"architecture_diagram\" | \"other\", \"confidence\": 0.0-1.0}"},{"type":"image_url","image_url":{"url":img_url}}]}]
        try:
            t0=time.time()
            raw=FigureExtractor()._call_and_parse(msgs)  # reuse retry/timeout logic via new instance
            # _call_and_parse expects vision prompt shape but works for classifier too
            label, conf = raw.get("label"), float(raw.get("confidence",0.0) or 0.0)
            if label not in ("architecture_diagram","other"): label="other"
            out.append({"figure":fig,"label":label,"confidence":conf,"time":time.time()-t0})
            log.info(f"classify {fig.id} p{fig.page} -> {label} {conf:.2f}")
        except Exception as e: log.warning(f"classify {fig.id} failed: {e}"); out.append({"figure":fig,"label":"other","confidence":0.0})
    # Log distribution
    try: log.info(f"classify distribution: {','.join(f'{x['confidence']:.2f}' for x in out)}")
    except Exception: pass
    return out

def pick_architecture_figure_classifier(figures, paper_title: str = ""):
    """Replace caption keyword scoring with classifier for every paper, no exceptions."""
    cands = classify_candidates(figures, paper_title)
    arch = [c for c in cands if c["label"]=="architecture_diagram"]
    if arch: return max(arch, key=lambda x: x["confidence"])["figure"]
    return None

# ── Extractor ────────────────────────────────────────────────────────
class FigureExtractor:
    """Extracts and verifies an ArchitectureGraph from a single paper figure."""

    def extract_architecture(
        self, figure, paper_title: str = ""
    ) -> Optional[ArchitectureGraph]:
        """Extract the architecture graph from a figure. Returns None on any
        failure (no key, malformed JSON, validation error, etc.) so the
        pipeline can fall back without raising."""
        if not settings.VISION_API_KEY:
            log.warning("[WARN] Vision API key not configured; skipping image-based extraction")
            return None
        if not figure or not figure.base64:
            return None

        image_url = (
            figure.base64
            if figure.base64.startswith("data:")
            else f"data:image/jpeg;base64,{figure.base64}"
        )

        is_whole = bool(getattr(figure, "is_whole_page_fallback", False))
        figure_context = "WHOLE PAGE (fallback) — locate diagram within page" if is_whole else "CROPPED FIGURE"
        system_prompt = VISION_GRAPH_PROMPT.replace("{figure_context}", figure_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Paper: {paper_title}\nFigure context: {figure_context}\nExtract the architecture graph shown in this figure as JSON."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        try:
            raw = self._call_and_parse(messages)
        except Exception as e:
            log.error(f"[ERROR] Vision architecture extraction failed: {e}", exc_info=True)
            return None

        try:
            return _parse_vision_payload(
                raw,
                paper_title=paper_title,
                figure_id=getattr(figure, "id", None),
                figure_page=getattr(figure, "page", None),
            )
        except (ValidationError, ValueError) as e:
            log.warning(f"[WARN] Vision payload failed validation: {e}")
            return None

    def verify_figure_graph(
        self, figure, graph: ArchitectureGraph, paper_title: str = ""
    ) -> GraphVerification:
        """Run the figure-faithfulness verifier against the source image.

        Always returns a GraphVerification (never None). When the API is
        unavailable or the payload is malformed, returns a verification
        with ``valid=False`` so the pipeline can fall back safely.
        """
        if not settings.VISION_API_KEY or not figure or not figure.base64:
            return GraphVerification(
                valid=False,
                reason="vision API unavailable; cannot verify figure graph",
            )
        if graph is None:
            return GraphVerification(
                valid=False,
                confidence=0.0,
                reason="no graph supplied for verification",
            )

        image_url = (
            figure.base64
            if figure.base64.startswith("data:")
            else f"data:image/jpeg;base64,{figure.base64}"
        )

        # Compact graph summary for the verifier
        graph_summary = {
            "version": graph.version,
            "source": graph.source,
            "groups": [g.model_dump(mode="json", exclude_none=True) for g in graph.groups],
            "nodes": [n.model_dump(mode="json", exclude_none=True) for n in graph.nodes],
            "edges": [e.model_dump(mode="json", exclude_none=True) for e in graph.edges],
        }

        is_whole = bool(getattr(figure, "is_whole_page_fallback", False))
        figure_context = "WHOLE PAGE — diagram is embedded in full page, ignore margins/text" if is_whole else "CROPPED FIGURE"
        messages = [
            {"role": "system", "content": VISION_VERIFICATION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Paper: {paper_title}\nFigure context: {figure_context}\nVerify this graph against the source figure.\n\nGraph JSON:\n{json.dumps(graph_summary, default=str)}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        try:
            raw = self._call_and_parse(messages, is_verification=True)
        except Exception as e:
            log.error(f"[ERROR] Vision verification call failed: {e}", exc_info=True)
            return GraphVerification(
                valid=False,
                confidence=0.0,
                reason=f"verification call failed: {e}",
            )

        return _parse_verification_payload(raw)

    def _call_and_parse(self, messages: list, is_verification: bool = False) -> dict:
        """POST to the vision LLM and return the JSON object from its reply."""
        payload = {
            "model": settings.VISION_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        last_err: Exception | None = None
        for attempt in range(settings.LLM_MAX_RETRIES):
            try:
                resp = requests.post(
                    settings.VISION_API_URL,
                    headers={
                        "Authorization": f"Bearer {settings.VISION_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=45,
                )
                if resp.status_code == 429:
                    time.sleep(settings.LLM_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # Persist raw LLM output for debugging (B.6)
                try:
                    import os, pathlib
                    raw_dir = pathlib.Path(__file__).parent.parent / "logs" / "vision_raw"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    prefix = "verify" if is_verification else "extract"
                    (raw_dir / f"{prefix}_{int(time.time())}.json").write_text(content, encoding="utf-8")
                    log.info(f"vision raw persisted: {prefix} {len(content)} chars")
                except Exception: pass
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ValueError(f"vision reply was not a JSON object: {type(data).__name__}")
                return data
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                last_err = e
                if attempt < settings.LLM_MAX_RETRIES - 1:
                    time.sleep(settings.LLM_RETRY_BACKOFF_BASE * (attempt + 1))
                    continue
                break
        raise RuntimeError(
            f"vision {'verification' if is_verification else 'extraction'} failed: {last_err}"
        )


figure_extractor = FigureExtractor()
