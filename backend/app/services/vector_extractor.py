"""Vector geometry extractor — ground truth from PDF vector data.

For born-digital PDFs the figure is vector art and PyMuPDF hands you the
ground truth. This module extracts exact label strings + bboxes and
rect/line geometry, so the VLM never needs to guess coordinates.
"""

from __future__ import annotations

import re
import math
from typing import Any

import pymupdf  # type: ignore[import-untyped]

# Regex for N× hint near group edges (notation parsing, not hard-coded)
_NX_RE = re.compile(r"(\d+|N|L)\s*[×xX]|[×xX]\s*(\d+|N|L)")

def _normalize_rect(rect: pymupdf.Rect, page_rect: pymupdf.Rect) -> dict[str, float]:
    """Normalize rect to [0,1] relative to page."""
    pw, ph = page_rect.width, page_rect.height
    if pw <= 0 or ph <= 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    return {
        "x": rect.x0 / pw,
        "y": rect.y0 / ph,
        "w": rect.width / pw,
        "h": rect.height / ph,
    }

def _snap_to_rect_border(pt: dict[str, float], rect: dict[str, float]) -> dict[str, float]:
    """Snap point to nearest border of rect (for edge endpoints)."""
    # Compute closest point on rect border
    cx, cy = rect["x"] + rect["w"] / 2, rect["y"] + rect["h"] / 2
    # Determine closest side
    dx = pt["x"] - cx
    dy = pt["y"] - cy
    # Normalize by half-size
    nx = abs(dx) / (rect["w"] / 2) if rect["w"] > 0 else 0
    ny = abs(dy) / (rect["h"] / 2) if rect["h"] > 0 else 0
    if nx > ny:
        # snap to left/right
        x = rect["x"] if dx < 0 else rect["x"] + rect["w"]
        y = max(rect["y"], min(rect["y"] + rect["h"], pt["y"]))
    else:
        # snap to top/bottom
        x = max(rect["x"], min(rect["x"] + rect["w"], pt["x"]))
        y = rect["y"] if dy < 0 else rect["y"] + rect["h"]
    return {"x": x, "y": y}

def extract_vector_geometry(
    pdf_bytes: bytes,
    page_index: int,
    fig_rect: pymupdf.Rect | None = None,
) -> dict[str, Any]:
    """
    Extract exact geometry from a PDF page.

    Returns dict with nodes, groups, edges, annotations, crop.
    Coordinates are normalized [0,1] relative to page (or fig_rect if given).
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_index < 0 or page_index >= len(doc):
            return {"nodes": [], "groups": [], "edges": [], "annotations": [], "crop": None}
        page = doc[page_index]
        page_rect = page.rect
        clip = fig_rect if fig_rect is not None else page_rect

        # Text dict gives exact label strings + bboxes
        text_dict = page.get_text("dict", clip=clip)
        # Drawings give vector rects/lines
        drawings = page.get_drawings()

        # Collect rects that look like boxes
        rects: list[dict[str, Any]] = []
        lines: list[dict[str, Any]] = []
        for dr in drawings:
            # Check if this drawing is inside clip
            drect = dr["rect"]
            # Simple clip test: intersect
            if drect.x1 < clip.x0 or drect.x0 > clip.x1 or drect.y1 < clip.y0 or drect.y0 > clip.y1:
                continue
            for item in dr.get("items", []):
                itype = item[0]
                if itype == "re":
                    # item can be ("re", x0,y0,x1,y1) or ("re", Rect)
                    if len(item) == 2 and isinstance(item[1], pymupdf.Rect):
                        rect = item[1]
                    elif len(item) == 5:
                        _, x0, y0, x1, y1 = item
                        rect = pymupdf.Rect(x0, y0, x1, y1)
                    else:
                        # Fallback: try to parse as rect from item[1]
                        try:
                            rect = pymupdf.Rect(item[1])
                        except:
                            continue
                    # Filter tiny rects (likely glyph backgrounds) and full-page
                    w, h = rect.width, rect.height
                    if w < 12 or h < 8:
                        continue
                    if w > page_rect.width * 0.92 and h > page_rect.height * 0.92:
                        continue
                    # Check dashes attribute for dashed groups
                    dashes = dr.get("dashes")
                    is_dashed = bool(dashes and dashes != "[]")
                    norm = _normalize_rect(rect, page_rect)
                    # Clip-normalize if fig_rect given
                    if fig_rect is not None:
                        # Convert to fig_rect normalized [0,1]
                        fw, fh = fig_rect.width, fig_rect.height
                        if fw > 0 and fh > 0:
                            norm = {
                                "x": (rect.x0 - fig_rect.x0) / fw,
                                "y": (rect.y0 - fig_rect.y0) / fh,
                                "w": w / fw,
                                "h": h / fh,
                            }
                    rects.append({
                        "rect": rect,
                        "bbox": norm,
                        "is_dashed": is_dashed,
                        "dashes": dashes,
                    })
                elif itype in ("l", "c"):
                    # line/curve: item contains points
                    # For 'l': ("l", x0,y0,x1,y1) ; for 'c': bezier
                    lines.append(dr)

        # Build groups vs nodes: rects that contain other rects are groups
        # Sort by area descending
        rects_sorted = sorted(rects, key=lambda r: r["rect"].width * r["rect"].height, reverse=True)
        groups: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []
        for r in rects_sorted:
            contained = sum(
                1 for o in rects
                if o is not r
                and o["rect"].x0 >= r["rect"].x0 - 1
                and o["rect"].y0 >= r["rect"].y0 - 1
                and o["rect"].x1 <= r["rect"].x1 + 1
                and o["rect"].y1 <= r["rect"].y1 + 1
            )
            # Heuristic: dashed or contains >=2 is group
            is_group = r["is_dashed"] or contained >= 2
            entry: dict[str, Any] = {
                "id": f"rect_{len(groups)+len(nodes)}",
                "bbox": r["bbox"],
                "is_dashed": r["is_dashed"],
                "contained": contained,
            }
            if is_group:
                entry["label"] = ""
                groups.append(entry)
            else:
                # Find label: text spans whose center falls inside rect
                label_parts: list[str] = []
                for block in text_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            sb = span["bbox"]  # [x0,y0,x1,y1] in page coords
                            cx = (sb[0] + sb[2]) / 2
                            cy = (sb[1] + sb[3]) / 2
                            if r["rect"].contains(pymupdf.Point(cx, cy)):
                                txt = span["text"].strip()
                                if txt:
                                    label_parts.append(txt)
                label = " ".join(label_parts).strip()
                entry["label"] = label
                entry["type"] = "other"
                nodes.append(entry)

        # Annotations: text spans not inside any rect
        annotations: list[dict[str, Any]] = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sb = span["bbox"]
                    cx = (sb[0] + sb[2]) / 2
                    cy = (sb[1] + sb[3]) / 2
                    inside = any(
                        r["rect"].contains(pymupdf.Point(cx, cy)) for r in rects
                    )
                    if not inside:
                        txt = span["text"].strip()
                        if txt and _NX_RE.search(txt):
                            # N× hint near group edge
                            norm = {
                                "x": sb[0] / page_rect.width,
                                "y": sb[1] / page_rect.height,
                                "w": (sb[2] - sb[0]) / page_rect.width,
                                "h": (sb[3] - sb[1]) / page_rect.height,
                            }
                            if fig_rect is not None:
                                fw, fh = fig_rect.width, fig_rect.height
                                norm = {
                                    "x": (sb[0] - fig_rect.x0) / fw if fw else 0,
                                    "y": (sb[1] - fig_rect.y0) / fh if fh else 0,
                                    "w": (sb[2] - sb[0]) / fw if fw else 0,
                                    "h": (sb[3] - sb[1]) / fh if fh else 0,
                                }
                            annotations.append({"text": txt, "bbox": norm})

        # Edges: chain line segments into polylines, snap to nearest rect
        # Simplified: collect all line endpoints and chain by proximity ~2pt
        polylines: list[list[dict[str, float]]] = []
        # Extract points from drawings
        points: list[dict[str, float]] = []
        for dr in drawings:
            drect = dr["rect"]
            if drect.x1 < clip.x0 or drect.x0 > clip.x1 or drect.y1 < clip.y0 or drect.y0 > clip.y1:
                continue
            for item in dr.get("items", []):
                if item[0] == "l":
                    # Handle ("l", Point, Point) or ("l", x0,y0,x1,y1)
                    if len(item) == 3 and hasattr(item[1], "x"):
                        p0_raw, p1_raw = item[1], item[2]
                        x0, y0 = float(p0_raw.x), float(p0_raw.y)
                        x1, y1 = float(p1_raw.x), float(p1_raw.y)
                    elif len(item) == 5:
                        _, x0, y0, x1, y1 = item
                    else:
                        continue
                    # Normalize to page or fig_rect
                    if fig_rect is not None:
                        fw, fh = fig_rect.width, fig_rect.height
                        p0 = {"x": (x0 - fig_rect.x0) / fw if fw else 0, "y": (y0 - fig_rect.y0) / fh if fh else 0}
                        p1 = {"x": (x1 - fig_rect.x0) / fw if fw else 0, "y": (y1 - fig_rect.y0) / fh if fh else 0}
                    else:
                        p0 = {"x": x0 / page_rect.width, "y": y0 / page_rect.height}
                        p1 = {"x": x1 / page_rect.width, "y": y1 / page_rect.height}
                    # Chain
                    added = False
                    for pl in polylines:
                        # Check proximity to either end within ~2pt normalized ~0.005
                        if math.hypot(pl[-1]["x"] - p0["x"], pl[-1]["y"] - p0["y"]) < 0.008:
                            pl.append(p1)
                            added = True
                            break
                        if math.hypot(pl[0]["x"] - p1["x"], pl[0]["y"] - p1["y"]) < 0.008:
                            pl.insert(0, p0)
                            added = True
                            break
                    if not added:
                        polylines.append([p0, p1])

        # Convert polylines to edges with snapped endpoints
        edges: list[dict[str, Any]] = []
        # Build rect lookup for snapping
        all_rects = rects
        for pl in polylines:
            if len(pl) < 2:
                continue
            # Only keep polylines that look like wires (length > 0.02, not tiny)
            length = sum(math.hypot(pl[i+1]["x"]-pl[i]["x"], pl[i+1]["y"]-pl[i]["y"]) for i in range(len(pl)-1))
            if length < 0.02:
                continue
            # Snap endpoints to nearest rect border
            p0, p1 = pl[0], pl[-1]
            # Find nearest rect for each end
            def nearest_rect(pt: dict[str, float]) -> dict[str, Any] | None:
                best = None
                best_dist = float("inf")
                for r in all_rects:
                    # Distance to rect border
                    cx, cy = r["bbox"]["x"] + r["bbox"]["w"]/2, r["bbox"]["y"] + r["bbox"]["h"]/2
                    dist = math.hypot(pt["x"]-cx, pt["y"]-cy)
                    if dist < best_dist:
                        best_dist = dist
                        best = r
                if best_dist < 0.15:
                    return best
                return None

            r0 = nearest_rect(p0)
            r1 = nearest_rect(p1)
            if r0:
                pl[0] = _snap_to_rect_border(p0, r0["bbox"])
            if r1:
                pl[-1] = _snap_to_rect_border(p1, r1["bbox"])
            edges.append({
                "id": f"edge_{len(edges)}",
                "path_points": pl,
                "source": None,
                "target": None,
            })

        # Determine crop dimensions for transform
        crop_w, crop_h = clip.width, clip.height

        return {
            "nodes": nodes,
            "groups": groups,
            "edges": edges,
            "annotations": annotations,
            "crop": {"w": crop_w, "h": crop_h, "rect": [clip.x0, clip.y0, clip.x1, clip.y1]},
            "page_rect": {"w": page_rect.width, "h": page_rect.height},
        }
    finally:
        doc.close()


def get_figure_crops(pdf_bytes: bytes, page_index: int | None = None) -> list[pymupdf.Rect]:
    """Score candidate figure regions by vector/raster density and proximity to Figure caption."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        candidates: list[pymupdf.Rect] = []
        pages = [page_index] if page_index is not None else range(len(doc))
        for pi in pages:
            if pi < 0 or pi >= len(doc):
                continue
            page = doc[pi]
            # Find Figure captions
            text_dict = page.get_text("dict")
            fig_spans: list[pymupdf.Rect] = []
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span["text"].strip()
                        if re.match(r"Figure\s+\d+", txt, re.IGNORECASE):
                            sb = span["bbox"]
                            fig_spans.append(pymupdf.Rect(sb[0], sb[1], sb[2], sb[3]))
            # Vector density heatmap: count drawings per region
            # Simplified: score whole page as one candidate, plus caption-proximal regions
            page_rect = page.rect
            # Whole page candidate (for fallback)
            candidates.append(page_rect)
            # Caption-proximal candidates: expand caption rect upward
            for cap_rect in fig_spans:
                # Figure is usually above caption
                candidate = pymupdf.Rect(
                    page_rect.x0, page_rect.y0,
                    page_rect.x1, cap_rect.y0 - 8
                )
                # Filter tiny candidates
                if candidate.width > 80 and candidate.height > 60:
                    candidates.append(candidate)
        # Downscale large crops to 1024 long edge for TPM: handled by caller
        return candidates
    finally:
        doc.close()
