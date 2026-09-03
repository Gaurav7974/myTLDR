"""Main pipeline orchestrator."""

import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logger import get_logger
log = get_logger(__name__)
from app.core.security import sanitize_arxiv_id
from app.services.text_extraction import text_extractor
from app.services.graph_extraction import graph_extractor
from app.services.explanation import explanation_service
from app.services.forward_pass import forward_pass_extractor
from app.services.ir_extraction import ir_extractor
from app.services.figure_extraction import (
    figure_extractor,
    is_verified_vision_graph,
    pick_architecture_figure_classifier,
)
from app.services.graph_merge import (
    merge_graph_metadata,
    should_render_reconstruction,
)
from app.services.flow_layout import graph_to_react_flow
from app.services.cache import cache_service, compute_content_hash
from app.services.rag import rag_service


async def process_paper(
    source_type: str,
    file_bytes: bytes | None,
    url: str | None,
    difficulty: str,
    db: AsyncSession
) -> AsyncGenerator[dict, None]:
    """
    Process a paper through the full pipeline.
    Yields SSE events for frontend consumption.
    """
    
    # ── Stage 1: Input routing & text extraction ──
    yield {"stage": "extracting", "message": "Downloading and extracting text..."}
    
    metadata = {}
    
    if source_type == "arxiv":
        arxiv_id = _extract_arxiv_id(url)
        metadata = await asyncio.to_thread(text_extractor.get_arxiv_metadata, arxiv_id)
        metadata["pdf_url"] = f"https://arxiv.org/pdf/{sanitize_arxiv_id(arxiv_id)}.pdf"
        metadata["arxiv_id"] = sanitize_arxiv_id(arxiv_id)
        pdf_bytes = await asyncio.to_thread(text_extractor.download_arxiv_pdf, arxiv_id)
        try: extraction = await asyncio.to_thread(text_extractor.extract_from_pdf_bytes, pdf_bytes)
        except Exception as e: yield {"stage": "error", "message": f"OCR failed: {str(e)}."}; return
    elif source_type == "url":
        metadata["pdf_url"] = url
        pdf_bytes = await asyncio.to_thread(text_extractor.download_pdf, url)
        try: extraction = await asyncio.to_thread(text_extractor.extract_from_pdf_bytes, pdf_bytes)
        except Exception as e: yield {"stage": "error", "message": f"OCR failed: {str(e)}."}; return
    elif source_type == "upload":
        if not file_bytes: yield {"stage": "error", "message": "No file provided"}; return
        try: extraction = await asyncio.to_thread(text_extractor.extract_from_pdf_bytes, file_bytes)
        except Exception as e: yield {"stage": "error", "message": f"OCR failed: {str(e)}."}; return
    else: yield {"stage": "error", "message": f"Unknown source type: {source_type}"}; return
    
    full_text = extraction.text
    figures = extraction.figures
    
    if not full_text or len(full_text.strip()) < 100:
        yield {"stage": "error", "message": "Could not extract meaningful text from the paper"}
        return
    
    if not metadata.get("title"):
        try: metadata.update(await asyncio.to_thread(text_extractor.extract_paper_metadata_from_text, full_text))
        except Exception: pass
    
    paper_title = metadata.get("title") or "Uploaded Research Paper"

    # ── Content hash & cache check ──
    content_hash = compute_content_hash(full_text)
    arch_text = text_extractor.find_architecture_section(full_text)
    arch_section_missing = arch_text is None
    if arch_section_missing:
        log.warning("[PIPELINE] No architecture section located — text graph extraction will be skipped, diagram may be vision-only")

    # ── Parsed text for the frontend reader (per-page, images inlined) ──
    cached_parsed = await cache_service.get_parsed_text(db, content_hash)
    if cached_parsed:
        parsed_text = cached_parsed
    else:
        parsed_text = {
            "page_count": extraction.page_count,
            "pages": [
                {
                    "page_number": p.page_number,
                    "markdown": p.markdown,
                    "source": p.source,
                }
                for p in extraction.pages
            ],
            "figures": [
                {"id": f.id, "page": f.page, "caption": f.caption}
                for f in figures
            ],
        }
        await cache_service.store_parsed_text(db, content_hash, parsed_text)
    
    # ── Index paper chunks into pgvector for RAG Q&A ──
    # Own session: this task outlives the request, so it must not reuse the
    # request-scoped db session (races with its teardown).
    async def _index_rag_background(hash_key: str, text_key: str):
        from app.core.database import async_session
        try:
            async with async_session() as session:
                await rag_service.index_paper(session, hash_key, text_key)
        except Exception as e:
            log.warning(f"[WARN] Background RAG indexing failed: {e}")

    asyncio.create_task(_index_rag_background(content_hash, full_text))
    
    # Check explanation cache
    cached_explanation = await cache_service.get_explanation(db, content_hash, difficulty)
    cached_graph = await cache_service.get_graph(db, content_hash)
    
    if cached_explanation and cached_graph is not None:
        # Full cache hit
        from app.models.graph import ArchitectureGraph
        graph = ArchitectureGraph(**cached_graph) if cached_graph else None
        if graph is not None and not should_render_reconstruction(graph):
            # Legacy or unverifiable cache must not masquerade as figure-faithful.
            if graph.layout_mode == "figure":
                graph.layout_mode = "auto"
            if graph.source is None:
                graph.source = "legacy"
        diagram_available = graph is not None and len(graph.nodes) > 0
        react_flow = graph_to_react_flow(graph) if graph and diagram_available else None
        yield {
            "stage": "done",
            "explanation": cached_explanation,
            "graph": react_flow,
            "diagram_available": diagram_available,
            "summary_available": cached_explanation is not None,
            "metadata": metadata,
            "parsed_text": parsed_text,
            "cached": True
        }
        return
    
    yield {"stage": "explaining", "message": f"Generating explanation & architecture for {paper_title[:40]}..."}
    arch_figure = None
    if cached_graph is None and figures:
        try: arch_figure = await asyncio.to_thread(pick_architecture_figure_classifier, figures, paper_title)
        except Exception as e: log.warning(f"classifier pick failed: {e}"); arch_figure = _pick_architecture_figure(figures)
        if not arch_figure: arch_figure = _pick_architecture_figure(figures)  # fallback to keyword
    # Single parallel batch: all independent LLM calls together
    tasks = {}
    if not cached_explanation: tasks["expl"] = asyncio.to_thread(explanation_service.generate, full_text, difficulty)
    tasks["eq"] = asyncio.to_thread(explanation_service.extract_equations, full_text)
    tasks["code"] = asyncio.to_thread(explanation_service.generate_pytorch_code, full_text, paper_title)
    tasks["ir"] = asyncio.to_thread(ir_extractor.extract, full_text, paper_title)
    tasks["bycroft"] = asyncio.to_thread(forward_pass_extractor.extract_3d_architecture, full_text, paper_title)
    if cached_graph is None and not arch_section_missing: tasks["tgraph"] = asyncio.to_thread(graph_extractor.extract_graph, arch_text, paper_title)
    if arch_figure: tasks["vgraph"] = asyncio.to_thread(figure_extractor.extract_architecture, arch_figure, paper_title)
    results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values(), return_exceptions=True)))
    def _ok(k): v=results.get(k); return None if isinstance(v, Exception) else v
    explanation = cached_explanation or _ok("expl")
    if explanation and not cached_explanation:
        try: await cache_service.store_explanation(db, content_hash, difficulty, explanation)
        except Exception: pass
    equations, code_snippets = _ok("eq") or [], _ok("code") or []
    canonical_ir, scene_graph = _ok("ir") or (None, None)
    if isinstance(canonical_ir, tuple): canonical_ir, scene_graph = canonical_ir
    if scene_graph is None:
        from app.models.architecture_ir import SceneGraph3D; scene_graph = SceneGraph3D(nodes_3d=[], edges_3d=[])
    if canonical_ir is None:
        from app.models.architecture_ir import ArchitectureIR, ArchitectureMetadata
        canonical_ir = ArchitectureIR(version="1.0", architecture=ArchitectureMetadata(name=paper_title, paradigm="other"), nodes=[], edges=[])
    bycroft_data = _ok("bycroft") or {}
    text_graph, vision_graph = _ok("tgraph"), _ok("vgraph")
    # Parallel verification (both independent, were sequential)
    text_verified = False
    if text_graph and len(text_graph.nodes) > 0 or vision_graph and len(getattr(vision_graph,'nodes',[])) > 0:
        vtasks = {}
        if vision_graph and len(vision_graph.nodes) > 0: vtasks["vv"] = asyncio.to_thread(figure_extractor.verify_figure_graph, arch_figure, vision_graph, paper_title)
        if text_graph and len(text_graph.nodes) > 0: vtasks["tv"] = asyncio.to_thread(graph_extractor.verify_graph, arch_text, text_graph, paper_title)
        if vtasks:
            vres = dict(zip(vtasks.keys(), await asyncio.gather(*vtasks.values(), return_exceptions=True)))
            if "vv" in vres and not isinstance(vres["vv"], Exception): vision_graph.verification = vres["vv"]
            elif "vv" in vres: log.warning(f"[WARN] Figure verification failed: {vres['vv']}")  # keep graph for amber badge, don't hide
            if "tv" in vres and not isinstance(vres["tv"], Exception): text_verified = bool(vres["tv"].get("valid", False))
    yield {"stage": "diagramming", "message": "Finalizing diagrams..."}

    graph = None
    diagram_available = False
    diagram_reason = "not_attempted"

    if cached_graph is not None:
        from app.models.graph import ArchitectureGraph
        graph = ArchitectureGraph(**cached_graph)
        # Only honor the cached graph if it is still a verifiable rich graph.
        # Legacy cached simple graphs fall back to auto-layout (still rendered).
        if graph.source in ("vision", "merged") and is_verified_vision_graph(graph):
            diagram_available = True
            diagram_reason = "cached_verified_vision"
        elif graph.layout_mode == "figure" and not is_verified_vision_graph(graph):
            # Stale figure-mode cache that no longer passes verification.
            # Re-merge: keep geometry as auto-layout to avoid trusting it.
            graph.layout_mode = "auto"
            graph.source = "legacy"
            diagram_available = True
            diagram_reason = "cached_downgraded_to_auto"
        else:
            diagram_available = len(graph.nodes) > 0
            diagram_reason = "cached_text_or_auto"
    else:
        # Explicit failure propagation (Fix #2): no architecture section + no vision = no diagram
        if arch_section_missing and not vision_graph:
            graph, diagram_available, diagram_reason = None, False, "no_architecture_section"
        elif is_verified_vision_graph(vision_graph):
            graph = merge_graph_metadata(vision_graph, text_graph, prefer="vision") if text_verified and text_graph else vision_graph
            diagram_available, diagram_reason = True, "verified_vision"
        elif vision_graph and len(vision_graph.nodes) > 0:
            # Unverified vision — enrich with text semantics for richer info card, show amber badge
            if text_graph and len(text_graph.nodes) > 0:
                try: graph = merge_graph_metadata(vision_graph, text_graph, prefer="vision")
                except Exception: graph = vision_graph
            else: graph = vision_graph
            diagram_available, diagram_reason = True, "unverified_vision"
        elif text_verified and text_graph and len(text_graph.nodes) > 0:
            graph, diagram_available, diagram_reason = text_graph, True, "verified_text"
        elif text_graph and len(text_graph.nodes) > 0:
            text_graph.layout_mode, text_graph.source = "auto", "text"
            graph, diagram_available, diagram_reason = text_graph, True, "text_auto_fallback"
        elif arch_section_missing:
            graph, diagram_available, diagram_reason = None, False, "no_architecture_section"
        else: graph, diagram_available, diagram_reason = None, False, "no_verified_graph"

        if diagram_available and graph is not None:
            await cache_service.store_graph(
                db, content_hash, graph.model_dump(), True
            )
        else:
            await cache_service.store_graph(db, content_hash, None, False)

    # If we still have a graph but the renderer can prove it is not
    # renderable (e.g. legacy payload with no source verification), down-
    # grade to auto-layout so the frontend never shows a fake reconstruction.
    if graph is not None and not should_render_reconstruction(graph):
        graph.layout_mode = "auto"
        graph.source = "legacy" if graph.source is None else graph.source
        if not diagram_available:
            diagram_available = len(graph.nodes) > 0

    log.info(f"[PIPELINE] diagram_available={diagram_available} reason={diagram_reason} "
          f"nodes={len(graph.nodes) if graph else 0} "
          f"source={getattr(graph, 'source', None)} "
          f"layout_mode={getattr(graph, 'layout_mode', None)}")
    
    # ── Done ──
    react_flow = graph_to_react_flow(graph) if graph and diagram_available else None
    
    yield {
        "stage": "done",
        "explanation": explanation,
        "graph": react_flow,
        "architecture_ir": canonical_ir.model_dump(),
        "scene_graph_3d": scene_graph.model_dump(),
        "bycroft_stages": bycroft_data.get("stages", []),
        "bycroft_highways": bycroft_data.get("residualHighways", []),
        "attention_heatmap": {
            "tokens": bycroft_data.get("heatmapTokens", []),
            "matrix": bycroft_data.get("heatmapMatrix", [])
        },
        "next_token_predictions": bycroft_data.get("nextTokenPredictions", []),
        "equations": equations,
        "code_snippets": code_snippets,
        "architecture_nodes": [n.model_dump() for n in canonical_ir.nodes] if hasattr(canonical_ir, "nodes") else [],
        "diagram_available": diagram_available,
        "summary_available": explanation is not None,
        "metadata": metadata,
        "parsed_text": parsed_text,
        "cached": False
    }


def _extract_arxiv_id(url: str) -> str:
    """Extract arXiv ID from URL."""
    import re
    match = re.search(r'(\d{4}\.\d{4,5})', url)
    if match:
        return match.group(1)
    # Handle arxiv.org/abs/ID format
    parts = url.rstrip('/').split('/')
    return parts[-1]


def _pick_architecture_figure(figures) -> object | None:
    """Pick the figure most likely to be the architecture diagram, by caption.

    Scores captions for architecture-relevant keywords and returns the best match,
    or None if no figure clears the threshold (so the caller falls back to text).
    """
    if not figures:
        return None

    best, best_score = None, 0
    for fig in figures:
        cap = (getattr(fig, "caption", None) or "").lower()
        if not cap:
            continue
        score = (3 if "architecture" in cap else 0) + (2 if "model" in cap else 0) + (2 if any(k in cap for k in ("overview","framework","network","pipeline","diagram","transformer","attention","block")) else 0) + (1 if "figure" in cap or "fig" in cap else 0) - (1 if getattr(fig, "page", 1) <= 1 else 0)
        if score > best_score: best, best_score = fig, score
    # Only return if any caption clears the threshold (>0); otherwise fall back to None so vision path is skipped
    return best
