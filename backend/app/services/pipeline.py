"""Main pipeline orchestrator."""

import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.security import sanitize_arxiv_id
from app.services.text_extraction import text_extractor
from app.services.graph_extraction import graph_extractor
from app.services.explanation import explanation_service
from app.services.forward_pass import forward_pass_extractor
from app.services.ir_extraction import ir_extractor
from app.services.figure_extraction import figure_extractor
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
        metadata = text_extractor.get_arxiv_metadata(arxiv_id)
        # Provide direct PDF URL for frontend iframe (no /api/pdf proxy needed for arXiv)
        metadata["pdf_url"] = f"https://arxiv.org/pdf/{sanitize_arxiv_id(arxiv_id)}.pdf"
        metadata["arxiv_id"] = sanitize_arxiv_id(arxiv_id)
        pdf_bytes = text_extractor.download_arxiv_pdf(arxiv_id)
        try:
            extraction = text_extractor.extract_from_pdf_bytes(pdf_bytes)
        except Exception as e:
            yield {"stage": "error", "message": f"OCR failed: {str(e)}. This paper appears to be image-based and could not be processed."}
            return
    elif source_type == "url":
        # Preserve direct PDF URL for iframe; if it's a PDF link the download will succeed
        if url and url.lower().endswith(".pdf"):
            metadata["pdf_url"] = url
        else:
            metadata["pdf_url"] = url
        pdf_bytes = text_extractor.download_pdf(url)
        try:
            extraction = text_extractor.extract_from_pdf_bytes(pdf_bytes)
        except Exception as e:
            yield {"stage": "error", "message": f"OCR failed: {str(e)}. This paper appears to be image-based and could not be processed."}
            return
    elif source_type == "upload":
        if not file_bytes:
            yield {"stage": "error", "message": "No file provided"}
            return
        try:
            extraction = text_extractor.extract_from_pdf_bytes(file_bytes)
        except Exception as e:
            yield {"stage": "error", "message": f"OCR failed: {str(e)}. This paper appears to be image-based and could not be processed."}
            return
    else:
        yield {"stage": "error", "message": f"Unknown source type: {source_type}"}
        return
    
    full_text = extraction.text
    figures = extraction.figures
    
    if not full_text or len(full_text.strip()) < 100:
        yield {"stage": "error", "message": "Could not extract meaningful text from the paper"}
        return
    
    # ── Fallback to LLM first-page metadata extraction if title is missing ──
    if not metadata.get("title"):
        extracted_meta = text_extractor.extract_paper_metadata_from_text(full_text)
        metadata.update(extracted_meta)
    
    paper_title = metadata.get("title") or "Uploaded Research Paper"

    # ── Content hash & cache check ──
    content_hash = compute_content_hash(full_text)
    arch_text = text_extractor.find_architecture_section(full_text)

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
            print(f"[WARN] Background RAG indexing failed: {e}")

    asyncio.create_task(_index_rag_background(content_hash, full_text))
    
    # Check explanation cache
    cached_explanation = await cache_service.get_explanation(db, content_hash, difficulty)
    cached_graph = await cache_service.get_graph(db, content_hash)
    
    if cached_explanation and cached_graph is not None:
        # Full cache hit
        from app.models.graph import ArchitectureGraph
        graph = ArchitectureGraph(**cached_graph) if cached_graph else None
        react_flow = graph_to_react_flow(graph) if graph else None
        yield {
            "stage": "done",
            "explanation": cached_explanation,
            "graph": react_flow,
            "diagram_available": cached_graph is not None,
            "summary_available": cached_explanation is not None,
            "metadata": metadata,
            "parsed_text": parsed_text,
            "cached": True
        }
        return
    
    # ── Stage 2: Explanation & Equation & Code Generation ──
    yield {"stage": "explaining", "message": f"Extracting mathematical formulations and pedagogical summary for {paper_title[:40]}..."}
    
    if cached_explanation:
        explanation = cached_explanation
    else:
        explanation = explanation_service.generate(full_text, difficulty)
        # Never cache a failed (None) summary — that would replay fabricated/empty data.
        if explanation is not None:
            await cache_service.store_explanation(db, content_hash, difficulty, explanation)
    
    equations = explanation_service.extract_equations(full_text)
    code_snippets = explanation_service.generate_pytorch_code(full_text, paper_title)

    # ── Stage 3: Canonical ArchitectureIR & 3D Scene Graph Extraction ──
    yield {"stage": "diagramming", "message": "Extracting canonical ArchitectureIR & compiling 3D SceneGraph..."}
    
    canonical_ir, scene_graph = ir_extractor.extract(full_text, paper_title)
    bycroft_data = forward_pass_extractor.extract_3d_architecture(full_text, paper_title)
    
    graph = None
    diagram_available = False
    
    if cached_graph is not None:
        from app.models.graph import ArchitectureGraph
        graph = ArchitectureGraph(**cached_graph)
        diagram_available = True
    else:
        # Prefer image-derived topology when an architecture figure is available
        # (the figure shows the real block layout / skip connections), and fall
        # back to the text section otherwise.
        arch_figure = _pick_architecture_figure(figures)
        vision_graph = (
            figure_extractor.extract_architecture(arch_figure, paper_title)
            if arch_figure else None
        )
        text_graph = graph_extractor.extract_graph(arch_text)

        if vision_graph and len(vision_graph.nodes) >= 2:
            graph = vision_graph
            diagram_available = True
            await cache_service.store_graph(
                db, content_hash, graph.model_dump(), True
            )
        elif text_graph and len(text_graph.nodes) > 0:
            # Stage 4: Verification (text path only — image nodes can't cite text)
            verification = graph_extractor.verify_graph(arch_text, text_graph)
            if verification.get("valid", False):
                graph = text_graph
                diagram_available = True
                await cache_service.store_graph(
                    db, content_hash, graph.model_dump(), True
                )
            else:
                # Verification failed — text only
                graph = None
                await cache_service.store_graph(db, content_hash, None, False)
        else:
            # Empty graph — text only
            graph = None
            await cache_service.store_graph(db, content_hash, None, False)
    
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

    best = None
    best_score = 1  # require at least a weak architecture signal
    for fig in figures:
        caption = (getattr(fig, "caption", None) or "").lower()
        score = 0
        if "architecture" in caption:
            score += 3
        if "model" in caption:
            score += 2
        if any(k in caption for k in ("overview", "framework", "network", "pipeline", "diagram")):
            score += 2
        if "figure" in caption or "fig" in caption:
            score += 1
        # Title-page figures are rarely the architecture diagram.
        if getattr(fig, "page", 1) <= 1:
            score -= 1
        if score > best_score:
            best_score = score
            best = fig
    return best
