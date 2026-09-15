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
from app.services.cache import cache_service, compute_content_hash
from app.services.rag import rag_service
from app.services.paper_validator import paper_validator


async def process_paper(
    source_type: str,
    file_bytes: bytes | None,
    url: str | None,
    difficulty: str,
    db: AsyncSession | None
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
        try:
            metadata = await asyncio.to_thread(text_extractor.get_arxiv_metadata, arxiv_id)
        except ValueError as e:
            yield {"stage": "error", "message": str(e)}; return
        metadata["pdf_url"] = f"https://arxiv.org/pdf/{sanitize_arxiv_id(arxiv_id)}.pdf"
        metadata["arxiv_id"] = sanitize_arxiv_id(arxiv_id)
        pdf_bytes = await asyncio.to_thread(text_extractor.download_arxiv_pdf, arxiv_id)
        try: extraction = await asyncio.to_thread(text_extractor.extract_from_pdf_bytes, pdf_bytes)
        except ValueError as e: yield {"stage": "error", "message": str(e)}; return
        except Exception as e: yield {"stage": "error", "message": f"Extraction failed: {str(e)}"}; return
    elif source_type == "url":
        metadata["pdf_url"] = url
        pdf_bytes = await asyncio.to_thread(text_extractor.download_pdf, url)
        try: extraction = await asyncio.to_thread(text_extractor.extract_from_pdf_bytes, pdf_bytes)
        except ValueError as e: yield {"stage": "error", "message": str(e)}; return
        except Exception as e: yield {"stage": "error", "message": f"Extraction failed: {str(e)}"}; return
    elif source_type == "upload":
        if not file_bytes: yield {"stage": "error", "message": "No file provided"}; return
        try: extraction = await asyncio.to_thread(text_extractor.extract_from_pdf_bytes, file_bytes)
        except ValueError as e: yield {"stage": "error", "message": str(e)}; return
        except Exception as e: yield {"stage": "error", "message": f"Extraction failed: {str(e)}"}; return
    else: yield {"stage": "error", "message": f"Unknown source type: {source_type}"}; return
    
    full_text = extraction.text
    figures = extraction.figures
    
    if not full_text or len(full_text.strip()) < 100:
        yield {"stage": "error", "message": "Could not extract meaningful text from the paper"}
        return

    # ── Academic research paper verification (upload gate) ──
    if source_type == "upload":
        is_paper, reject_reason = await asyncio.to_thread(paper_validator.validate_research_paper, full_text)
        if not is_paper:
            log.warning(f"[PIPELINE] Upload rejected as non-academic: {reject_reason}")
            yield {
                "stage": "error",
                "message": f"Uploaded document does not appear to be an academic research paper: {reject_reason}. Please upload a scientific or academic paper (PDF)."
            }
            return
    
    if not metadata.get("title"):
        try: metadata.update(await asyncio.to_thread(text_extractor.extract_paper_metadata_from_text, full_text))
        except Exception: pass
    
    paper_title = metadata.get("title") or "Uploaded Research Paper"

    # ── Content hash & cache check ──
    content_hash = compute_content_hash(full_text)

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
    async def _index_rag_background(hash_key: str, text_key: str):
        from app.core.database import async_session
        try:
            async with async_session() as session:
                await rag_service.index_paper(session, hash_key, text_key)
        except Exception as e:
            log.warning(f"[WARN] Background RAG indexing failed: {e}")

    asyncio.create_task(_index_rag_background(content_hash, full_text))
    
    # Check caches: explanation, graph, and difficulty-independent analysis bundle
    cached_explanation = await cache_service.get_explanation(db, content_hash, difficulty)
    cached_graph = await cache_service.get_graph(db, content_hash)
    cached_analysis = await cache_service.get_analysis(db, content_hash)
    
    if cached_explanation and cached_graph is not None and cached_analysis is not None:
        # Full cache hit
        from app.models.graph import ArchitectureGraph
        from app.models.architecture_ir import ArchitectureIR, ArchitectureMetadata, SceneGraph3D
        graph = ArchitectureGraph(**cached_graph) if cached_graph else None
        if graph is not None and not should_render_reconstruction(graph):
            if graph.layout_mode == "figure":
                graph.layout_mode = "auto"
            if graph.source is None:
                graph.source = "legacy"
        diagram_available = graph is not None and len(graph.nodes) > 0
        graph_payload = graph.model_dump() if graph and diagram_available else None
        
        c_eqs = cached_analysis.get("equations", [])
        c_snippets = cached_analysis.get("code_snippets", [])
        c_note = cached_analysis.get("code_note")
        c_ir_data = cached_analysis.get("canonical_ir") or {}
        c_ir = ArchitectureIR(**c_ir_data) if c_ir_data and "architecture" in c_ir_data else ArchitectureIR(version="1.0", architecture=ArchitectureMetadata(name=paper_title, paradigm="other"), nodes=[], edges=[])

        yield {
            "stage": "done",
            "explanation": cached_explanation,
            "graph": graph_payload,
            "architecture_ir": c_ir.model_dump(),
            "scene_graph_3d": SceneGraph3D(nodes_3d=[], edges_3d=[]).model_dump(),
            "bycroft_stages": [],
            "bycroft_highways": [],
            "attention_heatmap": {"tokens": [], "matrix": []},
            "next_token_predictions": [],
            "equations": c_eqs,
            "code_snippets": c_snippets,
            "code_note": c_note,
            "architecture_nodes": [n.model_dump() for n in c_ir.nodes] if hasattr(c_ir, "nodes") else [],
            "diagram_available": diagram_available,
            "summary_available": cached_explanation is not None,
            "metadata": metadata,
            "parsed_text": parsed_text,
            "content_hash": content_hash,
            "cached": True
        }
        return
    
    # ── Prep phase: arch section detection + figure classification ────────────
    # Both are independent of each other, so run them in parallel FIRST.
    # Having arch_text before the main batch lets equations, IR, and graph
    # extraction use the focused method section (typically 8-15k chars) instead
    # of the full 40k paper — a 3-5x token reduction for those calls with no
    # quality loss since those tasks only care about the architecture section.
    yield {"stage": "explaining", "message": f"Generating explanation & architecture for {paper_title[:40]}..."}
    arch_figure = None
    keyword_pick = None
    need_fig_classify = False

    if cached_graph is None and figures:
        keyword_pick = _pick_architecture_figure(figures)
        if keyword_pick and any(
            k in (getattr(keyword_pick, "caption", "") or "").lower()
            for k in ("architecture", "network architecture", "model architecture", "overall architecture")
        ):
            arch_figure = keyword_pick  # high-confidence keyword match — skip classifier
        else:
            need_fig_classify = True

    prep_tasks: dict = {}
    if cached_graph is None:
        prep_tasks["arch"] = asyncio.to_thread(text_extractor.find_architecture_section, full_text)
    if need_fig_classify:
        prep_tasks["figcls"] = asyncio.to_thread(pick_architecture_figure_classifier, figures, paper_title)

    _prep = (
        dict(zip(prep_tasks.keys(), await asyncio.gather(*prep_tasks.values(), return_exceptions=True)))
        if prep_tasks else {}
    )

    # Unpack: arch section
    arch_text: str | None = None
    arch_section_missing = False
    if "arch" in _prep:
        _ar = _prep["arch"]
        if isinstance(_ar, Exception):
            log.warning(f"[PIPELINE] arch section detection raised: {_ar}")
            arch_section_missing = True
        else:
            arch_text = _ar
            arch_section_missing = arch_text is None
        if arch_section_missing:
            log.warning("[PIPELINE] No architecture section located — text graph extraction skipped")

    # Unpack: figure classifier
    if "figcls" in _prep and arch_figure is None:
        _fr = _prep["figcls"]
        if isinstance(_fr, Exception):
            log.warning(f"[PIPELINE] classifier pick failed: {_fr}")
            arch_figure = keyword_pick
        else:
            arch_figure = _fr or keyword_pick
    elif need_fig_classify and arch_figure is None:
        arch_figure = keyword_pick  # fallback to keyword pick if classifier wasn't run

    # focused_text: arch section when found (10-30x smaller than full paper);
    # fall back to first 20k chars of stripped full_text (methods are in first half).
    focused_text: str = arch_text if arch_text else full_text[:20000]

    # ── Reuse cached analysis if available (e.g. difficulty switch) ───────────
    equations = cached_analysis.get("equations") if cached_analysis else None
    code_snippets = cached_analysis.get("code_snippets") if cached_analysis else None
    code_note = cached_analysis.get("code_note") if cached_analysis else None
    canonical_ir = None
    if cached_analysis and cached_analysis.get("canonical_ir"):
        from app.models.architecture_ir import ArchitectureIR
        try: canonical_ir = ArchitectureIR(**cached_analysis["canonical_ir"])
        except Exception: canonical_ir = None

    # ── Main LLM task batch (all parallel) ────────────────────────────────────
    # expl   → needs broad context: overview + results + conclusion → full_text
    # eq     → equations live in the method section              → focused_text
    # code   → PyTorch module from arch description              → arch_text (preferred)
    # ir     → IR purely from architecture/method section        → focused_text
    # tgraph → graph extraction from arch section                → arch_text (already scoped)
    # vgraph → vision graph from figure image                    → no text
    tasks: dict = {}
    if not cached_explanation:
        tasks["expl"] = asyncio.to_thread(explanation_service.generate, full_text, difficulty)
    if equations is None:
        tasks["eq"] = asyncio.to_thread(explanation_service.extract_equations, focused_text)
    if code_snippets is None:
        tasks["code"] = asyncio.to_thread(explanation_service.generate_pytorch_code, full_text, paper_title, arch_text)
    if canonical_ir is None:
        tasks["ir"] = asyncio.to_thread(ir_extractor.extract, focused_text, paper_title)
    if cached_graph is None and not arch_section_missing and arch_text:
        tasks["tgraph"] = asyncio.to_thread(graph_extractor.extract_graph, arch_text, paper_title)
    if arch_figure:
        tasks["vgraph"] = asyncio.to_thread(figure_extractor.extract_architecture, arch_figure, paper_title)

    results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values(), return_exceptions=True))) if tasks else {}
    def _ok(k): v = results.get(k); return None if isinstance(v, Exception) else v

    explanation = cached_explanation or _ok("expl")
    if explanation and not cached_explanation:
        try: await cache_service.store_explanation(db, content_hash, difficulty, explanation)
        except Exception: pass

    if equations is None:
        equations = _ok("eq") or []
    if code_snippets is None:
        code_res = _ok("code") or {}
        code_snippets, code_note = [], None
        if isinstance(code_res, dict):
            code_snippets = code_res.get("snippets", []) or []
            if not code_res.get("needs_code", True):
                code_note = code_res.get("reason") or "This paper has no implementable architecture to show as code."

    from app.models.architecture_ir import ArchitectureIR, ArchitectureMetadata, SceneGraph3D
    scene_graph = SceneGraph3D(nodes_3d=[], edges_3d=[])
    if canonical_ir is None:
        ir_res = _ok("ir")
        if isinstance(ir_res, tuple):
            canonical_ir, _ = ir_res
        elif isinstance(ir_res, ArchitectureIR):
            canonical_ir = ir_res
    if canonical_ir is None:
        canonical_ir = ArchitectureIR(version="1.0", architecture=ArchitectureMetadata(name=paper_title, paradigm="other"), nodes=[], edges=[])

    # Store analysis bundle in cache for future difficulty switches or instant loads
    if not cached_analysis:
        analysis_payload = {
            "equations": equations,
            "code_snippets": code_snippets,
            "code_note": code_note,
            "canonical_ir": canonical_ir.model_dump(),
        }
        try: await cache_service.store_analysis(db, content_hash, analysis_payload)
        except Exception as e: log.warning(f"[WARN] Failed storing analysis cache: {e}")

    text_graph, vision_graph = _ok("tgraph"), _ok("vgraph")
    # Parallel verification (both independent)
    text_verified = False
    if (text_graph and len(text_graph.nodes) > 0) or (vision_graph and len(getattr(vision_graph, 'nodes', [])) > 0):
        vtasks = {}
        if vision_graph and len(vision_graph.nodes) > 0:
            vtasks["vv"] = asyncio.to_thread(figure_extractor.verify_figure_graph, arch_figure, vision_graph, paper_title)
        if text_graph and len(text_graph.nodes) > 0 and arch_text:
            vtasks["tv"] = asyncio.to_thread(graph_extractor.verify_graph, arch_text, text_graph, paper_title)
        if vtasks:
            vres = dict(zip(vtasks.keys(), await asyncio.gather(*vtasks.values(), return_exceptions=True)))
            if "vv" in vres and not isinstance(vres["vv"], Exception):
                vision_graph.verification = vres["vv"]
            elif "vv" in vres:
                log.warning(f"[WARN] Figure verification failed: {vres['vv']}")
            if "tv" in vres and not isinstance(vres["tv"], Exception):
                text_verified = bool(vres["tv"].get("valid", False))
    yield {"stage": "diagramming", "message": "Finalizing diagrams..."}

    graph = None
    diagram_available = False
    diagram_reason = "not_attempted"

    if cached_graph is not None:
        from app.models.graph import ArchitectureGraph
        graph = ArchitectureGraph(**cached_graph)
        # Only honor the cached graph if it is still a verifiable rich graph.
        if graph.source in ("vision", "merged") and is_verified_vision_graph(graph):
            diagram_available = True
            diagram_reason = "cached_verified_vision"
        elif graph.layout_mode == "figure" and not is_verified_vision_graph(graph):
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
            # Inline-enrich from canonical_ir (zero API latency)
            if canonical_ir and hasattr(canonical_ir, "nodes") and canonical_ir.nodes:
                ir_nodes_by_id = {str(n.id).lower().strip(): n for n in canonical_ir.nodes if n.id}
                ir_nodes_by_label = {str(n.label).lower().strip(): n for n in canonical_ir.nodes if n.label}
                for node in graph.nodes:
                    nid = str(node.id).lower().strip()
                    nlbl = str(node.label).lower().strip()
                    match = ir_nodes_by_id.get(nid) or ir_nodes_by_label.get(nlbl)
                    if not match:
                        for k, v in ir_nodes_by_label.items():
                            if nlbl in k or k in nlbl:
                                match = v
                                break
                    if match:
                        if not node.description and getattr(match, "description", None):
                            node.description = str(match.description)
                        if not node.math_formula and getattr(match, "math_formula", None):
                            node.math_formula = str(match.math_formula)
                        if not getattr(node, "paper_term", None) and getattr(match, "paper_term", None):
                            node.paper_term = str(match.paper_term)
                        if not getattr(node, "parameters", None) and getattr(match, "parameters", None):
                            node.parameters = str(match.parameters)
                        if not getattr(node, "flops", None) and getattr(match, "flops", None):
                            node.flops = str(match.flops)

            # Only call LLM enrichment if nodes still lack descriptions
            if any(not n.description for n in graph.nodes):
                try:
                    graph = await asyncio.to_thread(
                        graph_extractor.enrich_nodes,
                        graph,
                        arch_text or full_text,
                        paper_title,
                    )
                except Exception as e:
                    log.warning(f"[WARN] Dynamic node enrichment skipped: {e}")

            await cache_service.store_graph(
                db, content_hash, graph.model_dump(), True
            )
        else:
            await cache_service.store_graph(db, content_hash, None, False)

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
    graph_payload = graph.model_dump() if graph and diagram_available else None
    
    yield {
        "stage": "done",
        "explanation": explanation,
        "graph": graph_payload,
        "architecture_ir": canonical_ir.model_dump(),
        "scene_graph_3d": scene_graph.model_dump(),
        "bycroft_stages": [],
        "bycroft_highways": [],
        "attention_heatmap": {
            "tokens": [],
            "matrix": []
        },
        "next_token_predictions": [],
        "equations": equations,
        "code_snippets": code_snippets,
        "code_note": code_note,
        "architecture_nodes": [n.model_dump() for n in canonical_ir.nodes] if hasattr(canonical_ir, "nodes") else [],
        "diagram_available": diagram_available,
        "summary_available": explanation is not None,
        "metadata": metadata,
        "parsed_text": parsed_text,
        "content_hash": content_hash,
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
