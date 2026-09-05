"""Lightweight arXiv identity lookup (no pipeline run).

Lets the frontend show "Found: <title> (arXiv:<id>)" and ask for confirmation
before running the expensive /stream pipeline, so a typo'd ID can never
silently deconstruct the wrong paper.
"""

import asyncio
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from app.core.config import settings
from app.core.security import limiter, sanitize_arxiv_id, verify_api_key
from app.core.logger import get_logger
from app.services.text_extraction import text_extractor

log = get_logger(__name__)

router = APIRouter()


@router.get("/arxiv/lookup")
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def arxiv_lookup(
    request: Request,
    arxiv_id: str = Query(..., min_length=9, max_length=60),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Return canonical arXiv metadata for an ID/URL without deconstructing.

    400s on malformed IDs (via sanitize_arxiv_id) and on unknown/mismatched
    IDs (via get_arxiv_metadata) — the frontend turns those into inline
    "check the ID" errors instead of streaming a wrong paper.
    """
    clean_id = sanitize_arxiv_id(arxiv_id)
    try:
        meta = await asyncio.to_thread(text_extractor.get_arxiv_metadata, clean_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    log.info(f"arxiv lookup ok id={meta.get('arxiv_id_base')}")
    return {
        "arxiv_id": meta.get("arxiv_id_base"),
        "title": meta.get("title"),
        "authors": meta.get("authors", []),
        "abstract": (meta.get("abstract") or "")[:2000],
        "categories": meta.get("categories", []),
    }
