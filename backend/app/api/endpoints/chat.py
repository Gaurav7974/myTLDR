"""Interactive Paper Chat RAG endpoint with rate limiting and security."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.core.security import limiter, sanitize_arxiv_id, verify_api_key
from app.core.logger import get_logger
from app.services.rag import rag_service
log = get_logger(__name__)

router = APIRouter()


class HistoryItem(BaseModel):
    """Accepts both {role, content} (standard) and {sender, text} (legacy frontend format)."""
    role: Optional[str] = None
    content: Optional[str] = None
    sender: Optional[str] = None
    text: Optional[str] = None

    def to_llm_turn(self) -> Optional[Dict[str, str]]:
        """Normalize to {role, content} for LLM. Returns None if unrecognizable."""
        if self.role in ("user", "assistant") and self.content:
            return {"role": self.role, "content": self.content}
        # legacy frontend format: {sender: "user"|"ai", text: "..."}
        if self.sender and self.text:
            role = "user" if self.sender == "user" else "assistant"
            return {"role": role, "content": self.text}
        return None


class ChatRequest(BaseModel):
    paper_hash: Optional[str] = None
    paper_id: Optional[str] = None
    paper_title: Optional[str] = "Research Paper"
    query: str
    history: Optional[List[HistoryItem]] = None
    difficulty: Optional[str] = "intermediate"
    # Optional: raw paper text for on-the-fly BM25 fallback when paper_hash has
    # no indexed chunks yet (e.g., first request immediately after processing).
    full_text: Optional[str] = None


VALID_DIFFICULTIES = {"beginner", "intermediate", "expert"}


class CitationItem(BaseModel):
    section_title: str
    content: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    indexing_in_progress: Optional[bool] = None


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat_with_paper(
    request: Request,
    req: ChatRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    RAG-based Interactive Paper Q&A:
    Retrieves semantic chunks via pgvector (primary) or BM25 keyword search (fallback)
    and returns a grounded LLM answer with verified citations.
    Rate limited by IP or API Key.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if req.difficulty and req.difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {req.difficulty}")

    # Sanitize paper hash/id
    raw_id = req.paper_hash or req.paper_id or "default"
    clean_hash = raw_id
    if "arxiv" in clean_hash.lower() or (re.search(r'^\d{4}\.\d{4,5}', clean_hash)):
        # Looks like an arXiv ID — not a content hash; log a warning
        log.warning(f"chat: received arxiv-style ID as paper_hash ({clean_hash[:20]}) — no semantic index will match; using BM25 only")
    elif "." in clean_hash and not clean_hash.startswith("sha"):
        try:
            clean_hash = sanitize_arxiv_id(clean_hash)
        except Exception:
            clean_hash = raw_id.strip()

    # Normalize history to [{role, content}] format
    normalized_history: List[Dict[str, str]] = []
    for h in (req.history or []):
        turn = h.to_llm_turn()
        if turn:
            normalized_history.append(turn)

    log.info(f"chat query hash={clean_hash[:12]} q={req.query[:60]}")

    try:
        from app.core.database import async_session
        async with async_session() as session:
            result = await rag_service.answer_question(
                session,
                clean_hash,
                req.paper_title or "Research Paper",
                req.query.strip(),
                normalized_history,
                req.difficulty or "intermediate",
                full_text=req.full_text,
            )
    except Exception as db_err:
        log.warning(f"Database session unavailable, using BM25/direct LLM: {db_err}")
        try:
            result = await rag_service.answer_question(
                None,
                clean_hash,
                req.paper_title or "Research Paper",
                req.query.strip(),
                normalized_history,
                req.difficulty or "intermediate",
                full_text=req.full_text,
            )
        except Exception as e:
            log.error(f"chat failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        answer=result["answer"],
        citations=[CitationItem(**c) for c in result.get("citations", [])],
        indexing_in_progress=result.get("indexing_in_progress"),
    )


import re
