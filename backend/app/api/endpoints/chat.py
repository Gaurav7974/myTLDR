"""Interactive Paper Chat RAG endpoint with rate limiting and security."""

from typing import List, Dict, Optional
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


class ChatRequest(BaseModel):
    paper_hash: Optional[str] = None
    paper_id: Optional[str] = None
    paper_title: Optional[str] = "Research Paper"
    query: str
    history: Optional[List[Dict[str, str]]] = None
    difficulty: Optional[str] = "intermediate"

VALID_DIFFICULTIES = {"beginner", "intermediate", "expert"}


class CitationItem(BaseModel):
    section_title: str
    content: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[CitationItem]


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat_with_paper(
    request: Request,
    req: ChatRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    RAG-based Interactive Paper Q&A:
    Retrieves semantic chunks via pgvector/memory and returns grounded LLM answer with verified citations.
    Rate limited by IP or API Key.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if req.difficulty and req.difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {req.difficulty}")

    # Sanitize paper hash/id if formatted like arXiv ID
    raw_id = req.paper_hash or req.paper_id or "default"
    clean_hash = raw_id
    if "arxiv" in clean_hash.lower() or "." in clean_hash:
        try:
            clean_hash = sanitize_arxiv_id(clean_hash)
        except Exception:
            clean_hash = raw_id.strip()

    log.info(f"chat query hash={clean_hash[:8]} q={req.query[:60]}")
    
    # Try using database session if available, else gracefully fallback to memory/direct LLM
    db_session = None
    try:
        from app.core.database import async_session
        async with async_session() as session:
            result = await rag_service.answer_question(
                session,
                clean_hash,
                req.paper_title or "Research Paper",
                req.query.strip(),
                req.history,
                req.difficulty or "intermediate"
            )
    except Exception as db_err:
        log.warning(f"Database session unavailable, using direct LLM synthesis: {db_err}")
        try:
            result = await rag_service.answer_question(
                None,
                clean_hash,
                req.paper_title or "Research Paper",
                req.query.strip(),
                req.history,
                req.difficulty or "intermediate"
            )
        except Exception as e:
            log.error(f"chat failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        answer=result["answer"],
        citations=[CitationItem(**c) for c in result.get("citations", [])]
    )
