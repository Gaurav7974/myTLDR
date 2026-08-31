"""Interactive Paper Chat RAG endpoint with rate limiting and security."""

from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.config import settings
from app.core.security import limiter, sanitize_arxiv_id, verify_api_key
from app.services.rag_service import rag_service

router = APIRouter()


class ChatRequest(BaseModel):
    paper_hash: str
    paper_title: Optional[str] = "Research Paper"
    query: str
    history: Optional[List[Dict[str, str]]] = None
    difficulty: Optional[str] = "intermediate"


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
    db: AsyncSession = Depends(get_db)
):
    """
    RAG-based Interactive Paper Q&A:
    Retrieves semantic chunks via pgvector and returns grounded LLM answer with verified citations.
    Rate limited by IP or API Key.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Sanitize paper hash/id if formatted like arXiv ID
    clean_hash = req.paper_hash
    if "arxiv" in clean_hash.lower() or "." in clean_hash:
        try:
            clean_hash = sanitize_arxiv_id(clean_hash)
        except Exception:
            clean_hash = req.paper_hash.strip()

    result = await rag_service.answer_question(
        db=db,
        paper_hash=clean_hash,
        paper_title=req.paper_title or "Research Paper",
        query=req.query.strip(),
        history=req.history,
        difficulty=req.difficulty or "intermediate"
    )

    return ChatResponse(
        answer=result["answer"],
        citations=[CitationItem(**c) for c in result.get("citations", [])]
    )
