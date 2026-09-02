"""Feedback wall persistence — 15-word limit, no auth."""
from typing import List
from pydantic import BaseModel, field_validator
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.core.database import get_db
from app.core.logger import get_logger
from app.models.orm import Feedback

log = get_logger(__name__)
router = APIRouter()

class FeedbackIn(BaseModel):
    content: str
    @field_validator("content")
    @classmethod
    def check(cls, v: str) -> str:
        t = v.strip()
        if not t: raise ValueError("empty")
        if len(t.split()) > 15: raise ValueError("max 15 words")
        if len(t) > 200: raise ValueError("too long")
        return t

class FeedbackOut(BaseModel):
    id: str
    content: str
    created_at: str
    class Config: from_attributes = True

@router.get("/feedback", response_model=List[FeedbackOut])
async def list_feedback(db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Feedback).order_by(desc(Feedback.created_at)).limit(100))
    rows = res.scalars().all()
    log.info(f"list feedback {len(rows)}")
    return [{"id": str(r.id), "content": r.content, "created_at": r.created_at.isoformat() if r.created_at else ""} for r in rows]

@router.post("/feedback", response_model=FeedbackOut)
async def create_feedback(body: FeedbackIn, db: AsyncSession = Depends(get_db)):
    fb = Feedback(content=body.content.strip())
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    log.info(f"create feedback {fb.id} {body.content[:30]}")
    return {"id": str(fb.id), "content": fb.content, "created_at": fb.created_at.isoformat() if fb.created_at else ""}
