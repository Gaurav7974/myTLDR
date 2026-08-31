"""Submit endpoint with rate limiting and security."""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.core.security import limiter, sanitize_arxiv_id, validate_and_sanitize_url, verify_api_key

router = APIRouter()


@router.post("/submit")
@limiter.limit(settings.RATE_LIMIT_SUBMIT)
async def submit_paper(
    request: Request,
    source: str = Form(...),  # "upload" | "url" | "arxiv"
    difficulty_level: str = Form("intermediate"),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    api_key: Optional[str] = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Submit a paper for validation and processing."""
    
    if source == "upload":
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        contents = await file.read()
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)"
            )
        
        return {"status": "received", "source": "upload", "filename": file.filename}
    
    elif source == "arxiv":
        if not url:
            raise HTTPException(status_code=400, detail="No arXiv ID or URL provided")
        clean_arxiv_id = sanitize_arxiv_id(url)
        return {"status": "received", "source": "arxiv", "arxiv_id": clean_arxiv_id}

    elif source == "url":
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")
        safe_url = validate_and_sanitize_url(url)
        return {"status": "received", "source": "url", "url": safe_url}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
