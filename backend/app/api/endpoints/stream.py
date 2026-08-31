"""SSE streaming endpoint with rate limiting and security."""

import json
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.core.security import limiter, sanitize_arxiv_id, validate_and_sanitize_url, verify_api_key
from app.services.pipeline import process_paper

router = APIRouter()


@router.post("/stream")
@limiter.limit(settings.RATE_LIMIT_STREAM)
async def stream_paper(
    request: Request,
    source: str = Form(...),
    difficulty_level: str = Form("intermediate"),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    api_key: Optional[str] = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """
    Stream paper processing results via SSE.
    Rate limited by IP/API Key, protected against SSRF & malicious input.
    """
    # Sanitize inputs
    sanitized_url = url
    if source == "arxiv" and url:
        sanitized_url = sanitize_arxiv_id(url)
    elif source == "url" and url:
        sanitized_url = validate_and_sanitize_url(url)

    file_bytes = None
    if file:
        file_bytes = await file.read()
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)"
            )
    
    async def event_generator():
        async for event in process_paper(
            source_type=source,
            file_bytes=file_bytes,
            url=sanitized_url,
            difficulty=difficulty_level,
            db=db
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
