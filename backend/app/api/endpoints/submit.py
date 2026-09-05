"""Submit endpoint with rate limiting and security."""

import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.core.security import limiter, sanitize_arxiv_id, validate_and_sanitize_url, verify_api_key
from app.core.logger import get_logger
log = get_logger(__name__)

VALID_DIFFICULTIES = {"beginner", "intermediate", "expert"}

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
    if difficulty_level not in VALID_DIFFICULTIES:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty_level: {difficulty_level}")

    if source == "upload":
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type: Only PDF documents (.pdf) are supported.")
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        total = 0
        chunks = []
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)")
            chunks.append(chunk)

        file_bytes = b"".join(chunks)
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if not file_bytes.lstrip().startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF (missing %PDF header)")
        try:
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            p_count = len(doc)
            doc.close()
            if p_count > settings.MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Paper has {p_count} pages, which exceeds the {settings.MAX_PDF_PAGES}-page limit. Please provide a paper with {settings.MAX_PDF_PAGES} pages or fewer."
                )
        except HTTPException:
            raise
        except Exception as e:
            log.warning(f"Could not inspect PDF page count in submit: {e}")

        return {"status": "received", "source": "upload", "filename": file.filename}

    elif source == "arxiv":
        if not url:
            raise HTTPException(status_code=400, detail="No arXiv ID or URL provided")
        clean_arxiv_id = sanitize_arxiv_id(url)
        return {"status": "received", "source": "arxiv", "arxiv_id": clean_arxiv_id}

    elif source == "url":
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")
        if "arxiv.org" not in url.lower():
            raise HTTPException(
                status_code=400,
                detail="Invalid URL: Only arXiv papers are supported (e.g. https://arxiv.org/abs/1706.03762). Please try again."
            )
        safe_url = await asyncio.to_thread(validate_and_sanitize_url, url)
        return {"status": "received", "source": "url", "url": safe_url}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
