"""SSE streaming endpoint with rate limiting and security."""

import asyncio
import json
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.core.security import limiter, sanitize_arxiv_id, validate_and_sanitize_url, verify_api_key
from app.core.logger import get_logger
from app.services.pipeline import process_paper
log = get_logger(__name__)

VALID_DIFFICULTIES = {"beginner", "intermediate", "expert"}

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
    if difficulty_level not in VALID_DIFFICULTIES:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty_level: {difficulty_level}")
    if source not in ("upload", "url", "arxiv"):
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

    # Sanitize inputs — DNS validation offloaded to threadpool to avoid blocking loop
    sanitized_url = url
    if source == "arxiv" and url:
        sanitized_url = sanitize_arxiv_id(url)
    elif source == "url" and url:
        if "arxiv.org" not in url.lower():
            raise HTTPException(
                status_code=400,
                detail="Invalid URL: Only arXiv papers are supported (e.g. https://arxiv.org/abs/1706.03762). Please try again."
            )
        sanitized_url = await asyncio.to_thread(validate_and_sanitize_url, url)

    # Stream file with incremental cap (avoid loading 1GB into memory)
    file_bytes = None
    if source == "upload":
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type: Only PDF documents (.pdf) are supported.")
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        chunks = []
        total = 0
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)")
            chunks.append(chunk)
        file_bytes = b"".join(chunks) if chunks else None
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
            log.warning(f"Could not inspect PDF page count in stream: {e}")
    
    log.info(f"stream start source={source} difficulty={difficulty_level} url={sanitized_url} file={bool(file_bytes)}")

    # Heartbeat wrapper: pipeline has 60-180s gaps with zero bytes
    # (LLM gather in pipeline.py) during which Railway/HTTP-2 kills the
    # idle stream -> ERR_HTTP2_PROTOCOL_ERROR. Emit `: ping` every 15s
    # so the connection never idles. Frontend already skips non-data lines.
    async def event_generator():
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _produce() -> None:
            try:
                async for event in process_paper(source_type=source, file_bytes=file_bytes, url=sanitized_url, difficulty=difficulty_level, db=db):
                    log.info(f"sse stage={event.get('stage')} diagram={event.get('diagram_available')}")
                    await queue.put(f"data: {json.dumps(event)}\n\n")
                await queue.put("data: [DONE]\n\n")
                log.info("stream done")
            except Exception as e:
                log.error(f"stream pipeline error: {e}", exc_info=True)
                # never leak internal details to client, but surface clean validation messages
                msg = str(e) if isinstance(e, ValueError) else 'Processing failed. Please retry.'
                await queue.put(f"data: {json.dumps({'stage': 'error', 'message': msg})}\n\n")
            finally:
                await queue.put(None)

        producer = asyncio.create_task(_produce())
        try:
            # Open the stream immediately so proxies start forwarding.
            yield ": connected\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                yield item
        finally:
            if not producer.done():
                producer.cancel()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
