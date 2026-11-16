"""API routes."""

from fastapi import APIRouter, Depends
from app.api.endpoints import submit, stream, chat, arxiv
from app.core.security import verify_api_key

# One enforcement point: a new endpoint cannot forget auth.
router = APIRouter(dependencies=[Depends(verify_api_key)])
router.include_router(submit.router, tags=["submit"])
router.include_router(stream.router, tags=["stream"])
router.include_router(chat.router, tags=["chat"])
router.include_router(arxiv.router, tags=["arxiv"])
