"""API routes."""

from fastapi import APIRouter
from app.api.endpoints import submit, stream, chat, feedback

router = APIRouter()
router.include_router(submit.router, tags=["submit"])
router.include_router(stream.router, tags=["stream"])
router.include_router(chat.router, tags=["chat"])
router.include_router(feedback.router, tags=["feedback"])
