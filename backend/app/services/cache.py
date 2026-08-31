"""Caching layer for graphs and explanations."""

import hashlib
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.orm import PaperCache, GraphCache
from app.services.text_extraction import text_extractor


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = text_extractor.normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


# Fast in-memory caches
_IN_MEMORY_EXPLANATIONS = {}
_IN_MEMORY_GRAPHS = {}


class CacheService:
    """Manages paper cache lookups and writes with database and in-memory fallbacks."""
    
    async def get_graph(self, db: AsyncSession, content_hash: str) -> dict | None:
        """Look up cached graph by content hash."""
        if content_hash in _IN_MEMORY_GRAPHS:
            return _IN_MEMORY_GRAPHS[content_hash]
        try:
            result = await db.execute(
                select(GraphCache).where(GraphCache.content_hash == content_hash)
            )
            cache = result.scalar_one_or_none()
            if cache and cache.graph_json:
                _IN_MEMORY_GRAPHS[content_hash] = cache.graph_json
                return cache.graph_json
        except Exception:
            pass
        return None
    
    async def get_explanation(self, db: AsyncSession, content_hash: str, 
                              difficulty: str) -> dict | None:
        """Look up cached explanation by hash + difficulty."""
        key = f"{content_hash}:{difficulty}"
        if key in _IN_MEMORY_EXPLANATIONS:
            return _IN_MEMORY_EXPLANATIONS[key]
        try:
            result = await db.execute(
                select(PaperCache).where(
                    PaperCache.content_hash == content_hash,
                    PaperCache.difficulty_level == difficulty
                )
            )
            cache = result.scalar_one_or_none()
            if cache:
                _IN_MEMORY_EXPLANATIONS[key] = cache.explanation
                return cache.explanation
        except Exception:
            pass
        return None
    
    async def store_graph(self, db: AsyncSession, content_hash: str,
                          graph_json: dict | None, diagram_available: bool):
        """Store extracted graph in cache."""
        _IN_MEMORY_GRAPHS[content_hash] = graph_json
        try:
            cache = GraphCache(
                content_hash=content_hash,
                graph_json=graph_json,
                diagram_available=diagram_available
            )
            db.add(cache)
            await db.commit()
        except Exception:
            pass
    
    async def store_explanation(self, db: AsyncSession, content_hash: str,
                                difficulty: str, explanation: dict | str):
        """Store generated explanation in cache."""
        key = f"{content_hash}:{difficulty}"
        _IN_MEMORY_EXPLANATIONS[key] = explanation
        try:
            cache = PaperCache(
                content_hash=content_hash,
                difficulty_level=difficulty,
                explanation=explanation if isinstance(explanation, str) else str(explanation),
                diagram_available=False
            )
            db.add(cache)
            await db.commit()
        except Exception:
            pass


cache_service = CacheService()
