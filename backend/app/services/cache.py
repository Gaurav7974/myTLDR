"""Caching layer for graphs and explanations.

The graph payload schema has changed: a ``version`` field plus
``layout_mode`` / ``verification`` are required to safely treat a cached
graph as a figure-faithful reconstruction. Caches that predate this
version are loaded as ``layout_mode='auto'`` (legacy) and never
masquerade as a verified diagram.
"""

import hashlib, logging
from typing import Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.graph import GRAPH_PAYLOAD_VERSION
from app.models.orm import PaperCache, GraphCache, ParsedTextCache
from app.services.text_extraction import text_extractor
from app.core.logger import get_logger
log = get_logger(__name__)


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = text_extractor.normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


# Bounded in-memory caches — Postgres is source of truth
try:
    from cachetools import LRUCache as _LRUCache  # type: ignore[import-untyped]
except ImportError:
    _LRUCache = None  # type: ignore[assignment]

def _make_cache(maxsize: int):
    if _LRUCache is not None:
        return _LRUCache(maxsize=maxsize)
    return {}

_IN_MEMORY_EXPLANATIONS = _make_cache(200)
_IN_MEMORY_GRAPHS = _make_cache(100)
_IN_MEMORY_PARSED_TEXT = _make_cache(100)


def _downgrade_legacy_graph(graph_json: dict | None) -> dict | None:
    """Return a copy of the cached graph with legacy layout_mode.

    Caches written before the figure-faithful schema update are not safe
    to render as a verified reconstruction. We force ``layout_mode='auto'``
    and ``source='legacy'`` so the frontend can still display the graph
    (with auto-layout) but never claims it is figure-faithful.

    The single exception: a payload that already carries the current
    ``GRAPH_PAYLOAD_VERSION`` AND a passing ``_is_rich_figure_graph``
    signature is preserved as-is.
    """
    if not isinstance(graph_json, dict):
        return graph_json
    out = dict(graph_json)
    if _is_rich_figure_graph(out):
        return out
    # Anything else is unsafe to claim as figure-faithful.
    if "version" not in out or out.get("version") != GRAPH_PAYLOAD_VERSION:
        out["version"] = "legacy"
    out["layout_mode"] = "auto"
    if not out.get("source"):
        out["source"] = "legacy"
    return out


def _is_rich_figure_graph(graph_json: dict | None) -> bool:
    """A cached graph is figure-faithful only if it has the current version
    AND explicit verification metadata."""
    if not isinstance(graph_json, dict):
        return False
    if graph_json.get("version") != GRAPH_PAYLOAD_VERSION:
        return False
    if graph_json.get("layout_mode") != "figure":
        return False
    v = graph_json.get("verification") or {}
    if not isinstance(v, dict) or not v.get("valid"):
        return False
    if float(v.get("confidence") or 0.0) < 0.65:
        return False
    if v.get("hallucinations"):
        return False
    return True


class CacheService:
    """Manages paper cache lookups and writes with database and in-memory fallbacks."""

    async def get_graph(self, db: AsyncSession, content_hash: str) -> dict | None:
        """Look up cached graph by content hash.

        Legacy (pre-versioned) payloads are returned with ``layout_mode='auto'``
        and ``source='legacy'`` so the renderer cannot mistake them for
        figure-faithful reconstructions.
        """
        if content_hash in _IN_MEMORY_GRAPHS:
            return _downgrade_legacy_graph(_IN_MEMORY_GRAPHS[content_hash])
        try:
            result = await db.execute(select(GraphCache).where(GraphCache.content_hash == content_hash))
            cache = result.scalar_one_or_none()
            if cache and cache.graph_json:
                _IN_MEMORY_GRAPHS[content_hash] = cache.graph_json
                return _downgrade_legacy_graph(cache.graph_json)
        except Exception:
            try: await db.rollback()
            except Exception: pass
        return None
    async def get_explanation(self, db: AsyncSession, content_hash: str, difficulty: str) -> dict | None:
        key = f"{content_hash}:{difficulty}"
        if key in _IN_MEMORY_EXPLANATIONS: return _IN_MEMORY_EXPLANATIONS[key]
        try:
            result = await db.execute(select(PaperCache).where(PaperCache.content_hash == content_hash, PaperCache.difficulty_level == difficulty))
            cache = result.scalar_one_or_none()
            if cache: _IN_MEMORY_EXPLANATIONS[key] = cache.explanation; return cache.explanation
        except Exception:
            try: await db.rollback()
            except Exception: pass
        return None
    
    async def store_graph(self, db: AsyncSession, content_hash: str, graph_json: dict | None, diagram_available: bool):
        payload = graph_json
        if isinstance(payload, dict):
            payload = dict(payload); payload.setdefault("version", GRAPH_PAYLOAD_VERSION)
            if payload.get("layout_mode") == "figure" and not _is_rich_figure_graph(payload): payload["layout_mode"] = "auto"; payload["source"] = payload.get("source") or "legacy"
        # Try request-scoped session first, then own session with upsert
        try:
            db.add(GraphCache(content_hash=content_hash, graph_json=payload, diagram_available=diagram_available))
            await db.commit()
            _IN_MEMORY_GRAPHS[content_hash] = payload  # only after commit
            return
        except Exception as e:
            try: await db.rollback()
            except Exception: pass
            # Unique violation → upsert path
            log.warning(f"store_graph request session failed: {e}")
        try:
            from app.core.database import async_session
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            async with async_session() as s:
                # Use pg upsert if available, else try add
                try:
                    stmt = pg_insert(GraphCache).values(content_hash=content_hash, graph_json=payload, diagram_available=diagram_available).on_conflict_do_nothing(index_elements=["content_hash"])
                    await s.execute(stmt)
                    await s.commit()
                except Exception:
                    await s.rollback()
                    # fallback for SQLite
                    try:
                        s.add(GraphCache(content_hash=content_hash, graph_json=payload, diagram_available=diagram_available))
                        await s.commit()
                    except Exception:
                        await s.rollback()
                        raise
                _IN_MEMORY_GRAPHS[content_hash] = payload
        except Exception as e:
            log.warning(f"store_graph own-session also failed: {e}")

    async def store_explanation(self, db: AsyncSession, content_hash: str, difficulty: str, explanation: dict | str):
        key = f"{content_hash}:{difficulty}"
        expl_str = explanation if isinstance(explanation, str) else str(explanation)
        try:
            db.add(PaperCache(content_hash=content_hash, difficulty_level=difficulty, explanation=expl_str, diagram_available=False))
            await db.commit()
            _IN_MEMORY_EXPLANATIONS[key] = explanation
            return
        except Exception as e:
            try: await db.rollback()
            except Exception: pass
            log.warning(f"store_explanation request session failed: {e}")
        try:
            from app.core.database import async_session
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            async with async_session() as s:
                try:
                    stmt = pg_insert(PaperCache).values(content_hash=content_hash, difficulty_level=difficulty, explanation=expl_str, diagram_available=False).on_conflict_do_nothing(index_elements=["content_hash", "difficulty_level"])
                    await s.execute(stmt)
                    await s.commit()
                except Exception:
                    await s.rollback()
                    try:
                        s.add(PaperCache(content_hash=content_hash, difficulty_level=difficulty, explanation=expl_str, diagram_available=False))
                        await s.commit()
                    except Exception:
                        await s.rollback()
                        raise
                _IN_MEMORY_EXPLANATIONS[key] = explanation
        except Exception as e:
            log.warning(f"store_explanation own-session also failed: {e}")

    async def get_parsed_text(self, db: AsyncSession, content_hash: str) -> dict | None:
        """Look up cached parsed text (per-page structure) by content hash."""
        if content_hash in _IN_MEMORY_PARSED_TEXT:
            return _IN_MEMORY_PARSED_TEXT[content_hash]
        try:
            result = await db.execute(select(ParsedTextCache).where(ParsedTextCache.content_hash == content_hash))
            cache = result.scalar_one_or_none()
            if cache and cache.parsed_json: _IN_MEMORY_PARSED_TEXT[content_hash] = cache.parsed_json; return cache.parsed_json
        except Exception:
            try: await db.rollback()
            except Exception: pass
        return None

    async def store_parsed_text(self, db: AsyncSession, content_hash: str, parsed: dict):
        """Store parsed text structure in cache (skips payloads over ~16MB)."""
        import json
        if len(json.dumps(parsed, default=str)) > 16 * 1024 * 1024:
            log.warning("[WARN] Parsed text payload too large to cache; skipping")
            return
        try:
            db.add(ParsedTextCache(content_hash=content_hash, parsed_json=parsed))
            await db.commit()
            _IN_MEMORY_PARSED_TEXT[content_hash] = parsed
            return
        except Exception as e:
            try: await db.rollback()
            except Exception: pass
            log.warning(f"[WARN] Failed to cache parsed text: {e}")
        try:
            from app.core.database import async_session
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            async with async_session() as s:
                try:
                    stmt = pg_insert(ParsedTextCache).values(content_hash=content_hash, parsed_json=parsed).on_conflict_do_nothing(index_elements=["content_hash"])
                    await s.execute(stmt)
                    await s.commit()
                except Exception:
                    await s.rollback()
                    try:
                        s.add(ParsedTextCache(content_hash=content_hash, parsed_json=parsed))
                        await s.commit()
                    except Exception:
                        await s.rollback()
                        raise
                _IN_MEMORY_PARSED_TEXT[content_hash] = parsed
        except Exception as e: log.warning(f"[WARN] Own-session cache also failed: {e}")


cache_service = CacheService()
