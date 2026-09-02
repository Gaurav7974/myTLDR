"""Database models."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None

Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc)


class PaperCache(Base):
    """Caches extracted graphs and explanations."""
    __tablename__ = "paper_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String, nullable=False, index=True)
    difficulty_level = Column(String, nullable=False)
    explanation = Column(Text, nullable=False)
    graph_json = Column(JSONB, nullable=True)  # NULL if diagram unavailable
    diagram_available = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        UniqueConstraint("content_hash", "difficulty_level", name="uq_paper_cache_hash_difficulty"),
        {"sqlite_autoincrement": True},
    )


class GraphCache(Base):
    """Caches extracted graphs separately (difficulty-independent)."""
    __tablename__ = "graph_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String, nullable=False, unique=True, index=True)
    graph_json = Column(JSONB, nullable=True)
    diagram_available = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class ParsedTextCache(Base):
    """Caches per-page parsed text (difficulty-independent) keyed by content hash."""
    __tablename__ = "parsed_text_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String, nullable=False, unique=True, index=True)
    parsed_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class PaperChunk(Base):
    """Stores paper semantic chunks with pgvector embeddings for RAG Q&A."""
    __tablename__ = "paper_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_hash = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    section_title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    # 3072-dimensional dense vector embeddings (Google gemini-embedding-001)
    embedding = Column(Vector(3072) if Vector is not None else Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        UniqueConstraint("paper_hash", "chunk_index", name="uq_paper_chunk_hash_index"),
    )


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
