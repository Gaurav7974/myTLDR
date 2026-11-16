"""Database models."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Text, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None

Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc)


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
    # Version metadata: needed to detect stale indexes after chunker/model changes.
    # NULL on legacy rows (predates migration) — treated as needing re-index, not as current.
    chunker_version = Column(String(16), nullable=True)
    embedding_model = Column(String(128), nullable=True)
    embedding_dim = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        UniqueConstraint("paper_hash", "chunk_index", name="uq_paper_chunk_hash_index"),
    )
