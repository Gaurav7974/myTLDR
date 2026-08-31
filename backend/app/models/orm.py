"""Database models."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None

Base = declarative_base()


class PaperCache(Base):
    """Caches extracted graphs and explanations."""
    __tablename__ = "paper_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String, nullable=False, index=True)
    difficulty_level = Column(String, nullable=False)
    explanation = Column(Text, nullable=False)
    graph_json = Column(JSONB, nullable=True)  # NULL if diagram unavailable
    diagram_available = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        {"sqlite_autoincrement": True},
    )


class GraphCache(Base):
    """Caches extracted graphs separately (difficulty-independent)."""
    __tablename__ = "graph_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String, nullable=False, unique=True, index=True)
    graph_json = Column(JSONB, nullable=True)
    diagram_available = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class PaperChunk(Base):
    """Stores paper semantic chunks with pgvector embeddings for RAG Q&A."""
    __tablename__ = "paper_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_hash = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    section_title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    # 384-dimensional dense vector embeddings (e.g. BAAI/bge-small-en-v1.5 / all-MiniLM-L6-v2)
    embedding = Column(Vector(384) if Vector is not None else Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
