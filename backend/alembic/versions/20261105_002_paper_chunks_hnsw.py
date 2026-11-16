"""paper_chunks vector index — deliberately not created.

Revision ID: rag_hnsw
Revises: rag_version_cols
Create Date: 2026-11-05

pgvector supports approximate indexes (HNSW / IVFFlat) only on vectors with at
most 2000 dimensions. The production embedding (gemini-embedding-001) is 3072
dimensions, so no such index can be built on paper_chunks.embedding. The
retrieval path relies on the existing paper_hash btree index (filters by
paper_hash first) plus a small sequential scan — fine for a few hundred chunks
per paper. If the embedding model is switched to a <=2000-dim one in future,
re-add a vector index here.
"""

from alembic import op

revision = "rag_hnsw"
down_revision = "rag_version_cols"
branch_labels = None
depends_on = None


def upgrade() -> None:
    return  # see module docstring — no 3072-dim index possible in pgvector


def downgrade() -> None:
    return
