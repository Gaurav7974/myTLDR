"""paper_chunks: dedupe legacy rows + add unique constraint (paper_hash, chunk_index)

Revision ID: rag_unique_hash_index
Revises: rag_hnsw
Create Date: 2026-11-05

The ORM declares __table_args__ UniqueConstraint("paper_hash", "chunk_index"),
but create_all never adds table_args to a pre-existing table, so legacy
databases lack the constraint. Production inserts rely on
ON CONFLICT (paper_hash, chunk_index) and fail without it ("there is no unique
or exclusion constraint matching the ON CONFLICT specification"), leaving
pgvector permanently empty.

This migration dedupes potential legacy duplicates (keeping the lowest id per
(paper_hash, chunk_index)) and adds the constraint if it is missing.
Postgres/pgvector only; no-op on other dialects (create_all handles fresh
databases).
"""

from alembic import op

revision = "rag_unique_hash_index"
down_revision = "rag_hnsw"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(
        """
        DELETE FROM paper_chunks a
        USING (
            SELECT paper_hash, chunk_index, MIN(id::text) AS keep
            FROM paper_chunks
            GROUP BY paper_hash, chunk_index
        ) k
        WHERE a.paper_hash = k.paper_hash
          AND a.chunk_index = k.chunk_index
          AND a.id <> k.keep::uuid
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'uq_paper_chunk_hash_index'
            ) THEN
                ALTER TABLE paper_chunks
                ADD CONSTRAINT uq_paper_chunk_hash_index UNIQUE (paper_hash, chunk_index);
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(
        "ALTER TABLE paper_chunks DROP CONSTRAINT IF EXISTS uq_paper_chunk_hash_index"
    )
