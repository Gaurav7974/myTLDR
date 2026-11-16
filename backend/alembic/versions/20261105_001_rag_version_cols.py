"""rag chunk versioning columns

Revision ID: rag_version_cols
Revises:
Create Date: 2026-11-05

Adds nullable version metadata columns to paper_chunks so stale indexes can be
detected after a chunker or embedding-model change. Idempotent: safe on fresh
databases where create_all already created the columns.
"""

from alembic import op
import sqlalchemy as sa

revision = "rag_version_cols"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("paper_chunks") as batch_op:
        batch_op.add_column(sa.Column("chunker_version", sa.String(16), nullable=True))
        batch_op.add_column(sa.Column("embedding_model", sa.String(128), nullable=True))
        batch_op.add_column(sa.Column("embedding_dim", sa.Integer, nullable=True))
        batch_op.add_column(sa.Column("page_number", sa.Integer, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("paper_chunks") as batch_op:
        batch_op.drop_column("chunker_version")
        batch_op.drop_column("embedding_model")
        batch_op.drop_column("embedding_dim")
        batch_op.drop_column("page_number")
