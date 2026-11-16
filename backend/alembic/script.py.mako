"""Alembic script template."""
from alembic import op
import sqlalchemy as sa

revision = %(revision)s
down_revision = %(down_revision)s
branch_labels = %(branch_labels)r
depends_on = %(depends_on)r


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
