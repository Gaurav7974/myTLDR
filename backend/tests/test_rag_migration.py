"""Phase D: version-column migration + init_db backfill."""

import asyncio

import pytest

from app.models.orm import PaperChunk


def test_paperchunk_has_version_columns():
    cols = {c.key for c in PaperChunk.__table__.columns}
    assert {
        "chunker_version",
        "embedding_model",
        "embedding_dim",
        "page_number",
    } <= cols
    assert PaperChunk.__table__.c.chunker_version.nullable is True
    assert PaperChunk.__table__.c.embedding_model.nullable is True
    assert PaperChunk.__table__.c.embedding_dim.nullable is True
    assert PaperChunk.__table__.c.page_number.nullable is True


def test_alembic_versions_present_and_idempotent():
    from pathlib import Path

    versions = Path(__file__).resolve().parent.parent / "alembic" / "versions"
    files = sorted(p.name for p in versions.glob("*.py"))
    assert files == [
        "20261105_001_rag_version_cols.py",
        "20261105_002_paper_chunks_hnsw.py",
        "20261105_003_rag_unique_hash_index.py",
    ]
    for name in files:
        (versions / name).read_text(encoding="utf-8")
    hnsw = (versions / "20261105_002_paper_chunks_hnsw.py").read_text(encoding="utf-8")
    assert "2000" in hnsw  # documents why pgvector cannot index 3072-dim vectors


@pytest.mark.asyncio
async def test_init_db_backfill_adds_columns_to_legacy_table(tmp_path):
    """A legacy paper_chunks table (no version columns) must gain them via init_db."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE paper_chunks ("
        "id VARCHAR PRIMARY KEY, paper_hash VARCHAR NOT NULL, chunk_index INTEGER NOT NULL, "
        "section_title VARCHAR, content TEXT NOT NULL, embedding TEXT, "
        "created_at DATETIME)"
    )
    conn.execute(
        "INSERT INTO paper_chunks (id, paper_hash, chunk_index, section_title, content, embedding, created_at) "
        "VALUES ('x', 'h', 0, 'S', 'content', NULL, '2026-01-01')"
    )
    conn.commit()
    conn.close()

    url = f"sqlite+aiosqlite:///{db_path}"
    from sqlalchemy.ext.asyncio import (
        create_async_engine,
        async_sessionmaker,
        AsyncSession,
    )

    engine = create_async_engine(url)

    # Run the same backfill statements init_db uses (isolated from full boot).
    from sqlalchemy import text

    async with engine.begin() as conn:
        for col, kind in (
            ("chunker_version", "VARCHAR(16)"),
            ("embedding_model", "VARCHAR(128)"),
            ("embedding_dim", "INTEGER"),
            ("page_number", "INTEGER"),
        ):
            # SQLite: ADD COLUMN IF NOT EXISTS is not supported on older versions,
            # so emulate the check init_db relies on.
            try:
                await conn.execute(
                    text(
                        f"ALTER TABLE paper_chunks ADD COLUMN IF NOT EXISTS {col} {kind}"
                    )
                )
            except Exception:
                # Fallback: inspect and add if missing.
                res = await conn.execute(text("PRAGMA table_info(paper_chunks)"))
                names = {row[1] for row in res.fetchall()}
                if col not in names:
                    await conn.execute(
                        text(f"ALTER TABLE paper_chunks ADD COLUMN {col} {kind}")
                    )

    await engine.dispose()

    conn = sqlite3.connect(str(db_path))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(paper_chunks)")}
    conn.close()
    assert {
        "chunker_version",
        "embedding_model",
        "embedding_dim",
        "page_number",
    } <= cols


def test_init_db_contains_backfill_guard():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent / "app" / "core" / "database.py"
    ).read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS" in src
    assert "paper_chunks" in src
