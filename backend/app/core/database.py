"""Database connection and session management."""

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)
from app.models.orm import Base

log = get_logger(__name__)

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create pgvector extension and all tables.

    Tables are created one by one in savepoints: a table whose DDL cannot
    run here (e.g. paper_chunks needs the pgvector extension, which this
    database image lacks) is skipped with a warning instead of aborting
    every other table. Endpoints already degrade gracefully when a table
    is absent; a missing table must never take the app down.
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    except Exception as e:
        # If dialect doesn't support extension (e.g. SQLite test), ignore
        log.warning(f"vector extension setup skipped: {e}")
    async with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            try:
                async with conn.begin_nested():
                    await conn.run_sync(table.create, checkfirst=True)
            except Exception as e:
                log.warning(
                    f"table {table.name} setup skipped: {type(e).__name__}: {e}"
                )
        # Backfill: create_all does not add columns or constraints to existing
        # tables, so older paper_chunks tables would miss the version columns and
        # the unique constraint (whose absence broke every ON CONFLICT insert).
        # ADD COLUMN IF NOT EXISTS is idempotent across Postgres and SQLite.
        try:
            for col, kind in (
                ("chunker_version", "VARCHAR(16)"),
                ("embedding_model", "VARCHAR(128)"),
                ("embedding_dim", "INTEGER"),
                ("page_number", "INTEGER"),
            ):
                await conn.execute(
                    text(
                        f"ALTER TABLE paper_chunks ADD COLUMN IF NOT EXISTS {col} {kind}"
                    )
                )
            if engine.dialect.name == "postgresql":
                await conn.execute(
                    text(
                        "DELETE FROM paper_chunks a USING "
                        "(SELECT paper_hash, chunk_index, MIN(id::text) AS keep "
                        " FROM paper_chunks GROUP BY paper_hash, chunk_index) k "
                        "WHERE a.paper_hash = k.paper_hash AND a.chunk_index = k.chunk_index "
                        "AND a.id <> k.keep::uuid"
                    )
                )
                await conn.execute(
                    text(
                        "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_constraint "
                        "WHERE conname = 'uq_paper_chunk_hash_index') THEN ALTER TABLE "
                        "paper_chunks ADD CONSTRAINT uq_paper_chunk_hash_index "
                        "UNIQUE (paper_hash, chunk_index); END IF; END $$;"
                    )
                )
        except Exception as e:
            # Fresh DBs have the columns already (create_all); SQLite dialect
            # differences also land here. Retrieval degrades gracefully.
            log.warning(f"paper_chunks backfill skipped: {e}")


async def get_db():
    try:
        session = await async_session().__aenter__()
    except Exception as e:
        log.warning(f"DB connection failed: {e}")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")
    try:
        yield session
    finally:
        try:
            await session.close()
        except Exception:
            pass


async def get_db_optional():
    """Yield a session or None without raising.

    Used by endpoints (e.g. /api/stream) that must run input validation
    even when the database is down. Cache/rag layers already degrade to
    in-memory fallbacks when db is None.
    """
    try:
        session = await async_session().__aenter__()
    except Exception as e:
        log.warning(f"DB connection failed (optional): {e}")
        yield None
        return
    try:
        yield session
    finally:
        try:
            await session.close()
        except Exception:
            pass
