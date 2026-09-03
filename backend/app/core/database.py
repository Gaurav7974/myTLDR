"""Database connection and session management."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings
from app.core.logger import get_logger
from app.models.orm import Base
log = get_logger(__name__)

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create pgvector extension and all tables."""
    # Extension first, in its own transaction: if it fails (e.g. no DDL
    # permission), the abort must not poison table creation below.
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    except Exception as e:
        # If dialect doesn't support extension (e.g. SQLite test), ignore
        log.warning(f"vector extension setup skipped: {e}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            try:
                await session.rollback()
            except Exception:
                pass
            log.error(f"DB session error: {e}", exc_info=True)
            raise
        finally:
            try:
                await session.close()
            except Exception:
                pass
