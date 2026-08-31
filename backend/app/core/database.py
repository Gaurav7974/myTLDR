"""Database connection and session management."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings
from app.models.orm import Base

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create pgvector extension and all tables."""
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception:
            # If dialect doesn't support extension (e.g. SQLite test), ignore
            pass
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Get a database session."""
    async with async_session() as session:
        yield session
