"""DATABASE_URL normalization for libpq-style URLs (Altare)."""

from app.core.config import _normalize_database_url


def test_postgres_scheme_becomes_asyncpg():
    assert _normalize_database_url("postgres://u:p@host/papers").startswith(
        "postgresql+asyncpg://"
    )


def test_plain_postgresql_becomes_asyncpg():
    assert _normalize_database_url("postgresql://u:p@host/papers").startswith(
        "postgresql+asyncpg://"
    )


def test_sslmode_becomes_ssl():
    out = _normalize_database_url("postgresql://u:p@host:5432/papers?sslmode=require")
    assert out == "postgresql+asyncpg://u:p@host:5432/papers?ssl=require"


def test_asyncpg_url_keeps_dialect():
    assert _normalize_database_url("postgresql+asyncpg://u:p@host/papers") == (
        "postgresql+asyncpg://u:p@host/papers"
    )


def test_other_schemes_untouched():
    assert (
        _normalize_database_url("sqlite+aiosqlite:///./t.db")
        == "sqlite+aiosqlite:///./t.db"
    )
