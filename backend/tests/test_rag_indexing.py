"""Phase B5 + Phase D: indexing outcome visibility and stale-index recovery."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import rag as rag_module
from app.services.rag import rag_service


@pytest.fixture(autouse=True)
def _clean():
    rag_module._MEMORY_CHUNK_STORE.clear()
    rag_module._INDEX_STATUS.clear()
    yield
    rag_module._MEMORY_CHUNK_STORE.clear()
    rag_module._INDEX_STATUS.clear()


TEXT = "Abstract. Residual networks learn identity mappings. " * 20


def _res_scalar(value):
    res = MagicMock()
    res.scalar.return_value = value
    return res


def _res_version(version="v1", model=None, dim=None):
    res = MagicMock()
    model = model if model is not None else rag_module.settings.GOOGLE_EMBEDDING_MODEL
    res.first.return_value = (version, model, dim)
    return res


def _fake_db_check_ok(existing: int = 0):
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[
            _res_scalar("id") if existing else _res_scalar(None),
        ]
        + ([_res_version()] if existing else [])
        + ([_res_scalar(existing)] if existing else [])
    )
    return db


@pytest.mark.asyncio
async def test_bm25_only_when_no_embedding_key():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = ""
    db = _fake_db_check_ok(0)
    with patch(
        "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
    ):
        n = await rag_service.index_paper(db, "h-b5-nokey", TEXT)
    assert n > 0
    assert rag_module._INDEX_STATUS["h-b5-nokey"]["status"] == "bm25_only"
    assert rag_module._INDEX_STATUS["h-b5-nokey"]["reason"] == "no_embedding_key"


@pytest.mark.asyncio
async def test_bm25_only_when_embeddings_fail():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    db = _fake_db_check_ok(0)
    with patch(
        "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
    ):
        n = await rag_service.index_paper(db, "h-b5-embfail", TEXT)
    assert n > 0
    assert rag_module._INDEX_STATUS["h-b5-embfail"]["status"] == "bm25_only"
    assert rag_module._INDEX_STATUS["h-b5-embfail"]["reason"] == "embedding_failed"


@pytest.mark.asyncio
async def test_db_unavailable_at_check_marks_bm25_only():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    db = MagicMock()
    db.execute = AsyncMock(side_effect=RuntimeError("db down"))
    n = await rag_service.index_paper(db, "h-b5-dbdown", TEXT)
    assert n > 0
    assert rag_module._INDEX_STATUS["h-b5-dbdown"]["status"] == "bm25_only"
    assert rag_module._INDEX_STATUS["h-b5-dbdown"]["reason"] == "db_check_failed"


@pytest.mark.asyncio
async def test_db_insert_failure_logs_error_and_marks_db_failed(caplog):
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[_res_scalar(None), RuntimeError("insert boom")])
    db.rollback = AsyncMock()
    with patch(
        "app.services.rag.compute_dense_embedding",
        new=AsyncMock(return_value=[0.1] * 3072),
    ):
        with caplog.at_level("ERROR"):
            n = await rag_service.index_paper(db, "h-b5-insertfail", TEXT)
    assert n > 0
    assert rag_module._INDEX_STATUS["h-b5-insertfail"]["status"] == "db_failed"
    assert any("insert failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_already_indexed_marks_indexed(caplog):
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    db = _fake_db_check_ok(10**9)  # more chunks exist than len(chunks)
    with patch(
        "app.services.rag.compute_dense_embedding",
        new=AsyncMock(return_value=[0.1] * 3072),
    ):
        n = await rag_service.index_paper(db, "h-d-already", TEXT)
    assert rag_module._INDEX_STATUS["h-d-already"]["status"] == "indexed"
    assert rag_module._INDEX_STATUS["h-d-already"]["source"] == "already_indexed"


@pytest.mark.asyncio
async def test_full_success_marks_indexed():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    db = MagicMock()
    insert_results = [MagicMock() for _ in range(500)]
    db.execute = AsyncMock(side_effect=[_res_scalar(None)] + insert_results)
    db.commit = AsyncMock()
    with patch(
        "app.services.rag.compute_dense_embedding",
        new=AsyncMock(return_value=[0.1] * 3072),
    ):
        n = await rag_service.index_paper(db, "h-b5-success", TEXT)
    assert rag_module._INDEX_STATUS["h-b5-success"]["status"] == "indexed"
    assert rag_module._INDEX_STATUS["h-b5-success"]["chunks"] == n
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_index_marks_partial_until_resolved():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = ""  # BM25-only path
    db = _fake_db_check_ok(0)  # scalar truthy but zero chunks present
    db.execute = AsyncMock(
        side_effect=[
            _res_scalar("existing-id"),
            _res_version(),  # matching current version
            _res_scalar(0),  # zero chunks present
        ]
    )
    await rag_service.index_paper(db, "h-b5-partial", TEXT)
    assert rag_module._INDEX_STATUS["h-b5-partial"]["status"] in (
        "partial",
        "bm25_only",
    )


@pytest.mark.asyncio
async def test_stale_version_wipes_rows_and_reindexes():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    executed = []

    async def _execute(stmt, *args, **kwargs):
        executed.append(str(stmt))
        if len(executed) == 1:
            return _res_scalar("existing-id")
        if len(executed) == 2:
            return _res_version(version="old-chunker", model="old-model")
        result = MagicMock()
        result.rowcount = 1
        return result

    db = MagicMock()
    db.execute = AsyncMock(side_effect=_execute)
    db.commit = AsyncMock()
    with patch(
        "app.services.rag.compute_dense_embedding",
        new=AsyncMock(return_value=[0.1] * 3072),
    ):
        n = await rag_service.index_paper(db, "h-d-stale", TEXT)

    assert n > 0
    assert any("DELETE" in s.upper() for s in executed), "stale index must be wiped"
    assert rag_module._INDEX_STATUS["h-d-stale"]["status"] == "indexed"


@pytest.mark.asyncio
async def test_current_version_match_does_not_wipe():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = ""
    executed = []

    async def _execute(stmt, *args, **kwargs):
        executed.append(str(stmt))
        if len(executed) == 1:
            return _res_scalar("existing-id")
        if len(executed) == 2:
            return _res_version(
                version=rag_module.settings.CHUNKER_VERSION,
                model=rag_module.settings.GOOGLE_EMBEDDING_MODEL,
                dim=3072,
            )
        return _res_scalar(10**9)  # plenty of chunks

    db = MagicMock()
    db.execute = AsyncMock(side_effect=_execute)
    n = await rag_service.index_paper(db, "h-d-fresh", TEXT)
    assert not any("DELETE" in s.upper() for s in executed)
    assert rag_module._INDEX_STATUS["h-d-fresh"]["source"] == "already_indexed"


def _stale_db(fail_on_insert=False):
    """DB double that reports a stale index; records every SQL statement."""
    executed = []

    async def _execute(stmt, *args, **kwargs):
        executed.append(str(stmt))
        n = len(executed)
        if n == 1:
            return _res_scalar("existing-id")
        if n == 2:
            return _res_version(version="old-chunker", model="old-model")
        if fail_on_insert and n >= 4:
            raise RuntimeError("insert boom")
        result = MagicMock()
        result.rowcount = 1
        return result

    db = MagicMock()
    db.execute = AsyncMock(side_effect=_execute)
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db, executed


@pytest.mark.asyncio
async def test_stale_index_embedding_failure_keeps_old_db_rows():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    db, executed = _stale_db()
    with patch(
        "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
    ):
        n = await rag_service.index_paper(db, "h-stale-embfail", TEXT)

    assert n > 0
    assert not any("DELETE" in s.upper() for s in executed), executed
    db.commit.assert_not_awaited()
    assert rag_module._INDEX_STATUS["h-stale-embfail"]["reason"] == "embedding_failed"


@pytest.mark.asyncio
async def test_no_embedding_key_on_stale_index_does_not_wipe():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = ""
    db, executed = _stale_db()
    with patch(
        "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
    ):
        n = await rag_service.index_paper(db, "h-stale-nokey", TEXT)

    assert n > 0
    assert not any("DELETE" in s.upper() for s in executed), executed
    db.commit.assert_not_awaited()
    assert rag_module._INDEX_STATUS["h-stale-nokey"]["reason"] == "no_embedding_key"


@pytest.mark.asyncio
async def test_stale_index_insert_failure_rolls_back_delete():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    db, executed = _stale_db(fail_on_insert=True)
    with patch(
        "app.services.rag.compute_dense_embedding",
        new=AsyncMock(return_value=[0.1] * 3072),
    ):
        n = await rag_service.index_paper(db, "h-stale-insertfail", TEXT)

    assert n > 0
    assert any("DELETE" in s.upper() for s in executed)
    db.commit.assert_not_awaited(), "delete must not be committed before inserts"
    db.rollback.assert_awaited_once()
    assert rag_module._INDEX_STATUS["h-stale-insertfail"]["status"] == "db_failed"


@pytest.mark.asyncio
async def test_stale_reindex_commits_delete_and_inserts_together():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "fake-key"
    db, executed = _stale_db()

    async def _commit():
        executed.append("COMMIT")

    db.commit = AsyncMock(side_effect=_commit)
    with patch(
        "app.services.rag.compute_dense_embedding",
        new=AsyncMock(return_value=[0.1] * 3072),
    ):
        n = await rag_service.index_paper(db, "h-stale-success", TEXT)

    assert n > 0
    delete_at = next(i for i, s in enumerate(executed) if "DELETE" in s.upper())
    insert_at = next(i for i, s in enumerate(executed) if "INSERT" in s.upper())
    assert delete_at < insert_at
    assert executed.count("COMMIT") == 1, executed
    assert executed[-1] == "COMMIT", executed
    assert rag_module._INDEX_STATUS["h-stale-success"]["status"] == "indexed"
