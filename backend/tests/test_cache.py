"""Cache integration tests (Todo 6 + 12).

Verifies the in-process cache round-trip honors the trust-first policy:
- store_graph refuses to persist a layout_mode='figure' payload that
  fails _is_rich_figure_graph.
- get_graph returns downgraded legacy payloads.
"""
import pytest

from app.services import cache
from app.models.graph import GRAPH_PAYLOAD_VERSION, GraphVerification


class _FakeDB:
    """Minimal stand-in for AsyncSession used by cache.store_graph."""
    def __init__(self):
        self.added = []
    def add(self, obj):
        self.added.append(obj)
    async def execute(self, *args, **kwargs):
        from unittest.mock import MagicMock
        return MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    async def commit(self):
        pass


@pytest.mark.asyncio
async def test_store_graph_persists_valid_figure_payload():
    db = _FakeDB()
    payload = {
        "version": GRAPH_PAYLOAD_VERSION, "source": "vision", "layout_mode": "figure",
        "nodes": [], "edges": [],
        "verification": {"valid": True, "confidence": 0.9, "hallucinations": []},
    }
    await cache.cache_service.store_graph(db, "h1", payload, True)
    stored = cache._IN_MEMORY_GRAPHS["h1"]
    assert stored["layout_mode"] == "figure"
    assert stored["version"] == GRAPH_PAYLOAD_VERSION


@pytest.mark.asyncio
async def test_store_graph_downgrades_unverifiable_figure_payload():
    db = _FakeDB()
    payload = {
        "version": GRAPH_PAYLOAD_VERSION, "source": "vision", "layout_mode": "figure",
        "nodes": [], "edges": [],
        "verification": {"valid": True, "confidence": 0.9, "hallucinations": ["x"]},
    }
    await cache.cache_service.store_graph(db, "h2", payload, True)
    stored = cache._IN_MEMORY_GRAPHS["h2"]
    assert stored["layout_mode"] == "auto", "hallucinated figure payload must be downgraded"


@pytest.mark.asyncio
async def test_store_graph_stamps_version_if_missing():
    db = _FakeDB()
    payload = {"nodes": [], "edges": []}  # no version
    await cache.cache_service.store_graph(db, "h3", payload, False)
    stored = cache._IN_MEMORY_GRAPHS["h3"]
    assert stored["version"] == GRAPH_PAYLOAD_VERSION


@pytest.mark.asyncio
async def test_get_graph_downgrades_legacy_payload():
    """An in-memory legacy cache entry (no version, no verification)
    is returned as layout_mode='auto' even if the stored value was
    different."""
    cache._IN_MEMORY_GRAPHS["legacy1"] = {
        "nodes": [{"id": "a", "label": "A", "type": "attention"}],
        "edges": [],
    }
    db = _FakeDB()
    got = await cache.cache_service.get_graph(db, "legacy1")
    assert got["layout_mode"] == "auto"
    assert got["source"] == "legacy"
    assert got["version"] == "legacy"


@pytest.mark.asyncio
async def test_get_graph_returns_valid_rich_payload_unchanged():
    valid = {
        "version": GRAPH_PAYLOAD_VERSION, "source": "vision", "layout_mode": "figure",
        "nodes": [], "edges": [],
        "verification": {"valid": True, "confidence": 0.92, "hallucinations": []},
    }
    cache._IN_MEMORY_GRAPHS["v1"] = valid
    db = _FakeDB()
    got = await cache.cache_service.get_graph(db, "v1")
    assert got["layout_mode"] == "figure"


@pytest.mark.asyncio
async def test_get_graph_handles_non_dict():
    cache._IN_MEMORY_GRAPHS["weird"] = "not a dict"
    db = _FakeDB()
    got = await cache.cache_service.get_graph(db, "weird")
    assert got == "not a dict"  # returned as-is, no crash


@pytest.mark.asyncio
async def test_get_graph_returns_none_when_missing():
    db = _FakeDB()
    got = await cache.cache_service.get_graph(db, "nope")
    assert got is None
