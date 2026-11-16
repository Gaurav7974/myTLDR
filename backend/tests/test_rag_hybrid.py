"""Phase C: RRF hybrid retrieval behind RAG_HYBRID_ENABLED (default off)."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services import rag as rag_module
from app.services.rag import rag_service


@pytest.fixture(autouse=True)
def _clean():
    rag_module._MEMORY_CHUNK_STORE.clear()
    rag_module._MEMORY_VECTOR_STORE.clear()
    rag_module._INDEX_STATUS.clear()
    yield
    rag_module._MEMORY_CHUNK_STORE.clear()
    rag_module._MEMORY_VECTOR_STORE.clear()
    rag_module._INDEX_STATUS.clear()


def test_rrf_fuse_prefers_agreement():
    a = [{"section_title": "S1", "content": "hello world attention", "similarity": 0.8}]
    b = [
        {"section_title": "S1", "content": "hello world attention", "similarity": 0.7},
        {"section_title": "S2", "content": "other content", "similarity": 0.9},
    ]
    fused = rag_module.rrf_fuse([a, b])
    assert fused
    assert "hello world attention" in fused[0]["content"]
    assert all("rrf" not in e for e in fused)
    assert all(e["similarity"] >= rag_module.RELEVANCE_THRESHOLD for e in fused)


def test_rrf_fuse_empty_inputs():
    assert rag_module.rrf_fuse([]) == []
    assert rag_module.rrf_fuse([[], []]) == []


@pytest.mark.asyncio
async def test_hybrid_off_is_a_pure_fallback():
    rag_module.settings.RAG_HYBRID_ENABLED = False
    rag_module._MEMORY_CHUNK_STORE["h-hybrid-off"] = [
        {
            "section_title": "Methods",
            "content": "residual bottleneck layers " * 20,
            "embedding": None,
        },
        {
            "section_title": "Other",
            "content": "galaxy nebula quasar " * 20,
            "embedding": None,
        },
    ]
    with patch(
        "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
    ):
        out = await rag_service.retrieve_chunks(
            None, "h-hybrid-off", "residual bottleneck", top_k=5
        )
    assert out
    assert all("rrf" not in e for e in out)


@pytest.mark.asyncio
async def test_hybrid_on_cannot_fuse_without_vectors():
    # Hybrid with no query embedding must degrade to BM25, not crash.
    rag_module.settings.RAG_HYBRID_ENABLED = True
    rag_module._MEMORY_CHUNK_STORE["h-hybrid-novec"] = [
        {
            "section_title": "Methods",
            "content": "residual bottleneck layers " * 20,
            "embedding": None,
        },
    ]
    with patch(
        "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
    ):
        out = await rag_service.retrieve_chunks(
            None, "h-hybrid-novec", "residual bottleneck", top_k=5
        )
    assert out
    rag_module.settings.RAG_HYBRID_ENABLED = False


@pytest.mark.asyncio
async def test_hybrid_on_fuses_when_both_modalities_present():
    rag_module.settings.RAG_HYBRID_ENABLED = True
    rag_module._MEMORY_CHUNK_STORE["h-hybrid-on"] = [
        {
            "section_title": "Methods",
            "content": "residual bottleneck layers " * 20,
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "section_title": "Other",
            "content": "galaxy nebula quasar " * 20,
            "embedding": [0.0, 1.0, 0.0],
        },
    ]
    # Query embedding aligns with chunk 0 → vector hits; BM25 also hits chunk 0.
    with patch(
        "app.services.rag.compute_dense_embedding",
        new=AsyncMock(return_value=[1.0, 0.0, 0.0]),
    ):
        out = await rag_service.retrieve_chunks(
            None, "h-hybrid-on", "residual bottleneck", top_k=5
        )
    assert out
    assert "residual" in out[0]["content"].lower()
    rag_module.settings.RAG_HYBRID_ENABLED = False
