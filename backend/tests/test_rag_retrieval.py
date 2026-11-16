"""Retrieval-generation path tests added alongside Phase B fixes (RAG consolidation)."""

import threading
from unittest.mock import AsyncMock, patch

import pytest

from app.services import rag as rag_module
from app.services.rag import rag_service


@pytest.fixture(autouse=True)
def _clean_memory():
    rag_module._MEMORY_CHUNK_STORE.clear()
    yield
    rag_module._MEMORY_CHUNK_STORE.clear()


def test_stem_converges_morphological_variants():
    """embeddings/embedding must produce the same token (B3 fix)."""
    assert rag_module._stem("embeddings") == rag_module._stem("embedding")
    assert (
        rag_module._stem("attends")
        == rag_module._stem("attending")
        == rag_module._stem("attended")
    )
    # unmatched trailing s preserved for double-s forms
    assert rag_module._stem("loss") == "loss"


def test_bm25_matches_morphological_variant_document():
    """Query 'embedding dimension' must match chunks containing 'embeddings'."""
    chunks = [
        {
            "section_title": "Model",
            "content": "The embedding layer maps tokens to vectors of size 512.",
        },
        {
            "section_title": "Background",
            "content": "galaxy nebula quasar pulsar comet asteroid orbit telescope spectrum "
            * 40,
        },
    ]
    out = rag_module.bm25_keyword_search(chunks, "embeddings dimension size")
    assert out
    assert "embedding layer" in out[0]["content"]


@pytest.mark.asyncio
async def test_llm_called_off_event_loop():
    """The sync requests-based LLM client must not run in the event-loop thread."""
    target = "residual bottleneck convolutional layers batch normalization gamma beta "
    filler = "galaxy nebula quasar pulsar comet asteroid orbit telescope spectrum "
    text = target * 10 + (filler * 12 + " ") * 5 + "References. Doe 2020."

    captured = {}

    class _Client:
        def call(self, messages, model, temperature):
            captured["thread"] = threading.get_ident()
            return "ok"

    with (
        patch(
            "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
        ),
        patch.object(rag_module, "groq_client", _Client()),
    ):
        result = await rag_service.answer_question(
            None,
            "h-async-loop",
            "Demo",
            "residual bottleneck convolutional layers batch normalization?",
            [],
            "intermediate",
            full_text=text,
        )

    assert result["answer"] == "ok"
    assert captured["thread"] != threading.main_thread().ident


@pytest.mark.asyncio
async def test_llm_exception_returns_fallback_message_and_preserves_citations():
    target = "residual bottleneck convolutional layers batch normalization gamma beta "
    filler = "galaxy nebula quasar pulsar comet asteroid orbit telescope spectrum "
    text = target * 10 + (filler * 12 + " ") * 5 + "References. Doe 2020."

    class _Client:
        def call(self, messages, model, temperature):
            raise RuntimeError("provider down")

    with (
        patch(
            "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
        ),
        patch.object(rag_module, "groq_client", _Client()),
    ):
        result = await rag_service.answer_question(
            None,
            "h-async-fail",
            "Demo",
            "residual bottleneck convolutional layers batch normalization?",
            [],
            "intermediate",
            full_text=text,
        )

    assert "temporarily unavailable" in result["answer"]
    assert result["citations"]  # citations survive the LLM outage


@pytest.mark.asyncio
async def test_llm_receives_at_most_six_history_turns():
    """Long conversation history must not grow unbounded (grounding protection)."""
    target = "residual bottleneck convolutional layers batch normalization gamma beta "
    filler = "galaxy nebula quasar pulsar comet asteroid orbit telescope spectrum "
    text = target * 10 + (filler * 12 + " ") * 5 + "References. Doe 2020."

    captured = {}

    class _Client:
        def call(self, messages, model, temperature):
            captured["count"] = len(messages)
            return "ok"

    history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    with (
        patch(
            "app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)
        ),
        patch.object(rag_module, "groq_client", _Client()),
    ):
        result = await rag_service.answer_question(
            None,
            "h-history-cap",
            "Demo",
            "residual bottleneck convolutional layers batch normalization?",
            history,
            "intermediate",
            full_text=text,
        )

    assert result["answer"] == "ok"
    assert captured["count"] <= 1 + 6 + 1  # system + last 6 turns + user prompt
