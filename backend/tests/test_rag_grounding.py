"""RAG grounding tests (issues: forced-fit citations, invented numbers)."""
from unittest.mock import patch, AsyncMock

import pytest

from app.services import rag as rag_module
from app.services.rag import bm25_keyword_search, rag_service


@pytest.fixture(autouse=True)
def _clean_memory_store():
    rag_module._MEMORY_CHUNK_STORE.clear()
    yield
    rag_module._MEMORY_CHUNK_STORE.clear()


def _chunk(section: str, content: str):
    return {"section_title": section, "content": content}


def test_bm25_applies_relevance_floor():
    """A single rare-token hit must not become a citation."""
    filler = " ".join(f"paddingword number{idx}" for idx in range(60))
    corpus = [_chunk("Methods", f"{filler} quixotic") if i == 0 else _chunk("Methods", filler)
              for i in range(10)]
    # 'quixotic' appears once in one long chunk -> weak normalized score
    out = bm25_keyword_search(corpus, "quixotic marvel explained")
    assert out == []
    # repeated distinctive terms -> strong match, honours the floor
    rich = "residual bottleneck convolutional layers batch normalization "
    strong_corpus = [_chunk("Methods", rich * 12) for _ in range(3)]
    strong_corpus += [_chunk("Methods", filler) for _ in range(7)]
    strong = bm25_keyword_search(strong_corpus, "residual bottleneck convolutional layers")
    assert strong
    assert all(c["similarity"] >= rag_module.RELEVANCE_THRESHOLD for c in strong)


@pytest.mark.asyncio
async def test_off_topic_query_refuses_without_forced_citations():
    text = (
        "Abstract. Residual networks learn identity mappings. "
        "Introduction. Deeper plain networks degrade. "
        "Methods. Bottleneck blocks use 1x1 convolutions. "
        "Experiments. ImageNet top-5 error 3.57 percent. "
        "References. He et al 2015."
    )
    with patch("app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)):
        result = await rag_service.answer_question(
            None, "test-hash", "ResNet", "chocolate cake recipe",
            [], "intermediate", full_text=text,
        )
    assert result["citations"] == []
    assert "only answer questions" in result["answer"]


@pytest.mark.asyncio
async def test_numbers_rule_present_in_grounding_prompt():
    target = ("residual bottleneck convolutional layers batch normalization gamma beta ")
    filler = ("galaxy nebula quasar pulsar comet asteroid orbit telescope spectrum ")
    text = target * 10 + (filler * 12 + " ") * 5 + "References. Doe 2020."
    captured = {}

    class _Client:
        def call(self, messages, model, temperature):
            captured["messages"] = messages
            return "ok"

    with patch("app.services.rag.compute_dense_embedding", new=AsyncMock(return_value=None)), \
         patch.object(rag_module, "groq_client", _Client()):
        result = await rag_service.answer_question(
            None, "test-hash-2", "Demo",
            "residual bottleneck convolutional layers batch normalization?",
            [], "intermediate", full_text=text,
        )
    assert result["answer"] == "ok"
    system = captured["messages"][0]["content"]
    assert "verbatim" in system
    assert "top-5" in system or "top-1" in system
