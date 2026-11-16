"""Phase B2: embedding cache versioning + dimension validation."""

import sys
from unittest.mock import AsyncMock, patch

import pytest

from app.services import rag as rag_module


@pytest.fixture(autouse=True)
def _clean_embedding_cache():
    rag_module._embedding_cache.clear()
    yield
    rag_module._embedding_cache.clear()


class _FakeResponse:
    def __init__(self, dim: int):
        self._values = [0.1] * dim

    def raise_for_status(self):
        return None

    def json(self):
        return {"embedding": {"values": self._values}}


class _FakeClient:
    def __init__(self, dim: int, requests: list[str]):
        self._response = _FakeResponse(dim)
        self._requests = requests

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self._requests.append(url)
        return self._response


def _fake_httpx(dim: int, requests: list[str]):
    class _Mod:
        AsyncClient = lambda *a, **k: _FakeClient(dim, requests)  # noqa: E731

    return _Mod()


@pytest.mark.asyncio
async def test_cache_key_pins_model_and_dim():
    base = rag_module._embedding_cache_key("hello")
    original = rag_module.settings.GOOGLE_EMBEDDING_MODEL
    original_dim = rag_module.settings.EMBEDDING_DIM
    try:
        rag_module.settings.GOOGLE_EMBEDDING_MODEL = "some-other-model"
        assert rag_module._embedding_cache_key("hello") != base
        rag_module.settings.GOOGLE_EMBEDDING_MODEL = original
        rag_module.settings.EMBEDDING_DIM = original_dim + 1
        assert rag_module._embedding_cache_key("hello") != base
    finally:
        rag_module.settings.GOOGLE_EMBEDDING_MODEL = original
        rag_module.settings.EMBEDDING_DIM = original_dim


@pytest.mark.asyncio
async def test_model_switch_does_not_serve_stale_vector():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "key"
    original = rag_module.settings.GOOGLE_EMBEDDING_MODEL
    req_a: list[str] = []
    req_b: list[str] = []
    try:
        rag_module.settings.GOOGLE_EMBEDDING_MODEL = "model-a"
        with patch.dict(sys.modules, {"httpx": _fake_httpx(3072, req_a)}):
            vec_a = await rag_module.compute_dense_embedding("same text")
        rag_module.settings.GOOGLE_EMBEDDING_MODEL = "model-b"
        with patch.dict(sys.modules, {"httpx": _fake_httpx(3072, req_b)}):
            vec_b = await rag_module.compute_dense_embedding("same text")
    finally:
        rag_module.settings.GOOGLE_EMBEDDING_MODEL = original

    assert len(req_a) == 1 and len(req_b) == 1  # cache was NOT shared across models
    assert len(vec_a) == len(vec_b) == 3072


@pytest.mark.asyncio
async def test_wrong_dimension_vector_rejected_and_not_cached():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "key"
    reqs: list[str] = []
    with patch.dict(sys.modules, {"httpx": _fake_httpx(512, reqs)}):
        assert await rag_module.compute_dense_embedding("dim check") is None
        assert await rag_module.compute_dense_embedding("dim check") is None
    assert len(reqs) == 2  # rejected vector must not be cached


class _RetryResp:
    def __init__(self, status: int):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code < 400:
            return None

        class _R:
            status_code = self.status_code

        e = Exception(f"http {self.status_code}")
        e.response = _R()
        raise e

    def json(self):
        return {"embedding": {"values": [0.1] * 3072}}


@pytest.mark.asyncio
async def test_embedding_retries_429_then_succeeds():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "key"
    calls = {"n": 0}

    class _Mod:
        class AsyncClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kwargs):
                calls["n"] += 1
                return _RetryResp(429 if calls["n"] == 1 else 200)

    with (
        patch.dict(sys.modules, {"httpx": _Mod()}),
        patch.object(rag_module.asyncio, "sleep", new=AsyncMock(return_value=None)),
    ):
        vec = await rag_module.compute_dense_embedding("retry check")
    assert vec is not None and len(vec) == 3072
    assert calls["n"] == 2  # one retry, as per backoff policy


@pytest.mark.asyncio
async def test_embedding_gives_up_after_retries():
    rag_module.settings.GOOGLE_EMBEDDING_KEY = "key"
    calls = {"n": 0}

    class _Mod:
        class AsyncClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kwargs):
                calls["n"] += 1
                return _RetryResp(429)

    with (
        patch.dict(sys.modules, {"httpx": _Mod()}),
        patch.object(rag_module.asyncio, "sleep", new=AsyncMock(return_value=None)),
    ):
        vec = await rag_module.compute_dense_embedding("doomed")
    assert vec is None
    assert calls["n"] == 3  # 1 initial + 2 retries
