"""Pytest configuration for backend tests.

Ensures the ``app`` package is importable from the tests directory and
registers the asyncio mode for async test functions.
"""
import sys
import os

# Add the backend root to sys.path so `app.*` imports resolve.
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import pytest


@pytest.fixture(autouse=True)
def _isolate_in_memory_caches():
    """Reset the RAG service's module-level in-memory dicts around every
    test so vectors/chunks from one test do not leak into another."""
    from app.services import rag
    rag._MEMORY_VECTOR_STORE.clear()
    rag._MEMORY_CHUNK_STORE.clear()
    rag._embedding_cache.clear()
    yield
    rag._MEMORY_VECTOR_STORE.clear()
    rag._MEMORY_CHUNK_STORE.clear()
    rag._embedding_cache.clear()
