"""Lifecycle of the background RAG indexing tasks started by process_paper."""

import asyncio

import pytest

from app.services import pipeline


@pytest.fixture(autouse=True)
def _clean_tasks():
    pipeline._INDEX_TASKS.clear()
    yield
    for task in list(pipeline._INDEX_TASKS.values()):
        task.cancel()
    pipeline._INDEX_TASKS.clear()


@pytest.mark.asyncio
async def test_index_task_is_tracked_while_running():
    gate = asyncio.Event()

    async def gated():
        await gate.wait()

    task = pipeline._start_index_task("h1", gated())

    assert isinstance(task, asyncio.Task)
    assert pipeline._INDEX_TASKS.get("h1") is task
    gate.set()
    await task


@pytest.mark.asyncio
async def test_index_task_completes_and_is_discarded():
    gate = asyncio.Event()
    finished = []

    async def gated():
        await gate.wait()
        finished.append(True)

    task = pipeline._start_index_task("h2", gated())
    gate.set()
    await task
    await asyncio.sleep(0)

    assert finished == [True]
    assert "h2" not in pipeline._INDEX_TASKS


@pytest.mark.asyncio
async def test_second_call_same_hash_does_not_grow_registry():
    gate = asyncio.Event()

    async def gated():
        await gate.wait()

    first = pipeline._start_index_task("h3", gated())
    duplicate = gated()
    assert pipeline._start_index_task("h3", duplicate) is None
    duplicate.close()

    assert len(pipeline._INDEX_TASKS) == 1
    assert pipeline._INDEX_TASKS["h3"] is first
    gate.set()
    await first
    await asyncio.sleep(0)
    assert pipeline._INDEX_TASKS == {}


@pytest.mark.asyncio
async def test_failed_index_task_is_also_discarded():
    async def boom():
        raise RuntimeError("index failed")

    task = pipeline._start_index_task("h4", boom())
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert "h4" not in pipeline._INDEX_TASKS
