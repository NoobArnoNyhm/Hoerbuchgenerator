import asyncio

import pytest

from hoerbuchgenerator.jobs import JobManager, JobSlotBusy


def test_concurrent_reservations_accept_exactly_one_job():
    async def reserve_both():
        manager = JobManager()

        async def reserve(kind):
            try:
                return await manager.reserve(kind)
            except JobSlotBusy:
                return None

        return await asyncio.gather(reserve("voice"), reserve("synthesis"))

    results = asyncio.run(reserve_both())

    assert sum(result is not None for result in results) == 1


def test_finished_job_releases_slot_and_remains_queryable():
    async def exercise():
        manager = JobManager()
        first = await manager.reserve("voice")
        assert await manager.finish(first["id"], "completed")
        second = await manager.reserve("synthesis")
        return first, second, await manager.get(first["id"])

    first, second, stored = asyncio.run(exercise())

    assert first["id"] != second["id"]
    assert stored["state"] == "completed"


def test_cancel_marks_active_job_and_unknown_job_returns_none():
    async def exercise():
        manager = JobManager()
        job = await manager.reserve("synthesis")
        cancelled = await manager.request_cancel(job["id"])
        missing = await manager.request_cancel("unknown")
        return cancelled, missing

    cancelled, missing = asyncio.run(exercise())

    assert cancelled["state"] == "cancel_requested"
    assert cancelled["phase"] == "cancelling"
    assert missing is None


def test_subscribers_receive_snapshots_and_are_cleaned_up():
    async def exercise():
        manager = JobManager()
        job = await manager.reserve("synthesis")
        first = await manager.subscribe(job["id"])
        second = await manager.subscribe(job["id"])
        assert first is not None
        assert second is not None
        initial = await first.get()
        assert initial["state"] == "queued"
        assert (await second.get())["state"] == "queued"

        await manager.update(job["id"], "synthesizing", total=4, completed=1)
        first_progress = await first.get()
        second_progress = await second.get()
        await manager.request_cancel(job["id"])
        cancellation = await first.get()
        await manager.finish(job["id"], "cancelled")
        terminal = await second.get()
        await manager.unsubscribe(job["id"], first)
        await manager.unsubscribe(job["id"], second)
        return (
            first_progress,
            second_progress,
            cancellation,
            terminal,
            manager._subscribers,
        )

    progress, second_progress, cancellation, terminal, subscribers = asyncio.run(
        exercise()
    )

    assert progress["completed"] == second_progress["completed"] == 1
    assert cancellation["state"] == "cancel_requested"
    assert terminal["state"] == "cancelled"
    assert subscribers == {}


def test_subscriber_queue_coalesces_progress_but_keeps_terminal_snapshot():
    async def exercise():
        manager = JobManager()
        job = await manager.reserve("synthesis")
        queue = await manager.subscribe(job["id"])
        assert queue is not None
        await queue.get()
        await manager.update(job["id"], "synthesizing", total=5, completed=1)
        await manager.update(job["id"], "synthesizing", total=5, completed=4)
        await manager.finish(job["id"], "completed")
        return await queue.get(), queue.empty()

    terminal, is_empty = asyncio.run(exercise())

    assert terminal["state"] == "completed"
    assert terminal["completed"] == 4
    assert is_empty