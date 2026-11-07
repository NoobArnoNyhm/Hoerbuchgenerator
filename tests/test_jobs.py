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