import asyncio
from dataclasses import asdict, dataclass
from threading import Event
from typing import Literal
from uuid import uuid4


JobState = Literal["queued", "running", "cancel_requested", "completed", "failed", "cancelled"]
TERMINAL_STATES = {"completed", "failed", "cancelled"}


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = "queued"
    phase: str = "accepted"
    completed: int = 0
    total: int | None = None
    error: str | None = None
    result: dict[str, object] | None = None


class JobSlotBusy(Exception):
    pass


class JobManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._jobs: dict[str, Job] = {}
        self._cancel_events: dict[str, Event] = {}
        self._active_id: str | None = None

    async def reserve(self, kind: str) -> dict[str, object]:
        async with self._lock:
            if self._active_id is not None:
                raise JobSlotBusy

            job = Job(id=str(uuid4()), kind=kind)
            self._jobs[job.id] = job
            self._cancel_events[job.id] = Event()
            self._active_id = job.id
            return asdict(job)

    async def get(self, job_id: str) -> dict[str, object] | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job else None

    async def active(self) -> dict[str, object] | None:
        async with self._lock:
            job = self._jobs.get(self._active_id) if self._active_id else None
            return asdict(job) if job else None

    async def request_cancel(self, job_id: str) -> dict[str, object] | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state not in TERMINAL_STATES:
                job.state = "cancel_requested"
                job.phase = "cancelling"
                self._cancel_events[job.id].set()
            return asdict(job)

    def cancel_event(self, job_id: str) -> Event:
        return self._cancel_events[job_id]

    async def update(self, job_id: str, phase: str, *, total: int | None = None) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or self._active_id != job_id or job.state == "cancel_requested":
                return False
            job.state = "running"
            job.phase = phase
            job.total = total
            return True

    async def finish(
        self,
        job_id: str,
        state: Literal["completed", "failed", "cancelled"],
        error: str | None = None,
        result: dict[str, object] | None = None,
    ) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or self._active_id != job_id:
                return False
            job.state = state
            job.phase = state
            job.error = error
            job.result = result
            self._active_id = None
            return True