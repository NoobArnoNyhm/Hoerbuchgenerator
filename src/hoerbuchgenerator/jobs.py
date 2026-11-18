import asyncio
from dataclasses import asdict, dataclass
from threading import Event
from typing import Literal, TypeAlias
from uuid import uuid4


JobState = Literal["queued", "running", "cancel_requested", "completed", "failed", "cancelled"]
TERMINAL_STATES = {"completed", "failed", "cancelled"}
JobSnapshot: TypeAlias = dict[str, object]


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
        self._subscribers: dict[str, set[asyncio.Queue[JobSnapshot]]] = {}
        self._active_id: str | None = None

    async def reserve(self, kind: str) -> JobSnapshot:
        async with self._lock:
            if self._active_id is not None:
                raise JobSlotBusy

            job = Job(id=str(uuid4()), kind=kind)
            self._jobs[job.id] = job
            self._cancel_events[job.id] = Event()
            self._active_id = job.id
            snapshot = asdict(job)
            self._publish(job.id, snapshot)
            return snapshot

    async def get(self, job_id: str) -> JobSnapshot | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job else None

    async def active(self) -> JobSnapshot | None:
        async with self._lock:
            job = self._jobs.get(self._active_id) if self._active_id else None
            return asdict(job) if job else None

    async def request_cancel(self, job_id: str) -> JobSnapshot | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state not in TERMINAL_STATES and job.state != "cancel_requested":
                job.state = "cancel_requested"
                job.phase = "cancelling"
                self._cancel_events[job.id].set()
                self._publish(job.id, asdict(job))
            return asdict(job)

    async def subscribe(self, job_id: str) -> asyncio.Queue[JobSnapshot] | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            queue: asyncio.Queue[JobSnapshot] = asyncio.Queue(maxsize=1)
            self._subscribers.setdefault(job_id, set()).add(queue)
            queue.put_nowait(asdict(job))
            return queue

    async def unsubscribe(
        self,
        job_id: str,
        queue: asyncio.Queue[JobSnapshot],
    ) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(job_id)
            if subscribers is None:
                return
            subscribers.discard(queue)
            if not subscribers:
                del self._subscribers[job_id]

    def cancel_event(self, job_id: str) -> Event:
        return self._cancel_events[job_id]

    async def update(
        self,
        job_id: str,
        phase: str,
        *,
        total: int | None = None,
        completed: int | None = None,
    ) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or self._active_id != job_id or job.state == "cancel_requested":
                return False
            previous = asdict(job)
            job.state = "running"
            job.phase = phase
            job.total = total
            if completed is not None:
                job.completed = completed
            snapshot = asdict(job)
            if snapshot != previous:
                self._publish(job.id, snapshot)
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
            self._publish(job.id, asdict(job))
            return True

    def _publish(self, job_id: str, snapshot: JobSnapshot) -> None:
        for queue in self._subscribers.get(job_id, ()):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(snapshot)