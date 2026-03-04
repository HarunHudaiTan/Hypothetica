"""
In-memory job state manager for analysis pipelines.
Stores pipeline instances and status per job_id.
"""
import uuid
import threading
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

from api.models import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.GENERATING_QUESTIONS
    progress: float = 0.0
    progress_message: str = ""
    pipeline: Any = None
    questions: Optional[List[Dict]] = None
    results: Optional[Dict] = None
    error: Optional[str] = None
    settings: Optional[Dict] = None
    user_idea: str = ""
    events: List[Dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push_event(self, event: Dict):
        with self.lock:
            self.events.append(event)

    def pop_events(self) -> List[Dict]:
        with self.lock:
            events = self.events[:]
            self.events.clear()
            return events


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, user_idea: str, settings: Dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, user_idea=user_idea, settings=settings)
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus):
        job = self.get_job(job_id)
        if job:
            job.status = status

    def update_progress(self, job_id: str, message: str, progress: float):
        job = self.get_job(job_id)
        if job:
            job.progress = progress
            job.progress_message = message
            job.push_event({
                "type": "progress",
                "message": message,
                "progress": progress,
            })

    def set_questions(self, job_id: str, questions: List[Dict]):
        job = self.get_job(job_id)
        if job:
            job.questions = questions
            job.status = JobStatus.WAITING_FOR_ANSWERS
            job.push_event({
                "type": "questions",
                "questions": questions,
            })

    def set_results(self, job_id: str, results: Dict):
        job = self.get_job(job_id)
        if job:
            job.results = results
            job.status = JobStatus.COMPLETED
            job.push_event({
                "type": "completed",
                "results": results,
            })

    def set_error(self, job_id: str, error: str):
        job = self.get_job(job_id)
        if job:
            job.error = error
            job.status = JobStatus.ERROR
            job.push_event({
                "type": "error",
                "error": error,
            })


job_manager = JobManager()
