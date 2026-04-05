"""
In-memory job state manager for analysis jobs.
"""
import uuid
import threading
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

from app.api.schemas.job import JobStatus
from app.models.paper import Paper
from app.models.analysis import Layer1Result, Layer2Result, CostBreakdown, GitHubAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Holds the current state of the analysis pipeline."""
    user_idea: str = ""
    enriched_idea: str = ""
    user_sentences: List[str] = field(default_factory=list)
    followup_questions: List[Dict] = field(default_factory=list)
    followup_answers: List[str] = field(default_factory=list)
    query_variants: List[Dict] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    all_papers: List[Dict] = field(default_factory=list)
    selected_papers: List[Paper] = field(default_factory=list)
    layer1_results: List[Layer1Result] = field(default_factory=list)
    layer2_result: Optional[Layer2Result] = None
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    reality_check_result: Dict = field(default_factory=dict)
    reality_check_warning: Optional[str] = None
    github_result: Optional[GitHubAnalysisResult] = None
    # Pipeline stats
    total_papers_fetched: int = 0
    unique_papers_after_dedup: int = 0
    papers_after_embedding: int = 0
    papers_after_rerank: int = 0
    # Literature retrieval (multi-source benchmark metadata)
    selected_sources: List[str] = field(default_factory=list)
    source_results: Dict[str, int] = field(default_factory=dict)
    search_funnel: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.GENERATING_QUESTIONS
    progress: float = 0.0
    progress_message: str = ""
    state: PipelineState = field(default_factory=PipelineState)
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
