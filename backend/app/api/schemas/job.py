from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class JobStatus(str, Enum):
    GENERATING_QUESTIONS = "generating_questions"
    WAITING_FOR_ANSWERS = "waiting_for_answers"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    progress_message: str = ""
    questions: Optional[List[Dict[str, Any]]] = None
    reality_check: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
