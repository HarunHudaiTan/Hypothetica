"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class AnalyzeRequest(BaseModel):
    user_idea: str = Field(..., min_length=50)
    papers_per_query: int = Field(default=150, ge=50, le=300)
    embedding_topk: int = Field(default=100, ge=50, le=200)
    rerank_topk: int = Field(default=20, ge=10, le=50)
    final_papers: int = Field(default=5, ge=3, le=10)
    use_reranker: bool = True


class AnswersRequest(BaseModel):
    answers: List[str]


class SentenceMatchRequest(BaseModel):
    sentence: str
    top_k: int = Field(default=5, ge=1, le=20)


class JobStatus(str, Enum):
    GENERATING_QUESTIONS = "generating_questions"
    WAITING_FOR_ANSWERS = "waiting_for_answers"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class ProgressEvent(BaseModel):
    type: str = "progress"
    message: str = ""
    progress: float = 0.0


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
