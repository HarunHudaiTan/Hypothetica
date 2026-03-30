"""
FastAPI backend for Hypothetica Research Originality Assessment.
"""
import json
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.api.schemas.analysis import AnalyzeRequest, AnswersRequest
from app.api.schemas.job import JobStatus, JobStatusResponse
from app.api.schemas.matches import SentenceMatchRequest

from app.api.managers.job_manager import job_manager
from app.services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Hypothetica API starting up")
    yield
    logger.info("Hypothetica API shutting down")


app = FastAPI(
    title="Hypothetica API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze")
async def start_analysis(req: AnalyzeRequest):
    """Start a new originality analysis job."""
    settings = req.model_dump(exclude={"user_idea"})
    
    # Debug logging for paper_sources
    logger.info(f"API received paper_sources: {settings.get('paper_sources', 'NOT_FOUND')}")
    logger.info(f"Full settings: {settings}")
    
    job_id = job_manager.create_job(req.user_idea, settings)

    AnalysisService.start_questions_phase(job_id)

    return {"job_id": job_id}


@app.get("/api/analyze/{job_id}/status")
async def get_status(job_id: str):
    """Poll for job status."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    response = JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        progress_message=job.progress_message,
        questions=job.questions,
        results=job.results,
        error=job.error,
    )

    if job.state.reality_check_warning:
        response.reality_check = {
            "warning": job.state.reality_check_warning,
            "result": job.state.reality_check_result,
        }

    if job.results:
        response.stats = job.results.get("stats")

    return response


@app.get("/api/analyze/{job_id}/stream")
async def stream_events(job_id: str):
    """SSE stream for real-time progress updates."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        while True:
            events = job.pop_events()
            for event in events:
                yield {
                    "event": event["type"],
                    "data": json.dumps(event),
                }
            if job.status in (JobStatus.COMPLETED, JobStatus.ERROR):
                yield {
                    "event": "done",
                    "data": json.dumps({"status": job.status.value}),
                }
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.post("/api/analyze/{job_id}/answers")
async def submit_answers(job_id: str, req: AnswersRequest):
    """Submit follow-up answers and start the analysis pipeline."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status != JobStatus.WAITING_FOR_ANSWERS:
        raise HTTPException(400, f"Job is not waiting for answers (status: {job.status.value})")

    AnalysisService.start_analysis_phase(job_id, req.answers)

    return {"status": "processing"}


@app.post("/api/analyze/{job_id}/matches")
async def get_sentence_matches(job_id: str, req: SentenceMatchRequest):
    """Get RAG matches for a specific sentence."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    matches = AnalysisService.get_matches_for_sentence(job_id, req.sentence, top_k=req.top_k)
    return {"matches": matches}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "hypothetica-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
