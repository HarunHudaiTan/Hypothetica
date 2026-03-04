"""
FastAPI backend for Hypothetica Research Originality Assessment.

Endpoints:
  POST /api/analyze              - Start a new analysis job
  GET  /api/analyze/{id}/status  - Get job status (poll)
  GET  /api/analyze/{id}/stream  - SSE event stream for real-time progress
  POST /api/analyze/{id}/answers - Submit follow-up answers
  POST /api/analyze/{id}/matches - Get RAG matches for a sentence
  GET  /api/health               - Health check
"""
import sys
import os
import json
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models import (
    AnalyzeRequest, AnswersRequest, SentenceMatchRequest,
    JobStatus, JobStatusResponse,
)
from api.job_manager import job_manager

from pipeline.originality_pipeline import OriginalityPipeline

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


def _run_questions_phase(job_id: str):
    """Background thread: generate follow-up questions (no reality check here)."""
    job = job_manager.get_job(job_id)
    if not job:
        return

    try:
        pipeline = OriginalityPipeline()
        job.pipeline = pipeline

        settings = job.settings or {}
        pipeline.papers_per_query = settings.get("papers_per_query", 150)
        pipeline.embedding_topk = settings.get("embedding_topk", 100)
        pipeline.rerank_topk = settings.get("rerank_topk", 20)
        pipeline.final_papers = settings.get("final_papers", 5)

        def progress_cb(message: str, progress: float):
            job_manager.update_progress(job_id, message, progress)

        pipeline.progress_callback = progress_cb

        job_manager.update_progress(job_id, "Generating follow-up questions...", 0.05)
        questions = pipeline.generate_followup_questions(job.user_idea)
        job_manager.set_questions(job_id, questions)

    except Exception as e:
        logger.exception("Error in questions phase")
        job_manager.set_error(job_id, str(e))


def _run_reality_check(job_id: str):
    """
    Background thread: run reality check in parallel with the main pipeline.
    This is advisory only — it never affects the final score.
    """
    job = job_manager.get_job(job_id)
    if not job or not job.pipeline:
        return

    try:
        pipeline = job.pipeline
        pipeline.run_reality_check(job.user_idea)

        rc = pipeline.state.reality_check_result
        warning = pipeline.state.reality_check_warning
        if rc:
            job.push_event({
                "type": "reality_check",
                "reality_check": rc,
                "warning": warning,
            })
    except Exception as e:
        logger.warning(f"Reality check failed (non-fatal): {e}")


def _run_analysis_phase(job_id: str, answers: list):
    """Background thread: run the full analysis after answers are submitted."""
    job = job_manager.get_job(job_id)
    if not job or not job.pipeline:
        return

    try:
        pipeline = job.pipeline
        job_manager.update_status(job_id, JobStatus.PROCESSING)

        # Kick off reality check in a separate thread (runs in parallel)
        rc_thread = threading.Thread(
            target=_run_reality_check, args=(job_id,), daemon=True
        )
        rc_thread.start()

        pipeline.process_answers(answers)
        pipeline.search_papers()
        pipeline.process_papers()
        pipeline.run_layer1_analysis()
        result = pipeline.run_layer2_analysis()

        # Wait for reality check to finish (it should be done by now)
        rc_thread.join(timeout=10)

        results_dict = result.to_dict()

        papers_detail = []
        for paper in pipeline.state.selected_papers:
            l1 = next(
                (r for r in pipeline.state.layer1_results if r.paper_id == paper.paper_id),
                None,
            )
            entry = {
                "paper_id": paper.paper_id,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "authors": paper.authors,
                "categories": paper.categories,
                "is_processed": paper.is_processed,
            }
            if l1:
                entry["overall_overlap_score"] = l1.overall_overlap_score
                entry["criteria_scores"] = l1.criteria_scores.to_dict()
            papers_detail.append(entry)

        results_dict["papers"] = papers_detail

        # Attach reality check as advisory info (does NOT affect scores)
        reality_rc = pipeline.state.reality_check_result
        if reality_rc:
            results_dict["reality_check"] = reality_rc
            results_dict["reality_check_warning"] = pipeline.state.reality_check_warning

        results_dict["stats"] = pipeline.get_stats()

        job_manager.set_results(job_id, results_dict)

    except Exception as e:
        logger.exception("Error in analysis phase")
        job_manager.set_error(job_id, str(e))


# =========================================================================
# ENDPOINTS
# =========================================================================

@app.post("/api/analyze")
async def start_analysis(req: AnalyzeRequest):
    """Start a new originality analysis job."""
    settings = req.model_dump(exclude={"user_idea"})
    job_id = job_manager.create_job(req.user_idea, settings)

    thread = threading.Thread(target=_run_questions_phase, args=(job_id,), daemon=True)
    thread.start()

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

    if job.pipeline and job.pipeline.state.reality_check_warning:
        response.reality_check = {
            "warning": job.pipeline.state.reality_check_warning,
            "result": job.pipeline.state.reality_check_result,
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

    thread = threading.Thread(target=_run_analysis_phase, args=(job_id, req.answers), daemon=True)
    thread.start()

    return {"status": "processing"}


@app.post("/api/analyze/{job_id}/matches")
async def get_sentence_matches(job_id: str, req: SentenceMatchRequest):
    """Get RAG matches for a specific sentence."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if not job.pipeline:
        raise HTTPException(400, "Pipeline not initialized")

    matches = job.pipeline.get_matches_for_sentence(req.sentence, top_k=req.top_k)
    return {"matches": matches}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "hypothetica-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
