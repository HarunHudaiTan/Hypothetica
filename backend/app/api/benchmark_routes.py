"""
Benchmark batch API: run preset JSON datasets through the pipeline, log to Supabase + disk.
"""
from fastapi import APIRouter, HTTPException

from app.api.schemas.benchmark import BenchmarkRunRequest, BenchmarkRunResponse
from app.services.benchmark_run_service import PRESET_ADAPTERS, get_run_state, start_background_benchmark

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


@router.post("/run", response_model=BenchmarkRunResponse)
async def start_benchmark_run(req: BenchmarkRunRequest):
    preset = req.preset.value
    if preset not in PRESET_ADAPTERS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}")
    run_id = start_background_benchmark(
        preset=preset,
        limit=req.limit,
        dataset_path_override=req.dataset_path,
        persist_supabase=req.persist_supabase,
        job_timeout_seconds=req.job_timeout_seconds,
    )
    return BenchmarkRunResponse(run_id=run_id, status="started")


@router.get("/run/{run_id}")
async def benchmark_run_status(run_id: str):
    st = get_run_state(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Run not found")
    return st
