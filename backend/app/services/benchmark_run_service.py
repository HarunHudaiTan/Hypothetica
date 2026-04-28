"""
Run flat benchmark JSON files (patents / openalex / github) through the analysis
pipeline, persist rows to Supabase, and write a consolidated JSON under benchmarks/results/runs/.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.api.managers.job_manager import job_manager
from app.api.schemas.analysis import AnalyzeRequest
from app.api.schemas.job import JobStatus
from app.db.benchmark_repository import (
    float_to_likert,
    predicted_label_from_result,
    save_benchmark_row,
)
from app.services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# preset name → (adapter name, default dataset path relative to repo root)
PRESET_ADAPTERS: dict[str, tuple[str, str]] = {
    "patents": ("google_patents", "benchmarks/patents_benchmark.json"),
    "openalex": ("openalex", "benchmarks/openalex_benchmark.json"),
    "github": ("github", "benchmarks/github_benchmark.json"),
}

_run_states: dict[str, dict[str, Any]] = {}
_run_lock = threading.Lock()


def get_run_state(run_id: str) -> Optional[dict[str, Any]]:
    with _run_lock:
        s = _run_states.get(run_id)
        return dict(s) if s else None


def _set_run_state(run_id: str, **kwargs: Any) -> None:
    with _run_lock:
        if run_id not in _run_states:
            _run_states[run_id] = {}
        _run_states[run_id].update(kwargs)


def load_benchmark_cases(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("cases", [])
    else:
        raw = []

    out: list[dict[str, Any]] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        cid = c.get("id") or c.get("case_id")
        if not cid:
            continue
        lbl = c.get("originality_label") or c.get("label") or ""
        idea = (c.get("idea") or "").strip()
        out.append(
            {
                "case_id": str(cid),
                "domain": str(c.get("domain", "") or ""),
                "originality_label": str(lbl),
                "idea": idea,
            }
        )
    return out


def resolve_dataset_path(preset: str, override: Optional[str]) -> Path:
    if override:
        raw = Path(override).expanduser()
        path = raw if raw.is_absolute() else _REPO_ROOT / raw
        path = path.resolve()
        root = _REPO_ROOT.resolve()
        try:
            path.relative_to(root)
        except ValueError as e:
            raise ValueError(f"dataset_path must be inside repository root ({root})") from e
        return path
    _adapter, rel = PRESET_ADAPTERS[preset]
    return (_REPO_ROOT / rel).resolve()


def _default_job_settings(adapter: str) -> dict[str, Any]:
    placeholder = "x" * 60
    req = AnalyzeRequest(user_idea=placeholder)
    settings = req.model_dump(exclude={"user_idea", "selected_sources", "selected_adapter"})
    settings["selected_adapter"] = adapter
    settings["selected_sources"] = [adapter]
    settings["benchmark_mode"] = True
    return settings


def _wait_for_job(
    job_id: str, timeout_sec: float
) -> tuple[JobStatus, Optional[dict[str, Any]], Optional[str]]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        job = job_manager.get_job(job_id)
        if not job:
            return JobStatus.ERROR, None, "job_missing"
        if job.status == JobStatus.COMPLETED:
            return job.status, job.results, None
        if job.status == JobStatus.ERROR:
            return job.status, None, job.error or "unknown_error"
        time.sleep(0.75)
    return JobStatus.ERROR, None, "timeout"


def run_one_benchmark_case(
    case: dict[str, Any], adapter: str, job_timeout_seconds: float
) -> dict[str, Any]:
    idea = (case.get("idea") or "").strip()
    if len(idea) < 50:
        idea = idea + (" " * (50 - len(idea)))

    settings = _default_job_settings(adapter)
    job_id = job_manager.create_job(idea, settings)
    AnalysisService.start_analysis_phase(job_id, [])

    status, results, err = _wait_for_job(job_id, job_timeout_seconds)
    base: dict[str, Any] = {
        "case_id": case["case_id"],
        "domain": case.get("domain", ""),
        "true_label": case.get("originality_label", ""),
        "idea": case.get("idea", ""),
        "job_id": job_id,
        "status": status.value,
        "error": err,
    }
    if results:
        agg = results.get("aggregated_criteria") or {}
        ps = agg.get("problem_similarity")
        ms = agg.get("method_similarity")
        do = agg.get("domain_similarity")
        cs = agg.get("contribution_similarity")
        base["likerts"] = {
            "problem_similarity": float_to_likert(ps) if ps is not None else None,
            "method_similarity": float_to_likert(ms) if ms is not None else None,
            "domain_overlap": float_to_likert(do) if do is not None else None,
            "contribution_similarity": float_to_likert(cs) if cs is not None else None,
        }
        base["predicted_label"] = predicted_label_from_result(results)
        base["api_label"] = results.get("label")
        base["full_result"] = results
    else:
        base["likerts"] = None
        base["predicted_label"] = None
        base["api_label"] = None
        base["full_result"] = None
    return base


def execute_benchmark_run(
    run_id: str,
    preset: str,
    limit: Optional[int],
    dataset_path_override: Optional[str],
    persist_supabase: bool,
    job_timeout_seconds: int,
) -> None:
    started = datetime.now(timezone.utc).isoformat()
    _set_run_state(
        run_id,
        status="running",
        preset=preset,
        started_at=started,
        cases_total=0,
        cases_done=0,
        error=None,
        output_path=None,
    )
    try:
        adapter, _rel = PRESET_ADAPTERS[preset]
        path = resolve_dataset_path(preset, dataset_path_override)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        cases = load_benchmark_cases(path)
        if limit is not None:
            cases = cases[: max(0, limit)]

        _set_run_state(run_id, cases_total=len(cases))

        case_logs: list[dict[str, Any]] = []
        for i, case in enumerate(cases):
            logger.info(
                "Benchmark run %s [%s/%s] %s",
                run_id,
                i + 1,
                len(cases),
                case["case_id"],
            )
            log_entry = run_one_benchmark_case(
                case, adapter, float(job_timeout_seconds)
            )

            if log_entry.get("full_result") and persist_supabase:
                save_benchmark_row(
                    case,
                    log_entry["full_result"],
                    adapter,
                    job_id=log_entry.get("job_id"),
                    benchmark_run_id=run_id,
                )

            case_logs.append(log_entry)
            _set_run_state(run_id, cases_done=i + 1)

        out_dir = _REPO_ROOT / "benchmarks" / "results" / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}_{preset}.json"

        completed = datetime.now(timezone.utc).isoformat()
        artifact: dict[str, Any] = {
            "run_id": run_id,
            "preset": preset,
            "adapter": adapter,
            "dataset_path": str(path),
            "started_at": started,
            "completed_at": completed,
            "cases_total": len(cases),
            "persist_supabase": persist_supabase,
            "cases": case_logs,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)

        rel_out = str(out_path.relative_to(_REPO_ROOT))
        _set_run_state(
            run_id,
            status="completed",
            output_path=rel_out,
            completed_at=completed,
        )

    except Exception as e:
        logger.exception("Benchmark run %s failed", run_id)
        _set_run_state(
            run_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


def start_background_benchmark(
    preset: str,
    limit: Optional[int],
    dataset_path_override: Optional[str],
    persist_supabase: bool,
    job_timeout_seconds: int,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    _set_run_state(run_id, status="queued")
    thread = threading.Thread(
        target=execute_benchmark_run,
        args=(
            run_id,
            preset,
            limit,
            dataset_path_override,
            persist_supabase,
            job_timeout_seconds,
        ),
        daemon=True,
    )
    thread.start()
    return run_id
