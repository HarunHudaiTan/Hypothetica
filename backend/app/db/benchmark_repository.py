"""
Persist benchmark evaluation rows to Supabase public.benchmark2.
"""
import logging
from typing import Any, Dict, Optional

from core import config

logger = logging.getLogger(__name__)

_client = None

# API originality labels (low/medium/high) → dataset labels
API_LABEL_TO_BENCHMARK = {
    "high": "novel",
    "medium": "incremental",
    "low": "already_exists",
}


def _get_client():
    global _client
    if _client is None:
        if not config.SUPABASE_SERVICE_ROLE_KEY:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY not set")
        from supabase import create_client

        _client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
    return _client


def float_to_likert(value: float) -> int:
    """0–1 float → Likert 1–5 (same mapping as frontend / run_benchmark)."""
    if value >= 1.0:
        return 5
    if value >= 0.75:
        return 4
    if value >= 0.5:
        return 3
    if value >= 0.25:
        return 2
    return 1


def predicted_label_from_result(result: Dict[str, Any]) -> str:
    api = (result.get("label") or "").strip().lower()
    return API_LABEL_TO_BENCHMARK.get(api, api or "")


def save_benchmark_row(
    case: Dict[str, Any],
    result: Dict[str, Any],
    source_name: str,
    job_id: Optional[str] = None,
    benchmark_run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Insert one benchmark row. Returns inserted row id, or None if skipped/failed.
    """
    if not config.SUPABASE_SERVICE_ROLE_KEY:
        logger.debug("Supabase not configured, skipping benchmark insert")
        return None

    pred_label = predicted_label_from_result(result)
    api_label = result.get("label", "")

    agg = result.get("aggregated_criteria") or {}
    ps = agg.get("problem_similarity")
    ms = agg.get("method_similarity")
    do = agg.get("domain_similarity")
    cs = agg.get("contribution_similarity")

    cost = result.get("cost") or {}

    row = {
        "benchmark_run_id": benchmark_run_id,
        "case_id": case["case_id"],
        "source": source_name,
        "domain": case.get("domain", ""),
        "idea": case.get("idea", ""),
        "true_label": case.get("originality_label", ""),
        "predicted_label": pred_label,
        "api_label": api_label,
        "originality_score": result.get("originality_score"),
        "global_similarity_score": result.get("global_similarity_score"),
        "likert_problem_similarity": float_to_likert(ps) if ps is not None else None,
        "likert_method_similarity": float_to_likert(ms) if ms is not None else None,
        "likert_domain_overlap": float_to_likert(do) if do is not None else None,
        "likert_contribution_similarity": float_to_likert(cs) if cs is not None else None,
        "criteria_problem_similarity": ps,
        "criteria_method_similarity": ms,
        "criteria_domain_overlap": do,
        "criteria_contribution_similarity": cs,
        "layer1_results": result.get("layer1_results"),
        "layer2_full": result.get("layer2_full"),
        "papers": result.get("papers"),
        "selected_sources": result.get("selected_sources"),
        "source_results": result.get("source_results"),
        "search_funnel": result.get("search_funnel"),
        "stats": result.get("stats"),
        "sentence_annotations": result.get("sentence_annotations"),
        "papers_analyzed": result.get("papers_analyzed", 0),
        "total_processing_time": result.get("total_processing_time"),
        "cost_breakdown": cost.get("breakdown"),
        "summary": result.get("summary", ""),
        "comprehensive_report": result.get("comprehensive_report", ""),
        "job_id": job_id,
    }

    try:
        client = _get_client()
        resp = client.table("benchmark2").insert(row).execute()
        if resp.data and len(resp.data) > 0:
            rid = resp.data[0].get("id")
            logger.info(
                "Benchmark row saved: run=%s case=%s id=%s",
                benchmark_run_id,
                case["case_id"],
                rid,
            )
            return str(rid) if rid else None
        return None
    except Exception as e:
        logger.warning("Benchmark Supabase insert failed (non-fatal): %s", e)
        return None
