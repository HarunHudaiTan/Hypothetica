"""
Supabase repository for persisting analysis queries.
Records every completed analysis: scores, outputs, cost, papers.
"""
import logging
from typing import Dict, Any, Optional

from core import config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-init Supabase client."""
    global _client
    if _client is None:
        if not config.SUPABASE_SERVICE_ROLE_KEY:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY not set - cannot persist queries")
        from supabase import create_client
        _client = create_client(
            config.SUPABASE_URL,
            config.SUPABASE_SERVICE_ROLE_KEY,
        )
    return _client


def save_analysis(job_id: str, results: Dict[str, Any]) -> Optional[str]:
    """
    Persist a completed analysis to Supabase.
    Returns UUID of inserted row, or None if not configured or insert fails.
    """
    if not config.SUPABASE_SERVICE_ROLE_KEY:
        logger.debug("Supabase not configured, skipping query persistence")
        return None

    try:
        client = _get_client()

        cost = results.get("cost") or {}
        cost_breakdown = cost.get("breakdown") or {}
        estimated_cost = cost.get("estimated_cost_usd", 0.0)

        row = {
            "job_id": job_id,
            "user_idea": results.get("user_idea", ""),
            "enriched_idea": results.get("enriched_idea"),
            "followup_questions": results.get("followup_questions"),
            "followup_answers": results.get("followup_answers"),
            "global_originality_score": results.get("originality_score"),
            "global_similarity_score": results.get("global_similarity_score"),
            "label": results.get("label"),
            "summary": results.get("summary"),
            "aggregated_criteria": results.get("aggregated_criteria"),
            "cost_breakdown": cost_breakdown,
            "estimated_cost_usd": estimated_cost,
            "papers_analyzed": results.get("papers_analyzed", 0),
            "total_processing_time": results.get("total_processing_time"),
            "reality_check": results.get("reality_check"),
            "reality_check_warning": results.get("reality_check_warning"),
            "stats": results.get("stats"),
            "sentence_annotations": results.get("sentence_annotations"),
            "papers": results.get("papers"),
            "github_analysis": results.get("github_analysis"),
        }

        resp = client.table("queries").insert(row).execute()
        if resp.data and len(resp.data) > 0:
            record_id = resp.data[0].get("id")
            logger.info("Persisted analysis to Supabase: job_id=%s, id=%s", job_id, record_id)
            return record_id
        return None

    except Exception as e:
        logger.warning("Failed to persist analysis to Supabase (non-fatal): %s", e)
        return None
