"""
Unified service for orchestrating evidence adapters through the analysis pipeline.
Replaces adapter-specific services with a generic approach.
"""
import logging
import math
from typing import Callable, List

from app.api.managers.job_manager import job_manager
from app.adapters import get_adapter
from app.retrieval.paper_search import QueryWrapper
from app.agents.relevant_paper_selector_agent import RelevantPaperSelectorAgent
from app.agents.query_variant_agent import QueryVariantAgent
from app.models.paper import Paper
from core import config

logger = logging.getLogger(__name__)

_CURRENT_YEAR = 2026  # used for recency normalisation


class AdapterService:
    """
    Generic service for running any evidence adapter through the analysis pipeline.
    """
    
    def __init__(self):
        """Initialize shared components."""
        self._query_wrapper = QueryWrapper(use_reranker=True)
        self._paper_selector = RelevantPaperSelectorAgent()
        self._query_variant_agent = QueryVariantAgent()
    
    @classmethod
    def search_evidence(
        cls,
        job_id: str,
        adapter_name: str,
        update_progress: Callable
    ):
        """
        Search for evidence using the specified adapter.
        
        Args:
            job_id: Job identifier
            adapter_name: Name of the adapter to use (e.g., 'arxiv', 'google_patents')
            update_progress: Progress callback function
        """
        service = cls()
        job = job_manager.get_job(job_id)
        if not job:
            return
        
        # Get the adapter
        adapter = get_adapter(adapter_name)
        if not adapter:
            logger.error(f"Adapter '{adapter_name}' not found")
            update_progress(job_id, f"Error: Adapter '{adapter_name}' not found", 0.15)
            return
        
        if not adapter.is_available:
            logger.error(f"Adapter '{adapter_name}' is not available")
            update_progress(job_id, f"Error: Adapter '{adapter_name}' is not available", 0.15)
            return
        
        idea_for_search = job.state.enriched_idea or job.user_idea
        settings = job.settings or {}

        # ── Per-adapter pipeline settings ──────────────────────────────────────
        is_openalex = adapter_name == "openalex"
        if is_openalex:
            # Steps 2, 6–9: fetch more per variant; wider embedding window;
            # full cross-encoder rerank; LLM sees top-20 of reranked 100.
            papers_per_variant = settings.get("papers_per_variant", 150)
            embedding_topk     = settings.get("embedding_topk", 200)
            rerank_topk        = settings.get("rerank_topk", 100)
            llm_topk           = settings.get("llm_topk", 20)
            final_papers_count = settings.get("final_papers", 7)
        else:
            n_variants_hint    = 4  # rough default before LLM runs
            papers_per_variant = settings.get("papers_per_query", 150) // n_variants_hint
            embedding_topk     = settings.get("embedding_topk", 100)
            rerank_topk        = settings.get("rerank_topk", 20)
            llm_topk           = rerank_topk
            final_papers_count = settings.get("final_papers", 5)

        if is_openalex:
            fetch_per_variant = papers_per_variant
            convert_per_variant = papers_per_variant
        else:
            convert_per_variant = int(
                settings.get("papers_per_variant_conversion", config.PAPERS_PER_VARIANT_CONVERSION)
            )
            fetch_per_variant = max(papers_per_variant, convert_per_variant)

        logger.info(f"Using adapter: {adapter_name}")
        update_progress(job_id, f"Generating query variants for {adapter.description}...", 0.15)

        # Step 1: Generate query variants (adapter-aware prompt)
        query_variants = service._query_variant_agent.generate_query_variants(
            idea_for_search, adapter_name=adapter_name
        )
        if not query_variants:
            logger.warning("Query variant LLM returned no variants; falling back to raw idea")
            query_variants = [{"type": "raw", "query": idea_for_search[:500]}]
        logger.info(
            f"Generated {len(query_variants)} query variants: {[v['query'] for v in query_variants]}"
        )
        job.state.query_variants = query_variants

        update_progress(job_id, f"Searching {adapter.description}...", 0.20)

        # Steps 2–3: Search each variant; track max relevance score per paper
        all_papers: List[Paper] = []
        openalex_max_relevance: dict = {}  # source_id → max relevance_score across variants

        for variant in query_variants:
            variant_query = variant["query"]
            logger.info(f"Searching {adapter_name} with query: {variant_query}")

            results = adapter.search(variant_query, max_results=fetch_per_variant)
            logger.info(f"  [{adapter_name}] '{variant_query[:80]}' → {len(results)} raw results")

            paper_models = adapter.convert_to_papers(results, limit=convert_per_variant)

            # Track per-paper max relevance for composite ranking (OpenAlex only)
            if is_openalex:
                for paper in paper_models:
                    rel = paper.metadata.get("relevance_score") or 0.0
                    prev = openalex_max_relevance.get(paper.source_id, 0.0)
                    if rel > prev:
                        openalex_max_relevance[paper.source_id] = rel

            all_papers.extend(paper_models)

        # Step 4: Deduplicate by OpenAlex ID and DOI
        seen_source_ids: set = set()
        seen_dois: set = set()
        unique_papers: List[Paper] = []
        for paper in all_papers:
            doi = (paper.metadata or {}).get("doi") or ""
            if paper.source_id in seen_source_ids:
                continue
            if doi and doi in seen_dois:
                continue
            unique_papers.append(paper)
            seen_source_ids.add(paper.source_id)
            if doi:
                seen_dois.add(doi)

        # Re-assign globally unique IDs after dedup
        for i, paper in enumerate(unique_papers):
            paper.paper_id = f"paper_{i+1:03d}"

        # Step 5: Composite pre-ranking for OpenAlex (relevance + citations + recency)
        if is_openalex:
            # Backfill max relevance scores gathered across variants
            for paper in unique_papers:
                paper.metadata["relevance_score"] = openalex_max_relevance.get(
                    paper.source_id, paper.metadata.get("relevance_score") or 0.0
                )
            unique_papers = service._openalex_prerank(unique_papers)
            # Cap at 500 before embedding to keep index build fast
            unique_papers = unique_papers[:500]

        logger.info(f"Total unique papers from {adapter_name}: {len(unique_papers)}")
        job.state.total_papers_fetched = len(unique_papers)
        job.state.selected_sources = [adapter_name]
        job.state.source_results = {adapter_name: len(unique_papers)}

        if not unique_papers:
            logger.warning(f"No papers found from {adapter_name}")
            update_progress(
                job_id,
                f"No {adapter.evidence_noun_plural} found from {adapter.description}",
                0.25,
            )
            job.state.selected_papers = []
            return

        update_progress(
            job_id,
            f"Running semantic search on {len(unique_papers)} {adapter.evidence_noun_plural}...",
            0.30,
        )

        # Steps 6–7: Embed title+abstract with E5, cross-encoder rerank top-100
        import json
        jsonl_papers = service._convert_papers_to_jsonl(unique_papers)
        service._save_papers_to_jsonl(jsonl_papers)

        search_results = service._query_wrapper.search_literature(
            idea_for_search,
            include_scores=True,
            embedding_topk=embedding_topk,
            rerank_topk=rerank_topk,
            force_rebuild=True,
        )

        search_results_list = json.loads(search_results)
        if not isinstance(search_results_list, list):
            logger.error(f"search_literature returned unexpected format: {type(search_results_list)}")
            search_results_list = []

        job.state.unique_papers_after_dedup = len(unique_papers)
        job.state.papers_after_embedding = len(search_results_list)
        job.state.papers_after_rerank = len(search_results_list)

        logger.info(
            f"After embedding+rerank: {len(search_results_list)} papers "
            f"(embedding_topk={embedding_topk}, rerank_topk={rerank_topk})"
        )

        if not search_results_list:
            logger.warning("No papers passed semantic search filter")
            update_progress(
                job_id,
                f"No relevant {adapter.evidence_noun_plural} found after filtering",
                0.35,
            )
            job.state.selected_papers = []
            return

        # Step 8: LLM judges the top-N of the reranked list
        llm_candidates = search_results_list[:llm_topk]

        update_progress(
            job_id,
            f"LLM selecting {final_papers_count} {adapter.evidence_noun_plural} "
            f"from top {len(llm_candidates)} candidates...",
            0.50,
        )

        selected_papers_data = service._paper_selector.select_papers(
            user_idea=idea_for_search,
            papers=llm_candidates,
            select_count=final_papers_count,
            adapter_name=adapter_name,
        )

        logger.info(
            f"LLM selected {len(selected_papers_data)} papers from {len(llm_candidates)} candidates"
        )

        # Step 9: Convert selected dicts back to Paper objects (PDFs downloaded later)
        selected_papers = service._convert_search_results_to_papers(selected_papers_data, unique_papers)

        job.state.selected_papers = selected_papers
        job.state.all_papers = search_results_list[:100]

        job.state.search_funnel = {
            "adapter": adapter_name,
            "total_fetched": len(unique_papers),
            "after_dedup": len(unique_papers),
            "after_embedding": job.state.papers_after_embedding,
            "after_rerank": job.state.papers_after_rerank,
            "llm_candidates": len(llm_candidates),
            "final_selected": len(selected_papers),
        }

        logger.info(f"Final selection: {len(selected_papers)} papers from {adapter_name}")
        update_progress(
            job_id,
            f"Selected {len(selected_papers)} {adapter.evidence_noun_plural} for analysis",
            0.60,
        )
    
    @staticmethod
    def _openalex_prerank(papers: List[Paper]) -> List[Paper]:
        """
        Step 5: composite pre-ranking for OpenAlex candidates.
        Score = 0.5 * norm_relevance + 0.3 * norm_log_citations + 0.2 * norm_recency
        All three components are normalised to [0, 1] within this batch.
        """
        if not papers:
            return papers

        relevances = [p.metadata.get("relevance_score") or 0.0 for p in papers]
        citations  = [p.metadata.get("cited_by_count") or 0   for p in papers]

        max_rel  = max(relevances) or 1.0
        max_cite = math.log1p(max(citations)) or 1.0

        def _year(p: Paper) -> int:
            try:
                return int((p.published_date or "2000")[:4])
            except (ValueError, TypeError):
                return 2000

        min_year = min(_year(p) for p in papers)
        year_span = max(_CURRENT_YEAR - min_year, 1)

        def _score(p: Paper) -> float:
            rel     = (p.metadata.get("relevance_score") or 0.0) / max_rel
            cite    = math.log1p(p.metadata.get("cited_by_count") or 0) / max_cite
            recency = (_year(p) - min_year) / year_span
            return 0.5 * rel + 0.3 * cite + 0.2 * recency

        ranked = sorted(papers, key=_score, reverse=True)
        logger.info(
            f"[OpenAlex pre-rank] top scores: "
            + ", ".join(f"{_score(p):.3f}" for p in ranked[:5])
        )
        return ranked

    def _convert_papers_to_jsonl(self, papers: List[Paper]) -> List[dict]:
        """Convert Paper models to JSONL format for semantic search."""
        from typing import Dict, Any
        jsonl_papers = []
        
        for paper in papers:
            paper_dict: Dict[str, Any] = {
                'id': paper.paper_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'url': paper.url,
                'pdf_url': paper.pdf_url,
                'source': paper.source,
                'source_id': paper.source_id,
            }
            if paper.published_date:
                paper_dict['year'] = paper.published_date[:4] if isinstance(paper.published_date, str) else str(paper.published_date)[:4]
            if paper.categories:
                paper_dict['categories'] = paper.categories
            
            jsonl_papers.append(paper_dict)
        
        return jsonl_papers
    
    def _save_papers_to_jsonl(self, jsonl_papers: List[dict], filename: str = None):
        """Save papers to JSONL file format."""
        import os
        import json
        
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(script_dir, '..', '..', 'app', 'retrieval', 'sample_papers.jsonl')
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for paper in jsonl_papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(jsonl_papers)} papers to {filename}")
    
    def _convert_search_results_to_papers(self, search_results: List[dict], all_papers: List[Paper]) -> List[Paper]:
        """Convert search results back to Paper objects with scores."""
        # Create lookup by ID
        paper_lookup = {p.paper_id: p for p in all_papers}
        
        reranked = []
        for result in search_results:
            paper_id = result.get('id')
            if paper_id in paper_lookup:
                paper = paper_lookup[paper_id]
                # Add scores to paper metadata
                paper.metadata['embedding_score'] = result.get('score', 0)
                paper.metadata['rerank_score'] = result.get('rerank_score', result.get('score', 0))
                reranked.append(paper)
        
        return reranked
