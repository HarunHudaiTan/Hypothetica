"""
Unified service for orchestrating evidence adapters through the analysis pipeline.
Replaces adapter-specific services with a generic approach.
"""
import logging
from typing import Callable, List

from app.api.managers.job_manager import job_manager
from app.adapters import get_adapter
from app.retrieval.paper_search import QueryWrapper
from app.agents.relevant_paper_selector_agent import RelevantPaperSelectorAgent
from app.agents.query_variant_agent import QueryVariantAgent
from app.models.paper import Paper

logger = logging.getLogger(__name__)


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
        papers_per_query = settings.get("papers_per_query", 150)
        embedding_topk = settings.get("embedding_topk", 100)
        rerank_topk = settings.get("rerank_topk", 20)
        final_papers_count = settings.get("final_papers", 5)
        
        logger.info(f"Using adapter: {adapter_name}")
        update_progress(job_id, f"Generating query variants for {adapter.description}...", 0.15)
        
        # Generate query variants for better recall (adapter-aware)
        query_variants = service._query_variant_agent.generate_query_variants(idea_for_search, adapter_name=adapter_name)
        logger.info(f"Generated {len(query_variants)} query variants: {[v['query'] for v in query_variants]}")
        job.state.query_variants = query_variants
        
        update_progress(job_id, f"Searching {adapter.description}...", 0.20)
        
        # Search using the adapter
        all_papers = []
        for variant in query_variants:
            variant_query = variant['query']
            logger.info(f"Searching {adapter_name} with query: {variant_query}")
            
            results = adapter.search(variant_query, max_results=papers_per_query // len(query_variants))
            logger.info(f"  [{adapter_name}] query='{variant_query[:80]}' → {len(results)} raw results")
            paper_models = adapter.convert_to_papers(results, limit=50)  # Limit per variant
            
            all_papers.extend(paper_models)
        
        # Deduplicate papers
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper.source_id not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper.source_id)
        
        logger.info(f"Total unique papers from {adapter_name}: {len(unique_papers)}")
        job.state.total_papers_fetched = len(unique_papers)
        job.state.selected_sources = [adapter_name]
        job.state.source_results = {adapter_name: len(unique_papers)}
        
        if not unique_papers:
            logger.warning(f"No papers found from {adapter_name}")
            update_progress(job_id, f"No papers found from {adapter.description}", 0.25)
            job.state.selected_papers = []
            return
        
        update_progress(job_id, f"Running semantic search on {len(unique_papers)} papers...", 0.30)
        
        # Convert papers to JSONL format for semantic search
        import json
        jsonl_papers = service._convert_papers_to_jsonl(unique_papers)
        service._save_papers_to_jsonl(jsonl_papers)
        
        # Run semantic search with embeddings + reranking
        search_results = service._query_wrapper.search_literature(
            idea_for_search,
            include_scores=True,
            embedding_topk=embedding_topk,
            rerank_topk=rerank_topk,
            force_rebuild=True
        )
        
        search_results_list = json.loads(search_results)
        if not isinstance(search_results_list, list):
            logger.error(f"search_literature returned unexpected format: {type(search_results_list)}")
            search_results_list = []
        
        job.state.unique_papers_after_dedup = len(unique_papers)
        job.state.papers_after_embedding = len(search_results_list)
        job.state.papers_after_rerank = len(search_results_list)
        
        logger.info(f"After semantic search: {len(search_results_list)} papers")
        
        if not search_results_list:
            logger.warning("No papers passed semantic search filter")
            update_progress(job_id, "No relevant papers found after filtering", 0.35)
            job.state.selected_papers = []
            return
        
        update_progress(job_id, f"Selecting final {final_papers_count} papers with LLM...", 0.50)
        
        logger.info(f"After reranking: {len(search_results_list)} papers")
        
        # LLM-based final selection (works with paper dicts)
        selected_papers_data = service._paper_selector.select_papers(
            user_idea=idea_for_search,
            papers=search_results_list,
            select_count=final_papers_count,
            adapter_name=adapter_name
        )
        
        logger.info(f"LLM selected {len(selected_papers_data)} papers from {len(search_results_list)} candidates")
        
        # Convert selected paper dicts back to Paper objects
        selected_papers = service._convert_search_results_to_papers(selected_papers_data, unique_papers)
        
        job.state.selected_papers = selected_papers
        job.state.all_papers = search_results_list[:100]  # Store first 100 dicts for reference
        
        # Build search funnel metadata
        job.state.search_funnel = {
            "adapter": adapter_name,
            "total_fetched": len(unique_papers),
            "after_dedup": len(unique_papers),
            "after_embedding": len(search_results_list),
            "after_rerank": len(search_results_list),
            "final_selected": len(selected_papers)
        }
        
        logger.info(f"Final selection: {len(selected_papers)} papers from {adapter_name}")
        update_progress(job_id, f"Selected {len(selected_papers)} papers for analysis", 0.60)
    
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
