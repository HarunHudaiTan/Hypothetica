import json
import logging
from typing import List, Callable, Dict, Any

from app.api.managers.job_manager import job_manager
from app.models.paper import Paper
from core.config import (
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_RESULTS_PER_QUERY
)

from app.processing.arxiv_search import ArxivReq
from app.processing.semantic_scholar_search import SemanticScholarSource
from app.retrieval.paper_search import QueryWrapper
from app.agents.relevant_paper_selector_agent import RelevantPaperSelectorAgent

logger = logging.getLogger(__name__)


class PaperSearchService:
    _query_wrapper = QueryWrapper(use_reranker=True)
    _paper_selector = RelevantPaperSelectorAgent()
    
    def __init__(self):
        """Initialize paper sources based on availability."""
        self._sources = self._initialize_sources()
    
    def _initialize_sources(self) -> Dict[str, Any]:
        """Initialize available paper sources."""
        sources = {}
        
        # ArXiv is always available
        sources["arxiv"] = ArxivReq()
        logger.info("ArXiv source initialized")
        
        # Semantic Scholar requires API key
        logger.info(f"Semantic Scholar API key present: {bool(SEMANTIC_SCHOLAR_API_KEY)}")
        if SEMANTIC_SCHOLAR_API_KEY:
            sources["semantic_scholar"] = SemanticScholarSource(SEMANTIC_SCHOLAR_API_KEY)
            logger.info("Semantic Scholar source initialized")
        else:
            logger.warning("Semantic Scholar API key not provided, skipping source")
            
        logger.info(f"Final available sources: {list(sources.keys())}")
        return sources

    @classmethod
    def search_papers(cls, job_id: str, update_progress: Callable):
        """Enhanced paper search with multi-source support and high-recall retrieval."""
        # Create instance to access sources
        service = cls()
        
        job = job_manager.get_job(job_id)
        if not job:
            return

        idea_for_search = job.state.enriched_idea or job.user_idea
        settings = job.settings or {}
        papers_per_query = settings.get("papers_per_query", 150)
        embedding_topk = settings.get("embedding_topk", 100)
        rerank_topk = settings.get("rerank_topk", 20)
        final_papers_count = settings.get("final_papers", 5)
        
        # Get enabled sources for this job
        enabled_sources = job.state.paper_sources
        logger.info(f"Job requested sources: {enabled_sources}")
        logger.info(f"Available sources: {list(service._sources.keys())}")
        
        available_sources = [s for s in enabled_sources if s in service._sources]
        logger.info(f"Filtered available sources: {available_sources}")
        
        if not available_sources:
            update_progress(job_id, "No paper sources available", 0.15)
            return
            
        update_progress(job_id, f"Searching {len(available_sources)} sources: {', '.join(available_sources)}...", 0.15)

        # Search across all enabled sources
        all_papers = []
        source_stats = {}
        
        for source_name in available_sources:
            source = service._sources[source_name]
            update_progress(job_id, f"Searching {source_name}...", 0.20)
            
            try:
                if source_name == "arxiv":
                    # Use existing arxiv search logic for compatibility
                    papers_json = source.get_papers(idea_for_search, papers_per_query=papers_per_query)
                    papers_summary = json.loads(papers_json)
                    source_papers = papers_summary.get('papers', [])
                    
                    # Store arxiv-specific data for compatibility
                    job.state.query_variants = papers_summary.get('query_variants', [])
                    job.state.keywords = [v['query'] for v in job.state.query_variants]
                    
                elif source_name == "semantic_scholar":
                    # Use multi-query search for Semantic Scholar
                    papers_summary = source.search_papers_multi_query(
                        idea_for_search, 
                        max_results=papers_per_query
                    )
                    source_papers = papers_summary.get('papers', [])
                    
                    # Store query variants for compatibility and reporting
                    if not job.state.query_variants:  # Don't overwrite if arxiv already set
                        job.state.query_variants = papers_summary.get('query_variants', [])
                        job.state.keywords = [v['query'] for v in job.state.query_variants]
                    
                    # Store source-specific stats
                    job.state.semantic_scholar_stats = {
                        'total_found': papers_summary.get('total_found', 0),
                        'unique_after_dedup': papers_summary.get('unique_after_dedup', 0),
                        'query_variants_count': len(papers_summary.get('query_variants', []))
                    }
                    
                else:
                    # Use standardized interface for other sources
                    source_papers = source.search_papers(
                        idea_for_search, 
                        max_results=papers_per_query
                    )
                
                all_papers.extend(source_papers)
                source_stats[source_name] = len(source_papers)
                logger.info(f"Retrieved {len(source_papers)} papers from {source_name}")
                
            except Exception as e:
                logger.error(f"Error searching {source_name}: {e}")
                source_stats[source_name] = 0
                continue
        
        # Deduplicate papers across sources
        unique_papers = service._deduplicate_papers(all_papers)
        job.state.total_papers_fetched = len(all_papers)
        job.state.unique_papers_after_dedup = len(unique_papers)
        
        # Store source statistics
        job.state.source_stats = source_stats
        
        update_progress(
            job_id,
            f"Found {len(unique_papers)} unique papers from {', '.join(available_sources)}",
            0.30
        )

        update_progress(job_id, "Running semantic search with embeddings + rerank...", 0.40)

        # Convert unique papers to format expected by search_literature
        papers_for_search = service._convert_to_search_format(unique_papers)
        
        search_results = cls._query_wrapper.search_literature(
            idea_for_search,
            include_scores=True,
            embedding_topk=embedding_topk,
            rerank_topk=rerank_topk,
            force_rebuild=True,
            papers_data=papers_for_search  # Pass our multi-source papers
        )

        search_results_list = json.loads(search_results)
        if not isinstance(search_results_list, list):
            logger.error(f"search_literature returned unexpected format: {type(search_results_list)}")
            search_results_list = []
        job.state.papers_after_rerank = len(search_results_list)

        update_progress(job_id, f"Semantic search: {embedding_topk} -> {job.state.papers_after_rerank}", 0.50)

        update_progress(job_id, f"LLM selecting final {final_papers_count} papers...", 0.55)

        selected_json = cls._paper_selector.generate_relevant_paper_selector_response(
            idea_for_search,
            search_results,
            final_count=final_papers_count
        )
        selected_papers_data = json.loads(selected_json)

        # Log LLM selection details
        logger.info(f"LLM selected {len(selected_papers_data)} papers from {job.state.papers_after_rerank} candidates")
        for i, paper in enumerate(selected_papers_data):
            title = paper.get('title', 'Unknown')[:60] + "..." if len(paper.get('title', '')) > 60 else paper.get('title', '')
            logger.info(f"Selected paper {i+1}: {title}")

        job.state.all_papers = search_results_list if isinstance(search_results_list, list) else []

        # Build search funnel data for transparency
        search_funnel = {
            'query_variants_count': len(job.state.query_variants),
            'query_variants': job.state.query_variants,
            'total_papers_fetched': job.state.total_papers_fetched,
            'unique_papers_after_dedup': job.state.unique_papers_after_dedup,
            'papers_after_rerank': job.state.papers_after_rerank,
            'final_papers_selected': len(selected_papers_data)
        }
        
        # Calculate efficiency rates
        if job.state.total_papers_fetched > 0:
            search_funnel['deduplication_rate'] = 1 - (job.state.unique_papers_after_dedup / job.state.total_papers_fetched)
        else:
            search_funnel['deduplication_rate'] = 0
            
        if job.state.unique_papers_after_dedup > 0:
            search_funnel['semantic_filter_rate'] = 1 - (job.state.papers_after_rerank / job.state.unique_papers_after_dedup)
        else:
            search_funnel['semantic_filter_rate'] = 0
            
        if job.state.papers_after_rerank > 0:
            search_funnel['llm_selection_rate'] = 1 - (len(selected_papers_data) / job.state.papers_after_rerank)
        else:
            search_funnel['llm_selection_rate'] = 0
        
        # Store funnel data for report generation
        job.state.search_funnel = search_funnel

        # Convert to Paper models
        papers = []
        for i, pd in enumerate(selected_papers_data):
            # Extract source information from the paper data
            source = pd.get('source', 'arxiv')
            source_id = pd.get('source_id', pd.get('id', ''))
            
            # Handle URL construction based on source
            url = pd.get('url', '')
            if not url and source_id:
                if source == 'arxiv':
                    url = f"https://arxiv.org/abs/{source_id}"
                elif source == 'semantic_scholar':
                    url = f"https://www.semanticscholar.org/paper/{source_id}"
            
            # Handle PDF URL construction
            pdf_url = pd.get('pdf_url', '')
            if not pdf_url and source_id and source == 'arxiv':
                pdf_url = url.replace('/abs/', '/pdf/') if url else f"https://arxiv.org/pdf/{source_id}"

            paper = Paper(
                paper_id=f"paper_{i+1:02d}",
                arxiv_id=source_id if source == 'arxiv' else '',  # Keep for compatibility
                title=pd.get('title', ''),
                abstract=pd.get('abstract', ''),
                url=url,
                pdf_url=pdf_url,
                authors=pd.get('authors', []),
                categories=pd.get('categories', []),
                published_date=pd.get('published_date', str(pd.get('year', '')) if pd.get('year') else None),
                source=source,
                source_id=source_id,
                year=pd.get('year'),
                venue=pd.get('venue'),
                citation_count=pd.get('citation_count'),
                doi=pd.get('doi')
            )
            papers.append(paper)

        job.state.selected_papers = papers
        update_progress(job_id, f"Selected {len(papers)} papers for detailed analysis", 0.65)

    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate papers across multiple sources based on title similarity and DOI.
        
        Args:
            papers: List of paper dictionaries from various sources
            
        Returns:
            List of unique papers
        """
        seen_titles = set()
        seen_dois = set()
        unique_papers = []
        
        for paper in papers:
            # Normalize title for comparison (lowercase, remove extra spaces)
            title = paper.get('title', '').lower().strip()
            doi = paper.get('doi', '').lower().strip()
            
            # Check if we've seen this paper before
            is_duplicate = False
            
            if title and title in seen_titles:
                is_duplicate = True
            elif doi and doi in seen_dois:
                is_duplicate = True
            
            if not is_duplicate:
                unique_papers.append(paper)
                if title:
                    seen_titles.add(title)
                if doi:
                    seen_dois.add(doi)
        
        logger.info(f"Deduplicated {len(papers)} papers to {len(unique_papers)} unique papers")
        return unique_papers
    
    def _convert_to_search_format(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert papers from various sources to format expected by search_literature.
        
        Args:
            papers: List of standardized paper dictionaries
            
        Returns:
            List of papers in search format
        """
        search_papers = []
        
        for paper in papers:
            search_paper = {
                "id": paper.get('source_id', paper.get('id', '')),
                "title": paper.get('title', ''),
                "abstract": paper.get('abstract', ''),
                "url": paper.get('url', ''),
                "year": paper.get('year'),
                "categories": paper.get('categories', []),
                "source": paper.get('source', ''),
            }
            search_papers.append(search_paper)
        
        return search_papers
