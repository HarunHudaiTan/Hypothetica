import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Dict, Any

from app.api.managers.job_manager import job_manager
from app.models.paper import Paper

from app.processing.base_paper_source import BasePaperSource
from app.processing.arxiv_client import ArxivClient
from app.processing.google_patents_client import GooglePatentsClient
from app.retrieval.paper_search import QueryWrapper
from app.agents.relevant_paper_selector_agent import RelevantPaperSelectorAgent
from app.agents.query_variant_agent import QueryVariantAgent

logger = logging.getLogger(__name__)


class PaperSearchService:
    """Enhanced paper search service supporting multiple sources."""
    
    def __init__(self):
        """Initialize available paper sources."""
        self.sources: Dict[str, BasePaperSource] = {
            "arxiv": ArxivClient(),
            "google_patents": GooglePatentsClient()
        }
        # Note: Don't filter out unavailable sources here - we need them for the API
        
        self._query_wrapper = QueryWrapper(use_reranker=True)
        self._paper_selector = RelevantPaperSelectorAgent()
        self._query_variant_agent = QueryVariantAgent()

    @classmethod
    def search_papers(cls, job_id: str, update_progress: Callable, selected_sources: List[str] = None):
        """
        Enhanced paper search with multiple source support.
        
        Args:
            job_id: Job identifier
            update_progress: Progress callback function
            selected_sources: List of source names to search (default: all available)
        """
        # Initialize service instance
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
        
        # Determine which sources to use
        selected_sources = selected_sources or list(service.sources.keys())
        available_sources = [s for s in selected_sources if s in service.sources and service.sources[s].is_available]
        
        if not available_sources:
            logger.error("No available paper sources selected")
            update_progress(job_id, "Error: No available paper sources", 0.15)
            return
        
        logger.info(f"Searching sources: {available_sources}")
        update_progress(job_id, f"Generating query variants for {len(available_sources)} source(s)...", 0.15)
        
        # Generate query variants for better recall
        query_variants = service._query_variant_agent.generate_query_variants(idea_for_search)
        logger.info(f"Generated {len(query_variants)} query variants: {[v['query'] for v in query_variants]}")
        job.state.query_variants = query_variants
        
        update_progress(job_id, f"Searching {len(available_sources)} source(s)...", 0.20)

        # Search papers from all selected sources
        all_papers = []
        source_results = {}
        
        for source_name in available_sources:
            source = service.sources[source_name]
            update_progress(job_id, f"Searching {source_name}...", 0.25)
            
            # Use query variants for better coverage — run in parallel
            source_papers = []
            max_per_variant = papers_per_query // max(len(query_variants), 1)

            def _search_variant(variant):
                q = variant['query']
                logger.info(f"Searching {source_name} with query: {q}")
                raw = source.search_papers(q, max_results=max_per_variant)
                logger.info(f"  [{source_name}] query='{q[:80]}' → {len(raw)} raw results")
                return source.convert_to_paper_models(raw, limit=50)

            with ThreadPoolExecutor(max_workers=min(len(query_variants), 4)) as executor:
                futures = {executor.submit(_search_variant, v): v for v in query_variants}
                for future in as_completed(futures):
                    try:
                        source_papers.extend(future.result())
                    except Exception as e:
                        logger.error(f"[{source_name}] Variant search failed: {e}")
            
            # Deduplicate papers within the same source
            seen_ids = set()
            unique_papers = []
            for paper in source_papers:
                if paper.source_id not in seen_ids:
                    seen_ids.add(paper.source_id)
                    # Patents without PDF URLs are still usable — text is fetched via details API
                    unique_papers.append(paper)
            
            all_papers.extend(unique_papers)
            source_results[source_name] = unique_papers
            
            # Log PDF filtering results
            total_before_filter = len(source_papers)
            total_after_filter = len(unique_papers)
            if total_before_filter != total_after_filter:
                excluded_count = total_before_filter - total_after_filter
                logger.info(f"Filtered out {excluded_count} papers from {source_name} due to missing PDFs")
            
            logger.info(f"Found {len(unique_papers)} unique papers from {source_name}")
            update_progress(job_id, f"Found {len(unique_papers)} papers from {source_name}", 0.30)
        
        # Store source information in job state
        job.state.selected_sources = available_sources
        job.state.source_results = {source: len(papers) for source, papers in source_results.items()}
        job.state.total_papers_fetched = len(all_papers)

        update_progress(
            job_id,
            f"Found {len(all_papers)} total papers from {len(available_sources)} source(s)",
            0.35
        )

        # Convert all papers to JSONL format for semantic search
        jsonl_papers = service._convert_papers_to_jsonl(all_papers)
        service._save_papers_to_jsonl(jsonl_papers)

        update_progress(job_id, "Running semantic search with embeddings + rerank...", 0.40)

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
        job.state.papers_after_rerank = len(search_results_list)

        update_progress(job_id, f"Semantic search: {embedding_topk} -> {job.state.papers_after_rerank}", 0.50)

        update_progress(job_id, f"LLM selecting final {final_papers_count} papers...", 0.55)

        selected_json = service._paper_selector.generate_relevant_paper_selector_response(
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
            'selected_sources': available_sources,
            'source_results': job.state.source_results,
            'total_papers_fetched': job.state.total_papers_fetched,
            'papers_after_rerank': job.state.papers_after_rerank,
            'final_papers_selected': len(selected_papers_data)
        }
        
        # Calculate efficiency rates
        if job.state.total_papers_fetched > 0:
            search_funnel['semantic_filter_rate'] = 1 - (job.state.papers_after_rerank / job.state.total_papers_fetched)
        else:
            search_funnel['semantic_filter_rate'] = 0
            
        if job.state.papers_after_rerank > 0:
            search_funnel['llm_selection_rate'] = 1 - (len(selected_papers_data) / job.state.papers_after_rerank)
        else:
            search_funnel['llm_selection_rate'] = 0
        
        # Store funnel data for report generation
        job.state.search_funnel = search_funnel

        # Convert LLM selection back to Paper objects
        papers = []
        
        for i, pd in enumerate(selected_papers_data):
            source_id = pd.get('id', '')
            
            # Determine source from source_id prefix or URL
            source = "unknown"
            if source_id and source_id.startswith('patent/'):
                source = "google_patents"
            elif source_id and (source_id.startswith('cs.') or (len(source_id.split('.')[0]) == 4 and source_id.split('.')[0].isdigit())):
                source = "arxiv"
            elif 'patent' in pd.get('url', '').lower():
                source = "google_patents"

            url = pd.get('url', '')
            if not url and source == "arxiv" and source_id:
                url = f"https://arxiv.org/abs/{source_id}"
            pdf_url = url.replace('/abs/', '/pdf/') if url and source == "arxiv" else pd.get('pdf_url')

            # arXiv papers must have a PDF URL; patents get text via API so pdf_url is optional
            if not pdf_url and source != "google_patents":
                logger.warning(f"FILTERING OUT paper {i+1} ({source}) - no PDF URL available")
                continue
            
            paper = Paper(
                paper_id=f"paper_{i+1:02d}",
                source=source,
                source_id=source_id,
                title=pd.get('title', ''),
                abstract=pd.get('abstract', ''),
                url=url,
                pdf_url=pdf_url,
                authors=pd.get('authors', []),
                categories=pd.get('categories', []),
                published_date=str(pd.get('year', '')) if pd.get('year') else None
            )
            papers.append(paper)

        # Ensure we have enough papers after filtering, if not, get more from candidates
        if len(papers) < final_papers_count:
            logger.warning(f"Only {len(papers)} papers selected. Need {final_papers_count}.")
            for result in search_results_list[len(selected_papers_data):]:
                if len(papers) >= final_papers_count:
                    break

                source_id = result.get('id', '')
                pdf_url = result.get('pdf_url')

                # Determine source from source_id prefix or URL
                source = "unknown"
                if source_id and source_id.startswith('patent/'):
                    source = "google_patents"
                elif source_id and (source_id.startswith('cs.') or (len(source_id.split('.')[0]) == 4 and source_id.split('.')[0].isdigit())):
                    source = "arxiv"
                elif 'patent' in result.get('url', '').lower():
                    source = "google_patents"

                # arXiv must have pdf_url; patents are fine without one
                if not pdf_url and source != "google_patents":
                    continue

                url = result.get('url', '')
                if not url and source == "arxiv" and source_id:
                    url = f"https://arxiv.org/abs/{source_id}"
                pdf_url_final = url.replace('/abs/', '/pdf/') if url and source == "arxiv" else pdf_url

                logger.info(f"Adding additional paper: {result.get('title', 'Unknown')[:40]}...")
                additional_paper = Paper(
                    paper_id=f"paper_{len(papers)+1:02d}",
                    source=source,
                    source_id=source_id,
                    title=result.get('title', ''),
                    abstract=result.get('abstract', ''),
                    url=url,
                    pdf_url=pdf_url_final,
                    authors=result.get('authors', []),
                    categories=result.get('categories', []),
                    published_date=str(result.get('year', '')) if result.get('year') else None
                )
                papers.append(additional_paper)

        job.state.selected_papers = papers
        logger.info(f"Final selection - {len(papers)} papers, with PDFs: {sum(1 for p in papers if p.pdf_url)}")
        update_progress(job_id, f"Selected {len(papers)} papers for detailed analysis", 0.60)
    
    def _convert_papers_to_jsonl(self, papers: List[Paper]) -> List[Dict[str, Any]]:
        """Convert Paper models to JSONL format for semantic search."""
        jsonl_papers = []
        
        for paper in papers:
            year = None
            if paper.published_date:
                try:
                    # Extract year from various date formats
                    if len(paper.published_date) >= 4:
                        year = int(paper.published_date[:4])
                except (ValueError, TypeError):
                    year = None

            jsonl_entry = {
                "id": paper.source_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "year": year,
                "categories": paper.categories,
                "source": paper.source,
            }

            jsonl_papers.append(jsonl_entry)

        return jsonl_papers
    
    def _save_papers_to_jsonl(self, jsonl_papers: List[Dict[str, Any]], filename: str = None):
        """Save papers to JSONL file format."""
        import os

        if filename is None:
            # Write to same directory where paper_search.py reads from
            retrieval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'retrieval')
            filename = os.path.join(retrieval_dir, "sample_papers.jsonl")

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            for paper in jsonl_papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(jsonl_papers)} papers to {filename}")
    
    @classmethod
    def get_available_sources(cls) -> Dict[str, bool]:
        """Get all available paper sources and their availability status."""
        service = cls()
        return {name: source.is_available for name, source in service.sources.items()}
