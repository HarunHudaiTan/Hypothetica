import json
import logging
from typing import List, Callable

from api.managers.job_manager import job_manager
from models.paper import Paper

from processing.arxiv_search import ArxivReq
from retrieval.paper_search import QueryWrapper
from Agents.relevant_paper_selector_agent import RelevantPaperSelectorAgent

logger = logging.getLogger(__name__)


class PaperSearchService:
    _arxiv_req = ArxivReq()
    _query_wrapper = QueryWrapper(use_reranker=True)
    _paper_selector = RelevantPaperSelectorAgent()

    @classmethod
    def search_papers(cls, job_id: str, update_progress: Callable):
        """Enhanced paper search with high-recall retrieval."""
        job = job_manager.get_job(job_id)
        if not job:
            return

        idea_for_search = job.state.enriched_idea or job.user_idea
        settings = job.settings or {}
        papers_per_query = settings.get("papers_per_query", 150)
        embedding_topk = settings.get("embedding_topk", 100)
        rerank_topk = settings.get("rerank_topk", 20)
        final_papers_count = settings.get("final_papers", 5)

        update_progress(job_id, "Generating query variants and searching arXiv...", 0.15)

        papers_json = cls._arxiv_req.get_papers(idea_for_search, papers_per_query=papers_per_query)
        papers_summary = json.loads(papers_json)

        job.state.query_variants = papers_summary.get('query_variants', [])
        job.state.keywords = [v['query'] for v in job.state.query_variants]
        job.state.total_papers_fetched = papers_summary.get('total_papers_fetched', 0)
        job.state.unique_papers_after_dedup = papers_summary.get('unique_papers', 0)

        update_progress(
            job_id,
            f"Found {job.state.unique_papers_after_dedup} unique papers",
            0.25
        )

        update_progress(job_id, "Running semantic search with embeddings + rerank...", 0.30)

        search_results = cls._query_wrapper.search_literature(
            idea_for_search,
            include_scores=True,
            embedding_topk=embedding_topk,
            rerank_topk=rerank_topk,
            force_rebuild=True
        )

        search_results_list = json.loads(search_results)
        job.state.papers_after_rerank = len(search_results_list) if isinstance(search_results_list, list) else 0

        update_progress(job_id, f"Semantic search: {embedding_topk} -> {job.state.papers_after_rerank}", 0.40)

        update_progress(job_id, f"LLM selecting final {final_papers_count} papers...", 0.45)

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

        # Convert to Paper models
        papers = []
        for i, pd in enumerate(selected_papers_data):
            arxiv_id = pd.get('id', '')
            url = pd.get('url', '')
            if not url and arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = url.replace('/abs/', '/pdf/') if url else f"https://arxiv.org/pdf/{arxiv_id}"

            paper = Paper(
                paper_id=f"paper_{i+1:02d}",
                arxiv_id=arxiv_id,
                title=pd.get('title', ''),
                abstract=pd.get('abstract', ''),
                url=url,
                pdf_url=pdf_url,
                authors=pd.get('authors', []),
                categories=pd.get('categories', []),
                published_date=str(pd.get('year', '')) if pd.get('year') else None
            )
            papers.append(paper)

        job.state.selected_papers = papers
        update_progress(job_id, f"Selected {len(papers)} papers for detailed analysis", 0.52)
