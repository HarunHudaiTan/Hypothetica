"""
ArXiv evidence adapter.
Wraps the ArXiv API client to implement the EvidenceAdapter interface.
"""
import logging
from typing import List, Dict, Any

import arxiv as arxiv_lib

from app.models.paper import Paper
from .base_adapter import EvidenceAdapter

logger = logging.getLogger(__name__)


class ArxivAdapter(EvidenceAdapter):
    """
    Adapter for arXiv academic papers and preprints.
    Uses the official arxiv library for automatic rate limiting.
    """
    
    # Single shared Client — reusing it keeps rate limiting state intact across calls
    _client = arxiv_lib.Client(
        page_size=100,
        delay_seconds=5.0,
        num_retries=10,
    )
    
    @property
    def name(self) -> str:
        return "arxiv"
    
    @property
    def description(self) -> str:
        return "arXiv academic papers and preprints"

    @property
    def display_name(self) -> str:
        return "arXiv"
    
    @property
    def is_available(self) -> bool:
        return True  # arXiv API is always available (no API key needed)
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        search_field: str = "all",
        sort_by: str = "relevance",
        sort_order: str = "descending",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search arXiv using the official arxiv library (rate limiting handled automatically)."""
        sort_criterion = {
            "relevance": arxiv_lib.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv_lib.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv_lib.SortCriterion.SubmittedDate,
        }.get(sort_by, arxiv_lib.SortCriterion.Relevance)
        
        sort_ord = (
            arxiv_lib.SortOrder.Ascending
            if sort_order == "ascending"
            else arxiv_lib.SortOrder.Descending
        )
        
        # Prefix with field specifier if not already present
        search_query = query if ":" in query else f"{search_field}:{query}"
        logger.info(f"[ArxivAdapter] Searching: {search_query[:80]} (max {max_results} results)")
        
        try:
            search = arxiv_lib.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_criterion,
                sort_order=sort_ord,
            )
            results = list(self._client.results(search))
            logger.info(f"[ArxivAdapter] Returned {len(results)} results")
            return [self._result_to_dict(r) for r in results]
        except Exception as e:
            logger.error(f"[ArxivAdapter] Search failed: {e}")
            return []
    
    def _result_to_dict(self, r: arxiv_lib.Result) -> Dict[str, Any]:
        """Convert arxiv.Result to the dict format the rest of the pipeline expects."""
        arxiv_id = r.entry_id.split("/abs/")[-1]
        return {
            "arxiv_id": arxiv_id,
            "title": r.title,
            "abstract": r.summary,
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
            "authors": [a.name for a in r.authors],
            "categories": r.categories,
            "primary_category": r.primary_category,
            "published_date": r.published.isoformat() if r.published else None,
        }
    
    def convert_to_papers(
        self,
        results: List[Dict[str, Any]],
        limit: int = None
    ) -> List[Paper]:
        """Convert arXiv search results to Paper objects."""
        cap = len(results) if limit is None else min(int(limit), len(results))
        papers = []
        
        for i, pd in enumerate(results[:cap]):
            paper = Paper(
                paper_id=f"paper_{i+1:02d}",
                source="arxiv",
                source_id=pd.get('arxiv_id', ''),
                title=pd.get('title', ''),
                abstract=pd.get('abstract', ''),
                url=pd.get('url', ''),
                pdf_url=pd.get('pdf_url', ''),
                authors=pd.get('authors', []),
                categories=pd.get('categories', []),
                published_date=pd.get('published_date'),
                metadata={
                    'primary_category': pd.get('primary_category'),
                }
            )
            papers.append(paper)
        
        return papers
