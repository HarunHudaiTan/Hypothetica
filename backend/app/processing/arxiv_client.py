"""
ArXiv API client for searching and retrieving papers.
Uses the official `arxiv` Python library which handles rate limiting automatically.
Implements BasePaperSource interface.
"""
import logging
from typing import List, Dict, Optional, Any

import arxiv as arxiv_lib

from core import config
from app.models.paper import Paper
from .base_paper_source import BasePaperSource

logger = logging.getLogger(__name__)


class ArxivClient(BasePaperSource):
    """
    Client for interacting with the arXiv API.
    Uses the official arxiv library for automatic rate limiting (5s between requests, 10 retries).
    """

    # Single shared Client — reusing it keeps rate limiting state intact across calls
    _client = arxiv_lib.Client(
        page_size=100,
        delay_seconds=5.0,
        num_retries=10,
    )

    @property
    def source_name(self) -> str:
        return "arxiv"

    @property
    def is_available(self) -> bool:
        return True  # arXiv API is always available (no API key needed)

    def __init__(self):
        pass

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_field: str = "all",
        sort_by: str = "relevance",
        sort_order: str = "descending"
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
        logger.info(f"Searching arXiv: {search_query[:80]} (max {max_results} results)")

        try:
            search = arxiv_lib.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_criterion,
                sort_order=sort_ord,
            )
            results = list(self._client.results(search))
            logger.info(f"ArXiv returned {len(results)} results")
            return [self._result_to_dict(r) for r in results]
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
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

    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        return self.search(query=query, max_results=max_results)

    def search_multiple_keywords(
        self,
        keywords: List[str],
        results_per_keyword: int = None
    ) -> List[Dict[str, Any]]:
        results_per_keyword = results_per_keyword or config.PAPERS_PER_KEYWORD
        all_papers = []
        seen_ids = set()

        for keyword in keywords:
            logger.info(f"Searching for keyword: {keyword}")
            papers = self.search(query=keyword, max_results=results_per_keyword)

            for paper in papers:
                arxiv_id = paper.get('arxiv_id', '')
                if arxiv_id and arxiv_id not in seen_ids:
                    paper['search_keyword'] = keyword
                    all_papers.append(paper)
                    seen_ids.add(arxiv_id)

        logger.info(f"Found {len(all_papers)} unique papers from {len(keywords)} keywords")
        return all_papers

    def convert_to_paper_models(
        self,
        paper_dicts: List[Dict[str, Any]],
        limit: int = None
    ) -> List[Paper]:
        limit = limit or config.MAX_PAPERS_TO_ANALYZE
        papers = []

        for i, pd in enumerate(paper_dicts[:limit]):
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

    def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        try:
            search = arxiv_lib.Search(id_list=[arxiv_id])
            results = list(self._client.results(search))
            if results:
                return self.convert_to_paper_models(
                    [self._result_to_dict(results[0])], limit=1
                )[0]
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve paper {arxiv_id}: {e}")
            return None

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        paper = self.get_paper_by_id(paper_id)
        if paper:
            return {
                'arxiv_id': paper.source_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'authors': paper.authors,
                'categories': paper.categories,
                'published_date': paper.published_date,
                'url': paper.url,
                'pdf_url': paper.pdf_url,
                'metadata': paper.metadata
            }
        return None
