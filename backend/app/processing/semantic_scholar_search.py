import requests
import json
import time
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

from app.processing.base_paper_source import PaperSource
from app.agents.query_variant_agent import QueryVariantAgent

logger = logging.getLogger(__name__)


class SemanticScholarSource(PaperSource):
    """
    Semantic Scholar paper retrieval with query variant expansion.
    Implements PaperSource interface for modular design.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_delay = 1.0  # Seconds between requests
        self._last_request_time = 0
        self.query_variant_agent = QueryVariantAgent()
        
        # Default fields to retrieve from Semantic Scholar API
        self.default_fields = [
            "paperId",
            "title", 
            "abstract",
            "authors",
            "url",
            "openAccessPdf",
            "publicationDate",
            "year",
            "venue",
            "fieldsOfStudy",
            "citationCount",
            "publicationTypes"
        ]

    def _wait_for_rate_limit(self):
        """Ensure minimum delay between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.api_delay:
            time.sleep(self.api_delay - elapsed)
        self._last_request_time = time.time()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including API key if available."""
        headers = {"User-Agent": "Hypothetica/1.0 (research originality assessment)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def search_papers_multi_query(self, query: str, max_results: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Search for papers using multiple query variants for high recall.
        
        Args:
            query: Original search query
            max_results: Maximum total results to return
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with papers and query variants (for compatibility with ArXiv)
        """
        # Generate query variants
        query_variants = self.query_variant_agent.generate_query_variants(query)
        logger.info(f"Generated {len(query_variants)} query variants for Semantic Scholar search")
        
        all_papers = []
        papers_per_variant = max(10, max_results // len(query_variants))
        
        # Search each query variant
        for i, variant in enumerate(query_variants):
            logger.info(f"Semantic Scholar variant {i+1}/{len(query_variants)}: {variant['query']}")
            
            try:
                variant_papers = self.search_papers(
                    variant['query'], 
                    max_results=papers_per_variant,
                    require_pdf=False  # Get all papers, no PDF filtering
                )
                all_papers.extend(variant_papers)
                
            except Exception as e:
                logger.error(f"Error searching Semantic Scholar variant '{variant['query']}': {e}")
                continue
        
        # Deduplicate papers
        unique_papers = self._deduplicate_papers(all_papers)
        
        return {
            'papers': unique_papers[:max_results],
            'query_variants': query_variants,
            'total_found': len(all_papers),
            'unique_after_dedup': len(unique_papers)
        }
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate papers based on title and paper ID."""
        seen_titles = set()
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            paper_id = paper.get('source_id', '')
            
            if title not in seen_titles and paper_id not in seen_ids:
                unique_papers.append(paper)
                seen_titles.add(title)
                if paper_id:
                    seen_ids.add(paper_id)
        
        return unique_papers

    def search_papers(self, query: str, max_results: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for papers using Semantic Scholar bulk search API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional parameters (year, fields_of_study, require_pdf, etc.)
            
        Returns:
            List of standardized paper dictionaries
        """
        self._wait_for_rate_limit()
        
        # Build query parameters
        params = {
            "query": query,
            "fields": ",".join(self.default_fields),
            "limit": min(100, max_results)  # API max per request is 100
        }
        
        # Add optional filters
        if "year" in kwargs:
            params["year"] = kwargs["year"]
        elif "year_from" in kwargs:
            params["year"] = f"{kwargs['year_from']}-"
            
        if "fields_of_study" in kwargs:
            params["fieldsOfStudy"] = kwargs["fields_of_study"]
            
        if "min_citations" in kwargs:
            params["minCitationCount"] = kwargs["min_citations"]
            
        if "publication_types" in kwargs:
            params["publicationTypes"] = kwargs["publication_types"]

        url = f"{self.base_url}/paper/search/bulk"
        
        try:
            all_papers = []
            remaining = max_results
            offset = 0
            
            # Fetch more papers initially to account for PDF filtering
            fetch_limit = min(1000, max_results * 3)  # Fetch 3x more to filter
            
            while remaining > 0 and len(all_papers) < fetch_limit:
                if offset > 0:
                    params["offset"] = offset
                    
                response = requests.get(
                    url, 
                    params=params, 
                    headers=self._get_headers(),
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"Semantic Scholar API error: {response.status_code} - {response.text}")
                    break
                
                data = response.json()
                batch_papers = data.get("data", [])
                
                if not batch_papers:
                    break
                
                # Standardize and filter papers
                for paper_data in batch_papers:
                    standardized = self.standardize_paper_data(paper_data)
                    
                    # Filter for PDF availability if requested
                    require_pdf = kwargs.get("require_pdf", False)
                    if require_pdf and not standardized.get("pdf_url"):
                        continue
                    
                    all_papers.append(standardized)
                    
                    if len(all_papers) >= max_results:
                        break
                
                offset += len(batch_papers)
                
                # Check if we got all results
                if len(batch_papers) < params.get("limit", 100):
                    break
            
            logger.info(f"Semantic Scholar search found {len(all_papers)} papers for query: {query}")
            return all_papers[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Standardized paper dictionary or None if not found
        """
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/paper/{paper_id}"
        params = {"fields": ",".join(self.default_fields)}
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                paper_data = response.json()
                return self.standardize_paper_data(paper_data)
            else:
                logger.error(f"Failed to get paper details: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting paper details: {e}")
            return None

    def download_pdf(self, paper_id: str) -> Optional[str]:
        """
        Download PDF for a paper if available through open access.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Local file path to downloaded PDF or None if unavailable
        """
        # First get paper details to find PDF URL
        paper_details = self.get_paper_details(paper_id)
        if not paper_details:
            return None
            
        pdf_url = paper_details.get('pdf_url')
        if not pdf_url:
            logger.info(f"No open access PDF available for paper {paper_id}")
            return None
            
        # Use existing PDF download logic from pdf_processor
        try:
            from app.processing.pdf_processor import PDFProcessor
            processor = PDFProcessor()
            return processor.download_pdf(pdf_url, f"semantic_scholar_{paper_id}")
        except Exception as e:
            logger.error(f"Error downloading PDF for {paper_id}: {e}")
            return None

    def get_source_name(self) -> str:
        """Return the source name."""
        return "semantic_scholar"

    def standardize_paper_data(self, raw_paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Semantic Scholar paper data to standardized format.
        
        Args:
            raw_paper: Raw paper data from Semantic Scholar API
            
        Returns:
            Standardized paper dictionary
        """
        # Extract authors
        authors = []
        if "authors" in raw_paper:
            authors = [author.get("name", "") for author in raw_paper["authors"] if author.get("name")]
            
        # Extract PDF URL
        pdf_url = None
        if "openAccessPdf" in raw_paper and raw_paper["openAccessPdf"]:
            pdf_url = raw_paper["openAccessPdf"].get("url")
            
        # Extract publication date
        published_date = raw_paper.get("publicationDate") or str(raw_paper.get("year", ""))
        
        # Extract categories/fields of study
        categories = []
        if "fieldsOfStudy" in raw_paper:
            categories = raw_paper["fieldsOfStudy"]
        elif "venue" in raw_paper and raw_paper["venue"]:
            categories = [raw_paper["venue"]]
            
        # Create standardized paper
        standardized = {
            "id": f"ss_{raw_paper.get('paperId', '')}",
            "title": raw_paper.get("title", ""),
            "abstract": raw_paper.get("abstract", ""),
            "authors": authors,
            "url": raw_paper.get("url", ""),
            "pdf_url": pdf_url,
            "published_date": published_date,
            "categories": categories,
            "source": self.get_source_name(),
            "source_id": raw_paper.get("paperId", ""),
            "citation_count": raw_paper.get("citationCount", 0),
            "venue": raw_paper.get("venue", ""),
            "publication_types": raw_paper.get("publicationTypes", []),
            "year": raw_paper.get("year"),
        }
        
        return standardized
