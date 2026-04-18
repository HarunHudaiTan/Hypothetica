"""
Google Patents evidence adapter.
Wraps the Google Patents API client to implement the EvidenceAdapter interface.
"""
import requests
import logging
import time
from typing import List, Dict, Any

from core import config
from app.models.paper import Paper
from .base_adapter import EvidenceAdapter

logger = logging.getLogger(__name__)


class PatentsAdapter(EvidenceAdapter):
    """
    Adapter for Google Patents database via SerpApi.
    """
    
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: str = None, delay_between_requests: float = 1.0):
        """
        Initialize Google Patents adapter.
        
        Args:
            api_key: SerpApi API key (defaults to config.SERPAPI_API_KEY)
            delay_between_requests: Seconds to wait between API calls
        """
        self.api_key = api_key or config.SERPAPI_API_KEY
        self.delay = delay_between_requests
        self._last_request_time = 0
    
    @property
    def name(self) -> str:
        return "google_patents"
    
    @property
    def description(self) -> str:
        return "Google Patents database"
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search Google Patents for patents matching a query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            **kwargs: Additional parameters (country, language, status, etc.)
            
        Returns:
            List of patent dictionaries
        """
        if not self.is_available:
            logger.error("[PatentsAdapter] Not available - missing API key")
            return []
        
        self._wait_for_rate_limit()
        
        params = {
            'engine': 'google_patents',
            'q': query,
            'num': max(min(max_results, 100), 10),  # SerpApi requires num between 10-100
            'api_key': self.api_key,
            **kwargs
        }
        
        logger.info(f"[PatentsAdapter] Searching: {query} (max {max_results} results)")
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check if data is a string (error message) or dict
            if isinstance(data, str):
                logger.error(f"[PatentsAdapter] API returned string instead of JSON: {data}")
                return []
            
            if not isinstance(data, dict):
                logger.error(f"[PatentsAdapter] API returned unexpected type: {type(data)}")
                return []
            
            if 'error' in data:
                logger.error(f"[PatentsAdapter] API error: {data['error']}")
                return []
            
            results = self._parse_search_results(data)
            logger.info(f"[PatentsAdapter] Returned {len(results)} results")
            return results
            
        except requests.RequestException as e:
            logger.error(f"[PatentsAdapter] Search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"[PatentsAdapter] Unexpected error: {e}")
            return []
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Google Patents search results."""
        patents = []
        
        for result in data.get('organic_results', []):
            try:
                patent_data = {
                    'patent_id': result.get('patent_id', ''),
                    'title': result.get('title', ''),
                    'abstract': result.get('snippet', ''),
                    'link': result.get('patent_link', ''),
                    'patent_number': result.get('publication_number', ''),
                    'publication_date': result.get('publication_date'),
                    'filing_date': result.get('filing_date'),
                    'assignee': result.get('assignee') if isinstance(result.get('assignee'), str) else None,
                    'inventors': [result.get('inventor', '')] if isinstance(result.get('inventor'), str) else [],
                    'classification': [],
                    'status': result.get('status', ''),
                    'pdf_link': result.get('pdf', '')
                }
                
                # Only include patents with basic required fields
                if patent_data['patent_id'] and patent_data['title']:
                    patents.append(patent_data)
                    
            except Exception as e:
                logger.error(f"[PatentsAdapter] Error parsing result: {e}")
                continue
        
        return patents
    
    def convert_to_papers(
        self,
        results: List[Dict[str, Any]],
        limit: int = None
    ) -> List[Paper]:
        """Convert patent search results to Paper objects."""
        limit = limit or len(results)
        papers = []
        
        logger.info(f"[PatentsAdapter] Converting {len(results)} patents to Paper models")
        
        for i, pd in enumerate(results[:limit]):
            try:
                if not isinstance(pd, dict):
                    logger.error(f"[PatentsAdapter] Expected dict, got {type(pd)}: {pd}")
                    continue
                
                paper = Paper(
                    paper_id=f"paper_{i+1:02d}",
                    source="google_patents",
                    source_id=pd.get('patent_id', ''),
                    title=pd.get('title', ''),
                    abstract=pd.get('abstract', ''),
                    url=pd.get('link', ''),
                    pdf_url=pd.get('pdf_link'),
                    authors=pd.get('inventors', []),
                    published_date=pd.get('publication_date'),
                    metadata={
                        'family_id': pd.get('family_id'),
                        'cited_by': pd.get('cited_by', []),
                        'legal_events': pd.get('legal_events', [])
                    }
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"[PatentsAdapter] Error converting patent: {e}")
                continue
        
        logger.info(f"[PatentsAdapter] Converted {len(papers)} patents to Paper models")
        return papers
