"""
Google Patents API client using SerpApi.
Implements BasePaperSource interface for patent search and retrieval.
"""
import requests
import logging
import time
from typing import List, Dict, Any, Optional

from core import config
from .base_paper_source import BasePaperSource
from app.models.paper import Paper

logger = logging.getLogger(__name__)


class GooglePatentsClient(BasePaperSource):
    """Client for interacting with Google Patents API via SerpApi."""
    
    BASE_URL = "https://serpapi.com/search"
    DETAILS_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: str = None, delay_between_requests: float = 0.1):
        """
        Initialize Google Patents client.

        Args:
            api_key: SerpApi API key (defaults to config.SERPAPI_API_KEY)
            delay_between_requests: Seconds to wait between API calls
        """
        self.api_key = api_key or config.SERPAPI_API_KEY
        self.delay = delay_between_requests
        self._last_request_time = 0
    
    @property
    def source_name(self) -> str:
        """Return the name of the paper source."""
        return "google_patents"
    
    @property
    def is_available(self) -> bool:
        """Check if the source is properly configured and available."""
        return bool(self.api_key)
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()
    
    def search_papers(
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
            logger.error("Google Patents client not available - missing API key")
            return []
        
        self._wait_for_rate_limit()
        
        params = {
            'engine': 'google_patents',
            'q': query,
            'num': max(min(max_results, 100), 10),  # SerpApi requires num between 10-100
            'api_key': self.api_key,
            **kwargs
        }
        
        logger.info(f"Searching Google Patents: {query} (max {max_results} results)")
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check if data is a string (error message) or dict
            if isinstance(data, str):
                logger.error(f"Google Patents API returned string instead of JSON: {data}")
                return []
            
            if not isinstance(data, dict):
                logger.error(f"Google Patents API returned unexpected type: {type(data)}")
                return []
            
            if 'error' in data:
                logger.error(f"Google Patents API error: {data['error']}")
                return []
            
            return self._parse_search_results(data)
            
        except requests.RequestException as e:
            logger.error(f"Google Patents search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Google Patents search: {e}")
            return []
    
    def get_paper_details(self, patent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed patent information.
        
        Args:
            patent_id: Patent ID (e.g., "patent/US11734097B1" or "US11734097B1")
            
        Returns:
            Detailed patent information or None if not found
        """
        if not self.is_available:
            logger.error("Google Patents client not available - missing API key")
            return None
        
        self._wait_for_rate_limit()
        
        # Normalize patent_id format
        if not patent_id.startswith('patent/'):
            patent_id = f"patent/{patent_id}"
        
        params = {
            'engine': 'google_patents_details',
            'patent_id': patent_id,
            'api_key': self.api_key
        }
        
        logger.info(f"Getting Google Patents details: {patent_id}")
        
        try:
            response = requests.get(self.DETAILS_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check if data is a string (error message) or dict
            if isinstance(data, str):
                logger.error(f"Google Patents details API returned string instead of JSON: {data}")
                return None
            
            if not isinstance(data, dict):
                logger.error(f"Google Patents details API returned unexpected type: {type(data)}")
                return None
            
            if 'error' in data:
                logger.error(f"Google Patents details API error: {data['error']}")
                return None
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"Google Patents details request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting patent details: {e}")
            return None
    
    def convert_to_paper_models(
        self,
        paper_dicts: List[Dict[str, Any]],
        limit: int = None
    ) -> List[Paper]:
        """
        Convert patent data to Paper model objects.
        
        Args:
            paper_dicts: List of patent dictionaries from search
            limit: Maximum number of papers to convert
            
        Returns:
            List of Paper model objects
        """
        limit = limit or len(paper_dicts)
        papers = []
        
        logger.info(f"Converting {len(paper_dicts)} patent dictionaries to Paper models")
        
        for i, pd in enumerate(paper_dicts[:limit]):
            try:
                # Debug: Check what type pd is
                if not isinstance(pd, dict):
                    logger.error(f"Expected dict, got {type(pd)}: {pd}")
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
                    metadata={  # Add metadata field for patent-specific info
                        'family_id': pd.get('family_id'),
                        'cited_by': pd.get('cited_by', []),
                        'legal_events': pd.get('legal_events', [])
                    }
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error converting patent to Paper model: {e}")
                logger.error(f"Patent data: {pd}")
                continue
        
        logger.info(f"Converted {len(papers)} patents to Paper models")
        return papers
    
    def fetch_description(self, patent_id: str) -> str:
        """
        Fetch patent abstract + description text via the details API.
        Returns structured markdown suitable for chunking, or empty string on failure.
        Avoids downloading the full patent PDF.
        """
        details = self.get_paper_details(patent_id)
        if not details:
            return ""

        parts = []
        abstract = details.get('abstract') or ''
        description = details.get('description') or ''

        if abstract:
            parts.append(f"## Abstract\n\n{abstract}")
        if description:
            parts.append(f"## Description\n\n{description}")

        return '\n\n'.join(parts)

    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse Google Patents search results.
        
        Args:
            data: Raw JSON response from SerpApi
            
        Returns:
            List of patent dictionaries
        """
        patents = []
        
        for result in data.get('organic_results', []):
            try:
                patent_data = {
                    'patent_id': result.get('patent_id', ''),
                    'title': result.get('title', ''),
                    'abstract': result.get('snippet', ''),
                    'link': result.get('patent_link', ''),  # Use patent_link instead of link
                    'patent_number': result.get('publication_number', ''),
                    'publication_date': result.get('publication_date'),
                    'filing_date': result.get('filing_date'),
                    'assignee': result.get('assignee') if isinstance(result.get('assignee'), str) else None,
                    'inventors': [result.get('inventor', '')] if isinstance(result.get('inventor'), str) else [],
                    'classification': [],  # classification is None in current response
                    'status': result.get('status', ''),
                    'pdf_link': result.get('pdf', '')  # Use the correct 'pdf' field from SerpApi
                }
                
                # Only include patents with basic required fields
                if patent_data['patent_id'] and patent_data['title']:
                    patents.append(patent_data)
                    
            except Exception as e:
                logger.error(f"Error parsing patent result: {e}")
                continue
        
        logger.info(f"Parsed {len(patents)} patents from search results")
        return patents
