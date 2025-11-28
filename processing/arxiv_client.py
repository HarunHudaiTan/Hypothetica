"""
ArXiv API client for searching and retrieving papers.
Refactored from ArxivReq.py with cleaner interface.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

import config
from models.paper import Paper

logger = logging.getLogger(__name__)


class ArxivClient:
    """
    Client for interacting with the arXiv API.
    Handles searching, parsing, and paper selection.
    """
    
    # XML namespaces for arXiv API responses
    NAMESPACES = {
        'atom': 'http://www.w3.org/2005/Atom',
        'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, delay_between_requests: float = 3.0):
        """
        Initialize ArXiv client.
        
        Args:
            delay_between_requests: Seconds to wait between API calls (arXiv recommends 3s)
        """
        self.delay = delay_between_requests
        self._last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed arXiv rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        search_field: str = "all",
        sort_by: str = "relevance",
        sort_order: str = "descending"
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching a query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            search_field: Field to search ("all", "ti", "abs", "au")
            sort_by: Sort method ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort direction ("ascending", "descending")
            
        Returns:
            List of paper dictionaries
        """
        self._wait_for_rate_limit()
        
        # Build query
        search_query = f"{search_field}:{query}"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        logger.info(f"Searching arXiv: {query} (max {max_results} results)")
        
        try:
            response = urllib.request.urlopen(url, timeout=30)
            xml_content = response.read().decode('utf-8')
            return self._parse_response(xml_content)
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
    
    def search_multiple_keywords(
        self,
        keywords: List[str],
        results_per_keyword: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for multiple keywords and combine results.
        
        Args:
            keywords: List of search keywords
            results_per_keyword: Max results per keyword
            
        Returns:
            Combined list of unique papers
        """
        results_per_keyword = results_per_keyword or config.PAPERS_PER_KEYWORD
        all_papers = []
        seen_ids = set()
        
        for keyword in keywords:
            logger.info(f"Searching for keyword: {keyword}")
            papers = self.search(
                query=keyword,
                max_results=results_per_keyword
            )
            
            # Add unique papers only
            for paper in papers:
                arxiv_id = paper.get('arxiv_id', '')
                if arxiv_id and arxiv_id not in seen_ids:
                    paper['search_keyword'] = keyword
                    all_papers.append(paper)
                    seen_ids.add(arxiv_id)
        
        logger.info(f"Found {len(all_papers)} unique papers from {len(keywords)} keywords")
        return all_papers
    
    def _parse_response(self, xml_string: str) -> List[Dict[str, Any]]:
        """
        Parse arXiv API XML response into paper dictionaries.
        
        Args:
            xml_string: Raw XML response from arXiv
            
        Returns:
            List of paper dictionaries
        """
        papers = []
        
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
            return []
        
        for entry in root.findall('atom:entry', self.NAMESPACES):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)
        
        return papers
    
    def _parse_entry(self, entry: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a single arXiv entry into a paper dictionary."""
        ns = self.NAMESPACES
        
        try:
            # Extract ID
            id_elem = entry.find('atom:id', ns)
            if id_elem is None:
                return None
            
            full_id = id_elem.text
            arxiv_id = full_id.split('/abs/')[-1] if '/abs/' in full_id else full_id
            
            # Extract basic info
            title_elem = entry.find('atom:title', ns)
            title = ' '.join(title_elem.text.split()) if title_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', ns)
            abstract = ' '.join(summary_elem.text.split()) if summary_elem is not None else ""
            
            # Extract dates
            published = entry.find('atom:published', ns)
            published_date = published.text if published is not None else None
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            # Extract links
            url = ""
            pdf_url = ""
            for link in entry.findall('atom:link', ns):
                href = link.get('href', '')
                title_attr = link.get('title', '')
                rel = link.get('rel', '')
                
                if title_attr == 'pdf':
                    pdf_url = href
                elif rel == 'alternate':
                    url = href
            
            # If no explicit PDF link, construct it
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            if not url and arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
            
            # Extract categories
            categories = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term')
                if term:
                    categories.append(term)
            
            # Primary category
            primary_cat = entry.find('arxiv:primary_category', ns)
            primary_category = primary_cat.get('term') if primary_cat is not None else ""
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'url': url,
                'pdf_url': pdf_url,
                'authors': authors,
                'categories': categories,
                'primary_category': primary_category,
                'published_date': published_date
            }
            
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None
    
    def papers_to_models(
        self,
        paper_dicts: List[Dict[str, Any]],
        limit: int = None
    ) -> List[Paper]:
        """
        Convert paper dictionaries to Paper model objects.
        
        Args:
            paper_dicts: List of paper dictionaries from search
            limit: Maximum number of papers to convert
            
        Returns:
            List of Paper model objects
        """
        limit = limit or config.MAX_PAPERS_TO_ANALYZE
        papers = []
        
        for i, pd in enumerate(paper_dicts[:limit]):
            paper = Paper(
                paper_id=f"paper_{i+1:02d}",
                arxiv_id=pd.get('arxiv_id', ''),
                title=pd.get('title', ''),
                abstract=pd.get('abstract', ''),
                url=pd.get('url', ''),
                pdf_url=pd.get('pdf_url', ''),
                authors=pd.get('authors', []),
                categories=pd.get('categories', []),
                published_date=pd.get('published_date')
            )
            papers.append(paper)
        
        return papers
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """
        Retrieve a specific paper by arXiv ID.
        
        Args:
            arxiv_id: The arXiv ID (e.g., "2401.12345")
            
        Returns:
            Paper object or None if not found
        """
        self._wait_for_rate_limit()
        
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        
        try:
            response = urllib.request.urlopen(url, timeout=30)
            xml_content = response.read().decode('utf-8')
            papers = self._parse_response(xml_content)
            
            if papers:
                return self.papers_to_models(papers, limit=1)[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve paper {arxiv_id}: {e}")
            return None

