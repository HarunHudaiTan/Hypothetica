from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class PaperSource(ABC):
    """
    Abstract base class for paper sources (ArXiv, Semantic Scholar, etc.).
    Ensures modular design where each source implements the same interface.
    """

    @abstractmethod
    def search_papers(self, query: str, max_results: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for papers based on query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional source-specific parameters
            
        Returns:
            List of paper dictionaries in standardized format
        """
        pass

    @abstractmethod
    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Paper details dictionary or None if not found
        """
        pass

    @abstractmethod
    def download_pdf(self, paper_id: str) -> Optional[str]:
        """
        Download PDF for a paper and return local path.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Local file path to downloaded PDF or None if unavailable
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """
        Return the name of this paper source.
        
        Returns:
            Source name string (e.g., "arxiv", "semantic_scholar")
        """
        pass

    def standardize_paper_data(self, raw_paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert source-specific paper data to standardized format.
        This method should be overridden by each source to ensure consistency.
        
        Standard format includes:
        - id: Unique identifier
        - title: Paper title
        - abstract: Paper abstract/summary
        - authors: List of author names
        - url: URL to paper page
        - pdf_url: URL to PDF (if available)
        - published_date: Publication date string
        - categories: List of categories/fields
        - source: Source name
        - source_id: Original source ID
        
        Args:
            raw_paper: Raw paper data from source API
            
        Returns:
            Standardized paper dictionary
        """
        return {
            'id': '',
            'title': '',
            'abstract': '',
            'authors': [],
            'url': '',
            'pdf_url': '',
            'published_date': '',
            'categories': [],
            'source': self.get_source_name(),
            'source_id': '',
            **raw_paper  # Include any additional fields
        }
