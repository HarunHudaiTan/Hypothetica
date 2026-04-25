"""
Abstract base class for evidence adapters.
All evidence sources (arXiv, Patents, Semantic Scholar, etc.) implement this interface.
"""
from abc import ABC, abstractmethod
from typing import List
from app.models.paper import Paper


class EvidenceAdapter(ABC):
    """
    Abstract base class for all evidence sources.
    
    Each adapter encapsulates:
    - Search logic (how to query the source)
    - Content fetching (how to retrieve full content)
    - Paper model conversion (how to map source data to Paper objects)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this adapter (e.g., 'arxiv', 'google_patents')."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this evidence source."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Short UI label (e.g. 'arXiv', 'Google Patents')."""
        pass

    @property
    def evidence_noun_plural(self) -> str:
        """Plural noun for progress messages (e.g. 'papers', 'patents')."""
        return "papers"

    @property
    def evidence_noun_singular(self) -> str:
        """Singular noun for progress messages (e.g. 'paper', 'patent')."""
        return "paper"
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the adapter is properly configured and available for use."""
        pass
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10, **kwargs) -> List[dict]:
        """
        Search the evidence source for items matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional adapter-specific parameters
            
        Returns:
            List of raw result dictionaries (adapter-specific format)
        """
        pass
    
    @abstractmethod
    def convert_to_papers(self, results: List[dict], limit: int = None) -> List[Paper]:
        """
        Convert raw search results to standardized Paper objects.
        
        Args:
            results: Raw search results from search()
            limit: Maximum number of papers to convert
            
        Returns:
            List of Paper model objects
        """
        pass
    
    def search_and_convert(self, query: str, max_results: int = 10, limit: int = None) -> List[Paper]:
        """
        Convenience method: search and convert to Paper objects in one call.
        
        Args:
            query: Search query string
            max_results: Maximum results to fetch from source
            limit: Maximum papers to convert
            
        Returns:
            List of Paper objects
        """
        results = self.search(query, max_results)
        return self.convert_to_papers(results, limit)
