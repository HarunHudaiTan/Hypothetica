"""
Abstract base class for paper retrieval sources.
Provides a common interface for arXiv, Google Patents, and future sources.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.models.paper import Paper


class BasePaperSource(ABC):
    """Abstract base class for paper retrieval sources."""
    
    @abstractmethod
    def search_papers(self, query: str, max_results: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for papers matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional source-specific parameters
            
        Returns:
            List of paper dictionaries with source-specific format
        """
        pass
    
    @abstractmethod
    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id: Source-specific paper identifier
            
        Returns:
            Detailed paper information or None if not found
        """
        pass
    
    @abstractmethod
    def convert_to_paper_models(self, paper_dicts: List[Dict[str, Any]], limit: int = None) -> List[Paper]:
        """
        Convert source-specific paper data to standardized Paper model objects.
        
        Args:
            paper_dicts: List of paper dictionaries from search
            limit: Maximum number of papers to convert
            
        Returns:
            List of Paper model objects
        """
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the paper source."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the source is properly configured and available."""
        pass
