"""
GitHub evidence adapter.
Fetches repository READMEs as markdown for direct Layer 1 analysis (no chunking).
"""
import hashlib
import logging
from typing import List, Dict, Any

from core import config
from app.models.paper import Paper
from app.processing.github_search import GitHubSearchClient
from .base_adapter import EvidenceAdapter

logger = logging.getLogger(__name__)


class GitHubAdapter(EvidenceAdapter):
    """
    Adapter for GitHub repositories.
    Unlike paper adapters, GitHub repos are analyzed as whole markdown documents
    without chunking - the README content goes directly to Layer 1.
    """
    
    def __init__(self):
        self._client = GitHubSearchClient()
    
    @property
    def name(self) -> str:
        return "github"
    
    @property
    def description(self) -> str:
        return "GitHub repositories and open source code"

    @property
    def display_name(self) -> str:
        return "GitHub"

    @property
    def evidence_noun_plural(self) -> str:
        return "repositories"

    @property
    def evidence_noun_singular(self) -> str:
        return "repository"
    
    @property
    def is_available(self) -> bool:
        # GitHub API works without token, but has lower rate limits
        return True
    
    def search(self, query: str, max_results: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search GitHub repositories.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (uses config defaults)
            **kwargs: Additional parameters
            
        Returns:
            List of repository dictionaries with README content
        """
        logger.info(f"[GitHubAdapter] Searching: {query} (max {max_results} results)")
        
        # For GitHub, we use the query agent to generate multiple queries
        # But for now, just search with the single query
        queries = [query]
        
        # Use the existing GitHub search client
        repos = self._client.search_and_filter(queries)
        
        # Fetch full README for each repo (not just preview)
        for repo in repos:
            owner = repo["owner"]["login"]
            name = repo["name"]
            full_readme = self._client.get_readme(owner, name)
            repo["_full_readme"] = full_readme or ""
        
        logger.info(f"[GitHubAdapter] Returned {len(repos)} repositories")
        return repos
    
    def convert_to_papers(
        self,
        results: List[Dict[str, Any]],
        limit: int = None
    ) -> List[Paper]:
        """
        Convert GitHub repositories to Paper objects.
        The README markdown becomes the 'abstract' and full content.
        """
        limit = limit or config.GITHUB_MAX_REPOS_TO_ANALYZE
        papers = []
        
        logger.info(f"[GitHubAdapter] Converting {len(results)} repos to Paper models")
        
        for i, repo in enumerate(results[:limit]):
            try:
                full_name = repo.get("full_name", "")
                readme = repo.get("_full_readme", "")
                stable_id = hashlib.sha1(full_name.encode("utf-8")).hexdigest()[:12] if full_name else f"repo_{i+1:02d}"
                
                # Use README as both abstract and markdown_content
                # This allows Layer 1 to analyze the full README without chunking
                paper = Paper(
                    paper_id=f"github_{stable_id}",
                    source="github",
                    source_id=full_name,
                    title=repo.get("name", ""),
                    abstract=readme[:1000] if readme else repo.get("description", ""),  # First 1000 chars as abstract
                    url=repo.get("html_url", ""),
                    authors=[repo.get("owner", {}).get("login", "")],
                    published_date=repo.get("created_at"),
                    metadata={
                        "full_name": full_name,
                        "stars": repo.get("stargazers_count", 0),
                        "language": repo.get("language"),
                        "topics": repo.get("topics", []),
                        "last_pushed": repo.get("pushed_at"),
                        "description": repo.get("description", ""),
                    },
                    # Store full README as markdown_content for Layer 1 analysis
                    markdown_content=readme,
                    is_processed=True  # Mark as processed since we have the markdown
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"[GitHubAdapter] Error converting repo: {e}")
                continue
        
        logger.info(f"[GitHubAdapter] Converted {len(papers)} repos to Paper models")
        return papers
