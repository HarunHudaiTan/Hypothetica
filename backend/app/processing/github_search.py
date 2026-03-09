"""
GitHub API client for searching repositories and fetching README content.
"""
import base64
import logging
from typing import List, Dict, Optional

import requests

from core import config

logger = logging.getLogger(__name__)


class GitHubSearchClient:
    """Handles GitHub API interactions: repo search and README fetching."""

    def __init__(self):
        self._headers = {
            "Accept": "application/vnd.github+json",
        }
        if config.GITHUB_TOKEN:
            self._headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"

    def search_repos(self, query: str, per_page: int = None) -> List[Dict]:
        per_page = per_page or config.GITHUB_RESULTS_PER_QUERY
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
        }
        try:
            r = requests.get(url, headers=self._headers, params=params, timeout=15)
            r.raise_for_status()
            return r.json().get("items", [])
        except Exception as e:
            logger.error(f"GitHub search failed for query '{query}': {e}")
            return []

    def get_readme(self, owner: str, repo: str) -> Optional[str]:
        url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        try:
            r = requests.get(url, headers=self._headers, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            content = data.get("content")
            if not content:
                return None
            if data.get("encoding") == "base64":
                return base64.b64decode(content).decode("utf-8", errors="ignore")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch README for {owner}/{repo}: {e}")
            return None

    def search_and_filter(
        self, queries: List[str]
    ) -> List[Dict]:
        """
        Search GitHub with multiple queries, dedupe, filter by stars/recency,
        and return a diverse top set.

        Strategy per the spec:
        - 10-15 results per query, ~40-60 total
        - Dedupe by full_name
        - Filter: stars >= GITHUB_MIN_STARS, last pushed >= GITHUB_MIN_PUSH_YEAR
        - Take top GITHUB_TOP_PER_QUERY per query for diversity, then merge
        - Final cap at GITHUB_MAX_REPOS_TO_ANALYZE
        """
        per_query_results: Dict[str, List[Dict]] = {}

        for query in queries:
            repos = self.search_repos(query)
            filtered = []
            for repo in repos:
                if repo["stargazers_count"] < config.GITHUB_MIN_STARS:
                    continue
                pushed_year = repo.get("pushed_at", "")[:4]
                if pushed_year and int(pushed_year) < config.GITHUB_MIN_PUSH_YEAR:
                    continue
                filtered.append(repo)
            per_query_results[query] = filtered
            logger.info(
                f"GitHub query '{query[:50]}': {len(repos)} raw -> {len(filtered)} after filter"
            )

        # Take top N per query for diversity
        seen_names = set()
        diverse_repos: List[Dict] = []
        for query, repos in per_query_results.items():
            added = 0
            for repo in repos:
                name = repo["full_name"]
                if name not in seen_names:
                    seen_names.add(name)
                    diverse_repos.append(repo)
                    added += 1
                    if added >= config.GITHUB_TOP_PER_QUERY:
                        break

        # If we still have room, fill from remaining repos across all queries
        if len(diverse_repos) < config.GITHUB_MAX_REPOS_TO_ANALYZE:
            all_remaining = []
            for repos in per_query_results.values():
                for repo in repos:
                    if repo["full_name"] not in seen_names:
                        all_remaining.append(repo)
            all_remaining.sort(key=lambda r: r["stargazers_count"], reverse=True)
            for repo in all_remaining:
                if len(diverse_repos) >= config.GITHUB_MAX_REPOS_TO_ANALYZE:
                    break
                if repo["full_name"] not in seen_names:
                    seen_names.add(repo["full_name"])
                    diverse_repos.append(repo)

        final = diverse_repos[: config.GITHUB_MAX_REPOS_TO_ANALYZE]
        logger.info(
            f"GitHub search: {sum(len(v) for v in per_query_results.values())} filtered total, "
            f"{len(final)} diverse repos selected"
        )

        # Enrich with README previews
        for repo in final:
            owner = repo["owner"]["login"]
            name = repo["name"]
            readme = self.get_readme(owner, name)
            repo["_readme_preview"] = (
                readme[: config.GITHUB_README_PREVIEW_CHARS] if readme else ""
            )

        return final
