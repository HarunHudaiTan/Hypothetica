"""
GitHub API client for searching repositories and fetching README content.
"""
import base64
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Dict, Optional

import requests

from core import config

logger = logging.getLogger(__name__)

TOPIC_HINTS = {
    # Framework names only — these are commonly self-tagged on GitHub
    "langchain": "langchain",
    "langgraph": "langchain",
    "llama_index": "llama-index",
    "llama-index": "llama-index",
    "llamaindex": "llama-index",
    "transformers": "transformers",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "haystack": "haystack",
    "autogen": "autogen",
    "crewai": "crewai",
    # Intentionally excluded: "agent", "agents", "llm", "openai", "rag", "multimodal"
    # — these GitHub topics are too broad and match unrelated popular repos
}


class GitHubSearchClient:
    """Handles GitHub API interactions: repo search and README fetching."""

    def __init__(self):
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",  # GitHub API version
        }
        if config.GITHUB_TOKEN:
            self._headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"

    def search_repos(self, query: str, per_page: int = None) -> List[Dict]:
        """
        Search GitHub repositories with enhanced metadata extraction.
        Returns repos sorted by stars with topics, visibility, and license info.
        """
        per_page = per_page or config.GITHUB_RESULTS_PER_QUERY
        url = "https://api.github.com/search/repositories"
        qualified_query = self._build_qualified_query(query)
        params = {
            "q": qualified_query,
            # No explicit sort — GitHub uses best-match relevance ranking by default
            "per_page": min(per_page, 100),  # API max is 100
        }
        try:
            r = requests.get(url, headers=self._headers, params=params, timeout=15)
            r.raise_for_status()
            
            items = r.json().get("items", [])
            
            # Enhance each repo with metadata for RAG
            for repo in items:
                # Extract topics (high-level categories)
                repo['_topics'] = repo.get('topics', [])
                
                # Extract visibility (public/private)
                repo['_visibility'] = 'private' if repo.get('private', False) else 'public'
                
                # Extract license for compliance
                license_info = repo.get('license')
                repo['_license'] = license_info.get('spdx_id') if license_info else None
                
                # Extract language
                repo['_language'] = repo.get('language', 'Unknown')
                
            return items
        except Exception as e:
            logger.error(f"GitHub search failed for query '{query}': {e}")
            return []

    def _build_qualified_query(self, query: str) -> str:
        tokens = [token.strip().lower() for token in query.split() if token.strip()]
        core_terms: List[str] = []
        topic_terms: List[str] = []

        for token in tokens:
            normalized = token.replace(":", "").replace(",", "")
            if normalized in TOPIC_HINTS:
                topic_terms.append(f"topic:{TOPIC_HINTS[normalized]}")
            else:
                core_terms.append(normalized)

        core_terms = core_terms[:5]
        qualifiers = [
            "in:name,description,readme",
            "archived:false",
            "fork:false",
            f"stars:>{config.GITHUB_QUERY_MIN_STARS}",
        ]
        qualified_parts = core_terms + topic_terms[:2] + qualifiers
        qualified_query = " ".join(qualified_parts).strip()
        logger.info(f"[GitHubSearchClient] Qualified query: {qualified_query}")
        return qualified_query

    @staticmethod
    def _parse_timestamp(value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        return [term.lower() for term in query.split() if term.strip()]

    def _repo_text(self, repo: Dict) -> str:
        parts = [
            repo.get("name", ""),
            repo.get("full_name", ""),
            repo.get("description", ""),
            " ".join(repo.get("topics", []) or []),
            repo.get("_readme_preview", ""),
        ]
        return " ".join(part.lower() for part in parts if part)

    def _repo_quality_score(self, repo: Dict, query_terms: List[str]) -> float:
        text = self._repo_text(repo)
        matched_terms = sum(1 for term in query_terms if term in text)
        match_ratio = matched_terms / max(len(query_terms), 1)

        stars = max(repo.get("stargazers_count", 0), 0)
        star_score = min(math.log10(stars + 1) / 4.0, 1.0)

        pushed_at = self._parse_timestamp(repo.get("pushed_at", ""))
        if pushed_at:
            days_since_push = max((datetime.now(timezone.utc) - pushed_at).days, 0)
            recency_score = max(0.0, 1.0 - min(days_since_push, 365 * 3) / (365 * 3))
        else:
            recency_score = 0.0

        readme_length = len(repo.get("_readme_preview", ""))
        readme_score = min(readme_length / max(config.GITHUB_README_PREVIEW_CHARS, 1), 1.0)
        topic_score = min(len(repo.get("topics", []) or []) / 5.0, 1.0)
        license_score = 1.0 if repo.get("_license") and repo.get("_license") != "NOASSERTION" else 0.0

        return (
            match_ratio * 0.45
            + star_score * 0.20
            + recency_score * 0.15
            + readme_score * 0.10
            + topic_score * 0.05
            + license_score * 0.05
        )

    def _repo_passes_quality_filters(self, repo: Dict) -> bool:
        if repo.get("archived") or repo.get("fork"):
            return False
        if repo.get("stargazers_count", 0) < config.GITHUB_MIN_STARS:
            return False

        pushed_at = self._parse_timestamp(repo.get("pushed_at", ""))
        if not pushed_at or pushed_at.year < config.GITHUB_MIN_PUSH_YEAR:
            return False

        readme_preview = repo.get("_readme_preview", "")
        if len(readme_preview) < config.GITHUB_MIN_README_CHARS:
            return False

        description = (repo.get("description") or "").strip()
        if not description and not repo.get("topics"):
            return False

        return True

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
        # Step 1: Run all queries in parallel
        query_to_repos: Dict[str, List[Dict]] = {}
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            futures = {executor.submit(self.search_repos, q): q for q in queries}
            for future in as_completed(futures):
                q = futures[future]
                try:
                    query_to_repos[q] = future.result()
                except Exception as e:
                    logger.error(f"GitHub query failed '{q}': {e}")
                    query_to_repos[q] = []

        # Step 2: Collect unique repos (dedup by full_name, preserve first-seen order)
        all_repos_by_name: Dict[str, Dict] = {}
        for q in queries:
            for repo in query_to_repos.get(q, []):
                name = repo["full_name"]
                if name not in all_repos_by_name:
                    all_repos_by_name[name] = repo

        # Step 3: Fetch all READMEs in parallel (single pass, no duplicate fetches)
        all_repos = list(all_repos_by_name.values())

        def _fetch_readme(repo):
            owner = repo["owner"]["login"]
            name = repo["name"]
            readme = self.get_readme(owner, name)
            repo["_readme_preview"] = readme[:config.GITHUB_README_PREVIEW_CHARS] if readme else ""

        with ThreadPoolExecutor(max_workers=min(len(all_repos), 10)) as executor:
            list(executor.map(_fetch_readme, all_repos))

        # Step 4: Filter and score per query (README data now available)
        per_query_results: Dict[str, List[Dict]] = {}
        for query in queries:
            query_terms = self._query_terms(query)
            filtered = []
            filtered_by_stars = 0
            filtered_by_date = 0

            for repo in query_to_repos.get(query, []):
                if repo.get("stargazers_count", 0) < config.GITHUB_MIN_STARS:
                    filtered_by_stars += 1
                    continue

                pushed_year = repo.get("pushed_at", "")[:4]
                if pushed_year and int(pushed_year) < config.GITHUB_MIN_PUSH_YEAR:
                    filtered_by_date += 1
                    logger.debug(
                        f"Filtered {repo['full_name']}: pushed in {pushed_year} (< {config.GITHUB_MIN_PUSH_YEAR})"
                    )
                    continue

                if not self._repo_passes_quality_filters(repo):
                    continue

                repo["_query_match_score"] = self._repo_quality_score(repo, query_terms)
                filtered.append(repo)

            filtered.sort(
                key=lambda repo: (
                    repo.get("_query_match_score", 0.0),
                    repo.get("stargazers_count", 0),
                    repo.get("pushed_at", ""),
                ),
                reverse=True,
            )
            per_query_results[query] = filtered
            logger.info(
                f"GitHub query '{query[:50]}': {len(query_to_repos.get(query, []))} raw -> {len(filtered)} after filter "
                f"(stars: -{filtered_by_stars}, date: -{filtered_by_date})"
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
            all_remaining.sort(
                key=lambda r: (
                    r.get("_query_match_score", 0.0),
                    r.get("stargazers_count", 0),
                    r.get("pushed_at", ""),
                ),
                reverse=True,
            )
            for repo in all_remaining:
                if len(diverse_repos) >= config.GITHUB_MAX_REPOS_TO_ANALYZE:
                    break
                if repo["full_name"] not in seen_names:
                    seen_names.add(repo["full_name"])
                    diverse_repos.append(repo)

        final = sorted(
            diverse_repos,
            key=lambda r: (
                r.get("_query_match_score", 0.0),
                r.get("stargazers_count", 0),
                r.get("pushed_at", ""),
            ),
            reverse=True,
        )[: config.GITHUB_MAX_REPOS_TO_ANALYZE]
        logger.info(
            f"GitHub search: {sum(len(v) for v in per_query_results.values())} filtered total, "
            f"{len(final)} diverse repos selected"
        )

        return final
