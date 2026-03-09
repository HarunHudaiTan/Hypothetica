"""
GitHubService: Orchestrates the full GitHub evidence pipeline.
Runs independently from and in parallel with the arXiv pipeline.

Pipeline: GitHubQueryAgent → GitHub API search → RepoRelevanceAgent (per repo) → GitHubSynthesisAgent
"""
import logging

from app.api.managers.job_manager import job_manager
from app.processing.github_search import GitHubSearchClient
from app.agents.github_query_agent import GitHubQueryAgent
from app.agents.repo_relevance_agent import RepoRelevanceAgent
from app.agents.github_synthesis_agent import GitHubSynthesisAgent
from app.models.analysis import GitHubAnalysisResult, RepoRelevanceResult

logger = logging.getLogger(__name__)


class GitHubService:
    _search_client = GitHubSearchClient()
    _query_agent = GitHubQueryAgent()
    _relevance_agent = RepoRelevanceAgent()
    _synthesis_agent = GitHubSynthesisAgent()

    @staticmethod
    def _push_github_progress(job_id: str, message: str):
        """Push a progress event for the GitHub pipeline without overwriting arXiv progress."""
        job = job_manager.get_job(job_id)
        if job:
            job.push_event({
                "type": "progress",
                "message": message,
                "progress": -1,
            })
            logger.info(f"Job {job_id} [GitHub] {message}")

    @classmethod
    def run_github_analysis(cls, job_id: str, _update_progress=None):
        """
        Full GitHub evidence pipeline. Designed to run in a background thread
        in parallel with the arXiv pipeline.
        """
        job = job_manager.get_job(job_id)
        if not job:
            return

        idea = job.state.enriched_idea or job.user_idea

        try:
            # Phase 1: Generate GitHub-optimized queries
            cls._push_github_progress(job_id, "Searching GitHub...")
            queries = cls._query_agent.generate_queries(idea)
            if not queries:
                logger.warning(f"GitHubQueryAgent returned no queries for job {job_id}")
                job.state.github_result = GitHubAnalysisResult(
                    synthesis="No GitHub queries could be generated for this idea.",
                    verdict="pursue_as_is",
                )
                return

            # Phase 2: Search GitHub API and filter/dedupe
            repos = cls._search_client.search_and_filter(queries)
            if not repos:
                logger.info(f"No GitHub repos found for job {job_id}")
                job.state.github_result = GitHubAnalysisResult(
                    synthesis="No relevant GitHub repositories were found for this research idea.",
                    verdict="pursue_as_is",
                )
                return

            # Phase 3: Assess each repo with RepoRelevanceAgent
            cls._push_github_progress(job_id, "Analyzing repositories...")
            cls._relevance_agent.total_input_tokens = 0
            cls._relevance_agent.total_output_tokens = 0

            repo_analyses = []
            for repo in repos:
                assessment = cls._relevance_agent.assess_repo(idea, repo)
                repo_analyses.append({
                    "repo_full_name": repo["full_name"],
                    "repo_url": repo["html_url"],
                    "stars": repo["stargazers_count"],
                    "description": repo.get("description") or "",
                    "last_pushed": repo.get("pushed_at", "")[:10],
                    "topics": repo.get("topics", []) or [],
                    **assessment,
                })

            # Filter out unrelated repos for synthesis
            relevant = [
                ra for ra in repo_analyses if ra["verdict"] != "unrelated"
            ]

            # Phase 4: Synthesis
            cls._push_github_progress(job_id, "Summarizing GitHub findings...")
            synthesis_result = cls._synthesis_agent.synthesize(idea, relevant)

            # Build result models
            repo_result_models = []
            for ra in repo_analyses:
                repo_result_models.append(RepoRelevanceResult(
                    repo_full_name=ra["repo_full_name"],
                    repo_url=ra["repo_url"],
                    stars=ra["stars"],
                    description=ra["description"],
                    last_pushed=ra["last_pushed"],
                    topics=ra["topics"],
                    overlap_score=ra["overlap_score"],
                    what_it_covers=ra["what_it_covers"],
                    what_it_misses=ra["what_it_misses"],
                    verdict=ra["verdict"],
                ))

            github_result = GitHubAnalysisResult(
                synthesis=synthesis_result.get("synthesis", ""),
                verdict=synthesis_result.get("verdict", "pursue_as_is"),
                repos_analyzed=len(repo_analyses),
                repos_relevant=len(relevant),
                repo_results=repo_result_models,
            )
            job.state.github_result = github_result

            # Accumulate costs
            total_github_cost = (
                cls._query_agent.get_cost()
                + cls._relevance_agent.get_cost()
                + cls._synthesis_agent.get_cost()
            )
            job.state.cost.github = total_github_cost

            logger.info(
                f"GitHub analysis complete for job {job_id}: "
                f"{len(repo_analyses)} repos analyzed, {len(relevant)} relevant, "
                f"verdict={github_result.verdict}, cost=${total_github_cost:.4f}"
            )

        except Exception as e:
            logger.exception(f"GitHub analysis failed for job {job_id}: {e}")
            job.state.github_result = GitHubAnalysisResult(
                synthesis=f"GitHub analysis encountered an error: {str(e)}",
                verdict="pursue_as_is",
            )
