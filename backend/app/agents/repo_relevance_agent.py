"""
RepoRelevanceAgent: For each repo returned by GitHub search, decides if it
genuinely overlaps with the research idea or is a false positive.
Outputs a structured judgment with overlap_score, what_it_covers,
what_it_misses, and verdict.
"""
import json
import logging
from typing import Dict

from app.agents.Agent import Agent
from app.agents import agent_config
from core import config

logger = logging.getLogger(__name__)


class RepoRelevanceAgent(Agent):

    def __init__(self):
        super().__init__(
            system_prompt="""You are a GitHub Repository Relevance Assessor for research originality analysis.

Your job is NOT to find similarity. Your job is to find the GAP.
Where does this repository stop, and where does the research idea begin?

For a given research idea and a GitHub repository (name, description, topics, README excerpt),
you assess whether this repo genuinely overlaps with the idea or is a false positive.

## Rules
- IGNORE star count as a quality signal
- IGNORE recency as a relevance signal
- IGNORE language/framework choice as overlap evidence
  (a PyTorch repo and a JAX repo solving the same problem are equally overlapping)
- Focus on WHAT the repo does, not HOW it's implemented
- Look for what the repo does NOT do that the idea proposes

## Output Format
Return ONLY valid JSON:
{
  "overlap_score": 0.7,
  "what_it_covers": "One sentence on what aspect of the idea this repo addresses.",
  "what_it_misses": "One sentence on what the idea proposes that this repo does NOT do.",
  "verdict": "partial_overlap"
}

## Verdict values (pick exactly one):
- "strong_overlap": The repo implements 70%+ of what the idea proposes, same approach
- "partial_overlap": The repo covers a significant component but misses key parts
- "tangential": Related area but fundamentally different goal or approach
- "unrelated": No meaningful overlap despite keyword similarity

## Scoring guidance
- overlap_score is 0.0 to 1.0
- Be precise: 0.8+ means the idea is largely implemented already
- 0.4-0.7 means meaningful partial overlap
- Below 0.3 means tangential at best
""",
            temperature=agent_config.REPO_RELEVANCE_TEMPERATURE,
            top_p=agent_config.REPO_RELEVANCE_TOP_P,
            top_k=agent_config.REPO_RELEVANCE_TOP_K,
            response_mime_type="application/json",
            create_chat=False,
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def assess_repo(self, user_idea: str, repo: Dict) -> Dict:
        """
        Assess a single repo's relevance to the research idea.
        Returns parsed JSON with overlap_score, what_it_covers, what_it_misses, verdict.
        """
        topics = ", ".join(repo.get("topics", []) or [])
        readme = repo.get("_readme_preview", "")

        prompt = f"""## RESEARCH IDEA
{user_idea}

## GITHUB REPOSITORY
Name: {repo.get('full_name', '')}
Description: {repo.get('description', 'No description')}
Topics: {topics or 'None listed'}

README (first ~2000 chars):
{readme if readme else 'No README available'}

Assess this repository's overlap with the research idea."""

        try:
            response = self.generate_text_generation_response(prompt)
            um = getattr(response, "usage_metadata", None)
            self.total_input_tokens += getattr(um, "prompt_token_count", 0) or 0
            self.total_output_tokens += getattr(um, "candidates_token_count", 0) or 0
            result = json.loads(response.text)

            result["overlap_score"] = max(0.0, min(1.0, float(result.get("overlap_score", 0))))
            if result.get("verdict") not in (
                "strong_overlap", "partial_overlap", "tangential", "unrelated"
            ):
                result["verdict"] = "unrelated"

            logger.info(
                f"RepoRelevance: {repo.get('full_name')} -> "
                f"verdict={result['verdict']}, score={result['overlap_score']:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"RepoRelevance failed for {repo.get('full_name')}: {e}")
            return {
                "overlap_score": 0.0,
                "what_it_covers": "",
                "what_it_misses": "Assessment failed",
                "verdict": "unrelated",
            }

    def get_cost(self) -> float:
        return (
            (self.total_input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            + (self.total_output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
        )
