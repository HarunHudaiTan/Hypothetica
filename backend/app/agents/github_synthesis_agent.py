"""
GitHubSynthesisAgent: Takes all per-repo judgments and writes the final
GitHub evidence paragraph for the originality report.
This is the output the user actually reads.
"""
import json
import logging
from typing import List, Dict

from app.agents.Agent import Agent
from app.agents import agent_config
from core import config

logger = logging.getLogger(__name__)


class GitHubSynthesisAgent(Agent):

    def __init__(self):
        super().__init__(
            system_prompt="""You are a research originality advisor synthesizing GitHub evidence.

You receive a research idea and analysis of relevant GitHub repositories.
Write the final assessment paragraph that goes into an originality report.

## Your output must be:

1. **Honest about existence** — if 3 high-star repos do 80% of what the idea proposes, say that clearly. Don't soften it.

2. **Specific about the gap** — name the specific capability, use case, or approach that existing repos don't cover. Vague gaps ("there is still room for innovation") are useless.

3. **Actionable** — end with one of three verdicts:
   - "pursue_as_is": The idea has clear novelty not covered by existing implementations
   - "refine_scope": Existing implementations cover parts — focus on the gap
   - "reconsider": This is largely solved as running code

## Critical distinctions to make:
- Repos that implement the SAME idea with the SAME approach = high threat to novelty
- Repos that implement the SAME application with a DIFFERENT method = low threat, different contribution
- Repos that implement a COMPONENT of the idea = partial overlap, novelty lives in the integration

## Tone
Do NOT be encouraging just to be nice. A researcher reading this is making a real decision about whether to spend months on this idea. Be honest.

## Output Format
Return ONLY valid JSON:
{
  "synthesis": "A single paragraph of 4-6 sentences summarizing the GitHub evidence.",
  "verdict": "pursue_as_is"
}

verdict must be exactly one of: "pursue_as_is", "refine_scope", "reconsider"
""",
            temperature=agent_config.GITHUB_SYNTHESIS_TEMPERATURE,
            top_p=agent_config.GITHUB_SYNTHESIS_TOP_P,
            top_k=agent_config.GITHUB_SYNTHESIS_TOP_K,
            response_mime_type="application/json",
            create_chat=False,
        )
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def synthesize(self, user_idea: str, repo_analyses: List[Dict]) -> Dict:
        """
        Produce the final GitHub evidence synthesis.
        repo_analyses: list of dicts with repo info + relevance assessment.
        """
        repo_lines = []
        for ra in repo_analyses:
            repo_lines.append(
                f"- **{ra['repo_full_name']}** ({ra['stars']} stars)\n"
                f"  Verdict: {ra['verdict']} | Overlap: {ra['overlap_score']:.1%}\n"
                f"  Covers: {ra['what_it_covers']}\n"
                f"  Misses: {ra['what_it_misses']}"
            )
        repos_block = "\n".join(repo_lines) if repo_lines else "No relevant repositories found."

        prompt = f"""## RESEARCH IDEA
{user_idea}

## REPOSITORY ANALYSES (only repos with verdict != 'unrelated')
{repos_block}

Write the synthesis paragraph and verdict."""

        try:
            response = self.generate_text_generation_response(prompt)
            um = getattr(response, "usage_metadata", None)
            self.last_input_tokens = getattr(um, "prompt_token_count", 0) or 0
            self.last_output_tokens = getattr(um, "candidates_token_count", 0) or 0
            result = json.loads(response.text)

            if result.get("verdict") not in ("pursue_as_is", "refine_scope", "reconsider"):
                result["verdict"] = "pursue_as_is"

            logger.info(f"GitHubSynthesis verdict: {result['verdict']}")
            return result
        except Exception as e:
            logger.error(f"GitHubSynthesis failed: {e}")
            return {
                "synthesis": "GitHub analysis could not be completed.",
                "verdict": "pursue_as_is",
            }

    def get_cost(self) -> float:
        return (
            (self.last_input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            + (self.last_output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
        )
