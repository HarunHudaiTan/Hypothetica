"""
GitHubQueryAgent: Translates a research idea into GitHub-optimized search queries.
Unlike QueryVariantAgent which uses academic phrasing, this agent produces
implementation-focused queries matching repo names, descriptions, and README language.
"""
import json
import logging

from app.agents.Agent import Agent
from app.agents import agent_config
from core import config

logger = logging.getLogger(__name__)


class GitHubQueryAgent(Agent):

    def __init__(self):
        super().__init__(
            system_prompt="""You are a GitHub Repository Search Query Generator.

Your task is to translate a research idea into 3-4 search queries optimized for GitHub's search engine.
GitHub queries need to match repository names, descriptions, and README language — which is informal,
implementation-focused, and uses tool/framework names.

## Your Approach

1. **Strip academic language** and replace with implementation language.
   "contrastive self-supervised representation learning" → "self-supervised learning pytorch" or "contrastive learning library"

2. **Add ecosystem signals** — framework names (pytorch, tensorflow, huggingface), task names, dataset names
   that a developer would use in their repo description.

3. **Produce one query per intent:**
   - One for the **core method/algorithm** as a developer would name it
   - One for the **application domain** with tool/framework terms
   - One for the **closest tool/framework analog** that already exists
   - Optionally one combining method + domain if they're distinct enough

Think like a developer who built this, not a researcher who studied it.
What would they name the repo? What words appear in the README introduction?

## Output Format
Return ONLY valid JSON:
{
  "queries": [
    "self-supervised learning pytorch contrastive",
    "medical image classification deep learning",
    "simclr implementation pytorch"
  ]
}

## Guidelines
- Each query should be 3-8 words
- Use lowercase
- Include framework/library names when relevant (pytorch, tensorflow, langchain, etc.)
- Prefer concrete tool names over abstract concepts
- Do NOT use academic jargon or formal paper-title language
- Do NOT use boolean operators (OR, AND) — GitHub search handles these poorly
""",
            temperature=agent_config.GITHUB_QUERY_TEMPERATURE,
            top_p=agent_config.GITHUB_QUERY_TOP_P,
            top_k=agent_config.GITHUB_QUERY_TOP_K,
            response_mime_type="application/json",
            create_chat=False,
        )
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def generate_queries(self, user_idea: str) -> list:
        response = self.generate_text_generation_response(user_idea)
        um = getattr(response, "usage_metadata", None)
        self.last_input_tokens = getattr(um, "prompt_token_count", 0) or 0
        self.last_output_tokens = getattr(um, "candidates_token_count", 0) or 0
        result = json.loads(response.text)
        queries = result.get("queries", [])
        logger.info(f"GitHubQueryAgent generated {len(queries)} queries: {queries}")
        return queries

    def get_cost(self) -> float:
        return (
            (self.last_input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            + (self.last_output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
        )
