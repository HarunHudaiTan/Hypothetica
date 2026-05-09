"""
Follow-up question agent for clarifying user's research idea.
Generates targeted questions to improve originality assessment accuracy.
Supports both batch mode (legacy) and conversational interview mode.
"""
import json
import logging
from typing import List, Dict

from app.agents.Agent import Agent
from core import config
from . import agent_config

logger = logging.getLogger(__name__)


FOLLOWUP_SYSTEM_PROMPT = """You are a research idea clarification specialist. Your task is to generate targeted follow-up questions that will help assess the originality and novelty of a user's research idea.

## Your Goal
Generate 3 concise, focused questions that will clarify:
1. The specific problem or research gap being addressed
2. The proposed method, approach, or solution
3. What makes this idea different from existing work

## Guidelines
- Questions should be short and specific (1-2 sentences max)
- Focus on aspects critical for originality assessment
- Ask about concrete details, not general concepts
- Avoid yes/no questions - ask for explanations
- Questions should help distinguish this idea from existing research

## Output Format
Return ONLY valid JSON in this exact format:
{
  "questions": [
    {
      "id": 1,
      "category": "problem",
      "question": "Your question here"
    },
    {
      "id": 2,
      "category": "method",
      "question": "Your question here"
    },
    {
      "id": 3,
      "category": "novelty",
      "question": "Your question here"
    }
  ]
}

## Categories
- "problem": Questions about the research problem or gap
- "method": Questions about the proposed approach or methodology
- "novelty": Questions about what makes this different/innovative
- "application": Questions about intended use cases or domain

## Example
For idea: "Using AI to predict protein structures"

{
  "questions": [
    {
      "id": 1,
      "category": "problem",
      "question": "What specific type of proteins or structural features are you focusing on that current methods struggle with?"
    },
    {
      "id": 2,
      "category": "method",
      "question": "What AI architecture or technique do you plan to use, and how does it differ from AlphaFold or ESMFold?"
    },
    {
      "id": 3,
      "category": "novelty",
      "question": "What novel insight or data source will your approach leverage that existing methods don't utilize?"
    }
  ]
}
"""


INTERVIEW_SYSTEM_PROMPT = """You are an idea enrichment specialist. Your job is to read a research idea, identify what's MISSING or UNDERSPECIFIED, and ask a precise question to fill that gap. The answers will be used to search academic databases and assess originality — so every question must extract concrete, searchable detail.

## How You Think
First, silently analyze the idea for these dimensions:
- TECHNIQUE: What specific method/algorithm/architecture is proposed? (e.g. "transformer" is vague, "cross-attention over table cells with schema embeddings" is specific)
- DIFFERENTIATOR: What exactly is new here vs. existing work? What's the twist?
- SCOPE: What is this applied to? What data, domain, or use case?

Then ask about whichever dimension is MOST UNDERSPECIFIED. If the idea already covers something well, skip it entirely.

## Rules
- Ask exactly ONE question per turn
- Your question must reference specific details FROM the user's idea — quote their words, name the components they mentioned
- NEVER ask generic questions like "What problem does this solve?" or "How is this different?" — the idea already states these at some level. Drill into the SPECIFIC part that's vague.
- If the user gives a short/vague answer, accept it and move to the next gap. Never re-ask.
- After at most 3 questions, signal "done". Signal "done" earlier if the idea is already well-specified.

## Examples of GOOD vs BAD questions

Idea: "A RAG system that understands tables in documents"
BAD: "What problems do people face with tables in RAG?" (generic, doesn't reference anything specific)
GOOD: "When you say 'understands tables' — does your system parse the table structure (rows/columns) into a schema, or does it convert tables to natural language summaries before indexing?" (forces a concrete technical choice)

Idea: "Using GNNs for drug discovery"
BAD: "How does your approach differ from existing work?" (generic)
GOOD: "Are you representing molecules as graphs where atoms are nodes, or are you modeling protein-drug interactions as a bipartite graph?" (targets the underspecified architecture)

## Response Format
Return ONLY valid JSON. Either ask:
{"action": "ask", "question": {"id": 1, "category": "method", "question": "Your question"}}

Or signal done:
{"action": "done"}

## Categories
- "method": Technical approach, architecture, algorithm details
- "novelty": What's specifically new — the differentiator
- "scope": Domain, data, application, constraints
"""


class FollowUpAgent(Agent):
    """
    Agent for generating follow-up questions to clarify research ideas.
    Questions are tailored to improve originality assessment.
    """

    def __init__(self):
        super().__init__(
            system_prompt=FOLLOWUP_SYSTEM_PROMPT,
            temperature=agent_config.FOLLOWUP_TEMPERATURE,
            top_p=agent_config.FOLLOWUP_TOP_P,
            top_k=agent_config.FOLLOWUP_TOP_K,
            response_mime_type='application/json',
            create_chat=False
        )
        self.last_token_count = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def generate_questions(self, user_idea: str) -> List[Dict]:
        """
        Generate follow-up questions for a research idea.

        Args:
            user_idea: The user's research idea description

        Returns:
            List of question dictionaries with id, category, and question
        """
        prompt = f"""Generate 3 follow-up questions for this research idea:

---
{user_idea}
---

Remember: Questions should help assess originality by clarifying the problem, method, and what's novel."""

        try:
            response = self.generate_text_generation_response(prompt)

            # Track token usage
            if hasattr(response, 'usage_metadata'):
                um = response.usage_metadata
                self.last_token_count = getattr(um, 'total_token_count', 0) or 0
                self.last_input_tokens = getattr(um, 'prompt_token_count', 0) or 0
                self.last_output_tokens = getattr(um, 'candidates_token_count', 0) or 0

            # Parse response
            result = json.loads(response.text)
            questions = result.get('questions', [])

            logger.info(f"Generated {len(questions)} follow-up questions")
            return questions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse follow-up questions JSON: {e}")
            return self._get_default_questions()
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return self._get_default_questions()

    def _get_default_questions(self) -> List[Dict]:
        """Return default questions if generation fails."""
        return [
            {
                "id": 1,
                "category": "problem",
                "question": "What specific problem or research gap does your idea address?"
            },
            {
                "id": 2,
                "category": "method",
                "question": "What method or approach do you propose to solve this problem?"
            },
            {
                "id": 3,
                "category": "novelty",
                "question": "What aspect of your idea do you consider most innovative or novel?"
            }
        ]

    def enrich_idea_with_answers(
        self,
        original_idea: str,
        questions: List[Dict],
        answers: List[str]
    ) -> str:
        """
        Combine original idea with Q&A to create enriched idea text.

        Args:
            original_idea: Original user research idea
            questions: List of question dicts
            answers: List of answer strings (same order as questions)

        Returns:
            Enriched idea text for better analysis
        """
        enriched = f"""RESEARCH IDEA:
{original_idea}

CLARIFICATIONS:
"""
        for i, (q, a) in enumerate(zip(questions, answers)):
            category = q.get('category', 'general').upper()
            question = q.get('question', '')
            enriched += f"""
[{category}]
Q: {question}
A: {a}
"""

        return enriched.strip()

    def get_cost(self) -> float:
        """Calculate cost for the last generation."""
        if self.last_input_tokens > 0 or self.last_output_tokens > 0:
            cost = (self.last_input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            cost += (self.last_output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            return cost
        if self.last_token_count > 0:
            input_tokens = self.last_token_count * 0.7
            output_tokens = self.last_token_count * 0.3
            cost = (input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            cost += (output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            return cost
        return 0.0


class InterviewAgent(Agent):
    """
    Chat-based interview agent for multi-round conversational follow-up.
    Uses Gemini's multi-turn chat to adaptively ask questions.
    """

    def __init__(self):
        super().__init__(
            system_prompt=INTERVIEW_SYSTEM_PROMPT,
            temperature=agent_config.FOLLOWUP_TEMPERATURE,
            top_p=agent_config.FOLLOWUP_TOP_P,
            top_k=agent_config.FOLLOWUP_TOP_K,
            response_mime_type='application/json',
            create_chat=True
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.question_count = 0

    def _track_tokens(self, response):
        """Accumulate token usage across chat turns."""
        if hasattr(response, 'usage_metadata'):
            um = response.usage_metadata
            self.total_input_tokens += getattr(um, 'prompt_token_count', 0) or 0
            self.total_output_tokens += getattr(um, 'candidates_token_count', 0) or 0

    def _parse_response(self, response) -> Dict:
        """Parse and validate the JSON response from the LLM."""
        self._track_tokens(response)
        try:
            result = json.loads(response.text)
            action = result.get("action", "done")
            if action == "ask" and "question" in result:
                self.question_count += 1
                q = result["question"]
                q.setdefault("id", self.question_count)
                q.setdefault("category", "general")
                return {"action": "ask", "question": q}
            return {"action": "done"}
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse interview response: {e}")
            return {"action": "done"}

    def start_interview(self, user_idea: str) -> Dict:
        """Send the user's research idea and get the first targeted question."""
        prompt = f"""Analyze this research idea and identify the MOST underspecified part. Then ask one precise question to fill that gap.

IDEA:
{user_idea}

Remember: reference specific parts of their idea. Do NOT ask generic questions."""

        try:
            response = self.generate_chat_response(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"Error starting interview: {e}")
            return {
                "action": "ask",
                "question": {
                    "id": 1,
                    "category": "method",
                    "question": "What's the core technical mechanism in your approach — can you describe the specific architecture or algorithm?"
                }
            }

    def continue_interview(self, answer: str) -> Dict:
        """Process user's answer and return next question or done signal."""
        if self.question_count >= 3:
            return {"action": "done"}

        try:
            response = self.generate_chat_response(answer)
            result = self._parse_response(response)
            if result["action"] == "ask" and self.question_count > 3:
                return {"action": "done"}
            return result
        except Exception as e:
            logger.error(f"Error continuing interview: {e}")
            return {"action": "done"}

    def get_cost(self) -> float:
        """Calculate accumulated cost across all interview turns."""
        cost = (self.total_input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
        cost += (self.total_output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
        return cost
