"""
Follow-up question agent for clarifying user's research idea.
Generates targeted questions to improve originality assessment accuracy.
"""
import json
import logging
from typing import List, Dict

from Agents.Agent import Agent
import config

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


class FollowUpAgent(Agent):
    """
    Agent for generating follow-up questions to clarify research ideas.
    Questions are tailored to improve originality assessment.
    """
    
    def __init__(self):
        super().__init__(
            system_prompt=FOLLOWUP_SYSTEM_PROMPT,
            temperature=config.FOLLOWUP_TEMPERATURE,
            top_p=config.FOLLOWUP_TOP_P,
            top_k=config.FOLLOWUP_TOP_K,
            response_mime_type='application/json',
            create_chat=False
        )
        self.last_token_count = 0
    
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
                self.last_token_count = response.usage_metadata.total_token_count
            
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
        if self.last_token_count > 0:
            # Approximate input/output split
            input_tokens = self.last_token_count * 0.7
            output_tokens = self.last_token_count * 0.3
            
            cost = (input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            cost += (output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            return cost
        return 0.0

