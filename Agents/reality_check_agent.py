"""
Reality Check Agent: Uses LLM's general knowledge to check if an idea
already exists as a known product, service, or well-established concept.

This is critical because arXiv only contains academic papers, not information
about existing products like Facebook, Google, Uber, etc.
"""
import json
import logging
from typing import Dict, Optional

from Agents.Agent import Agent
import config

logger = logging.getLogger(__name__)


REALITY_CHECK_PROMPT = """You are a research originality assessor with broad knowledge of existing technologies, products, services, and well-known research areas.

## Your Task
Determine if the user's research/product idea already exists or closely resembles something well-known.

## Important
- Consider existing products (Facebook, Google, Uber, etc.)
- Consider established research areas and solved problems
- Consider patents and known technologies
- Be honest - if the idea is essentially describing something that exists, say so

## Output Format
Return ONLY valid JSON:

{
  "already_exists": true/false,
  "confidence": 0.0-1.0,
  "existing_examples": [
    {
      "name": "Name of existing product/research",
      "similarity": 0.0-1.0,
      "description": "Brief description of how it relates"
    }
  ],
  "assessment": "Brief explanation of your assessment",
  "novelty_aspects": ["List any aspects that might still be novel"],
  "recommendation": "Your recommendation for the user"
}

## Examples

Input: "A mobile app where users can request rides from nearby drivers using GPS"
Output:
{
  "already_exists": true,
  "confidence": 0.95,
  "existing_examples": [
    {"name": "Uber", "similarity": 0.98, "description": "Ride-hailing app using GPS to connect riders with drivers"},
    {"name": "Lyft", "similarity": 0.95, "description": "Similar ride-sharing platform"},
    {"name": "Grab", "similarity": 0.90, "description": "Southeast Asian ride-hailing service"}
  ],
  "assessment": "This idea is essentially describing existing ride-hailing services like Uber and Lyft, which have been operating since 2009-2012.",
  "novelty_aspects": [],
  "recommendation": "This concept already exists as a major industry. Consider what specific innovation or improvement you would bring that differs from existing services."
}

Input: "Using quantum entanglement for faster-than-light communication"
Output:
{
  "already_exists": false,
  "confidence": 0.90,
  "existing_examples": [
    {"name": "Quantum Communication Research", "similarity": 0.40, "description": "Existing research uses entanglement for secure communication, but not FTL"}
  ],
  "assessment": "While quantum entanglement is well-studied, using it for FTL communication violates known physics (no-communication theorem). This is not an existing product but may be physically impossible.",
  "novelty_aspects": ["The FTL aspect is novel but likely impossible"],
  "recommendation": "Review the no-communication theorem in quantum mechanics. Consider focusing on quantum key distribution (QKD) which is achievable."
}
"""


class RealityCheckAgent(Agent):
    """
    Agent that checks if an idea already exists using LLM's general knowledge.
    This catches cases where arXiv papers won't help (e.g., Facebook, Uber).
    """
    
    def __init__(self):
        super().__init__(
            system_prompt=REALITY_CHECK_PROMPT,
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            response_mime_type='application/json',
            create_chat=False
        )
        self.last_token_count = 0
    
    def check_idea(self, user_idea: str) -> Dict:
        """
        Check if an idea already exists in the real world.
        
        Args:
            user_idea: The user's research/product idea
            
        Returns:
            Dict with existence check results
        """
        prompt = f"""Analyze this idea and determine if it already exists:

---
{user_idea}
---

Be thorough - check against known products, services, technologies, and research areas.
Return your assessment as JSON."""

        try:
            response = self.generate_text_generation_response(prompt)
            
            if hasattr(response, 'usage_metadata'):
                self.last_token_count = response.usage_metadata.total_token_count
            
            result = json.loads(response.text)
            
            # Ensure required fields exist
            result.setdefault('already_exists', False)
            result.setdefault('confidence', 0.0)
            result.setdefault('existing_examples', [])
            result.setdefault('assessment', '')
            result.setdefault('novelty_aspects', [])
            result.setdefault('recommendation', '')
            
            logger.info(f"Reality check: already_exists={result['already_exists']}, confidence={result['confidence']}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reality check JSON: {e}")
            return self._default_response()
        except Exception as e:
            logger.error(f"Reality check failed: {e}")
            return self._default_response()
    
    def _default_response(self) -> Dict:
        """Return default response if check fails."""
        return {
            "already_exists": False,
            "confidence": 0.0,
            "existing_examples": [],
            "assessment": "Unable to perform reality check. Proceeding with paper analysis.",
            "novelty_aspects": [],
            "recommendation": "Consider manually verifying if similar products/research exist."
        }
    
    def get_warning_message(self, result: Dict) -> Optional[str]:
        """
        Generate a warning message if the idea likely already exists.
        
        Args:
            result: The check result dict
            
        Returns:
            Warning message string or None
        """
        if not result.get('already_exists', False):
            return None
        
        confidence = result.get('confidence', 0)
        examples = result.get('existing_examples', [])
        
        if confidence >= 0.8 and examples:
            top_example = examples[0]
            return (
                f"⚠️ **Warning: This idea may already exist!**\n\n"
                f"Your idea closely resembles **{top_example.get('name', 'existing products')}** "
                f"(similarity: {top_example.get('similarity', 0):.0%}).\n\n"
                f"*{result.get('assessment', '')}*\n\n"
                f"**Recommendation:** {result.get('recommendation', 'Consider what makes your approach unique.')}"
            )
        elif confidence >= 0.5:
            return (
                f"⚠️ **Note:** Similar concepts may exist.\n\n"
                f"*{result.get('assessment', '')}*"
            )
        
        return None
    
    def adjust_originality_score(self, original_score: int, result: Dict) -> int:
        """
        Adjust the originality score based on reality check.
        
        Args:
            original_score: Score from paper analysis (0-100)
            result: Reality check result
            
        Returns:
            Adjusted score
        """
        if not result.get('already_exists', False):
            return original_score
        
        confidence = result.get('confidence', 0)
        examples = result.get('existing_examples', [])
        
        if examples:
            max_similarity = max(ex.get('similarity', 0) for ex in examples)
        else:
            max_similarity = 0
        
        # Calculate penalty based on confidence and similarity
        # High confidence + high similarity = big penalty
        penalty_factor = confidence * max_similarity
        
        # Penalty can reduce score by up to 80%
        penalty = int(original_score * penalty_factor * 0.8)
        
        adjusted = max(5, original_score - penalty)  # Minimum score of 5
        
        logger.info(f"Score adjusted: {original_score} -> {adjusted} (penalty factor: {penalty_factor:.2f})")
        
        return adjusted
    
    def get_cost(self) -> float:
        """Calculate cost for the check."""
        if self.last_token_count > 0:
            input_tokens = self.last_token_count * 0.7
            output_tokens = self.last_token_count * 0.3
            
            cost = (input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            cost += (output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            return cost
        return 0.0

