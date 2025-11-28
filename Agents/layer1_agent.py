"""
Layer 1 Agent: Per-paper originality analysis.
Evaluates how similar a single paper is to the user's research idea.
"""
import json
import logging
from typing import List, Optional

from Agents.Agent import Agent
import config
from models.paper import Paper
from models.analysis import (
    Layer1Result, 
    CriteriaScores, 
    SentenceAnalysis,
    MatchedSection
)

logger = logging.getLogger(__name__)


LAYER1_SYSTEM_PROMPT = """You are an academic originality assessor, similar to a TÜBİTAK grant reviewer. Your task is to evaluate how similar a research paper is to a user's research idea.

## Your Role
Analyze the relationship between a user's research idea and a given paper. Determine overlap in:
- Problem definition / research question
- Proposed methodology / approach
- Application domain
- Claimed contributions

## Input You Will Receive
1. User's research idea (possibly with clarifications)
2. Paper information: title, abstract, and extracted sections
3. ArXiv categories/keywords

## Evaluation Criteria (Score 0.0 to 1.0, where higher = MORE SIMILAR / LESS ORIGINAL)

1. **problem_similarity**: How similar is the research problem or question?
   - 0.0 = Completely different problems
   - 0.5 = Related but distinct problems
   - 1.0 = Identical problem being addressed

2. **method_similarity**: How similar is the proposed method/approach?
   - 0.0 = Completely different techniques
   - 0.5 = Same general family of methods
   - 1.0 = Identical methodology

3. **domain_overlap**: How much do application domains overlap?
   - 0.0 = Different fields entirely
   - 0.5 = Related fields
   - 1.0 = Same specific domain/application

4. **contribution_similarity**: How similar are the claimed contributions?
   - 0.0 = Different contributions
   - 0.5 = Partial overlap in contributions
   - 1.0 = Same contributions claimed

## Sentence-Level Analysis
For EACH sentence in the user's idea, evaluate:
- overlap_score: How much this specific sentence overlaps with paper content (0.0-1.0)
- matched_sections: Which paper sections relate to this sentence

## Output Format
Return ONLY valid JSON:

{
  "paper_id": "paper_01",
  "overall_overlap_score": 0.45,
  "criteria_scores": {
    "problem_similarity": 0.70,
    "method_similarity": 0.20,
    "domain_overlap": 0.50,
    "contribution_similarity": 0.40
  },
  "sentence_level": [
    {
      "sentence_index": 0,
      "sentence": "The user's first sentence.",
      "overlap_score": 0.65,
      "matched_sections": [
        {
          "heading": "INTRODUCTION",
          "reason": "Similar problem statement",
          "similarity": 0.68
        }
      ]
    }
  ],
  "analysis_notes": "Brief explanation of key overlaps and differences"
}

## Important Guidelines
- Be objective and evidence-based
- Reference specific parts of the paper when identifying overlaps
- If no overlap exists for a criterion, score it near 0.0
- overall_overlap_score should be weighted average: problem(0.3) + method(0.3) + domain(0.2) + contribution(0.2)
- For sentence_level, include ALL sentences from the user's idea
- DO NOT hallucinate paper content - only reference what is provided
"""


class Layer1Agent(Agent):
    """
    Layer 1 Agent for per-paper originality analysis.
    Compares user's idea against a single paper.
    """
    
    def __init__(self):
        super().__init__(
            system_prompt=LAYER1_SYSTEM_PROMPT,
            temperature=config.LAYER1_TEMPERATURE,
            top_p=config.LAYER1_TOP_P,
            top_k=config.LAYER1_TOP_K,
            response_mime_type='application/json',
            create_chat=False
        )
        self.last_token_count = 0
    
    def analyze_paper(
        self,
        user_idea: str,
        user_sentences: List[str],
        paper: Paper,
        paper_context: str = ""
    ) -> Layer1Result:
        """
        Analyze a single paper against the user's idea.
        
        Args:
            user_idea: Full enriched user idea text
            user_sentences: User's idea split into sentences
            paper: Paper object to analyze
            paper_context: Extracted relevant sections from paper
            
        Returns:
            Layer1Result with scores and sentence-level analysis
        """
        # Build prompt with paper information
        prompt = self._build_analysis_prompt(
            user_idea=user_idea,
            user_sentences=user_sentences,
            paper=paper,
            paper_context=paper_context
        )
        
        try:
            response = self.generate_text_generation_response(prompt)
            
            # Track tokens
            if hasattr(response, 'usage_metadata'):
                self.last_token_count = response.usage_metadata.total_token_count
            
            # Parse response
            result_dict = json.loads(response.text)
            return self._parse_result(result_dict, paper, user_sentences)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Layer1 JSON for {paper.paper_id}: {e}")
            return self._create_error_result(paper, str(e))
        except Exception as e:
            logger.error(f"Layer1 analysis failed for {paper.paper_id}: {e}")
            return self._create_error_result(paper, str(e))
    
    def _build_analysis_prompt(
        self,
        user_idea: str,
        user_sentences: List[str],
        paper: Paper,
        paper_context: str
    ) -> str:
        """Build the analysis prompt with all relevant information."""
        
        # Format sentences with indices
        sentences_text = "\n".join([
            f"[{i}] {sent}" for i, sent in enumerate(user_sentences)
        ])
        
        # Format paper sections
        sections_text = ""
        for heading in paper.headings:
            if heading.section_text and heading.is_valid:
                sections_text += f"\n### {heading.text}\n{heading.section_text[:1500]}...\n"
        
        prompt = f"""Analyze the following paper against the user's research idea.

## USER'S RESEARCH IDEA
{user_idea}

## USER'S IDEA SENTENCES (analyze each one)
{sentences_text}

## PAPER TO ANALYZE
Paper ID: {paper.paper_id}
ArXiv ID: {paper.arxiv_id}
Title: {paper.title}
Categories: {', '.join(paper.categories)}

### ABSTRACT
{paper.abstract}

### EXTRACTED SECTIONS
{sections_text if sections_text else paper_context if paper_context else "No sections extracted"}

## TASK
1. Evaluate criteria_scores (problem, method, domain, contribution similarity)
2. Calculate overall_overlap_score as weighted average
3. For EACH sentence in the user's idea, assess overlap with this paper
4. Provide brief analysis_notes

Return valid JSON only."""

        return prompt
    
    def _parse_result(
        self,
        result_dict: dict,
        paper: Paper,
        user_sentences: List[str]
    ) -> Layer1Result:
        """Parse JSON response into Layer1Result object."""
        
        # Parse criteria scores
        criteria_dict = result_dict.get('criteria_scores', {})
        criteria = CriteriaScores(
            problem_similarity=float(criteria_dict.get('problem_similarity', 0.0)),
            method_similarity=float(criteria_dict.get('method_similarity', 0.0)),
            domain_overlap=float(criteria_dict.get('domain_overlap', 0.0)),
            contribution_similarity=float(criteria_dict.get('contribution_similarity', 0.0))
        )
        
        # Parse sentence-level analysis
        sentence_analyses = []
        for sent_data in result_dict.get('sentence_level', []):
            idx = sent_data.get('sentence_index', 0)
            
            # Parse matched sections
            matched = []
            for match in sent_data.get('matched_sections', []):
                matched.append(MatchedSection(
                    chunk_id="",  # Will be linked by RAG later
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    heading=match.get('heading', ''),
                    text_snippet="",  # Will be filled by RAG
                    similarity=float(match.get('similarity', 0.0)),
                    reason=match.get('reason', '')
                ))
            
            # Get sentence text
            sentence = sent_data.get('sentence', '')
            if not sentence and idx < len(user_sentences):
                sentence = user_sentences[idx]
            
            sentence_analyses.append(SentenceAnalysis(
                sentence=sentence,
                sentence_index=idx,
                overlap_score=float(sent_data.get('overlap_score', 0.0)),
                matched_sections=matched
            ))
        
        # Ensure we have analysis for all sentences
        analyzed_indices = {sa.sentence_index for sa in sentence_analyses}
        for i, sent in enumerate(user_sentences):
            if i not in analyzed_indices:
                sentence_analyses.append(SentenceAnalysis(
                    sentence=sent,
                    sentence_index=i,
                    overlap_score=0.0,
                    matched_sections=[]
                ))
        
        # Sort by index
        sentence_analyses.sort(key=lambda x: x.sentence_index)
        
        return Layer1Result(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            arxiv_id=paper.arxiv_id,
            overall_overlap_score=float(result_dict.get('overall_overlap_score', criteria.average)),
            criteria_scores=criteria,
            sentence_analyses=sentence_analyses,
            tokens_used=self.last_token_count
        )
    
    def _create_error_result(self, paper: Paper, error: str) -> Layer1Result:
        """Create a result object for failed analysis."""
        return Layer1Result(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            arxiv_id=paper.arxiv_id,
            overall_overlap_score=0.0,
            criteria_scores=CriteriaScores(
                problem_similarity=0.0,
                method_similarity=0.0,
                domain_overlap=0.0,
                contribution_similarity=0.0
            ),
            sentence_analyses=[]
        )
    
    def get_cost(self) -> float:
        """Calculate cost for the last analysis."""
        if self.last_token_count > 0:
            input_tokens = self.last_token_count * 0.8  # More input for analysis
            output_tokens = self.last_token_count * 0.2
            
            cost = (input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            cost += (output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            return cost
        return 0.0

