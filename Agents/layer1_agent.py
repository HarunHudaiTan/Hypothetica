"""
Layer 1 Agent: Per-paper originality analysis.
HYBRID APPROACH: Combines 11-question framework with sentence-level analysis.

Evaluates how similar a single paper is to the user's research idea using:
1. Rigorous dimension-based analysis (11 questions with evidence)
2. Sentence-level mapping for UI highlighting and RAG
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
    MatchedSection,
    PaperMetadata,
    DimensionAnalysis,
    QuestionAnalysis,
    SimilarityCategory
)

logger = logging.getLogger(__name__)


LAYER1_SYSTEM_PROMPT = """You are an expert **Intellectual Property Analyst** specializing in research originality assessment. Your role is to evaluate research ideas with the rigor of a patent examiner, determining novelty against existing academic literature.

## Your Mission
Analyze a research paper against a user's research idea to identify **prior art** and assess **originality conflicts**.

## PART 1: DIMENSION ANALYSIS (11 Questions)

Evaluate FOUR DIMENSIONS using categorical scores:

**SIMILARITY SCALE:**
- **NONE**: No overlap whatsoever (0.0)
- **MINIMAL**: Slight overlap in broad concepts (0.25)
- **MODERATE**: Some meaningful overlap but significant differences (0.5)
- **SIGNIFICANT**: Substantial overlap with minor differences (0.75)
- **COMPLETE**: Identical or nearly identical (1.0)

### DIMENSION 1: Technical Problem Novelty (3 questions)

**Q1.1: Does the paper address the exact same research question?**
**Q1.2: Are the problem constraints and requirements identical?**
**Q1.3: Is the motivation and problem context the same?**

### DIMENSION 2: Methodological Innovation (3 questions)

**Q2.1: Are the core algorithms, models, or techniques the same?**
**Q2.2: Does the paper use the same architectural choices or design patterns?**
**Q2.3: Are the implementation strategies and technical approaches identical?**

### DIMENSION 3: Application Domain Overlap (3 questions)

**Q3.1: Do both target the same industry, field, or application area?**
**Q3.2: Are the end-users and stakeholders the same?**
**Q3.3: Do they solve problems for the same market segment?**

### DIMENSION 4: Innovation Claims Overlap (2 questions)

**Q4.1: Do both claim the same primary innovations or breakthroughs?**
**Q4.2: Are the technical advantages and benefits identical?**

---

## PART 2: SENTENCE-LEVEL ANALYSIS

For EACH sentence in the user's idea:
1. Determine overlap_score (0.0-1.0) based on dimension analysis
2. Identify which dimension is most relevant (primary_dimension)
3. Provide 1-2 evidence quotes from the paper supporting the overlap
4. List matched paper sections with FULL CONTEXT:
   - heading: The section heading
   - text_snippet: The ACTUAL TEXT from the paper that matches (2-3 sentences)
   - reason: Why this text matches the user's sentence
   - dimension: Which dimension this match relates to (problem/method/domain/claims)

---

## OUTPUT FORMAT

Return ONLY valid JSON:

```json
{
  "paper_id": "paper_01",
  "dimension_analyses": {
    "technical_problem_novelty": [
      {
        "question": "Does the paper address the exact same research question?",
        "score": "MODERATE",
        "answer": "Both address NLP tasks but different specific problems.",
        "evidence": ["Quote from paper...", "Another quote..."]
      },
      {
        "question": "Are the problem constraints and requirements identical?",
        "score": "MINIMAL",
        "answer": "Different scale and performance requirements.",
        "evidence": ["Quote..."]
      },
      {
        "question": "Is the motivation and problem context the same?",
        "score": "MODERATE",
        "answer": "Similar industry motivation but different contexts.",
        "evidence": ["Quote..."]
      }
    ],
    "methodological_innovation": [
      {
        "question": "Are the core algorithms, models, or techniques the same?",
        "score": "SIGNIFICANT",
        "answer": "Both use transformer architectures with similar approaches.",
        "evidence": ["Quote..."]
      },
      {
        "question": "Does the paper use the same architectural choices?",
        "score": "MODERATE",
        "answer": "Similar high-level architecture but different components.",
        "evidence": ["Quote..."]
      },
      {
        "question": "Are the implementation strategies identical?",
        "score": "MINIMAL",
        "answer": "Different implementation details and optimization strategies.",
        "evidence": ["Quote..."]
      }
    ],
    "application_domain_overlap": [
      {
        "question": "Do both target the same industry or field?",
        "score": "SIGNIFICANT",
        "answer": "Both target healthcare/biomedical applications.",
        "evidence": ["Quote..."]
      },
      {
        "question": "Are the end-users and stakeholders the same?",
        "score": "MODERATE",
        "answer": "Similar user base but different primary users.",
        "evidence": ["Quote..."]
      },
      {
        "question": "Do they solve problems for the same market segment?",
        "score": "MODERATE",
        "answer": "Overlapping market segments.",
        "evidence": ["Quote..."]
      }
    ],
    "innovation_claims_overlap": [
      {
        "question": "Do both claim the same primary innovations?",
        "score": "MINIMAL",
        "answer": "Different primary contributions claimed.",
        "evidence": ["Quote..."]
      },
      {
        "question": "Are the technical advantages and benefits identical?",
        "score": "MODERATE",
        "answer": "Some overlapping benefits claimed.",
        "evidence": ["Quote..."]
      }
    ]
  },
  "sentence_level": [
    {
      "sentence_index": 0,
      "sentence": "The user's first sentence.",
      "overlap_score": 0.65,
      "primary_dimension": "methodological_innovation",
      "evidence": ["Relevant quote from paper..."],
      "matched_sections": [
        {
          "heading": "METHODOLOGY",
          "text_snippet": "We propose a transformer-based architecture that processes input sequences through self-attention mechanisms, enabling the model to capture long-range dependencies.",
          "reason": "Both use transformer architecture with self-attention for sequence processing",
          "dimension": "methodological_innovation"
        }
      ]
    }
  ],
  "analysis_notes": "Brief 1-2 sentence summary of key overlaps and differences"
}
```

## CRITICAL RULES

1. **Score ALL 11 questions** - never skip any
2. **Provide 1-4 evidence quotes per question** - actual text from the paper
3. **Be conservative** - when uncertain, choose higher similarity to flag potential prior art
4. **Map EVERY user sentence** to overlap score and primary dimension
5. **Evidence-based only** - do NOT hallucinate paper content
6. **Use categorical scores** - NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE only
7. **Sentence overlap scores** should reflect the dimension analysis (e.g., if methodology is SIGNIFICANT=0.75, sentences about methods should be ~0.75)
8. **ALWAYS include text_snippet** in matched_sections - copy 2-3 actual sentences from the paper that match
9. **ALWAYS include dimension** in matched_sections - specify which dimension (problem/method/domain/claims) the match relates to


"""


class Layer1Agent(Agent):
    """
    Layer 1 Agent for per-paper originality analysis.
    HYBRID: Combines dimension analysis with sentence-level mapping.
    """
    
    # Category to numerical mapping
    CATEGORY_MAP = {
        "NONE": 0.0,
        "MINIMAL": 0.25,
        "MODERATE": 0.5,
        "SIGNIFICANT": 0.75,
        "COMPLETE": 1.0
    }
    
    # Dimension weights for overall score
    DIMENSION_WEIGHTS = {
        "technical_problem_novelty": 0.30,
        "methodological_innovation": 0.40,
        "application_domain_overlap": 0.20,
        "innovation_claims_overlap": 0.10
    }
    
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
            Layer1Result with dimension analysis and sentence-level mapping
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
            return self._create_error_result(paper, user_sentences, str(e))
        except Exception as e:
            logger.error(f"Layer1 analysis failed for {paper.paper_id}: {e}")
            return self._create_error_result(paper, user_sentences, str(e))
    
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
        if paper.headings:
            for heading in paper.headings:
                if heading.section_text and heading.is_valid:
                    # Truncate long sections
                    section_content = heading.section_text[:2000]
                    if len(heading.section_text) > 2000:
                        section_content += "...[truncated]"
                    sections_text += f"\n### {heading.text}\n{section_content}\n"
        
        prompt = f"""Analyze the following paper against the user's research idea.

## USER'S RESEARCH IDEA
{user_idea}

## USER'S IDEA SENTENCES (analyze each one for sentence_level output)
{sentences_text}

## PAPER TO ANALYZE
Paper ID: {paper.paper_id}
ArXiv ID: {paper.arxiv_id}
Title: {paper.title}
Categories: {', '.join(paper.categories)}

### ABSTRACT
{paper.abstract}

### EXTRACTED SECTIONS
{sections_text if sections_text else paper_context if paper_context else "No detailed sections available - analyze based on title and abstract."}

## TASK
1. Answer ALL 11 questions in the dimension_analyses with categorical scores and evidence
2. For EACH user sentence (indices 0 to {len(user_sentences)-1}), provide sentence_level analysis
3. Provide brief analysis_notes summarizing key overlaps/differences

Return valid JSON only."""

        return prompt
    
    def _category_to_float(self, category: str) -> float:
        """Convert categorical score to float."""
        return self.CATEGORY_MAP.get(category.upper(), 0.0)
    
    def _parse_result(
        self,
        result_dict: dict,
        paper: Paper,
        user_sentences: List[str]
    ) -> Layer1Result:
        """Parse JSON response into Layer1Result object."""
        
        # Parse dimension analyses
        dimension_analyses = []
        dim_scores = {}
        
        raw_dimensions = result_dict.get('dimension_analyses', {})
        
        for dim_name, questions_list in raw_dimensions.items():
            questions = []
            scores_for_dim = []
            
            for q_data in questions_list:
                score_str = q_data.get('score', 'NONE')
                score_float = self._category_to_float(score_str)
                scores_for_dim.append(score_float)
                
                questions.append(QuestionAnalysis(
                    question=q_data.get('question', ''),
                    score=SimilarityCategory.from_string(score_str),
                    answer=q_data.get('answer', ''),
                    evidence=q_data.get('evidence', [])
                ))
            
            dimension_analyses.append(DimensionAnalysis(
                dimension_name=dim_name,
                questions=questions
            ))
            
            # Calculate average for this dimension
            if scores_for_dim:
                dim_scores[dim_name] = sum(scores_for_dim) / len(scores_for_dim)
            else:
                dim_scores[dim_name] = 0.0
        
        # Build CriteriaScores from dimension averages
        criteria = CriteriaScores(
            problem_similarity=dim_scores.get('technical_problem_novelty', 0.0),
            method_similarity=dim_scores.get('methodological_innovation', 0.0),
            domain_overlap=dim_scores.get('application_domain_overlap', 0.0),
            contribution_similarity=dim_scores.get('innovation_claims_overlap', 0.0)
        )
        
        # Calculate weighted overall score
        overall_overlap = criteria.weighted_average
        
        # Create paper metadata for grounded RAG display
        paper_metadata = PaperMetadata(
            paper_id=paper.paper_id,
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors if hasattr(paper, 'authors') else [],
            publication_date=paper.published if hasattr(paper, 'published') else "",
            categories=paper.categories if hasattr(paper, 'categories') else [],
            abstract=paper.abstract if hasattr(paper, 'abstract') else ""
        )
        
        # Parse sentence-level analysis
        sentence_analyses = []
        for sent_data in result_dict.get('sentence_level', []):
            idx = sent_data.get('sentence_index', 0)
            
            # Parse matched sections with full context
            matched = []
            for match in sent_data.get('matched_sections', []):
                matched.append(MatchedSection(
                    chunk_id="",  # Will be linked by RAG later
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    heading=match.get('heading', ''),
                    heading_hierarchy=[match.get('heading', '')],  # Single heading for now
                    text_snippet=match.get('text_snippet', ''),  # Now populated by LLM
                    context_before="",  # Can be filled by RAG later
                    context_after="",   # Can be filled by RAG later
                    similarity=sent_data.get('overlap_score', 0.0),
                    reason=match.get('reason', ''),
                    dimension=match.get('dimension', sent_data.get('primary_dimension', '')),
                    paper_metadata=paper_metadata
                ))
            
            # Get sentence text
            sentence = sent_data.get('sentence', '')
            if not sentence and idx < len(user_sentences):
                sentence = user_sentences[idx]
            
            sentence_analyses.append(SentenceAnalysis(
                sentence=sentence,
                sentence_index=idx,
                overlap_score=float(sent_data.get('overlap_score', 0.0)),
                primary_dimension=sent_data.get('primary_dimension', ''),
                matched_sections=matched,
                evidence=sent_data.get('evidence', [])
            ))
        
        # Ensure we have analysis for all sentences
        analyzed_indices = {sa.sentence_index for sa in sentence_analyses}
        for i, sent in enumerate(user_sentences):
            if i not in analyzed_indices:
                sentence_analyses.append(SentenceAnalysis(
                    sentence=sent,
                    sentence_index=i,
                    overlap_score=0.0,
                    primary_dimension="",
                    matched_sections=[],
                    evidence=[]
                ))
        
        # Sort by index
        sentence_analyses.sort(key=lambda x: x.sentence_index)
        
        return Layer1Result(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            arxiv_id=paper.arxiv_id,
            overall_overlap_score=overall_overlap,
            criteria_scores=criteria,
            dimension_analyses=dimension_analyses,
            sentence_analyses=sentence_analyses,
            analysis_notes=result_dict.get('analysis_notes', ''),
            tokens_used=self.last_token_count
        )
    
    def _create_error_result(
        self, 
        paper: Paper, 
        user_sentences: List[str],
        error: str
    ) -> Layer1Result:
        """Create a result object for failed analysis."""
        # Create empty sentence analyses for all sentences
        sentence_analyses = [
            SentenceAnalysis(
                sentence=sent,
                sentence_index=i,
                overlap_score=0.0,
                primary_dimension="",
                matched_sections=[],
                evidence=[]
            )
            for i, sent in enumerate(user_sentences)
        ]
        
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
            dimension_analyses=[],
            sentence_analyses=sentence_analyses,
            analysis_notes=f"Analysis failed: {error}"
        )
    
    def get_cost(self) -> float:
        """Calculate cost for the last analysis."""
        if self.last_token_count > 0:
            # Hybrid approach uses more tokens (larger output)
            input_tokens = self.last_token_count * 0.6
            output_tokens = self.last_token_count * 0.4
            
            cost = (input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            cost += (output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            return cost
        return 0.0


# Utility functions for external use

def categorical_to_numerical(category: str) -> float:
    """Convert categorical similarity score to numerical value (0.0-1.0)."""
    mapping = {
        "NONE": 0.0,
        "MINIMAL": 0.25,
        "MODERATE": 0.5,
        "SIGNIFICANT": 0.75,
        "COMPLETE": 1.0
    }
    return mapping.get(category.upper(), 0.0)


def calculate_dimension_scores(result: Layer1Result) -> dict:
    """
    Extract dimension scores from Layer1Result.
    
    Returns:
        Dictionary with dimension averages and overall score
    """
    dim_scores = {}
    
    for dim in result.dimension_analyses:
        dim_scores[dim.dimension_name] = dim.average_score
    
    # Calculate weighted overall
    overall = (
        dim_scores.get('technical_problem_novelty', 0.0) * 0.30 +
        dim_scores.get('methodological_innovation', 0.0) * 0.40 +
        dim_scores.get('application_domain_overlap', 0.0) * 0.20 +
        dim_scores.get('innovation_claims_overlap', 0.0) * 0.10
    )
    
    return {
        "dimension_scores": {k: round(v, 3) for k, v in dim_scores.items()},
        "overall_overlap_score": round(overall, 3),
        "originality_score": round((1 - overall) * 100, 1)
    }
