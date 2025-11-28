"""
Layer 2: Global originality aggregation.
Combines Layer 1 results with hard-coded scoring logic.
"""
import logging
from typing import List, Dict, Optional

from Agents.Agent import Agent
import config
from models.analysis import (
    Layer1Result,
    Layer2Result,
    CriteriaScores,
    SentenceAnnotation,
    MatchedSection,
    OriginalityLabel,
    CostBreakdown
)

logger = logging.getLogger(__name__)


LAYER2_SUMMARY_PROMPT = """You are a research originality summarizer. Based on the analysis results provided, write a brief 1-2 sentence summary explaining the originality assessment.

## Input
You will receive:
- Global originality score (0-100, higher = more original)
- Number of papers analyzed
- Aggregated criteria scores
- Sentence-level labels (red = low originality, yellow = medium, green = high)

## Output
Write ONLY a 1-2 sentence summary. Be specific about:
- Main areas of overlap (if any)
- Main areas of originality (if any)
- Actionable insight for the researcher

Examples:
- "Your idea shows strong originality in methodology, but the problem formulation has significant overlap with existing work in X. Consider differentiating your approach to Y."
- "This research idea appears highly original across all criteria. The closest related work focuses on Z, which differs from your proposed approach."
- "Moderate originality detected. While your application domain is novel, the core method shares similarities with papers on A and B."

Do NOT include any JSON or formatting. Return only plain text summary.
"""


class Layer2Aggregator:
    """
    Layer 2: Aggregates Layer 1 results and produces final originality assessment.
    Uses hard-coded logic for scoring, LLM only for summary generation.
    """
    
    def __init__(self):
        """Initialize with optional LLM for summary generation."""
        self.summary_agent = None
        self.last_token_count = 0
    
    def _init_summary_agent(self):
        """Lazy initialization of summary agent."""
        if self.summary_agent is None:
            self.summary_agent = Agent(
                system_prompt=LAYER2_SUMMARY_PROMPT,
                temperature=config.LAYER2_TEMPERATURE,
                top_p=config.LAYER2_TOP_P,
                top_k=config.LAYER2_TOP_K,
                response_mime_type='text/plain',
                create_chat=False
            )
    
    def aggregate(
        self,
        layer1_results: List[Layer1Result],
        user_sentences: List[str],
        cost_breakdown: CostBreakdown = None
    ) -> Layer2Result:
        """
        Aggregate Layer 1 results into final originality assessment.
        
        Args:
            layer1_results: List of Layer1Result from each paper
            user_sentences: Original user sentences
            cost_breakdown: Optional cost tracking object
            
        Returns:
            Layer2Result with global scores and sentence annotations
        """
        if not layer1_results:
            return self._create_empty_result(user_sentences)
        
        # Step 1: Aggregate criteria scores (average across papers)
        aggregated_criteria = self._aggregate_criteria(layer1_results)
        
        # Step 2: Compute sentence-level originality using hard-coded logic
        sentence_annotations = self._compute_sentence_annotations(
            layer1_results, user_sentences
        )
        
        # Step 3: Compute global scores
        global_overlap = self._compute_global_overlap(sentence_annotations)
        global_originality = self._overlap_to_originality_score(global_overlap)
        global_label = self._score_to_label(global_originality)
        
        # Step 4: Generate summary using LLM
        summary = self._generate_summary(
            global_originality=global_originality,
            aggregated_criteria=aggregated_criteria,
            sentence_annotations=sentence_annotations,
            num_papers=len(layer1_results)
        )
        
        # Update cost breakdown
        if cost_breakdown:
            cost_breakdown.layer2 = self.get_cost()
            cost_breakdown.total = (
                cost_breakdown.followup +
                cost_breakdown.keywords +
                cost_breakdown.layer1 +
                cost_breakdown.layer2
            )
        
        return Layer2Result(
            global_originality_score=global_originality,
            global_overlap_score=global_overlap,
            label=global_label,
            sentence_annotations=sentence_annotations,
            summary=summary,
            aggregated_criteria=aggregated_criteria,
            papers_analyzed=len(layer1_results),
            cost=cost_breakdown or CostBreakdown()
        )
    
    def _aggregate_criteria(self, results: List[Layer1Result]) -> CriteriaScores:
        """Average criteria scores across all papers."""
        if not results:
            return CriteriaScores(0.0, 0.0, 0.0, 0.0)
        
        problem_sum = sum(r.criteria_scores.problem_similarity for r in results)
        method_sum = sum(r.criteria_scores.method_similarity for r in results)
        domain_sum = sum(r.criteria_scores.domain_overlap for r in results)
        contrib_sum = sum(r.criteria_scores.contribution_similarity for r in results)
        
        n = len(results)
        return CriteriaScores(
            problem_similarity=problem_sum / n,
            method_similarity=method_sum / n,
            domain_overlap=domain_sum / n,
            contribution_similarity=contrib_sum / n
        )
    
    def _compute_sentence_annotations(
        self,
        results: List[Layer1Result],
        user_sentences: List[str]
    ) -> List[SentenceAnnotation]:
        """
        Compute sentence-level originality using hard-coded logic.
        
        For each sentence:
        - Find MAX overlap score across all papers (worst case)
        - Classify based on thresholds
        - Collect all matched sections
        """
        annotations = []
        
        for idx, sentence in enumerate(user_sentences):
            # Collect overlap scores and matches from all papers
            overlap_scores = []
            all_matches = []
            
            for result in results:
                for sent_analysis in result.sentence_analyses:
                    if sent_analysis.sentence_index == idx:
                        overlap_scores.append(sent_analysis.overlap_score)
                        all_matches.extend(sent_analysis.matched_sections)
                        break
            
            # Use MAX overlap (worst case for originality)
            max_overlap = max(overlap_scores) if overlap_scores else 0.0
            
            # Convert to originality (inverse)
            originality = 1.0 - max_overlap
            
            # Classify using hard-coded thresholds
            # NOTE: Thresholds are on OVERLAP, not originality
            if max_overlap >= config.HIGH_OVERLAP_THRESHOLD:
                label = OriginalityLabel.LOW  # Red - high overlap
            elif max_overlap >= config.MEDIUM_OVERLAP_THRESHOLD:
                label = OriginalityLabel.MEDIUM  # Yellow
            else:
                label = OriginalityLabel.HIGH  # Green - low overlap
            
            # Sort matches by similarity, take top ones
            all_matches.sort(key=lambda x: x.similarity, reverse=True)
            top_matches = all_matches[:5]  # Limit to top 5 matches
            
            annotations.append(SentenceAnnotation(
                index=idx,
                sentence=sentence,
                originality_score=originality,
                overlap_score=max_overlap,
                label=label,
                linked_sections=top_matches
            ))
        
        return annotations
    
    def _compute_global_overlap(
        self,
        annotations: List[SentenceAnnotation]
    ) -> float:
        """
        Compute global overlap score from sentence annotations.
        Uses weighted average - problem-related sentences weighted higher.
        """
        if not annotations:
            return 0.0
        
        # Simple average for now
        # Could be enhanced with importance weighting
        total_overlap = sum(a.overlap_score for a in annotations)
        return total_overlap / len(annotations)
    
    def _overlap_to_originality_score(self, overlap: float) -> int:
        """
        Convert overlap (0-1) to originality score (0-100).
        Higher originality = lower overlap.
        """
        originality = (1.0 - overlap) * 100
        return int(max(0, min(100, originality)))
    
    def _score_to_label(self, originality_score: int) -> OriginalityLabel:
        """Convert originality score to label."""
        if originality_score >= config.SCORE_YELLOW_MAX:
            return OriginalityLabel.HIGH
        elif originality_score >= config.SCORE_RED_MAX:
            return OriginalityLabel.MEDIUM
        else:
            return OriginalityLabel.LOW
    
    def _generate_summary(
        self,
        global_originality: int,
        aggregated_criteria: CriteriaScores,
        sentence_annotations: List[SentenceAnnotation],
        num_papers: int
    ) -> str:
        """Generate natural language summary using LLM."""
        self._init_summary_agent()
        
        # Count labels
        red_count = len([a for a in sentence_annotations if a.label == OriginalityLabel.LOW])
        yellow_count = len([a for a in sentence_annotations if a.label == OriginalityLabel.MEDIUM])
        green_count = len([a for a in sentence_annotations if a.label == OriginalityLabel.HIGH])
        
        prompt = f"""Generate a brief summary for this originality assessment:

Global Originality Score: {global_originality}/100
Papers Analyzed: {num_papers}

Criteria Scores (0-1, higher = more similar to existing work):
- Problem Similarity: {aggregated_criteria.problem_similarity:.2f}
- Method Similarity: {aggregated_criteria.method_similarity:.2f}
- Domain Overlap: {aggregated_criteria.domain_overlap:.2f}
- Contribution Similarity: {aggregated_criteria.contribution_similarity:.2f}

Sentence Labels:
- Red (low originality): {red_count} sentences
- Yellow (medium): {yellow_count} sentences
- Green (high originality): {green_count} sentences

Write a 1-2 sentence summary explaining the assessment and giving actionable insight."""

        try:
            response = self.summary_agent.generate_text_generation_response(prompt)
            
            if hasattr(response, 'usage_metadata'):
                self.last_token_count = response.usage_metadata.total_token_count
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._generate_fallback_summary(
                global_originality, aggregated_criteria, red_count, yellow_count, green_count
            )
    
    def _generate_fallback_summary(
        self,
        score: int,
        criteria: CriteriaScores,
        red: int,
        yellow: int,
        green: int
    ) -> str:
        """Generate fallback summary if LLM fails."""
        if score >= 70:
            level = "high"
        elif score >= 40:
            level = "moderate"
        else:
            level = "low"
        
        # Find highest overlap criterion
        criteria_names = {
            'problem definition': criteria.problem_similarity,
            'methodology': criteria.method_similarity,
            'application domain': criteria.domain_overlap,
            'contributions': criteria.contribution_similarity
        }
        max_criterion = max(criteria_names.items(), key=lambda x: x[1])
        
        if max_criterion[1] > 0.5:
            overlap_note = f" Main overlap detected in {max_criterion[0]}."
        else:
            overlap_note = ""
        
        return f"Your idea shows {level} originality (score: {score}/100).{overlap_note} {red} sentences have significant overlap, {yellow} have moderate overlap, and {green} appear novel."
    
    def _create_empty_result(self, user_sentences: List[str]) -> Layer2Result:
        """Create result when no papers were analyzed."""
        annotations = [
            SentenceAnnotation(
                index=i,
                sentence=sent,
                originality_score=1.0,
                overlap_score=0.0,
                label=OriginalityLabel.HIGH,
                linked_sections=[]
            )
            for i, sent in enumerate(user_sentences)
        ]
        
        return Layer2Result(
            global_originality_score=100,
            global_overlap_score=0.0,
            label=OriginalityLabel.HIGH,
            sentence_annotations=annotations,
            summary="No similar papers were found. Your idea appears to be highly original, though this may indicate a gap in the search rather than true novelty.",
            papers_analyzed=0,
            cost=CostBreakdown()
        )
    
    def get_cost(self) -> float:
        """Calculate cost for summary generation."""
        if self.last_token_count > 0:
            input_tokens = self.last_token_count * 0.7
            output_tokens = self.last_token_count * 0.3
            
            cost = (input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
            cost += (output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            return cost
        return 0.0

