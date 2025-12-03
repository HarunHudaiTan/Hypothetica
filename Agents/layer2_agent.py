"""
Layer 2: Global originality aggregation.
Combines Layer 1 results with hard-coded scoring logic.
Enhanced to handle dimension analyses and evidence from hybrid Layer 1.
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
- Dimension scores (problem, method, domain, claims - each 0-1 where higher = more overlap)
- Sentence-level labels (red = low originality, yellow = medium, green = high)

## Output
Write ONLY a 1-2 sentence summary. Be specific about:
- Main areas of overlap (if any) - reference which dimension is highest
- Main areas of originality (if any) - reference which dimension is lowest
- Actionable insight for the researcher

Examples:
- "Your idea shows strong originality in problem formulation and application domain, but the methodology has significant overlap with transformer-based approaches in existing literature. Consider exploring alternative architectures or novel combinations."
- "This research idea appears highly original across all dimensions. The closest related work focuses on different user populations and market segments."
- "Moderate originality detected. While your technical contributions are distinct, the research problem and motivation closely mirror prior work on X. Emphasize what makes your approach uniquely valuable."

Do NOT include any JSON or formatting. Return only plain text summary.
"""


class Layer2Aggregator:
    """
    Layer 2: Aggregates Layer 1 results and produces final originality assessment.
    Uses hard-coded logic for scoring, LLM only for summary generation.
    
    Enhanced to aggregate dimension-level analyses and preserve evidence.
    """
    
    # Dimension names mapping
    DIMENSION_NAMES = {
        "technical_problem_novelty": "Problem Novelty",
        "methodological_innovation": "Methodology",
        "application_domain_overlap": "Application Domain",
        "innovation_claims_overlap": "Innovation Claims"
    }
    
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
        
        # Step 2: Aggregate dimension analyses
        aggregated_dimensions = self._aggregate_dimensions(layer1_results)
        
        # Step 3: Compute sentence-level originality using hard-coded logic
        sentence_annotations = self._compute_sentence_annotations(
            layer1_results, user_sentences
        )
        
        # Step 4: Compute global scores
        global_overlap = self._compute_global_overlap(sentence_annotations)
        global_originality = self._overlap_to_originality_score(global_overlap)
        global_label = self._score_to_label(global_originality)
        
        # Step 5: Generate summary using LLM
        summary = self._generate_summary(
            global_originality=global_originality,
            aggregated_criteria=aggregated_criteria,
            aggregated_dimensions=aggregated_dimensions,
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
                cost_breakdown.layer2 +
                cost_breakdown.reality_check
            )
        
        return Layer2Result(
            global_originality_score=global_originality,
            global_overlap_score=global_overlap,
            label=global_label,
            sentence_annotations=sentence_annotations,
            summary=summary,
            aggregated_criteria=aggregated_criteria,
            aggregated_dimensions=aggregated_dimensions,
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
    
    def _aggregate_dimensions(self, results: List[Layer1Result]) -> Dict[str, float]:
        """
        Aggregate dimension analyses across all papers.
        Returns MAX score per dimension (worst case).
        """
        dimensions = {
            "technical_problem_novelty": [],
            "methodological_innovation": [],
            "application_domain_overlap": [],
            "innovation_claims_overlap": []
        }
        
        for result in results:
            for dim in result.dimension_analyses:
                if dim.dimension_name in dimensions:
                    dimensions[dim.dimension_name].append(dim.average_score)
        
        # Use MAX (worst case for originality)
        aggregated = {}
        for dim_name, scores in dimensions.items():
            if scores:
                aggregated[dim_name] = max(scores)
            else:
                aggregated[dim_name] = 0.0
        
        return aggregated
    
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
        - Collect all matched sections and evidence
        - Track primary dimension
        """
        annotations = []
        
        for idx, sentence in enumerate(user_sentences):
            # Collect overlap scores, matches, evidence from all papers
            overlap_scores = []
            all_matches = []
            all_evidence = []
            primary_dimensions = []
            
            for result in results:
                for sent_analysis in result.sentence_analyses:
                    if sent_analysis.sentence_index == idx:
                        overlap_scores.append(sent_analysis.overlap_score)
                        all_matches.extend(sent_analysis.matched_sections)
                        all_evidence.extend(sent_analysis.evidence)
                        if sent_analysis.primary_dimension:
                            primary_dimensions.append(sent_analysis.primary_dimension)
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
            
            # Deduplicate evidence
            unique_evidence = list(dict.fromkeys(all_evidence))[:5]
            
            # Get most common primary dimension
            primary_dim = ""
            if primary_dimensions:
                from collections import Counter
                primary_dim = Counter(primary_dimensions).most_common(1)[0][0]
            
            annotations.append(SentenceAnnotation(
                index=idx,
                sentence=sentence,
                originality_score=originality,
                overlap_score=max_overlap,
                label=label,
                primary_dimension=primary_dim,
                linked_sections=top_matches,
                evidence=unique_evidence
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
        # Could be enhanced with importance weighting by primary_dimension
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
        aggregated_dimensions: Dict[str, float],
        sentence_annotations: List[SentenceAnnotation],
        num_papers: int
    ) -> str:
        """Generate natural language summary using LLM."""
        self._init_summary_agent()
        
        # Count labels
        red_count = len([a for a in sentence_annotations if a.label == OriginalityLabel.LOW])
        yellow_count = len([a for a in sentence_annotations if a.label == OriginalityLabel.MEDIUM])
        green_count = len([a for a in sentence_annotations if a.label == OriginalityLabel.HIGH])
        
        # Format dimension scores
        dim_text = ""
        for dim_key, score in aggregated_dimensions.items():
            dim_name = self.DIMENSION_NAMES.get(dim_key, dim_key)
            dim_text += f"- {dim_name}: {score:.2f}\n"
        
        # Find highest and lowest overlap dimensions
        if aggregated_dimensions:
            max_dim = max(aggregated_dimensions.items(), key=lambda x: x[1])
            min_dim = min(aggregated_dimensions.items(), key=lambda x: x[1])
            max_dim_name = self.DIMENSION_NAMES.get(max_dim[0], max_dim[0])
            min_dim_name = self.DIMENSION_NAMES.get(min_dim[0], min_dim[0])
        else:
            max_dim_name = "N/A"
            min_dim_name = "N/A"
        
        prompt = f"""Generate a brief summary for this originality assessment:

Global Originality Score: {global_originality}/100
Papers Analyzed: {num_papers}

Dimension Overlap Scores (0-1, higher = more overlap with existing work):
{dim_text}
Highest overlap: {max_dim_name}
Lowest overlap (most original): {min_dim_name}

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
                global_originality, aggregated_criteria, aggregated_dimensions,
                red_count, yellow_count, green_count
            )
    
    def _generate_fallback_summary(
        self,
        score: int,
        criteria: CriteriaScores,
        dimensions: Dict[str, float],
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
        
        # Find highest overlap dimension
        if dimensions:
            max_dim = max(dimensions.items(), key=lambda x: x[1])
            max_dim_name = self.DIMENSION_NAMES.get(max_dim[0], max_dim[0])
            
            if max_dim[1] > 0.5:
                overlap_note = f" Main overlap detected in {max_dim_name}."
            else:
                overlap_note = ""
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
                primary_dimension="",
                linked_sections=[],
                evidence=[]
            )
            for i, sent in enumerate(user_sentences)
        ]
        
        return Layer2Result(
            global_originality_score=100,
            global_overlap_score=0.0,
            label=OriginalityLabel.HIGH,
            sentence_annotations=annotations,
            summary="No similar papers were found. Your idea appears to be highly original, though this may indicate a gap in the search rather than true novelty.",
            aggregated_dimensions={},
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
