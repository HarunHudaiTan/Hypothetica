"""
Layer 2: Global originality aggregation.
Per-paper weighted criteria, then max across papers; mapped to 0–100 originality.
Final score is derived from criteria, not sentences. Sentence annotations are UI-only.
"""
import logging
from typing import List, Dict, Optional

from app.agents.Agent import Agent
from core import config
from . import agent_config
from app.models.analysis import (
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
- Per-paper threat levels and criteria scores
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
    Layer 2: Aggregates Layer 1 results into final originality assessment.

    Scoring pipeline:
      1. Per-paper similarity = weighted mean of four criteria (config.CRITERIA_WEIGHTS)
      2. Global similarity = max over papers (a single close match is enough to cap originality)
      3. Convert to 0-100 originality via (1 - global ** power) (config.OVERLAP_CURVE_POWER)
      4. Compute sentence annotations (for UI only — does NOT affect final score)
      5. Generate summary via LLM
    """

    def __init__(self):
        self.summary_agent = None
        self.last_token_count = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def _init_summary_agent(self):
        if self.summary_agent is None:
            self.summary_agent = Agent(
                system_prompt=LAYER2_SUMMARY_PROMPT,
                temperature=agent_config.LAYER2_TEMPERATURE,
                top_p=agent_config.LAYER2_TOP_P,
                top_k=agent_config.LAYER2_TOP_K,
                response_mime_type='text/plain',
                create_chat=False
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        layer1_results: List[Layer1Result],
        user_sentences: List[str],
        cost_breakdown: CostBreakdown = None,
        benchmark_mode: bool = False,
    ) -> Layer2Result:
        if not layer1_results:
            return self._create_empty_result(user_sentences)

        # Step 1: Compute per-paper similarity scores
        paper_similarity_score_list = [
            self._compute_paper_similarity(r) for r in layer1_results
        ]
        for r, paper_similarity in zip(layer1_results, paper_similarity_score_list):
            logger.info(
                f"[Layer2] Paper similarity — {r.paper_title[:50]}: {paper_similarity:.3f} "
                f"(p={r.criteria_scores.problem_similarity:.2f} "
                f"m={r.criteria_scores.method_similarity:.2f} "
                f"d={r.criteria_scores.domain_similarity:.2f} "
                f"c={r.criteria_scores.contribution_similarity:.2f})"
            )

        # Step 2: Global similarity = max per-paper (one strong match cannot be averaged away)
        global_similarity = self._compute_global_similarity(paper_similarity_score_list)
        mean_s = sum(paper_similarity_score_list) / len(paper_similarity_score_list)
        logger.info(
            f"[Layer2] Global similarity: {global_similarity:.3f} "
            f"(max per paper={max(paper_similarity_score_list):.3f}, mean for ref={mean_s:.3f})"
        )

        # Step 3: Convert to 0-100 originality score
        global_originality = self._overlap_to_originality_score(global_similarity)
        global_label = self._score_to_label(global_originality)
        logger.info(
            f"[Layer2] Final score: similarity={global_similarity:.3f} → "
            f"originality={global_originality}/100 → label={global_label.value}"
        )

        # Step 4: Aggregate criteria for UI display
        aggregated_criteria = self._aggregate_criteria(layer1_results)

        # Step 5: Sentence annotations for UI (skipped in benchmark mode)
        if benchmark_mode:
            logger.info("[Layer2] Skipping sentence annotations and summary (benchmark_mode)")
            sentence_annotations = []
            summary = ""
        else:
            sentence_annotations = self._compute_sentence_annotations(
                layer1_results, user_sentences
            )
            red = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.LOW)
            yellow = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.MEDIUM)
            green = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.HIGH)
            logger.info(f"[Layer2] Sentence annotations: {red} RED, {yellow} YELLOW, {green} GREEN")

            # Step 6: Generate summary
            summary = self._generate_summary(
                global_originality=global_originality,
                aggregated_criteria=aggregated_criteria,
                sentence_annotations=sentence_annotations,
                num_papers=len(layer1_results),
                layer1_results=layer1_results,
                paper_similarity_score_list=paper_similarity_score_list,
            )

        # Update cost
        if cost_breakdown:
            cost_breakdown.layer2 = self.get_cost()
            cost_breakdown.total = (
                cost_breakdown.followup
                + cost_breakdown.keywords
                + cost_breakdown.layer1
                + cost_breakdown.layer2
                + cost_breakdown.reality_check
                + cost_breakdown.github
            )

        return Layer2Result(
            originality_score=global_originality,
            global_similarity_score=global_similarity,
            label=global_label,
            sentence_annotations=sentence_annotations,
            summary=summary,
            aggregated_criteria=aggregated_criteria,
            papers_analyzed=len(layer1_results),
            cost=cost_breakdown or CostBreakdown(),
        )

    # ------------------------------------------------------------------
    # Paper threat computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_paper_similarity(result: Layer1Result) -> float:
        """
        Compute how similar a single paper is to the idea.

        Formula: weighted mean of the 4 criteria, using config.CRITERIA_WEIGHTS.

            paper_sim = w_p * problem + w_m * method + w_d * domain + w_c * contribution

        Recalibrated 2026-04-19: previously also added a max-term
        (PAPER_SIMILARITY_MAX_WEIGHT * max(criteria)), but a sweep over 125k
        configs showed the max-term inflated single-criterion noise (especially
        domain, which is near-constant due to retrieval bias) without improving
        macro-F1. The plain weighted mean is both simpler and more accurate.
        """
        cs = result.criteria_scores
        w = config.CRITERIA_WEIGHTS
        return (
            w["problem"] * cs.problem_similarity
            + w["method"] * cs.method_similarity
            + w["domain"] * cs.domain_similarity
            + w["contribution"] * cs.contribution_similarity
        )

    # ------------------------------------------------------------------
    # Global overlap (across papers)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_global_similarity(paper_similarity_score_list: List[float]) -> float:
        """
        Global similarity = GLOBAL_MAX_WEIGHT * max + GLOBAL_MEAN_WEIGHT * mean
        of per-paper similarity scores.

        Pure max was too sensitive to a single noisy outlier paper inflating the result.
        The blend preserves the signal that one strong match is enough to flag overlap,
        while dampening the effect of one overscored irrelevant paper.
        """
        if not paper_similarity_score_list:
            return 0.0
        best = max(paper_similarity_score_list)
        mean = sum(paper_similarity_score_list) / len(paper_similarity_score_list)
        return config.GLOBAL_MAX_WEIGHT * best + config.GLOBAL_MEAN_WEIGHT * mean

    # ------------------------------------------------------------------
    # Categorical guardrails
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Criteria aggregation (for UI display)
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_criteria(results: List[Layer1Result]) -> CriteriaScores:
        """Max-weighted average per criterion across papers (for UI display)."""
        if not results:
            return CriteriaScores(0.0, 0.0, 0.0, 0.0)

        def _mwa(scores: List[float]) -> float:
            if not scores:
                return 0.0
            w = config.CRITERIA_MAX_WEIGHT
            return w * max(scores) + (1 - w) * (sum(scores) / len(scores))

        return CriteriaScores(
            problem_similarity=_mwa([r.criteria_scores.problem_similarity for r in results]),
            method_similarity=_mwa([r.criteria_scores.method_similarity for r in results]),
            domain_similarity=_mwa([r.criteria_scores.domain_similarity for r in results]),
            contribution_similarity=_mwa([r.criteria_scores.contribution_similarity for r in results]),
        )

    # ------------------------------------------------------------------
    # Sentence annotations (UI only — does NOT affect final score)
    # ------------------------------------------------------------------

    def _compute_sentence_annotations(
        self,
        results: List[Layer1Result],
        user_sentences: List[str]
    ) -> List[SentenceAnnotation]:
        """
        Per-sentence criterion-based annotations for UI highlighting.
        For each sentence and each criterion, takes max score across all papers.
        Only criteria that passed the Layer1 filter (score > 0) are considered.
        """
        annotations = []

        CRITERIA_KEYS = [
            "problem_similarity",
            "method_similarity",
            "domain_similarity",
            "contribution_similarity",
        ]

        for idx, sentence in enumerate(user_sentences):
            # Collect per-criterion max scores and best matched_sections across papers
            criterion_max: Dict[str, float] = {k: 0.0 for k in CRITERIA_KEYS}
            criterion_best_match: Dict[str, MatchedSection] = {}

            for result in results:
                for sa in result.sentence_analyses:
                    if sa.sentence_index != idx:
                        continue
                    if sa.sentence_criteria_scores is None:
                        continue

                    sc = sa.sentence_criteria_scores
                    scores_map = {
                        "problem_similarity": sc.problem_similarity,
                        "method_similarity": sc.method_similarity,
                        "domain_similarity": sc.domain_similarity,
                        "contribution_similarity": sc.contribution_similarity,
                    }

                    for criterion_key, score in scores_map.items():
                        if score <= 0.0:
                            continue  # filtered out in Layer1
                        if score > criterion_max[criterion_key]:
                            criterion_max[criterion_key] = score
                            # Find the best matched_section for this criterion
                            best = self._best_match_for_criterion(sa.matched_sections, criterion_key)
                            if best:
                                criterion_best_match[criterion_key] = best
                    break

            # Build criteria_labels — only criteria that pass the display threshold
            criteria_labels: Dict[str, OriginalityLabel] = {}
            linked_sections: List[MatchedSection] = []

            for criterion_key, max_score in criterion_max.items():
                likert = self._float_to_likert(max_score)
                if likert >= config.SENTENCE_CRITERION_RED_MIN:
                    criteria_labels[criterion_key] = OriginalityLabel.LOW
                    if criterion_key in criterion_best_match:
                        linked_sections.append(criterion_best_match[criterion_key])
                elif likert >= config.SENTENCE_CRITERION_YELLOW_MIN:
                    criteria_labels[criterion_key] = OriginalityLabel.MEDIUM
                    if criterion_key in criterion_best_match:
                        linked_sections.append(criterion_best_match[criterion_key])
                else:
                    criteria_labels[criterion_key] = OriginalityLabel.HIGH  # green — no overlap

            # Overall label: worst across passing criteria
            if any(v == OriginalityLabel.LOW for v in criteria_labels.values()):
                label = OriginalityLabel.LOW
            elif any(v == OriginalityLabel.MEDIUM for v in criteria_labels.values()):
                label = OriginalityLabel.MEDIUM
            else:
                label = OriginalityLabel.HIGH

            logger.info(
                f"[Layer2] sentence[{idx}] criteria_max={{{', '.join(f'{k}={v:.2f}(L{self._float_to_likert(v)})' for k, v in criterion_max.items())}}} "
                f"criteria_labels={{{', '.join(f'{k}={v.value}' for k, v in criteria_labels.items())}}} "
                f"label={label.value} — \"{sentence[:60]}\""
            )

            # Compute effective overlap for backward compat fields
            passing_scores = [s for s in criterion_max.values() if s > 0.0]
            effective_overlap = max(passing_scores) if passing_scores else 0.0
            originality = 1.0 - effective_overlap

            annotations.append(SentenceAnnotation(
                index=idx,
                sentence=sentence,
                originality_score=originality,
                similarity_score=effective_overlap,
                label=label,
                linked_sections=linked_sections,
                criteria_labels=criteria_labels,
            ))

        return annotations

    @staticmethod
    def _best_match_for_criterion(
        matched_sections: List[MatchedSection],
        criterion_key: str,
    ) -> Optional[MatchedSection]:
        """Return the highest-similarity matched_section for a given criterion."""
        candidates = [m for m in matched_sections if m.criterion == criterion_key]
        if not candidates:
            # Fallback: return highest similarity match regardless of criterion
            candidates = matched_sections
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.similarity)

    @staticmethod
    def _float_to_likert(value: float) -> int:
        """Convert 0-1 float to Likert integer (1-5)."""
        reverse = {0.0: 1, 0.25: 2, 0.5: 3, 0.75: 4, 1.0: 5}
        rounded = round(value * 4) / 4
        return reverse.get(rounded, 1)

    # ------------------------------------------------------------------
    # Score conversions
    # ------------------------------------------------------------------

    @staticmethod
    def _overlap_to_originality_score(overlap: float) -> int:
        """Convert overlap (0-1) → originality (0-100) via power curve."""
        power = config.OVERLAP_CURVE_POWER
        originality = (1.0 - overlap ** power) * 100
        return int(max(0, min(100, originality)))

    @staticmethod
    def _score_to_label(originality_score: int) -> OriginalityLabel:
        if originality_score >= config.SCORE_YELLOW_MAX:
            return OriginalityLabel.HIGH
        elif originality_score >= config.SCORE_RED_MAX:
            return OriginalityLabel.MEDIUM
        else:
            return OriginalityLabel.LOW

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def _generate_summary(
        self,
        global_originality: int,
        aggregated_criteria: CriteriaScores,
        sentence_annotations: List[SentenceAnnotation],
        num_papers: int,
        layer1_results: List[Layer1Result],
        paper_similarity_score_list: List[float],
    ) -> str:
        self._init_summary_agent()

        red = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.LOW)
        yellow = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.MEDIUM)
        green = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.HIGH)

        # Per-paper similarity info
        threat_lines = []
        for r, t in zip(layer1_results, paper_similarity_score_list):
            threat_lines.append(
                f"- {r.paper_title[:60]} — similarity: {t:.2f}, similarity_level: {r.similarity_level}"
            )
        threat_block = "\n".join(threat_lines)

        prompt = f"""Generate a brief summary for this originality assessment:

Global Originality Score: {global_originality}/100
Papers Analyzed: {num_papers}

Aggregated Criteria Scores (0-1, higher = more similar to existing work):
- Problem Similarity: {aggregated_criteria.problem_similarity:.2f}
- Method Similarity: {aggregated_criteria.method_similarity:.2f}
- Domain Overlap: {aggregated_criteria.domain_similarity:.2f}
- Contribution Similarity: {aggregated_criteria.contribution_similarity:.2f}

Per-Paper Threat Assessment:
{threat_block}

Sentence Labels:
- Red (low originality): {red} sentences
- Yellow (medium): {yellow} sentences
- Green (high originality): {green} sentences

Write a 1-2 sentence summary explaining the assessment and giving actionable insight."""

        try:
            response = self.summary_agent.generate_text_generation_response(prompt)
            if hasattr(response, 'usage_metadata'):
                um = response.usage_metadata
                self.last_token_count = getattr(um, 'total_token_count', 0) or 0
                self.last_input_tokens = getattr(um, 'prompt_token_count', 0) or 0
                self.last_output_tokens = getattr(um, 'candidates_token_count', 0) or 0
            return response.text.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._generate_fallback_summary(
                global_originality, aggregated_criteria, red, yellow, green
            )

    @staticmethod
    def _generate_fallback_summary(
        score: int,
        criteria: CriteriaScores,
        red: int,
        yellow: int,
        green: int
    ) -> str:
        if score >= 70:
            level = "high"
        elif score >= 40:
            level = "moderate"
        else:
            level = "low"

        criteria_names = {
            'problem definition': criteria.problem_similarity,
            'methodology': criteria.method_similarity,
            'application domain': criteria.domain_similarity,
            'contributions': criteria.contribution_similarity,
        }
        max_criterion = max(criteria_names.items(), key=lambda x: x[1])

        overlap_note = ""
        if max_criterion[1] > 0.5:
            overlap_note = f" Main overlap detected in {max_criterion[0]}."

        return (
            f"Your idea shows {level} originality (score: {score}/100)."
            f"{overlap_note} {red} sentences have significant overlap, "
            f"{yellow} have moderate overlap, and {green} appear novel."
        )

    # ------------------------------------------------------------------
    # Empty / error results
    # ------------------------------------------------------------------

    @staticmethod
    def _create_empty_result(user_sentences: List[str]) -> Layer2Result:
        annotations = [
            SentenceAnnotation(
                index=i, sentence=sent,
                originality_score=1.0, similarity_score=0.0,
                label=OriginalityLabel.HIGH, linked_sections=[],
            )
            for i, sent in enumerate(user_sentences)
        ]
        return Layer2Result(
            originality_score=100,
            global_similarity_score=0.0,
            label=OriginalityLabel.HIGH,
            sentence_annotations=annotations,
            summary=(
                "No similar papers were found. Your idea appears to be highly "
                "original, though this may indicate a gap in the search rather "
                "than true novelty."
            ),
            papers_analyzed=0,
            cost=CostBreakdown(),
        )

    def get_cost(self) -> float:
        if self.last_input_tokens > 0 or self.last_output_tokens > 0:
            return (
                (self.last_input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
                + (self.last_output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            )
        if self.last_token_count > 0:
            inp = self.last_token_count * 0.7
            out = self.last_token_count * 0.3
            return (
                (inp / 1_000_000) * config.INPUT_TOKEN_PRICE
                + (out / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            )
        return 0.0
