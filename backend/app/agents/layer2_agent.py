"""
Layer 2: Global originality aggregation.
Uses paper-threat-based scoring (worst-case driven) with categorical guardrails.
Final score is derived from criteria, NOT sentences.
Sentence annotations are kept for UI display only.
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
      1. Compute per-paper threat (worst-case driven within each paper)
      2. Compute global overlap from paper threats (worst-case driven across papers)
      3. Apply categorical guardrails
      4. Convert to 0-100 originality score
      5. Compute sentence annotations (for UI only — does NOT affect final score)
      6. Generate summary via LLM
    """

    def __init__(self):
        self.summary_agent = None
        self.last_token_count = 0

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
        cost_breakdown: CostBreakdown = None
    ) -> Layer2Result:
        if not layer1_results:
            return self._create_empty_result(user_sentences)

        # Step 1: Compute per-paper threat scores
        paper_threats = [
            self._compute_paper_threat(r) for r in layer1_results
        ]
        for r, threat in zip(layer1_results, paper_threats):
            logger.info(
                f"[Layer2] Paper threat — {r.paper_title[:50]}: {threat:.3f} "
                f"(p={r.criteria_scores.problem_similarity:.2f} "
                f"m={r.criteria_scores.method_similarity:.2f} "
                f"d={r.criteria_scores.domain_overlap:.2f} "
                f"c={r.criteria_scores.contribution_similarity:.2f})"
            )

        # Step 2: Global overlap from paper threats (worst-case driven)
        global_overlap = self._compute_global_overlap(paper_threats)
        logger.info(
            f"[Layer2] Global overlap (before guardrails): {global_overlap:.3f} "
            f"(max_threat={max(paper_threats):.3f}, mean_threat={sum(paper_threats)/len(paper_threats):.3f})"
        )

        # Step 3: Categorical guardrails — may override global_overlap
        overlap_before = global_overlap
        global_overlap = self._apply_guardrails(global_overlap, layer1_results)
        if global_overlap != overlap_before:
            logger.info(f"[Layer2] Guardrail applied: overlap {overlap_before:.3f} → {global_overlap:.3f}")

        # Step 4: Convert to 0-100 originality score
        global_originality = self._overlap_to_originality_score(global_overlap)
        global_label = self._score_to_label(global_originality)
        logger.info(
            f"[Layer2] Final score: overlap={global_overlap:.3f} → "
            f"originality={global_originality}/100 → label={global_label.value}"
        )

        # Step 5: Aggregate criteria for UI display
        aggregated_criteria = self._aggregate_criteria(layer1_results)

        # Step 6: Sentence annotations for UI (does NOT affect final score)
        sentence_annotations = self._compute_sentence_annotations(
            layer1_results, user_sentences
        )
        red = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.LOW)
        yellow = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.MEDIUM)
        green = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.HIGH)
        logger.info(f"[Layer2] Sentence annotations: {red} RED, {yellow} YELLOW, {green} GREEN")

        # Step 7: Generate summary
        summary = self._generate_summary(
            global_originality=global_originality,
            aggregated_criteria=aggregated_criteria,
            sentence_annotations=sentence_annotations,
            num_papers=len(layer1_results),
            layer1_results=layer1_results,
            paper_threats=paper_threats,
        )

        # Update cost
        if cost_breakdown:
            cost_breakdown.layer2 = self.get_cost()
            cost_breakdown.total = (
                cost_breakdown.followup
                + cost_breakdown.keywords
                + cost_breakdown.layer1
                + cost_breakdown.layer2
            )

        return Layer2Result(
            global_originality_score=global_originality,
            global_overlap_score=global_overlap,
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
    def _compute_paper_threat(result: Layer1Result) -> float:
        """
        Compute how much a single paper threatens the idea's originality.

        Formula: PAPER_THREAT_MAX_WEIGHT * max(criteria) +
                 (1 - PAPER_THREAT_MAX_WEIGHT) * weighted_mean(criteria)

        Worst-case driven: if a paper has contribution_similarity=1.0 (Likert 5),
        the max term dominates regardless of other criteria being low.
        """
        scores = [
            result.criteria_scores.problem_similarity,
            result.criteria_scores.method_similarity,
            result.criteria_scores.domain_overlap,
            result.criteria_scores.contribution_similarity,
        ]
        w = config.CRITERIA_WEIGHTS
        weighted_mean = (
            w["problem"] * result.criteria_scores.problem_similarity
            + w["method"] * result.criteria_scores.method_similarity
            + w["domain"] * result.criteria_scores.domain_overlap
            + w["contribution"] * result.criteria_scores.contribution_similarity
        )
        max_score = max(scores)
        t = config.PAPER_THREAT_MAX_WEIGHT
        return t * max_score + (1 - t) * weighted_mean

    # ------------------------------------------------------------------
    # Global overlap (across papers)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_global_overlap(paper_threats: List[float]) -> float:
        """
        Global overlap = GLOBAL_THREAT_MAX_WEIGHT * max(threats)
                        + (1 - GLOBAL_THREAT_MAX_WEIGHT) * mean(threats)

        One very threatening paper dominates the final score.
        """
        if not paper_threats:
            return 0.0
        g = config.GLOBAL_THREAT_MAX_WEIGHT
        return g * max(paper_threats) + (1 - g) * (sum(paper_threats) / len(paper_threats))

    # ------------------------------------------------------------------
    # Categorical guardrails
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_guardrails(
        global_overlap: float,
        results: List[Layer1Result],
    ) -> float:
        """
        Hard rules that override or cap the computed overlap:
        - Any criterion = 5 (Likert) on any paper → overlap floor
        - 2+ criteria >= 4 on a single paper → overlap floor
        """
        for r in results:
            raw_scores = [
                r.criteria_scores.problem_similarity,
                r.criteria_scores.method_similarity,
                r.criteria_scores.domain_overlap,
                r.criteria_scores.contribution_similarity,
            ]
            # Any criterion at max (1.0 = Likert 5)
            if any(s >= 1.0 for s in raw_scores):
                floor = config.GUARDRAIL_CRITICAL_FLOOR
                if global_overlap < floor:
                    logger.warning(
                        f"Guardrail: paper '{r.paper_title[:40]}' has a criterion at 5. "
                        f"Raising global overlap from {global_overlap:.3f} to {floor:.3f}"
                    )
                    global_overlap = floor

            # 2+ criteria at >= 0.75 (Likert 4+)
            high_count = sum(1 for s in raw_scores if s >= 0.75)
            if high_count >= config.GUARDRAIL_HIGH_COUNT:
                floor = config.GUARDRAIL_HIGH_FLOOR
                if global_overlap < floor:
                    logger.warning(
                        f"Guardrail: paper '{r.paper_title[:40]}' has {high_count} criteria >= 4. "
                        f"Raising global overlap from {global_overlap:.3f} to {floor:.3f}"
                    )
                    global_overlap = floor

        return min(global_overlap, 1.0)

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
            domain_overlap=_mwa([r.criteria_scores.domain_overlap for r in results]),
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
        Per-sentence overlap for UI highlighting.
        Uses top-K averaging across papers, then threshold-based classification.
        """
        annotations = []
        k = config.SENTENCE_OVERLAP_TOP_K

        for idx, sentence in enumerate(user_sentences):
            overlap_scores = []
            all_matches = []

            for result in results:
                for sa in result.sentence_analyses:
                    if sa.sentence_index == idx:
                        overlap_scores.append(sa.overlap_score)
                        all_matches.extend(sa.matched_sections)
                        break

            if overlap_scores:
                top_k = sorted(overlap_scores, reverse=True)[:k]
                effective_overlap = sum(top_k) / len(top_k)
            else:
                effective_overlap = 0.0

            originality = 1.0 - effective_overlap

            if effective_overlap >= config.HIGH_OVERLAP_THRESHOLD:
                label = OriginalityLabel.LOW
            elif effective_overlap >= config.MEDIUM_OVERLAP_THRESHOLD:
                label = OriginalityLabel.MEDIUM
            else:
                label = OriginalityLabel.HIGH

            all_matches.sort(key=lambda x: x.similarity, reverse=True)

            annotations.append(SentenceAnnotation(
                index=idx,
                sentence=sentence,
                originality_score=originality,
                overlap_score=effective_overlap,
                label=label,
                linked_sections=all_matches[:5],
            ))

        return annotations

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
        paper_threats: List[float],
    ) -> str:
        self._init_summary_agent()

        red = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.LOW)
        yellow = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.MEDIUM)
        green = sum(1 for a in sentence_annotations if a.label == OriginalityLabel.HIGH)

        # Per-paper threat info
        threat_lines = []
        for r, t in zip(layer1_results, paper_threats):
            threat_lines.append(
                f"- {r.paper_title[:60]} — threat: {t:.2f}, originality_threat: {r.originality_threat}"
            )
        threat_block = "\n".join(threat_lines)

        prompt = f"""Generate a brief summary for this originality assessment:

Global Originality Score: {global_originality}/100
Papers Analyzed: {num_papers}

Aggregated Criteria Scores (0-1, higher = more similar to existing work):
- Problem Similarity: {aggregated_criteria.problem_similarity:.2f}
- Method Similarity: {aggregated_criteria.method_similarity:.2f}
- Domain Overlap: {aggregated_criteria.domain_overlap:.2f}
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
                self.last_token_count = response.usage_metadata.total_token_count
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
            'application domain': criteria.domain_overlap,
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
                originality_score=1.0, overlap_score=0.0,
                label=OriginalityLabel.HIGH, linked_sections=[],
            )
            for i, sent in enumerate(user_sentences)
        ]
        return Layer2Result(
            global_originality_score=100,
            global_overlap_score=0.0,
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
        if self.last_token_count > 0:
            inp = self.last_token_count * 0.7
            out = self.last_token_count * 0.3
            return (
                (inp / 1_000_000) * config.INPUT_TOKEN_PRICE
                + (out / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            )
        return 0.0
