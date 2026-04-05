"""
Layer 1 Agent: Per-paper originality analysis.
Uses 4 focused criterion calls + 1 sentence-level analysis call per paper.
Each criterion is evaluated independently with its own Likert rubric,
then sentence analysis is anchored to the computed criterion scores.
"""
import json
import logging
from typing import List, Tuple, Dict

from app.agents.Agent import Agent
from core import config
from . import agent_config
from app.models.paper import Paper
from app.models.analysis import (
    Layer1Result,
    CriteriaScores,
    SentenceAnalysis,
    MatchedSection
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts (minimal — criterion-specific rubrics go in user prompts)
# ---------------------------------------------------------------------------

CRITERION_SYSTEM_PROMPT = (
    "You are an academic originality assessor. You evaluate how similar a "
    "research paper is to a user's research idea on a single criterion. "
    "Be objective, evidence-based, and precise. Return ONLY valid JSON."
)

SENTENCE_SYSTEM_PROMPT = (
    "You are an academic originality assessor performing sentence-level "
    "overlap analysis. You compare each sentence of a user's research idea "
    "against a paper's content. Be objective and evidence-based. "
    "Return ONLY valid JSON."
)

# ---------------------------------------------------------------------------
# Per-criterion Likert rubrics
# ---------------------------------------------------------------------------

CRITERION_RUBRICS: Dict[str, Dict[str, str]] = {
    "problem_similarity": {
        "description": "How similar is the research problem or question?",
        "rubric": (
            "1 = Completely unrelated question. The paper addresses a fundamentally different research gap.\n"
            "2 = Loosely related area but a clearly distinct research question with different goals.\n"
            "3 = Same broad topic, but different specific problem or different angle on the problem.\n"
            "4 = Very similar problem statement, differing only in scope, constraints, or minor framing.\n"
            "5 = Same problem, same framing. The paper asks essentially the same question."
        ),
    },
    "method_similarity": {
        "description": "How similar is the proposed method or approach?",
        "rubric": (
            "1 = Completely different techniques. No methodological overlap whatsoever.\n"
            "2 = Methods share a broad category (e.g., both use deep learning) but differ in architecture, training, and application.\n"
            "3 = Same general method family with meaningful differences in design, components, or pipeline.\n"
            "4 = Very similar methodology; differences are incremental (e.g., a different loss function or minor architectural tweak).\n"
            "5 = Identical methodology with the same implementation approach. Only trivial differences if any."
        ),
    },
    "domain_overlap": {
        "description": "How much do the application domains overlap?",
        "rubric": (
            "1 = Different fields entirely (e.g., NLP vs. robotics).\n"
            "2 = Related disciplines but different application contexts (e.g., both in healthcare but radiology vs. genomics).\n"
            "3 = Same discipline, different sub-area or application target.\n"
            "4 = Same sub-area with closely related application targets.\n"
            "5 = Same specific application area, same data type, same target population or system."
        ),
    },
    "contribution_similarity": {
        "description": "How similar are the claimed contributions?",
        "rubric": (
            "1 = Unrelated contributions addressing different gaps.\n"
            "2 = Contributions in the same general direction but clearly different claims.\n"
            "3 = Partial overlap: some contributions are related, others are distinct.\n"
            "4 = Most contributions overlap, with only minor novel additions in the user's idea.\n"
            "5 = Same claims and results. The paper already demonstrates what the user proposes to contribute."
        ),
    },
}

CRITERION_ORDER = [
    "problem_similarity",
    "method_similarity",
    "domain_overlap",
    "contribution_similarity",
]


class Layer1Agent:
    """
    Layer 1: Per-paper originality analysis.

    For each paper runs 5 stateless text-generation calls:
      1-4. One per criterion  (problem / method / domain / contribution)
      5.   Sentence-level analysis anchored to the four criterion scores
    """

    def __init__(self):
        self._criterion_agent = Agent(
            system_prompt=CRITERION_SYSTEM_PROMPT,
            temperature=agent_config.LAYER1_TEMPERATURE,
            top_p=agent_config.LAYER1_TOP_P,
            top_k=agent_config.LAYER1_TOP_K,
            response_mime_type="application/json",
            create_chat=False,
        )
        self._sentence_agent = Agent(
            system_prompt=SENTENCE_SYSTEM_PROMPT,
            temperature=agent_config.LAYER1_TEMPERATURE,
            top_p=agent_config.LAYER1_TOP_P,
            top_k=agent_config.LAYER1_TOP_K,
            response_mime_type="application/json",
            create_chat=False,
        )
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0



    def analyze_paper(
        self,
        user_idea: str,
        user_sentences: List[str],
        paper: Paper,
        paper_context: str = "",
    ) -> Layer1Result:
        """
        Full analysis of one paper: 4 criterion calls + 1 sentence call.
        """
        paper_text = self._format_paper(paper, paper_context)
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        logger.info(f"[Layer1] Starting analysis for paper: {paper.paper_id} — \"{paper.title[:60]}\"")

        # Phase 1 — score each criterion independently
        logger.info(f"[Layer1] Phase 1: Scoring 4 criteria for {paper.paper_id}")
        criteria_results: Dict[str, Dict] = {}
        for criterion_name in CRITERION_ORDER:
            score, justification, tokens, inp, out = self._score_criterion(
                criterion_name, user_idea, paper_text, paper
            )
            criteria_results[criterion_name] = {
                "score": score,
                "justification": justification,
            }
            self.total_tokens += tokens
            self.total_input_tokens += inp
            self.total_output_tokens += out

        # Build CriteriaScores (0-1 floats)
        criteria = CriteriaScores(
            problem_similarity=self._likert_to_float(criteria_results["problem_similarity"]["score"]),
            method_similarity=self._likert_to_float(criteria_results["method_similarity"]["score"]),
            domain_overlap=self._likert_to_float(criteria_results["domain_overlap"]["score"]),
            contribution_similarity=self._likert_to_float(criteria_results["contribution_similarity"]["score"]),
        )

        # Overall overlap (weighted average of criteria)
        w = config.CRITERIA_WEIGHTS
        overall_overlap = (
            w["problem"] * criteria.problem_similarity
            + w["method"] * criteria.method_similarity
            + w["domain"] * criteria.domain_overlap
            + w["contribution"] * criteria.contribution_similarity
        )

        logger.info(
            f"[Layer1] Phase 1 complete for {paper.paper_id}: "
            f"p={criteria_results['problem_similarity']['score']}/5, "
            f"m={criteria_results['method_similarity']['score']}/5, "
            f"d={criteria_results['domain_overlap']['score']}/5, "
            f"c={criteria_results['contribution_similarity']['score']}/5 "
            f"→ overall_overlap={overall_overlap:.2f}"
        )

        # Derive confidence / similarity_level from raw Likert scores
        high_count = sum(
            1 for v in criteria_results.values() if v["score"] >= 4
        )
        if high_count >= 3:
            similarity_level = "high"
            confidence = "high"
        elif high_count >= 1:
            similarity_level = "moderate"
            confidence = "medium"
        else:
            similarity_level = "low"
            confidence = "medium"

        # Phase 2 — sentence-level analysis (independent, no anchoring)
        logger.info(f"[Layer1] Phase 2: Sentence analysis for {paper.paper_id} ({len(user_sentences)} sentences)")
        sentence_analyses, sent_tokens, sent_inp, sent_out = self._analyze_sentences(
            user_idea, user_sentences, paper_text, paper
        )
        self.total_tokens += sent_tokens
        self.total_input_tokens += sent_inp
        self.total_output_tokens += sent_out

        # Phase 3 — filter sentence criteria against paper-level scores
        sentence_analyses = self._filter_sentence_criteria(sentence_analyses, criteria_results)

        for sa in sentence_analyses:
            sc = sa.sentence_criteria_scores
            criteria_str = (
                f"p={self._float_to_likert(sc.problem_similarity)} "
                f"m={self._float_to_likert(sc.method_similarity)} "
                f"d={self._float_to_likert(sc.domain_overlap)} "
                f"c={self._float_to_likert(sc.contribution_similarity)}"
            ) if sc else "criteria=None"
            logger.info(
                f"[Layer1]   sentence[{sa.sentence_index}] similarity={sa.similarity_score:.2f} "
                f"{criteria_str} matches={len(sa.matched_sections)} "
                f"— \"{sa.sentence[:60]}...\""
            )

        # Build reason string from per-criterion justifications
        reason_parts = []
        criterion_labels = {
            "problem_similarity": "Problem",
            "method_similarity": "Method",
            "domain_overlap": "Domain",
            "contribution_similarity": "Contribution",
        }
        for key, label in criterion_labels.items():
            j = criteria_results[key].get("justification", "").strip()
            if j:
                reason_parts.append(f"{label}: {j}")
        reason = " | ".join(reason_parts)

        result = Layer1Result(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            arxiv_id=paper.source_id,  # Use source_id instead of arxiv_id
            paper_similarity_score=overall_overlap,
            reason=reason,
            criteria_scores=criteria,
            sentence_analyses=sentence_analyses,
            confidence=confidence,
            similarity_level=similarity_level,
            tokens_used=self.total_tokens,
        )

        logger.info(
            f"[Layer1] Result JSON for {paper.paper_id}:\n"
            f"{json.dumps(result.to_dict(), indent=2, ensure_ascii=False)}"
        )

        return result

    def get_cost(self) -> float:
        """Cost estimate for the last analyze_paper call (all 5 sub-calls)."""
        if self.total_input_tokens > 0 or self.total_output_tokens > 0:
            return (
                (self.total_input_tokens / 1_000_000) * config.INPUT_TOKEN_PRICE
                + (self.total_output_tokens / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            )
        if self.total_tokens > 0:
            inp = self.total_tokens * 0.8
            out = self.total_tokens * 0.2
            return (
                (inp / 1_000_000) * config.INPUT_TOKEN_PRICE
                + (out / 1_000_000) * config.OUTPUT_TOKEN_PRICE
            )
        return 0.0

    # ------------------------------------------------------------------
    # Internals — paper formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_paper(paper: Paper, paper_context: str) -> str:
        sections_text = ""
        for heading in paper.headings:
            if heading.section_text and heading.is_valid:
                sections_text += f"\n### {heading.text}\n{heading.section_text[:3000]}\n"

        return (
            f"Title: {paper.title}\n"
            f"Source: {paper.source}\n"
            f"Source ID: {paper.source_id}\n"
            f"Categories: {', '.join(paper.categories)}\n\n"
            f"### ABSTRACT\n{paper.abstract}\n\n"
            f"### EXTRACTED SECTIONS\n"
            f"{sections_text if sections_text else paper_context if paper_context else 'No sections extracted'}"
        )

    # ------------------------------------------------------------------
    # Internals — criterion scoring (1 call per criterion)
    # ------------------------------------------------------------------

    def _score_criterion(
        self,
        criterion_name: str,
        user_idea: str,
        paper_text: str,
        paper: Paper,
    ) -> Tuple[int, str, int, int, int]:
        """Score a single criterion. Returns (likert_score, justification, total_tokens, input_tokens, output_tokens)."""
        rubric_info = CRITERION_RUBRICS[criterion_name]

        prompt = f"""## USER'S RESEARCH IDEA
{user_idea}

## PAPER TO EVALUATE
{paper_text}

## CRITERION: {criterion_name}
{rubric_info["description"]}

### Scoring Rubric (1-5 Likert, higher = MORE similar to existing work)
{rubric_info["rubric"]}

## CONSTRAINTS
- Be objective and evidence-based. A score above 1 must be justified by specific paper content.
- DO NOT hallucinate paper content — only reference text that appears above.
- If paper content is missing or too short to evaluate this criterion, default to 1.
- When uncertain between two adjacent scores, choose the lower one.
- A score of 5 requires near-verbatim or structurally identical overlap.

## OUTPUT FORMAT
Return ONLY valid JSON:
{{"score": <integer 1-5>, "justification": "<2-3 sentences explaining why this score, citing specific paper content>"}}"""

        try:
            response = self._criterion_agent.generate_text_generation_response(prompt)
            um = getattr(response, "usage_metadata", None)
            tokens = getattr(um, "total_token_count", 0) or 0
            inp = getattr(um, "prompt_token_count", 0) or 0
            out = getattr(um, "candidates_token_count", 0) or 0
            result = json.loads(response.text)
            score = max(1, min(5, int(result.get("score", 1))))
            justification = result.get("justification", "")
            logger.info(f"  {criterion_name}: {score}/5 — {justification[:100]}")
            return score, justification, tokens, inp, out
        except Exception as e:
            logger.error(f"Failed to score {criterion_name} for {paper.paper_id}: {e}")
            return 1, f"Error: {e}", 0, 0, 0

    # ------------------------------------------------------------------
    # Internals — sentence-level analysis (5th call, anchored to criteria)
    # ------------------------------------------------------------------

    def _analyze_sentences(
        self,
        user_idea: str,
        user_sentences: List[str],
        paper_text: str,
        paper: Paper,
    ) -> Tuple[List[SentenceAnalysis], int, int, int]:
        """Independent sentence-level analysis. Returns (analyses, total_tokens, input_tokens, output_tokens)."""
        sentences_text = "\n".join(
            f"[{i}] {sent}" for i, sent in enumerate(user_sentences)
        )

        prompt = f"""## USER'S RESEARCH IDEA
{user_idea}

## USER'S IDEA SENTENCES (analyze every one)
{sentences_text}

## PAPER
{paper_text}

## SCORING RUBRICS
For each sentence, score it on all four criteria using these rubrics (1-5 Likert, higher = MORE overlap):

**problem_score** — {CRITERION_RUBRICS["problem_similarity"]["description"]}
{CRITERION_RUBRICS["problem_similarity"]["rubric"]}

**method_score** — {CRITERION_RUBRICS["method_similarity"]["description"]}
{CRITERION_RUBRICS["method_similarity"]["rubric"]}

**domain_score** — {CRITERION_RUBRICS["domain_overlap"]["description"]}
{CRITERION_RUBRICS["domain_overlap"]["rubric"]}

**contribution_score** — {CRITERION_RUBRICS["contribution_similarity"]["description"]}
{CRITERION_RUBRICS["contribution_similarity"]["rubric"]}

## TASK
For EACH sentence in the user's idea above, independently evaluate its overlap with the paper on all four criteria.

For each sentence provide:
- **problem_score**, **method_score**, **domain_score**, **contribution_score** (integers 1-5, using rubrics above)
- **matched_sections**: List of overlapping passages from the paper. For each match include:
  - criterion: One of "problem_similarity", "method_similarity", "domain_overlap", "contribution_similarity"
  - heading: Paper section heading where overlap was found
  - similar_text: Quote or closely paraphrase the SPECIFIC passage (1-3 sentences). NEVER leave empty.
  - reason: Explain WHAT is similar
  - similarity: Integer 1-5

## CONSTRAINTS
- DO NOT hallucinate paper content — only reference text that appears above.
- A sentence may score high on one criterion and low on others.

## OUTPUT FORMAT
Return ONLY valid JSON:
{{"sentence_level": [
    {{"sentence_index": 0,
      "problem_score": 1, "method_score": 4, "domain_score": 2, "contribution_score": 1,
      "matched_sections": [
        {{"criterion": "method_similarity", "heading": "...", "similar_text": "...", "reason": "...", "similarity": 4}}
      ]
    }}
]}}"""

        try:
            response = self._sentence_agent.generate_text_generation_response(prompt)
            um = getattr(response, "usage_metadata", None)
            tokens = getattr(um, "total_token_count", 0) or 0
            inp = getattr(um, "prompt_token_count", 0) or 0
            out = getattr(um, "candidates_token_count", 0) or 0
            result = json.loads(response.text)
            analyses = self._parse_sentence_results(
                result, paper, user_sentences
            )
            return analyses, tokens, inp, out
        except Exception as e:
            logger.error(f"Sentence analysis failed for {paper.paper_id}: {e}")
            fallback = [
                SentenceAnalysis(
                    sentence=s, sentence_index=i,
                    similarity_score=0.0, matched_sections=[],
                )
                for i, s in enumerate(user_sentences)
            ]
            return fallback, 0, 0, 0

    # ------------------------------------------------------------------
    # Internals — parsing helpers
    # ------------------------------------------------------------------

    def _parse_sentence_results(
        self,
        result: dict,
        paper: Paper,
        user_sentences: List[str],
    ) -> List[SentenceAnalysis]:
        analyses: List[SentenceAnalysis] = []

        for sent_data in result.get("sentence_level", []):
            idx = sent_data.get("sentence_index", 0)

            matched = [
                MatchedSection(
                    chunk_id="",
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    heading=m.get("heading", ""),
                    text_snippet=m.get("similar_text", ""),
                    similarity=self._likert_to_float(m.get("similarity", 1)),
                    reason=m.get("reason", ""),
                    criterion=m.get("criterion", ""),
                )
                for m in sent_data.get("matched_sections", [])
            ]

            sentence = sent_data.get("sentence", "")
            if not sentence and idx < len(user_sentences):
                sentence = user_sentences[idx]

            # Build per-criterion scores for this sentence
            p = sent_data.get("problem_score", 1)
            m = sent_data.get("method_score", 1)
            d = sent_data.get("domain_score", 1)
            c = sent_data.get("contribution_score", 1)
            sentence_criteria = CriteriaScores(
                problem_similarity=self._likert_to_float(p),
                method_similarity=self._likert_to_float(m),
                domain_overlap=self._likert_to_float(d),
                contribution_similarity=self._likert_to_float(c),
            )

            # Compute overall overlap as weighted avg of sentence criteria
            w = config.CRITERIA_WEIGHTS
            overlap = (
                w["problem"] * sentence_criteria.problem_similarity
                + w["method"] * sentence_criteria.method_similarity
                + w["domain"] * sentence_criteria.domain_overlap
                + w["contribution"] * sentence_criteria.contribution_similarity
            )

            analyses.append(
                SentenceAnalysis(
                    sentence=sentence,
                    sentence_index=idx,
                    similarity_score=overlap,
                    matched_sections=matched,
                    sentence_criteria_scores=sentence_criteria,
                )
            )

        # Fill any missing sentences
        analyzed = {sa.sentence_index for sa in analyses}
        for i, sent in enumerate(user_sentences):
            if i not in analyzed:
                analyses.append(
                    SentenceAnalysis(
                        sentence=sent, sentence_index=i,
                        similarity_score=0.0, matched_sections=[],
                    )
                )

        analyses.sort(key=lambda x: x.sentence_index)
        return analyses

    def _filter_sentence_criteria(
        self,
        sentence_analyses: List[SentenceAnalysis],
        criteria_results: Dict[str, Dict],
    ) -> List[SentenceAnalysis]:
        """
        Filter sentence criteria against paper-level scores.
        If sentence criterion score < paper-level score - 1 (Likert tolerance),
        zero out that criterion so Layer2 ignores it.
        """
        paper_likert = {
            "problem_similarity": criteria_results["problem_similarity"]["score"],
            "method_similarity": criteria_results["method_similarity"]["score"],
            "domain_overlap": criteria_results["domain_overlap"]["score"],
            "contribution_similarity": criteria_results["contribution_similarity"]["score"],
        }

        for sa in sentence_analyses:
            if sa.sentence_criteria_scores is None:
                continue

            sc = sa.sentence_criteria_scores
            # Convert sentence 0-1 floats back to Likert for comparison
            sent_likert = {
                "problem_similarity": self._float_to_likert(sc.problem_similarity),
                "method_similarity": self._float_to_likert(sc.method_similarity),
                "domain_overlap": self._float_to_likert(sc.domain_overlap),
                "contribution_similarity": self._float_to_likert(sc.contribution_similarity),
            }

            passing = {
                "problem_similarity": sent_likert["problem_similarity"] >= paper_likert["problem_similarity"] - 1,
                "method_similarity": sent_likert["method_similarity"] >= paper_likert["method_similarity"] - 1,
                "domain_overlap": sent_likert["domain_overlap"] >= paper_likert["domain_overlap"] - 1,
                "contribution_similarity": sent_likert["contribution_similarity"] >= paper_likert["contribution_similarity"] - 1,
            }

            filtered = CriteriaScores(
                problem_similarity=sc.problem_similarity if passing["problem_similarity"] else 0.0,
                method_similarity=sc.method_similarity if passing["method_similarity"] else 0.0,
                domain_overlap=sc.domain_overlap if passing["domain_overlap"] else 0.0,
                contribution_similarity=sc.contribution_similarity if passing["contribution_similarity"] else 0.0,
            )
            sa.sentence_criteria_scores = filtered

            # Remove matched_sections for criteria that were zeroed out
            sa.matched_sections = [
                ms for ms in sa.matched_sections
                if passing.get(ms.criterion, False)
            ]

        return sentence_analyses

    @staticmethod
    def _float_to_likert(value: float) -> int:
        """Convert 0-1 float back to Likert integer (1-5)."""
        reverse = {0.0: 1, 0.25: 2, 0.5: 3, 0.75: 4, 1.0: 5}
        rounded = round(value * 4) / 4  # snap to nearest 0.25
        return reverse.get(rounded, 1)

    @staticmethod
    def _likert_to_float(value) -> float:
        """Convert Likert integer (1-5) or legacy float (0-1) to a 0-1 float."""
        v = float(value)
        int_v = int(v)
        if int_v == v and int_v in config.LIKERT_TO_FLOAT:
            return config.LIKERT_TO_FLOAT[int_v]
        if 0.0 <= v <= 1.0:
            return v
        clamped = max(1, min(5, int(round(v))))
        return config.LIKERT_TO_FLOAT[clamped]

    def _create_error_result(self, paper: Paper, error: str) -> Layer1Result:
        """Create a result object for a completely failed analysis."""
        return Layer1Result(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            arxiv_id=paper.source_id,  # Use source_id instead of arxiv_id
            paper_similarity_score=0.0,
            criteria_scores=CriteriaScores(0.0, 0.0, 0.0, 0.0),
            sentence_analyses=[],
        )
