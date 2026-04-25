"""
Test script for Layer 2 Aggregator (paper-threat-based scoring).

Uses mock Layer1Results — NO API calls needed for scoring.
(Summary generation does call the API, but falls back to template if unavailable.)

Usage:
    cd backend
    python -m tests.test_layer2
"""
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from app.agents.layer2_agent import Layer2Aggregator
from app.models.analysis import (
    Layer1Result,
    CriteriaScores,
    SentenceAnalysis,
    MatchedSection,
    OriginalityLabel,
)
from core import config


def _likert_to_float(v: int) -> float:
    return config.LIKERT_TO_FLOAT[v]


def make_l1_result(
    paper_id: str,
    title: str,
    problem: int,
    method: int,
    domain: int,
    contribution: int,
    sentence_overlaps: list[int] = None,
) -> Layer1Result:
    """Helper to create a Layer1Result from Likert integers."""
    cs = CriteriaScores(
        problem_similarity=_likert_to_float(problem),
        method_similarity=_likert_to_float(method),
        domain_similarity=_likert_to_float(domain),
        contribution_similarity=_likert_to_float(contribution),
    )
    w = config.CRITERIA_WEIGHTS
    overall = (
        w["problem"] * cs.problem_similarity
        + w["method"] * cs.method_similarity
        + w["domain"] * cs.domain_similarity
        + w["contribution"] * cs.contribution_similarity
    )

    # Derive threat
    high_count = sum(1 for s in [problem, method, domain, contribution] if s >= 4)
    if high_count >= 3:
        threat = "high"
    elif high_count >= 1:
        threat = "moderate"
    else:
        threat = "low"

    # Simple sentence analyses
    overlaps = sentence_overlaps or [1, 1, 1]
    sentence_analyses = [
        SentenceAnalysis(
            sentence=f"Sentence {i}",
            sentence_index=i,
            similarity_score=_likert_to_float(s),
            matched_sections=[],
        )
        for i, s in enumerate(overlaps)
    ]

    return Layer1Result(
        paper_id=paper_id,
        paper_title=title,
        arxiv_id=f"2401.{paper_id}",
        idea_similarity_score=overall,
        criteria_scores=cs,
        sentence_analyses=sentence_analyses,
        confidence="medium",
        similarity_level=threat,
    )


def print_result(label, result):
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {label}")
    print(f"{'=' * 60}")
    print(f"Originality score:        {result.originality_score}/100")
    print(f"Global similarity score:  {result.global_similarity_score:.3f}")
    print(f"Label:                    {result.label.value}")
    if result.aggregated_criteria:
        ac = result.aggregated_criteria
        print(f"Aggregated criteria:")
        print(f"  problem:      {ac.problem_similarity:.2f}")
        print(f"  method:       {ac.method_similarity:.2f}")
        print(f"  domain:       {ac.domain_similarity:.2f}")
        print(f"  contribution: {ac.contribution_similarity:.2f}")
    print(f"Sentence labels: ", end="")
    for sa in result.sentence_annotations:
        color = {"high": "GREEN", "medium": "YELLOW", "low": "RED"}
        print(f"[{sa.index}]={color[sa.label.value]} ", end="")
    print()
    print(f"Summary: {result.summary[:200]}")


def test_all_low_overlap():
    """All papers score 1-2 across all criteria → HIGH originality."""
    results = [
        make_l1_result("p1", "Unrelated paper A", problem=1, method=1, domain=2, contribution=1,
                       sentence_overlaps=[1, 1, 1]),
        make_l1_result("p2", "Unrelated paper B", problem=2, method=1, domain=1, contribution=1,
                       sentence_overlaps=[1, 2, 1]),
        make_l1_result("p3", "Loosely related C",  problem=2, method=2, domain=2, contribution=1,
                       sentence_overlaps=[2, 2, 1]),
    ]
    return results


def test_one_threatening_paper():
    """4 papers low, 1 paper very high overlap → should still flag concern."""
    results = [
        make_l1_result("p1", "Unrelated A", problem=1, method=1, domain=1, contribution=1,
                       sentence_overlaps=[1, 1, 1]),
        make_l1_result("p2", "Unrelated B", problem=1, method=2, domain=1, contribution=1,
                       sentence_overlaps=[1, 1, 1]),
        make_l1_result("p3", "Unrelated C", problem=2, method=1, domain=2, contribution=1,
                       sentence_overlaps=[1, 2, 1]),
        make_l1_result("p4", "Unrelated D", problem=1, method=1, domain=1, contribution=1,
                       sentence_overlaps=[1, 1, 1]),
        # This paper is very similar
        make_l1_result("p5", "VERY SIMILAR PAPER", problem=5, method=4, domain=5, contribution=4,
                       sentence_overlaps=[5, 4, 5]),
    ]
    return results


def test_moderate_overlap():
    """Several papers with moderate overlap → MEDIUM originality."""
    results = [
        make_l1_result("p1", "Related A", problem=3, method=2, domain=3, contribution=2,
                       sentence_overlaps=[3, 2, 2]),
        make_l1_result("p2", "Related B", problem=2, method=3, domain=3, contribution=2,
                       sentence_overlaps=[2, 3, 2]),
        make_l1_result("p3", "Related C", problem=3, method=3, domain=2, contribution=3,
                       sentence_overlaps=[3, 3, 3]),
    ]
    return results


def test_guardrail_critical():
    """One paper has a criterion at 5 → guardrail should enforce floor."""
    results = [
        make_l1_result("p1", "Unrelated A", problem=1, method=1, domain=1, contribution=1,
                       sentence_overlaps=[1, 1, 1]),
        # This paper has contribution=5 but otherwise low
        make_l1_result("p2", "Same contribution", problem=2, method=1, domain=2, contribution=5,
                       sentence_overlaps=[1, 1, 4]),
    ]
    return results


def test_guardrail_high_count():
    """One paper has 2+ criteria at 4 → high guardrail triggers."""
    results = [
        make_l1_result("p1", "Unrelated A", problem=1, method=1, domain=1, contribution=1,
                       sentence_overlaps=[1, 1, 1]),
        # 3 criteria at 4
        make_l1_result("p2", "Multi-high paper", problem=4, method=4, domain=4, contribution=2,
                       sentence_overlaps=[4, 4, 3]),
    ]
    return results


if __name__ == "__main__":
    user_sentences = ["Sentence 0", "Sentence 1", "Sentence 2"]
    aggregator = Layer2Aggregator()

    scenarios = [
        ("All Low Overlap (expect HIGH originality)", test_all_low_overlap()),
        ("One Threatening Paper (expect LOW/MEDIUM originality)", test_one_threatening_paper()),
        ("Moderate Overlap (expect MEDIUM originality)", test_moderate_overlap()),
        ("Guardrail: Criterion at 5 (expect floor applied)", test_guardrail_critical()),
        ("Guardrail: 2+ Criteria at 4 (expect floor applied)", test_guardrail_high_count()),
    ]

    all_results = []
    for label, l1_results in scenarios:
        result = aggregator.aggregate(l1_results, user_sentences)
        print_result(label, result)
        all_results.append((label, result))

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Scenario':<50} {'Score':>5} {'Label':>8} {'Similarity':>10}")
    print("-" * 77)
    for label, r in all_results:
        short_label = label.split("(")[0].strip()
        print(f"{short_label:<50} {r.originality_score:>5} {r.label.value:>8} {r.global_similarity_score:>10.3f}")

    # Sanity checks
    low_r = all_results[0][1]
    threat_r = all_results[1][1]
    mod_r = all_results[2][1]

    assert low_r.originality_score > mod_r.originality_score, \
        "All-low should score higher originality than moderate"
    assert low_r.originality_score > threat_r.originality_score, \
        "All-low should score higher originality than one-threatening"
    assert threat_r.originality_score <= 50, \
        "One very similar paper should bring originality below 50"

    # Guardrail checks
    guardrail_crit = all_results[3][1]
    assert guardrail_crit.global_similarity_score >= config.GUARDRAIL_CRITICAL_FLOOR, \
        f"Critical guardrail should enforce floor of {config.GUARDRAIL_CRITICAL_FLOOR}"

    guardrail_high = all_results[4][1]
    assert guardrail_high.global_similarity_score >= config.GUARDRAIL_HIGH_FLOOR, \
        f"High guardrail should enforce floor of {config.GUARDRAIL_HIGH_FLOOR}"

    print("\nAll sanity checks passed!")
