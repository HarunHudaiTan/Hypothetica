"""
Test script for Layer 1 Agent (criterion-per-paper scoring).

Requires GOOGLE_API_KEY in envfiles/.env.
Makes real API calls — 5 calls per paper (4 criteria + 1 sentence analysis).

Usage:
    cd backend
    python -m tests.test_layer1
"""
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from app.agents.layer1_agent import Layer1Agent
from app.models.paper import Paper, Heading


def make_mock_paper() -> Paper:
    """Create a mock paper for testing."""
    return Paper(
        paper_id="test_paper_01",
        arxiv_id="2401.99999",
        title="Graph Attention Networks for Molecular Property Prediction",
        abstract=(
            "We propose a graph attention network (GAT) architecture for predicting "
            "molecular properties from 2D molecular graphs. Our method leverages "
            "multi-head attention to capture long-range atomic interactions and achieves "
            "state-of-the-art results on the MoleculeNet benchmark. We demonstrate that "
            "attention-based message passing outperforms traditional GCN approaches by "
            "15% on BACE and 8% on BBBP classification tasks."
        ),
        url="https://arxiv.org/abs/2401.99999",
        pdf_url="https://arxiv.org/pdf/2401.99999",
        authors=["Alice Smith", "Bob Jones"],
        categories=["cs.LG", "q-bio.BM"],
        headings=[
            Heading(
                heading_id="test_paper_01_h00",
                paper_id="test_paper_01",
                index=0,
                level=2,
                text="Introduction",
                raw_text="Introduction",
                section_text=(
                    "Predicting molecular properties is a fundamental task in drug discovery. "
                    "Traditional approaches rely on hand-crafted molecular descriptors, but "
                    "recent work has shown that graph neural networks can learn directly from "
                    "molecular graphs. We build on this line of work by introducing attention "
                    "mechanisms into the message passing framework."
                ),
                is_valid=True,
            ),
            Heading(
                heading_id="test_paper_01_h01",
                paper_id="test_paper_01",
                index=1,
                level=2,
                text="Methods",
                raw_text="Methods",
                section_text=(
                    "Our architecture consists of a multi-head graph attention layer followed "
                    "by a readout function and MLP classifier. We use 4 attention heads with "
                    "64-dimensional hidden states. The attention mechanism computes pairwise "
                    "attention coefficients between bonded atoms, allowing the model to weigh "
                    "the importance of different neighbors during message passing. We train "
                    "using binary cross-entropy loss with Adam optimizer (lr=1e-3)."
                ),
                is_valid=True,
            ),
            Heading(
                heading_id="test_paper_01_h02",
                paper_id="test_paper_01",
                index=2,
                level=2,
                text="Results",
                raw_text="Results",
                section_text=(
                    "On the MoleculeNet benchmark, our GAT model achieves AUC-ROC of 0.89 "
                    "on BACE and 0.92 on BBBP, outperforming baseline GCN by 15% and 8% "
                    "respectively. Ablation studies show that multi-head attention is critical "
                    "for capturing long-range interactions in large molecules."
                ),
                is_valid=True,
            ),
        ],
        is_processed=True,
    )


def test_high_overlap():
    """Test with a user idea that closely matches the mock paper."""
    print("\n" + "=" * 70)
    print("TEST: HIGH OVERLAP SCENARIO")
    print("=" * 70)

    user_idea = (
        "I want to use graph attention networks to predict molecular properties. "
        "My approach uses multi-head attention on molecular graphs to capture "
        "atomic interactions. I plan to evaluate on the MoleculeNet benchmark "
        "including BACE and BBBP datasets."
    )
    user_sentences = [
        "I want to use graph attention networks to predict molecular properties.",
        "My approach uses multi-head attention on molecular graphs to capture atomic interactions.",
        "I plan to evaluate on the MoleculeNet benchmark including BACE and BBBP datasets.",
    ]

    agent = Layer1Agent()
    paper = make_mock_paper()
    result = agent.analyze_paper(user_idea, user_sentences, paper)

    print(f"\nIdea similarity: {result.idea_similarity_score:.2f}")
    print(f"Similarity level: {result.similarity_level}")
    print(f"Confidence: {result.confidence}")
    print(f"\nCriteria scores:")
    print(f"  problem_similarity:      {result.criteria_scores.problem_similarity:.2f}")
    print(f"  method_similarity:       {result.criteria_scores.method_similarity:.2f}")
    print(f"  domain_overlap:          {result.criteria_scores.domain_overlap:.2f}")
    print(f"  contribution_similarity: {result.criteria_scores.contribution_similarity:.2f}")
    print(f"\nSentence analyses:")
    for sa in result.sentence_analyses:
        print(f"  [{sa.sentence_index}] similarity={sa.similarity_score:.2f}")
        print(f"      \"{sa.sentence[:80]}...\"")
        for ms in sa.matched_sections[:2]:
            print(f"      -> {ms.heading}: {ms.reason[:80]}")
    print(f"\nTokens used: {result.tokens_used}")
    print(f"Cost: ${agent.get_cost():.4f}")
    return result


def test_low_overlap():
    """Test with a user idea that is unrelated to the mock paper."""
    print("\n" + "=" * 70)
    print("TEST: LOW OVERLAP SCENARIO")
    print("=" * 70)

    user_idea = (
        "I propose a reinforcement learning approach for autonomous drone "
        "navigation in indoor environments. The system uses depth cameras and "
        "IMU sensors to build a 3D map while learning obstacle avoidance policies."
    )
    user_sentences = [
        "I propose a reinforcement learning approach for autonomous drone navigation in indoor environments.",
        "The system uses depth cameras and IMU sensors to build a 3D map while learning obstacle avoidance policies.",
    ]

    agent = Layer1Agent()
    paper = make_mock_paper()
    result = agent.analyze_paper(user_idea, user_sentences, paper)

    print(f"\nIdea similarity: {result.idea_similarity_score:.2f}")
    print(f"Similarity level: {result.similarity_level}")
    print(f"\nCriteria scores:")
    print(f"  problem_similarity:      {result.criteria_scores.problem_similarity:.2f}")
    print(f"  method_similarity:       {result.criteria_scores.method_similarity:.2f}")
    print(f"  domain_overlap:          {result.criteria_scores.domain_overlap:.2f}")
    print(f"  contribution_similarity: {result.criteria_scores.contribution_similarity:.2f}")
    print(f"\nTokens used: {result.tokens_used}")
    print(f"Cost: ${agent.get_cost():.4f}")
    return result


if __name__ == "__main__":
    print("Layer 1 Agent Test — Criterion-per-paper scoring")
    print("Each paper triggers 5 API calls (4 criteria + 1 sentence analysis)\n")

    r1 = test_high_overlap()
    r2 = test_low_overlap()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"High overlap scenario: overall={r1.idea_similarity_score:.2f}, level={r1.similarity_level}")
    print(f"Low overlap scenario:  overall={r2.idea_similarity_score:.2f}, level={r2.similarity_level}")

    # Sanity check
    assert r1.idea_similarity_score > r2.idea_similarity_score, \
        "High overlap scenario should have higher similarity than low overlap scenario!"
    print("\nSanity check passed.")
