"""
Data models for originality analysis results.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
from enum import Enum


class OriginalityLabel(str, Enum):
    """Originality classification labels."""
    LOW = "low"       # Red - high overlap
    MEDIUM = "medium"  # Yellow - moderate overlap
    HIGH = "high"      # Green - low overlap/novel


@dataclass
class MatchedSection:
    """
    A section/chunk that matches a user's sentence.
    """
    chunk_id: str
    paper_id: str
    paper_title: str
    heading: str
    text_snippet: str              # Relevant excerpt from chunk
    similarity: float              # Cosine similarity score
    reason: str                    # Why this matches
    criterion: str = ""            # Which criterion this match belongs to (e.g. "method_similarity")


@dataclass
class SentenceAnalysis:
    """
    Layer 1 analysis of a single user sentence against a paper.
    """
    sentence: str
    sentence_index: int
    similarity_score: float        # 0-1 score (weighted avg of criteria, kept for backward compat)
    matched_sections: List[MatchedSection] = field(default_factory=list)
    sentence_criteria_scores: Optional["CriteriaScores"] = None  # Per-criterion scores (0-1) for this sentence


@dataclass
class CriteriaScores:
    """
    TÜBİTAK-style originality criteria scores.
    All scores are 0-1 where higher = more similar (less original).
    """
    problem_similarity: float      # How similar is the problem definition
    method_similarity: float       # How similar is the proposed method
    domain_overlap: float          # How much domain/application overlap
    contribution_similarity: float # How similar are the claimed contributions
    
    def to_dict(self) -> dict:
        return {
            "problem_similarity": self.problem_similarity,
            "method_similarity": self.method_similarity,
            "domain_overlap": self.domain_overlap,
            "contribution_similarity": self.contribution_similarity
        }
    
    @property
    def average(self) -> float:
        """Average of all criteria scores."""
        return (
            self.problem_similarity + 
            self.method_similarity + 
            self.domain_overlap + 
            self.contribution_similarity
        ) / 4


@dataclass
class Layer1Result:
    """
    Complete Layer 1 analysis result for a single paper.
    Answers: "How similar is this paper to the user's idea?"
    """
    paper_id: str
    paper_title: str
    arxiv_id: str  # Keep as arxiv_id for backward compatibility, but will store source_id
    
    # Overall scores
    paper_similarity_score: float   # 0-1, higher = more similar
    criteria_scores: CriteriaScores

    # Sentence-level analysis
    sentence_analyses: List[SentenceAnalysis] = field(default_factory=list)

    # Human-readable reason explaining the scores
    reason: str = ""

    # Confidence and threat assessment
    confidence: str = "medium"          # low, medium, high
    similarity_level: str = "low"       # low, moderate, high

    # Processing metadata
    tokens_used: int = 0
    processing_time: float = 0.0   # seconds
    
    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "arxiv_id": self.arxiv_id,
            "paper_similarity_score": self.paper_similarity_score,
            "reason": self.reason,
            "criteria_scores": self.criteria_scores.to_dict(),
            "sentence_level": [
                {
                    "sentence": sa.sentence,
                    "sentence_index": sa.sentence_index,
                    "similarity_score": sa.similarity_score,
                    "matched_sections": [
                        {
                            "chunk_id": ms.chunk_id,
                            "paper_id": ms.paper_id,
                            "heading": ms.heading,
                            "text_snippet": ms.text_snippet,
                            "similarity": ms.similarity,
                            "reason": ms.reason
                        }
                        for ms in sa.matched_sections
                    ]
                }
                for sa in self.sentence_analyses
            ],
            "confidence": self.confidence,
            "similarity_level": self.similarity_level
        }


@dataclass
class SentenceAnnotation:
    """
    Final annotation for a user sentence after Layer 2 processing.
    """
    index: int
    sentence: str
    originality_score: float       # 0-1, higher = MORE original (inverted from overlap)
    similarity_score: float        # 0-1, higher = more overlap (less original)
    label: OriginalityLabel
    linked_sections: List[MatchedSection] = field(default_factory=list)
    criteria_labels: Dict[str, OriginalityLabel] = field(default_factory=dict)  # Per-criterion labels (only passing ones)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "sentence": self.sentence,
            "originality_score": self.originality_score,
            "similarity_score": self.similarity_score,
            "label": self.label.value,
            "criteria_labels": {k: v.value for k, v in self.criteria_labels.items()},
            "linked_sections": [
                {
                    "chunk_id": ls.chunk_id,
                    "paper_id": ls.paper_id,
                    "paper_title": ls.paper_title,
                    "heading": ls.heading,
                    "text_snippet": ls.text_snippet,
                    "similarity": ls.similarity,
                    "reason": ls.reason,
                    "criterion": ls.criterion,
                }
                for ls in self.linked_sections
            ]
        }


# =============================================================================
# GitHub Analysis Models
# =============================================================================

@dataclass
class RepoRelevanceResult:
    """Result of RepoRelevanceAgent analysis for a single GitHub repo."""
    repo_full_name: str
    repo_url: str
    stars: int
    description: str
    last_pushed: str
    topics: List[str] = field(default_factory=list)
    overlap_score: float = 0.0
    what_it_covers: str = ""
    what_it_misses: str = ""
    verdict: str = "unrelated"  # strong_overlap, partial_overlap, tangential, unrelated

    def to_dict(self) -> dict:
        return {
            "repo_full_name": self.repo_full_name,
            "repo_url": self.repo_url,
            "stars": self.stars,
            "description": self.description,
            "last_pushed": self.last_pushed,
            "topics": self.topics,
            "overlap_score": self.overlap_score,
            "what_it_covers": self.what_it_covers,
            "what_it_misses": self.what_it_misses,
            "verdict": self.verdict,
        }


@dataclass
class GitHubAnalysisResult:
    """Complete GitHub evidence analysis."""
    synthesis: str = ""
    verdict: str = ""  # pursue_as_is, refine_scope, reconsider
    repos_analyzed: int = 0
    repos_relevant: int = 0
    repo_results: List[RepoRelevanceResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "synthesis": self.synthesis,
            "verdict": self.verdict,
            "repos_analyzed": self.repos_analyzed,
            "repos_relevant": self.repos_relevant,
            "repo_results": [r.to_dict() for r in self.repo_results],
        }


@dataclass
class CostBreakdown:
    """
    Token cost breakdown for the analysis.
    """
    retrieval: float = 0.0
    layer1: float = 0.0
    layer2: float = 0.0
    followup: float = 0.0
    keywords: float = 0.0
    reality_check: float = 0.0
    github: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "estimated_cost_usd": round(self.total, 4),
            "breakdown": {
                "retrieval": round(self.retrieval, 4),
                "layer1": round(self.layer1, 4),
                "layer2": round(self.layer2, 4),
                "followup": round(self.followup, 4),
                "keywords": round(self.keywords, 4),
                "reality_check": round(self.reality_check, 4),
                "github": round(self.github, 4),
            }
        }


@dataclass
class Layer2Result:
    """
    Complete Layer 2 result - global originality assessment.
    """
    # Global scores
    originality_score: int         # 0-100, higher = more original
    global_similarity_score: float # 0-1, average overlap
    label: OriginalityLabel        # Overall label
    
    # Sentence annotations
    sentence_annotations: List[SentenceAnnotation] = field(default_factory=list)
    
    # Summary and comprehensive report
    summary: str = ""              # 1-2 sentence explanation
    comprehensive_report: str = ""  # Detailed analysis report
    
    # Aggregated criteria (averaged across papers)
    aggregated_criteria: Optional[CriteriaScores] = None
    
    # Papers analyzed
    papers_analyzed: int = 0
    
    # Cost tracking
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    
    # Processing metadata
    total_processing_time: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "originality_score": self.originality_score,
            "global_similarity_score": self.global_similarity_score,
            "label": self.label.value,
            "sentence_annotations": [sa.to_dict() for sa in self.sentence_annotations],
            "summary": self.summary,
            "comprehensive_report": self.comprehensive_report,
            "aggregated_criteria": self.aggregated_criteria.to_dict() if self.aggregated_criteria else None,
            "papers_analyzed": self.papers_analyzed,
            "cost": self.cost.to_dict(),
            "total_processing_time": self.total_processing_time
        }
    
    def get_sentences_by_label(self, label: OriginalityLabel) -> List[SentenceAnnotation]:
        """Get all sentences with a specific label."""
        return [sa for sa in self.sentence_annotations if sa.label == label]
    
    @property
    def red_sentences(self) -> List[SentenceAnnotation]:
        """Sentences with low originality (high overlap)."""
        return self.get_sentences_by_label(OriginalityLabel.LOW)
    
    @property
    def yellow_sentences(self) -> List[SentenceAnnotation]:
        """Sentences with medium originality."""
        return self.get_sentences_by_label(OriginalityLabel.MEDIUM)
    
    @property
    def green_sentences(self) -> List[SentenceAnnotation]:
        """Sentences with high originality."""
        return self.get_sentences_by_label(OriginalityLabel.HIGH)

