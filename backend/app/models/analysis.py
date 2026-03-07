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


@dataclass
class SentenceAnalysis:
    """
    Layer 1 analysis of a single user sentence against a paper.
    """
    sentence: str
    sentence_index: int
    overlap_score: float           # 0-1 score
    matched_sections: List[MatchedSection] = field(default_factory=list)


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
    arxiv_id: str
    
    # Overall scores
    overall_overlap_score: float   # 0-1, higher = more similar
    criteria_scores: CriteriaScores
    
    # Sentence-level analysis
    sentence_analyses: List[SentenceAnalysis] = field(default_factory=list)
    
    # Processing metadata
    tokens_used: int = 0
    processing_time: float = 0.0   # seconds
    
    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "arxiv_id": self.arxiv_id,
            "overall_overlap_score": self.overall_overlap_score,
            "criteria_scores": self.criteria_scores.to_dict(),
            "sentence_level": [
                {
                    "sentence": sa.sentence,
                    "sentence_index": sa.sentence_index,
                    "overlap_score": sa.overlap_score,
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
            ]
        }


@dataclass
class SentenceAnnotation:
    """
    Final annotation for a user sentence after Layer 2 processing.
    """
    index: int
    sentence: str
    originality_score: float       # 0-1, higher = MORE original (inverted from overlap)
    overlap_score: float           # 0-1, higher = more overlap (less original)
    label: OriginalityLabel
    linked_sections: List[MatchedSection] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "sentence": self.sentence,
            "originality_score": self.originality_score,
            "overlap_score": self.overlap_score,
            "label": self.label.value,
            "linked_sections": [
                {
                    "chunk_id": ls.chunk_id,
                    "paper_id": ls.paper_id,
                    "paper_title": ls.paper_title,
                    "heading": ls.heading,
                    "text_snippet": ls.text_snippet,
                    "similarity": ls.similarity,
                    "reason": ls.reason
                }
                for ls in self.linked_sections
            ]
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
    total: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "estimated_cost_usd": round(self.total, 4),
            "breakdown": {
                "retrieval": round(self.retrieval, 4),
                "layer1": round(self.layer1, 4),
                "layer2": round(self.layer2, 4),
                "followup": round(self.followup, 4),
                "keywords": round(self.keywords, 4)
            }
        }


@dataclass
class Layer2Result:
    """
    Complete Layer 2 result - global originality assessment.
    """
    # Global scores
    global_originality_score: int  # 0-100, higher = more original
    global_overlap_score: float    # 0-1, average overlap
    label: OriginalityLabel        # Overall label
    
    # Sentence annotations
    sentence_annotations: List[SentenceAnnotation] = field(default_factory=list)
    
    # Summary
    summary: str = ""              # 1-2 sentence explanation
    
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
            "global_originality_score": self.global_originality_score,
            "global_overlap_score": self.global_overlap_score,
            "label": self.label.value,
            "sentence_annotations": [sa.to_dict() for sa in self.sentence_annotations],
            "summary": self.summary,
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

