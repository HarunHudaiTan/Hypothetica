"""
Data models for originality analysis results.
Enhanced with dimension-level analysis and evidence support.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
from enum import Enum


class OriginalityLabel(str, Enum):
    """Originality classification labels."""
    LOW = "low"       # Red - high overlap
    MEDIUM = "medium"  # Yellow - moderate overlap
    HIGH = "high"      # Green - low overlap/novel


class SimilarityCategory(str, Enum):
    """Categorical similarity scores (more reliable than continuous)."""
    NONE = "NONE"           # 0.0 - No overlap
    MINIMAL = "MINIMAL"     # 0.25 - Slight overlap
    MODERATE = "MODERATE"   # 0.5 - Some meaningful overlap
    SIGNIFICANT = "SIGNIFICANT"  # 0.75 - Substantial overlap
    COMPLETE = "COMPLETE"   # 1.0 - Identical
    
    @property
    def numerical(self) -> float:
        """Convert category to numerical value."""
        mapping = {
            "NONE": 0.0,
            "MINIMAL": 0.25,
            "MODERATE": 0.5,
            "SIGNIFICANT": 0.75,
            "COMPLETE": 1.0
        }
        return mapping.get(self.value, 0.0)
    
    @classmethod
    def from_string(cls, value: str) -> 'SimilarityCategory':
        """Convert string to SimilarityCategory."""
        try:
            return cls(value.upper())
        except ValueError:
            return cls.NONE


@dataclass
class QuestionAnalysis:
    """
    Analysis of a single question in the dimension framework.
    """
    question: str
    score: SimilarityCategory
    answer: str                         # 1-2 sentence explanation
    evidence: List[str] = field(default_factory=list)  # 2-4 quotes from paper
    
    @property
    def numerical_score(self) -> float:
        return self.score.numerical if isinstance(self.score, SimilarityCategory) else SimilarityCategory.from_string(str(self.score)).numerical
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "score": self.score.value if isinstance(self.score, SimilarityCategory) else str(self.score),
            "answer": self.answer,
            "evidence": self.evidence
        }


@dataclass
class DimensionAnalysis:
    """
    Analysis of one dimension (Technical Problem, Method, Domain, or Claims).
    Contains multiple questions.
    """
    dimension_name: str
    questions: List[QuestionAnalysis] = field(default_factory=list)
    
    @property
    def average_score(self) -> float:
        """Average numerical score across all questions."""
        if not self.questions:
            return 0.0
        return sum(q.numerical_score for q in self.questions) / len(self.questions)
    
    def to_dict(self) -> dict:
        return {
            "dimension_name": self.dimension_name,
            "average_score": round(self.average_score, 3),
            "questions": [q.to_dict() for q in self.questions]
        }


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
    Enhanced with dimension mapping and evidence.
    """
    sentence: str
    sentence_index: int
    overlap_score: float           # 0-1 score (derived from dimension analysis)
    primary_dimension: str = ""    # Which dimension this sentence mainly relates to
    matched_sections: List[MatchedSection] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # Supporting quotes
    
    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "sentence_index": self.sentence_index,
            "overlap_score": self.overlap_score,
            "primary_dimension": self.primary_dimension,
            "evidence": self.evidence,
            "matched_sections": [
                {
                    "chunk_id": ms.chunk_id,
                    "paper_id": ms.paper_id,
                    "heading": ms.heading,
                    "text_snippet": ms.text_snippet,
                    "similarity": ms.similarity,
                    "reason": ms.reason
                }
                for ms in self.matched_sections
            ]
        }


@dataclass
class CriteriaScores:
    """
    TÜBİTAK-style originality criteria scores.
    All scores are 0-1 where higher = more similar (less original).
    Now derived from dimension analysis.
    """
    problem_similarity: float      # Technical Problem Novelty dimension
    method_similarity: float       # Methodological Innovation dimension
    domain_overlap: float          # Application Domain Overlap dimension
    contribution_similarity: float # Innovation Claims Overlap dimension
    
    def to_dict(self) -> dict:
        return {
            "problem_similarity": round(self.problem_similarity, 3),
            "method_similarity": round(self.method_similarity, 3),
            "domain_overlap": round(self.domain_overlap, 3),
            "contribution_similarity": round(self.contribution_similarity, 3)
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
    
    @property
    def weighted_average(self) -> float:
        """
        Weighted average: Method(0.4) + Problem(0.3) + Domain(0.2) + Claims(0.1)
        Method is weighted highest as it's most important for novelty.
        """
        return (
            self.problem_similarity * 0.30 +
            self.method_similarity * 0.40 +
            self.domain_overlap * 0.20 +
            self.contribution_similarity * 0.10
        )


@dataclass
class Layer1Result:
    """
    Complete Layer 1 analysis result for a single paper.
    Answers: "How similar is this paper to the user's idea?"
    
    HYBRID: Contains both dimension analysis (for explainability) 
    and sentence-level analysis (for UI/RAG).
    """
    paper_id: str
    paper_title: str
    arxiv_id: str
    
    # Overall scores
    overall_overlap_score: float   # 0-1, higher = more similar
    criteria_scores: CriteriaScores
    
    # Dimension analysis (11 questions with evidence)
    dimension_analyses: List[DimensionAnalysis] = field(default_factory=list)
    
    # Sentence-level analysis (for UI highlighting and RAG)
    sentence_analyses: List[SentenceAnalysis] = field(default_factory=list)
    
    # Brief analysis notes
    analysis_notes: str = ""
    
    # Processing metadata
    tokens_used: int = 0
    processing_time: float = 0.0   # seconds
    
    def get_dimension(self, name: str) -> Optional[DimensionAnalysis]:
        """Get a specific dimension analysis by name."""
        for dim in self.dimension_analyses:
            if dim.dimension_name.lower() == name.lower():
                return dim
        return None
    
    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "arxiv_id": self.arxiv_id,
            "overall_overlap_score": round(self.overall_overlap_score, 3),
            "criteria_scores": self.criteria_scores.to_dict(),
            "dimension_analyses": [d.to_dict() for d in self.dimension_analyses],
            "sentence_level": [sa.to_dict() for sa in self.sentence_analyses],
            "analysis_notes": self.analysis_notes,
            "tokens_used": self.tokens_used,
            "processing_time": round(self.processing_time, 2)
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
    primary_dimension: str = ""    # Which dimension caused the overlap
    linked_sections: List[MatchedSection] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # Supporting quotes from papers
    
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "sentence": self.sentence,
            "originality_score": round(self.originality_score, 3),
            "overlap_score": round(self.overlap_score, 3),
            "label": self.label.value,
            "primary_dimension": self.primary_dimension,
            "evidence": self.evidence,
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
    reality_check: float = 0.0
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
                "reality_check": round(self.reality_check, 4)
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
    
    # Aggregated dimension analyses
    aggregated_dimensions: Dict[str, float] = field(default_factory=dict)
    
    # Papers analyzed
    papers_analyzed: int = 0
    
    # Cost tracking
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    
    # Processing metadata
    total_processing_time: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "global_originality_score": self.global_originality_score,
            "global_overlap_score": round(self.global_overlap_score, 3),
            "label": self.label.value,
            "sentence_annotations": [sa.to_dict() for sa in self.sentence_annotations],
            "summary": self.summary,
            "aggregated_criteria": self.aggregated_criteria.to_dict() if self.aggregated_criteria else None,
            "aggregated_dimensions": {k: round(v, 3) for k, v in self.aggregated_dimensions.items()},
            "papers_analyzed": self.papers_analyzed,
            "cost": self.cost.to_dict(),
            "total_processing_time": round(self.total_processing_time, 2)
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
