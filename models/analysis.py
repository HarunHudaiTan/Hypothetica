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
class PaperMetadata:
    """
    Full metadata for a paper - used for grounded RAG display.
    """
    paper_id: str
    arxiv_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    publication_date: str = ""
    categories: List[str] = field(default_factory=list)
    abstract: str = ""
    
    @property
    def arxiv_url(self) -> str:
        """Direct link to arxiv paper."""
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return ""
    
    @property
    def pdf_url(self) -> str:
        """Direct link to PDF."""
        if self.arxiv_id:
            return f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        return ""
    
    @property
    def authors_str(self) -> str:
        """Formatted author string."""
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al."
    
    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "authors_str": self.authors_str,
            "publication_date": self.publication_date,
            "categories": self.categories,
            "arxiv_url": self.arxiv_url,
            "pdf_url": self.pdf_url
        }


@dataclass
class MatchedSection:
    """
    A section/chunk that matches a user's sentence.
    Enhanced with full paper metadata and context for grounded RAG display.
    """
    # Chunk identification
    chunk_id: str
    paper_id: str
    paper_title: str
    
    # Section context
    heading: str                   # Section heading (e.g., "3.1 Methodology")
    heading_hierarchy: List[str] = field(default_factory=list)  # ["3. Methods", "3.1 Methodology"]
    
    # Text content with context
    text_snippet: str              # The matching excerpt
    context_before: str = ""       # 1-2 sentences before for context
    context_after: str = ""        # 1-2 sentences after for context
    highlight_start: int = 0       # Character position where match starts
    highlight_end: int = 0         # Character position where match ends
    
    # Matching info
    similarity: float = 0.0        # Cosine similarity score (0-1)
    reason: str = ""               # Why this matches (from LLM)
    dimension: str = ""            # Which dimension this relates to (problem/method/domain/claims)
    
    # Full paper metadata (for grounded display)
    paper_metadata: Optional[PaperMetadata] = None
    
    @property
    def full_context(self) -> str:
        """Get full context with the matched part."""
        parts = []
        if self.context_before:
            parts.append(f"...{self.context_before}")
        parts.append(self.text_snippet)
        if self.context_after:
            parts.append(f"{self.context_after}...")
        return " ".join(parts)
    
    @property 
    def arxiv_url(self) -> str:
        """Get arxiv URL from paper metadata."""
        if self.paper_metadata:
            return self.paper_metadata.arxiv_url
        return ""
    
    @property
    def similarity_percent(self) -> str:
        """Similarity as percentage string."""
        return f"{self.similarity * 100:.1f}%"
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "heading": self.heading,
            "heading_hierarchy": self.heading_hierarchy,
            "text_snippet": self.text_snippet,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "full_context": self.full_context,
            "similarity": self.similarity,
            "similarity_percent": self.similarity_percent,
            "reason": self.reason,
            "dimension": self.dimension,
            "arxiv_url": self.arxiv_url,
            "paper_metadata": self.paper_metadata.to_dict() if self.paper_metadata else None
        }
    
    def to_display_dict(self) -> dict:
        """Formatted dict for UI display."""
        return {
            "paper": {
                "title": self.paper_title,
                "arxiv_id": self.paper_metadata.arxiv_id if self.paper_metadata else "",
                "authors": self.paper_metadata.authors_str if self.paper_metadata else "",
                "date": self.paper_metadata.publication_date if self.paper_metadata else "",
                "url": self.arxiv_url,
                "categories": self.paper_metadata.categories if self.paper_metadata else []
            },
            "match": {
                "section": self.heading,
                "section_path": " > ".join(self.heading_hierarchy) if self.heading_hierarchy else self.heading,
                "similarity": self.similarity_percent,
                "dimension": self.dimension,
                "reason": self.reason
            },
            "content": {
                "snippet": self.text_snippet,
                "full_context": self.full_context,
                "context_before": self.context_before,
                "context_after": self.context_after
            }
        }


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


# ============================================================================
# RAG Display Formatters
# ============================================================================

class RAGDisplayFormatter:
    """
    Utility class to format RAG results for grounded display.
    """
    
    DIMENSION_LABELS = {
        "technical_problem_novelty": "🔬 Problem Novelty",
        "methodological_innovation": "⚙️ Methodology",
        "application_domain_overlap": "🎯 Application Domain",
        "innovation_claims_overlap": "💡 Innovation Claims",
        "problem": "🔬 Problem",
        "method": "⚙️ Method",
        "domain": "🎯 Domain",
        "claims": "💡 Claims"
    }
    
    @classmethod
    def format_matched_section_for_display(cls, section: MatchedSection) -> dict:
        """
        Format a single matched section for rich UI display.
        """
        # Get dimension label
        dim_key = section.dimension.lower() if section.dimension else ""
        dim_label = cls.DIMENSION_LABELS.get(dim_key, section.dimension or "General")
        
        return {
            "paper_info": {
                "title": section.paper_title,
                "arxiv_id": section.paper_metadata.arxiv_id if section.paper_metadata else "",
                "authors": section.paper_metadata.authors_str if section.paper_metadata else "Unknown",
                "date": section.paper_metadata.publication_date if section.paper_metadata else "",
                "url": section.arxiv_url,
                "pdf_url": section.paper_metadata.pdf_url if section.paper_metadata else ""
            },
            "section_info": {
                "heading": section.heading,
                "path": " → ".join(section.heading_hierarchy) if section.heading_hierarchy else section.heading,
                "dimension": dim_label,
                "similarity": f"{section.similarity * 100:.1f}%"
            },
            "content": {
                "matched_text": section.text_snippet,
                "context_before": section.context_before,
                "context_after": section.context_after,
                "full_context": section.full_context
            },
            "explanation": {
                "reason": section.reason,
                "dimension_raw": section.dimension
            }
        }
    
    @classmethod
    def format_sentence_matches_for_display(cls, annotation: 'SentenceAnnotation') -> dict:
        """
        Format all matches for a sentence into a structured display format.
        """
        # Group matches by paper
        papers = {}
        for section in annotation.linked_sections:
            paper_key = section.paper_id
            if paper_key not in papers:
                papers[paper_key] = {
                    "paper_info": {
                        "title": section.paper_title,
                        "arxiv_id": section.paper_metadata.arxiv_id if section.paper_metadata else "",
                        "authors": section.paper_metadata.authors_str if section.paper_metadata else "Unknown",
                        "url": section.arxiv_url
                    },
                    "matches": []
                }
            papers[paper_key]["matches"].append(cls.format_matched_section_for_display(section))
        
        # Get dimension label for primary dimension
        dim_key = annotation.primary_dimension.lower() if annotation.primary_dimension else ""
        primary_dim_label = cls.DIMENSION_LABELS.get(dim_key, annotation.primary_dimension or "General")
        
        return {
            "sentence": {
                "text": annotation.sentence,
                "index": annotation.index,
                "overlap_score": f"{annotation.overlap_score * 100:.0f}%",
                "originality_score": f"{annotation.originality_score * 100:.0f}%",
                "label": annotation.label.value,
                "primary_dimension": primary_dim_label
            },
            "evidence": annotation.evidence[:5],  # Top 5 evidence quotes
            "papers": list(papers.values()),
            "total_matches": len(annotation.linked_sections)
        }
    
    @classmethod
    def generate_markdown_report(cls, annotation: 'SentenceAnnotation') -> str:
        """
        Generate a markdown-formatted report for a sentence's matches.
        """
        data = cls.format_sentence_matches_for_display(annotation)
        
        lines = [
            f"## Matching Sources for Sentence",
            f"",
            f"**Your sentence:** \"{data['sentence']['text']}\"",
            f"",
            f"**Overlap Score:** {data['sentence']['overlap_score']} | **Primary Dimension:** {data['sentence']['primary_dimension']}",
            f"",
            f"---",
            f""
        ]
        
        if data['evidence']:
            lines.append("### Key Evidence")
            for i, ev in enumerate(data['evidence'], 1):
                lines.append(f"{i}. \"{ev}\"")
            lines.append("")
        
        lines.append(f"### Matching Papers ({len(data['papers'])} found)")
        lines.append("")
        
        for paper in data['papers']:
            info = paper['paper_info']
            lines.append(f"#### 📄 {info['title']}")
            lines.append(f"")
            lines.append(f"- **Authors:** {info['authors']}")
            lines.append(f"- **ArXiv:** [{info['arxiv_id']}]({info['url']})" if info['arxiv_id'] else "")
            lines.append("")
            
            for match in paper['matches']:
                section = match['section_info']
                content = match['content']
                explanation = match['explanation']
                
                lines.append(f"**Section:** {section['path']} | **Similarity:** {section['similarity']} | **Dimension:** {section['dimension']}")
                lines.append("")
                lines.append(f"> {content['full_context']}")
                lines.append("")
                if explanation['reason']:
                    lines.append(f"*Why this matches:* {explanation['reason']}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
