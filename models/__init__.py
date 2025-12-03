"""
Data models for the Hypothetica Originality System.
"""
from models.paper import Paper, Heading, Chunk
from models.analysis import (
    Layer1Result,
    Layer2Result,
    SentenceAnalysis,
    CriteriaScores,
    MatchedSection,
    SentenceAnnotation,
    CostBreakdown
)

__all__ = [
    'Paper',
    'Heading', 
    'Chunk',
    'Layer1Result',
    'Layer2Result',
    'SentenceAnalysis',
    'CriteriaScores',
    'MatchedSection',
    'SentenceAnnotation',
    'CostBreakdown'
]

