from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class PaperSource(str, Enum):
    ARXIV = "arxiv"
    GOOGLE_PATENTS = "google_patents"

class AnalyzeRequest(BaseModel):
    user_idea: str = Field(..., min_length=50)
    papers_per_query: int = Field(default=150, ge=50, le=300)
    papers_per_variant_conversion: int = Field(
        default=40,
        ge=10,
        le=200,
        description="How many hits to keep from each search query variant (arXiv, patents, etc.) before dedup.",
    )
    embedding_topk: int = Field(default=100, ge=50, le=200)
    rerank_topk: int = Field(default=20, ge=10, le=50)
    final_papers: int = Field(default=5, ge=3, le=10)
    use_reranker: bool = True
    selected_adapter: Optional[str] = Field(
        default="arxiv",
        description="Evidence adapter to use (e.g., 'arxiv', 'google_patents'). Defaults to 'arxiv'."
    )
    selected_sources: Optional[List[PaperSource]] = Field(
        default=None,
        description="(Deprecated) List of paper sources to use. Use selected_adapter instead."
    )
    benchmark_mode: bool = Field(
        default=False,
        description="Skip follow-up questions, GitHub analysis, and report generation for faster batch runs."
    )

class AnswersRequest(BaseModel):
    answers: List[str]
