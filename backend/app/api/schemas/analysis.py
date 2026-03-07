from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class AnalyzeRequest(BaseModel):
    user_idea: str = Field(..., min_length=50)
    papers_per_query: int = Field(default=150, ge=50, le=300)
    embedding_topk: int = Field(default=100, ge=50, le=200)
    rerank_topk: int = Field(default=20, ge=10, le=50)
    final_papers: int = Field(default=5, ge=3, le=10)
    use_reranker: bool = True

class AnswersRequest(BaseModel):
    answers: List[str]
