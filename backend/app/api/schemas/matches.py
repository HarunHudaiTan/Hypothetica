from pydantic import BaseModel, Field

class SentenceMatchRequest(BaseModel):
    sentence: str
    top_k: int = Field(default=5, ge=1, le=20)
