from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BenchmarkPreset(str, Enum):
    patents = "patents"
    openalex = "openalex"
    github = "github"


class BenchmarkRunRequest(BaseModel):
    preset: BenchmarkPreset
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Run only the first N cases from the dataset.",
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="Override JSON path, relative to repo root or absolute under repo root.",
    )
    persist_supabase: bool = True
    job_timeout_seconds: int = Field(default=900, ge=60, le=7200)


class BenchmarkRunResponse(BaseModel):
    run_id: str
    status: str
