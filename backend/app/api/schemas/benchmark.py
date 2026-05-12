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
    table_name: str = Field(
        ...,
        min_length=1,
        description="Supabase table to insert benchmark rows into (e.g. 'benchmark2', 'openalex_v3_benchmark').",
    )


class BenchmarkRunResponse(BaseModel):
    run_id: str
    status: str
