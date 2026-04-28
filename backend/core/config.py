"""
Central configuration for Hypothetica Research Originality System.
Global settings, API keys, models, pipeline parameters, and thresholds.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load .env: try envfiles/.env first (Docker), then project root .env (local dev)
_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / "envfiles" / ".env")
# Local .env should win over any duplicate keys in envfiles/ (default override=False is wrong for that)
load_dotenv(_root / ".env", override=True)

# =============================================================================
# API KEYS
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Supabase (for persisting analysis queries)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jqdjzvnqvkwyaiqednyf.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# GitHub
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# SerpApi for Google Patents
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# OpenAlex (https://developers.openalex.org/) — optional mailto improves rate limits
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
def _detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

EMBEDDING_DEVICE = _detect_device()

# =============================================================================
# PIPELINE PARAMETERS
# =============================================================================
# Enhanced ArXiv Search (High-Recall Retrieval)
PAPERS_PER_QUERY_VARIANT = 150  # Papers to fetch per query variant
# Raw hits retained per query variant after search (AdapterService, non-OpenAlex).
# Fetch size is max(this, papers_per_query // 4) so the API returns enough rows.
PAPERS_PER_VARIANT_CONVERSION = 40
EMBEDDING_TOPK = 100  # Candidates from embedding search
RERANK_TOPK = 20  # Papers after cross-encoder reranking
MAX_PAPERS_TO_ANALYZE = 5  # Final papers for detailed analysis

# Legacy (backward compatibility)
NUM_KEYWORDS = 7
PAPERS_PER_KEYWORD = 10

# Chunking
MAX_CHUNK_SIZE = 512  # tokens approximately (characters / 4)
CHUNK_OVERLAP = 50    # characters overlap between chunks
MIN_CHUNK_SIZE = 100  # minimum characters for a valid chunk

# Section Quality Thresholds
MIN_SECTION_LENGTH = 200  # minimum characters for a meaningful section
ABSTRACT_SIMILARITY_THRESHOLD = 0.3  # flag sections below this similarity to abstract

# =============================================================================
# ORIGINALITY THRESHOLDS
# =============================================================================
# Sentence-level classification (legacy thresholds, kept for fallback)
HIGH_OVERLAP_THRESHOLD = 0.7    # >= this = RED (low originality)
MEDIUM_OVERLAP_THRESHOLD = 0.4  # >= this = YELLOW (medium originality)
# Below MEDIUM_OVERLAP_THRESHOLD = GREEN (high originality)

# Per-criterion sentence labeling (Likert scale)
SENTENCE_CRITERION_RED_MIN = 4    # Likert 4 or 5 → RED
SENTENCE_CRITERION_YELLOW_MIN = 3 # Likert 3 → YELLOW
# Below SENTENCE_CRITERION_YELLOW_MIN → not shown (filtered out)

# Global score ranges (for display)
SCORE_RED_MAX = 40      # 0-40 = low originality
SCORE_YELLOW_MAX = 70   # 40-70 = medium originality
# 70-100 = high originality

# =============================================================================
# LIKERT SCALE & AGGREGATION
# =============================================================================
# 5-point Likert → 0-1 float mapping
LIKERT_TO_FLOAT = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}

# Criteria weights for computing overall overlap score
# Recalibrated 2026-04-19 against arXiv benchmark (40 cases).
# Rationale: contribution is the strongest discriminator (Δ=+0.30 between
# novel and already_exists), method second (Δ=+0.23). Problem and domain are
# kept as context signals with low weights — domain is near-constant because
# retrieval already restricts to the same field, so it carries little info.
CRITERIA_WEIGHTS = {
    "problem":      0.15,    # context (low — overlaps with method/contribution)
    "method":       0.30,    # novelty signal
    "domain":       0.10,    # context (retrieval already guarantees same field)
    "contribution": 0.45,    # strongest novelty signal
}

SENTENCE_OVERLAP_TOP_K = 2       # Top-K scores to average per sentence (UI annotations)
CRITERIA_MAX_WEIGHT = 0.6        # Weight given to max score in criteria aggregation (UI display)
# originality = (1 - global_similarity ** OVERLAP_CURVE_POWER) * 100
# Power=1 is linear. Power>1 convex-ifies low similarities (inflates “novel” when mean was used).
# Power=2 with mean(global) let irrelevant papers dilute one real match; we now use max(global).
OVERLAP_CURVE_POWER = 1.5

# Layer 2 aggregation policy
# Per-paper similarity = weighted mean of the 4 criteria (no max within paper).
# Global similarity   = max over analyzed papers (best match wins — mean diluted real overlap).

# Categorical guardrails
GUARDRAIL_CRITICAL_FLOOR = 0.65  # Min overlap when any criterion = Likert 5 (1.0)
GUARDRAIL_HIGH_COUNT = 2         # Number of criteria >= Likert 4 to trigger high guardrail
GUARDRAIL_HIGH_FLOOR = 0.50      # Min overlap when GUARDRAIL_HIGH_COUNT criteria >= Likert 4

# =============================================================================
# CHROMADB CONFIGURATION
# =============================================================================
CHROMA_COLLECTION_NAME = "paper_chunks"
CHROMA_PERSIST_DIR = None  # None = in-memory for demo

# =============================================================================
# COST TRACKING (Gemini 2.5 Flash pricing per 1M tokens)
# See https://ai.google.dev/gemini-api/docs/pricing for current rates.
# =============================================================================
INPUT_TOKEN_PRICE = 0.30    # $0.30 per 1M input tokens
OUTPUT_TOKEN_PRICE = 2.50   # $2.50 per 1M output tokens

# =============================================================================
# GITHUB SEARCH CONFIGURATION
# =============================================================================
GITHUB_RESULTS_PER_QUERY = 30       # Results per query from GitHub API (increased)
GITHUB_MIN_STARS = 10               # Filter out abandoned/no-signal repositories
GITHUB_MIN_PUSH_YEAR = 2021         # Prefer actively maintained repositories
GITHUB_TOP_PER_QUERY = 8            # Keep diversity, but bias toward higher quality
GITHUB_MAX_REPOS_TO_ANALYZE = 24    # Smaller, cleaner candidate pool
GITHUB_README_PREVIEW_CHARS = 2000  # First N chars of README for LLM analysis
GITHUB_MIN_README_CHARS = 400       # Basic documentation quality gate
GITHUB_MAX_REPOS_FOR_RELEVANCE = 12 # Cap expensive LLM repo assessments
GITHUB_QUERY_MIN_STARS = 10         # Add stars:>N directly into GitHub search query

# =============================================================================
# RAG CONFIGURATION
# =============================================================================
RAG_TOP_K = 5  # Number of chunks to retrieve per query

# =============================================================================
# UI CONFIGURATION
# =============================================================================
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds between progress updates
