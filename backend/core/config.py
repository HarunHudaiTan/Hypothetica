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
load_dotenv(_root / ".env")  # Fallback for local development

# =============================================================================
# API KEYS
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Supabase (for persisting analysis queries)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jqdjzvnqvkwyaiqednyf.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

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
# Sentence-level classification
HIGH_OVERLAP_THRESHOLD = 0.7    # >= this = RED (low originality)
MEDIUM_OVERLAP_THRESHOLD = 0.4  # >= this = YELLOW (medium originality)
# Below MEDIUM_OVERLAP_THRESHOLD = GREEN (high originality)

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
CRITERIA_WEIGHTS = {
    "problem": 0.3,
    "method": 0.3,
    "domain": 0.2,
    "contribution": 0.2,
}

SENTENCE_OVERLAP_TOP_K = 2       # Top-K scores to average per sentence (UI annotations)
CRITERIA_MAX_WEIGHT = 0.6        # Weight given to max score in criteria aggregation (UI display)
OVERLAP_CURVE_POWER = 1.5        # Exponent for non-linear overlap→originality mapping

# Paper-threat-based scoring (Layer 2)
PAPER_THREAT_MAX_WEIGHT = 0.5    # In per-paper threat: weight for max criterion vs weighted mean
GLOBAL_THREAT_MAX_WEIGHT = 0.7   # In global overlap: weight for most threatening paper vs mean

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
# RAG CONFIGURATION
# =============================================================================
RAG_TOP_K = 5  # Number of chunks to retrieve per query

# =============================================================================
# UI CONFIGURATION
# =============================================================================
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds between progress updates

