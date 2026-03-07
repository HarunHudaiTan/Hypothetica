"""
Central configuration for Hypothetica Research Originality System.
Global settings, API keys, models, pipeline parameters, and thresholds.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
EMBEDDING_DEVICE = "mps"  # Use "cuda" for NVIDIA, "cpu" for fallback

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
# CHROMADB CONFIGURATION
# =============================================================================
CHROMA_COLLECTION_NAME = "paper_chunks"
CHROMA_PERSIST_DIR = None  # None = in-memory for demo

# =============================================================================
# COST TRACKING (Gemini 2.5 Flash pricing per 1M tokens)
# =============================================================================
INPUT_TOKEN_PRICE = 0.075   # $0.075 per 1M input tokens
OUTPUT_TOKEN_PRICE = 0.30   # $0.30 per 1M output tokens

# =============================================================================
# RAG CONFIGURATION
# =============================================================================
RAG_TOP_K = 5  # Number of chunks to retrieve per query

# =============================================================================
# UI CONFIGURATION
# =============================================================================
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds between progress updates

