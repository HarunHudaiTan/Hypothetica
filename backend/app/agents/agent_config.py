"""
Agent-specific configuration parameters for Hypothetica Research Originality System.
All agent tuning parameters (temperature, top_p, top_k) are centralized here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# FOLLOW-UP AGENT CONFIG
# =============================================================================
FOLLOWUP_TEMPERATURE = 0.7
FOLLOWUP_TOP_P = 0.9
FOLLOWUP_TOP_K = 40

# =============================================================================
# KEYWORD AGENT CONFIG (legacy - not currently used)
# =============================================================================
KEYWORD_TEMPERATURE = 0.3
KEYWORD_TOP_P = 0.85
KEYWORD_TOP_K = 40

# =============================================================================
# LAYER 1 AGENT CONFIG (per-paper analysis)
# =============================================================================
LAYER1_TEMPERATURE = 0.2
LAYER1_TOP_P = 0.8
LAYER1_TOP_K = 30

# =============================================================================
# LAYER 2 AGENT CONFIG (summary generation only)
# =============================================================================
LAYER2_TEMPERATURE = 0.5
LAYER2_TOP_P = 0.9
LAYER2_TOP_K = 40

# =============================================================================
# REALITY CHECK AGENT CONFIG
# =============================================================================
REALITY_CHECK_TEMPERATURE = 0.3
REALITY_CHECK_TOP_P = 0.8
REALITY_CHECK_TOP_K = 40

# =============================================================================
# QUERY VARIANT AGENT CONFIG
# =============================================================================
QUERY_VARIANT_TEMPERATURE = 0.4
QUERY_VARIANT_TOP_P = 0.85
QUERY_VARIANT_TOP_K = 40

# =============================================================================
# RELEVANT PAPER SELECTOR AGENT CONFIG
# =============================================================================
PAPER_SELECTOR_TEMPERATURE = 0.3
PAPER_SELECTOR_TOP_P = 0.8
PAPER_SELECTOR_TOP_K = 40

# =============================================================================
# HEADING SELECTOR AGENT CONFIG (not currently used)
# =============================================================================
HEADING_SELECTOR_TEMPERATURE = 0.4
HEADING_SELECTOR_TOP_P = 0.85
HEADING_SELECTOR_TOP_K = 40
