"""
Evidence adapter registry and discovery.
"""
from typing import Dict, List
from .base_adapter import EvidenceAdapter

# Adapter registry - populated by importing concrete adapters
_ADAPTERS: Dict[str, EvidenceAdapter] = {}


def register_adapter(adapter: EvidenceAdapter):
    """Register an evidence adapter."""
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> EvidenceAdapter:
    """Get an adapter by name."""
    return _ADAPTERS.get(name)


def get_all_adapters() -> Dict[str, EvidenceAdapter]:
    """Get all registered adapters."""
    return _ADAPTERS.copy()


def get_available_adapters() -> List[Dict[str, any]]:
    """Get list of available adapters with their metadata."""
    return [
        {
            "name": adapter.name,
            "description": adapter.description,
            "display_name": adapter.display_name,
            "evidence_noun_plural": adapter.evidence_noun_plural,
            "evidence_noun_singular": adapter.evidence_noun_singular,
            "available": adapter.is_available
        }
        for adapter in _ADAPTERS.values()
    ]


# Import and register concrete adapters
# Use try-except to handle missing dependencies gracefully
try:
    from .arxiv_adapter import ArxivAdapter
    register_adapter(ArxivAdapter())
except ImportError as e:
    import logging
    logging.warning(f"Failed to load ArxivAdapter: {e}")

try:
    from .patents_adapter import PatentsAdapter
    register_adapter(PatentsAdapter())
except ImportError as e:
    import logging
    logging.warning(f"Failed to load PatentsAdapter: {e}")

try:
    from .github_adapter import GitHubAdapter
    register_adapter(GitHubAdapter())
except ImportError as e:
    import logging
    logging.warning(f"Failed to load GitHubAdapter: {e}")

try:
    from .openalex_adapter import OpenAlexAdapter
    register_adapter(OpenAlexAdapter())
except ImportError as e:
    import logging
    logging.warning(f"Failed to load OpenAlexAdapter: {e}")
