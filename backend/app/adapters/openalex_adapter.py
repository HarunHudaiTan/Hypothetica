"""
OpenAlex evidence adapter.
Searches the OpenAlex works catalog (see https://developers.openalex.org/).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import requests

from core import config
from app.models.paper import Paper
from .base_adapter import EvidenceAdapter

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# Applied to every search unless the caller passes filter= explicitly.
# has_abstract:true   — skip papers with no abstract (useless for scoring)
# is_retracted:false  — exclude retracted works
# type:article|preprint — journals + arXiv/bioRxiv style preprints only
OPENALEX_DEFAULT_FILTER = "has_abstract:true,is_retracted:false,type:article|preprint"


def _reconstruct_abstract_from_inverted_index(inv: Optional[Dict[str, List[int]]]) -> str:
    if not inv:
        return ""
    parts: List[tuple[int, str]] = []
    for word, positions in inv.items():
        for pos in positions:
            parts.append((pos, word))
    parts.sort(key=lambda x: x[0])
    return " ".join(w for _, w in parts)


def _openalex_work_id(work: Dict[str, Any]) -> str:
    url = work.get("id") or (work.get("ids") or {}).get("openalex", "")
    if not url:
        return ""
    return url.rsplit("/", 1)[-1]


def _first_pdf_url(work: Dict[str, Any]) -> Optional[str]:
    oa = work.get("best_oa_location") or {}
    if oa.get("pdf_url"):
        return oa["pdf_url"]
    pl = work.get("primary_location") or {}
    if pl.get("pdf_url"):
        return pl["pdf_url"]
    for loc in work.get("locations") or []:
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    # e.g. "oa_url": "https://arxiv.org/pdf/…" (not always on primary_location)
    oa_status = work.get("open_access") or {}
    oa_url = oa_status.get("oa_url")
    if isinstance(oa_url, str) and oa_url.startswith("http") and (
        "/pdf/" in oa_url or oa_url.lower().rstrip("/").endswith(".pdf")
    ):
        return oa_url
    return None


def _parse_arxiv_paper_id(raw: Any) -> Optional[str]:
    """Normalize ids.arxiv (URL or bare id) to an arXiv id usable in /abs/ and /pdf/ paths."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if "arxiv.org/abs/" in s:
        tail = s.split("arxiv.org/abs/", 1)[-1]
        return tail.split("?")[0].rstrip("/")
    if "arxiv.org/pdf/" in s:
        tail = s.split("arxiv.org/pdf/", 1)[-1]
        return tail.split("?")[0].rstrip("/").removesuffix(".pdf")
    if s.startswith("http"):
        return None
    if re.match(r"^[\w./+-]+$", s):
        return s
    return None


def _extract_arxiv_paper_id_from_work(work: Dict[str, Any]) -> Optional[str]:
    """
    arXiv id from OpenAlex work: ids.arxiv, or DOI 10.48550/arxiv.* (arXiv DataCite),
    or legacy ids.mag in rare cases.
    """
    ids = work.get("ids") or {}
    raw_arxiv = ids.get("arxiv")
    if raw_arxiv:
        pid = _parse_arxiv_paper_id(raw_arxiv)
        if pid:
            return pid
    doi = ids.get("doi")
    if isinstance(doi, str) and "10.48550/arxiv." in doi:
        return doi.split("10.48550/arxiv.", 1)[-1].split("?")[0].rstrip("/")
    if isinstance(doi, str) and "arxiv.org/abs/" in doi:
        return _parse_arxiv_paper_id(doi)
    return None


def _arxiv_abs_and_pdf(arxiv_paper_id: str) -> tuple[str, str]:
    return (
        f"https://arxiv.org/abs/{arxiv_paper_id}",
        f"https://arxiv.org/pdf/{arxiv_paper_id}.pdf",
    )


def _is_openalex_only_landing(u: str, work: Dict[str, Any]) -> bool:
    u = (u or "").strip()
    if not u:
        return True
    wid = (work.get("id") or "").strip()
    return bool(wid and u == wid) or (
        "openalex.org" in u and "/W" in u and u.startswith("https://")
    )


def _landing_url(work: Dict[str, Any]) -> str:
    pl = work.get("primary_location") or {}
    if pl.get("landing_page_url"):
        return pl["landing_page_url"]
    oa = work.get("best_oa_location") or {}
    if oa.get("landing_page_url"):
        return oa["landing_page_url"]
    return work.get("id", "") or ""


def _authors(work: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for a in work.get("authorships") or []:
        auth = a.get("author") or {}
        name = auth.get("display_name")
        if name:
            out.append(name)
    return out


def _concept_labels(work: Dict[str, Any], limit: int = 5) -> List[str]:
    concepts = work.get("concepts") or []
    scored = sorted(
        (c for c in concepts if c.get("display_name") and c.get("score", 0) is not None),
        key=lambda c: float(c.get("score", 0)),
        reverse=True,
    )
    return [c["display_name"] for c in scored[:limit]]


def _strip_env_value(raw: str | None) -> str:
    if not raw:
        return ""
    k = raw.strip()
    if len(k) >= 2 and k[0] in "'\"" and k[0] == k[-1]:
        k = k[1:-1].strip()
    return k.replace("\n", "").replace("\r", "")


class OpenAlexAdapter(EvidenceAdapter):
    """
    Academic works via the OpenAlex API (metadata + best OA PDF when available).
    """

    def __init__(self, api_key: str | None = None, mailto: str | None = None):
        self._api_key = _strip_env_value(
            api_key if api_key is not None else (config.OPENALEX_API_KEY or "")
        )
        self._mailto = _strip_env_value(
            mailto if mailto is not None else (config.OPENALEX_MAILTO or "")
        )

    @property
    def name(self) -> str:
        return "openalex"

    @property
    def description(self) -> str:
        return "OpenAlex scholarly works (journals, conferences, preprints, and more)"

    @property
    def display_name(self) -> str:
        return "OpenAlex"

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _work_to_dict(self, work: Dict[str, Any]) -> Dict[str, Any]:
        inv = work.get("abstract_inverted_index")
        abstract = _reconstruct_abstract_from_inverted_index(inv)
        pub = work.get("publication_date") or ""
        if not pub and work.get("publication_year"):
            pub = f"{int(work['publication_year'])}-01-01"
        oa_id = _openalex_work_id(work)
        ids = work.get("ids") or {}
        arxiv_pid = _extract_arxiv_paper_id_from_work(work)
        pdf_url = _first_pdf_url(work)
        if not pdf_url and arxiv_pid:
            _, pdf_url = _arxiv_abs_and_pdf(arxiv_pid)
        page_url = _landing_url(work)
        if arxiv_pid and _is_openalex_only_landing(page_url, work):
            page_url, _ = _arxiv_abs_and_pdf(arxiv_pid)
        return {
            "openalex_id": oa_id,
            "title": (work.get("display_name") or "").strip(),
            "abstract": abstract,
            "url": page_url,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_pid,
            "authors": _authors(work),
            "categories": _concept_labels(work),
            "published_date": pub or None,
            "doi": ids.get("doi"),
            "cited_by_count": work.get("cited_by_count"),
            "work_type": work.get("type"),
            "is_oa": (work.get("open_access") or {}).get("is_oa"),
            "relevance_score": work.get("relevance_score"),
        }

    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not self.is_available:
            logger.error("[OpenAlexAdapter] Missing OPENALEX_API_KEY")
            return []

        out: List[Dict[str, Any]] = []
        cursor: str | None = None
        cap = max(1, min(int(max_results), 1000))
        per_page_max = 200
        # Copy so we do not mutate caller kwargs
        extra = dict(kwargs)
        # Quality filter: only works with abstracts, non-retracted, article/preprint types.
        # Caller can override by passing filter= explicitly.
        if "filter" not in extra:
            extra["filter"] = OPENALEX_DEFAULT_FILTER

        while len(out) < cap:
            batch = min(per_page_max, cap - len(out))
            params: Dict[str, Any] = {
                "search": query,
                "per_page": batch,
                "api_key": self._api_key,
            }
            if self._mailto:
                params["mailto"] = self._mailto
            if cursor:
                params["cursor"] = cursor
            params.update(extra)

            logger.info(f"[OpenAlexAdapter] search={query[:80]!r} (batch {batch}, have {len(out)})")
            try:
                r = requests.get(OPENALEX_WORKS_URL, params=params, timeout=60)
                if r.status_code == 401:
                    logger.error(
                        "[OpenAlexAdapter] HTTP 401 Unauthorized — OpenAlex rejected the API key. "
                        "Create or rotate a key at https://openalex.org/settings/api and set OPENALEX_API_KEY in .env"
                    )
                    break
                if r.status_code == 403:
                    logger.error(
                        "[OpenAlexAdapter] HTTP 403 — request blocked (quota, policy, or bad key). Body: %s",
                        (r.text or "")[:500],
                    )
                    break
                r.raise_for_status()
                data = r.json()
            except requests.RequestException as e:
                logger.error(f"[OpenAlexAdapter] Request failed: {e}")
                break

            if not isinstance(data, dict):
                logger.error(f"[OpenAlexAdapter] Unexpected response type: {type(data)}")
                break

            page = data.get("results") or []
            for w in page:
                if len(out) >= cap:
                    break
                if not isinstance(w, dict):
                    continue
                if not _openalex_work_id(w):
                    continue
                out.append(self._work_to_dict(w))

            meta = data.get("meta") or {}
            cursor = meta.get("next_cursor")
            if not page or not cursor:
                break

        logger.info(f"[OpenAlexAdapter] Returned {len(out)} results")
        return out

    def convert_to_papers(
        self,
        results: List[Dict[str, Any]],
        limit: int | None = None,
    ) -> List[Paper]:
        # Do not cap here — the pipeline controls funnel size after dedup + ranking.
        cap = limit if limit is not None else len(results)
        papers: List[Paper] = []
        for i, pd in enumerate(results[:cap]):
            papers.append(
                Paper(
                    paper_id=f"paper_{i+1:03d}",
                    source="openalex",
                    source_id=pd.get("openalex_id", ""),
                    title=pd.get("title", ""),
                    abstract=pd.get("abstract", ""),
                    url=pd.get("url", ""),
                    pdf_url=pd.get("pdf_url"),
                    authors=pd.get("authors", []),
                    categories=pd.get("categories", []),
                    published_date=pd.get("published_date"),
                    metadata={
                        "doi": pd.get("doi"),
                        "openalex_id": pd.get("openalex_id", ""),
                        "arxiv_id": pd.get("arxiv_id"),
                        "cited_by_count": pd.get("cited_by_count"),
                        "type": pd.get("work_type"),
                        "is_oa": pd.get("is_oa"),
                        "relevance_score": pd.get("relevance_score"),
                    },
                )
            )
        return papers
