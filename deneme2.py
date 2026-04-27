import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# Same pattern as backend/core/config.py, but the final .env overwrites the shell so
# a stale or wrong exported SEMANTIC_SCHOLAR_API_KEY does not block the file (403 common).
_root = Path(__file__).resolve().parent
load_dotenv(_root / "envfiles" / ".env", override=True)
load_dotenv(_root / ".env", override=True)  # root wins for duplicate keys


def _normalize_api_key(raw: str) -> str:
    k = (raw or "").strip()
    if len(k) >= 2 and k[0] in "\"'" and k[0] == k[-1]:
        k = k[1:-1].strip()
    k = k.replace("\n", "").replace("\r", "")
    return k


S2 = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = _normalize_api_key(
    os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
)
if not SEMANTIC_SCHOLAR_API_KEY:
    print(
        "Set SEMANTIC_SCHOLAR_API_KEY in your .env file (project root or envfiles/.env).",
        file=sys.stderr,
    )
    sys.exit(1)


def _s2_headers() -> dict[str, str]:
    return {
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        "User-Agent": "Hypothetica2-deneme/0.1 (S2 API)",
        "Accept": "application/json",
    }


def _s2_get_json(url: str) -> dict:
    """GET JSON; on HTTP errors print the S2 message body and re-raise."""
    req = urllib.request.Request(url, headers=_s2_headers())
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode(errors="replace")
        except OSError as read_err:  # pragma: no cover
            body = f"(could not read body: {read_err})"
        print(f"HTTP {e.code} {e.reason!r} — {url}\n{body}\n", file=sys.stderr)
        if e.code == 403:
            print(
                "403 = auth rejected. Copy a new key from "
                "https://www.semanticscholar.org/product/api — or run "
                "'unset SEMANTIC_SCHOLAR_API_KEY' if the shell has an old value "
                "('.env' is loaded with override).",
                file=sys.stderr,
            )
        raise


def s2_paper(arxiv_id: str) -> dict:
    """GET /graph/v1/paper/ARXIV:{id} — https://api.semanticscholar.org/api-docs/"""
    q = urllib.parse.urlencode(
        {
            "fields": "title,abstract,year,authors,authors.name,citationCount,venue,"
            "externalIds,url,publicationDate,fieldsOfStudy"
        }
    )
    url = f"{S2}/paper/ARXIV:{arxiv_id}?{q}"
    return _s2_get_json(url)


def s2_paper_search(query: str, limit: int = 3) -> dict:
    """GET /graph/v1/paper/search?query=..."""
    q = urllib.parse.urlencode(
        {
            "query": query,
            "limit": str(limit),
            "fields": "title,year,authors.name,citationCount,externalIds",
        }
    )
    url = f"{S2}/paper/search?{q}"
    return _s2_get_json(url)


def main() -> None:
    print("--- paper/ARXIV:1512.03385 (Deep Residual Learning) ---\n")
    try:
        print(json.dumps(s2_paper("1512.03385"), indent=2, ensure_ascii=False))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
        print("Request failed:", e, file=sys.stderr)
        raise SystemExit(1) from e

    print("\n--- paper/search (query=ResNet, limit=3) ---\n")
    try:
        print(json.dumps(s2_paper_search("ResNet", limit=3), indent=2, ensure_ascii=False))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
        print("Request failed:", e, file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
