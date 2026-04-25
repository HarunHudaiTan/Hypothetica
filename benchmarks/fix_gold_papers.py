"""
fix_gold_papers.py
==================
For every mismatched gold paper, searches arXiv by the CLAIMED title to find
the real arXiv ID. Search order: strict title phrase → keyword AND (3–4 terms)
→ optional author+keyword if `author` / `authors` exists on the gold_paper.
Writes fixed JSON, fix_report.json, and manual_review.json (all not auto-fixed).
"""

import json
import re
import time
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# Short common words + arXiv-title filler — not used for keyword AND queries
_STOPWORDS = frozenset({
    "a", "an", "the", "for", "and", "or", "with", "via", "using", "from", "to", "of", "in",
    "on", "at", "by", "as", "is", "are", "be", "been", "being", "we", "our", "their", "its",
    "this", "that", "these", "those", "not", "no", "but", "if", "when", "how", "what", "has",
    "have", "can", "may", "new", "based", "toward", "towards", "into", "over", "under",
})

# https://info.arxiv.org/help/api/user-manual.html — identify yourself; avoid generic agents.
ARXIV_SEARCH = "https://export.arxiv.org/api/query"
HTTP_HEADERS = {
    "User-Agent": "Hypothetica2-benchmark/1.0 (gold-paper fix; respectful API use)",
}
DEFAULT_DELAY = 4  # seconds between searches — arXiv asks for max ~1 request per 3s


def arxiv_api_get(
    params: dict, label: str
) -> tuple[requests.Response | None, str | None]:
    """
    GET with retries. Returns (response, None) on success, or (None, reason) on hard failure.
    `reason` is 'rate_limited' | 'http_error' | 'network_error'.
    """
    for attempt in range(10):
        try:
            resp = requests.get(
                ARXIV_SEARCH, params=params, headers=HTTP_HEADERS, timeout=45
            )
        except requests.RequestException as e:
            print(f"    [!] Network error ({label}): {e}")
            if attempt < 9:
                wait = 10 + 10 * attempt
                print(f"    … retry in {wait}s")
                time.sleep(wait)
            else:
                return None, "network_error"
            continue
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After", "").strip()
            if ra.isdigit():
                wait = int(ra)
            else:
                wait = min(30 + 25 * attempt, 180)
            print(
                f"    [!] arXiv rate limit (429) — waiting {wait}s before retry "
                f"({attempt + 1}/10)"
            )
            time.sleep(wait)
            continue
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"    [!] HTTP error ({label}): {e}")
            return None, "http_error"
        return resp, None
    print(
        "    [!] Gave up after repeated 429s — will mark as RATE_LIMITED (re-run with "
        "higher --delay or after a break)."
    )
    return None, "rate_limited"


def _word_overlap(claimed: str, found: str) -> float:
    def words(s: str) -> set:
        t = s.lower().replace(":", " ").replace("-", " ")
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return {w for w in t.split() if len(w) > 0}

    cw, fw = words(claimed), words(found)
    if not cw:
        return 0.0
    return len(cw & fw) / max(len(cw), 1)


def _feed_hits_scored(claimed_title: str, resp: requests.Response) -> list[dict]:
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    hits: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        entry_id_el = entry.find("atom:id", ns)
        title_el = entry.find("atom:title", ns)
        if entry_id_el is None or title_el is None:
            continue
        raw_id = entry_id_el.text or ""
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0].strip()
        real_title = (title_el.text or "").strip().replace("\n", " ")
        ov = _word_overlap(claimed_title, real_title)
        hits.append({"arxiv_id": arxiv_id, "real_title": real_title, "overlap": ov})
    hits.sort(key=lambda h: -h["overlap"])
    return hits


def _significant_keywords(title: str, n: int = 4) -> list[str]:
    """3–4 content words: long tokens, drop stopwords, keep names like PruneFL."""
    raw = re.sub(r"[\$\{\}\\]", " ", title)
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]+|[0-9]+[a-z0-9.\-]*", raw)
    scored: list[str] = []
    for t in toks:
        low = t.lower()
        if len(low) < 2 or low in _STOPWORDS:
            continue
        if low not in {x.lower() for x in scored}:
            scored.append(t)
    scored.sort(key=lambda x: len(x), reverse=True)
    uniq: list[str] = []
    seen: set = set()
    for t in scored:
        if t.lower() not in seen:
            seen.add(t.lower())
            uniq.append(t)
    if len(uniq) < 2:
        for t in toks:
            low = t.lower()
            if low in _STOPWORDS or len(low) < 2:
                continue
            if low not in seen:
                seen.add(low)
                uniq.append(t)
    return uniq[:n]


def _arxiv_tok(s: str) -> str:
    t = s.strip()
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9\-\.]*$", t):
        return t
    return f'"{t}"'


def _author_lastname(author: str) -> str | None:
    s = author.strip()
    if not s:
        return None
    part = s.split(",")[0].strip() if "," in s else s
    parts = re.split(r"\s+", part)
    return parts[-1] if parts else None


def _build_keyword_query(keywords: list[str]) -> str:
    toks = [_arxiv_tok(k) for k in keywords if k.strip()]
    if not toks:
        return ""
    return " AND ".join(f"all:{t}" for t in toks)


def _hit_to_outcome(
    hit: dict | None, search_tier: str
) -> dict:
    if hit is None:
        return {"outcome": "no_hits", "search_tier": search_tier}
    base = {**hit, "search_tier": search_tier}
    if hit["overlap"] < 0.35:
        base["outcome"] = "low_overlap"
    else:
        base["outcome"] = "ok"
    return base


def _strict_search(claimed_title: str) -> dict:
    safe = claimed_title.replace('"', " ")
    params = {
        "search_query": f'ti:"{safe}"',
        "max_results": 5,
        "sortBy": "relevance",
    }
    label = f'strict:{claimed_title[:45]}'
    return _search_query(params, label, claimed_title, "strict")


def _search_query(
    params: dict,
    label: str,
    claimed_for_overlap: str,
    search_tier: str,
) -> dict:
    """One API call → outcome dict (ok / low_overlap / no_hits / api_error)."""
    resp, err = arxiv_api_get(params, label)
    if resp is None:
        return {"outcome": "api_error", "reason": err or "http_error", "search_tier": search_tier}
    hits = _feed_hits_scored(claimed_for_overlap, resp)
    if not hits:
        return {"outcome": "no_hits", "search_tier": search_tier}
    best = hits[0]
    return _hit_to_outcome(best, search_tier)


def _keyword_fallback(claimed_title: str, delay_sec: float) -> dict:
    kws = _significant_keywords(claimed_title, n=4)
    if len(kws) < 2:
        kws = _significant_keywords(claimed_title, n=6)
    last: dict = {"outcome": "no_hits", "search_tier": "keyword"}
    seen_q: set[str] = set()
    for take in (4, 3, 2):
        chunk = kws[:take] if len(kws) >= take else kws
        if len(chunk) < 2:
            continue
        q = _build_keyword_query(chunk)
        if not q or q in seen_q:
            continue
        seen_q.add(q)
        params = {
            "search_query": q,
            "max_results": 10,
            "sortBy": "relevance",
        }
        label = f"kw:{chunk}"
        r = _search_query(params, label, claimed_title, "keyword")
        r["keyword_query_tried"] = q
        last = r
        time.sleep(delay_sec)
        if r.get("outcome") not in ("no_hits",):
            return r
    last["search_tier"] = "keyword"
    return last


def _author_keyword_fallback(claimed_title: str, author: str) -> dict:
    last = _author_lastname(author)
    if not last:
        return {"outcome": "no_hits", "search_tier": "author_keyword"}
    kws = _significant_keywords(claimed_title, n=2)[:2]
    if not kws:
        return {"outcome": "no_hits", "search_tier": "author_keyword"}
    toks = [f"au:{_arxiv_tok(last)}"] + [f"all:{_arxiv_tok(k)}" for k in kws]
    q = " AND ".join(toks)
    params = {
        "search_query": q,
        "max_results": 8,
        "sortBy": "relevance",
    }
    r = _search_query(params, f"au+kw:{last}", claimed_title, "author_keyword")
    r["author_keyword_query_tried"] = q
    return r


def _gold_paper_author(gp: dict) -> str | None:
    a = gp.get("author")
    if isinstance(a, str) and a.strip():
        return a
    a = gp.get("authors")
    if isinstance(a, str) and a.strip():
        return a
    if isinstance(a, list) and a:
        if isinstance(a[0], str):
            return a[0]
        if isinstance(a[0], dict) and a[0].get("name"):
            return str(a[0]["name"])
    return None


def search_for_paper(claimed_title: str, gp: dict, delay_sec: float) -> dict:
    """
    Strict `ti:"..."` first; on `no_hits` try keyword AND; then author+keyword
    if `author` / `authors` is present on the gold_paper. Sleeps `delay_sec`
    after each arXiv HTTP call.
    """
    r = _strict_search(claimed_title)
    time.sleep(delay_sec)

    if r.get("outcome") == "api_error":
        return r
    if r.get("outcome") in ("ok", "low_overlap"):
        return r

    if r.get("outcome") == "no_hits":
        print("    → keyword fallback (free-text AND)…")
        r2 = _keyword_fallback(claimed_title, delay_sec)
        r2["attempts"] = ["strict", "keyword"]
        if r2.get("outcome") == "api_error":
            return r2
        if r2.get("outcome") in ("ok", "low_overlap"):
            return r2

        author = _gold_paper_author(gp)
        if author:
            print(
                f"    → author + keyword (au:… AND all:…) — author: {author[:50]}…"
            )
            r3 = _author_keyword_fallback(claimed_title, author)
            time.sleep(delay_sec)
            r3["attempts"] = ["strict", "keyword", "author_keyword"]
            return r3
        return r2

    return r


def _apply_search_result(
    case: dict,
    gp: dict,
    claimed_title: str,
    old_id: str,
    r: dict,
    fix_log: list,
    failed_fixes: list,
) -> None:
    """Mutates gp, fix_log, failed_fixes based on a search result dict."""
    outcome = r.get("outcome")
    if outcome == "ok":
        new_id = r["arxiv_id"]
        print(f"    ✓ Found: {new_id} — '{r['real_title'][:60]}' (overlap={r['overlap']:.2f})")
        rec = {
            "case_id": case["id"],
            "claimed_title": claimed_title,
            "old_id": old_id,
            "new_id": new_id,
            "found_title": r["real_title"],
            "overlap": r["overlap"],
        }
        if r.get("search_tier"):
            rec["search_tier"] = r["search_tier"]
        if r.get("keyword_query_tried"):
            rec["keyword_query_tried"] = r["keyword_query_tried"]
        if r.get("author_keyword_query_tried"):
            rec["author_keyword_query_tried"] = r["author_keyword_query_tried"]
        fix_log.append(rec)
        gp["arxiv_id"] = new_id
        gp["title"] = r["real_title"]
        gp.pop("_verification", None)
        gp.pop("_fix_status", None)
        return

    if outcome == "low_overlap":
        print(f"    ✗ Low title overlap (overlap={r['overlap']:.2f}) — not auto-applying")
        row = {
            "case_id": case["id"],
            "claimed_title": claimed_title,
            "old_id": old_id,
            "failure": "low_overlap",
            "search_result": {
                "arxiv_id": r["arxiv_id"],
                "real_title": r["real_title"],
                "overlap": r["overlap"],
            },
        }
        for k in (
            "attempts",
            "keyword_query_tried",
            "search_tier",
        ):
            if r.get(k) is not None:
                row[k] = r[k]
        failed_fixes.append(row)
        gp["_fix_status"] = "NEEDS_MANUAL_FIX"
        gp["_search_suggestion"] = f"{r['arxiv_id']} — {r['real_title']}"
        return

    if outcome == "no_hits":
        at = r.get("attempts", ["strict"])
        print(f"    ✗ No arXiv hit after: {', '.join(at)} (see manual_review.json)")
        row = {
            "case_id": case["id"],
            "claimed_title": claimed_title,
            "old_id": old_id,
            "failure": "no_hits",
            "search_result": None,
        }
        for k in (
            "attempts",
            "keyword_query_tried",
            "author_keyword_query_tried",
            "search_tier",
        ):
            if r.get(k) is not None:
                row[k] = r[k]
        failed_fixes.append(row)
        gp["_fix_status"] = "NEEDS_MANUAL_FIX"
        return

    if outcome == "api_error":
        reason = r.get("reason", "http_error")
        if reason == "rate_limited":
            print("    ✗ arXiv still rate-limited (not a missing paper). Re-run with --delay 10+ or after a long break.")
        else:
            print(f"    ✗ API error ({reason})")
        failed_fixes.append(
            {
                "case_id": case["id"],
                "claimed_title": claimed_title,
                "old_id": old_id,
                "failure": reason,
                "search_result": None,
            }
        )
        gp["_fix_status"] = "RATE_LIMITED_RETRY" if reason == "rate_limited" else "NEEDS_MANUAL_FIX"
        return


def fix_dataset(benchmark_path: str, report_path: str, delay_sec: float) -> None:
    benchmark = json.loads(Path(benchmark_path).read_text())
    report = json.loads(Path(report_path).read_text())

    # Build lookup: (case_id, claimed_title) -> needs fixing
    mismatched = {
        (r["case_id"], r["claimed_title"]): r
        for r in report["mismatched_ids"]
    }

    fix_log: list = []  # what we changed
    failed_fixes: list = []  # what we could not auto-fix
    rate_limited_titles: list[tuple[dict, dict, str, str]] = []  # case, gp, title, old_id

    print(f"\n{'='*60}")
    print(
        f"Fixing {len(mismatched)} mismatched gold papers (delay {delay_sec}s between requests)"
    )
    print(f"{'='*60}\n")

    for case in benchmark["cases"]:
        for gp in case.get("gold_papers", []):
            key = (case["id"], gp["title"])
            if key not in mismatched:
                continue  # already OK, skip

            old_id = gp["arxiv_id"]
            claimed_title = gp["title"]
            print(f"  [{case['id']}] Searching: '{claimed_title[:70]}'")

            r = search_for_paper(claimed_title, gp, delay_sec)

            if r.get("outcome") == "api_error" and r.get("reason") == "rate_limited":
                print(
                    "    ⏳ Rate limited (429) — queued for cool-down retry (not ‘no results’ on arXiv)"
                )
                gp["_fix_status"] = "RATE_LIMITED_RETRY"
                rate_limited_titles.append((case, gp, claimed_title, old_id))
                continue

            _apply_search_result(
                case, gp, claimed_title, old_id, r, fix_log, failed_fixes
            )

    if rate_limited_titles:
        print(f"\n{'='*60}")
        print(
            f"⏳ Cool-down 90s, then one retry for {len(rate_limited_titles)} title(s) that hit 429"
        )
        print(f"{'='*60}")
        time.sleep(90)
        retry_delay = max(delay_sec, 8.0)
        for case, gp, claimed_title, old_id in rate_limited_titles:
            key = (case["id"], claimed_title)
            if key not in mismatched:
                continue
            print(f"  [retry {case['id']}] '{claimed_title[:70]}'")
            r = search_for_paper(claimed_title, gp, retry_delay)
            _apply_search_result(
                case, gp, claimed_title, old_id, r, fix_log, failed_fixes
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FIX SUMMARY")
    print(f"{'='*60}")
    print(f"  ✅ Fixed automatically : {len(fix_log)}")
    print(f"  ⚠️  Not auto-fixed      : {len(failed_fixes)}")

    if failed_fixes:
        print(f"\n⚠️  STILL NEED ATTENTION (by failure type):")
        for f in failed_fixes:
            fail = f.get("failure", "unknown")
            print(f"   [{f['case_id']}] {fail}: '{f['claimed_title'][:70]}'")
            if f.get("search_result"):
                sr = f["search_result"]
                print(
                    f"       Best guess: {sr['arxiv_id']} — '{sr['real_title'][:60]}' "
                    f"(overlap={sr.get('overlap', 0):.2f})"
                )
            elif fail == "rate_limited":
                print("       arXiv still throttling — re-run the script after a long break; use --delay 10+")
            elif fail in ("http_error", "network_error"):
                print("       Network/server error — re-run; not a ‘missing paper’ on arXiv")
            elif fail == "no_hits":
                print("       No arXiv API hit for exact title phrase — try manual search or shorter query")

    # ── Save outputs ──────────────────────────────────────────────────────────
    base = Path(benchmark_path).parent

    in_stem = Path(benchmark_path).stem
    fixed_path = base / f"{in_stem}_fixed.json"
    # Strip any leftover _verification fields from OK entries
    for case in benchmark["cases"]:
        for gp in case.get("gold_papers", []):
            gp.pop("_verification", None)
    fixed_path.write_text(json.dumps(benchmark, indent=2))
    print(f"\n📄 Fixed dataset saved  → {fixed_path}")

    fix_report_path = base / "fix_report.json"
    fix_report_path.write_text(json.dumps({
        "fixed": fix_log,
        "needs_manual_fix": failed_fixes,
    }, indent=2))
    print(f"📄 Fix report saved     → {fix_report_path}")

    manual_path = base / "manual_review.json"
    manual_path.write_text(
        json.dumps(
            {
                "description": (
                    "Items the script could not auto-fix. For failure=rate_limited, "
                    "re-run with a higher --delay. For no_hits / low_overlap, look up the "
                    "paper on arXiv and set the correct arxiv_id (and title) in the benchmark JSON."
                ),
                "count": len(failed_fixes),
                "entries": failed_fixes,
            },
            indent=2,
        )
    )
    print(f"📄 Manual review list     → {manual_path}")
    print("\nNext: run verify_gold_papers.py on the fixed dataset to confirm.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default=str(
            Path(__file__).resolve().parent
            / "datasets"
            / "hypothetica_benchmark_arxiv_gold_clean.json"
        ),
    )
    parser.add_argument(
        "--report",
        default="/Users/alpmalkoc/Desktop/Hypothetica2/benchmarks/datasets/verification_report.json",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds to sleep after each arXiv request (default {DEFAULT_DELAY}; increase if you still see 429).",
    )
    args = parser.parse_args()
    fix_dataset(args.benchmark, args.report, args.delay)