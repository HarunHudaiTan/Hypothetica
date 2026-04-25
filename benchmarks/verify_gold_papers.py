"""
verify_gold_papers.py
=====================
Checks every gold paper arXiv ID in the benchmark dataset against the real
arXiv API. Produces a verification report and a cleaned dataset with dead IDs
flagged (but not yet removed — you review first).

Usage (from repository root):
    pip install requests
    python benchmarks/verify_gold_papers.py

Or from the benchmarks directory:
    cd benchmarks && python verify_gold_papers.py

Custom input file:
    python benchmarks/verify_gold_papers.py --input path/to/benchmark.json
"""

import json
import time
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
BATCH_SIZE = 20          # arXiv allows up to 100 IDs per query; stay conservative
DELAY_BETWEEN_BATCHES = 3  # seconds — be polite to arXiv servers

@dataclass
class VerificationResult:
    arxiv_id: str
    claimed_title: str
    case_id: str
    relevance: str
    exists: bool
    real_title: Optional[str]
    mismatch: bool          # exists but title is very different
    note: str


def batch_check(ids: list[str]) -> dict[str, dict]:
    """
    Query arXiv API for a batch of IDs.
    Returns dict: arxiv_id -> {"exists": bool, "real_title": str|None}
    """
    id_str = ",".join(ids)
    try:
        resp = requests.get(
            ARXIV_API,
            params={"id_list": id_str, "max_results": len(ids)},
            timeout=15
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] API error: {e}")
        return {id_: {"exists": False, "real_title": None} for id_ in ids}

    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    found = {}
    for entry in root.findall("atom:entry", ns):
        # arXiv returns an error entry if the ID doesn't exist
        entry_id_el = entry.find("atom:id", ns)
        if entry_id_el is None:
            continue
        entry_id = entry_id_el.text or ""

        # Check for "not found" error entries
        if "abs/ERROR" in entry_id or "abs/error" in entry_id:
            continue

        # Extract the clean ID from the URL e.g. http://arxiv.org/abs/1706.03762v5
        clean_id = entry_id.split("/abs/")[-1].split("v")[0].strip()

        title_el = entry.find("atom:title", ns)
        real_title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else None
        found[clean_id] = {"exists": True, "real_title": real_title}

    # Any ID not in found didn't come back → doesn't exist
    result = {}
    for id_ in ids:
        # Normalize: strip version suffix if user included one
        clean = id_.split("v")[0].strip()
        if clean in found:
            result[id_] = found[clean]
        else:
            result[id_] = {"exists": False, "real_title": None}

    return result


def title_mismatch(claimed: str, real: str, threshold: float = 0.4) -> bool:
    """
    Rough word-overlap check. Returns True if titles share fewer than
    `threshold` fraction of words (case-insensitive).
    """
    def words(s):
        return set(s.lower().replace(":", "").replace("-", " ").split())

    c = words(claimed)
    r = words(real)
    if not c or not r:
        return True
    overlap = len(c & r) / max(len(c), len(r))
    return overlap < threshold


def verify_dataset(input_path: str) -> None:
    data = json.loads(Path(input_path).read_text())
    cases = data["cases"]

    # Collect all (case_id, arxiv_id, claimed_title, relevance) tuples
    all_papers = []
    for case in cases:
        for gp in case.get("gold_papers", []):
            all_papers.append({
                "case_id": case["id"],
                "arxiv_id": gp["arxiv_id"],
                "claimed_title": gp["title"],
                "relevance": gp["relevance"],
            })

    total = len(all_papers)
    print(f"\n{'='*60}")
    print(f"Verifying {total} gold papers across {len(cases)} cases")
    print(f"{'='*60}\n")

    # Batch the IDs
    results: list[VerificationResult] = []
    for batch_start in range(0, total, BATCH_SIZE):
        batch = all_papers[batch_start : batch_start + BATCH_SIZE]
        ids = [p["arxiv_id"] for p in batch]
        print(f"  Checking batch {batch_start // BATCH_SIZE + 1}: IDs {batch_start+1}–{min(batch_start+BATCH_SIZE, total)}")

        api_results = batch_check(ids)

        for p in batch:
            ar = api_results.get(p["arxiv_id"], {"exists": False, "real_title": None})
            exists = ar["exists"]
            real_title = ar["real_title"]

            mismatch = False
            if exists and real_title:
                mismatch = title_mismatch(p["claimed_title"], real_title)

            note = ""
            if not exists:
                note = "ID does not exist on arXiv"
            elif mismatch:
                note = f"Title mismatch — real title: '{real_title}'"
            else:
                note = "OK"

            results.append(VerificationResult(
                arxiv_id=p["arxiv_id"],
                claimed_title=p["claimed_title"],
                case_id=p["case_id"],
                relevance=p["relevance"],
                exists=exists,
                real_title=real_title,
                mismatch=mismatch,
                note=note,
            ))

        if batch_start + BATCH_SIZE < total:
            time.sleep(DELAY_BETWEEN_BATCHES)

    # ── Summary ──────────────────────────────────────────────────────────────
    dead     = [r for r in results if not r.exists]
    mismatched = [r for r in results if r.exists and r.mismatch]
    ok       = [r for r in results if r.exists and not r.mismatch]

    print(f"\n{'='*60}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total papers checked : {total}")
    print(f"  ✅ OK                : {len(ok)}")
    print(f"  ⚠️  Title mismatch   : {len(mismatched)}  (paper exists but title is wrong)")
    print(f"  ❌ Does not exist    : {len(dead)}")
    print()

    if dead:
        print("❌ DEAD IDs (must replace):")
        for r in dead:
            print(f"   [{r.case_id}] {r.arxiv_id}  —  claimed: '{r.claimed_title}'")

    if mismatched:
        print("\n⚠️  TITLE MISMATCHES (verify manually):")
        for r in mismatched:
            print(f"   [{r.case_id}] {r.arxiv_id}")
            print(f"       Claimed : {r.claimed_title}")
            print(f"       Real    : {r.real_title}")

    # ── Save full report ──────────────────────────────────────────────────────
    report_path = Path(input_path).parent / "verification_report.json"
    report = {
        "summary": {
            "total": total,
            "ok": len(ok),
            "title_mismatch": len(mismatched),
            "dead": len(dead),
        },
        "dead_ids": [asdict(r) for r in dead],
        "mismatched_ids": [asdict(r) for r in mismatched],
        "all_results": [asdict(r) for r in results],
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n📄 Full report saved → {report_path}")

    # ── Save annotated dataset ─────────────────────────────────────────────
    # Mark each gold paper with verification status so you can review before replacing
    dead_set = {r.arxiv_id for r in dead}
    mismatch_set = {r.arxiv_id for r in mismatched}

    for case in cases:
        for gp in case.get("gold_papers", []):
            aid = gp["arxiv_id"]
            if aid in dead_set:
                gp["_verification"] = "DEAD — replace"
            elif aid in mismatch_set:
                result_obj = next(r for r in mismatched if r.arxiv_id == aid)
                gp["_verification"] = f"TITLE_MISMATCH — real: {result_obj.real_title}"
            else:
                gp["_verification"] = "OK"

    stem = Path(input_path).stem
    annotated_path = Path(input_path).parent / f"{stem}_annotated.json"
    annotated_path.write_text(json.dumps(data, indent=2))
    print(f"📄 Annotated dataset saved → {annotated_path}")
    print("\nNext step: review the annotated file, replace DEAD IDs, then re-run to confirm.")


if __name__ == "__main__":
    _default_in = (
        Path(__file__).resolve().parent / "datasets" / "hypothetica_benchmark_arxiv_gold_clean.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(_default_in),
        help="Path to the benchmark JSON file",
    )
    args = parser.parse_args()
    verify_dataset(args.input)