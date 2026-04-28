"""
Hypothetica Benchmark Runner
=============================
Runs benchmark cases against the live Hypothetica API (same flow as the UI:
analyze → follow-up questions → submit answers → processing → results).

Default arXiv dataset (single source of truth):
    benchmarks/datasets/hypothetica_benchmark_arxiv_gold_clean.json
Also:
    benchmarks/datasets/hypothetica_benchmark_patents.json

Usage:
    python run_benchmark.py
    python run_benchmark.py --source arxiv
    # Optional: separate results tree for an experiment:
    python run_benchmark.py --source arxiv --results-dir benchmarks/results_experiment_a
    python run_benchmark.py --source google_patents
    python run_benchmark.py --metrics-only
    python run_benchmark.py --upload-only          # backfill Supabase from saved raw results

    Ad-hoc API smoke tests (any text; not from dataset JSON; gold metrics are N/A):
    python run_benchmark.py --source arxiv --idea "Your research idea in one string"
    python run_benchmark.py --source arxiv --ideas-file /path/to/ideas.txt
    # ideas.txt: one research idea per line (omit blank lines)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load env vars (same lookup order as the backend)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / "envfiles" / ".env")
load_dotenv(_root / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8005"
DELAY_BETWEEN_JOBS = 5
POLL_INTERVAL = 2
QUESTIONS_TIMEOUT = 120
JOB_TIMEOUT = 600
K_VALUES = [3, 5, 10]

# API originality labels → benchmark dataset labels
API_LABEL_TO_BENCHMARK = {
    "high": "novel",
    "medium": "incremental",
    "low": "already_exists",
}

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"


def _default_arxiv_dataset() -> Path:
    env = os.environ.get("BENCHMARK_ARXIV_DATASET", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (DATASETS_DIR / "hypothetica_benchmark_arxiv_gold_clean.json").resolve()


def build_source_config(
    results_dir: Path,
    arxiv_dataset: Path | None = None,
) -> dict:
    arxiv_path = (arxiv_dataset or _default_arxiv_dataset()).resolve()
    return {
        "arxiv": {
            "dataset_json": arxiv_path,
            "raw_dir": results_dir / "arxiv",
            "results_csv": results_dir / "arxiv_results.csv",
            "metrics_csv": results_dir / "metrics_arxiv.csv",
            "api_source": "arxiv",
        },
        "google_patents": {
            "dataset_json": DATASETS_DIR / "hypothetica_benchmark_patents.json",
            "raw_dir": results_dir / "google_patents",
            "results_csv": results_dir / "google_patents_results.csv",
            "metrics_csv": results_dir / "metrics_google_patents.csv",
            "api_source": "google_patents",
        },
    }


# Default config (overridden in main() if --results-dir given)
SOURCE_CONFIG = build_source_config(RESULTS_DIR)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_dataset_json(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Returns (cases_for_runner, gold_map)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cases_out = []
    gold_map: dict[str, list[dict]] = {}

    for c in data.get("cases", []):
        cid = c["id"]
        cases_out.append(
            {
                "case_id": cid,
                "domain": c.get("domain", ""),
                "originality_label": c.get("originality_label", ""),
                "idea": c.get("idea", ""),
            }
        )
        # arXiv dataset uses gold_papers; patents dataset uses gold_patents
        gold_map[cid] = list(c.get("gold_papers") or c.get("gold_patents") or [])

    return cases_out, gold_map


def already_done(raw_dir: Path, case_id: str) -> bool:
    return (raw_dir / f"{case_id}.json").exists()


def save_raw(raw_dir: Path, case_id: str, data: dict):
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / f"{case_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_raw(raw_dir: Path, case_id: str) -> dict | None:
    p = raw_dir / f"{case_id}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def normalize_result_blob(blob: dict) -> dict:
    """Flatten API status payload so metrics see the analysis dict."""
    r = blob.get("results")
    if isinstance(r, dict):
        return r
    return blob


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jqdjzvnqvkwyaiqednyf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_sb_client = None


def _get_supabase():
    global _sb_client
    if _sb_client is None:
        if not SUPABASE_KEY:
            log("WARNING: SUPABASE_SERVICE_ROLE_KEY not set — skipping DB upload")
            return None
        from supabase import create_client
        _sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _sb_client


def _float_to_likert(value: float) -> int:
    """0-1 float → Likert 1-5 (same mapping as frontend/backend)."""
    if value >= 1.0:
        return 5
    if value >= 0.75:
        return 4
    if value >= 0.5:
        return 3
    if value >= 0.25:
        return 2
    return 1


def extract_predicted_label_benchmark(result: dict) -> str:
    """Map API label (low/medium/high) to benchmark gold label names."""
    api = (result.get("label") or "").strip().lower()
    return API_LABEL_TO_BENCHMARK.get(api, api or "")


def upload_benchmark_row(
    case: dict,
    result: dict,
    source_name: str,
    job_id: str | None = None,
) -> bool:
    """Insert one benchmark row into public.benchmark2. Returns True on success."""
    client = _get_supabase()
    if client is None:
        return False

    pred_label = extract_predicted_label_benchmark(result)
    api_label = result.get("label", "")

    agg = result.get("aggregated_criteria") or {}
    ps = agg.get("problem_similarity")
    ms = agg.get("method_similarity")
    do = agg.get("domain_similarity")
    cs = agg.get("contribution_similarity")

    cost = result.get("cost") or {}

    row = {
        "case_id": case["case_id"],
        "source": source_name,
        "domain": case.get("domain", ""),
        "idea": case.get("idea", ""),
        "true_label": case.get("originality_label", ""),
        "predicted_label": pred_label,
        "api_label": api_label,
        "originality_score": result.get("originality_score"),
        "global_similarity_score": result.get("global_similarity_score"),
        "likert_problem_similarity": _float_to_likert(ps) if ps is not None else None,
        "likert_method_similarity": _float_to_likert(ms) if ms is not None else None,
        "likert_domain_overlap": _float_to_likert(do) if do is not None else None,
        "likert_contribution_similarity": _float_to_likert(cs) if cs is not None else None,
        "criteria_problem_similarity": ps,
        "criteria_method_similarity": ms,
        "criteria_domain_overlap": do,
        "criteria_contribution_similarity": cs,
        "layer1_results": result.get("layer1_results"),
        "layer2_full": result.get("layer2_full"),
        "papers": result.get("papers"),
        "selected_sources": result.get("selected_sources"),
        "source_results": result.get("source_results"),
        "search_funnel": result.get("search_funnel"),
        "stats": result.get("stats"),
        "sentence_annotations": result.get("sentence_annotations"),
        "papers_analyzed": result.get("papers_analyzed", 0),
        "total_processing_time": result.get("total_processing_time"),
        "cost_breakdown": cost.get("breakdown"),
        "summary": result.get("summary", ""),
        "comprehensive_report": result.get("comprehensive_report", ""),
        "job_id": job_id,
    }

    try:
        resp = client.table("benchmark2").insert(row).execute()
        if resp.data:
            log(f"  ✓ Uploaded {case['case_id']} to Supabase benchmark table")
            return True
        log(f"  ✗ Supabase insert returned no data for {case['case_id']}")
        return False
    except Exception as e:
        log(f"  ✗ Supabase insert failed for {case['case_id']}: {e}")
        return False


def upload_all_from_raw(cases: list[dict], raw_dir: Path, source_name: str):
    """Upload all existing raw results to Supabase (for backfill / --upload-only)."""
    uploaded = skipped = 0
    for case in cases:
        cid = case["case_id"]
        blob = load_raw(raw_dir, cid)
        if blob is None:
            skipped += 1
            continue
        result = normalize_result_blob(blob)
        job_id = None
        snapshot = blob.get("_job_status_snapshot")
        if isinstance(snapshot, dict):
            job_id = snapshot.get("job_id")
        if upload_benchmark_row(case, result, source_name, job_id=job_id):
            uploaded += 1
        else:
            skipped += 1
    log(f"Upload complete: {uploaded} uploaded, {skipped} skipped/failed")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def post_analyze(idea: str, source: str) -> str | None:
    payload = {
        "user_idea": idea,
        "selected_sources": [source],
        "benchmark_mode": True,
    }
    try:
        r = requests.post(f"{API_BASE}/api/analyze", json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("job_id")
    except Exception as e:
        log(f"  ERROR POST /api/analyze: {e}")
        return None


def get_status(job_id: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/api/analyze/{job_id}/status", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"  ERROR GET status: {e}")
        return None


def wait_for_questions(job_id: str) -> list | None:
    deadline = time.time() + QUESTIONS_TIMEOUT
    while time.time() < deadline:
        data = get_status(job_id)
        if not data:
            time.sleep(POLL_INTERVAL)
            continue
        st = data.get("status", "")
        if st == "error":
            log(f"  Job {job_id} ERROR during questions: {data.get('error', '')}")
            return None
        if st == "waiting_for_answers" and data.get("questions"):
            return data["questions"]
        time.sleep(POLL_INTERVAL)
    log(f"  TIMEOUT waiting for follow-up questions ({job_id})")
    return None


def post_answers(job_id: str, answers: list[str]) -> bool:
    """POST body matches FastAPI AnswersRequest: {\"answers\": List[str]} in question order."""
    try:
        r = requests.post(
            f"{API_BASE}/api/analyze/{job_id}/answers",
            json={"answers": answers},
            timeout=30,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        log(f"  ERROR POST answers: {e}")
        return False


def poll_until_done(job_id: str) -> dict | None:
    deadline = time.time() + JOB_TIMEOUT
    while time.time() < deadline:
        data = get_status(job_id)
        if not data:
            time.sleep(POLL_INTERVAL)
            continue
        st = data.get("status", "")
        if st == "completed":
            return data
        if st == "error":
            log(f"  Job {job_id} ERROR: {data.get('error', 'unknown')}")
            return None
        time.sleep(POLL_INTERVAL)
    log(f"  TIMEOUT waiting for completion ({job_id})")
    return None


def run_one_case(idea: str, api_source: str) -> dict | None:
    job_id = post_analyze(idea, api_source)
    if not job_id:
        return None

    # benchmark_mode=True skips questions — job goes straight to processing
    return poll_until_done(job_id)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_arxiv_id(raw: str) -> str:
    """
    Canonical arXiv id for matching gold vs retrieved: strip version suffix
    and optional abs/pdf URL. New-style: 1234.56789, old-style: cs/0601001.
    """
    s = (raw or "").strip()
    if not s:
        return s
    # New-style id anywhere in the string (e.g. full URL to abs page)
    m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", s)
    if m:
        return m.group(1)
    return re.sub(r"v\d+$", "", s, flags=re.IGNORECASE)


def extract_retrieved_ids(result: dict, source: str) -> list[str]:
    """IDs in list order for overlap with gold arxiv_id / patent_id."""
    ids: list[str] = []
    papers = result.get("papers") or []
    if not isinstance(papers, list):
        return ids

    for p in papers:
        if not isinstance(p, dict):
            continue
        val = None
        for key in ("arxiv_id", "patent_id", "source_id"):
            v = p.get(key)
            if v:
                val = str(v).strip()
                break
        if val and not val.startswith("paper_"):
            ids.append(normalize_arxiv_id(val))

    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def precision_at_k(retrieved: list[str], gold_ids: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in gold_ids)
    return hits / k


def recall_at_k(retrieved: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in gold_ids)
    return hits / len(gold_ids)


def average_precision(retrieved: list[str], gold_ids: set[str]) -> float:
    if not gold_ids or not retrieved:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(retrieved, 1):
        if r in gold_ids:
            hits += 1
            score += hits / i
    return score / len(gold_ids)


def extract_predicted_label_benchmark(result: dict) -> str:
    """Map API label (high/medium/low) to dataset vocabulary."""
    label = result.get("label")
    if isinstance(label, str) and label:
        return API_LABEL_TO_BENCHMARK.get(label.lower().strip(), "unknown")

    score = result.get("originality_score")
    if score is not None:
        try:
            s = float(score)
            if s >= 70:
                return "novel"
            if s >= 40:
                return "incremental"
            return "already_exists"
        except (ValueError, TypeError):
            pass
    return "unknown"


def gold_id_sets(gold_items: list[dict]) -> tuple[set[str], set[str]]:
    all_ids: set[str] = set()
    core_ids: set[str] = set()
    for g in gold_items:
        gid = (g.get("arxiv_id") or g.get("patent_id") or "").strip()
        if not gid:
            continue
        nid = normalize_arxiv_id(gid) if g.get("arxiv_id") else gid
        all_ids.add(nid)
        if g.get("relevance") == "core":
            core_ids.add(nid)
    return all_ids, core_ids


def compute_metrics(
    cases: list[dict],
    gold_map: dict[str, list[dict]],
    raw_dir: Path,
    source: str,
) -> list[dict]:
    rows = []
    for case in cases:
        cid = case["case_id"]
        blob = load_raw(raw_dir, cid)
        if blob is None:
            log(f"  SKIP metrics for {cid} — no result file")
            continue

        result = normalize_result_blob(blob)
        gold_items = gold_map.get(cid, [])
        gold_all, gold_core = gold_id_sets(gold_items)

        retrieved = extract_retrieved_ids(result, source)
        pred_label = extract_predicted_label_benchmark(result)
        true_label = case["originality_label"]

        row = {
            "case_id": cid,
            "domain": case["domain"],
            "true_label": true_label,
            "predicted_label": pred_label,
            "label_correct": int(pred_label == true_label),
            "n_retrieved": len(retrieved),
            "n_gold_all": len(gold_all),
            "n_gold_core": len(gold_core),
            "avg_precision_all": round(average_precision(retrieved, gold_all), 4),
            "avg_precision_core": round(average_precision(retrieved, gold_core), 4),
        }

        for k in K_VALUES:
            row[f"P@{k}_all"] = round(precision_at_k(retrieved, gold_all, k), 4)
            row[f"R@{k}_all"] = round(recall_at_k(retrieved, gold_all, k), 4)
            row[f"P@{k}_core"] = round(precision_at_k(retrieved, gold_core, k), 4)
            row[f"R@{k}_core"] = round(recall_at_k(retrieved, gold_core, k), 4)

        rows.append(row)
    return rows


def save_metrics(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict], source: str):
    if not rows:
        print("No results to summarize.")
        return

    n = len(rows)
    acc = sum(r["label_correct"] for r in rows) / n

    print(f"\n{'=' * 55}")
    print(f"  RESULTS SUMMARY — {source.upper()}  ({n} cases)")
    print(f"{'=' * 55}")
    print(f"  Label accuracy:       {acc:.1%}")

    for k in K_VALUES:
        p_all = sum(r[f"P@{k}_all"] for r in rows) / n
        r_all = sum(r[f"R@{k}_all"] for r in rows) / n
        p_core = sum(r[f"P@{k}_core"] for r in rows) / n
        r_core = sum(r[f"R@{k}_core"] for r in rows) / n
        print(f"  P@{k} (all gold):     {p_all:.3f}   R@{k}: {r_all:.3f}")
        print(f"  P@{k} (core only):    {p_core:.3f}   R@{k}: {r_core:.3f}")

    map_all = sum(r["avg_precision_all"] for r in rows) / n
    map_core = sum(r["avg_precision_core"] for r in rows) / n
    print(f"  MAP (all gold):       {map_all:.3f}")
    print(f"  MAP (core only):      {map_core:.3f}")

    print("\n  --- Label breakdown ---")
    for label in ["novel", "incremental", "already_exists"]:
        subset = [r for r in rows if r["true_label"] == label]
        if subset:
            sub_acc = sum(r["label_correct"] for r in subset) / len(subset)
            sub_map = sum(r["avg_precision_all"] for r in subset) / len(subset)
            print(f"  {label:<20} n={len(subset):2d}  acc={sub_acc:.1%}  MAP={sub_map:.3f}")

    print(f"{'=' * 55}\n")


def adhoc_cases_single(idea: str) -> tuple[list[dict], dict[str, list[dict]]]:
    cid = "ad_hoc_001"
    cases = [
        {
            "case_id": cid,
            "domain": "",
            "originality_label": "",
            "idea": idea.strip(),
        }
    ]
    return cases, {cid: []}


def adhoc_cases_from_ideas_file(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"No non-empty lines in {path}")
    cases = []
    gold: dict[str, list[dict]] = {}
    for i, line in enumerate(lines, 1):
        cid = f"ad_hoc_{i:03d}"
        cases.append(
            {
                "case_id": cid,
                "domain": "",
                "originality_label": "",
                "idea": line,
            }
        )
        gold[cid] = []
    return cases, gold


def run_source(
    source_name: str,
    metrics_only: bool = False,
    upload_only: bool = False,
    *,
    cases: list[dict] | None = None,
    gold_map: dict[str, list[dict]] | None = None,
    adhoc: bool = False,
):
    cfg = SOURCE_CONFIG[source_name]
    path = cfg["dataset_json"]
    if cases is None or gold_map is None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        cases, gold_map = load_dataset_json(path)
    raw_dir = cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if upload_only:
        log(f"Uploading existing raw results for {source_name} to Supabase...")
        upload_all_from_raw(cases, raw_dir, source_name)
        return

    if not metrics_only:
        total = len(cases)
        skipped = submitted = failed = 0

        log(f"\n{'─' * 50}")
        log(f"Starting benchmark: {source_name.upper()}  ({total} cases)")
        if adhoc:
            log("Dataset: (ad-hoc ideas — not from benchmark JSON)")
        else:
            log(f"Dataset: {path}")
        log(f"{'─' * 50}")

        for i, case in enumerate(cases, 1):
            cid = case["case_id"]
            idea = case["idea"]

            if already_done(raw_dir, cid):
                log(f"[{i:02d}/{total}] SKIP {cid} (already done)")
                skipped += 1
                continue

            log(f"[{i:02d}/{total}] {cid} — running job...")
            status_payload = run_one_case(idea, cfg["api_source"])

            if status_payload is None:
                log(f"  FAILED {cid}")
                failed += 1
            else:
                flat = normalize_result_blob(status_payload)
                save_raw(
                    raw_dir,
                    cid,
                    {
                        "_job_status_snapshot": {
                            "job_id": status_payload.get("job_id"),
                            "status": status_payload.get("status"),
                        },
                        **flat,
                    },
                )
                submitted += 1
                score = flat.get("originality_score", "?")
                pred = extract_predicted_label_benchmark(flat)
                log(f"  DONE — score={score}  mapped_label={pred}  true={case['originality_label']}")

                upload_benchmark_row(
                    case, flat, source_name,
                    job_id=status_payload.get("job_id"),
                )

            time.sleep(DELAY_BETWEEN_JOBS)

        log(f"\nJobs: {submitted} completed, {skipped} skipped, {failed} failed")

    log(f"Computing metrics for {source_name}...")
    metric_rows = compute_metrics(cases, gold_map, raw_dir, source_name)
    save_metrics(metric_rows, cfg["metrics_csv"])
    log(f"Metrics saved → {cfg['metrics_csv']}")

    print_summary(metric_rows, source_name)


def main():
    parser = argparse.ArgumentParser(description="Hypothetica benchmark runner")
    parser.add_argument("--source", choices=["arxiv", "google_patents"])
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument(
        "--upload-only", action="store_true",
        help="Upload existing raw results to Supabase benchmark table (no API calls)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=(
            "Directory to read/write benchmark results. Defaults to "
            "'benchmarks/results'. Use a different path so multiple benchmark "
            "runs or dataset experiments do not overwrite the same raw JSON/CSVs."
        ),
    )
    parser.add_argument(
        "--arxiv-dataset",
        type=Path,
        default=None,
        help=(
            "Path to the arXiv benchmark JSON (default: "
            "hypothetica_benchmark_arxiv_gold_clean.json). "
            "Override with BENCHMARK_ARXIV_DATASET if set."
        ),
    )
    parser.add_argument(
        "--idea",
        type=str,
        default=None,
        help="Run a single ad-hoc idea through the live API (not from a dataset file).",
    )
    parser.add_argument(
        "--ideas-file",
        type=Path,
        default=None,
        help="Text file: one research idea per non-empty line; runs each as a separate ad-hoc case.",
    )
    args = parser.parse_args()

    if (args.idea or args.ideas_file) and args.upload_only:
        raise SystemExit("--upload-only is not valid with --idea or --ideas-file")

    if args.idea and args.ideas_file:
        raise SystemExit("Use only one of --idea and --ideas-file")

    arxiv_ds = args.arxiv_dataset
    if arxiv_ds is not None:
        arxiv_ds = arxiv_ds.resolve()

    # Override the module-level SOURCE_CONFIG so run_source() picks up the
    # caller-supplied results directory and optional arXiv dataset path.
    global SOURCE_CONFIG
    SOURCE_CONFIG = build_source_config(args.results_dir, arxiv_dataset=arxiv_ds)
    log(f"Using results directory: {args.results_dir.resolve()}")

    if args.idea is not None or args.ideas_file is not None:
        source = args.source or "arxiv"
        if args.source is None:
            log("No --source for ad-hoc run; using arxiv")
        if args.ideas_file is not None:
            p = args.ideas_file.resolve()
            if not p.is_file():
                raise SystemExit(f"Not a file: {p}")
            cases, gold_map = adhoc_cases_from_ideas_file(p)
        else:
            if not (args.idea or "").strip():
                raise SystemExit("--idea must be non-empty")
            cases, gold_map = adhoc_cases_single(args.idea)
        run_source(
            source,
            metrics_only=args.metrics_only,
            upload_only=False,
            cases=cases,
            gold_map=gold_map,
            adhoc=True,
        )
        log("All done.")
        return

    sources = [args.source] if args.source else ["arxiv", "google_patents"]
    if "arxiv" in sources:
        log(f"arXiv dataset: {SOURCE_CONFIG['arxiv']['dataset_json']}")

    for source in sources:
        run_source(source, metrics_only=args.metrics_only, upload_only=args.upload_only)

    log("All done.")


if __name__ == "__main__":
    main()
