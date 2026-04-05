"""
Hypothetica Benchmark Runner
=============================
Runs benchmark cases against the live Hypothetica API (same flow as the UI:
analyze → follow-up questions → submit answers → processing → results).

Loads datasets from:
    benchmarks/datasets/hypothetica_benchmark_arxiv.json
    benchmarks/datasets/hypothetica_benchmark_patents.json

Usage:
    python run_benchmark.py
    python run_benchmark.py --source arxiv
    python run_benchmark.py --source google_patents
    python run_benchmark.py --metrics-only
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8005"
DELAY_BETWEEN_JOBS = 8
POLL_INTERVAL = 5
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

SOURCE_CONFIG = {
    "arxiv": {
        "dataset_json": DATASETS_DIR / "hypothetica_benchmark_arxiv.json",
        "raw_dir": RESULTS_DIR / "arxiv",
        "results_csv": RESULTS_DIR / "arxiv_results.csv",
        "metrics_csv": RESULTS_DIR / "metrics_arxiv.csv",
        "api_source": "arxiv",
    },
    "google_patents": {
        "dataset_json": DATASETS_DIR / "hypothetica_benchmark_patents.json",
        "raw_dir": RESULTS_DIR / "google_patents",
        "results_csv": RESULTS_DIR / "google_patents_results.csv",
        "metrics_csv": RESULTS_DIR / "metrics_google_patents.csv",
        "api_source": "google_patents",
    },
}


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
# API
# ---------------------------------------------------------------------------

def post_analyze(idea: str, source: str) -> str | None:
    payload = {
        "user_idea": idea,
        "selected_sources": [source],
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

    questions = wait_for_questions(job_id)
    if questions is None:
        return None

    answers = [
        "Benchmark mode — no further clarification; please proceed with the idea as stated."
    ] * len(questions)
    if not post_answers(job_id, answers):
        return None

    return poll_until_done(job_id)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
            ids.append(val)

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
        all_ids.add(gid)
        if g.get("relevance") == "core":
            core_ids.add(gid)
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


def run_source(source_name: str, metrics_only: bool = False):
    cfg = SOURCE_CONFIG[source_name]
    path = cfg["dataset_json"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases, gold_map = load_dataset_json(path)
    raw_dir = cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_only:
        total = len(cases)
        skipped = submitted = failed = 0

        log(f"\n{'─' * 50}")
        log(f"Starting benchmark: {source_name.upper()}  ({total} cases)")
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
    args = parser.parse_args()

    sources = [args.source] if args.source else ["arxiv", "google_patents"]

    for source in sources:
        run_source(source, metrics_only=args.metrics_only)

    log("All done.")


if __name__ == "__main__":
    main()
