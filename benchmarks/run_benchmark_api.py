"""
Run preset benchmark JSON files through the Hypothetica benchmark API with checkpoint/resume.

Checkpoints are written after every case to:
  benchmarks/results/runs/{run_id}_{preset}.json
  benchmarks/results_v3/{run_id}_{preset}.json
If SerpApi quota is hit during a patents run, the server pauses automatically.

Usage:
    # Start patents benchmark (backend must be running on :8005)
    python benchmarks/run_benchmark_api.py --preset patents

    # Resume after quota pause or manual stop
    python benchmarks/run_benchmark_api.py --preset patents --resume RUN_ID

    # Smoke test first N cases, skip Supabase
    python benchmarks/run_benchmark_api.py --preset patents --limit 3 --no-supabase

    # Poll an existing run and print results when done
    python benchmarks/run_benchmark_api.py --status RUN_ID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "envfiles" / ".env")
load_dotenv(_ROOT / ".env")

API_BASE = "http://localhost:8005"
POLL_INTERVAL = 3
TERMINAL_STATUSES = {"completed", "failed", "paused"}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def start_run(
    api_base: str,
    preset: str,
    *,
    limit: int | None,
    dataset_path: str | None,
    resume_run_id: str | None,
    persist_supabase: bool,
    table_name: str,
    pause_on_quota: bool,
) -> str:
    payload = {
        "preset": preset,
        "persist_supabase": persist_supabase,
        "table_name": table_name,
        "pause_on_quota": pause_on_quota,
    }
    if limit is not None:
        payload["limit"] = limit
    if dataset_path:
        payload["dataset_path"] = dataset_path
    if resume_run_id:
        payload["resume_run_id"] = resume_run_id

    r = requests.post(f"{api_base}/api/benchmark/run", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["run_id"]


def get_status(api_base: str, run_id: str) -> dict:
    r = requests.get(f"{api_base}/api/benchmark/run/{run_id}", timeout=15)
    r.raise_for_status()
    return r.json()


def wait_for_terminal(api_base: str, run_id: str) -> dict:
    while True:
        st = get_status(api_base, run_id)
        status = st.get("status", "unknown")
        done = st.get("cases_done", 0)
        total = st.get("cases_total", "?")
        log(f"run {run_id}: status={status}  progress={done}/{total}")
        if status in TERMINAL_STATUSES:
            return st
        time.sleep(POLL_INTERVAL)


def load_artifact(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_artifact(run_id: str, preset: str | None) -> Path | None:
    search_dirs = [
        _ROOT / "benchmarks" / "results" / "runs",
        _ROOT / "benchmarks" / "results_v3",
    ]
    if preset:
        for runs_dir in search_dirs:
            p = runs_dir / f"{run_id}_{preset}.json"
            if p.is_file():
                return p
    for runs_dir in search_dirs:
        matches = sorted(runs_dir.glob(f"{run_id}_*.json"))
        if matches:
            return matches[0]
    return None


def print_summary(artifact: dict) -> None:
    cases = artifact.get("cases") or []
    completed = [c for c in cases if c.get("status") == "completed"]
    failed = [c for c in cases if c.get("status") != "completed"]

    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK RUN {artifact.get('run_id')} — {artifact.get('preset', '').upper()}")
    print(f"{'=' * 60}")
    print(f"  Status:        {artifact.get('status', artifact.get('run_id') and 'completed')}")
    print(f"  Dataset:       {artifact.get('dataset_path', '')}")
    print(f"  Cases total:   {artifact.get('cases_total', len(cases))}")
    print(f"  Cases done:    {artifact.get('cases_done', len(completed))}")
    if artifact.get("error"):
        print(f"  Note:          {artifact['error']}")
    print(f"  Output (runs): benchmarks/results/runs/{artifact.get('run_id')}_{artifact.get('preset')}.json")
    print(f"  Output (v3):   benchmarks/results_v3/{artifact.get('run_id')}_{artifact.get('preset')}.json")

    if completed:
        correct = sum(
            1
            for c in completed
            if c.get("predicted_label") and c.get("predicted_label") == c.get("true_label")
        )
        acc = correct / len(completed)
        print(f"\n  Label accuracy: {acc:.1%} ({correct}/{len(completed)} completed cases)")

        pred_counts = Counter(c.get("predicted_label") for c in completed)
        true_counts = Counter(c.get("true_label") for c in completed)
        print(f"  True labels:      {dict(true_counts)}")
        print(f"  Predicted labels: {dict(pred_counts)}")

        print("\n  --- Per-case ---")
        for c in cases:
            cid = c.get("case_id", "?")
            st = c.get("status", "?")
            true_l = c.get("true_label", "")
            pred_l = c.get("predicted_label", "")
            score = ""
            fr = c.get("full_result") or {}
            if fr.get("originality_score") is not None:
                score = f"  score={fr['originality_score']}"
            mark = "✓" if pred_l and pred_l == true_l else "·"
            err = f"  err={c['error']}" if c.get("error") else ""
            print(f"  {mark} {cid:<12} {st:<10} true={true_l:<16} pred={pred_l or '?':<16}{score}{err}")

    if failed:
        print(f"\n  Incomplete/failed: {len(failed)} case(s)")
        for c in failed:
            if c.get("status") != "completed":
                print(f"    - {c.get('case_id')}: {c.get('status')} {c.get('error') or ''}")

    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hypothetica benchmark via API")
    parser.add_argument("--preset", choices=["patents", "openalex", "github"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--resume", metavar="RUN_ID", default=None)
    parser.add_argument("--status", metavar="RUN_ID", default=None, help="Poll existing run only")
    parser.add_argument("--no-supabase", action="store_true")
    parser.add_argument("--table-name", default="benchmark2")
    parser.add_argument("--no-pause-on-quota", action="store_true")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full checkpoint JSON to stdout when the run finishes",
    )
    parser.add_argument("--api-base", default=API_BASE)
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    run_id = args.status or args.resume
    preset = args.preset

    if args.status:
        st = wait_for_terminal(api_base, args.status)
        run_id = args.status
        preset = preset or st.get("preset")
    else:
        if not preset:
            parser.error("--preset is required unless using --status")
        log(f"Starting benchmark preset={preset} resume={args.resume or 'no'}")
        run_id = start_run(
            api_base,
            preset,
            limit=args.limit,
            dataset_path=args.dataset_path,
            resume_run_id=args.resume,
            persist_supabase=not args.no_supabase,
            table_name=args.table_name,
            pause_on_quota=not args.no_pause_on_quota,
        )
        log(f"Run started: run_id={run_id}")
        st = wait_for_terminal(api_base, run_id)
        preset = preset or st.get("preset")

    artifact_path = find_artifact(run_id, preset)
    if artifact_path is None:
        out_rel = st.get("output_path")
        if out_rel:
            artifact_path = _ROOT / out_rel
    if artifact_path is None or not artifact_path.is_file():
        log(f"WARNING: checkpoint file not found for run {run_id}")
        print(json.dumps(st, indent=2))
        if st.get("status") == "paused":
            log(f"Resume with: python benchmarks/run_benchmark_api.py --preset {preset} --resume {run_id}")
            sys.exit(2)
        sys.exit(1 if st.get("status") == "failed" else 0)

    artifact = load_artifact(artifact_path)
    if artifact is None:
        log("ERROR: could not read checkpoint artifact")
        sys.exit(1)

    print_summary(artifact)

    if args.print_json:
        print(json.dumps(artifact, indent=2, ensure_ascii=False))

    status = st.get("status") or artifact.get("status")
    if status == "paused":
        log(f"Run paused — resume with: python benchmarks/run_benchmark_api.py --preset {preset} --resume {run_id}")
        sys.exit(2)
    if status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
