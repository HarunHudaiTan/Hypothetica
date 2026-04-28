"""
Sweep post-Layer1 scoring parameters on a benchmark CSV WITHOUT touching the
LLM Likert output. Reads per-paper criteria_scores from layer1_results JSON,
then re-computes paper_similarity → global_similarity → originality_score →
label using configurable parameters and reports accuracy.

Parameters explored (all four can be varied independently or jointly):
  - CRITERIA_WEIGHTS       (problem, method, domain, contribution)  — Layer1+Layer2 paper sim
  - GLOBAL_MAX_WEIGHT      (the 0.7 in `0.7*max + 0.3*mean`)        — Layer2 global aggregation
  - OVERLAP_CURVE_POWER    (the 1.5 exponent)                        — Layer2 score curve
  - SCORE_RED_MAX, SCORE_YELLOW_MAX (label cutoffs in 0-100 space)   — Layer2 label mapping

Usage:
  python benchmarks/sweep_scoring_params.py [path_to_csv]
Default csv: benchmarks/results/runs/benchmark_patents_rows.csv
"""
from __future__ import annotations

import csv
import json
import sys
import itertools
from pathlib import Path
from typing import List, Dict, Tuple

csv.field_size_limit(2**31 - 1)

DEFAULT_CSV = "benchmarks/results/runs/benchmark_patents_rows.csv"

# Current production values (from backend/core/config.py and layer2_agent.py)
DEFAULT = {
    "w_p": 0.15, "w_m": 0.30, "w_d": 0.10, "w_c": 0.45,
    "global_max_w": 0.7,         # 0.7*max + 0.3*mean
    "curve_power": 1.5,           # (1 - gsim^power) * 100
    "red_max": 40,                # < red_max → already_exists
    "yellow_max": 70,             # >= yellow_max → novel; in between → incremental
}


def load_cases(csv_path: str) -> List[Dict]:
    """Each case = {true_label, papers: [{p,m,d,c}, ...]}"""
    cases = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            true = (r.get("true_label") or "").strip()
            if not true:
                continue
            try:
                papers = json.loads(r["layer1_results"])
            except Exception:
                continue
            paper_scores = []
            for p in papers:
                cs = p.get("criteria_scores") or {}
                paper_scores.append({
                    "p": float(cs.get("problem_similarity", 0.0)),
                    "m": float(cs.get("method_similarity", 0.0)),
                    "d": float(cs.get("domain_similarity", 0.0)),
                    "c": float(cs.get("contribution_similarity", 0.0)),
                })
            if paper_scores:
                cases.append({"true": true, "papers": paper_scores, "case_id": r.get("case_id")})
    return cases


def predict(case: Dict, params: Dict) -> Tuple[str, float, float, int]:
    """Return (predicted_label, global_sim, originality, paper_count)."""
    wp, wm, wd, wc = params["w_p"], params["w_m"], params["w_d"], params["w_c"]
    s = wp + wm + wd + wc
    wp, wm, wd, wc = wp / s, wm / s, wd / s, wc / s

    paper_sims = [
        wp * x["p"] + wm * x["m"] + wd * x["d"] + wc * x["c"]
        for x in case["papers"]
    ]
    if not paper_sims:
        return "novel", 0.0, 100, 0

    mx = max(paper_sims)
    mn = sum(paper_sims) / len(paper_sims)
    gsim = params["global_max_w"] * mx + (1 - params["global_max_w"]) * mn

    orig = (1.0 - gsim ** params["curve_power"]) * 100
    orig = max(0, min(100, orig))

    if orig >= params["yellow_max"]:
        label = "novel"
    elif orig >= params["red_max"]:
        label = "incremental"
    else:
        label = "already_exists"
    return label, gsim, orig, len(paper_sims)


def accuracy(cases: List[Dict], params: Dict) -> int:
    return sum(1 for c in cases if predict(c, params)[0] == c["true"])


def per_class_breakdown(cases: List[Dict], params: Dict) -> Dict:
    by = {l: {"n": 0, "ok": 0} for l in ("novel", "incremental", "already_exists")}
    for c in cases:
        pred, *_ = predict(c, params)
        by[c["true"]]["n"] += 1
        if pred == c["true"]:
            by[c["true"]]["ok"] += 1
    return by


def confusion(cases: List[Dict], params: Dict) -> Dict:
    cm = {}
    for c in cases:
        pred, *_ = predict(c, params)
        cm[(c["true"], pred)] = cm.get((c["true"], pred), 0) + 1
    return cm


def report(label: str, cases, params: Dict):
    n = len(cases)
    a = accuracy(cases, params)
    by = per_class_breakdown(cases, params)
    parts = " | ".join(f"{k}={v['ok']}/{v['n']}" for k, v in by.items())
    print(f"  {label:<60}  {a}/{n} = {a/n*100:5.1f}%   ({parts})")


def sweep_one(cases, base: Dict, key: str, values: List, label_fmt: str):
    print(f"\n--- vary {key} (others held at default) ---")
    for v in values:
        p = dict(base)
        p[key] = v
        report(label_fmt.format(v), cases, p)


def sweep_thresholds(cases, base: Dict, top_n=10):
    print(f"\n--- threshold-only sweep (red_max × yellow_max) ---")
    results = []
    for red in range(20, 65, 2):
        for yel in range(red + 4, 92, 2):
            p = dict(base)
            p["red_max"] = red
            p["yellow_max"] = yel
            results.append((accuracy(cases, p), red, yel))
    results.sort(reverse=True)
    print(f"  current red={base['red_max']}, yellow={base['yellow_max']}  → {accuracy(cases, base)}/{len(cases)}")
    for a, red, yel in results[:top_n]:
        print(f"  red={red:>3} yellow={yel:>3}  → {a}/{len(cases)} = {a/len(cases)*100:5.1f}%")


def sweep_global_blend(cases, base: Dict):
    print(f"\n--- global aggregation: max-blend weight ---")
    for mw in [0.0, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]:
        p = dict(base)
        p["global_max_w"] = mw
        # also re-tune thresholds for fair comparison
        best = max(
            ((accuracy(cases, {**p, "red_max": r, "yellow_max": y}), r, y)
             for r in range(20, 65, 2) for y in range(r + 4, 92, 2)),
        )
        a, r, y = best
        kind = "pure mean" if mw == 0 else "pure max" if mw == 1 else f"{mw:.2f}*max + {1-mw:.2f}*mean"
        print(f"  {kind:<30}  best @ red={r:>3} yel={y:>3}  → {a}/{len(cases)} = {a/len(cases)*100:5.1f}%")


def sweep_curve(cases, base: Dict):
    print(f"\n--- curve power (with thresholds re-tuned) ---")
    for power in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        p = dict(base)
        p["curve_power"] = power
        best = max(
            ((accuracy(cases, {**p, "red_max": r, "yellow_max": y}), r, y)
             for r in range(20, 65, 2) for y in range(r + 4, 92, 2)),
        )
        a, r, y = best
        print(f"  power={power:>4.2f}  best @ red={r:>3} yel={y:>3}  → {a}/{len(cases)} = {a/len(cases)*100:5.1f}%")


def sweep_weights_with_thresholds(cases, base: Dict, top_n=10):
    print(f"\n--- coarse weight sweep (each combo with optimal thresholds) ---")
    results = []
    grid_p = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    grid_m = [0.10, 0.20, 0.30, 0.40, 0.50]
    grid_d = [0.0, 0.05, 0.10, 0.15]
    grid_c = [0.20, 0.30, 0.40, 0.50, 0.60]
    for wp, wm, wd, wc in itertools.product(grid_p, grid_m, grid_d, grid_c):
        s = wp + wm + wd + wc
        if s < 0.5 or s > 1.5:
            continue
        p = dict(base, w_p=wp, w_m=wm, w_d=wd, w_c=wc)
        best_a, best_r, best_y = 0, 0, 0
        for r in range(20, 65, 4):
            for y in range(r + 4, 92, 4):
                a = accuracy(cases, {**p, "red_max": r, "yellow_max": y})
                if a > best_a:
                    best_a, best_r, best_y = a, r, y
        results.append((best_a, wp, wm, wd, wc, best_r, best_y))
    results.sort(reverse=True)
    print(f"  current weights p=0.15 m=0.30 d=0.10 c=0.45  → {accuracy(cases, base)}/{len(cases)}")
    print(f"  {'acc':>5}  {'p':>5}{'m':>5}{'d':>5}{'c':>5}   thresholds")
    for a, wp, wm, wd, wc, r, y in results[:top_n]:
        s = wp + wm + wd + wc
        print(f"  {a:>5}  {wp/s:>5.2f}{wm/s:>5.2f}{wd/s:>5.2f}{wc/s:>5.2f}   red={r:>3} yel={y:>3}")


def joint_optimum(cases, base: Dict):
    print(f"\n--- joint optimum (weights + max_blend + power + thresholds, coarse) ---")
    grid_p = [0.0, 0.10, 0.15, 0.20]
    grid_m = [0.20, 0.30, 0.40]
    grid_d = [0.0, 0.10]
    grid_c = [0.30, 0.45, 0.55]
    grid_mw = [0.3, 0.5, 0.7, 1.0]
    grid_pw = [1.0, 1.5, 2.0]
    best = (0, None)
    seen = 0
    for wp, wm, wd, wc, mw, pw in itertools.product(
        grid_p, grid_m, grid_d, grid_c, grid_mw, grid_pw
    ):
        if wp + wm + wd + wc < 0.5:
            continue
        seen += 1
        for r in range(20, 65, 4):
            for y in range(r + 4, 92, 4):
                params = {
                    "w_p": wp, "w_m": wm, "w_d": wd, "w_c": wc,
                    "global_max_w": mw, "curve_power": pw,
                    "red_max": r, "yellow_max": y,
                }
                a = accuracy(cases, params)
                if a > best[0]:
                    best = (a, params)
    a, p = best
    s = p["w_p"] + p["w_m"] + p["w_d"] + p["w_c"]
    print(f"  searched {seen} weight combos × ~{17*17} thresholds")
    print(f"  best: {a}/{len(cases)} = {a/len(cases)*100:.1f}%")
    print(f"    weights p={p['w_p']/s:.2f} m={p['w_m']/s:.2f} d={p['w_d']/s:.2f} c={p['w_c']/s:.2f}")
    print(f"    global_max_w={p['global_max_w']}  curve_power={p['curve_power']}  red={p['red_max']} yel={p['yellow_max']}")
    print(f"    confusion (true → pred):")
    cm = confusion(cases, p)
    for (t, pr), n in sorted(cm.items()):
        print(f"      {t:>15} → {pr:<15}  {n}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not Path(csv_path).exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
    cases = load_cases(csv_path)
    print(f"Loaded {len(cases)} cases from {csv_path}")
    n_papers = sum(len(c["papers"]) for c in cases)
    print(f"Total per-paper rows: {n_papers}  (avg {n_papers/len(cases):.1f} papers/case)")

    print("\n" + "=" * 78)
    print("BASELINE (current production values)")
    print("=" * 78)
    report("default", cases, DEFAULT)

    sweep_thresholds(cases, DEFAULT, top_n=8)
    sweep_global_blend(cases, DEFAULT)
    sweep_curve(cases, DEFAULT)
    sweep_weights_with_thresholds(cases, DEFAULT, top_n=8)
    joint_optimum(cases, DEFAULT)


if __name__ == "__main__":
    main()
