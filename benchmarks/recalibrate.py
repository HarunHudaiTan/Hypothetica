"""
Layer 2 offline recalibration sweep.

Reads saved benchmark results (per-paper Layer 1 criteria scores) and the labeled
dataset, then sweeps the Layer 2 aggregation parameters to find the configuration
that best matches the ground-truth originality labels.

No API calls, no LLM cost — pure math replay over stored criteria scores.

Usage:
    python benchmarks/recalibrate.py
    python benchmarks/recalibrate.py --source arxiv
    python benchmarks/recalibrate.py --top 20 --output recalibration_results.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

SOURCE_CONFIG = {
    "arxiv": {
        "dataset_json": DATASETS_DIR / "hypothetica_benchmark_arxiv_gold_clean.json",
        "results_dir": RESULTS_DIR / "arxiv",
    },
    "google_patents": {
        "dataset_json": DATASETS_DIR / "hypothetica_benchmark_patents.json",
        "results_dir": RESULTS_DIR / "google_patents",
    },
}

LABELS = ["novel", "incremental", "already_exists"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Case:
    case_id: str
    true_label: str
    # List of 5 papers, each a dict with p/m/d/c floats in [0,1]
    papers: List[Dict[str, float]]


def load_cases(
    source: str,
    dataset_json: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> List[Case]:
    cfg = SOURCE_CONFIG[source]
    ds_path = dataset_json or cfg["dataset_json"]
    res_dir = results_dir or cfg["results_dir"]
    with open(ds_path, encoding="utf-8") as f:
        dataset = json.load(f)

    true_labels = {c["id"]: c["originality_label"] for c in dataset["cases"]}
    cases: List[Case] = []

    for result_file in sorted(res_dir.glob("*.json")):
        case_id = result_file.stem
        if case_id not in true_labels:
            continue

        with open(result_file, encoding="utf-8") as f:
            blob = json.load(f)

        # Extract per-paper criteria scores from top-level "papers" array
        papers_raw = blob.get("papers") or []
        papers: List[Dict[str, float]] = []
        for p in papers_raw:
            cs = p.get("criteria_scores")
            if not cs:
                continue
            papers.append({
                "p": float(cs.get("problem_similarity", 0.0)),
                "m": float(cs.get("method_similarity", 0.0)),
                "d": float(cs.get("domain_similarity") or cs.get("domain_overlap", 0.0)),
                "c": float(cs.get("contribution_similarity", 0.0)),
            })

        if not papers:
            # Skip cases with no scored papers
            continue

        cases.append(Case(
            case_id=case_id,
            true_label=true_labels[case_id],
            papers=papers,
        ))

    return cases


# ---------------------------------------------------------------------------
# Layer 2 formulas — mirror of backend/app/agents/layer2_agent.py
# ---------------------------------------------------------------------------

def compute_originality(
    papers: List[Dict[str, float]],
    w_p: float, w_m: float, w_d: float, w_c: float,
    alpha: float,   # PAPER_SIMILARITY_MAX_WEIGHT
    beta: float,    # GLOBAL_SIMILARITY_MAX_WEIGHT
    gamma: float,   # OVERLAP_CURVE_POWER
) -> Tuple[int, float]:
    """Returns (originality_score_0_100, global_similarity_0_1)."""
    paper_sims: List[float] = []
    for p in papers:
        weighted_mean = w_p * p["p"] + w_m * p["m"] + w_d * p["d"] + w_c * p["c"]
        max_score = max(p["p"], p["m"], p["d"], p["c"])
        paper_sim = alpha * max_score + (1 - alpha) * weighted_mean
        paper_sims.append(paper_sim)

    max_sim = max(paper_sims)
    mean_sim = sum(paper_sims) / len(paper_sims)
    global_sim = beta * max_sim + (1 - beta) * mean_sim

    originality = (1.0 - global_sim ** gamma) * 100
    originality = int(max(0, min(100, originality)))
    return originality, global_sim


def score_to_label(score: int, t_low: int, t_high: int) -> str:
    if score >= t_high:
        return "novel"
    if score >= t_low:
        return "incremental"
    return "already_exists"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(cases: List[Case], predictions: List[str]) -> Dict[str, Dict[str, int]]:
    cm = {t: {p: 0 for p in LABELS} for t in LABELS}
    for case, pred in zip(cases, predictions):
        if case.true_label in cm and pred in cm[case.true_label]:
            cm[case.true_label][pred] += 1
    return cm


def per_class_f1(cm: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    f1s: Dict[str, float] = {}
    for cls in LABELS:
        tp = cm[cls][cls]
        fn = sum(cm[cls][p] for p in LABELS if p != cls)
        fp = sum(cm[t][cls] for t in LABELS if t != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s[cls] = f1
    return f1s


def evaluate_config(cases: List[Case], params: Dict) -> Dict:
    predictions: List[str] = []
    scores: List[int] = []
    for case in cases:
        score, _ = compute_originality(
            case.papers,
            params["w_p"], params["w_m"], params["w_d"], params["w_c"],
            params["alpha"], params["beta"], params["gamma"],
        )
        scores.append(score)
        predictions.append(score_to_label(score, params["t_low"], params["t_high"]))

    cm = confusion_matrix(cases, predictions)
    f1s = per_class_f1(cm)
    macro_f1 = sum(f1s.values()) / len(f1s)
    accuracy = sum(1 for c, p in zip(cases, predictions) if c.true_label == p) / len(cases)
    # Balanced accuracy = mean recall
    recalls = []
    for cls in LABELS:
        tp = cm[cls][cls]
        fn = sum(cm[cls][p] for p in LABELS if p != cls)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    balanced_acc = sum(recalls) / len(recalls)

    return {
        **params,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "balanced_acc": balanced_acc,
        "f1_novel": f1s["novel"],
        "f1_incremental": f1s["incremental"],
        "f1_already_exists": f1s["already_exists"],
        "confusion": cm,
        "scores": scores,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Layer 1 diagnostic (tells us if Layer 1 scores are discriminative)
# ---------------------------------------------------------------------------

def diagnose_layer1(cases: List[Case]):
    by_label: Dict[str, List[Dict[str, float]]] = {l: [] for l in LABELS}
    for case in cases:
        if case.true_label not in by_label:
            continue
        # Collect all per-paper criteria for cases of this label
        for paper in case.papers:
            by_label[case.true_label].append(paper)

    print("\n" + "=" * 72)
    print("  LAYER 1 DIAGNOSTIC — Mean criteria scores per true label")
    print("  (across all retrieved papers in all cases of that label)")
    print("=" * 72)
    print(f"  {'label':<18} {'n_papers':>9}  {'problem':>8} {'method':>8} {'domain':>8} {'contrib':>8}")
    print("  " + "-" * 64)
    for label in LABELS:
        papers = by_label[label]
        if not papers:
            print(f"  {label:<18} {'-':>9}  {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            continue
        mean_p = statistics.mean(p["p"] for p in papers)
        mean_m = statistics.mean(p["m"] for p in papers)
        mean_d = statistics.mean(p["d"] for p in papers)
        mean_c = statistics.mean(p["c"] for p in papers)
        print(f"  {label:<18} {len(papers):>9}  {mean_p:>8.3f} {mean_m:>8.3f} {mean_d:>8.3f} {mean_c:>8.3f}")

    # Gap analysis: how different are novel vs already_exists?
    if by_label["novel"] and by_label["already_exists"]:
        print("\n  Gap (already_exists − novel):")
        for key, name in [("p", "problem"), ("m", "method"), ("d", "domain"), ("c", "contribution")]:
            nov_mean = statistics.mean(p[key] for p in by_label["novel"])
            exist_mean = statistics.mean(p[key] for p in by_label["already_exists"])
            gap = exist_mean - nov_mean
            flag = "✓ discriminative" if abs(gap) >= 0.05 else "✗ flat (not useful)"
            print(f"    {name:<12} Δ = {gap:+.3f}   {flag}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def weight_combinations() -> List[Tuple[float, float, float, float]]:
    """Generate (w_p, w_m, w_d, w_c) combos that sum to 1.0 (within tolerance)."""
    grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    combos = []
    for w_p, w_m, w_d, w_c in itertools.product(grid, repeat=4):
        total = w_p + w_m + w_d + w_c
        if abs(total - 1.0) < 1e-9:
            combos.append((w_p, w_m, w_d, w_c))
    return combos


def sweep(cases: List[Case]) -> List[Dict]:
    alpha_grid = [0.0, 0.2, 0.3, 0.5]
    beta_grid = [0.0, 0.3, 0.5, 0.7]
    gamma_grid = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    threshold_grid = [(30, 60), (35, 65), (40, 70), (45, 75), (30, 55), (25, 50)]
    weights = weight_combinations()

    total = len(weights) * len(alpha_grid) * len(beta_grid) * len(gamma_grid) * len(threshold_grid)
    print(f"\nSweeping {total:,} configurations over {len(cases)} cases...")

    results: List[Dict] = []
    for (w_p, w_m, w_d, w_c) in weights:
        for alpha in alpha_grid:
            for beta in beta_grid:
                for gamma in gamma_grid:
                    for t_low, t_high in threshold_grid:
                        params = {
                            "w_p": w_p, "w_m": w_m, "w_d": w_d, "w_c": w_c,
                            "alpha": alpha, "beta": beta, "gamma": gamma,
                            "t_low": t_low, "t_high": t_high,
                        }
                        results.append(evaluate_config(cases, params))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CURRENT_CONFIG = {
    "w_p": 0.3, "w_m": 0.3, "w_d": 0.2, "w_c": 0.2,
    "alpha": 0.5, "beta": 0.7, "gamma": 1.5,
    "t_low": 40, "t_high": 70,
}


# ---------------------------------------------------------------------------
# NEW FORMULA — user-proposed 2-branch global + piecewise linear curve
# ---------------------------------------------------------------------------

NEW_FORMULA_WEIGHTS = {
    "p": 0.15,   # problem (context)
    "d": 0.10,   # domain  (context, retrieval already guarantees)
    "m": 0.30,   # method  (novelty signal)
    "c": 0.45,   # contribution (strongest novelty signal)
}
NEW_EVIDENCE_THRESHOLD = 0.85       # top >= this → "evidence found, use top"
NEW_THRESHOLDS = (40, 70)           # t_low, t_high


def new_paper_similarity(paper: Dict[str, float], w: Dict[str, float]) -> float:
    """Step 1: weighted mean only (no max term)."""
    return w["p"] * paper["p"] + w["d"] * paper["d"] + w["m"] * paper["m"] + w["c"] * paper["c"]


def new_global_similarity(
    paper_sims: List[float],
    evidence_threshold: float,
    mode: str = "2branch",
    outlier_gap: float = 0.25,
) -> float:
    """Step 2: collapse per-paper similarities into a single global score.

    mode:
      '2branch' (user): top >= evidence_threshold → top ; else mean(all)
      '3branch'      : top >= evidence_threshold  → top
                       elif (top - rest_mean) > outlier_gap → rest_mean
                       else → (top + rest_mean) / 2
      'mean'         : plain mean (baseline)
      'max'          : plain max  (baseline)
    """
    top = max(paper_sims)
    if len(paper_sims) <= 1:
        return top

    if mode == "max":
        return top

    if mode == "mean":
        return sum(paper_sims) / len(paper_sims)

    if mode == "2branch":
        if top >= evidence_threshold:
            return top
        return sum(paper_sims) / len(paper_sims)

    # 3branch
    sorted_sims = sorted(paper_sims, reverse=True)
    rest = sorted_sims[1:]
    rest_mean = sum(rest) / len(rest)

    if top >= evidence_threshold:
        return top                       # strong evidence
    if (top - rest_mean) > outlier_gap:
        return rest_mean                 # drop outlier top
    return (top + rest_mean) / 2         # balanced blend


def new_overlap_to_originality(overlap: float, curve: str = "piecewise", gamma: float = 2.0) -> int:
    """Step 3: convert 0-1 overlap to 0-100 originality.
    curve:
      'linear'    → (1 - overlap) * 100
      'power'     → (1 - overlap^gamma) * 100
      'piecewise' → 4-piece linear, aligned with label thresholds
    """
    if curve == "linear":
        val = (1.0 - overlap) * 100
    elif curve == "power":
        val = (1.0 - overlap ** gamma) * 100
    else:  # piecewise (default)
        if overlap <= 0.30:
            val = 100.0
        elif overlap <= 0.50:
            val = 100.0 - (overlap - 0.30) * 150    # 100 → 70
        elif overlap <= 0.70:
            val = 70.0  - (overlap - 0.50) * 150    # 70 → 40
        elif overlap <= 0.90:
            val = 40.0  - (overlap - 0.70) * 175    # 40 → 5
        else:
            val = 5.0
    return int(max(0, min(100, val)))


def new_score_to_label(score: int, t_low: int, t_high: int) -> str:
    if score >= t_high:
        return "novel"
    if score >= t_low:
        return "incremental"
    return "already_exists"


def evaluate_new_formula(
    cases: List[Case],
    weights: Dict[str, float] = NEW_FORMULA_WEIGHTS,
    evidence_threshold: float = NEW_EVIDENCE_THRESHOLD,
    thresholds: Tuple[int, int] = NEW_THRESHOLDS,
    curve: str = "piecewise",
    gamma: float = 2.0,
    global_mode: str = "2branch",
    outlier_gap: float = 0.25,
) -> Dict:
    predictions: List[str] = []
    scores: List[int] = []
    for case in cases:
        paper_sims = [new_paper_similarity(p, weights) for p in case.papers]
        global_sim = new_global_similarity(
            paper_sims, evidence_threshold, mode=global_mode, outlier_gap=outlier_gap
        )
        score = new_overlap_to_originality(global_sim, curve=curve, gamma=gamma)
        scores.append(score)
        predictions.append(new_score_to_label(score, thresholds[0], thresholds[1]))

    cm = confusion_matrix(cases, predictions)
    f1s = per_class_f1(cm)
    macro_f1 = sum(f1s.values()) / len(f1s)
    accuracy = sum(1 for c, p in zip(cases, predictions) if c.true_label == p) / len(cases)
    recalls = []
    for cls in LABELS:
        tp = cm[cls][cls]
        fn = sum(cm[cls][p] for p in LABELS if p != cls)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    balanced_acc = sum(recalls) / len(recalls)

    return {
        "formula": f"NEW (global={global_mode}, curve={curve})",
        "curve": curve,
        "gamma": gamma,
        "global_mode": global_mode,
        "outlier_gap": outlier_gap,
        "weights": weights,
        "evidence_threshold": evidence_threshold,
        "thresholds": thresholds,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "balanced_acc": balanced_acc,
        "f1_novel": f1s["novel"],
        "f1_incremental": f1s["incremental"],
        "f1_already_exists": f1s["already_exists"],
        "confusion": cm,
        "scores": scores,
        "predictions": predictions,
    }


def sweep_new_formula(cases: List[Case]) -> List[Dict]:
    """Small sweep over new-formula knobs only."""
    weight_grid = [
        {"p": 0.15, "d": 0.10, "m": 0.30, "c": 0.45},   # default
        {"p": 0.10, "d": 0.10, "m": 0.30, "c": 0.50},   # c-heavy
        {"p": 0.20, "d": 0.10, "m": 0.30, "c": 0.40},   # p-heavier
        {"p": 0.10, "d": 0.05, "m": 0.35, "c": 0.50},   # novelty-focused
        {"p": 0.20, "d": 0.20, "m": 0.30, "c": 0.30},   # balanced
        {"p": 0.15, "d": 0.15, "m": 0.35, "c": 0.35},   # mid
        {"p": 0.00, "d": 0.10, "m": 0.40, "c": 0.50},   # drop problem
        {"p": 0.25, "d": 0.00, "m": 0.35, "c": 0.40},   # drop domain
    ]
    evidence_grid = [0.75, 0.80, 0.85, 0.90]
    threshold_grid = [(30, 60), (35, 65), (40, 70), (45, 75)]
    curve_grid = [
        ("linear", 1.0),
        ("piecewise", 1.0),
        ("power", 1.5),
        ("power", 2.0),
        ("power", 3.0),
    ]
    global_mode_grid = [
        ("2branch", 0.25),
        ("3branch", 0.20),
        ("3branch", 0.25),
        ("3branch", 0.30),
        ("mean", 0.0),
        ("max", 0.0),
    ]

    results = []
    for w in weight_grid:
        for ev in evidence_grid:
            for th in threshold_grid:
                for curve, gamma in curve_grid:
                    for gmode, ogap in global_mode_grid:
                        results.append(evaluate_new_formula(
                            cases, w, ev, th,
                            curve=curve, gamma=gamma,
                            global_mode=gmode, outlier_gap=ogap,
                        ))
    return results


def format_confusion(cm: Dict[str, Dict[str, int]]) -> str:
    lines = ["        " + "  ".join(f"{p[:8]:>10}" for p in LABELS)]
    for t in LABELS:
        row = f"  {t[:8]:<8}" + "  ".join(f"{cm[t][p]:>10}" for p in LABELS)
        lines.append(row)
    return "\n".join(lines)


def print_result(label: str, r: Dict):
    print(f"\n[{label}]")
    print(f"  weights  problem={r['w_p']:.2f} method={r['w_m']:.2f} "
          f"domain={r['w_d']:.2f} contrib={r['w_c']:.2f}")
    print(f"  alpha={r['alpha']:.2f} (paper-max)  "
          f"beta={r['beta']:.2f} (global-max)  "
          f"gamma={r['gamma']:.1f} (curve-power)")
    print(f"  thresholds  t_low={r['t_low']}  t_high={r['t_high']}")
    print(f"  macro_f1={r['macro_f1']:.3f}  accuracy={r['accuracy']:.3f}  "
          f"balanced_acc={r['balanced_acc']:.3f}")
    print(f"  F1 per class: novel={r['f1_novel']:.3f}  "
          f"incremental={r['f1_incremental']:.3f}  "
          f"already_exists={r['f1_already_exists']:.3f}")
    print(f"  confusion (rows=true, cols=predicted):")
    print(format_confusion(r["confusion"]))


def save_csv(results: List[Dict], path: Path, top_n: int = 100):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "macro_f1", "accuracy", "balanced_acc",
        "f1_novel", "f1_incremental", "f1_already_exists",
        "w_p", "w_m", "w_d", "w_c",
        "alpha", "beta", "gamma", "t_low", "t_high",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results[:top_n]:
            w.writerow({k: round(r[k], 4) if isinstance(r[k], float) else r[k] for k in fields})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    source: str,
    top_n: int,
    output: Path,
    dataset_json: Optional[Path] = None,
    arxiv_results_dir: Optional[Path] = None,
):
    print(f"\nLoading cases for source={source} ...")
    if source == "arxiv" and dataset_json is None:
        env_ds = os.environ.get("BENCHMARK_ARXIV_DATASET", "").strip()
        if env_ds:
            dataset_json = Path(env_ds).expanduser().resolve()
    res_override = arxiv_results_dir if source == "arxiv" else None
    cases = load_cases(source, dataset_json=dataset_json, results_dir=res_override)
    label_dist = {l: sum(1 for c in cases if c.true_label == l) for l in LABELS}
    print(f"Loaded {len(cases)} cases. Label distribution: {label_dist}")

    diagnose_layer1(cases)

    current = evaluate_config(cases, CURRENT_CONFIG)
    print_result("CURRENT CONFIG (old formula, old params)", current)

    # ---- PREFERRED SIMPLE FORMULA ----
    preferred = evaluate_new_formula(
        cases,
        weights={"p": 0.15, "d": 0.10, "m": 0.30, "c": 0.45},
        evidence_threshold=0.85,   # unused in mean mode
        thresholds=(40, 70),
        curve="power", gamma=2.0,
        global_mode="mean",
    )
    print("\n" + "=" * 72)
    print("  [Production uses weighted-mean per paper, max across papers, power γ=1.0; this script is for sweeps only]")
    print("=" * 72)
    print(f"  paper_similarity = 0.15·p + 0.10·d + 0.30·m + 0.45·c")
    print(f"  (legacy sweep) global_similarity = mean(5 paper sims) — not production")
    print(f"  originality = (1 - global_sim^2.0) × 100")
    print(f"  thresholds = (40, 70)")
    print(f"  macro_f1={preferred['macro_f1']:.3f}  "
          f"accuracy={preferred['accuracy']:.3f}  "
          f"balanced_acc={preferred['balanced_acc']:.3f}")
    print(f"  F1 per class: novel={preferred['f1_novel']:.3f}  "
          f"incremental={preferred['f1_incremental']:.3f}  "
          f"already_exists={preferred['f1_already_exists']:.3f}")
    print(f"  confusion (rows=true, cols=predicted):")
    print(format_confusion(preferred["confusion"]))

    # ---- NEW FORMULA ----
    new_default = evaluate_new_formula(cases)
    print("\n" + "=" * 72)
    print("  NEW FORMULA — user-proposed 2-branch global + piecewise linear")
    print("=" * 72)
    print(f"\n[NEW FORMULA — DEFAULT]")
    print(f"  weights  problem={new_default['weights']['p']:.2f} "
          f"domain={new_default['weights']['d']:.2f} "
          f"method={new_default['weights']['m']:.2f} "
          f"contrib={new_default['weights']['c']:.2f}")
    print(f"  evidence_threshold={new_default['evidence_threshold']:.2f}  "
          f"thresholds={new_default['thresholds']}")
    print(f"  macro_f1={new_default['macro_f1']:.3f}  "
          f"accuracy={new_default['accuracy']:.3f}  "
          f"balanced_acc={new_default['balanced_acc']:.3f}")
    print(f"  F1 per class: novel={new_default['f1_novel']:.3f}  "
          f"incremental={new_default['f1_incremental']:.3f}  "
          f"already_exists={new_default['f1_already_exists']:.3f}")
    print(f"  confusion (rows=true, cols=predicted):")
    print(format_confusion(new_default["confusion"]))

    # Small sweep over new-formula knobs
    new_results = sweep_new_formula(cases)
    new_results.sort(key=lambda r: (r["macro_f1"], r["balanced_acc"]), reverse=True)
    best_new = new_results[0]

    # Best per curve type
    curve_names = ["linear", "piecewise", "power"]
    best_per_curve = {}
    for cn in curve_names:
        candidates = [r for r in new_results if r["curve"] == cn]
        if candidates:
            best_per_curve[cn] = candidates[0]  # already sorted

    print(f"\n[NEW FORMULA — BEST PER CURVE TYPE]")
    for cn in curve_names:
        r = best_per_curve.get(cn)
        if not r:
            continue
        curve_label = cn if cn != "power" else f"power (γ={r['gamma']})"
        print(f"\n  {curve_label.upper()}")
        print(f"    weights  p={r['weights']['p']:.2f} d={r['weights']['d']:.2f} "
              f"m={r['weights']['m']:.2f} c={r['weights']['c']:.2f}")
        print(f"    global_mode={r['global_mode']}  "
              f"evidence_threshold={r['evidence_threshold']:.2f}  "
              f"thresholds={r['thresholds']}")
        print(f"    macro_f1={r['macro_f1']:.3f}  accuracy={r['accuracy']:.3f}  "
              f"balanced_acc={r['balanced_acc']:.3f}")
        print(f"    F1: novel={r['f1_novel']:.3f}  "
              f"incremental={r['f1_incremental']:.3f}  "
              f"already_exists={r['f1_already_exists']:.3f}")

    # Best per global_mode type
    print(f"\n[NEW FORMULA — BEST PER GLOBAL_MODE]")
    for gm in ["2branch", "3branch", "mean", "max"]:
        candidates = [r for r in new_results if r["global_mode"] == gm]
        if not candidates:
            continue
        r = candidates[0]
        curve_label = r["curve"] if r["curve"] != "power" else f"power (γ={r['gamma']})"
        extra = f"  outlier_gap={r['outlier_gap']}" if gm == "3branch" else ""
        print(f"\n  {gm.upper()}{extra}")
        print(f"    weights  p={r['weights']['p']:.2f} d={r['weights']['d']:.2f} "
              f"m={r['weights']['m']:.2f} c={r['weights']['c']:.2f}")
        print(f"    curve={curve_label}  "
              f"evidence_threshold={r['evidence_threshold']:.2f}  "
              f"thresholds={r['thresholds']}")
        print(f"    macro_f1={r['macro_f1']:.3f}  accuracy={r['accuracy']:.3f}  "
              f"balanced_acc={r['balanced_acc']:.3f}")
        print(f"    F1: novel={r['f1_novel']:.3f}  "
              f"incremental={r['f1_incremental']:.3f}  "
              f"already_exists={r['f1_already_exists']:.3f}")

    print(f"\n[NEW FORMULA — OVERALL BEST of {len(new_results)} small-sweep configs]")
    curve_label = best_new["curve"] if best_new["curve"] != "power" else f"power (γ={best_new['gamma']})"
    print(f"  global_mode={best_new['global_mode']}  curve={curve_label}")
    print(f"  weights  problem={best_new['weights']['p']:.2f} "
          f"domain={best_new['weights']['d']:.2f} "
          f"method={best_new['weights']['m']:.2f} "
          f"contrib={best_new['weights']['c']:.2f}")
    print(f"  evidence_threshold={best_new['evidence_threshold']:.2f}  "
          f"thresholds={best_new['thresholds']}")
    print(f"  macro_f1={best_new['macro_f1']:.3f}  "
          f"accuracy={best_new['accuracy']:.3f}  "
          f"balanced_acc={best_new['balanced_acc']:.3f}")
    print(f"  F1 per class: novel={best_new['f1_novel']:.3f}  "
          f"incremental={best_new['f1_incremental']:.3f}  "
          f"already_exists={best_new['f1_already_exists']:.3f}")
    print(f"  confusion (rows=true, cols=predicted):")
    print(format_confusion(best_new["confusion"]))

    # ---- OLD FORMULA SWEEP (for comparison) ----
    all_results = sweep(cases)
    all_results.sort(key=lambda r: (r["macro_f1"], r["balanced_acc"]), reverse=True)

    print("\n" + "=" * 72)
    print(f"  TOP {top_n} CONFIGURATIONS BY MACRO-F1")
    print("=" * 72)
    for i, r in enumerate(all_results[:top_n], 1):
        print_result(f"RANK {i}", r)

    save_csv(all_results, output, top_n=500)
    print(f"\nSaved top-500 configs to {output}")

    # ---- FINAL COMPARISON ----
    print("\n" + "=" * 72)
    print("  FINAL COMPARISON  (macro-F1, higher is better)")
    print("=" * 72)
    print(f"  Current config (OLD formula, OLD params):     {current['macro_f1']:.3f}")
    print(f"  OLD formula — best sweep config:              {all_results[0]['macro_f1']:.3f}")
    print(f"  NEW formula — default weights:                {new_default['macro_f1']:.3f}")
    print(f"  NEW formula — best small-sweep config:        {best_new['macro_f1']:.3f}")
    print(f"  sweep baseline (wmean + mean + power γ=2.0): {preferred['macro_f1']:.3f}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Layer 2 recalibration sweep")
    parser.add_argument("--source", default="arxiv", choices=list(SOURCE_CONFIG.keys()))
    parser.add_argument("--top", type=int, default=5, help="Top-N configs to print")
    parser.add_argument(
        "--output", type=Path,
        default=BASE_DIR / "recalibration_results.csv",
        help="CSV output for top-500 configs",
    )
    parser.add_argument(
        "--arxiv-dataset",
        type=Path,
        default=None,
        help="arXiv benchmark JSON (default: datasets/hypothetica_benchmark_arxiv_gold_clean.json).",
    )
    parser.add_argument(
        "--arxiv-results",
        type=Path,
        default=None,
        help=(
            "Directory with per-case raw results (*.json) for arxiv, e.g. "
            "benchmarks/results_arxiv_gold_clean/arxiv"
        ),
    )
    args = parser.parse_args()
    run(
        args.source,
        args.top,
        args.output,
        dataset_json=args.arxiv_dataset,
        arxiv_results_dir=args.arxiv_results,
    )


if __name__ == "__main__":
    main()
