#!/usr/bin/env python3
"""
Performance benchmark for the reranker (cross-encoder) model.
Measures throughput (pairs/sec) and latency (ms per rerank call).
"""
import sys
import time
from pathlib import Path

# Add backend to path when run from project root
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "backend"))

from sentence_transformers import CrossEncoder
from core.config import RERANKER_MODEL, EMBEDDING_DEVICE


def benchmark(
    model_name: str = RERANKER_MODEL,
    device: str = EMBEDDING_DEVICE,
    num_pairs: int = 100,
    batch_size: int = 32,
) -> dict:
    """
    Run performance benchmark on the reranker model.

    Returns:
        Dict with throughput and latency metrics.
    """
    print(f"Loading reranker: {model_name} (device={device})")
    model = CrossEncoder(model_name, device=device)

    # Sample texts similar to paper search (query + title+abstract)
    query = "neural network for drug discovery"
    sample_doc = (
        "This paper presents a novel approach to machine learning for drug discovery. "
        "We propose a graph neural network that captures molecular structure. "
        "Experiments on MoleculeNet benchmarks show state-of-the-art results."
    )

    pairs = [[query, sample_doc]] * num_pairs

    # Warmup
    print("Warmup...")
    model.predict(pairs[:1], show_progress_bar=False)

    # Batch throughput (typical: 100 embedding results -> rerank)
    print(f"\nReranking {num_pairs} pairs (batch_size={batch_size})...")
    t0 = time.perf_counter()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    throughput = num_pairs / elapsed

    # Latency: single rerank call (1 pair, simulates minimal rerank)
    print(f"\nSingle pair latency (10 runs)...")
    t0 = time.perf_counter()
    for _ in range(10):
        model.predict([pairs[0]], show_progress_bar=False)
    single_elapsed = (time.perf_counter() - t0) / 10
    latency_ms = single_elapsed * 1000

    results = {
        "model": model_name,
        "device": device,
        "throughput_pairs_per_sec": round(throughput, 1),
        "total_pairs": num_pairs,
        "total_sec": round(elapsed, 2),
        "latency_ms_per_pair": round(latency_ms, 1),
    }

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Batch: {num_pairs} pairs in {elapsed:.2f}s")
    print(f"         → {throughput:.1f} pairs/sec")
    print(f"  Single pair: {latency_ms:.0f} ms")
    print("=" * 50)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark reranker performance")
    parser.add_argument("--model", default=RERANKER_MODEL, help="Cross-encoder model name")
    parser.add_argument("--device", default=EMBEDDING_DEVICE, help="Device (cuda, mps, cpu)")
    parser.add_argument("--pairs", type=int, default=100, help="Number of query-doc pairs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for scoring")
    args = parser.parse_args()

    benchmark(
        model_name=args.model,
        device=args.device,
        num_pairs=args.pairs,
        batch_size=args.batch_size,
    )
