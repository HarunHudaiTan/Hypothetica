#!/usr/bin/env python3
"""
Performance benchmark for the embedding model.
Measures throughput (texts/sec) and latency (ms per query).
"""
import sys
import time
from pathlib import Path

# Add backend to path when run from project root
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "backend"))

from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL, EMBEDDING_DEVICE


def benchmark(
    model_name: str = EMBEDDING_MODEL,
    device: str = EMBEDDING_DEVICE,
    num_passages: int = 100,
    num_queries: int = 10,
    batch_size: int = 32,
) -> dict:
    """
    Run performance benchmark on the embedding model.

    Returns:
        Dict with throughput and latency metrics.
    """
    use_e5 = "e5" in model_name.lower()
    prefix_passage = "passage: " if use_e5 else ""
    prefix_query = "query: " if use_e5 else ""

    print(f"Loading model: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)

    # Sample texts similar to paper abstracts
    sample_abstract = (
        "This paper presents a novel approach to machine learning for drug discovery. "
        "We propose a graph neural network that captures molecular structure. "
        "Experiments on MoleculeNet benchmarks show state-of-the-art results."
    )
    passages = [f"{prefix_passage}{sample_abstract}"] * num_passages
    queries = [f"{prefix_query}neural network for drug discovery"] * num_queries

    # Warmup
    print("Warmup...")
    model.encode(queries[:1], normalize_embeddings=True)

    # Passage throughput (batch encoding)
    print(f"\nEncoding {num_passages} passages (batch_size={batch_size})...")
    t0 = time.perf_counter()
    embs = model.encode(passages, batch_size=batch_size, normalize_embeddings=True)
    passage_elapsed = time.perf_counter() - t0
    passage_throughput = num_passages / passage_elapsed

    # Query latency (single encodes, simulates real-time search)
    print(f"\nEncoding {num_queries} queries (one at a time)...")
    t0 = time.perf_counter()
    for q in queries:
        model.encode([q], normalize_embeddings=True)
    query_elapsed = time.perf_counter() - t0
    query_latency_ms = (query_elapsed / num_queries) * 1000

    results = {
        "model": model_name,
        "device": device,
        "passage_throughput_per_sec": round(passage_throughput, 1),
        "passage_total_sec": round(passage_elapsed, 2),
        "query_latency_ms": round(query_latency_ms, 1),
        "embedding_dim": len(embs[0]),
    }

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Passages: {num_passages} texts in {passage_elapsed:.2f}s")
    print(f"           → {passage_throughput:.1f} texts/sec")
    print(f"  Queries:  {num_queries} single encodes in {query_elapsed:.2f}s")
    print(f"           → {query_latency_ms:.0f} ms per query")
    print(f"  Embedding dim: {results['embedding_dim']}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark embedding model performance")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Model name")
    parser.add_argument("--device", default=EMBEDDING_DEVICE, help="Device (cuda, mps, cpu)")
    parser.add_argument("--passages", type=int, default=100, help="Number of passages to encode")
    parser.add_argument("--queries", type=int, default=10, help="Number of queries for latency test")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for passage encoding")
    args = parser.parse_args()

    benchmark(
        model_name=args.model,
        device=args.device,
        num_passages=args.passages,
        num_queries=args.queries,
        batch_size=args.batch_size,
    )
