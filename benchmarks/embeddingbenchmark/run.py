#!/usr/bin/env python3
"""
Run retrieval benchmarks (embedding + reranker).
Usage: python benchmarks/embeddingbenchmark/run.py [--embedding] [--reranker] [--all]
"""
import sys
import argparse
from pathlib import Path

# Add project root and backend to path
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "backend"))


def main():
    parser = argparse.ArgumentParser(description="Run retrieval benchmarks")
    parser.add_argument("--embedding", action="store_true", help="Run embedding benchmark")
    parser.add_argument("--reranker", action="store_true", help="Run reranker benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    run_all = args.all or (not args.embedding and not args.reranker)

    if args.embedding or run_all:
        print("\n>>> Embedding benchmark\n")
        from benchmarks.embeddingbenchmark.performance_benchmark import benchmark
        benchmark()
        print()

    if args.reranker or run_all:
        print("\n>>> Reranker benchmark\n")
        from benchmarks.embeddingbenchmark.reranker_benchmark import benchmark as reranker_benchmark
        reranker_benchmark()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
