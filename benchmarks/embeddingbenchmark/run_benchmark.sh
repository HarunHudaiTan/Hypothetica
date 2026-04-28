#!/bin/bash
# Run retrieval benchmarks (embedding + reranker) using the hypothetica conda environment.
# Usage: ./run_benchmark.sh [embedding|reranker|all]
set -e
cd "$(dirname "$0")/../.."

RUN="${1:-all}"

case "$RUN" in
  embedding)
    conda run -n hypothetica python benchmarks/embeddingbenchmark/performance_benchmark.py
    ;;
  reranker)
    conda run -n hypothetica python benchmarks/embeddingbenchmark/reranker_benchmark.py
    ;;
  all)
    echo ">>> Embedding benchmark"
    conda run -n hypothetica python benchmarks/embeddingbenchmark/performance_benchmark.py
    echo ""
    echo ">>> Reranker benchmark"
    conda run -n hypothetica python benchmarks/embeddingbenchmark/reranker_benchmark.py
    ;;
  *)
    echo "Usage: $0 [embedding|reranker|all]"
    exit 1
    ;;
esac
