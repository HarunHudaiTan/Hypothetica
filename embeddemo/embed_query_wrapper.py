#!/usr/bin/env python3
"""
Wrapper for embed_mvp.py to run queries programmatically without command line args.
"""
import os
import sys
import subprocess
from typing import List, Dict, Any, Optional
import json

class QueryWrapper:
    def query_embeddings(
        self,
        query_text: str,
        backend: str = "st",
        model: str = "intfloat/e5-base-v2",
        index_dir: str = "index_dir",
        topk: int = 5,
        device: Optional[str] = "mps",
        cache_path: str = "./.embed_cache/cache.sqlite3"
    ) -> List[Dict[str, Any]]:
        """
        Query the embedding index programmatically.

        Args:
            query_text: The text to search for
            backend: Either "st" (SentenceTransformers) or "openai"
            model: Model name (e.g., "intfloat/e5-base-v2")
            index_dir: Path to the index directory
            topk: Number of results to return
            device: Device for computation (e.g., "mps", "cuda", "cpu")
            cache_path: Path to embedding cache

        Returns:
            List of search results with metadata
        """
        # Build the command
        cmd = [
            sys.executable,
            "embed_mvp.py",
            "query",
            query_text,
            "--backend", backend,
            "--model", model,
            "--index", index_dir,
            "--topk", str(topk),
            "--cache", cache_path
        ]

        if device:
            cmd.extend(["--device", device])

        try:
            # Run the command and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            # Parse the JSON output
            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"Error running query: {e}")
            print(f"Stderr: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output: {e}")
            print(f"Raw output: {result.stdout}")
            raise

    def build_index(self,
        jsonl_path: str,
        backend: str = "st",
        model: str = "intfloat/e5-base-v2",
        output_dir: str = "./index_dir",
        device: Optional[str] = "mps",
        cache_path: str = "./.embed_cache/cache.sqlite3"
    ) -> None:
        """
        Build an embedding index programmatically.

        Args:
            jsonl_path: Path to the JSONL file with papers
            backend: Either "st" (SentenceTransformers) or "openai"
            model: Model name
            output_dir: Where to save the index
            device: Device for computation
            cache_path: Path to embedding cache
        """
        cmd = [
            sys.executable,
            "embed_mvp.py",
            "build",
            jsonl_path,
            "--backend", backend,
            "--model", model,
            "--out", output_dir,
            "--cache", cache_path
        ]

        if device:
            cmd.extend(["--device", device])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"Error building index: {e}")
            print(f"Stderr: {e.stderr}")
            raise

    def search_literature(self, query: str = "novelty in retrieval-augmented literature mapping", include_scores: bool = True) -> str:
        """
        Single method to search literature and return JSON results.

        Args:
            query: The search query text
            include_scores: Whether to include similarity scores in the results

        Returns:
            JSON string with search results
        """
        print("users idea"+f"{query}")
        try:
            # Get the absolute path to sample_papers.jsonl
            script_dir = os.path.dirname(os.path.abspath(__file__))
            jsonl_path = os.path.join(script_dir, "sample_papers.jsonl")
            index_dir = os.path.join(script_dir, "index_dir")

            # Clean index directory to ensure fresh rebuild
            if os.path.exists(index_dir):
                import shutil
                shutil.rmtree(index_dir)
            os.makedirs(index_dir, exist_ok=True)

            # Rebuild index every time to ensure we have the full dataset
            self.build_index(
                jsonl_path=jsonl_path,
                backend="st",
                model="intfloat/e5-base-v2",
                output_dir=index_dir,
                device="mps"
            )

            results = self.query_embeddings(
                query_text=query,
                backend="st",
                model="intfloat/e5-base-v2",
                index_dir=index_dir,
                topk=15,
                device="mps"
            )

            # Remove scores if not requested
            if not include_scores:
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and 'score' in result:
                            del result['score']
                elif isinstance(results, dict) and 'results' in results:
                    for result in results['results']:
                        if isinstance(result, dict) and 'score' in result:
                            del result['score']

            # Return as formatted JSON string
            return json.dumps(results, indent=2, ensure_ascii=False)

        except Exception as e:
            error_result = {
                "error": str(e),
                "query": query,
                "results": []
            }
            return json.dumps(error_result, indent=2, ensure_ascii=False)

def main():
    """Main method that calls the single search method"""

    prompt=''' Theoretical Bounds on Sample Complexity for Few-Shot Learning

I'm exploring the theoretical foundations of few-shot learning - specifically, what are 
the fundamental limits on how few examples are needed to learn a new task? I want to 
derive sample complexity bounds that depend on task similarity, model capacity, and the 
structure of the meta-learning algorithm. This could help explain why certain meta-learning 
architectures (like MAML or Prototypical Networks) work better than others and guide the 
design of more sample-efficient algorithms.'''
    wrapper = QueryWrapper()
    result_json = wrapper.search_literature(prompt)
    print(result_json)


# if __name__ == "__main__":
#
#
#     main()
