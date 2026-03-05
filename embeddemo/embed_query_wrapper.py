#!/usr/bin/env python3
"""
Enhanced embedding query wrapper with:
- Persistent embedding cache (no rebuild every run)
- Cross-encoder reranking for precision
- Large pool semantic search
"""
import os
import sys
import json
import shutil
from typing import List, Dict, Any, Optional

import config


class QueryWrapper:
    """
    Wrapper for embedding-based paper search with caching and reranking.
    """
    
    def __init__(self, 
                 backend: str = "st",
                 model: str = "intfloat/e5-base-v2",
                 device: str = None,
                 cache_path: str = None,
                 index_dir: str = None,
                 use_reranker: bool = True,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the query wrapper.
        
        Args:
            backend: "st" for SentenceTransformers or "openai"
            model: Model name for embeddings
            device: Device for computation ("mps", "cuda", "cpu"). If None, uses config.EMBEDDING_DEVICE
            cache_path: Path to embedding cache SQLite file
            index_dir: Directory for FAISS index
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: Cross-encoder model for reranking
        """
        self.backend = backend
        self.model = model
        self.device = device or config.EMBEDDING_DEVICE
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        
        # Set default paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_path = cache_path or os.path.join(script_dir, ".embed_cache", "cache.sqlite3")
        self.index_dir = index_dir or os.path.join(script_dir, "index_dir")
        self.jsonl_path = os.path.join(script_dir, "sample_papers.jsonl")
        
        # Lazy load reranker
        self._reranker = None
        
    def _get_reranker(self):
        """Lazy load cross-encoder reranker."""
        if self._reranker is None and self.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading reranker: {self.reranker_model}")
                self._reranker = CrossEncoder(self.reranker_model, device=self.device)
            except Exception as e:
                print(f"Warning: Could not load reranker: {e}")
                self._reranker = False  # Mark as unavailable
        return self._reranker if self._reranker else None

    def _get_embedding_backend(self):
        """Create embedding backend."""
        # Import from embed_mvp
        from embeddemo.embed_mvp import STBackend, OpenAIBackend
        
        if self.backend == "st":
            return STBackend(self.model, device=self.device)
        else:
            return OpenAIBackend(self.model)
    
    def _get_pipeline(self):
        """Create embedding pipeline."""
        from embeddemo.embed_mvp import EmbedPipeline
        
        backend = self._get_embedding_backend()
        return EmbedPipeline(backend, cache_path=self.cache_path, out_dir=self.index_dir)
    
    def _load_papers_from_jsonl(self) -> List[Dict]:
        """Load papers from JSONL file."""
        papers = []
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        papers.append(json.loads(line))
        return papers
    
    def _get_cached_paper_ids(self) -> set:
        """Get set of paper IDs already in the index."""
        meta_path = os.path.join(self.index_dir, "meta.jsonl")
        cached_ids = set()
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            paper = json.loads(line)
                            cached_ids.add(paper.get('id', ''))
                        except:
                            pass
        return cached_ids
    
    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Build or update the embedding index.
        Only embeds new papers not already in cache.
        
        Args:
            force_rebuild: If True, rebuild entire index from scratch
            
        Returns:
            Number of papers in index
        """
        from embeddemo.embed_mvp import PaperDoc, FaissIndexer, EmbedPipeline
        
        papers = self._load_papers_from_jsonl()
        
        if not papers:
            print("No papers found in JSONL file")
            return 0
        
        if force_rebuild and os.path.exists(self.index_dir):
            print("Force rebuild: removing existing index...")
            shutil.rmtree(self.index_dir)
        
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Check which papers are already indexed
        cached_ids = self._get_cached_paper_ids() if not force_rebuild else set()
        new_papers = [p for p in papers if p.get('id', '') not in cached_ids]
        
        if not new_papers and cached_ids:
            print(f"All {len(papers)} papers already indexed. Skipping build.")
            return len(papers)
        
        print(f"Building index: {len(new_papers)} new papers (total: {len(papers)})")
        
        # If we have cached papers and new ones, we need to rebuild
        # because FAISS doesn't support incremental adds well with our setup
        if cached_ids and new_papers:
            print("Index update needed - rebuilding full index...")
            if os.path.exists(self.index_dir):
                shutil.rmtree(self.index_dir)
            os.makedirs(self.index_dir, exist_ok=True)
            papers_to_index = papers
        else:
            papers_to_index = new_papers if new_papers else papers
        
        # Convert to PaperDoc objects
        docs = [
            PaperDoc(
                id=str(p.get('id', '')),
                title=p.get('title', ''),
                abstract=p.get('abstract', ''),
                url=p.get('url'),
                year=p.get('year'),
                categories=p.get('categories'),
            )
            for p in papers_to_index
        ]
        
        # Build index
        pipeline = self._get_pipeline()
        indexer, count = pipeline.build(docs)
        
        print(f"Index built with {count} papers")
        return count
    
    def query_embeddings(self, query_text: str, topk: int = 100) -> List[Dict]:
        """
        Query the embedding index.
        
        Args:
            query_text: Search query
            topk: Number of results to return
            
        Returns:
            List of paper results with scores
        """
        pipeline = self._get_pipeline()
        
        try:
            results = pipeline.query(
                index_dir=self.index_dir,
                queries=[query_text],
                topk=topk
            )
            return results[0] if results else []
        except Exception as e:
            print(f"Error querying index: {e}")
            return []
    
    def rerank_results(self, query: str, results: List[Dict], topk: int = 30) -> List[Dict]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Original query
            results: List of paper results from embedding search
            topk: Number of top results to return after reranking
            
        Returns:
            Reranked list of papers
        """
        reranker = self._get_reranker()
        
        if not reranker or not results:
            return results[:topk]
        
        print(f"Reranking {len(results)} results with cross-encoder...")
        
        # Prepare pairs for cross-encoder
        pairs = []
        for paper in results:
            # Combine title and abstract for reranking
            doc_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            pairs.append([query, doc_text])
        
        # Get cross-encoder scores
        try:
            scores = reranker.predict(pairs, show_progress_bar=False)
            
            # Add rerank scores and sort
            for i, paper in enumerate(results):
                paper['rerank_score'] = float(scores[i])
                paper['embedding_score'] = paper.get('score', 0)
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            print(f"Reranking complete. Top score: {reranked[0].get('rerank_score', 0):.4f}")
            return reranked[:topk]
            
        except Exception as e:
            print(f"Warning: Reranking failed: {e}")
            return results[:topk]
    
    def search_literature(self, 
                         query: str, 
                         include_scores: bool = True,
                         embedding_topk: int = 100,
                         rerank_topk: int = 30,
                         force_rebuild: bool = False) -> str:
        """
        Main search method: build index if needed, search, and rerank.
        
        Args:
            query: Search query text
            include_scores: Whether to include similarity scores
            embedding_topk: Number of results from embedding search
            rerank_topk: Number of results after reranking
            force_rebuild: Force rebuild of index
            
        Returns:
            JSON string with search results
        """
        print(f"\n{'='*60}")
        print(f"Searching literature for: {query[:100]}...")
        
        try:
            # Step 1: Build/update index
            self.build_index(force_rebuild=force_rebuild)
            
            # Step 2: Embedding search (high recall)
            print(f"\nStep 1: Embedding search (top {embedding_topk})...")
            results = self.query_embeddings(query, topk=embedding_topk)
            
            if not results:
                return json.dumps({"error": "No results found", "query": query, "results": []})
            
            print(f"Found {len(results)} candidates from embedding search")
            
            # Step 3: Rerank for precision
            if self.use_reranker:
                print(f"\nStep 2: Cross-encoder reranking (top {rerank_topk})...")
                results = self.rerank_results(query, results, topk=rerank_topk)
            else:
                results = results[:rerank_topk]
            
            # Remove scores if not requested
            if not include_scores:
                for result in results:
                    result.pop('score', None)
                    result.pop('rerank_score', None)
                    result.pop('embedding_score', None)
            
            print(f"\nReturning {len(results)} papers")
            return json.dumps(results, indent=2, ensure_ascii=False)
            
        except Exception as e:
            import traceback
            error_result = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "query": query,
                "results": []
            }
            return json.dumps(error_result, indent=2, ensure_ascii=False)


def main():
    """Test the search functionality."""
    prompt = '''Theoretical Bounds on Sample Complexity for Few-Shot Learning

I'm exploring the theoretical foundations of few-shot learning - specifically, what are 
the fundamental limits on how few examples are needed to learn a new task? I want to 
derive sample complexity bounds that depend on task similarity, model capacity, and the 
structure of the meta-learning algorithm.'''

    wrapper = QueryWrapper(device=config.EMBEDDING_DEVICE, use_reranker=True)
    result_json = wrapper.search_literature(prompt, include_scores=True)
    
    results = json.loads(result_json)
    if isinstance(results, list):
        print(f"\n{'='*60}")
        print(f"Top 5 results:")
        for i, paper in enumerate(results[:5]):
            print(f"\n{i+1}. {paper.get('title', 'No title')}")
            print(f"   Score: {paper.get('rerank_score', paper.get('score', 'N/A'))}")
            print(f"   URL: {paper.get('url', 'N/A')}")


if __name__ == "__main__":
    main()
