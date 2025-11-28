"""
Retriever module for RAG queries.
Provides high-level interface for finding relevant content.
"""
import logging
from typing import List, Dict, Any, Optional

import config
from rag.chroma_store import ChromaStore
from models.analysis import MatchedSection

logger = logging.getLogger(__name__)


class Retriever:
    """
    High-level retriever for originality analysis.
    Wraps ChromaStore with analysis-specific functionality.
    """
    
    def __init__(self, store: ChromaStore):
        """
        Initialize retriever with a ChromaStore.
        
        Args:
            store: ChromaStore instance to use for retrieval
        """
        self.store = store
    
    def find_matches_for_sentence(
        self,
        sentence: str,
        top_k: int = None,
        similarity_threshold: float = 0.3
    ) -> List[MatchedSection]:
        """
        Find matching paper sections for a user's sentence.
        
        Args:
            sentence: The sentence to find matches for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity to include
            
        Returns:
            List of MatchedSection objects
        """
        top_k = top_k or config.RAG_TOP_K
        
        # Search ChromaDB
        results = self.store.search(query=sentence, n_results=top_k)
        
        # Filter by threshold and convert to MatchedSection
        matches = []
        for result in results:
            if result['similarity'] >= similarity_threshold:
                metadata = result.get('metadata', {})
                
                matches.append(MatchedSection(
                    chunk_id=result['chunk_id'],
                    paper_id=metadata.get('paper_id', ''),
                    paper_title=metadata.get('paper_title', ''),
                    heading=metadata.get('heading', ''),
                    text_snippet=result['text'][:500],  # Truncate for display
                    similarity=result['similarity'],
                    reason=f"Semantic similarity: {result['similarity']:.2f}"
                ))
        
        return matches
    
    def find_matches_for_idea(
        self,
        idea: str,
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[MatchedSection]:
        """
        Find matching sections for the full user idea.
        Useful for overall relevance assessment.
        
        Args:
            idea: Full user research idea
            top_k: Number of results
            similarity_threshold: Minimum similarity
            
        Returns:
            List of MatchedSection objects
        """
        return self.find_matches_for_sentence(
            sentence=idea,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
    
    def get_context_for_paper(
        self,
        paper_id: str,
        query: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context from a specific paper.
        If query provided, returns most relevant chunks.
        Otherwise returns all chunks.
        
        Args:
            paper_id: Paper to get context from
            query: Optional query to filter by relevance
            
        Returns:
            List of chunk data
        """
        if query:
            # Search within paper
            results = self.store.search(
                query=query,
                n_results=config.RAG_TOP_K,
                filter_paper_id=paper_id
            )
            return results
        else:
            # Return all chunks for paper
            return self.store.get_chunks_by_paper(paper_id)
    
    def get_evidence_for_match(
        self,
        chunk_id: str,
        expand_context: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed evidence for a specific match.
        Optionally includes surrounding context.
        
        Args:
            chunk_id: The chunk ID to get evidence for
            expand_context: Whether to include neighboring chunks
            
        Returns:
            Dict with chunk text, metadata, and optional context
        """
        chunk = self.store.get_chunk_by_id(chunk_id)
        if not chunk:
            return {}
        
        result = {
            "chunk_id": chunk_id,
            "text": chunk['text'],
            "metadata": chunk['metadata']
        }
        
        if expand_context:
            # Try to get surrounding chunks for context
            metadata = chunk.get('metadata', {})
            paper_id = metadata.get('paper_id', '')
            heading_index = metadata.get('heading_index', 0)
            chunk_index = metadata.get('chunk_index', 0)
            
            # Get previous and next chunk IDs
            prev_id = f"{paper_id}_h{heading_index:02d}_c{chunk_index-1:02d}"
            next_id = f"{paper_id}_h{heading_index:02d}_c{chunk_index+1:02d}"
            
            prev_chunk = self.store.get_chunk_by_id(prev_id)
            next_chunk = self.store.get_chunk_by_id(next_id)
            
            if prev_chunk:
                result['context_before'] = prev_chunk['text']
            if next_chunk:
                result['context_after'] = next_chunk['text']
        
        return result
    
    def batch_search_sentences(
        self,
        sentences: List[str],
        top_k_per_sentence: int = 3
    ) -> Dict[int, List[MatchedSection]]:
        """
        Search for matches for multiple sentences at once.
        
        Args:
            sentences: List of sentences to search
            top_k_per_sentence: Results per sentence
            
        Returns:
            Dict mapping sentence index to list of matches
        """
        results = {}
        for idx, sentence in enumerate(sentences):
            matches = self.find_matches_for_sentence(
                sentence=sentence,
                top_k=top_k_per_sentence
            )
            results[idx] = matches
        
        return results
    
    def compute_idea_paper_similarity(
        self,
        idea: str,
        paper_id: str
    ) -> float:
        """
        Compute overall similarity between user idea and a paper.
        Uses the best matching chunk's similarity as proxy.
        
        Args:
            idea: User's research idea
            paper_id: Paper to compare against
            
        Returns:
            Similarity score (0-1)
        """
        results = self.store.search(
            query=idea,
            n_results=1,
            filter_paper_id=paper_id
        )
        
        if results:
            return results[0]['similarity']
        return 0.0

