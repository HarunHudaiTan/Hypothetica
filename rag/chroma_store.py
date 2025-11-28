"""
ChromaDB vector store for paper chunks.
Handles embedding storage and similarity search.
"""
import logging
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

import config
from models.paper import Paper, Chunk

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    ChromaDB-based vector store for paper chunks.
    Supports in-memory storage for demo, with optional persistence.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_dir: str = None,
        embedding_model: str = None
    ):
        """
        Initialize ChromaDB store.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_dir: Directory for persistence (None = in-memory)
            embedding_model: SentenceTransformer model name
        """
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=config.EMBEDDING_DEVICE
        )
        
        # Initialize ChromaDB client
        if self.persist_dir:
            logger.info(f"Initializing persistent ChromaDB at: {self.persist_dir}")
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            logger.info("Initializing in-memory ChromaDB")
            self.client = chromadb.Client()
        
        # Create/get collection with custom embedding function
        self._init_collection()
        
    def _init_collection(self):
        """Initialize or get the ChromaDB collection."""
        # Delete existing collection if it exists (for fresh demo runs)
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        logger.info(f"Created collection: {self.collection_name}")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using E5 model.
        E5 models expect "passage: " prefix for documents.
        """
        # Add E5-style prefix for passages
        if 'e5' in self.embedding_model_name.lower():
            texts = [f"passage: {t}" for t in texts]
        
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        E5 models expect "query: " prefix for queries.
        """
        if 'e5' in self.embedding_model_name.lower():
            query = f"query: {query}"
        
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def add_paper(self, paper: Paper) -> int:
        """
        Add all chunks from a paper to the store.
        
        Args:
            paper: Paper object with extracted chunks
            
        Returns:
            Number of chunks added
        """
        if not paper.headings:
            logger.warning(f"Paper {paper.paper_id} has no headings")
            return 0
        
        chunks = paper.valid_chunks
        if not chunks:
            logger.warning(f"Paper {paper.paper_id} has no valid chunks")
            return 0
        
        # Prepare data for ChromaDB
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []
        
        for chunk in chunks:
            # Find parent heading
            heading = next(
                (h for h in paper.headings if h.index == chunk.heading_index),
                None
            )
            
            metadatas.append({
                "chunk_id": chunk.chunk_id,
                "paper_id": paper.paper_id,
                "arxiv_id": paper.arxiv_id,
                "paper_title": paper.title,
                "heading": heading.text if heading else "",
                "heading_index": chunk.heading_index,
                "chunk_index": chunk.chunk_index,
                "categories": ",".join(paper.categories),
                "abstract": paper.abstract[:500] if paper.abstract else "",
            })
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} chunks from {paper.paper_id}")
        embeddings = self._embed_texts(documents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        logger.info(f"Added {len(chunks)} chunks from paper {paper.paper_id}")
        return len(chunks)
    
    def add_papers(self, papers: List[Paper]) -> int:
        """
        Add chunks from multiple papers.
        
        Args:
            papers: List of Paper objects
            
        Returns:
            Total number of chunks added
        """
        total = 0
        for paper in papers:
            total += self.add_paper(paper)
        return total
    
    def search(
        self,
        query: str,
        n_results: int = None,
        filter_paper_id: str = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_paper_id: Optional filter by paper ID
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of result dicts with chunk info and similarity scores
        """
        n_results = n_results or config.RAG_TOP_K
        
        # Build where filter
        where_filter = None
        if filter_paper_id:
            where_filter = {"paper_id": filter_paper_id}
        
        # Generate query embedding
        query_embedding = self._embed_query(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results and results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                # Convert distance to similarity (ChromaDB returns distances)
                # For cosine distance: similarity = 1 - distance
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance
                
                formatted.append({
                    "chunk_id": chunk_id,
                    "text": results['documents'][0][i] if results['documents'] else "",
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "similarity": similarity,
                    "distance": distance
                })
        
        return formatted
    
    def search_by_sentence(
        self,
        sentence: str,
        n_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks matching a specific sentence.
        Wrapper around search() for sentence-level RAG.
        
        Args:
            sentence: User's sentence to find matches for
            n_results: Number of results
            
        Returns:
            List of matching chunks with similarity scores
        """
        return self.search(query=sentence, n_results=n_results)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: The chunk ID to retrieve
            
        Returns:
            Chunk data dict or None if not found
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if result and result['ids']:
                return {
                    "chunk_id": result['ids'][0],
                    "text": result['documents'][0] if result['documents'] else "",
                    "metadata": result['metadatas'][0] if result['metadatas'] else {}
                }
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
        
        return None
    
    def get_chunks_by_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific paper.
        
        Args:
            paper_id: Paper ID to filter by
            
        Returns:
            List of chunk data dicts
        """
        try:
            result = self.collection.get(
                where={"paper_id": paper_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if result and result['ids']:
                for i, chunk_id in enumerate(result['ids']):
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": result['documents'][i] if result['documents'] else "",
                        "metadata": result['metadatas'][i] if result['metadatas'] else {}
                    })
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks for paper {paper_id}: {e}")
            return []
    
    def count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()
    
    def clear(self):
        """Clear all data from the collection."""
        self._init_collection()
        logger.info("Cleared all data from ChromaDB collection")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.count(),
            "embedding_model": self.embedding_model_name,
            "persist_dir": self.persist_dir or "in-memory"
        }

