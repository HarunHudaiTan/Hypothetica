"""
Data models for papers, headings, and chunks.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Chunk:
    """
    A chunk of text extracted from a paper section.
    Chunks are the atomic units for embedding and retrieval.
    """
    chunk_id: str                    # Unique ID: paper_id_heading_idx_chunk_idx
    paper_id: str                    # Parent paper ID
    heading: str                     # Parent heading text
    heading_index: int               # Index of heading in paper
    chunk_index: int                 # Index of chunk within heading
    text: str                        # The actual chunk text
    char_start: int                  # Start position in original section text
    char_end: int                    # End position in original section text
    
    # Quality metrics
    is_valid: bool = True            # Whether chunk meets quality thresholds
    quality_reason: Optional[str] = None  # Reason if invalid
    
    def __post_init__(self):
        """Generate chunk_id if not provided."""
        if not self.chunk_id:
            self.chunk_id = f"{self.paper_id}_h{self.heading_index:02d}_c{self.chunk_index:02d}"


@dataclass
class Heading:
    """
    A heading/section extracted from a paper.
    Contains the heading text and all chunks under it.
    """
    heading_id: str                  # Unique ID: paper_id_heading_idx
    paper_id: str                    # Parent paper ID
    index: int                       # Position in paper (0-indexed)
    level: int                       # Heading level (1-6)
    text: str                        # Heading text (cleaned)
    raw_text: str                    # Original heading text
    section_text: str                # Full text under this heading
    chunks: List[Chunk] = field(default_factory=list)
    
    # Quality metrics
    is_valid: bool = True
    quality_score: float = 1.0       # 0-1 score based on content quality
    abstract_similarity: Optional[float] = None  # Similarity to paper abstract
    
    def __post_init__(self):
        """Generate heading_id if not provided."""
        if not self.heading_id:
            self.heading_id = f"{self.paper_id}_h{self.index:02d}"


@dataclass
class Paper:
    """
    A research paper with all its extracted content.
    """
    paper_id: str                    # Internal ID (e.g., "paper_01")
    arxiv_id: str                    # ArXiv ID (e.g., "2401.12345")
    title: str
    abstract: str
    url: str                         # ArXiv URL
    pdf_url: str                     # PDF URL
    
    # Metadata
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)  # ArXiv categories
    published_date: Optional[str] = None
    
    # Extracted content
    headings: List[Heading] = field(default_factory=list)
    markdown_content: Optional[str] = None
    
    # Processing status
    is_processed: bool = False
    processing_error: Optional[str] = None
    processed_at: Optional[datetime] = None
    
    # Computed properties
    @property
    def total_chunks(self) -> int:
        """Total number of chunks across all headings."""
        return sum(len(h.chunks) for h in self.headings)
    
    @property
    def valid_chunks(self) -> List[Chunk]:
        """All valid chunks from all headings."""
        chunks = []
        for heading in self.headings:
            chunks.extend([c for c in heading.chunks if c.is_valid])
        return chunks
    
    @property
    def chunk_ids(self) -> List[str]:
        """List of all chunk IDs for ChromaDB reference."""
        return [c.chunk_id for h in self.headings for c in h.chunks]
    
    def get_chunk_metadata(self) -> List[dict]:
        """
        Get metadata dict for all chunks (for ChromaDB storage).
        """
        metadata_list = []
        for heading in self.headings:
            for chunk in heading.chunks:
                metadata_list.append({
                    "chunk_id": chunk.chunk_id,
                    "paper_id": self.paper_id,
                    "arxiv_id": self.arxiv_id,
                    "paper_title": self.title,
                    "heading": heading.text,
                    "heading_index": heading.index,
                    "chunk_index": chunk.chunk_index,
                    "categories": ",".join(self.categories),
                    "abstract": self.abstract[:500],  # Truncate for metadata
                    "is_valid": chunk.is_valid
                })
        return metadata_list
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "paper_id": self.paper_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "authors": self.authors,
            "categories": self.categories,
            "published_date": self.published_date,
            "is_processed": self.is_processed,
            "processing_error": self.processing_error,
            "num_headings": len(self.headings),
            "total_chunks": self.total_chunks
        }

