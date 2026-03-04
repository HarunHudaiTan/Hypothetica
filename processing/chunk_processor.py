"""
Chunk processor for splitting paper sections into embeddable chunks.
Uses heading-level structure with paragraph/sentence-level chunks.
"""
import re
import logging
from typing import List, Tuple, Optional

import config
from models.paper import Paper, Heading, Chunk

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """
    Processes paper headings into smaller chunks for embedding.
    
    Strategy:
    - Headings provide structure/context
    - Chunks are paragraph or sentence blocks within headings
    - Each chunk maintains reference to parent heading
    """
    
    def __init__(
        self,
        max_chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None
    ):
        """
        Initialize chunk processor.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            min_chunk_size: Minimum characters for valid chunk
        """
        self.max_chunk_size = max_chunk_size or config.MAX_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or config.MIN_CHUNK_SIZE
    
    def process_paper(self, paper: Paper) -> Paper:
        """
        Process all headings in a paper into chunks.
        
        Args:
            paper: Paper with extracted headings (section_text populated)
            
        Returns:
            Paper with chunks added to each heading
        """
        for heading in paper.headings:
            if heading.section_text:
                chunks = self._chunk_section(
                    text=heading.section_text,
                    paper_id=paper.paper_id,
                    heading=heading
                )
                heading.chunks = chunks
                
                # Log quality issues
                valid_chunks = [c for c in chunks if c.is_valid]
                if len(valid_chunks) < len(chunks):
                    logger.warning(
                        f"Paper {paper.paper_id}, heading '{heading.text}': "
                        f"{len(chunks) - len(valid_chunks)}/{len(chunks)} chunks invalid"
                    )
            else:
                logger.warning(
                    f"Paper {paper.paper_id}, heading '{heading.text}' has no section text"
                )
        
        return paper
    
    def _chunk_section(
        self,
        text: str,
        paper_id: str,
        heading: Heading
    ) -> List[Chunk]:
        """
        Split section text into chunks.
        
        Strategy:
        1. First try to split by paragraphs (double newlines)
        2. If paragraphs are too long, split by sentences
        3. If still too long, split by fixed size with overlap
        
        Args:
            text: Section text to chunk
            paper_id: Parent paper ID
            heading: Parent heading object
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # Clean the text
        text = self._clean_text(text)
        
        if not text or len(text.strip()) < self.min_chunk_size:
            # Section too short - create single chunk and mark potentially invalid
            chunk = Chunk(
                chunk_id="",
                paper_id=paper_id,
                heading=heading.text,
                heading_index=heading.index,
                chunk_index=0,
                text=text.strip(),
                char_start=0,
                char_end=len(text),
                is_valid=len(text.strip()) >= self.min_chunk_size // 2,
                quality_reason="Section too short" if len(text.strip()) < self.min_chunk_size // 2 else None
            )
            return [chunk]
        
        # Split by paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        # Process paragraphs into chunks
        current_chunk_text = ""
        current_start = 0
        char_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max size
            if current_chunk_text and len(current_chunk_text) + len(para) + 2 > self.max_chunk_size:
                # Save current chunk
                if len(current_chunk_text.strip()) >= self.min_chunk_size:
                    chunk = Chunk(
                        chunk_id="",
                        paper_id=paper_id,
                        heading=heading.text,
                        heading_index=heading.index,
                        chunk_index=len(chunks),
                        text=current_chunk_text.strip(),
                        char_start=current_start,
                        char_end=char_pos,
                        is_valid=True
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text)
                current_chunk_text = overlap_text + para
                current_start = char_pos - len(overlap_text)
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
                    current_start = char_pos
            
            char_pos += len(para) + 2  # +2 for paragraph separator
        
        # Save final chunk
        if current_chunk_text.strip():
            chunk = Chunk(
                chunk_id="",
                paper_id=paper_id,
                heading=heading.text,
                heading_index=heading.index,
                chunk_index=len(chunks),
                text=current_chunk_text.strip(),
                char_start=current_start,
                char_end=len(text),
                is_valid=len(current_chunk_text.strip()) >= self.min_chunk_size
            )
            chunks.append(chunk)
        
        # If we got very large chunks, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk.text) > self.max_chunk_size * 1.5:
                # Split by sentences
                sub_chunks = self._split_large_chunk(chunk, paper_id, heading, len(final_chunks))
                final_chunks.extend(sub_chunks)
            else:
                chunk.chunk_index = len(final_chunks)
                final_chunks.append(chunk)
        
        # Regenerate chunk IDs
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_index = i
            chunk.chunk_id = f"{paper_id}_h{heading.index:02d}_c{i:02d}"
        
        return final_chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs (double newlines or markdown patterns)."""
        # Split by double newlines or markdown-style breaks
        paragraphs = re.split(r'\n\s*\n|\n(?=#+\s)', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - handles common cases
        # Could be improved with NLTK or spaCy for production
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_large_chunk(
        self,
        chunk: Chunk,
        paper_id: str,
        heading: Heading,
        start_index: int
    ) -> List[Chunk]:
        """Split a large chunk into smaller pieces by sentences."""
        sentences = self._split_sentences(chunk.text)
        
        sub_chunks = []
        current_text = ""
        current_start = chunk.char_start
        
        for sent in sentences:
            if len(current_text) + len(sent) + 1 > self.max_chunk_size:
                if current_text.strip():
                    sub_chunk = Chunk(
                        chunk_id="",
                        paper_id=paper_id,
                        heading=heading.text,
                        heading_index=heading.index,
                        chunk_index=start_index + len(sub_chunks),
                        text=current_text.strip(),
                        char_start=current_start,
                        char_end=current_start + len(current_text),
                        is_valid=len(current_text.strip()) >= self.min_chunk_size
                    )
                    sub_chunks.append(sub_chunk)
                
                current_text = sent
                current_start = current_start + len(current_text)
            else:
                current_text += " " + sent if current_text else sent
        
        # Add remaining
        if current_text.strip():
            sub_chunk = Chunk(
                chunk_id="",
                paper_id=paper_id,
                heading=heading.text,
                heading_index=heading.index,
                chunk_index=start_index + len(sub_chunks),
                text=current_text.strip(),
                char_start=current_start,
                char_end=chunk.char_end,
                is_valid=len(current_text.strip()) >= self.min_chunk_size
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of chunk for continuity."""
        if len(text) <= self.chunk_overlap:
            return ""
        
        # Try to break at word boundary
        overlap_region = text[-self.chunk_overlap:]
        space_idx = overlap_region.find(' ')
        if space_idx > 0:
            return overlap_region[space_idx + 1:]
        return overlap_region
    
    def _clean_text(self, text: str) -> str:
        """Clean text for chunking."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation markers like [1], [23]
        text = re.sub(r'Fig\.\s*\d+', 'Figure', text)  # Normalize figure references
        
        return text.strip()
    
    def validate_chunk_quality(
        self,
        chunk: Chunk,
        abstract: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate chunk quality.
        
        Args:
            chunk: Chunk to validate
            abstract: Paper abstract for relevance check
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check minimum length
        if len(chunk.text.strip()) < self.min_chunk_size:
            return False, "Chunk too short"
        
        # Check if mostly numbers/special characters
        alpha_ratio = sum(c.isalpha() for c in chunk.text) / max(len(chunk.text), 1)
        if alpha_ratio < 0.5:
            return False, "Too few alphabetic characters"
        
        # Check for repetitive content
        words = chunk.text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False, "Too repetitive"
        
        return True, None
    
    def get_chunk_stats(self, paper: Paper) -> dict:
        """Get statistics about chunks in a paper."""
        all_chunks = []
        for heading in paper.headings:
            all_chunks.extend(heading.chunks)
        
        valid_chunks = [c for c in all_chunks if c.is_valid]
        
        return {
            "total_chunks": len(all_chunks),
            "valid_chunks": len(valid_chunks),
            "invalid_chunks": len(all_chunks) - len(valid_chunks),
            "avg_chunk_length": sum(len(c.text) for c in all_chunks) / max(len(all_chunks), 1),
            "headings_with_chunks": sum(1 for h in paper.headings if h.chunks)
        }

