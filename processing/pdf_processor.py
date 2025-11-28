"""
PDF processor for extracting content from research papers.
Uses Docling for PDF to Markdown conversion and heading extraction.
"""
import re
import logging
from typing import List, Optional, Tuple
from datetime import datetime

from docling.document_converter import DocumentConverter

import config
from models.paper import Paper, Heading

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Processes PDF papers into structured content.
    Extracts headings and section text for chunking.
    """
    
    # Headings to skip (common non-content sections)
    SKIP_HEADINGS = {
        'references', 'bibliography', 'acknowledgments', 'acknowledgements',
        'appendix', 'appendices', 'supplementary', 'supplemental'
    }
    
    def __init__(self):
        """Initialize PDF processor with Docling converter."""
        self.converter = DocumentConverter()
    
    def process_paper(self, paper: Paper) -> Paper:
        """
        Process a paper's PDF and extract structured content.
        
        Args:
            paper: Paper object with pdf_url set
            
        Returns:
            Paper with markdown_content and headings populated
        """
        if not paper.pdf_url:
            paper.processing_error = "No PDF URL provided"
            return paper
        
        try:
            logger.info(f"Processing PDF: {paper.title[:50]}...")
            
            # Convert PDF to markdown
            markdown = self._convert_to_markdown(paper.pdf_url)
            if not markdown:
                paper.processing_error = "Failed to convert PDF to markdown"
                return paper
            
            paper.markdown_content = markdown
            
            # Extract headings with section text
            headings = self._extract_headings_with_content(markdown, paper.paper_id)
            paper.headings = headings
            
            # Mark as processed
            paper.is_processed = True
            paper.processed_at = datetime.now()
            
            logger.info(f"Extracted {len(headings)} headings from {paper.paper_id}")
            
        except Exception as e:
            logger.error(f"Error processing paper {paper.paper_id}: {e}")
            paper.processing_error = str(e)
        
        return paper
    
    def _convert_to_markdown(self, source: str) -> Optional[str]:
        """
        Convert PDF to markdown using Docling.
        
        Args:
            source: PDF URL or file path
            
        Returns:
            Markdown string or None on failure
        """
        try:
            result = self.converter.convert(source)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            return None
    
    def _extract_headings_with_content(
        self,
        markdown: str,
        paper_id: str
    ) -> List[Heading]:
        """
        Extract headings and their section content from markdown.
        
        Args:
            markdown: Markdown text from PDF conversion
            paper_id: Parent paper ID
            
        Returns:
            List of Heading objects with section_text populated
        """
        headings = []
        lines = markdown.split('\n')
        
        # Pattern for markdown headings
        heading_pattern = r'^(#{1,6})\s*(.*)$'
        
        # First pass: find all headings with their line numbers
        heading_positions = []
        for line_num, line in enumerate(lines):
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                text = self._clean_heading_text(match.group(2))
                heading_positions.append({
                    'line_num': line_num,
                    'level': level,
                    'text': text,
                    'raw': line.strip()
                })
        
        # Second pass: extract content between headings
        for i, h_pos in enumerate(heading_positions):
            # Skip unwanted sections
            if self._should_skip_heading(h_pos['text']):
                continue
            
            # Determine end line (next heading or end of document)
            start_line = h_pos['line_num'] + 1  # Start after heading line
            if i + 1 < len(heading_positions):
                end_line = heading_positions[i + 1]['line_num']
            else:
                end_line = len(lines)
            
            # Extract section text
            section_lines = lines[start_line:end_line]
            section_text = '\n'.join(section_lines).strip()
            
            # Create Heading object
            heading = Heading(
                heading_id="",  # Will be generated in __post_init__
                paper_id=paper_id,
                index=len(headings),
                level=h_pos['level'],
                text=h_pos['text'],
                raw_text=h_pos['raw'],
                section_text=section_text,
                is_valid=len(section_text) >= config.MIN_SECTION_LENGTH
            )
            
            # Calculate quality score
            heading.quality_score = self._calculate_section_quality(section_text)
            
            headings.append(heading)
        
        return headings
    
    def _clean_heading_text(self, text: str) -> str:
        """
        Clean heading text by removing numbering and extra whitespace.
        
        Args:
            text: Raw heading text
            
        Returns:
            Cleaned heading text
        """
        if not text:
            return ""
        
        # Remove common numbering patterns
        # e.g., "1.", "1.2", "I.", "A.", "1)", "(1)"
        text = re.sub(r'^[\d.]+\s*', '', text)  # "1." or "1.2.3"
        text = re.sub(r'^[IVX]+\.\s*', '', text)  # "I." "II." Roman numerals
        text = re.sub(r'^[A-Z]\.\s*', '', text)  # "A." "B."
        text = re.sub(r'^\(\d+\)\s*', '', text)  # "(1)"
        text = re.sub(r'^\d+\)\s*', '', text)  # "1)"
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (for headings that include content)
        if ':' in text and len(text) > 100:
            text = text.split(':')[0]
        
        if len(text) > 150:
            text = text[:150].rsplit(' ', 1)[0] + '...'
        
        return text.strip()
    
    def _should_skip_heading(self, heading_text: str) -> bool:
        """Check if a heading should be skipped (references, etc.)."""
        lower_text = heading_text.lower()
        for skip in self.SKIP_HEADINGS:
            if skip in lower_text:
                return True
        return False
    
    def _calculate_section_quality(self, section_text: str) -> float:
        """
        Calculate a quality score for section content.
        
        Returns:
            Float 0-1 where 1 is highest quality
        """
        if not section_text:
            return 0.0
        
        score = 1.0
        
        # Penalize very short sections
        if len(section_text) < config.MIN_SECTION_LENGTH:
            score *= 0.5
        elif len(section_text) < config.MIN_SECTION_LENGTH * 2:
            score *= 0.8
        
        # Check alphabetic content ratio
        alpha_count = sum(c.isalpha() for c in section_text)
        alpha_ratio = alpha_count / max(len(section_text), 1)
        if alpha_ratio < 0.5:
            score *= 0.6
        
        # Check for common low-quality patterns
        words = section_text.split()
        if len(words) > 10:
            unique_ratio = len(set(w.lower() for w in words)) / len(words)
            if unique_ratio < 0.3:
                score *= 0.5  # Very repetitive
        
        return min(max(score, 0.0), 1.0)
    
    def get_text_between_headings(
        self,
        markdown: str,
        start_heading: str,
        end_heading: Optional[str] = None
    ) -> str:
        """
        Extract text between two headings.
        
        Args:
            markdown: Full markdown text
            start_heading: Starting heading text (partial match)
            end_heading: Ending heading text (partial match, exclusive)
            
        Returns:
            Text between the headings
        """
        lines = markdown.split('\n')
        heading_pattern = r'^#{1,6}\s*(.*)$'
        
        start_line = None
        end_line = len(lines)
        
        for i, line in enumerate(lines):
            match = re.match(heading_pattern, line.strip())
            if match:
                heading_text = match.group(1).lower()
                
                if start_line is None and start_heading.lower() in heading_text:
                    start_line = i + 1  # Start after the heading
                elif start_line is not None and end_heading:
                    if end_heading.lower() in heading_text:
                        end_line = i
                        break
        
        if start_line is None:
            return ""
        
        return '\n'.join(lines[start_line:end_line]).strip()
    
    def extract_abstract_from_markdown(self, markdown: str) -> str:
        """
        Try to extract abstract from markdown if not already available.
        
        Args:
            markdown: Full markdown text
            
        Returns:
            Abstract text or empty string
        """
        # Look for explicit abstract section
        abstract = self.get_text_between_headings(
            markdown, 'abstract', 'introduction'
        )
        
        if abstract:
            return abstract[:2000]  # Limit length
        
        # Try to find abstract in first few paragraphs
        lines = markdown.split('\n')
        in_abstract = False
        abstract_lines = []
        
        for line in lines[:50]:  # Only check first 50 lines
            lower_line = line.lower().strip()
            
            if 'abstract' in lower_line and line.startswith('#'):
                in_abstract = True
                continue
            
            if in_abstract:
                if line.startswith('#'):
                    break
                if line.strip():
                    abstract_lines.append(line.strip())
        
        return ' '.join(abstract_lines)[:2000]

