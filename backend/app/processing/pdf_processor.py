"""
PDF processor for extracting content from research papers.
Uses Docling for PDF to Markdown conversion and heading extraction.
Supports parallel processing for faster throughput.
"""
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from datetime import datetime

from docling.document_converter import DocumentConverter

from core import config
from app.models.paper import Paper, Heading

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
        Source-aware processing with fallback for different paper sources.
        
        Args:
            paper: Paper object with pdf_url set
            
        Returns:
            Paper with markdown_content and headings populated
        """
        try:
            logger.info(f"Processing PDF: {paper.title[:50]}... (source: {paper.source})")
            
            # Handle papers without PDF URLs
            if not paper.pdf_url:
                if paper.abstract:
                    # All sources: Use abstract fallback when no PDF URL
                    logger.info(f"No PDF URL for {paper.paper_id}, using abstract fallback")
                    markdown = self._create_abstract_fallback(paper)
                    if markdown:
                        paper.markdown_content = markdown
                        paper.headings = self._extract_headings_with_content(markdown, paper.paper_id)
                        paper.is_processed = True
                        paper.processed_at = datetime.now()
                        logger.info(f"Abstract fallback successful for {paper.paper_id}")
                        return paper
                else:
                    # No abstract either: Mark as error
                    paper.processing_error = "No PDF URL or abstract provided"
                    return paper
            
            # Convert PDF to markdown with source-aware handling
            markdown = self._convert_to_markdown_with_source(paper)
            if not markdown:
                # PDF conversion failed, try abstract fallback
                if paper.abstract:
                    logger.warning(f"PDF conversion failed for {paper.paper_id}, trying abstract fallback")
                    markdown = self._create_abstract_fallback(paper)
                    if markdown:
                        paper.markdown_content = markdown
                        paper.headings = self._extract_headings_with_content(markdown, paper.paper_id)
                        paper.is_processed = True
                        paper.processed_at = datetime.now()
                        logger.info(f"Abstract fallback successful for {paper.paper_id}")
                        return paper
                else:
                    paper.processing_error = "Failed to convert PDF to markdown and no abstract available"
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
    
    def _process_paper_isolated(self, paper: Paper) -> Paper:
        """
        Process a single paper with a fresh DocumentConverter.
        Thread-safe: each call uses its own converter to avoid shared state.
        Source-aware processing with fallback.
        """
        try:
            logger.info(f"Processing PDF: {paper.title[:50]}... (source: {paper.source})")
            
            # Handle papers without PDF URLs
            if not paper.pdf_url:
                if paper.abstract:
                    # All sources: Use abstract fallback when no PDF URL
                    logger.info(f"No PDF URL for {paper.paper_id}, using abstract fallback")
                    markdown = self._create_abstract_fallback(paper)
                    if markdown:
                        paper.markdown_content = markdown
                        paper.headings = self._extract_headings_with_content(markdown, paper.paper_id)
                        paper.is_processed = True
                        paper.processed_at = datetime.now()
                        logger.info(f"Abstract fallback successful for {paper.paper_id}")
                        return paper
                else:
                    # No abstract either: Mark as error
                    paper.processing_error = "No PDF URL or abstract provided"
                    return paper
            
            converter = DocumentConverter()
            
            # Source-aware conversion
            markdown = self._convert_to_markdown_with_converter(paper, converter)
            
            if not markdown:
                # PDF conversion failed, try abstract fallback
                if paper.abstract:
                    logger.warning(f"PDF conversion failed for {paper.paper_id}, trying abstract fallback")
                    markdown = self._create_abstract_fallback(paper)
                    if markdown:
                        paper.markdown_content = markdown
                        paper.headings = self._extract_headings_with_content(markdown, paper.paper_id)
                        paper.is_processed = True
                        paper.processed_at = datetime.now()
                        logger.info(f"Abstract fallback successful for {paper.paper_id}")
                        return paper
                else:
                    paper.processing_error = "Failed to convert PDF to markdown and no abstract available"
                    return paper
            
            paper.markdown_content = markdown
            headings = self._extract_headings_with_content(markdown, paper.paper_id)
            paper.headings = headings
            paper.is_processed = True
            paper.processed_at = datetime.now()
            logger.info(f"Extracted {len(headings)} headings from {paper.paper_id}")
            
        except Exception as e:
            logger.error(f"Error processing paper {paper.paper_id}: {e}")
            paper.processing_error = str(e)
        
        return paper
    
    def process_papers_parallel(
        self,
        papers: List[Paper],
        max_workers: int = 5
    ) -> List[Paper]:
        """
        Process multiple papers in parallel (PDF conversion + heading extraction).
        Uses a fresh DocumentConverter per paper for thread safety.
        
        Args:
            papers: List of Paper objects to process
            max_workers: Maximum number of concurrent workers
            
        Returns:
            The same papers with markdown_content and headings populated
        """
        if not papers:
            return papers
        
        workers = min(max_workers, len(papers))
        logger.info(f"Processing {len(papers)} PDFs in parallel (max_workers={workers})")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_paper = {
                executor.submit(self._process_paper_isolated, p): p
                for p in papers
            }
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing paper {paper.paper_id}: {e}")
                    paper.processing_error = str(e)
        
        return papers
    
    def _convert_to_markdown(self, source: str) -> Optional[str]:
        """
        Convert PDF to markdown using Docling.
        Legacy method - use _convert_to_markdown_with_source for new code.
        
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
    
    def _convert_to_markdown_with_source(self, paper: Paper) -> Optional[str]:
        """
        Convert PDF to markdown with source-aware handling.
        Different strategies for different paper sources.
        
        Args:
            paper: Paper object with source and pdf_url
            
        Returns:
            Markdown string or None on failure
        """
        if paper.source == "arxiv":
            # ArXiv: Use existing logic (unchanged)
            return self._convert_to_markdown(paper.pdf_url)
        
        elif paper.source == "semantic_scholar":
            # Semantic Scholar: Try multiple strategies
            return self._convert_semantic_scholar_pdf(paper)
        
        else:
            # Unknown source: Use default logic
            return self._convert_to_markdown(paper.pdf_url)
    
    def _convert_semantic_scholar_pdf(self, paper: Paper) -> Optional[str]:
        """
        Convert Semantic Scholar PDF with multiple fallback strategies.
        
        Args:
            paper: Semantic Scholar paper object
            
        Returns:
            Markdown string or None on failure
        """
        pdf_url = paper.pdf_url
        logger.info(f"Converting Semantic Scholar PDF: {paper.title[:50]}...")
        
        # Strategy 1: Try direct conversion
        try:
            result = self.converter.convert(pdf_url)
            markdown = result.document.export_to_markdown()
            if markdown and len(markdown.strip()) > 100:
                logger.info(f"Direct conversion successful for {paper.paper_id}")
                return markdown
        except Exception as e:
            logger.warning(f"Direct conversion failed for {paper.paper_id}: {e}")
        
        # Strategy 2: Try DOI redirect (if URL is DOI)
        if "doi.org" in pdf_url:
            try:
                logger.info(f"Trying DOI redirect for {paper.paper_id}")
                result = self.converter.convert(pdf_url)
                markdown = result.document.export_to_markdown()
                if markdown and len(markdown.strip()) > 100:
                    logger.info(f"DOI conversion successful for {paper.paper_id}")
                    return markdown
            except Exception as e:
                logger.warning(f"DOI conversion failed for {paper.paper_id}: {e}")
        
        # Strategy 3: Fallback to abstract-only processing
        logger.warning(f"PDF conversion failed for {paper.paper_id}, falling back to abstract-only")
        return self._create_abstract_fallback(paper)
    
    def _convert_to_markdown_with_converter(self, paper: Paper, converter) -> Optional[str]:
        """
        Convert PDF to markdown using a specific converter instance.
        Source-aware processing with fallback.
        """
        if paper.source == "arxiv":
            # ArXiv: Use existing logic
            try:
                result = converter.convert(paper.pdf_url)
                return result.document.export_to_markdown()
            except Exception as e:
                logger.error(f"ArXiv conversion failed: {e}")
                return None
        
        elif paper.source == "semantic_scholar":
            # Semantic Scholar: Try multiple strategies
            return self._convert_semantic_scholar_pdf_with_converter(paper, converter)
        
        else:
            # Unknown source: Use default logic
            try:
                result = converter.convert(paper.pdf_url)
                return result.document.export_to_markdown()
            except Exception as e:
                logger.error(f"Default conversion failed: {e}")
                return None
    
    def _convert_semantic_scholar_pdf_with_converter(self, paper: Paper, converter) -> Optional[str]:
        """
        Convert Semantic Scholar PDF using specific converter with fallback.
        """
        pdf_url = paper.pdf_url
        logger.info(f"Converting Semantic Scholar PDF: {paper.title[:50]}...")
        
        # Strategy 1: Try direct conversion
        try:
            result = converter.convert(pdf_url)
            markdown = result.document.export_to_markdown()
            if markdown and len(markdown.strip()) > 100:
                logger.info(f"Direct conversion successful for {paper.paper_id}")
                return markdown
        except Exception as e:
            logger.warning(f"Direct conversion failed for {paper.paper_id}: {e}")
        
        # Strategy 2: Try DOI redirect
        if "doi.org" in pdf_url:
            try:
                logger.info(f"Trying DOI redirect for {paper.paper_id}")
                result = converter.convert(pdf_url)
                markdown = result.document.export_to_markdown()
                if markdown and len(markdown.strip()) > 100:
                    logger.info(f"DOI conversion successful for {paper.paper_id}")
                    return markdown
            except Exception as e:
                logger.warning(f"DOI conversion failed for {paper.paper_id}: {e}")
        
        # Strategy 3: Fallback to abstract-only
        logger.warning(f"PDF conversion failed for {paper.paper_id}, falling back to abstract-only")
        return self._create_abstract_fallback(paper)
    
    def _create_abstract_fallback(self, paper: Paper) -> Optional[str]:
        """
        Create markdown from abstract when PDF conversion fails.
        This ensures the paper can still be processed for analysis.
        
        Args:
            paper: Paper object with abstract
            
        Returns:
            Markdown string with abstract content
        """
        if not paper.abstract:
            logger.error(f"No abstract available for fallback processing {paper.paper_id}")
            return None
        
        # Ensure categories is never None
        if paper.categories is None:
            paper.categories = []
        
        # Create markdown structure from abstract
        markdown = f"""# {paper.title}

## Abstract

{paper.abstract}

## Authors

{', '.join(paper.authors) if paper.authors else 'Unknown'}

## Source

{paper.source} - {paper.source_id}

## Note

*This paper was processed using abstract-only content due to PDF conversion limitations.*
"""
        
        logger.info(f"Created abstract fallback for {paper.paper_id} ({len(markdown)} chars)")
        return markdown
    
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

