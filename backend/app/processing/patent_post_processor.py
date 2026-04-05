"""
Patent-specific post-processor for Google Patents papers.

Runs AFTER Docling PDF→Markdown conversion but BEFORE heading extraction.
Handles non-English text (translation), OCR artifacts, patent structure
normalization, and legal boilerplate that the generic PDF processor
doesn't address.

Only activated for papers with source == "google_patents".
arXiv papers are never touched by this module.
"""
import re
import logging
import unicodedata
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PatentProcessingResult:
    """Result of patent post-processing with metadata about what changed."""
    markdown: str
    had_non_english: bool = False
    translated_sections_count: int = 0
    detected_languages: List[str] = field(default_factory=list)
    ocr_artifacts_removed: int = 0
    headings_normalized: int = 0
    boilerplate_sections_removed: int = 0

    @property
    def has_warnings(self) -> bool:
        return self.had_non_english or self.ocr_artifacts_removed > 5

    @property
    def warning_message(self) -> Optional[str]:
        if not self.has_warnings:
            return None

        parts = []
        if self.had_non_english:
            lang_str = ", ".join(self.detected_languages) if self.detected_languages else "non-English"
            parts.append(
                f"This patent contained {self.translated_sections_count} section(s) "
                f"in {lang_str} that were machine-translated to English. "
                f"Translated text may reduce embedding accuracy and affect "
                f"the global originality score."
            )
        if self.ocr_artifacts_removed > 5:
            parts.append(
                f"{self.ocr_artifacts_removed} OCR artifacts were cleaned up, "
                f"which may indicate lower-quality text extraction."
            )
        return " ".join(parts)


class PatentPostProcessor:
    """
    Post-processes patent markdown content to improve quality
    before heading extraction and chunking.
    """

    # Patent-specific heading mappings (uppercase variants → normalized markdown)
    HEADING_NORMALIZATION = {
        # Abstract variants
        r'ABSTRACT\s*(?:OF\s*(?:THE\s*)?DISCLOSURE)?': '## Abstract',
        r'SUMMARY\s*(?:OF\s*(?:THE\s*)?(?:INVENTION|DISCLOSURE))?': '## Summary',
        # Background variants
        r'BACKGROUND\s*(?:OF\s*(?:THE\s*)?INVENTION)?': '## Background',
        r'FIELD\s*(?:OF\s*(?:THE\s*)?INVENTION)?': '## Field of the Invention',
        r'PRIOR\s*ART': '## Prior Art',
        r'RELATED\s*ART': '## Related Art',
        # Description variants
        r'DETAILED\s*DESCRIPTION\s*(?:OF\s*(?:THE\s*)?(?:PREFERRED\s*)?EMBODIMENTS?)?': '## Detailed Description',
        r'DESCRIPTION\s*(?:OF\s*(?:THE\s*)?(?:PREFERRED\s*)?EMBODIMENTS?)?': '## Description',
        r'BRIEF\s*DESCRIPTION\s*(?:OF\s*(?:THE\s*)?DRAWINGS?)?': '## Brief Description of the Drawings',
        # Claims
        r'CLAIMS?': '## Claims',
        r'WHAT\s*IS\s*CLAIMED\s*IS': '## Claims',
        # Other patent sections
        r'TECHNICAL\s*FIELD': '## Technical Field',
        r'CROSS[\-\s]?REFERENCE\s*TO\s*RELATED\s*APPLICATIONS?': '## Cross-Reference to Related Applications',
        r'STATEMENT\s*(?:REGARDING|OF)\s*FEDERALLY\s*SPONSORED': '## Federally Sponsored Research',
    }

    # Legal boilerplate patterns to remove entirely
    BOILERPLATE_PATTERNS = [
        # Patent number / filing lines
        r'^(?:United States Patent|U\.S\. Patent)\s*(?:No\.)?\s*[\d,]+.*$',
        r'^(?:Patent\s*No\.|Application\s*No\.)\s*:?\s*[\w\d/,\-]+.*$',
        r'^(?:Filed|Date of Patent|Appl\. No\.):\s*.*$',
        # Assignee / inventor header blocks
        r'^(?:Assignee|Inventor(?:s)?|Attorney|Agent|Examiner)\s*:.*$',
        # Legal preamble
        r'^(?:This application (?:claims|is a continuation).*?)$',
        r'^(?:\(\d+\)\s*(?:Field of|Int\.\s*Cl\.).*?)$',
        # Foreign patent classification codes
        r'^[A-Z]\d{2}[A-Z]\s*\d+/\d+.*$',
    ]

    # Unicode script ranges for language detection
    SCRIPT_RANGES = {
        'CJK': [
            (0x4E00, 0x9FFF),    # CJK Unified Ideographs
            (0x3400, 0x4DBF),    # CJK Extension A
            (0x3000, 0x303F),    # CJK Symbols and Punctuation
            (0x3040, 0x309F),    # Hiragana
            (0x30A0, 0x30FF),    # Katakana
            (0xAC00, 0xD7AF),    # Hangul Syllables
        ],
        'Arabic': [
            (0x0600, 0x06FF),    # Arabic
            (0x0750, 0x077F),    # Arabic Supplement
        ],
        'Cyrillic': [
            (0x0400, 0x04FF),    # Cyrillic
            (0x0500, 0x052F),    # Cyrillic Supplement
        ],
        'Devanagari': [
            (0x0900, 0x097F),    # Devanagari
        ],
        'Thai': [
            (0x0E00, 0x0E7F),    # Thai
        ],
    }

    @classmethod
    def process(cls, markdown: str, paper_title: str = "") -> PatentProcessingResult:
        """
        Run the full patent post-processing pipeline on markdown content.

        Args:
            markdown: Raw markdown from Docling conversion
            paper_title: Paper title for logging context

        Returns:
            PatentProcessingResult with processed markdown and metadata
        """
        result = PatentProcessingResult(markdown=markdown)
        title_short = paper_title[:40] + "..." if len(paper_title) > 40 else paper_title
        logger.info(f"[PatentPostProcessor] Processing patent: {title_short}")

        # Step 1: Remove legal boilerplate
        result.markdown, boilerplate_count = cls._remove_boilerplate(result.markdown)
        result.boilerplate_sections_removed = boilerplate_count
        if boilerplate_count > 0:
            logger.info(f"[PatentPostProcessor] Removed {boilerplate_count} boilerplate lines")

        # Step 2: Normalize patent headings
        result.markdown, headings_count = cls._normalize_headings(result.markdown)
        result.headings_normalized = headings_count
        if headings_count > 0:
            logger.info(f"[PatentPostProcessor] Normalized {headings_count} patent headings")

        # Step 3: Clean OCR artifacts
        result.markdown, ocr_count = cls._clean_ocr_artifacts(result.markdown)
        result.ocr_artifacts_removed = ocr_count
        if ocr_count > 0:
            logger.info(f"[PatentPostProcessor] Cleaned {ocr_count} OCR artifacts")

        # Step 4: Clean reference numerals
        result.markdown = cls._clean_reference_numerals(result.markdown)

        # Step 5: Detect and translate non-English sections
        result.markdown, non_english_info = cls._handle_non_english_sections(result.markdown)
        result.had_non_english = non_english_info['had_non_english']
        result.translated_sections_count = non_english_info['translated_count']
        result.detected_languages = non_english_info['detected_languages']
        if result.had_non_english:
            logger.warning(
                f"[PatentPostProcessor] Translated {result.translated_sections_count} "
                f"non-English section(s) (languages: {', '.join(result.detected_languages)})"
            )

        # Step 6: Final cleanup
        result.markdown = cls._final_cleanup(result.markdown)

        logger.info(
            f"[PatentPostProcessor] Done: {result.headings_normalized} headings normalized, "
            f"{result.ocr_artifacts_removed} OCR artifacts cleaned, "
            f"{result.translated_sections_count} sections translated"
        )
        return result

    @classmethod
    def _remove_boilerplate(cls, markdown: str) -> Tuple[str, int]:
        """Remove legal boilerplate lines from patent text."""
        lines = markdown.split('\n')
        cleaned_lines = []
        removed_count = 0

        for line in lines:
            stripped = line.strip()
            is_boilerplate = False

            for pattern in cls.BOILERPLATE_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    is_boilerplate = True
                    removed_count += 1
                    break

            if not is_boilerplate:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines), removed_count

    @classmethod
    def _normalize_headings(cls, markdown: str) -> Tuple[str, int]:
        """
        Normalize patent-specific headings into standard markdown headings.
        Patent PDFs often have ALL-CAPS headings without # markers.
        """
        lines = markdown.split('\n')
        normalized_lines = []
        normalized_count = 0

        for line in lines:
            stripped = line.strip()
            matched = False

            # Check if line is already a markdown heading
            if stripped.startswith('#'):
                # Still try to normalize the heading text
                heading_match = re.match(r'^(#{1,6})\s*(.*)', stripped)
                if heading_match:
                    level = heading_match.group(1)
                    text = heading_match.group(2)
                    for pattern, replacement in cls.HEADING_NORMALIZATION.items():
                        if re.match(pattern, text, re.IGNORECASE):
                            normalized_lines.append(replacement)
                            normalized_count += 1
                            matched = True
                            break

            # Check for standalone ALL-CAPS lines (likely headings)
            if not matched and stripped and not stripped.startswith('#'):
                for pattern, replacement in cls.HEADING_NORMALIZATION.items():
                    if re.match(r'^' + pattern + r'\s*$', stripped, re.IGNORECASE):
                        normalized_lines.append(replacement)
                        normalized_count += 1
                        matched = True
                        break

                # Generic ALL-CAPS detection for other headings
                if not matched and cls._is_likely_heading(stripped):
                    # Convert to title case heading
                    title_text = stripped.title()
                    normalized_lines.append(f"## {title_text}")
                    normalized_count += 1
                    matched = True

            if not matched:
                normalized_lines.append(line)

        return '\n'.join(normalized_lines), normalized_count

    @staticmethod
    def _is_likely_heading(text: str) -> bool:
        """
        Heuristic to detect if a line is likely a heading.
        Patents often have ALL-CAPS section headers without markdown markers.
        """
        if not text or len(text) > 80 or len(text) < 3:
            return False

        # Must be mostly uppercase alphabetic
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return False

        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio < 0.8:
            return False

        # Should be short-ish (headings, not paragraphs)
        word_count = len(text.split())
        if word_count > 10:
            return False

        # Should not end with a period (likely a sentence, not a heading)
        if text.rstrip().endswith('.'):
            return False

        return True

    @classmethod
    def _clean_ocr_artifacts(cls, markdown: str) -> Tuple[str, int]:
        """
        Remove or clean OCR artifacts — garbled text, broken encodings,
        and nonsensical character sequences.
        """
        lines = markdown.split('\n')
        cleaned_lines = []
        artifact_count = 0

        for line in lines:
            stripped = line.strip()

            if not stripped:
                cleaned_lines.append(line)
                continue

            # Skip very short lines (less than 3 chars that aren't headings)
            if len(stripped) < 3 and not stripped.startswith('#'):
                artifact_count += 1
                continue

            # Check alphabetic content ratio (very low = likely garbage)
            if len(stripped) > 5:
                alpha_count = sum(c.isalpha() or c.isspace() for c in stripped)
                alpha_ratio = alpha_count / len(stripped)
                if alpha_ratio < 0.3:
                    artifact_count += 1
                    continue

            # Check for excessive consecutive special characters
            if re.search(r'[^\w\s]{5,}', stripped) and not stripped.startswith('#'):
                # Clean the line by removing excessive special char runs
                cleaned = re.sub(r'[^\w\s.,;:!?()"\'-]{3,}', ' ', stripped)
                if len(cleaned.strip()) > 10:
                    cleaned_lines.append(cleaned)
                    artifact_count += 1
                else:
                    artifact_count += 1
                continue

            # Check for repeated character patterns (broken OCR)
            if re.search(r'(.)\1{4,}', stripped):
                artifact_count += 1
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines), artifact_count

    @classmethod
    def _clean_reference_numerals(cls, markdown: str) -> str:
        """
        Clean up inline patent reference numerals that clutter text.
        e.g., "element 102", "component (34)", "reference numeral 210"
        """
        # Remove standalone reference numerals in parentheses: (102), (34a)
        markdown = re.sub(r'\(\s*\d{1,4}[a-zA-Z]?\s*\)', '', markdown)

        # Clean "reference numeral NNN" patterns
        markdown = re.sub(
            r'(?:reference\s+)?(?:numeral|number)\s+\d{1,4}[a-zA-Z]?\b',
            '',
            markdown,
            flags=re.IGNORECASE
        )

        # Clean up resulting double spaces
        markdown = re.sub(r'  +', ' ', markdown)

        return markdown

    @classmethod
    def _handle_non_english_sections(cls, markdown: str) -> Tuple[str, dict]:
        """
        Detect non-English paragraphs and translate them to English.
        Keeps translated text in place with a marker note.

        Returns:
            Tuple of (processed_markdown, info_dict)
        """
        info = {
            'had_non_english': False,
            'translated_count': 0,
            'detected_languages': [],
        }

        paragraphs = re.split(r'(\n\s*\n)', markdown)
        processed_parts = []

        for part in paragraphs:
            # Preserve whitespace separators
            if not part.strip() or re.match(r'^\s*$', part):
                processed_parts.append(part)
                continue

            # Skip markdown headings
            if part.strip().startswith('#'):
                processed_parts.append(part)
                continue

            # Detect the dominant script/language
            detected_script = cls._detect_non_english_script(part)

            if detected_script:
                info['had_non_english'] = True
                info['translated_count'] += 1
                if detected_script not in info['detected_languages']:
                    info['detected_languages'].append(detected_script)

                # Translate the paragraph
                translated = cls._translate_text(part.strip(), detected_script)

                if translated and translated != part.strip():
                    # Replace with translated text + marker
                    processed_parts.append(
                        f"[Translated from {detected_script}] {translated}"
                    )
                else:
                    # Translation failed — keep original with warning marker
                    processed_parts.append(
                        f"[Original {detected_script} text — translation unavailable] {part.strip()}"
                    )
            else:
                processed_parts.append(part)

        return ''.join(processed_parts), info

    @classmethod
    def _detect_non_english_script(cls, text: str) -> Optional[str]:
        """
        Detect if a text block is primarily in a non-English script.

        Returns:
            Script name (e.g., 'CJK', 'Arabic') if non-English, None if English/Latin
        """
        if not text or len(text.strip()) < 10:
            return None

        # Count characters in each script range
        script_counts = {script: 0 for script in cls.SCRIPT_RANGES}
        latin_count = 0
        total_alpha = 0

        for char in text:
            if not char.isalpha():
                continue
            total_alpha += 1
            code_point = ord(char)

            # Check Latin range
            if code_point < 0x0250:  # Basic Latin + Latin Extended
                latin_count += 1
                continue

            # Check non-Latin scripts
            for script, ranges in cls.SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= code_point <= end:
                        script_counts[script] += 1
                        break

        if total_alpha < 5:
            return None

        # Find the dominant non-Latin script
        for script, count in script_counts.items():
            ratio = count / total_alpha
            if ratio > 0.3:  # More than 30% of characters are in this script
                return script

        return None

    @classmethod
    def _translate_text(cls, text: str, source_script: str) -> Optional[str]:
        """
        Translate non-English text to English using deep-translator.

        Args:
            text: Text to translate
            source_script: Detected script name for logging

        Returns:
            Translated text or None if translation failed
        """
        try:
            from deep_translator import GoogleTranslator

            # deep-translator auto-detects source language
            # Translate in chunks if text is too long (Google Translate has a 5000 char limit)
            max_chunk = 4500

            if len(text) <= max_chunk:
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                return translated
            else:
                # Split into chunks at sentence boundaries
                chunks = cls._split_for_translation(text, max_chunk)
                translated_chunks = []
                for chunk in chunks:
                    translated_chunk = GoogleTranslator(source='auto', target='en').translate(chunk)
                    if translated_chunk:
                        translated_chunks.append(translated_chunk)
                    else:
                        translated_chunks.append(chunk)  # Keep original if chunk fails
                return ' '.join(translated_chunks)

        except ImportError:
            logger.warning(
                "[PatentPostProcessor] deep-translator not installed. "
                "Non-English text will be kept as-is. "
                "Install with: pip install deep-translator"
            )
            return None
        except Exception as e:
            logger.error(f"[PatentPostProcessor] Translation failed ({source_script}): {e}")
            return None

    @staticmethod
    def _split_for_translation(text: str, max_len: int) -> List[str]:
        """Split text into chunks at sentence boundaries for translation."""
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_len:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    @staticmethod
    def _final_cleanup(markdown: str) -> str:
        """Final cleanup pass — fix whitespace and formatting issues."""
        # Remove excessive blank lines (more than 2 consecutive)
        markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)

        # Remove trailing whitespace on each line
        lines = [line.rstrip() for line in markdown.split('\n')]
        markdown = '\n'.join(lines)

        # Ensure headings have blank lines before them
        markdown = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', markdown)

        return markdown.strip()
