import json
import re
import asyncio
import os
from datetime import datetime
from docling.document_converter import DocumentConverter


class HeadingExtractor:
    def __init__(self):
        self.converter=DocumentConverter()


    def convert_to_markdown(self,source):
        result=self.converter.convert(source)
        markdown = result.document.export_to_markdown()
        return markdown

    async def convert_to_markdown_async(self, source):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.converter.convert, source)
        markdown = result.document.export_to_markdown()
        return markdown

    async def extract_headings_async(self, markdown_text):
        loop = asyncio.get_event_loop()
        headings = await loop.run_in_executor(None, self.extract_headings, markdown_text)
        return headings

    async def process_pdfs_async(self, pdf_urls):
        """Process multiple PDFs asynchronously and extract headings"""
        all_headings = []
        
        async def process_single_pdf(url):
            try:
                markdown = await self.convert_to_markdown_async(url)
                headings = await self.extract_headings_async(markdown)
                return headings
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return []
        
        tasks = [process_single_pdf(url) for url in pdf_urls]
        results = await asyncio.gather(*tasks)
        
        for headings in results:
            all_headings.extend(headings)
        
        return all_headings

    def clean_heading_text(self,text, is_title=False):
        """Clean heading text by removing content after colons and limiting length."""
        if not text:
            return text

        if is_title:
            if len(text) > 200:
                for i in range(180, min(200, len(text))):
                    if text[i] in ' ,.':
                        text = text[:i].strip()
                        break
                else:
                    text = text[:180].strip()
            return text

        if ':' in text:
            text = text.split(':', 1)[0].strip()

        if len(text) > 100:
            for i in range(80, min(100, len(text))):
                if text[i] in ' ,.':
                    text = text[:i].strip()
                    break
            else:
                text = text[:80].strip()

        return text


    def extract_headings(self,markdown_text):
        headings = []
        markdown_pattern = r'^(#{1,6})\s*(.*)$'

        lines = markdown_text.split('\n')
        is_first_heading = True

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()

            # ONLY check for markdown headers
            md_match = re.match(markdown_pattern, line_stripped)
            if md_match:
                level = len(md_match.group(1))
                heading_text = self.clean_heading_text(md_match.group(2).strip(), is_first_heading)
                headings.append({
                    'type': 'markdown',
                    'level': level,
                    'number': None,
                    'text': heading_text,
                    'raw': line_stripped,
                    'line_num': line_num
                })
                is_first_heading = False

        return headings


    def get_text_between_headings(self,markdown_text, start_heading, end_heading=None):
        """
        Extract text between two headings.

        Args:
            markdown_text: The full markdown text
            start_heading: Text of the starting heading (partial match)
            end_heading: Text of the ending heading (partial match). If None, gets to end of document

        Returns:
            String containing the text between the headings
        """
        lines = markdown_text.split('\n')
        headings = self.extract_headings(markdown_text)

        # Find the start heading
        start_idx = None
        for h in headings:
            if start_heading.lower() in h['text'].lower():
                start_idx = h['line_num']
                break

        if start_idx is None:
            return f"Start heading '{start_heading}' not found"

        # Find the end heading
        end_idx = len(lines)
        if end_heading:
            for h in headings:
                if h['line_num'] > start_idx and end_heading.lower() in h['text'].lower():
                    end_idx = h['line_num']
                    break

        # Extract text between headings (including the headings themselves)
        content_lines = lines[start_idx:end_idx]
        return '\n'.join(content_lines).strip()


    def extract_introduction(self,markdown_text):
        """
        Extract the introduction section from markdown text.

        This function looks for common variations of introduction headings such as:
        - Introduction
        - I. Introduction
        - 1. Introduction

        Args:
            markdown_text: The full markdown text

        Returns:
            String containing the introduction text, or an error message if not found
        """
        lines = markdown_text.split('\n')
        headings = self.extract_headings(markdown_text)

        # Common introduction heading patterns (case-insensitive)
        intro_patterns = ['introduction', 'intro']

        # Find the introduction heading
        start_idx = None
        start_level = None
        for h in headings:
            heading_text_lower = h['text'].lower()
            for pattern in intro_patterns:
                if pattern in heading_text_lower:
                    start_idx = h['line_num']
                    start_level = h['level']
                    break
            if start_idx is not None:
                break

        if start_idx is None:
            return "Introduction section not found"

        # Find the next heading of the same or higher level (lower number = higher level)
        end_idx = len(lines)
        for h in headings:
            if h['line_num'] > start_idx and h['level'] <= start_level:
                end_idx = h['line_num']
                break

        # Extract the introduction content
        content_lines = lines[start_idx + 1:end_idx]
        return '\n'.join(content_lines).strip()


    def get_headings_json(self,headings):
        heading_names = {
            'headings': [h['text'] for h in headings]
        }
        return json.dumps(heading_names, indent=2)
    def extract_conclusion(self,markdown_text):
        """
        Extract the conclusion section from markdown text.

        This function looks for common variations of conclusion headings such as:
        - Conclusion
        - Conclusions
        - Concluding Remarks
        - Summary
        - VI. Conclusion

        Args:
            markdown_text: The full markdown text

        Returns:
            String containing the conclusion text, or an error message if not found
        """
        lines = markdown_text.split('\n')
        headings = self.extract_headings(markdown_text)

        # Common conclusion heading patterns (case-insensitive)
        conclusion_patterns = ['conclusion', 'concluding', 'summary', 'closing']

        # Find the conclusion heading
        start_idx = None
        start_level = None
        for h in headings:
            heading_text_lower = h['text'].lower()
            for pattern in conclusion_patterns:
                if pattern in heading_text_lower:
                    start_idx = h['line_num']
                    start_level = h['level']
                    break
            if start_idx is not None:
                break

        if start_idx is None:
            return "Conclusion section not found"

        # Find the next heading of the same or higher level (lower number = higher level)
        # For conclusion, often it's the last section before references/acknowledgments
        end_idx = len(lines)
        for h in headings:
            if h['line_num'] > start_idx and h['level'] <= start_level:
                # Skip common post-conclusion sections
                heading_lower = h['text'].lower()
                if any(skip in heading_lower for skip in ['reference', 'acknowledgment', 'acknowledgement', 'appendix']):
                    end_idx = h['line_num']
                    break
                # If it's another major section that's not a post-conclusion section, use it as end
                if h['level'] == start_level:
                    end_idx = h['line_num']
                    break

        # Extract the conclusion content
        content_lines = lines[start_idx + 1:end_idx]
        return '\n'.join(content_lines).strip()

    def save_extracted_text_to_file(self, extracted_text_list, filename="extracted_headings.txt"):
        """
        Save the extracted heading text to a txt file.

        Args:
            extracted_text_list (list): List of extracted text strings from headings
            filename (str): Name of the output file (default: "extracted_headings.txt")
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for i, text in enumerate(extracted_text_list, 1):
                    f.write(f"=== EXTRACTED SECTION {i} ===\n")
                    f.write(text)
                    f.write("\n\n" + "=" * 50 + "\n\n")
            print(f"Extracted text saved to: {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")

    def save_pdf_info_to_txt(self, paper_info, extracted_texts, user_idea, paper_index):
        """
        Save PDF information to a text file in the extracted_pdf_info directory.
        
        Args:
            paper_info: Dictionary containing paper title, abstract, url
            extracted_texts: List of extracted text sections
            user_idea: The original research idea
            paper_index: Index of the paper for unique filename
        
        Returns:
            str: Path to the saved file, or None if failed
        """
        # Ensure the directory exists
        output_dir = "Agents/extracted_pdf_info"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a safe filename from paper title
        safe_title = re.sub(r'[^\w\s-]', '', paper_info['title']).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:50]  # Limit length
        
        # Create filename with timestamp and index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"paper_{paper_index:03d}_{safe_title}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare the content
        content = f"""RESEARCH IDEA: {user_idea}

{'='*80}
PAPER INFORMATION
{'='*80}

TITLE: {paper_info['title']}

URL: {paper_info['url']}

ABSTRACT:
{paper_info['abstract']}

{'='*80}
SELECTED HEADINGS AND EXTRACTED CONTENT
{'='*80}

"""
        
        # Add extracted texts
        for i, text in enumerate(extracted_texts, 1):
            content += f"\n{'-'*60}\nSECTION {i}\n{'-'*60}\n\n{text}\n\n"
        
        # Add footer
        content += f"\n{'='*80}\nEXTRACTED ON: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
        
        # Write to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Saved PDF info to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving PDF info to {filepath}: {e}")
            return None

if __name__ == '__main__':
    extractor = HeadingExtractor()
    source = "https://arxiv.org/pdf/2311.07582v1"
    markdown = extractor.convert_to_markdown(source)
    headings = extractor.extract_headings(markdown)
    headings_json = extractor.get_headings_json(headings)
    print(headings_json)

#
#     # # Example: Extract text between two headings
#     # print("\n=== TEXT BETWEEN HEADINGS ===")
#     text = extractor.get_text_between_headings(markdown, "Related Work", "Methods")
#     print(text)

    # # Example: Extract introduction
    # print("\n=== INTRODUCTION ===")
    # introduction = extractor.extract_introduction(markdown)
    # # print(introduction)
    #
    # print("\n=== CONCLUSION ===")
    # conclusion = extract_conclusion(markdown)
    # print(conclusion)