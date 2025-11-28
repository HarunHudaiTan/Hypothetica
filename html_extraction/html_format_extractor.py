import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import json
import os
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
from ArxivReq import last_days


def parse_arxiv_response(xml_response):
    """
    Parse arXiv API XML response and extract paper information.
    
    Args:
        xml_response: XML string from arXiv API
        
    Returns:
        List of dictionaries containing paper information
    """
    papers = []
    root = ET.fromstring(xml_response)
    
    # Define namespaces
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    for entry in root.findall('atom:entry', ns):
        paper = {}
        
        # Get basic information
        paper['title'] = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        paper['summary'] = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        paper['published'] = entry.find('atom:published', ns).text.strip()
        paper['updated'] = entry.find('atom:updated', ns).text.strip()
        
        # Get arXiv ID
        arxiv_id = entry.find('atom:id', ns).text.strip()
        paper['arxiv_id'] = arxiv_id.split('/abs/')[-1]
        
        # Get authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text.strip())
        paper['authors'] = authors
        
        # Get all links and check for HTML format
        links = {}
        has_html_link = False
        for link in entry.findall('atom:link', ns):
            href = link.get('href')
            link_type = link.get('type')
            title = link.get('title')
            
            # Store link if it has a title
            if title and href:
                links[title] = href
            
            # Check if this is an HTML link by type or URL pattern
            if href:
                # Check by MIME type - arXiv indicates HTML availability with type="text/html"
                if link_type == 'text/html':
                    has_html_link = True
                # Or check by URL pattern (arxiv.org/html/)
                elif '/html/' in href:
                    has_html_link = True
        
        paper['links'] = links
        
        # If API indicates HTML is available, construct the proper HTML URL
        # arXiv HTML format: https://arxiv.org/html/{arxiv_id}
        if has_html_link:
            paper['html_url'] = f"https://arxiv.org/html/{paper['arxiv_id']}"
            paper['has_html'] = True
        else:
            paper['has_html'] = False
        
        papers.append(paper)
    
    return papers


def fetch_html_content(html_url):
    """
    Fetch HTML content from arXiv HTML URL.
    
    Args:
        html_url: URL to the HTML version of the paper
        
    Returns:
        HTML content as string
    """
    try:
        response = urllib.request.urlopen(html_url)
        return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching HTML from {html_url}: {e}")
        return None


def debug_print_headings(html_content):
    """Debug function to print all headings found in HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    print("  DEBUG - Found headings:")
    for h in headings[:10]:  # Print first 10
        text = h.get_text().strip()
        clean = re.sub(r'^\d+\.?\s*', '', text)
        print(f"    {h.name}: {clean[:60]}")
    if len(headings) > 10:
        print(f"    ... and {len(headings) - 10} more")


def extract_section_from_html(html_content, section_patterns):
    """
    Extract a specific section from HTML content using a robust multi-strategy approach.
    
    Args:
        html_content: HTML content as string
        section_patterns: List of patterns to match section headings (e.g., ['introduction', 'intro'])
        
    Returns:
        Extracted section text or error message
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Strategy 1: Look for section elements with IDs (arXiv often uses <section id="S1">)
    for section in soup.find_all('section'):
        # Check the heading in this section
        heading = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if heading:
            heading_text = heading.get_text().strip()
            clean_heading = re.sub(r'^\d+\.?\s*', '', heading_text).lower()
            
            # Check if it matches any pattern
            for pattern in section_patterns:
                if pattern.lower() in clean_heading or clean_heading.startswith(pattern.lower()):
                    # Extract all text from this section, excluding subsection headings
                    content_parts = []
                    
                    # Get direct text and paragraph elements
                    for elem in section.find_all(['p', 'div']):
                        # Skip if it's a heading container
                        if elem.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                            continue
                        
                        text = elem.get_text().strip()
                        text = re.sub(r'\s+', ' ', text)
                        
                        # Filter out references, citations, and very short text
                        if text and len(text) > 20 and not text.startswith('[') and not text.startswith('Figure'):
                            content_parts.append(text)
                    
                    if content_parts:
                        return '\n\n'.join(content_parts[:50])  # Limit to first 50 paragraphs
    
    # Strategy 2: Text-based search - find the heading text and extract paragraphs after it
    # Get the full text to search
    full_text = soup.get_text()
    
    for pattern in section_patterns:
        # Search for the pattern in the full text (case-insensitive)
        # Match patterns like "1. Introduction" or "Introduction" at the start of a line
        regex_patterns = [
            rf'(?:^|\n)\s*\d+\.?\s*{re.escape(pattern)}\s*(?:\n|$)',  # "1. Introduction"
            rf'(?:^|\n)\s*{re.escape(pattern)}\s*(?:\n|$)',  # "Introduction"
        ]
        
        match = None
        for regex_pat in regex_patterns:
            match = re.search(regex_pat, full_text, re.IGNORECASE | re.MULTILINE)
            if match:
                break
        
        if match:
            # Find where this section ends (next section heading)
            section_start = match.end()
            section_text = full_text[section_start:]
            
            # Find the end of this section (next major heading)
            end_patterns = r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:' + '|'.join([
                'Introduction', 'Methodology', 'Method', 'Approach', 'Results', 
                'Discussion', 'Conclusion', 'References', 'Related Work', 'Background',
                'Experiments', 'Evaluation', 'Implementation'
            ]) + r')\s*(?:\n|$)'
            
            end_match = re.search(end_patterns, section_text, re.IGNORECASE | re.MULTILINE)
            if end_match:
                section_text = section_text[:end_match.start()]
            
            # Clean up the text
            section_text = section_text.strip()
            section_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', section_text)  # Remove excessive newlines
            section_text = re.sub(r'[ \t]+', ' ', section_text)  # Clean up spaces
            
            if section_text and len(section_text) > 50:
                return section_text[:10000]  # Limit to reasonable size
    
    # Strategy 3: Find heading elements and extract sibling paragraphs
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        for heading in soup.find_all(tag):
            heading_text = heading.get_text().strip()
            clean_heading = re.sub(r'^\d+\.?\s*', '', heading_text).lower()
            
            for pattern in section_patterns:
                if pattern.lower() in clean_heading:
                    content_parts = []
                    
                    # Get following siblings
                    for sibling in heading.find_next_siblings():
                        # Stop at next heading of same or higher level
                        if sibling.name and sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            sibling_level = int(sibling.name[1])
                            heading_level = int(tag[1])
                            if sibling_level <= heading_level:
                                break
                        
                        # Extract text from paragraphs
                        if sibling.name == 'p':
                            text = sibling.get_text().strip()
                            text = re.sub(r'\s+', ' ', text)
                            if text and len(text) > 20:
                                content_parts.append(text)
                    
                    if content_parts:
                        return '\n\n'.join(content_parts[:50])
    
    return f"Section not found (searched for: {', '.join(section_patterns)})"


def extract_introduction(html_content):
    """Extract the Introduction section from HTML content."""
    patterns = ['introduction', 'intro']
    return extract_section_from_html(html_content, patterns)


def extract_methodology(html_content):
    """Extract the Methodology section from HTML content."""
    patterns = ['methodology', 'method', 'approach']
    return extract_section_from_html(html_content, patterns)


def extract_conclusion(html_content):
    """Extract the Conclusion section from HTML content."""
    patterns = ['conclusion', 'conclusions']
    return extract_section_from_html(html_content, patterns)


def process_html_paper(paper):
    """
    Process a single paper with HTML format and extract sections.
    
    Args:
        paper: Dictionary containing paper information
        
    Returns:
        Dictionary with extracted sections or None if processing fails
    """
    if not paper['has_html']:
        return None
    
    print(f"Processing: {paper['title'][:80]}...")
    
    html_content = fetch_html_content(paper['html_url'])
    if not html_content:
        return None
    
    # Debug: print available headings (disabled after successful testing)
    # debug_print_headings(html_content)
    
    # Extract sections with status reporting
    intro = extract_introduction(html_content)
    method = extract_methodology(html_content)
    concl = extract_conclusion(html_content)
    
    # Show extraction status
    status = []
    if not intro.startswith("Section not found"):
        status.append("✓ Intro")
    if not method.startswith("Section not found"):
        status.append("✓ Method")
    if not concl.startswith("Section not found"):
        status.append("✓ Concl")
    
    if status:
        print(f"  Extracted: {', '.join(status)}")
    else:
        print(f"  ⚠ No sections extracted")
    
    result = {
        'arxiv_id': paper['arxiv_id'],
        'title': paper['title'],
        'authors': paper['authors'],
        'published': paper['published'],
        'updated': paper['updated'],
        'summary': paper['summary'],
        'html_url': paper['html_url'],
        'sections': {
            'introduction': intro,
            'methodology': method,
            'conclusion': concl
        }
    }
    
    return result


def save_results_to_json(results, topic, output_folder='arxiv_html_papers'):
    """
    Save extracted results to JSON file in a folder.
    
    Args:
        results: List of processed papers
        topic: Search topic for filename
        output_folder: Folder to save results
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create filename with topic and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
    filename = f"{safe_topic}_{timestamp}.json"
    filepath = os.path.join(output_folder, filename)
    
    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'topic': topic,
            'timestamp': timestamp,
            'total_papers': len(results),
            'papers': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {filepath}")
    print(f"  Total papers with HTML format: {len(results)}")


def search_arxiv_with_retry(terms, max_results=100, date_from=None, max_retries=3):
    """
    Search arXiv with retry logic for handling temporary failures.
    
    Args:
        terms: Search terms
        max_results: Maximum number of results
        date_from: Optional date filter
        max_retries: Maximum number of retry attempts
        
    Returns:
        XML response string
    """
    for attempt in range(max_retries):
        try:
            # Build URL manually with User-Agent header (arXiv recommends this)
            import urllib.parse
            
            query_parts = []
            if isinstance(terms, str):
                query_parts.append(f"all:{terms}")
            
            if date_from:
                from datetime import datetime
                if isinstance(date_from, datetime):
                    date_from_str = date_from.strftime("%Y%m%d%H%M")
                else:
                    date_from_str = date_from
                date_to_str = datetime.now().strftime("%Y%m%d%H%M")
                query_parts.append(f"submittedDate:[{date_from_str} TO {date_to_str}]")
            
            if len(query_parts) > 1:
                query = " AND ".join([f"({part})" for part in query_parts])
            else:
                query = query_parts[0]
            
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results
            }
            
            url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
            
            # Create request with User-Agent header
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (compatible; ArxivHTMLExtractor/1.0)')
            
            response = urllib.request.urlopen(req, timeout=30)
            return response.read().decode('utf-8')
            
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff: 5, 10, 15 seconds
                print(f"\n⚠ arXiv server busy (503 error). Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"\n⚠ Error: {e}. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise


def search_html_papers(topic, max_results=100, date_from=None):
    """
    Search for papers on a specific topic and extract sections from HTML versions.
    
    Args:
        topic: Search topic/query
        max_results: Maximum number of results to retrieve
        date_from: Optional date filter (datetime object or string)
        
    Returns:
        List of processed papers with extracted sections
    """
    print(f"\n🔍 Searching arXiv for: '{topic}'")
    print(f"   Max results: {max_results}")
    
    try:
        # Search arXiv with retry logic
        xml_response = search_arxiv_with_retry(terms=topic, max_results=max_results, date_from=date_from)
    except urllib.error.HTTPError as e:
        print(f"\n❌ Error: arXiv API returned HTTP {e.code}: {e.reason}")
        print("   This could be due to:")
        print("   • Temporary server overload")
        print("   • Rate limiting")
        print("   • Service maintenance")
        print("\n   Please try again in a few minutes.")
        return []
    except Exception as e:
        print(f"\n❌ Error searching arXiv: {e}")
        return []
    
    # Parse response
    try:
        papers = parse_arxiv_response(xml_response)
        print(f"   Found {len(papers)} total papers")
    except Exception as e:
        print(f"\n❌ Error parsing arXiv response: {e}")
        return []
    
    # Filter papers with HTML format
    html_papers = [p for p in papers if p['has_html']]
    print(f"   Papers with HTML format: {len(html_papers)}")
    
    if not html_papers:
        print("\n⚠ No papers with HTML format found for this query.")
        return []
    
    # Process each HTML paper
    results = []
    for i, paper in enumerate(html_papers, 1):
        print(f"\n[{i}/{len(html_papers)}] ", end="")
        result = process_html_paper(paper)
        if result:
            results.append(result)
        
        # Be nice to arXiv servers
        time.sleep(1)
    
    return results


def main():
    """Main function to run the HTML format extractor."""
    print("=" * 70)
    print("arXiv HTML Format Paper Extractor")
    print("=" * 70)
    print("\nThis tool searches arXiv for papers and extracts:")
    print("  • Introduction")
    print("  • Methodology")
    print("  • Conclusion")
    print("\nNote: Only papers with HTML format on arXiv will be processed.")
    print("=" * 70)
    
    # Ask user for topic
    topic = input("\n📝 What topic do you want to search for? ").strip()
    
    if not topic:
        print("❌ Error: Topic cannot be empty")
        return
    
    # Ask for number of results
    max_results_input = input("\n📊 Maximum number of papers to retrieve (press Enter for no limit/10000): ").strip()
    if max_results_input:
        try:
            max_results = int(max_results_input)
        except ValueError:
            print("⚠ Invalid number, using default: 10000")
            max_results = 10000
    else:
        max_results = 10000
    
    # Ask for date filter (optional)
    date_filter = input("\n📅 Filter by date? (e.g., '1' for last 1 year, '2' for last 2 years, or press Enter to skip): ").strip()
    date_from = None
    if date_filter:
        try:
            years = int(date_filter)
            days = years * 365  # Convert years to approximate days
            date_from = last_days(days)
            print(f"   Filtering papers from last {years} year(s) (~{days} days)")
        except ValueError:
            print("⚠ Invalid date, searching all papers")
    
    # Search and process
    print("\n" + "=" * 70)
    results = search_html_papers(topic, max_results=max_results, date_from=date_from)
    
    # Save results
    if results:
        save_results_to_json(results, topic)
        print("\n✅ Done!")
    else:
        print("\n⚠ No results to save.")


if __name__ == "__main__":
    main()

