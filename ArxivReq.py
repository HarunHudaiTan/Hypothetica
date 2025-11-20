import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import json

from Agents.keyword_agent import KeywordAgent


class ArxivReq:
    def __init__(self):
     self.keyword_agent = KeywordAgent()
    def search_arxiv(self,terms=None, operator=None, category=None, search_in="all",
                     max_results=10, start=0, sort_by=None, sort_order="descending",
                     date_from=None, date_to=None):
        """
        Search arXiv API for papers.

        Args:
            terms: Single term (str) or list of terms, e.g., "RAG" or ["RAG", "agents"]
            operator: "AND" or "OR" - how to combine multiple terms
            category: arXiv category to filter by, e.g., "cs.AI"
            search_in: Where to search - "all", "ti" (title), "abs" (abstract), "au" (author)
            max_results: Number of results to return (default: 10)
            start: Starting index for pagination (default: 0)
            sort_by: "relevance", "lastUpdatedDate", "submittedDate"
            sort_order: "ascending" or "descending" (default: "descending")
            date_from: Start date for filtering - datetime object or "YYYYMMDDHHMM" string
            date_to: End date for filtering - datetime object or "YYYYMMDDHHMM" string

        Returns:
            String containing the Atom XML response

        Examples:
            # Papers from last 6 months
            search_arxiv("RAG", date_from=datetime.now() - timedelta(days=180))

            # Papers from 2024
            search_arxiv("agents", date_from="202401010000", date_to="202412312359")

            # Recent AI papers from last year
            search_arxiv(category="cs.AI", date_from=datetime.now() - timedelta(days=365))
        """

        query_parts = []

        # Build search query from terms
        if terms:
            if isinstance(terms, str):
                terms = [terms]

            if len(terms) == 1:
                query_parts.append(f"{search_in}:{terms[0]}")
            elif operator:
                query_parts.append(f" {operator} ".join([f"{search_in}:{term}" for term in terms]))
            else:
                query_parts.append(" ".join([f"{search_in}:{term}" for term in terms]))

        # Add category filter if specified
        if category:
            query_parts.append(f"cat:{category}")

        # Add date range filter if specified
        if date_from or date_to:
            # Convert datetime objects to string format if needed
            if isinstance(date_from, datetime):
                date_from = date_from.strftime("%Y%m%d%H%M")
            if isinstance(date_to, datetime):
                date_to = date_to.strftime("%Y%m%d%H%M")

            # Set defaults if only one date is provided
            if not date_from:
                date_from = "200001010000"  # arXiv started in 2000
            if not date_to:
                date_to = datetime.now().strftime("%Y%m%d%H%M")

            query_parts.append(f"submittedDate:[{date_from} TO {date_to}]")

        # Combine all parts
        if len(query_parts) > 1:
            query = " AND ".join([f"({part})" for part in query_parts])
        elif len(query_parts) == 1:
            query = query_parts[0]
        else:
            raise ValueError("Must provide either terms, category, or date range")

        # Build the full URL with parameters
        params = {
            'search_query': query,
            'start': start,
            'max_results': max_results
        }

        # Add sorting parameters if specified
        if sort_by:
            params['sortBy'] = sort_by
            params['sortOrder'] = sort_order

        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"

        # Make the request
        response = urllib.request.urlopen(url)
        return response.read().decode('utf-8')


    def parse_arxiv_xml_to_json(self,xml_string):
        """
        Parse arXiv API XML response and convert to JSON format.

        Args:
            xml_string: XML string from arXiv API response

        Returns:
            Dictionary containing parsed paper information
        """
        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        # Parse XML
        root = ET.fromstring(xml_string)

        # Extract feed-level metadata
        feed_link = root.find('atom:link[@rel="self"]', namespaces)
        feed_title = root.find('atom:title', namespaces)
        feed_id = root.find('atom:id', namespaces)
        feed_updated = root.find('atom:updated', namespaces)

        # Extract opensearch metadata
        total_results = root.find('opensearch:totalResults', namespaces)
        start_index = root.find('opensearch:startIndex', namespaces)
        items_per_page = root.find('opensearch:itemsPerPage', namespaces)

        result = {
            'feed_link': feed_link.get('href') if feed_link is not None else None,
            'feed_title': feed_title.text if feed_title is not None else None,
            'feed_id': feed_id.text if feed_id is not None else None,
            'feed_updated': feed_updated.text if feed_updated is not None else None,
            'total_results': int(total_results.text) if total_results is not None else 0,
            'start_index': int(start_index.text) if start_index is not None else 0,
            'items_per_page': int(items_per_page.text) if items_per_page is not None else 0,
            'papers': []
        }

        # Extract paper entries
        for entry in root.findall('atom:entry', namespaces):
            paper = {}

            # Extract ID and convert to arxiv_id
            id_elem = entry.find('atom:id', namespaces)
            if id_elem is not None:
                paper['id'] = id_elem.text
                # Extract arxiv_id from URL (e.g., "2205.06168v1" from "http://arxiv.org/abs/2205.06168v1")
                paper['arxiv_id'] = id_elem.text.split('/abs/')[-1]

            # Extract dates
            published = entry.find('atom:published', namespaces)
            if published is not None:
                paper['published'] = published.text

            updated = entry.find('atom:updated', namespaces)
            if updated is not None:
                paper['updated'] = updated.text

            # Extract title
            title = entry.find('atom:title', namespaces)
            if title is not None:
                # Clean up title (remove extra whitespace and newlines)
                paper['title'] = ' '.join(title.text.split())

            # Extract summary/abstract
            summary = entry.find('atom:summary', namespaces)
            if summary is not None:
                # Clean up summary
                paper['summary'] = ' '.join(summary.text.split())

            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name = author.find('atom:name', namespaces)
                if name is not None:
                    authors.append(name.text)
            paper['authors'] = authors

            # Extract links
            links = {}
            for link in entry.findall('atom:link', namespaces):
                rel = link.get('rel')
                title_attr = link.get('title')
                href = link.get('href')

                if title_attr == 'pdf':
                    links['pdf'] = href
                elif rel == 'alternate':
                    links['html'] = href
            paper['links'] = links

            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    categories.append(term)
            paper['categories'] = categories

            # Extract primary category
            primary_cat = entry.find('arxiv:primary_category', namespaces)
            if primary_cat is not None:
                paper['primary_category'] = primary_cat.get('term')

            # Extract comment if present
            comment = entry.find('arxiv:comment', namespaces)
            if comment is not None:
                paper['comment'] = comment.text

            # Extract journal reference if present
            journal_ref = entry.find('arxiv:journal_ref', namespaces)
            if journal_ref is not None:
                paper['journal_ref'] = journal_ref.text

            # Extract DOI if present
            doi = entry.find('arxiv:doi', namespaces)
            if doi is not None:
                paper['doi'] = doi.text

            result['papers'].append(paper)

        return result


    # Helper functions for common date ranges
    def last_days(self,days):
        """Get datetime for N days ago"""
        return datetime.now() - timedelta(days=days)


    def last_months(self,months):
        """Get approximate datetime for N months ago (30 days per month)"""
        return datetime.now() - timedelta(days=months * 30)


    def search_multiple_topics(self,topics, return_json=True, **kwargs):
        """
        Search multiple topics separately with individual API calls.

        Args:
            topics: List of topics to search separately
            return_json: If True, return parsed JSON; if False, return raw XML (default: True)
            **kwargs: Additional parameters passed to search_arxiv

        Returns:
            Dictionary with results for each topic (JSON format if return_json=True)
        """
        results = {}
        for topic in topics:
            print(f"Searching for: {topic}...")
            xml_result = self.search_arxiv(topic, **kwargs)

            if return_json:
                # Parse XML to JSON
                json_result = self.parse_arxiv_xml_to_json(xml_result)
                results[topic] = json_result
                print(f"Found {json_result['total_results']} results for '{topic}', retrieved {len(json_result['papers'])} papers")
            else:
                results[topic] = xml_result
                # Count results from XML response
                import re
                total_results = re.search(r'<opensearch:totalResults[^>]*>(\d+)</opensearch:totalResults>', xml_result)
                if total_results:
                    count = total_results.group(1)
                    print(f"Found {count} results for '{topic}'")

            # Be nice to the API - add a delay between requests
            import time
            time.sleep(3)  # 3 second delay as recommended

        return results

    def convert_to_jsonl_format(self, search_results):
        """
        Convert ArXiv search results to JSONL format compatible with embed_mvp.py
        
        Args:
            search_results: Dictionary of search results from search_multiple_topics
            
        Returns:
            List of dictionaries in JSONL format
        """
        jsonl_papers = []
        
        for topic, topic_results in search_results.items():
            for paper in topic_results.get('papers', []):
                # Extract year from published date (format: "2025-02-25T05:55:15Z")
                year = None
                if 'published' in paper:
                    try:
                        year = int(paper['published'][:4])
                    except (ValueError, TypeError):
                        year = None
                
                # Create JSONL format entry
                jsonl_entry = {
                    "search_keyword": topic,
                    "id": paper.get('arxiv_id', ''),
                    "title": paper.get('title', ''),
                    "abstract": paper.get('summary', ''),  # ArXiv uses 'summary' for abstract
                    "url": paper.get('links', {}).get('html', ''),
                    "year": year,
                    "categories": paper.get('categories', []),
                    # Add the keyword that was used to find this paper
                }
                
                jsonl_papers.append(jsonl_entry)
        
        return jsonl_papers
    
    def save_to_jsonl_file(self, jsonl_papers, filename="embeddemo/sample_papers.jsonl"):
        """
        Save papers to JSONL file format
        
        Args:
            jsonl_papers: List of paper dictionaries in JSONL format
            filename: Output filename (default: embeddemo/sample_papers.jsonl)
        """
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for paper in jsonl_papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(jsonl_papers)} papers to {filename}")

    def get_papers(self,user_idea):
        keywords= self.keyword_agent.generate_keyword_agent_response(user_idea)
        print("keywords: ", keywords)
        search_results = self.search_multiple_topics(keywords)
        
        # Convert to JSONL format and save to file
        jsonl_papers = self.convert_to_jsonl_format(search_results)
        self.save_to_jsonl_file(jsonl_papers)
        
        return json.dumps(search_results,indent=4)


# Example usage:
# if __name__ == "__main__":
#
#     prompt=''' Theoretical Bounds on Sample Complexity for Few-Shot Learning
#
# I'm exploring the theoretical foundations of few-shot learning - specifically, what are
# the fundamental limits on how few examples are needed to learn a new task? I want to
# derive sample complexity bounds that depend on task similarity, model capacity, and the
# structure of the meta-learning algorithm. This could help explain why certain meta-learning
# architectures (like MAML or Prototypical Networks) work better than others and guide the
# design of more sample-efficient algorithms.'''
#
#     request=ArxivReq()
#     print(request.get_papers(prompt))






    # print("\n" + "=" * 50 + "\n")

    # # Papers from 2024 only
    # print("All RAG papers from 2024:")
    # result = search_arxiv("RAG", date_from="202401010000", date_to="202412312359", max_results=10)
    # print(result[:500])

    # print("\n" + "=" * 50 + "\n")

    # # Last 30 days, newest first
    # print("Recent papers from last 30 days:")
    # result = search_arxiv(["RAG", "retrieval"],
    #                       date_from=last_days(30),
    #                       sort_by="submittedDate",
    #                       sort_order="descending",
    #                       max_results=5)
    # print(result[:500])