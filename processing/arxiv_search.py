import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import json
import time

from Agents.query_variant_agent import QueryVariantAgent


class ArxivReq:
    """
    ArXiv paper retrieval with high-recall multi-query search and deduplication.
    """

    def __init__(self):
        self.query_variant_agent = QueryVariantAgent()
        # Configurable parameters for retrieval
        self.papers_per_query = 150  # Increased from 10 for high recall
        self.api_delay = 3  # Seconds between API calls (arXiv recommendation)

    def search_arxiv(self, terms=None, operator=None, category=None, search_in="all",
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
            if isinstance(date_from, datetime):
                date_from = date_from.strftime("%Y%m%d%H%M")
            if isinstance(date_to, datetime):
                date_to = date_to.strftime("%Y%m%d%H%M")

            if not date_from:
                date_from = "200001010000"
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

        if sort_by:
            params['sortBy'] = sort_by
            params['sortOrder'] = sort_order

        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"

        response = urllib.request.urlopen(url)
        return response.read().decode('utf-8')

    def parse_arxiv_xml_to_json(self, xml_string):
        """
        Parse arXiv API XML response and convert to JSON format.
        """
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        root = ET.fromstring(xml_string)

        feed_link = root.find('atom:link[@rel="self"]', namespaces)
        feed_title = root.find('atom:title', namespaces)
        feed_id = root.find('atom:id', namespaces)
        feed_updated = root.find('atom:updated', namespaces)

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

        for entry in root.findall('atom:entry', namespaces):
            paper = {}

            id_elem = entry.find('atom:id', namespaces)
            if id_elem is not None:
                paper['id'] = id_elem.text
                paper['arxiv_id'] = id_elem.text.split('/abs/')[-1]

            published = entry.find('atom:published', namespaces)
            if published is not None:
                paper['published'] = published.text

            updated = entry.find('atom:updated', namespaces)
            if updated is not None:
                paper['updated'] = updated.text

            title = entry.find('atom:title', namespaces)
            if title is not None:
                paper['title'] = ' '.join(title.text.split())

            summary = entry.find('atom:summary', namespaces)
            if summary is not None:
                paper['summary'] = ' '.join(summary.text.split())

            authors = []
            for author in entry.findall('atom:author', namespaces):
                name = author.find('atom:name', namespaces)
                if name is not None:
                    authors.append(name.text)
            paper['authors'] = authors

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

            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    categories.append(term)
            paper['categories'] = categories

            primary_cat = entry.find('arxiv:primary_category', namespaces)
            if primary_cat is not None:
                paper['primary_category'] = primary_cat.get('term')

            comment = entry.find('arxiv:comment', namespaces)
            if comment is not None:
                paper['comment'] = comment.text

            journal_ref = entry.find('arxiv:journal_ref', namespaces)
            if journal_ref is not None:
                paper['journal_ref'] = journal_ref.text

            doi = entry.find('arxiv:doi', namespaces)
            if doi is not None:
                paper['doi'] = doi.text

            result['papers'].append(paper)

        return result

    def last_days(self, days):
        """Get datetime for N days ago"""
        return datetime.now() - timedelta(days=days)

    def last_months(self, months):
        """Get approximate datetime for N months ago (30 days per month)"""
        return datetime.now() - timedelta(days=months * 30)

    def search_with_pagination(self, query: str, max_papers: int = 150, **kwargs) -> list:
        """
        Search arXiv with pagination to get more results.

        Args:
            query: Search query string
            max_papers: Maximum number of papers to retrieve
            **kwargs: Additional parameters for search_arxiv

        Returns:
            List of paper dictionaries
        """
        all_papers = []
        batch_size = min(100, max_papers)  # arXiv max per request is 100
        start = 0

        while len(all_papers) < max_papers:
            try:
                xml_result = self.search_arxiv(
                    terms=query,
                    max_results=batch_size,
                    start=start,
                    sort_by="relevance",
                    **kwargs
                )
                json_result = self.parse_arxiv_xml_to_json(xml_result)
                papers = json_result.get('papers', [])

                if not papers:
                    break

                all_papers.extend(papers)
                start += batch_size

                # Check if we got all available results
                if len(papers) < batch_size:
                    break

                # Rate limiting
                time.sleep(self.api_delay)

            except Exception as e:
                print(f"Error fetching papers for query '{query}': {e}")
                break

        return all_papers[:max_papers]

    def search_multiple_queries(self, queries: list, papers_per_query: int = None) -> dict:
        """
        Search multiple query variants and collect results.

        Args:
            queries: List of query variant dictionaries with 'type' and 'query' keys
            papers_per_query: Number of papers to fetch per query (default: self.papers_per_query)

        Returns:
            Dictionary with query as key and results as value
        """
        if papers_per_query is None:
            papers_per_query = self.papers_per_query

        results = {}

        for variant in queries:
            query = variant['query']
            query_type = variant['type']

            print(f"Searching [{query_type}]: {query}")

            papers = self.search_with_pagination(query, max_papers=papers_per_query)
            results[query] = {
                'type': query_type,
                'papers': papers,
                'count': len(papers)
            }

            print(f"  Found {len(papers)} papers")
            time.sleep(self.api_delay)

        return results

    def deduplicate_papers(self, search_results: dict) -> list:
        """
        Merge and deduplicate papers from multiple query results.

        Args:
            search_results: Dictionary of search results from search_multiple_queries

        Returns:
            List of unique paper dictionaries with source query info
        """
        seen_ids = set()
        unique_papers = []

        for query, data in search_results.items():
            for paper in data.get('papers', []):
                arxiv_id = paper.get('arxiv_id', '')

                # Normalize arXiv ID (remove version suffix for deduplication)
                base_id = arxiv_id.split('v')[0] if arxiv_id else ''

                if base_id and base_id not in seen_ids:
                    seen_ids.add(base_id)
                    # Add source query information
                    paper['source_query'] = query
                    paper['source_type'] = data.get('type', 'unknown')
                    unique_papers.append(paper)

        print(f"Deduplicated: {sum(d['count'] for d in search_results.values())} → {len(unique_papers)} unique papers")
        return unique_papers

    def convert_to_jsonl_format(self, papers: list) -> list:
        """
        Convert papers to JSONL format compatible with retrieval/paper_search.py

        Args:
            papers: List of paper dictionaries (already deduplicated)

        Returns:
            List of dictionaries in JSONL format
        """
        jsonl_papers = []

        for paper in papers:
            year = None
            if 'published' in paper:
                try:
                    year = int(paper['published'][:4])
                except (ValueError, TypeError):
                    year = None

            jsonl_entry = {
                "id": paper.get('arxiv_id', ''),
                "title": paper.get('title', ''),
                "abstract": paper.get('summary', ''),
                "url": paper.get('links', {}).get('html', ''),
                "year": year,
                "categories": paper.get('categories', []),
                "source_query": paper.get('source_query', ''),
                "source_type": paper.get('source_type', ''),
            }

            jsonl_papers.append(jsonl_entry)

        return jsonl_papers

    def save_to_jsonl_file(self, jsonl_papers: list, filename: str = "retrieval/sample_papers.jsonl"):
        """
        Save papers to JSONL file format
        """
        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            for paper in jsonl_papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')

        print(f"Saved {len(jsonl_papers)} papers to {filename}")

    def get_papers(self, user_idea: str, papers_per_query: int = None) -> str:
        """
        Main entry point: Generate query variants, search arXiv, deduplicate, and save.

        Args:
            user_idea: User's research idea/description
            papers_per_query: Number of papers to fetch per query variant

        Returns:
            JSON string with search results summary
        """
        # Step 1: Generate query variants
        print("=" * 60)
        print("Step 1: Generating query variants...")
        query_variants = self.query_variant_agent.generate_query_variants(user_idea)
        print(f"Generated {len(query_variants)} query variants:")
        for v in query_variants:
            print(f"  [{v['type']}] {v['query']}")

        # Step 2: Search arXiv with all variants
        print("\n" + "=" * 60)
        print("Step 2: Searching arXiv...")
        search_results = self.search_multiple_queries(query_variants, papers_per_query)

        # Step 3: Deduplicate
        print("\n" + "=" * 60)
        print("Step 3: Deduplicating papers...")
        unique_papers = self.deduplicate_papers(search_results)

        # Step 4: Convert and save to JSONL
        print("\n" + "=" * 60)
        print("Step 4: Saving to JSONL...")
        jsonl_papers = self.convert_to_jsonl_format(unique_papers)
        self.save_to_jsonl_file(jsonl_papers)

        # Return summary
        summary = {
            "query_variants": query_variants,
            "total_papers_fetched": sum(d['count'] for d in search_results.values()),
            "unique_papers": len(unique_papers),
            "papers": jsonl_papers
        }

        return json.dumps(summary, indent=2)
