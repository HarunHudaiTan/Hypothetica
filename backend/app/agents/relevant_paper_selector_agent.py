import json
from app.agents.Agent import Agent, QuotaExhaustedError


class RelevantPaperSelectorAgent(Agent):
    """
    LLM paper selector that receives papers already filtered by cross-encoder.
    Reviews full abstracts and selects final papers.
    
    Pipeline: Embedding Top 100 → Cross-Encoder Top 20 → LLM (this) → Final 5
    """
    
    def __init__(self):
        super().__init__(
            system_prompt="""You are a research paper relevance selector. Your task is to identify the most relevant papers from a pre-filtered shortlist based on the user's research idea.

## Context
You receive papers that have already been filtered by semantic similarity (embeddings) and re-ranked by a cross-encoder. Your job is the final human-like judgment on which papers are truly most relevant.

## Selection Criteria
- Prioritize papers that share core concepts, methodologies, or theoretical frameworks with the user's idea
- Focus on conceptual alignment over superficial keyword matches
- Consider papers that address similar problems or use comparable approaches
- Value papers that could provide foundational knowledge or complementary techniques
- Consider recency and impact when relevant

## Output Format
Return ONLY valid JSON - a list of selected papers with their IDs and reasons:

{
  "selected": [
    {"id": "arxiv_id_1", "reason": "brief reason for selection"},
    {"id": "arxiv_id_2", "reason": "brief reason for selection"},
    ...
  ]
}

Be selective and critical. Quality over quantity.
""",
            top_p=0.7,
            top_k=30,
            temperature=0.2,
            response_mime_type="application/json"
        )

    def _prepare_papers_for_llm(self, papers: list) -> str:
        """Prepare papers with full abstracts for LLM review."""
        lines = []
        for paper in papers:
            paper_id = paper.get('id', 'unknown')
            title = paper.get('title', 'No title')
            abstract = paper.get('abstract', 'No abstract')
            url = paper.get('url', '')
            year = paper.get('year', 'N/A')
            
            # Include rerank score if available
            score_info = ""
            if 'rerank_score' in paper:
                score_info = f"\nRerank Score: {paper['rerank_score']:.4f}"
            elif 'score' in paper:
                score_info = f"\nSimilarity Score: {paper['score']:.4f}"
            
            lines.append(f"[{paper_id}] {title}\nYear: {year} | URL: {url}{score_info}\nAbstract: {abstract}")
        
        return "\n\n---\n\n".join(lines)

    def select_papers(self, user_idea: str, papers: list, select_count: int = 5, use_llm: bool = True, adapter_name: str = "arxiv") -> list:
        """
        Select final papers from cross-encoder filtered shortlist.
        
        Args:
            user_idea: User's research idea
            papers: List of paper dicts (already filtered by cross-encoder, ~20 papers)
            select_count: Number of final papers to select
            use_llm: Whether to use LLM for selection (if False, use cross-encoder ranking)
            
        Returns:
            List of selected paper dictionaries with full info
        """
        if len(papers) <= select_count:
            return papers
        
        # If LLM is disabled, just return top papers from cross-encoder ranking
        if not use_llm:
            print("Using cross-encoder ranking directly (LLM disabled)")
            return papers[:select_count]
        
        papers_text = self._prepare_papers_for_llm(papers)
        
        # GitHub-specific selection criteria
        if adapter_name == "github":
            selection_criteria = """Consider:
1. **Implementation quality** - Well-maintained, documented, actively used repos
2. **Feature completeness** - Repos that implement the core concepts
3. **Code maturity** - Production-ready vs experimental prototypes
4. **Relevance to use case** - Direct applicability to the user's idea
5. **Popular frameworks** - Prefer established tools (LlamaIndex, LangChain, etc.) over one-off projects"""
            content_type = "REPOSITORIES"
        else:
            selection_criteria = """Consider:
1. Direct methodological relevance
2. Theoretical foundations the user might need
3. Similar problem domains and approaches
4. Complementary techniques that could be useful"""
            content_type = "PAPERS"
        
        prompt = f"""USER'S RESEARCH IDEA:
{user_idea}

SHORTLISTED {content_type} ({len(papers)} items, pre-filtered by semantic similarity):
{papers_text}

These {content_type.lower()} have already been filtered by embedding similarity and cross-encoder reranking. 
Your task is to select the {select_count} MOST relevant {content_type.lower()} for understanding and advancing the user's research idea.

{selection_criteria}

Return the {select_count} best {content_type.lower()} with brief reasons for each selection."""

        try:
            response = self.generate_text_generation_response(prompt)
            result = json.loads(response.text)
            
            print(f"[DEBUG] LLM response: {json.dumps(result, indent=2)}")
            
            # Build lookup and preserve LLM's ranking order
            id_to_paper = {p.get('id'): p for p in papers}
            selected_papers = []
            
            selected_list = result.get('selected', [])
            print(f"[DEBUG] Selected list from LLM: {len(selected_list)} items")
            print(f"[DEBUG] Available paper IDs: {list(id_to_paper.keys())[:5]}...")
            
            for item in selected_list:
                paper_id = item.get('id')
                print(f"[DEBUG] Looking for paper_id: {paper_id}")
                if paper_id in id_to_paper:
                    paper = id_to_paper[paper_id].copy()
                    paper['selection_reason'] = item.get('reason', '')
                    selected_papers.append(paper)
                else:
                    print(f"[DEBUG] Paper ID {paper_id} not found in lookup!")
            
            print(f"[DEBUG] Final selected papers: {len(selected_papers)}")
            
            if not selected_papers and papers:
                print(f"[WARNING] LLM selected 0 papers, using fallback to top {select_count}")
                return papers[:select_count]
            
            return selected_papers[:select_count]
            
        except QuotaExhaustedError as e:
            # Quota exhausted - use cross-encoder results (they're already very good!)
            print(f"⚠️ API quota exhausted. Using cross-encoder ranking (still high quality!)")
            for paper in papers[:select_count]:
                paper['selection_reason'] = 'Selected by cross-encoder ranking (API quota limit reached)'
            return papers[:select_count]
            
        except Exception as e:
            # Other failures - fallback to cross-encoder ranking
            print(f"LLM selection failed ({e}), falling back to cross-encoder ranking")
            return papers[:select_count]

    def generate_relevant_paper_selector_response(self, 
                                                   user_idea: str, 
                                                   papers: str,
                                                   final_count: int = 5) -> str:
        """
        Main entry point: Select final papers from cross-encoder filtered list.
        
        Pipeline assumed:
        - Embedding search → Top 100
        - Cross-encoder rerank → Top 20 (input to this method)
        - This LLM selector → Final 5
        
        Args:
            user_idea: User's research idea
            papers: JSON string of papers from cross-encoder reranking (~20 papers)
            final_count: Final number of papers to select
            
        Returns:
            JSON string of selected papers
        """
        # Parse input papers
        if isinstance(papers, str):
            papers_list = json.loads(papers)
        else:
            papers_list = papers
            
        # Handle different input formats
        if isinstance(papers_list, dict):
            if 'results' in papers_list:
                papers_list = papers_list['results']
            elif 'papers' in papers_list:
                papers_list = papers_list['papers']
        
        if not papers_list:
            return json.dumps([])
        
        print(f"\n{'='*60}")
        print(f"LLM Paper Selection")
        print(f"Input: {len(papers_list)} papers (from cross-encoder)")
        
        # Single-stage selection (cross-encoder already did the heavy lifting)
        print(f"Selecting final {final_count} papers...")
        final_papers = self.select_papers(user_idea, papers_list, final_count)
        print(f"Selected {len(final_papers)} papers")
        
        # Format output
        output = []
        for paper in final_papers:
            output.append({
                'id': paper.get('id', ''),
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'url': paper.get('url', ''),
                'year': paper.get('year'),
                'categories': paper.get('categories', []),
                'selection_reason': paper.get('selection_reason', ''),
                'rerank_score': paper.get('rerank_score'),
            })
        
        return json.dumps(output, indent=2)


# Testing
if __name__ == '__main__':
    from app.retrieval.paper_search import QueryWrapper
    
    selector = RelevantPaperSelectorAgent()
    wrapper = QueryWrapper()
    
    user_idea = """I'm exploring the theoretical foundations of few-shot learning - specifically, what are
the fundamental limits on how few examples are needed to learn a new task? I want to
derive sample complexity bounds that depend on task similarity, model capacity, and the 
structure of the meta-learning algorithm."""

    # Get papers from embedding search + cross-encoder (already filtered to ~20-30)
    papers = wrapper.search_literature(
        user_idea, 
        include_scores=True,
        embedding_topk=100,
        rerank_topk=20  # Cross-encoder gives us 20
    )
    
    print('\n' + '='*60)
    print('Running LLM selection on cross-encoder results...')
    
    selected = selector.generate_relevant_paper_selector_response(user_idea, papers, final_count=5)
    
    print('\n' + '='*60)
    print('FINAL SELECTED PAPERS:')
    print(selected)
