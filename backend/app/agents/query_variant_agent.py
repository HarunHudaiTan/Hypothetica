import json
from app.agents.Agent import Agent


class QueryVariantAgent(Agent):
    """
    Generates multiple search query variants from user input for high-recall paper retrieval.
    Replaces the old KeywordAgent with a more comprehensive approach.
    """
    
    def __init__(self):
        super().__init__(
            system_prompt="""You are an Academic Search Query Generator for arXiv literature retrieval.

Your task is to generate 4-5 search query variants from a user's research idea to maximize paper discovery (high recall).

## Query Variant Types

1. **RAW** (1 query): The user's core idea condensed into a searchable phrase (5-15 words)
   - Extract the main research concept directly
   - Keep it natural, not keyword-stuffed

2. **ACADEMIC** (2 queries): Formal academic reformulations
   - Translate into standard academic terminology used in paper titles/abstracts
   - Use established research field vocabulary
   - One focused on methodology, one on application domain

3. **SYNONYMS** (1 query): OR-based query with abbreviations and alternative terms
   - Include common abbreviations (e.g., "RAG" for retrieval-augmented generation)
   - Include synonyms and related terms
   - Format: "term1 OR term2 OR term3 OR term4"
   - **Maximum 4 OR terms** — keep it short to avoid timeouts

## Output Format
Return ONLY valid JSON:

{
  "variants": [
    {"type": "raw", "query": "<condensed user idea>"},
    {"type": "academic", "query": "<formal methodology-focused query>"},
    {"type": "academic", "query": "<formal domain-focused query>"},
    {"type": "synonyms", "query": "<term1 OR term2 OR abbreviation OR synonym>"}
  ]
}

## Guidelines
- Each query should be 3-20 words (except synonym queries which can be longer)
- Use terminology that appears in arXiv paper titles and abstracts
- Avoid overly specific phrases that won't match anything
- Avoid overly broad terms that match everything
- The queries should complement each other, not duplicate

## Example

User Input: "I want to build a system that uses LLMs to answer questions from medical documents with images like X-rays"

Output:
{
  "variants": [
    {"type": "raw", "query": "LLM question answering medical documents with images"},
    {"type": "academic", "query": "retrieval augmented generation multimodal medical question answering"},
    {"type": "academic", "query": "vision language models clinical document understanding radiology"},
    {"type": "synonyms", "query": "RAG OR retrieval-augmented OR medical VQA OR clinical NLP"}
  ]
}
""",
            temperature=0.4,
            top_p=0.85,
            top_k=40,
            response_mime_type='application/json',
            create_chat=False
        )

    def generate_query_variants(self, user_idea: str, adapter_name: str = "arxiv") -> list:
        """
        Generate search query variants from user's research idea.

        Args:
            user_idea: The user's research idea/description
            adapter_name: The adapter being used ("arxiv", "github", "google_patents", "openalex")

        Returns:
            List of query variant dictionaries with 'type' and 'query' keys
        """
        if adapter_name == "github":
            return self._generate_github_queries(user_idea)
        if adapter_name == "openalex":
            return self._generate_openalex_queries(user_idea)

        # Default academic query generation for arXiv and patents
        response = self.generate_text_generation_response(user_idea)
        result = json.loads(response.text)
        return result.get('variants', [])
    
    def _generate_openalex_queries(self, user_idea: str) -> list:
        """
        Generate 5 OpenAlex-optimised search variants.
        OpenAlex covers 250M+ works across all disciplines and supports full-text
        search over titles + abstracts, so variants should span vocabulary, synonyms,
        and conceptual framing rather than just keyword permutations.
        """
        import logging
        logger = logging.getLogger(__name__)

        prompt = f"""You are an OpenAlex Academic Search Query Generator.

OpenAlex indexes 250M+ scholarly works (journals, conferences, preprints, datasets) across all disciplines.
Queries run as relevance-ranked full-text search over titles and abstracts.

Your task: generate exactly 5 query variants that maximise recall of relevant papers while staying precise enough to avoid noise.

## Variant types to produce

1. **raw** — Core research concept in plain language (5–15 words). What the idea is fundamentally about.

2. **academic_method** — Formal methodology query: algorithms, architectures, model families, or frameworks used.
   Use vocabulary found in paper titles and abstract first sentences.

3. **academic_domain** — Application domain / problem framing: what field this applies to, what real-world problem it solves.

4. **synonyms** — OR-based query with acronyms and alternative terms.
   Format: "term1 OR term2 OR abbreviation OR synonym". Maximum 5 OR terms.

5. **concept** — Established research sub-field name, benchmark dataset name, or survey-level framing.
   Catches papers that study the same phenomenon from a different angle.

## Output format — return ONLY valid JSON
{{
  "variants": [
    {{"type": "raw",             "query": "..."}},
    {{"type": "academic_method", "query": "..."}},
    {{"type": "academic_domain", "query": "..."}},
    {{"type": "synonyms",        "query": "..."}},
    {{"type": "concept",         "query": "..."}}
  ]
}}

## Rules
- Each query: 3–20 words (synonym queries may be slightly longer)
- Queries must complement each other — no near-duplicates
- Use terminology that appears in published paper titles and abstracts
- Do NOT quote phrases unless essential

## Example
Idea: "Graph neural networks to predict drug-drug interactions from patient EHR co-prescription data"

{{
  "variants": [
    {{"type": "raw",             "query": "graph neural network drug-drug interaction prediction electronic health records"}},
    {{"type": "academic_method", "query": "graph convolutional network link prediction biomedical knowledge graph"}},
    {{"type": "academic_domain", "query": "polypharmacy side effect prediction clinical co-prescription data"}},
    {{"type": "synonyms",        "query": "DDI OR polypharmacy OR drug interaction OR GNN OR EHR"}},
    {{"type": "concept",         "query": "drug safety pharmacovigilance interaction network embedding"}}
  ]
}}

## User's research idea
{user_idea}

Generate exactly 5 variants."""

        response = self.generate_text_generation_response(prompt)
        result = json.loads(response.text)
        variants = result.get('variants', [])

        logger.info(f"[OpenAlex Query Generation] {len(variants)} variants generated")
        for v in variants:
            logger.info(f"  [{v.get('type')}] {v.get('query')}")

        return variants

    def _generate_github_queries(self, user_idea: str) -> list:
        """
        Generate GitHub-optimized search queries based on 2026 API best practices.
        Uses implementation-focused language and proper GitHub search syntax.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        github_prompt = f"""You are a GitHub Repository Search Query Generator.

Your task: Translate a research idea into 4 GitHub search queries that find repositories implementing the SPECIFIC combination of techniques described — not just any popular repo in the same broad area.

## Step 1 — Identify the RARE terms
Look at the idea and extract the 3-4 most SPECIFIC, RARE terms that together uniquely identify it.
- RARE terms: technical operations, algorithms, data structures, file types, domain-specific nouns
  Examples: "ast", "dependency graph", "refactor plan", "rollback", "patch suggestion", "call graph", "symbol rename"
- COMMON terms to AVOID as standalone searches: "agent", "llm", "code", "ai", "tool", "model", "github"
  These match thousands of unrelated repos.

## Step 2 — Build queries around the rare terms
Every query MUST contain at least 2 rare/specific terms. Framework names (langchain, llama_index, etc.) can appear but must be paired with rare terms.

## Framework Name Rules
- Only use official names: "langchain", "langgraph", "llama_index", "haystack", "autogen", "crewai"
- These get converted to GitHub topic filters by the backend
- All other terms are searched as text in repo name/description/readme

## GitHub Search Syntax
- Spaces = AND (all terms must appear)
- Do NOT add qualifiers — the backend adds: `in:name,description,readme archived:false fork:false stars:>N`
- Do NOT use OR operators

## Output Format
Return ONLY valid JSON:
{{
  "variants": [
    {{"type": "framework_rare", "query": "<framework name + 2-3 rare specific terms>"}},
    {{"type": "rare_only", "query": "<3-4 rare terms, no framework name>"}},
    {{"type": "rare_alternate", "query": "<2-3 different rare terms from the idea>"}},
    {{"type": "framework_broader", "query": "<framework name + 1-2 rare terms>"}}
  ]
}}

## Worked Example

**Idea**: "LangChain agent that browses a GitHub repository, tracks function/class dependencies via AST, and generates a step-by-step refactor plan with patch suggestions and rollback guidance"

Rare terms identified: "ast", "dependency graph", "refactor plan", "call graph", "patch", "rollback", "symbol", "codebase traversal"
Common terms to avoid alone: "agent", "llm", "code", "github"

Good queries:
{{
  "variants": [
    {{"type": "framework_rare", "query": "langchain ast dependency refactor plan"}},
    {{"type": "rare_only", "query": "ast call graph refactor patch rollback"}},
    {{"type": "rare_alternate", "query": "codebase dependency graph refactor planning llm"}},
    {{"type": "framework_broader", "query": "langchain codebase refactor dependency"}}
  ]
}}

Bad queries (too generic — DO NOT generate these):
- "langchain code agent" → matches every LangChain tool ever made
- "code refactor llm" → matches every AI coding assistant
- "github agent" → matches every repo that mentions GitHub

## User's Research Idea
{user_idea}

First identify the rare/specific terms, then generate 4 queries. Each query must be specific enough to exclude generic AI coding tools."""
        
        response = self.generate_text_generation_response(github_prompt)
        result = json.loads(response.text)
        variants = result.get('variants', [])
        
        logger.info(f"[GitHub Query Generation] Generated {len(variants)} queries:")
        for v in variants:
            logger.info(f"  - [{v.get('type', 'unknown')}] {v.get('query', '')}")
        
        return variants


# Testing
if __name__ == '__main__':
    agent = QueryVariantAgent()
    test_idea = '''I want to develop a multimodal retrieval-augmented generation (RAG) system 
    that can process and reason over both text documents and images simultaneously. 
    The idea is to use vision-language models to extract semantic information from diagrams, 
    charts, and figures in scientific papers, and then integrate this visual understanding 
    with the textual content for more comprehensive question-answering.'''
    
    variants = agent.generate_query_variants(test_idea)
    print(json.dumps(variants, indent=2))
