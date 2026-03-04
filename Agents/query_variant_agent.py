import json
from Agents.Agent import Agent


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
    {"type": "synonyms", "query": "RAG OR retrieval-augmented OR multimodal QA OR medical VQA OR clinical NLP OR radiology AI"}
  ]
}
""",
            temperature=0.4,
            top_p=0.85,
            top_k=40,
            response_mime_type='application/json',
            create_chat=False
        )

    def generate_query_variants(self, user_idea: str) -> list:
        """
        Generate search query variants from user's research idea.
        
        Args:
            user_idea: The user's research idea/description
            
        Returns:
            List of query variant dictionaries with 'type' and 'query' keys
        """
        response = self.generate_text_generation_response(user_idea)
        response_json = json.loads(response.text)
        return response_json['variants']


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
