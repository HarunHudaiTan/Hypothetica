import json

from Agents.Agent import Agent


class KeywordAgent(Agent):
    def __init__(self):
        super().__init__(
system_prompt="""You are an Academic Search Specialist for arXiv literature reviews.

Analyze the research idea and identify 7 search topics: 4 SPECIFIC terms followed by 3 BROAD terms to enable comprehensive literature discovery.

SEARCH STRATEGY:
- Keywords 1-4: SPECIFIC TERMS - Precise methods, techniques, and domain-specific concepts
  * Exact algorithms, specific approaches, technical terminology
  * Domain + technique combinations
  * Concrete problem formulations
- Keywords 5-7: BROAD TERMS - General fields and overarching areas
  * General research fields
  * Broader problem categories
  * Established research domains

CRITICAL: Do not just extract phrases from the user's text. Instead, translate their idea into standard academic terminology that researchers use in paper titles and abstracts.

Output Requirements:
Return ONLY valid JSON with no additional text:

{
  "keywords": [
    "specific term 1",
    "specific term 2",
    "specific term 3",
    "specific term 4",
    "broad term 1",
    "broad term 2",
    "broad term 3"
  ]
}

Guidelines:
- Use ESTABLISHED academic terminology, not buzzwords or new slang
- Think: "What would researchers call this in paper titles?"
- First 4 keywords: Target the user's SPECIFIC research direction
- Last 3 keywords: Capture the GENERAL research areas involved
- Prefer well-known terms over hyper-specific phrases
- Each topic should be 1-4 words maximum
- Avoid repetition across all 7 keywords
- Prioritize terms that will find the most relevant papers

Examples of Translation:
- User says "RAG system" → Specific: "retrieval augmented generation", "document retrieval" → Broad: "information retrieval", "question answering"
- User says "cross-modal attention" → Specific: "cross modal attention", "multimodal fusion" → Broad: "multimodal learning", "deep learning"
- User says "medical imaging" → Specific: "medical image analysis", "radiology AI" → Broad: "computer vision", "medical AI"

Example Output for Multimodal Medical Retrieval:
{
  "keywords": [
    "medical visual question answering",
    "vision language models",
    "multimodal medical retrieval",
    "cross modal fusion",
    "multimodal learning",
    "computer vision",
    "information retrieval"
  ]
}
""",temperature=0.3,top_p=0.85,top_k=40,response_mime_type='application/json',create_chat=False)


    def generate_keyword_agent_response(self, prompt):
        response=self.generate_text_generation_response(prompt)
        # print(response.text)
        response_json=json.loads(response.text)
        keyword_list=response_json['keywords']
        return keyword_list


# if __name__ == '__main__':
#     keyworda=KeywordAgent()
#     response=keyworda.generate_keyword_agent_response('''I want to develop a multimodal retrieval-augmented generation (RAG) system that can process and reason over both text documents and images simultaneously. The idea is to use vision-language models to extract semantic information from diagrams, charts, and figures in scientific papers, and then integrate this visual understanding with the textual content for more comprehensive question-answering. I'm particularly interested in applying this to medical literature where visual data like X-rays, MRI scans, and anatomical diagrams are crucial for understanding. The system should be able to answer complex queries that require correlating information from both text passages and medical images, potentially using cross-modal attention mechanisms. I'm also curious about efficient indexing strategies for this kind of multimodal data and how to handle cases where the visual and textual information might be contradictory.''')
#     print(response)