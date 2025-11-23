import json

from Agents.Agent import Agent


class KeywordAgent(Agent):
    def __init__(self):
        super().__init__(
system_prompt="""You are an Academic Search Specialist for arXiv literature reviews.

Generate 7 SPECIFIC search keywords by combining the user's research domains and methods. Create targeted compound terms that researchers use in their papers.

STRATEGY:
- Combine domain + technique (e.g., "medical AI", "financial forecasting")
- Use field-specific terminology (e.g., "clinical NLP", "satellite imagery")
- Create precise multi-word phrases that appear in paper titles
- Each keyword should be 2-4 words targeting the exact research area
- Avoid single generic words like "AI" or "learning"

Output ONLY valid JSON:

{
  "keywords": [
    "domain technique 1",
    "domain technique 2",
    "domain technique 3",
    "domain technique 4",
    "domain technique 5",
    "domain technique 6",
    "domain technique 7"
  ]
}

Examples:

Input: "AI for medical diagnosis"
Output:
{
  "keywords": [
    "medical AI",
    "clinical diagnosis AI",
    "medical image classification",
    "disease prediction models",
    "healthcare machine learning",
    "diagnostic decision support",
    "biomedical deep learning"
  ]
}

Input: "NLP for financial analysis"
Output:
{
  "keywords": [
    "financial NLP",
    "stock prediction models",
    "sentiment analysis finance",
    "financial text mining",
    "market forecasting AI",
    "trading signal generation",
    "financial document analysis"
  ]
}

Input: "Computer vision for agriculture"
Output:
{
  "keywords": [
    "agricultural computer vision",
    "crop disease detection",
    "precision agriculture AI",
    "plant phenotyping",
    "drone image analysis",
    "agricultural robotics vision",
    "field monitoring systems"
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