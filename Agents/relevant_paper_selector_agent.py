import json

from langchain_experimental.graph_transformers.llm import system_prompt

from Agents.Agent import Agent
from embeddemo.embed_query_wrapper import QueryWrapper


class RelevantPaperSelectorAgent(Agent):
    def __init__(self):
        super().__init__(
            system_prompt="""
You are a research paper relevance selector. Your task is to identify the 5 most relevant papers from a set of 15 candidates based on the user's research idea.

## Selection Criteria
- Prioritize papers that share core concepts, methodologies, or theoretical frameworks with the user's idea
- Focus on conceptual alignment over superficial keyword matches
- Consider papers that address similar problems or use comparable approaches

## Input
- User's project/research idea (JSON format)
- 15 arXiv papers with titles, abstracts, and URLs

## Output
Return exactly 5 papers that best align with the user's research direction, ranked by relevance.
            """,
            top_p=0.7,
            top_k=30,
            temperature=0.2,
            response_mime_type="application/json"
        )


    def generate_relevant_paper_selector_response(self, user_idea,papers):
        response=self.generate_text_generation_response("Users idea"+f"{user_idea}"+"papers"+f"{papers}")
        parsed_response = json.loads(response.text)
        return json.dumps(parsed_response, indent=2)

# if __name__ == '__main__':
#     relevant_paper_selector = RelevantPaperSelectorAgent()
#     wrapper=QueryWrapper()
#     user_idea="""I'm exploring the theoretical foundations of few-shot learning - specifically, what are
# the fundamental limits on how few examples are needed to learn a new task? I want to
# derive sample complexity bounds that depend on task similarity, model capacity, and the
# structure of the meta-learning algorithm. This could help explain why certain meta-learning
# architectures (like MAML or Prototypical Networks) work better than others and guide the """
#     papers=wrapper.search_literature(user_idea, include_scores=False)
#     print('-----------------------')
#     print(papers)

#     selected_papers_from_agent=relevant_paper_selector.generate_relevant_paper_selector_response(user_idea,papers)
#     print(selected_papers_from_agent)