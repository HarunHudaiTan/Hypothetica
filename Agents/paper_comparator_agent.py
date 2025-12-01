from Agents.Agent import Agent


class PaperComparatorAgent(Agent):
    def __init__(self):
        super().__init__(
            system_prompt="""

            
    """, temperature=0.3, top_p=0.85, top_k=40, response_mime_type='application/json', create_chat=False)