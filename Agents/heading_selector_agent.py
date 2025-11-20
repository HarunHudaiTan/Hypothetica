import json

from Agents.Agent import Agent
from heading_extraction.heading_extractor import HeadingExtractor


class HeadingSelectorAgent(Agent):
    def __init__(self):
        super().__init__(system_prompt='''
## Role
You are a specialized Heading Selector Agent designed to identify the most relevant section headings within academic papers based on users idea title, abstract, and headings list.
The main purpose of you is to find the relevant headings of the paper that can match users idea. This way you will fasten the literature review process 

## Input
You will receive:
1. **User's Research Idea**: A description of the research concept, question, or hypothesis the user is exploring
2. **Paper Title**: The title of the academic paper
3. **Paper Abstract**: A summary of the paper's objectives, methods, and findings
4. **List of Headings**: All section headings from the paper (without the actual content)

## Task
Based solely on the project idea title, abstract, and heading names, identify exactly **3 heading intervals** most likely to contain content that would help determine if the user's research idea is present in or related to this paper.

### Interval Definition
- An interval is defined by two headings: `from_heading` (inclusive) and `to_heading` (exclusive)
- The interval includes all content from `from_heading` up to (but NOT including) `to_heading`
- Example: `"from_heading": "METHODOLOGY", "to_heading": "RESULTS"` includes the METHODOLOGY section but stops before RESULTS

### Heading Format Rules
- **Strip all numbering prefixes** from headings (e.g., "4.", "III.", "A.", etc.)
- Return only the heading text in uppercase
- Example: "4. EXPERIMENTS" → return as "EXPERIMENTS"
- Example: "III. OUR PROPOSED APPROACH" → return as "OUR PROPOSED APPROACH"

## What to avoid
- Avoid selecting INTRODUCTION and CONCLUSION in from heading instance 

### Consecutive Section Handling

**When to Combine Sections:**
- If multiple sections that appear **one after another** in the paper are all relevant to the user's idea, combine them into a **single interval** instead of creating separate intervals for each section
- This is more efficient and captures continuous content as one cohesive block

**How to Combine Consecutive Sections:**
1. Identify the **first relevant section** - this becomes your `from_heading`
2. Identify the **last relevant section** in the consecutive sequence
3. Find the section that comes **immediately after** the last relevant section - this becomes your `to_heading`
4. The interval will include everything from `from_heading` up to (but not including) `to_heading`

**Visual Example:**

Given this paper structure:
I. INTRODUCTION
II. RELATED WORK
III. OUR PROPOSED APPROACH
IV. EXPERIMENTS
V. RESULTS ANALYSIS
VI. CONCLUSION
**Scenario 1:** You want sections III, IV, and V (three consecutive sections)
```json
{
  "from_heading": "OUR PROPOSED APPROACH",
  "to_heading": "CONCLUSION"
}
Scenario 2: You want sections II and III (two consecutive sections)
{
  "from_heading": "RELATED WORK",
  "to_heading": "EXPERIMENTS"
}
Scenario 3: You want sections II and V (non-consecutive sections)
[
  {
    "from_heading": "RELATED WORK",
    "to_heading": "OUR PROPOSED APPROACH"
  },
  {
    "from_heading": "RESULTS ANALYSIS",
    "to_heading": "CONCLUSION"
  }
]

        ''',top_p=0.85,top_k=40,temperature=0.1,response_mime_type='application/json',create_chat=False)

    def generate_heading_selector_agent_response(self,users_idea,headings,title_and_abstract):
        response=self.generate_text_generation_response('users idea:'+f'{users_idea}'+'title and abstract'+f'{title_and_abstract}'+'headings'+f'{headings}')
        return json.loads(response.text)
    



#
#
# if __name__ == '__main__':
#      extractor = HeadingExtractor()
#      source = "https://arxiv.org/pdf/2311.07582v1"
#      markdown = extractor.convert_to_markdown(source)
#      headings = extractor.extract_headings(markdown)
#      headings_json = extractor.get_headings_json(headings)
#      print(headings_json)
#      abstract="""
# Recent advances in Large Language Models (LLMs) have presented new opportunities for
# integrating Artificial General Intelligence (AGI) into biological research and education. This study
# evaluated the capabilities of leading LLMs, including GPT-4, GPT-3.5, PaLM2, Claude2, and
# SenseNova, in answering conceptual biology questions. The models were tested on a 108-
# question multiple-choice exam covering biology topics in molecular biology, biological techniques,
# metabolic engineering, and synthetic biology. Among the models, GPT-4 achieved the highest
# average score of 90 and demonstrated the greatest consistency across trials with different
# prompts. The results indicated GPT-4's proficiency in logical reasoning and its potential to aid
# biology research through capabilities like data analysis, hypothesis generation, and knowledge
# integration. However, further development and validation are still required before the promise of
# LLMs in accelerating biological discovery can be realized
#
#
#      """
#      project_idea="""
# Large language models could be adapted to predict protein-protein interaction networks by treating amino acid sequences as a specialized language with its own grammar and semantics. While current approaches often rely on computationally expensive molecular dynamics simulations or limited experimental data, an LLM fine-tuned on millions of known protein interactions, evolutionary conservation patterns, and structural motifs could learn the implicit rules governing how proteins recognize and bind to each other. The model could be trained on a multi-modal dataset combining sequence data, 3D structural information encoded as distance matrices or graph representations, and existing interaction databases, with the goal of not just predicting binary interactions but also estimating binding affinities, identifying specific binding sites, and suggesting mutations that could modulate interaction strength. This approach could accelerate drug discovery by rapidly screening potential therapeutic targets, help understand disease mechanisms driven by aberrant protein interactions, and reveal previously unknown signaling pathways—all while potentially uncovering the underlying "syntax" of molecular recognition that has been refined through billions of years of evolution.
#      """
#      agent=HeadingSelectorAgent()
#      heading_interval_list= agent.generate_heading_selector_agent_response(project_idea, headings_json, abstract)
#      print(heading_interval_list)
#
# #      extracted_heading_text=[]
# #      for interval in heading_interval_list:
# #          text=extractor.get_text_between_headings(markdown,interval['from_heading'],interval['to_heading'])
# #          extracted_heading_text.append(text)
# #
# #      print(extracted_heading_text)
# #
# #      # Save extracted text to file
# #      extractor.save_extracted_text_to_file(extracted_heading_text)
# #
# #      # heading_interval_list=json.loads(agent_response)
# #      # print(heading_interval_list)
# #      # print(type(heading_interval_list))
# #
# # #
# # #
# # #
