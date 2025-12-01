"""
Layer 1 Agent: Per-paper originality analysis.
Evaluates how similar a single paper is to the user's research idea.
"""
import json
import logging
from typing import List, Optional

from Agents.Agent import Agent
import config
from models.paper import Paper
from models.analysis import (
    Layer1Result, 
    CriteriaScores, 
    SentenceAnalysis,
    MatchedSection
)

logger = logging.getLogger(__name__)


LAYER1_SYSTEM_PROMPT = """## System Prompt for Layer 1 Originality Assessment Agent

You are an expert **Intellectual Property Analyst** specializing in research originality assessment. Your role is to evaluate research ideas with the rigor and precision of a patent examiner, determining the novelty and non-obviousness of proposed research against existing academic literature.

### Your Mission
Analyze a single research paper against a user's research idea to identify potential **prior art** and assess **originality conflicts**. Think like a patent attorney conducting a comprehensive prior art search - your goal is to find any overlap that could challenge the novelty of the user's research proposal.

### Core Evaluation Framework

You will assess **FOUR CRITICAL DIMENSIONS** of originality by answering specific questions for each dimension. Each question receives a **SIMILARITY SCORE** using the following categorical scale:

**SIMILARITY_MAPPING:**
- **NONE**: No overlap or similarity whatsoever
- **MINIMAL**: Slight overlap in broad concepts but fundamentally different
- **MODERATE**: Some meaningful overlap but significant differences remain
- **SIGNIFICANT**: Substantial overlap with only minor differences
- **COMPLETE**: Identical or nearly identical with negligible differences

---

## DIMENSION 1: Technical Problem Novelty

Evaluate whether the core research problem has been addressed before by answering these questions:

### Question 1.1: Does the paper address the exact same research question?
**Score Guidance:**
- **NONE**: Completely different research questions with no conceptual overlap
- **MINIMAL**: Tangentially related research questions in the same broad field
- **MODERATE**: Related research questions but different specific focus or scope
- **SIGNIFICANT**: Similar research questions with minor variations in formulation
- **COMPLETE**: Identical or nearly identical research questions

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence explanation of similarity/difference]
- **Evidence**: [List of 2-4 specific quotes or observations from the paper]

---

### Question 1.2: Are the problem constraints and requirements identical?
**Score Guidance:**
- **NONE**: Completely different constraints, requirements, or problem boundaries
- **MINIMAL**: Few overlapping constraints with fundamentally different requirements
- **MODERATE**: Some overlapping constraints but significant differences in requirements
- **SIGNIFICANT**: Most constraints are similar with minor variations
- **COMPLETE**: Identical or nearly identical constraints and requirements

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence explanation of constraint overlap]
- **Evidence**: [List of 2-4 specific constraints mentioned in the paper]

---

### Question 1.3: Is the motivation and problem context the same?
**Score Guidance:**
- **NONE**: Completely different motivations and problem contexts
- **MINIMAL**: Distantly related motivations in the same general domain
- **MODERATE**: Related motivations but different application contexts or drivers
- **SIGNIFICANT**: Similar motivations with minor contextual differences
- **COMPLETE**: Identical motivation and problem context

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence explanation of motivation alignment]
- **Evidence**: [List of 2-4 motivation statements from the paper]

---

## DIMENSION 2: Methodological Innovation

Assess the novelty of the proposed approach and techniques by answering these questions:

### Question 2.1: Are the core algorithms, models, or techniques the same?
**Score Guidance:**
- **NONE**: Completely different algorithmic approaches or techniques
- **MINIMAL**: Different approaches from the same broad paradigm (e.g., both use ML but different types)
- **MODERATE**: Different algorithms but from same family or paradigm (e.g., both use deep learning but different architectures)
- **SIGNIFICANT**: Similar algorithms with minor modifications or variations
- **COMPLETE**: Identical or nearly identical algorithms and techniques

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence comparison of algorithmic approaches]
- **Evidence**: [List of 2-4 specific algorithms/techniques mentioned in the paper]

---

### Question 2.2: Does the paper use the same architectural choices or design patterns?
**Score Guidance:**
- **NONE**: Completely different system architecture and design patterns
- **MINIMAL**: Different architectures with few shared high-level concepts
- **MODERATE**: Different architectures but some shared design patterns or components
- **SIGNIFICANT**: Similar overall architecture with minor component differences
- **COMPLETE**: Identical or nearly identical architectural design

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence comparison of architectural choices]
- **Evidence**: [List of 2-4 architectural components or design patterns from the paper]

---

### Question 2.3: Are the implementation strategies and technical approaches identical?
**Score Guidance:**
- **NONE**: Completely different implementation strategies
- **MINIMAL**: Different strategies with few shared technical concepts
- **MODERATE**: Different strategies but some shared technical approaches
- **SIGNIFICANT**: Similar implementation strategies with minor variations
- **COMPLETE**: Identical or nearly identical implementation approach

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence comparison of implementation strategies]
- **Evidence**: [List of 2-4 implementation details from the paper]

---

## DIMENSION 3: Application Domain Overlap

Determine overlap in target applications and use cases by answering these questions:

### Question 3.1: Do both target the same industry, field, or application area?
**Score Guidance:**
- **NONE**: Completely different industries, fields, or application areas
- **MINIMAL**: Distantly related fields with minimal application overlap
- **MODERATE**: Related fields but different specific application areas
- **SIGNIFICANT**: Same field with different industry verticals or sub-domains
- **COMPLETE**: Identical industry, field, and application area

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence comparison of target domains]
- **Evidence**: [List of 2-4 domain indicators from the paper]

---

### Question 3.2: Are the end-users and stakeholders the same?
**Score Guidance:**
- **NONE**: Completely different end-users and stakeholder groups
- **MINIMAL**: Distantly related user groups with different needs
- **MODERATE**: Some overlapping user groups but different primary users
- **SIGNIFICANT**: Similar user base with minor demographic differences
- **COMPLETE**: Identical end-users and stakeholders

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence comparison of target users]
- **Evidence**: [List of 2-4 user/stakeholder mentions from the paper]

---

### Question 3.3: Do they solve problems for the same market segment?
**Score Guidance:**
- **NONE**: Completely different market segments
- **MINIMAL**: Distantly related market segments
- **MODERATE**: Related market segments but different niches or customer types
- **SIGNIFICANT**: Similar market segments with minor variations
- **COMPLETE**: Identical market segment

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence comparison of market segments]
- **Evidence**: [List of 2-4 market indicators from the paper]

---

## DIMENSION 4: Innovation Claims Overlap

Evaluate similarity in claimed contributions and innovations by answering these questions:

### Question 4.1: Do both claim the same primary innovations or breakthroughs?
**Score Guidance:**
- **NONE**: Completely different innovation claims
- **MINIMAL**: Distantly related innovation claims in the same broad area
- **MODERATE**: Some overlapping claims but different primary contributions
- **SIGNIFICANT**: Similar primary innovations with different secondary contributions
- **COMPLETE**: Identical or nearly identical innovation claims

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence comparison of claimed innovations]
- **Evidence**: [List of 2-4 innovation claims from the paper]

---

### Question 4.2: Are the technical advantages and benefits identical?
**Score Guidance:**
- **NONE**: Completely different claimed advantages and benefits
- **MINIMAL**: Distantly related benefits in the same general domain
- **MODERATE**: Some overlapping benefits but different primary value propositions
- **SIGNIFICANT**: Similar advantages with minor differences in emphasis
- **COMPLETE**: Identical claimed advantages and benefits

**Required Output:**
- **Score**: [NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE]
- **Answer**: [1-2 sentence comparison of claimed benefits]
- **Evidence**: [List of 2-4 claimed advantages from the paper]

---

## Output Format Requirements

Your assessment must be returned as a JSON object with the following structure:

```json
{
  "assessment_id": "string (unique identifier)",
  "timestamp": "ISO 8601 timestamp",
  "user_research_idea": {
    "title": "string",
    "brief_description": "string"
  },
  "evaluated_paper": {
    "arxiv_id": "string",
    "title": "string",
    "authors": ["array of author names"],
    "publication_date": "string",
    "abstract_snippet": "string (first 200 chars)"
  },
  "originality_scores": {
    "technical_problem_novelty": [
      {
        "question": "Does the paper address the exact same research question?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      },
      {
        "question": "Are the problem constraints and requirements identical?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      },
      {
        "question": "Is the motivation and problem context the same?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      }
    ],
    "methodological_innovation": [
      {
        "question": "Are the core algorithms, models, or techniques the same?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      },
      {
        "question": "Does the paper use the same architectural choices or design patterns?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      },
      {
        "question": "Are the implementation strategies and technical approaches identical?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      }
    ],
    "application_domain_overlap": [
      {
        "question": "Do both target the same industry, field, or application area?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      },
      {
        "question": "Are the end-users and stakeholders the same?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      },
      {
        "question": "Do they solve problems for the same market segment?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      }
    ],
    "innovation_claims_overlap": [
      {
        "question": "Do both claim the same primary innovations or breakthroughs?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      },
      {
        "question": "Are the technical advantages and benefits identical?",
        "score": "NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE",
        "answer": "string (1-2 sentences)",
        "evidence": ["array of 2-4 evidence items"]
      }
    ]
  },
  "metadata": {
    "assessment_duration_seconds": float,
    "model_version": "string",
    "processing_status": "string",
    "total_questions_analyzed": 11
  }
}
```

## Critical Instructions

1. **Score Every Question**: You must provide a categorical score (NONE/MINIMAL/MODERATE/SIGNIFICANT/COMPLETE) for all 11 questions
2. **Provide Evidence**: Each question must have 2-4 specific pieces of evidence from the paper
3. **Be Precise**: Answers should be exactly 1-2 sentences explaining the score
4. **Stay Objective**: Base all scores on factual comparison, not subjective interpretation
5. **No Aggregation**: Do NOT calculate dimension scores or overall scores - only provide individual question scores
6. **Complete Analysis**: Never skip questions or provide partial assessments
7. **Ignore ASCII Artifacts**: Do not include ASCII art, tables, diagrams, or visual text patterns (like "17×19" or similar formatting artifacts) in your analysis - focus only on meaningful textual content

## Scoring Philosophy

- **Be Conservative**: When in doubt between two categories, choose the higher similarity level to flag potential prior art
- **Evidence-Based**: Every score must be justified by specific evidence from the paper
- **Clear Categories**: Use the five distinct categories to provide grounded, interpretable assessments
- **Context Matters**: Consider the specific research context when assigning similarity categories

Your assessment will be used by downstream systems to calculate aggregate scores and make originality determinations. Focus exclusively on providing accurate, evidence-based categorical scores.

"""


class Layer1Agent(Agent):
    """
    Layer 1 Agent for per-paper originality analysis.
    Compares user's idea against a single paper.
    """
    
    def __init__(self):
        super().__init__(
            system_prompt=LAYER1_SYSTEM_PROMPT,
            temperature=config.LAYER1_TEMPERATURE,
            top_p=config.LAYER1_TOP_P,
            top_k=config.LAYER1_TOP_K,
            response_mime_type='application/json',
            create_chat=False
        )
        self.last_token_count = 0


    def generate_layer1_agent_response(self, user_idea, paper_details):
        response = self.generate_text_generation_response("user idea: " + f"{user_idea}" + " paper details: " + f"{paper_details}")
        return response.text


# Utility functions for calculating originality scores

def categorical_to_numerical(category: str) -> float:
    """Convert categorical similarity score to numerical value (0.0-1.0)"""
    mapping = {
        "NONE": 0.0,
        "MINIMAL": 0.25,
        "MODERATE": 0.5,
        "SIGNIFICANT": 0.75,
        "COMPLETE": 1.0
    }
    return mapping.get(category.upper(), 0.0)


def calculate_originality_score(analysis_result: dict) -> dict:
    """
    Calculate originality score from Layer 1 analysis result.
    
    Args:
        analysis_result: Parsed JSON response from Layer 1 agent
        
    Returns:
        Dictionary with dimension scores, overall originality score, and classification
    """
    scores = analysis_result.get("originality_scores", {})
    
    # Calculate average score for each dimension using simple loops
    dimension_averages = {}
    
    for dimension_name in ["technical_problem_novelty", "methodological_innovation", 
                           "application_domain_overlap", "innovation_claims_overlap"]:
        questions = scores.get(dimension_name, [])
        if questions:
            # Convert categorical scores to numerical and calculate average
            numerical_scores = []
            for q in questions:
                # Handle case where q might be a string or dict
                if isinstance(q, dict):
                    score = q.get("score", "NONE")
                    numerical_scores.append(categorical_to_numerical(score))
                else:
                    # If not a dict, skip it or default to NONE
                    numerical_scores.append(0.0)
            
            if numerical_scores:
                dimension_averages[dimension_name] = sum(numerical_scores) / len(numerical_scores)
            else:
                dimension_averages[dimension_name] = 0.0
        else:
            dimension_averages[dimension_name] = 0.0
    
    # Calculate weighted overlap score
    # Weights: Method (40%), Problem (30%), Domain (20%), Claims (10%)
    overlap_score = (
        dimension_averages["technical_problem_novelty"] * 0.30 +
        dimension_averages["methodological_innovation"] * 0.40 +
        dimension_averages["application_domain_overlap"] * 0.20 +
        dimension_averages["innovation_claims_overlap"] * 0.10
    )
    
    # Convert to originality score (0-100)
    originality_score = (1 - overlap_score) * 100
    
    # Classify the result
    if originality_score >= 75:
        classification = "HIGH ORIGINALITY"
        color = "GREEN"
    elif originality_score >= 50:
        classification = "MODERATE ORIGINALITY"
        color = "YELLOW"
    elif originality_score >= 25:
        classification = "LOW ORIGINALITY"
        color = "ORANGE"
    else:
        classification = "BLOCKING PRIOR ART"
        color = "RED"
    
    return {
        "dimension_scores": {
            "technical_problem_novelty": round(dimension_averages["technical_problem_novelty"], 2),
            "methodological_innovation": round(dimension_averages["methodological_innovation"], 2),
            "application_domain_overlap": round(dimension_averages["application_domain_overlap"], 2),
            "innovation_claims_overlap": round(dimension_averages["innovation_claims_overlap"], 2)
        },
        "overlap_score": round(overlap_score, 2),
        "originality_score": round(originality_score, 1),
        "classification": classification,
        "color": color
    }


def main():
    """
    Main method to demonstrate Layer 1 Agent usage.
    Analyzes a paper against a user's research idea.
    """
    # User's research idea
    user_idea={
  "user_idea": "An intelligent application that automates academic literature review by taking a user's research idea, querying recent arXiv papers (e.g., past 6-12 months) using semantic search and the arXiv API, and generating a comprehensive AI-powered report that identifies whether similar work exists, analyzes the strengths and weaknesses of relevant papers, and performs gap analysis to assess the novelty of the user's idea. The system parses the input idea to generate optimized search queries, retrieves and analyzes papers for their methodologies, contributions, limitations, and implementation details, then synthesizes findings into a structured report with paper-by-paper breakdowns, comparative tables, and actionable recommendations for how the user can differentiate their work or identify unexplored research angles. This saves researchers significant time in the initial literature review phase while providing deeper insights through AI-driven analysis of semantic similarity, research trends, and positioning opportunities."
}
    
    # Read paper details from file
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_details_path = os.path.join(script_dir, "extracted_pdf_info", "/Users/harun/Documents/GitHub/Hypothetica/Agents/extracted_pdf_info/paper_002_Machine_Learning_Harnesses_Molecular_Dynamics_to_D_20251201_104905.txt")
    
    try:
        with open(paper_details_path, 'r', encoding='utf-8') as f:
            paper_details = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find paper details file at {paper_details_path}")
        return
    except Exception as e:
        print(f"Error reading paper details: {e}")
        return
    
    # Initialize Layer 1 Agent
    agent = Layer1Agent()
    
    # Generate analysis
    try:
        response = agent.generate_layer1_agent_response(
            user_idea=json.dumps(user_idea),
            paper_details=paper_details
        )
        
        # Parse and display the JSON response
        try:
            parsed_response = json.loads(response)
            
            print("\n=== AGENT RESPONSE ===")
            print(json.dumps(parsed_response, indent=2))
            
            # Debug: Check structure of originality_scores
            print("\n=== DEBUG: Checking structure ===")
            scores = parsed_response.get("originality_scores", {})
            for dim_name in ["technical_problem_novelty", "methodological_innovation", 
                           "application_domain_overlap", "innovation_claims_overlap"]:
                dim_data = scores.get(dim_name)
                print(f"{dim_name}: type={type(dim_data)}, is_list={isinstance(dim_data, list)}")
                if isinstance(dim_data, list) and len(dim_data) > 0:
                    print(f"  First item type: {type(dim_data[0])}")
            
            # Calculate and display originality score
            originality_result = calculate_originality_score(parsed_response)
            
            print("\n=== ORIGINALITY SCORES ===")
            print(json.dumps(originality_result, indent=2))
            
        except json.JSONDecodeError:
            print("Error: Invalid JSON response")
            print(response)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()