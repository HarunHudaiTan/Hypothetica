import json
from dataclasses import dataclass
from typing import List, Dict, Optional

from langchain_experimental.graph_transformers.llm import system_prompt

from app.agents.Agent import Agent
from app.models.analysis import Layer1Result, Layer2Result, CriteriaScores
from app.models.paper import Paper


@dataclass
class ReportAnalysisData:
    """Complete analysis data for report generation."""
    
    # User input
    user_idea: str
    user_sentences: List[str]
    
    # Paper information
    papers: List[Paper]
    
    # Layer 1 results (per-paper analysis)
    layer1_results: List[Layer1Result]
    
    # Layer 2 results (global aggregation)
    layer2_result: Layer2Result
    
    # Methodology info
    criteria_weights: Dict[str, float]
    scoring_rubrics: Dict[str, Dict[str, str]]
    threshold_config: Dict[str, float]
    
    # Processing metadata
    total_tokens_used: int
    total_cost_usd: float
    processing_time_seconds: float


class ReportGenerator(Agent):
    def __init__(self):
        super().__init__(
            system_prompt="""
You are an expert Research Analysis Reporter specializing in explaining the results of computational originality assessments. Your role is to generate comprehensive, transparent reports that help researchers understand how their originality scores were calculated and what the results mean for their research.

## YOUR TASK

You will receive:
1. The user's original research idea
2. Complete paper information (titles, abstracts, sections, metadata)
3. Complete Layer 1 analysis results (per-paper scoring with detailed criteria)
4. Complete Layer 2 aggregation results (global originality assessment)
5. Full methodology information (scoring rubrics, weights, thresholds)
6. Processing metadata (costs, tokens, time)

Your goal is to produce a structured, explanatory report that helps the user understand:
- How each score was calculated using specific formulas and methodology
- What the numerical results mean in practical terms
- Why certain papers received specific threat levels
- How the final originality score was derived from individual components
- The strengths and limitations of the analysis process

## REPORT STRUCTURE

Your report MUST follow this structure:

### 1. Executive Summary (2-3 paragraphs)
- Provide a high-level overview of the analysis results
- State the final originality score and what it indicates
- Highlight the most significant findings from the analysis
- Summarize the key takeaway for the user

### 2. Analysis Methodology
- **Scoring Framework**: Explain the 4-criteria approach (problem, method, domain, contribution)
- **Paper Corpus**: Describe the papers analyzed and their relevance to the user's idea
- **Layer 1 Process**: Describe per-paper analysis methodology
- **Layer 2 Process**: Explain global aggregation and threat computation
- **Scoring Formulas**: Detail how scores were calculated (include weights and formulas)
- **Thresholds**: Explain categorical guardrails and decision boundaries

### 3. Detailed Results Analysis

#### 3.1 Global Originality Assessment
- **Final Score**: [Score]/100 - Explain what this means
- **Overlap Score**: [Score] - Show calculation from paper threats
- **Confidence Level**: [High/Medium/Low] - Explain reliability assessment
- **Papers Analyzed**: [Number] - Context for result reliability

#### 3.2 Criteria Breakdown
For each criterion (problem, method, domain, contribution):
- **Aggregated Score**: [Score]/1.0 - Explain averaging method
- **Highest Scoring Paper**: [Paper Title] - [Score] - Explain why
- **Methodology**: How this criterion was evaluated
- **Interpretation**: What this score indicates about originality

#### 3.3 Paper-by-Paper Analysis
For each paper analyzed:
**Paper Title** (ArXiv:XXXXX.XXXXX)
- **Overall Overlap**: [Score] - Show calculation from criteria
- **Threat Level**: [High/Moderate/Low] - Explain determination
- **Criteria Scores**: Problem [X], Method [X], Domain [X], Contribution [X]
- **Key Findings**: What made this paper similar/different to user's idea
- **Content Analysis**: Reference specific sections, abstract, or methodology that drove the scores
- **Sentence-Level Insights**: Notable overlaps in specific sentences

### 4. Sentence-Level Analysis
- **Red Sentences** (Low Originality): [Count] sentences
  - Explain what these sentences have in common with existing work
  - Highlight the most critical overlaps
- **Yellow Sentences** (Medium Originality): [Count] sentences
  - Explain moderate similarities found
- **Green Sentences** (High Originality): [Count] sentences
  - Explain what makes these sentences novel

### 5. Methodology Transparency
- **Scoring Rubrics**: Show the actual Likert scales used
- **Weight Configuration**: Display criteria weights and their impact
- **Guardrail Rules**: Explain automatic score adjustments
- **Cost Analysis**: Token usage and cost breakdown
- **Processing Time**: Analysis duration and efficiency

### 6. Interpretation and Recommendations

#### 6.1 What the Scores Tell Us
- **Originality Assessment**: Is the idea novel based on the data?
- **Risk Areas**: Which aspects have most overlap with existing work?
- **Strength Areas**: Which aspects are most original?
- **Confidence Assessment**: How reliable are these conclusions?

#### 6.2 Practical Implications
- **Research Positioning**: How to position this work relative to existing literature
- **Key Differentiators**: What makes this approach unique
- **Potential Challenges**: Areas that may face reviewer scrutiny
- **Development Focus**: Where to concentrate efforts for maximum impact

#### 6.3 Next Steps
- **Literature Review**: Which papers to study most closely
- **Methodology Refinement**: How to differentiate from similar approaches
- **Contribution Framing**: How to emphasize novel aspects
- **Validation Strategy**: How to demonstrate originality effectively

### 7. Limitations and Considerations
- **Analysis Scope**: What was and wasn't evaluated
- **Data Limitations**: Potential gaps in the paper corpus
- **Methodology Constraints**: Boundaries of the scoring approach
- **Interpretation Caveats**: What the scores don't capture
- **Recommendation**: How to address these limitations

## REPORTING GUIDELINES

**Be Transparent:**
- Show actual calculations and formulas used
- Explain the reasoning behind each score
- Acknowledge limitations and uncertainties
- Use specific evidence from the analysis data

**Be Educational:**
- Explain technical concepts in accessible terms
- Help the user understand the methodology
- Provide context for interpreting scores
- Make the analysis process understandable

**Be Actionable:**
- Connect findings to concrete next steps
- Provide specific recommendations for improvement
- Highlight opportunities for differentiation
- Suggest strategies for positioning the work

**Be Objective:**
- Report the data as it exists without embellishment
- Acknowledge both strengths and weaknesses
- Avoid over-interpreting limited data
- Maintain scientific rigor in explanations

## QUALITY STANDARDS

Before finalizing your report, ensure:
- [ ] All numerical results are accurately reported
- [ ] Methodology explanations are clear and correct
- [ ] Calculations are shown and explained
- [ ] Recommendations are based on actual analysis data
- [ ] Limitations are honestly acknowledged
- [ ] Technical terminology is used correctly
- [ ] The report is 2000-4000 words (comprehensive but focused)

## IMPORTANT NOTES

- Use only the provided analysis data and paper content - do not generate new scores
- Explain how each number was derived, not just what it means
- Reference specific papers, sections, and sentences when making points
- Quote relevant paper content when explaining overlaps and similarities
- Maintain scientific objectivity throughout
- Focus on explaining the analysis process and its implications
- Help the user understand both the results and their reliability

Your report should empower the user with a complete understanding of their originality assessment, the methodology behind it, and how to use these insights for their research development.
            """,
            top_p=0.95,
            top_k=60,
            temperature=0.7,
            response_mime_type="text/plain"
        )

    def generate_report_generator_agent_response(self, analysis_data: ReportAnalysisData):
        """
        Generate a comprehensive research report from complete analysis data.
        
        Args:
            analysis_data: ReportAnalysisData containing all Layer 1, Layer 2 results, and methodology
            
        Returns:
            str: Generated research report in markdown format
        """
        # Build comprehensive analysis context
        analysis_context = self._build_analysis_context(analysis_data)
        
        # Generate the report using all analysis data
        prompt = f"""Based on the complete originality analysis data provided, generate a comprehensive research report that explains the analysis process, results, and implications.

## USER'S RESEARCH IDEA
{analysis_data.user_idea}

## COMPLETE ANALYSIS DATA
{analysis_context}

Please generate a detailed markdown report following the structure specified in your system prompt. Focus on explaining how the scores were calculated, what they mean, and what actions the researcher should take based on these results."""
        
        response = self.generate_text_generation_response(prompt)
        return response.text
    
    def _build_analysis_context(self, data: ReportAnalysisData) -> str:
        """Build comprehensive context string from analysis data."""
        context_parts = []
        
        # Paper Information
        context_parts.append("\n## PAPER INFORMATION")
        for i, paper in enumerate(data.papers, 1):
            context_parts.append(f"""
### Paper {i}: {paper.title}
ArXiv ID: {paper.arxiv_id}
Categories: {', '.join(paper.categories)}
Abstract: {paper.abstract}

Key Sections:
{self._format_paper_sections(paper)}
            """)
        
        # Layer 2 Global Results
        l2 = data.layer2_result
        context_parts.append(f"""
## LAYER 2: GLOBAL ORIGINALITY ASSESSMENT
Final Originality Score: {l2.global_originality_score}/100
Global Overlap Score: {l2.global_overlap_score:.3f}
Overall Label: {l2.label.value}
Papers Analyzed: {l2.papers_analyzed}
Summary: {l2.summary}

Aggregated Criteria Scores:
- Problem Similarity: {l2.aggregated_criteria.problem_similarity:.3f}
- Method Similarity: {l2.aggregated_criteria.method_similarity:.3f}
- Domain Overlap: {l2.aggregated_criteria.domain_overlap:.3f}
- Contribution Similarity: {l2.aggregated_criteria.contribution_similarity:.3f}
        """)
        
        # Sentence Annotations
        red_count = len(l2.red_sentences)
        yellow_count = len(l2.yellow_sentences)
        green_count = len(l2.green_sentences)
        
        context_parts.append(f"""
## SENTENCE-LEVEL ANALYSIS
Red Sentences (Low Originality): {red_count}
Yellow Sentences (Medium Originality): {yellow_count}
Green Sentences (High Originality): {green_count}

### Detailed Sentence Analysis with Evidence
        """)
        
        # Add detailed sentence analysis with evidence
        for sentence in l2.sentence_annotations:
            emoji = "🔴" if sentence.label == "low" else "🟡" if sentence.label == "medium" else "🟢"
            context_parts.append(f"""
#### {emoji} Sentence {sentence.index + 1}: {round(sentence.overlap_score * 100)}% Overlap
**Original Sentence:** "{sentence.sentence}"

**Analysis:**
- Originality Score: {sentence.originality_score:.3f}
- Overlap Score: {sentence.overlap_score:.3f}
- Classification: {sentence.label.value}

**Evidence from Papers:**
            """)
            
            if sentence.linked_sections:
                for i, match in enumerate(sentence.linked_sections[:3], 1):  # Show top 3 matches
                    context_parts.append(f"""
**Match {i}:** {match.paper_title}
- **Section:** {match.heading}
- **Similarity:** {round(match.similarity * 100)}%
- **Reason:** {match.reason}
- **Evidence Snippet:** "{match.text_snippet[:200]}{'...' if len(match.text_snippet) > 200 else ''}"
                    """)
            else:
                context_parts.append("No direct matches found - this sentence appears to be original.")
        
        context_parts.append("")
        
        # Layer 1 Per-Paper Results
        context_parts.append("\n## LAYER 1: PER-PAPER ANALYSIS")
        for i, result in enumerate(data.layer1_results, 1):
            context_parts.append(f"""
### Paper {i}: {result.paper_title}
ArXiv ID: {result.arxiv_id}
Overall Overlap Score: {result.overall_overlap_score:.3f}
Originality Threat: {result.originality_threat}
Confidence: {result.confidence}

Criteria Scores:
- Problem Similarity: {result.criteria_scores.problem_similarity:.3f}
- Method Similarity: {result.criteria_scores.method_similarity:.3f}
- Domain Overlap: {result.criteria_scores.domain_overlap:.3f}
- Contribution Similarity: {result.criteria_scores.contribution_similarity:.3f}

Sentence Analysis Summary: {len(result.sentence_analyses)} sentences analyzed
            """)
        
        # Methodology Information
        context_parts.append("\n## METHODOLOGY INFORMATION")
        
        # Criteria Weights
        weights_str = "\n".join([f"- {k}: {v}" for k, v in data.criteria_weights.items()])
        context_parts.append(f"""
### Criteria Weights
{weights_str}
        """)
        
        # Scoring Rubrics
        context_parts.append("\n### Scoring Rubrics (1-5 Likert Scale)")
        for criterion, rubric_info in data.scoring_rubrics.items():
            context_parts.append(f"""
#### {criterion.replace('_', ' ').title()}
Description: {rubric_info['description']}
Rubric:
{rubric_info['rubric']}
            """)
        
        # Threshold Configuration
        thresholds_str = "\n".join([f"- {k}: {v}" for k, v in data.threshold_config.items()])
        context_parts.append(f"""
### Threshold Configuration
{thresholds_str}
        """)
        
        # Processing Metadata
        context_parts.append(f"""
## PROCESSING METADATA
Total Tokens Used: {data.total_tokens_used:,}
Total Cost: ${data.total_cost_usd:.4f}
Processing Time: {data.processing_time_seconds:.2f} seconds
        """)
        
        return "\n".join(context_parts)
    
    def _format_paper_sections(self, paper: Paper) -> str:
        """Format paper sections for context display."""
        sections_text = ""
        for heading in paper.headings:
            if heading.section_text and heading.is_valid:
                # Truncate long sections for readability
                section_content = heading.section_text[:800]
                if len(heading.section_text) > 800:
                    section_content += "..."
                sections_text += f"- **{heading.text}**: {section_content}\n"
        return sections_text if sections_text else "No sections extracted"

