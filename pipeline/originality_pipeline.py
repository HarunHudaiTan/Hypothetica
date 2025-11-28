"""
Main orchestrator for the originality analysis pipeline.
Coordinates all components and provides real-time progress updates.
"""
import re
import time
import logging
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass

import config
from models.paper import Paper
from models.analysis import Layer1Result, Layer2Result, CostBreakdown

# Processing components
from processing.arxiv_client import ArxivClient
from processing.pdf_processor import PDFProcessor
from processing.chunk_processor import ChunkProcessor

# RAG components
from rag.chroma_store import ChromaStore
from rag.retriever import Retriever

# Agents
from Agents.keyword_agent import KeywordAgent
from Agents.followup_agent import FollowUpAgent
from Agents.layer1_agent import Layer1Agent
from Agents.layer2_agent import Layer2Aggregator
from Agents.reality_check_agent import RealityCheckAgent

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Holds the current state of the pipeline."""
    user_idea: str = ""
    enriched_idea: str = ""
    user_sentences: List[str] = None
    followup_questions: List[Dict] = None
    followup_answers: List[str] = None
    keywords: List[str] = None
    all_papers: List[Dict] = None
    selected_papers: List[Paper] = None
    layer1_results: List[Layer1Result] = None
    layer2_result: Layer2Result = None
    cost: CostBreakdown = None
    reality_check_result: Dict = None  # NEW: Reality check result
    reality_check_warning: str = None  # NEW: Warning message if idea exists
    
    def __post_init__(self):
        self.user_sentences = self.user_sentences or []
        self.followup_questions = self.followup_questions or []
        self.followup_answers = self.followup_answers or []
        self.keywords = self.keywords or []
        self.all_papers = self.all_papers or []
        self.selected_papers = self.selected_papers or []
        self.layer1_results = self.layer1_results or []
        self.cost = self.cost or CostBreakdown()
        self.reality_check_result = self.reality_check_result or {}


class OriginalityPipeline:
    """
    Main pipeline for research idea originality analysis.
    
    Flow:
    1. Generate follow-up questions → User answers
    2. Enrich idea with answers
    3. Generate search keywords
    4. Search arXiv for papers
    5. Select top papers
    6. Process PDFs → Extract headings → Chunk
    7. Store chunks in ChromaDB
    8. Layer 1: Analyze each paper
    9. Layer 2: Aggregate results
    10. Return final assessment
    """
    
    def __init__(self, progress_callback: Callable[[str, float], None] = None):
        """
        Initialize pipeline.
        
        Args:
            progress_callback: Function(message, progress_pct) for real-time updates
        """
        self.progress_callback = progress_callback or (lambda msg, pct: None)
        
        # Initialize components
        self.arxiv_client = ArxivClient()
        self.pdf_processor = PDFProcessor()
        self.chunk_processor = ChunkProcessor()
        self.chroma_store = None  # Initialized lazily
        self.retriever = None
        
        # Initialize agents
        self.keyword_agent = KeywordAgent()
        self.followup_agent = FollowUpAgent()
        self.layer1_agent = Layer1Agent()
        self.layer2_aggregator = Layer2Aggregator()
        self.reality_check_agent = RealityCheckAgent()  # NEW
        
        # State
        self.state = PipelineState()
    
    def _update_progress(self, message: str, progress: float):
        """Send progress update."""
        self.progress_callback(message, progress)
        logger.info(f"[{progress:.0%}] {message}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Filter and clean
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # =========================================================================
    # STEP 0: Reality Check (NEW)
    # =========================================================================
    def run_reality_check(self, user_idea: str) -> Dict:
        """
        Check if the idea already exists using LLM's general knowledge.
        This catches cases where arXiv won't help (e.g., Facebook, Uber).
        
        Args:
            user_idea: The user's research idea
            
        Returns:
            Dict with reality check results
        """
        self._update_progress("Checking if similar products/research already exist...", 0.02)
        
        result = self.reality_check_agent.check_idea(user_idea)
        self.state.reality_check_result = result
        
        # Generate warning if needed
        warning = self.reality_check_agent.get_warning_message(result)
        self.state.reality_check_warning = warning
        
        if result.get('already_exists', False):
            confidence = result.get('confidence', 0)
            examples = result.get('existing_examples', [])
            if examples:
                top = examples[0].get('name', 'similar products')
                self._update_progress(
                    f"⚠️ Found potential match: {top} (confidence: {confidence:.0%})",
                    0.04
                )
            else:
                self._update_progress(
                    f"⚠️ Similar concepts may exist (confidence: {confidence:.0%})",
                    0.04
                )
        else:
            self._update_progress("No obvious existing products found. Proceeding with analysis.", 0.04)
        
        return result
    
    # =========================================================================
    # STEP 1: Generate Follow-up Questions
    # =========================================================================
    def generate_followup_questions(self, user_idea: str) -> List[Dict]:
        """
        Generate follow-up questions to clarify the research idea.
        
        Args:
            user_idea: Initial user research idea
            
        Returns:
            List of question dicts
        """
        self._update_progress("Generating follow-up questions...", 0.05)
        
        self.state.user_idea = user_idea
        questions = self.followup_agent.generate_questions(user_idea)
        self.state.followup_questions = questions
        
        # Track cost
        self.state.cost.followup = self.followup_agent.get_cost()
        
        self._update_progress(f"Generated {len(questions)} follow-up questions", 0.10)
        return questions
    
    # =========================================================================
    # STEP 2: Process Answers and Enrich Idea
    # =========================================================================
    def process_answers(self, answers: List[str]) -> str:
        """
        Process user answers and create enriched idea.
        
        Args:
            answers: User's answers to follow-up questions
            
        Returns:
            Enriched idea text
        """
        self._update_progress("Processing your answers...", 0.12)
        
        self.state.followup_answers = answers
        
        enriched = self.followup_agent.enrich_idea_with_answers(
            self.state.user_idea,
            self.state.followup_questions,
            answers
        )
        self.state.enriched_idea = enriched
        
        # Split into sentences for analysis
        self.state.user_sentences = self._split_into_sentences(self.state.user_idea)
        
        self._update_progress("Idea enriched with clarifications", 0.15)
        return enriched
    
    # =========================================================================
    # STEP 3: Generate Keywords and Search
    # =========================================================================
    def search_papers(self) -> List[Paper]:
        """
        Generate keywords and search arXiv for relevant papers.
        
        Returns:
            List of selected Paper objects
        """
        # Generate keywords
        self._update_progress("Generating search keywords...", 0.18)
        
        idea_for_keywords = self.state.enriched_idea or self.state.user_idea
        keywords = self.keyword_agent.generate_keyword_agent_response(idea_for_keywords)
        self.state.keywords = keywords
        
        # Track cost
        self.state.cost.keywords = 0.001  # Keyword agent cost is minimal
        
        self._update_progress(f"Generated {len(keywords)} keywords: {', '.join(keywords[:3])}...", 0.22)
        
        # Search arXiv
        self._update_progress("Searching arXiv...", 0.25)
        
        all_papers = self.arxiv_client.search_multiple_keywords(
            keywords=keywords,
            results_per_keyword=config.PAPERS_PER_KEYWORD
        )
        self.state.all_papers = all_papers
        
        self._update_progress(f"Found {len(all_papers)} papers", 0.30)
        
        # Convert to Paper models and limit
        selected = self.arxiv_client.papers_to_models(
            all_papers,
            limit=config.MAX_PAPERS_TO_ANALYZE
        )
        self.state.selected_papers = selected
        
        self._update_progress(f"Selected top {len(selected)} papers for analysis", 0.32)
        return selected
    
    # =========================================================================
    # STEP 4: Process PDFs and Build RAG Index
    # =========================================================================
    def process_papers(self) -> int:
        """
        Process PDFs, extract content, chunk, and index in ChromaDB.
        
        Returns:
            Total number of chunks indexed
        """
        # Initialize ChromaDB
        self._update_progress("Initializing vector store...", 0.35)
        self.chroma_store = ChromaStore()
        self.retriever = Retriever(self.chroma_store)
        
        total_chunks = 0
        num_papers = len(self.state.selected_papers)
        
        for i, paper in enumerate(self.state.selected_papers):
            progress = 0.35 + (0.30 * (i / num_papers))  # 0.35 to 0.65
            
            # Process PDF
            self._update_progress(
                f"Processing paper {i+1}/{num_papers}: {paper.title[:40]}...",
                progress
            )
            
            try:
                # Extract content
                self.pdf_processor.process_paper(paper)
                
                if paper.is_processed and paper.headings:
                    # Chunk the content
                    self.chunk_processor.process_paper(paper)
                    
                    # Index in ChromaDB
                    chunks_added = self.chroma_store.add_paper(paper)
                    total_chunks += chunks_added
                    
                    self._update_progress(
                        f"Indexed {chunks_added} chunks from paper {i+1}",
                        progress + 0.02
                    )
                else:
                    logger.warning(f"Paper {paper.paper_id} failed to process")
                    
            except Exception as e:
                logger.error(f"Error processing paper {paper.paper_id}: {e}")
                paper.processing_error = str(e)
        
        self._update_progress(f"Indexed {total_chunks} total chunks", 0.65)
        return total_chunks
    
    # =========================================================================
    # STEP 5: Layer 1 Analysis
    # =========================================================================
    def run_layer1_analysis(self) -> List[Layer1Result]:
        """
        Run Layer 1 analysis on each paper.
        
        Returns:
            List of Layer1Result objects
        """
        results = []
        num_papers = len(self.state.selected_papers)
        processed_papers = [p for p in self.state.selected_papers if p.is_processed]
        
        self._update_progress(f"Analyzing {len(processed_papers)} papers...", 0.68)
        
        layer1_cost = 0.0
        
        for i, paper in enumerate(processed_papers):
            progress = 0.68 + (0.20 * (i / len(processed_papers)))  # 0.68 to 0.88
            
            self._update_progress(
                f"Layer 1 analysis: Paper {i+1}/{len(processed_papers)}",
                progress
            )
            
            # Get relevant context from ChromaDB
            context_chunks = self.retriever.get_context_for_paper(
                paper_id=paper.paper_id,
                query=self.state.enriched_idea or self.state.user_idea
            )
            
            context_text = "\n\n".join([
                f"[{c.get('metadata', {}).get('heading', 'Section')}]\n{c.get('text', '')[:800]}"
                for c in context_chunks[:5]
            ])
            
            # Run analysis
            result = self.layer1_agent.analyze_paper(
                user_idea=self.state.enriched_idea or self.state.user_idea,
                user_sentences=self.state.user_sentences,
                paper=paper,
                paper_context=context_text
            )
            
            results.append(result)
            layer1_cost += self.layer1_agent.get_cost()
            
            self._update_progress(
                f"Paper {i+1} overlap score: {result.overall_overlap_score:.2f}",
                progress + 0.02
            )
        
        self.state.layer1_results = results
        self.state.cost.layer1 = layer1_cost
        
        self._update_progress(f"Completed Layer 1 analysis for {len(results)} papers", 0.88)
        return results
    
    # =========================================================================
    # STEP 6: Layer 2 Aggregation
    # =========================================================================
    def run_layer2_analysis(self) -> Layer2Result:
        """
        Run Layer 2 aggregation to produce final results.
        
        Returns:
            Layer2Result with final assessment
        """
        self._update_progress("Computing global originality score...", 0.90)
        
        result = self.layer2_aggregator.aggregate(
            layer1_results=self.state.layer1_results,
            user_sentences=self.state.user_sentences,
            cost_breakdown=self.state.cost
        )
        
        # NEW: Adjust score based on reality check
        if self.state.reality_check_result:
            original_score = result.global_originality_score
            adjusted_score = self.reality_check_agent.adjust_originality_score(
                original_score,
                self.state.reality_check_result
            )
            
            if adjusted_score != original_score:
                result.global_originality_score = adjusted_score
                
                # Update summary to mention reality check
                rc = self.state.reality_check_result
                if rc.get('existing_examples'):
                    top_example = rc['existing_examples'][0].get('name', 'existing products')
                    result.summary = (
                        f"⚠️ This idea closely resembles {top_example}. "
                        f"{result.summary} "
                        f"Score adjusted from {original_score} to {adjusted_score} due to existing similar products."
                    )
        
        self.state.layer2_result = result
        
        self._update_progress(
            f"Originality score: {result.global_originality_score}/100",
            0.95
        )
        
        return result
    
    # =========================================================================
    # CONVENIENCE: Run Full Pipeline
    # =========================================================================
    def run_full_analysis(
        self,
        user_idea: str,
        followup_answers: List[str] = None
    ) -> Layer2Result:
        """
        Run the complete analysis pipeline.
        
        Args:
            user_idea: User's research idea
            followup_answers: Pre-provided answers (skip question step if provided)
            
        Returns:
            Final Layer2Result
        """
        start_time = time.time()
        
        try:
            # Step 0 (NEW): Reality Check - check if idea already exists
            self.run_reality_check(user_idea)
            
            # Step 1 & 2: Questions and enrichment
            if followup_answers:
                self.state.user_idea = user_idea
                questions = self.generate_followup_questions(user_idea)
                self.process_answers(followup_answers)
            else:
                # Just use the idea as-is if no answers provided
                self.state.user_idea = user_idea
                self.state.enriched_idea = user_idea
                self.state.user_sentences = self._split_into_sentences(user_idea)
            
            # Step 3: Search papers
            self.search_papers()
            
            # Step 4: Process PDFs and index
            self.process_papers()
            
            # Step 5: Layer 1
            self.run_layer1_analysis()
            
            # Step 6: Layer 2
            result = self.run_layer2_analysis()
            
            # Final update
            elapsed = time.time() - start_time
            result.total_processing_time = elapsed
            
            self._update_progress(
                f"Analysis complete! Score: {result.global_originality_score}/100",
                1.0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self._update_progress(f"Error: {str(e)}", -1)
            raise
    
    # =========================================================================
    # RAG QUERY (for UI click-through)
    # =========================================================================
    def get_matches_for_sentence(
        self,
        sentence: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get matching chunks for a specific sentence (RAG query).
        Used when user clicks on a highlighted sentence.
        
        Args:
            sentence: The sentence to find matches for
            top_k: Number of results
            
        Returns:
            List of matching chunks with metadata
        """
        if not self.retriever:
            return []
        
        matches = self.retriever.find_matches_for_sentence(
            sentence=sentence,
            top_k=top_k
        )
        
        return [
            {
                "paper_title": m.paper_title,
                "heading": m.heading,
                "text": m.text_snippet,
                "similarity": m.similarity,
                "reason": m.reason
            }
            for m in matches
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return {
            "papers_found": len(self.state.all_papers),
            "papers_analyzed": len(self.state.selected_papers),
            "papers_processed": len([p for p in self.state.selected_papers if p.is_processed]),
            "total_chunks": self.chroma_store.count() if self.chroma_store else 0,
            "keywords": self.state.keywords,
            "cost": self.state.cost.to_dict() if self.state.cost else {}
        }

