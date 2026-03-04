"""
Main orchestrator for the originality analysis pipeline.
Coordinates all components and provides real-time progress updates.

Enhanced retrieval pipeline:
1. Query variants (not keywords) → High-recall arXiv search
2. Embedding similarity → Top 100
3. Cross-encoder rerank → Top 20
4. LLM selection → Final 5
"""
import re
import time
import json
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

# Enhanced retrieval components
from ArxivReq import ArxivReq
from embeddemo.embed_query_wrapper import QueryWrapper
from Agents.relevant_paper_selector_agent import RelevantPaperSelectorAgent

# Agents
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
    query_variants: List[Dict] = None  # NEW: Query variants instead of keywords
    keywords: List[str] = None  # Keep for backward compatibility
    all_papers: List[Dict] = None
    selected_papers: List[Paper] = None
    layer1_results: List[Layer1Result] = None
    layer2_result: Layer2Result = None
    cost: CostBreakdown = None
    reality_check_result: Dict = None
    reality_check_warning: str = None
    # Pipeline stats
    total_papers_fetched: int = 0
    unique_papers_after_dedup: int = 0
    papers_after_embedding: int = 0
    papers_after_rerank: int = 0
    
    def __post_init__(self):
        self.user_sentences = self.user_sentences or []
        self.followup_questions = self.followup_questions or []
        self.followup_answers = self.followup_answers or []
        self.query_variants = self.query_variants or []
        self.keywords = self.keywords or []
        self.all_papers = self.all_papers or []
        self.selected_papers = self.selected_papers or []
        self.layer1_results = self.layer1_results or []
        self.cost = self.cost or CostBreakdown()
        self.reality_check_result = self.reality_check_result or {}


class OriginalityPipeline:
    """
    Main pipeline for research idea originality analysis.
    
    Enhanced Flow:
    1. Reality check (LLM general knowledge)
    2. Generate follow-up questions → User answers
    3. Enrich idea with answers
    4. Generate query variants (raw, academic, synonyms)
    5. Search arXiv with high recall (150+ papers per variant)
    6. Deduplicate papers
    7. Embedding search → Top 100
    8. Cross-encoder rerank → Top 20
    9. LLM selection → Final 5
    10. Process PDFs → Extract headings → Chunk
    11. Store chunks in ChromaDB
    12. Layer 1: Analyze each paper
    13. Layer 2: Aggregate results
    """
    
    def __init__(self, progress_callback: Callable[[str, float], None] = None):
        """
        Initialize pipeline.
        
        Args:
            progress_callback: Function(message, progress_pct) for real-time updates
        """
        self.progress_callback = progress_callback or (lambda msg, pct: None)
        
        # Legacy components (for PDF processing)
        self.arxiv_client = ArxivClient()
        self.pdf_processor = PDFProcessor()
        self.chunk_processor = ChunkProcessor()
        self.chroma_store = None
        self.retriever = None
        
        # Enhanced retrieval components
        self.arxiv_req = ArxivReq()
        self.query_wrapper = QueryWrapper(use_reranker=True)
        self.paper_selector = RelevantPaperSelectorAgent()
        
        # Analysis agents
        self.followup_agent = FollowUpAgent()
        self.layer1_agent = Layer1Agent()
        self.layer2_aggregator = Layer2Aggregator()
        self.reality_check_agent = RealityCheckAgent()
        
        # State
        self.state = PipelineState()
        
        # Configuration
        self.papers_per_query = 150  # High recall
        self.embedding_topk = 100
        self.rerank_topk = 20
        self.final_papers = config.MAX_PAPERS_TO_ANALYZE
    
    def _update_progress(self, message: str, progress: float):
        """Send progress update."""
        self.progress_callback(message, progress)
        logger.info(f"[{progress:.0%}] {message}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # =========================================================================
    # STEP 0: Reality Check
    # =========================================================================
    def run_reality_check(self, user_idea: str) -> Dict:
        """
        Check if the idea already exists using LLM's general knowledge.
        """
        self._update_progress("Checking if similar products/research already exist...", 0.02)
        
        result = self.reality_check_agent.check_idea(user_idea)
        self.state.reality_check_result = result
        
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
        """Generate follow-up questions to clarify the research idea."""
        self._update_progress("Generating follow-up questions...", 0.05)
        
        self.state.user_idea = user_idea
        questions = self.followup_agent.generate_questions(user_idea)
        self.state.followup_questions = questions
        
        self.state.cost.followup = self.followup_agent.get_cost()
        
        self._update_progress(f"Generated {len(questions)} follow-up questions", 0.08)
        return questions
    
    # =========================================================================
    # STEP 2: Process Answers and Enrich Idea
    # =========================================================================
    def process_answers(self, answers: List[str]) -> str:
        """Process user answers and create enriched idea."""
        self._update_progress("Processing your answers...", 0.10)
        
        self.state.followup_answers = answers
        
        enriched = self.followup_agent.enrich_idea_with_answers(
            self.state.user_idea,
            self.state.followup_questions,
            answers
        )
        self.state.enriched_idea = enriched
        self.state.user_sentences = self._split_into_sentences(self.state.user_idea)
        
        self._update_progress("Idea enriched with clarifications", 0.12)
        return enriched
    
    # =========================================================================
    # STEP 3: ENHANCED PAPER SEARCH
    # =========================================================================
    def search_papers(self) -> List[Paper]:
        """
        Enhanced paper search with high-recall retrieval.
        
        Pipeline:
        1. Generate query variants (raw, academic, synonyms)
        2. Search arXiv with large pool per variant
        3. Deduplicate by arXiv ID
        4. Embedding search → Top 100
        5. Cross-encoder rerank → Top 20
        6. LLM selection → Final 5
        
        Returns:
            List of selected Paper objects
        """
        idea_for_search = self.state.enriched_idea or self.state.user_idea
        
        # ----- Step 3a: High-recall arXiv search -----
        self._update_progress("Generating query variants and searching arXiv...", 0.15)
        
        # This uses QueryVariantAgent internally and handles deduplication
        papers_json = self.arxiv_req.get_papers(idea_for_search, papers_per_query=self.papers_per_query)
        papers_summary = json.loads(papers_json)
        
        # Store query variants for stats
        self.state.query_variants = papers_summary.get('query_variants', [])
        self.state.keywords = [v['query'] for v in self.state.query_variants]  # Backward compat
        self.state.total_papers_fetched = papers_summary.get('total_papers_fetched', 0)
        self.state.unique_papers_after_dedup = papers_summary.get('unique_papers', 0)
        
        self._update_progress(
            f"Generated {len(self.state.query_variants)} variants, found {self.state.unique_papers_after_dedup} unique papers",
            0.25
        )
        
        # ----- Step 3b: Embedding search + Cross-encoder rerank -----
        self._update_progress("Running semantic search with embeddings + cross-encoder...", 0.30)
        
        search_results = self.query_wrapper.search_literature(
            idea_for_search,
            include_scores=True,
            embedding_topk=self.embedding_topk,
            rerank_topk=self.rerank_topk,
            force_rebuild=True  # Rebuild since we have new papers
        )
        
        search_results_list = json.loads(search_results)
        self.state.papers_after_rerank = len(search_results_list) if isinstance(search_results_list, list) else 0
        
        self._update_progress(
            f"Semantic search: {self.embedding_topk} candidates → {self.state.papers_after_rerank} after rerank",
            0.40
        )
        
        # ----- Step 3c: LLM final selection -----
        self._update_progress(f"LLM selecting final {self.final_papers} papers...", 0.45)
        
        selected_json = self.paper_selector.generate_relevant_paper_selector_response(
            idea_for_search,
            search_results,
            final_count=self.final_papers
        )
        selected_papers_data = json.loads(selected_json)
        
        self._update_progress(f"Selected {len(selected_papers_data)} papers for detailed analysis", 0.50)
        
        # Store all papers for reference
        self.state.all_papers = search_results_list if isinstance(search_results_list, list) else []
        
        # ----- Convert to Paper models -----
        papers = []
        for i, pd in enumerate(selected_papers_data):
            # Construct proper URLs
            arxiv_id = pd.get('id', '')
            url = pd.get('url', '')
            if not url and arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = url.replace('/abs/', '/pdf/') if url else f"https://arxiv.org/pdf/{arxiv_id}"
            
            paper = Paper(
                paper_id=f"paper_{i+1:02d}",
                arxiv_id=arxiv_id,
                title=pd.get('title', ''),
                abstract=pd.get('abstract', ''),
                url=url,
                pdf_url=pdf_url,
                authors=pd.get('authors', []),
                categories=pd.get('categories', []),
                published_date=str(pd.get('year', '')) if pd.get('year') else None
            )
            papers.append(paper)
        
        self.state.selected_papers = papers
        
        self._update_progress(f"Ready to analyze {len(papers)} papers", 0.52)
        return papers
    
    # =========================================================================
    # STEP 4: Process PDFs and Build RAG Index
    # =========================================================================
    def process_papers(self) -> int:
        """Process PDFs, extract content, chunk, and index in ChromaDB."""
        self._update_progress("Initializing vector store...", 0.55)
        self.chroma_store = ChromaStore()
        self.retriever = Retriever(self.chroma_store)
        
        total_chunks = 0
        num_papers = len(self.state.selected_papers)
        
        for i, paper in enumerate(self.state.selected_papers):
            progress = 0.55 + (0.20 * (i / num_papers))
            
            self._update_progress(
                f"Processing paper {i+1}/{num_papers}: {paper.title[:40]}...",
                progress
            )
            
            try:
                self.pdf_processor.process_paper(paper)
                
                if paper.is_processed and paper.headings:
                    self.chunk_processor.process_paper(paper)
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
        
        self._update_progress(f"Indexed {total_chunks} total chunks", 0.75)
        return total_chunks
    
    # =========================================================================
    # STEP 5: Layer 1 Analysis
    # =========================================================================
    def run_layer1_analysis(self) -> List[Layer1Result]:
        """Run Layer 1 analysis on each paper."""
        results = []
        processed_papers = [p for p in self.state.selected_papers if p.is_processed]
        
        self._update_progress(f"Analyzing {len(processed_papers)} papers...", 0.78)
        
        layer1_cost = 0.0
        
        for i, paper in enumerate(processed_papers):
            progress = 0.78 + (0.12 * (i / len(processed_papers)))
            
            self._update_progress(
                f"Layer 1 analysis: Paper {i+1}/{len(processed_papers)}",
                progress
            )
            
            context_chunks = self.retriever.get_context_for_paper(
                paper_id=paper.paper_id,
                query=self.state.enriched_idea or self.state.user_idea
            )
            
            context_text = "\n\n".join([
                f"[{c.get('metadata', {}).get('heading', 'Section')}]\n{c.get('text', '')[:800]}"
                for c in context_chunks[:5]
            ])
            
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
                progress + 0.01
            )
        
        self._update_progress("Enriching matches with source text...", 0.88)
        self._enrich_matched_sections(results)

        self.state.layer1_results = results
        self.state.cost.layer1 = layer1_cost
        
        self._update_progress(f"Completed Layer 1 analysis for {len(results)} papers", 0.90)
        return results

    def _enrich_matched_sections(self, results: List[Layer1Result]):
        """
        Fill empty text_snippet fields in matched sections using RAG retrieval.
        The LLM should already provide similar_text, but this is a fallback
        for any sections it missed.
        """
        if not self.retriever:
            return

        for result in results:
            for sent_analysis in result.sentence_analyses:
                for match in sent_analysis.matched_sections:
                    if match.text_snippet:
                        continue
                    # Fallback: fetch from ChromaDB using the sentence as query
                    rag_matches = self.retriever.find_matches_for_sentence(
                        sentence=sent_analysis.sentence,
                        top_k=1,
                        similarity_threshold=0.0,
                    )
                    if rag_matches:
                        match.text_snippet = rag_matches[0].text_snippet
                        if not match.chunk_id:
                            match.chunk_id = rag_matches[0].chunk_id
    
    # =========================================================================
    # STEP 6: Layer 2 Aggregation
    # =========================================================================
    def run_layer2_analysis(self) -> Layer2Result:
        """Run Layer 2 aggregation to produce final results."""
        self._update_progress("Computing global originality score...", 0.92)
        
        result = self.layer2_aggregator.aggregate(
            layer1_results=self.state.layer1_results,
            user_sentences=self.state.user_sentences,
            cost_breakdown=self.state.cost
        )
        
        self.state.layer2_result = result
        
        self._update_progress(
            f"Originality score: {result.global_originality_score}/100",
            0.98
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
        """Run the complete analysis pipeline."""
        start_time = time.time()
        
        try:
            # Step 0: Reality Check
            self.run_reality_check(user_idea)
            
            # Steps 1 & 2: Questions and enrichment
            if followup_answers:
                self.state.user_idea = user_idea
                self.generate_followup_questions(user_idea)
                self.process_answers(followup_answers)
            else:
                self.state.user_idea = user_idea
                self.state.enriched_idea = user_idea
                self.state.user_sentences = self._split_into_sentences(user_idea)
            
            # Step 3: Enhanced paper search
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
        """Get matching chunks for a specific sentence."""
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
            "query_variants": len(self.state.query_variants),
            "total_fetched": self.state.total_papers_fetched,
            "unique_after_dedup": self.state.unique_papers_after_dedup,
            "after_rerank": self.state.papers_after_rerank,
            "papers_found": len(self.state.all_papers),
            "papers_analyzed": len(self.state.selected_papers),
            "papers_processed": len([p for p in self.state.selected_papers if p.is_processed]),
            "total_chunks": self.chroma_store.count() if self.chroma_store else 0,
            "keywords": self.state.keywords,
            "cost": self.state.cost.to_dict() if self.state.cost else {}
        }
