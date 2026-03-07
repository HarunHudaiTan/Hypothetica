import re
import time
import logging
import threading
from typing import List, Dict, Any

from api.managers.job_manager import job_manager
from api.schemas.job import JobStatus

from retrieval.chroma_store import ChromaStore
from retrieval.retriever import Retriever

from Agents.followup_agent import FollowUpAgent
from Agents.reality_check_agent import RealityCheckAgent

from api.services.paper_search_service import PaperSearchService
from api.services.paper_processing_service import PaperProcessingService
from api.services.originality_service import OriginalityService

logger = logging.getLogger(__name__)


class AnalysisService:
    _followup_agent = FollowUpAgent()
    _reality_check_agent = RealityCheckAgent()

    @staticmethod
    def _get_retriever(job_id: str):
        """Helper to get ChromaStore and Retriever."""
        store = ChromaStore()
        return store, Retriever(store)

    @staticmethod
    def _update_progress(job_id: str, message: str, progress: float):
        """Update job progress and log."""
        job_manager.update_progress(job_id, message, progress)
        logger.info(f"Job {job_id} [{progress:.0%}] {message}")

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    @classmethod
    def run_reality_check(cls, job_id: str):
        """Check if the idea already exists using LLM's general knowledge."""
        job = job_manager.get_job(job_id)
        if not job: return

        cls._update_progress(job_id, "Checking if similar products/research already exist...", 0.02)

        result = cls._reality_check_agent.check_idea(job.user_idea)
        job.state.reality_check_result = result

        warning = cls._reality_check_agent.get_warning_message(result)
        job.state.reality_check_warning = warning

        # Push event for real-time UI warning
        job.push_event({
            "type": "reality_check",
            "reality_check": result,
            "warning": warning,
        })

        if result.get('already_exists', False):
            confidence = result.get('confidence', 0)
            examples = result.get('existing_examples', [])
            msg = f"⚠️ Potential match found (confidence: {confidence:.0%})"
            if examples:
                msg = f"⚠️ Found potential match: {examples[0].get('name')} (confidence: {confidence:.0%})"
            cls._update_progress(job_id, msg, 0.04)
        else:
            cls._update_progress(job_id, "No obvious existing products found. Proceeding.", 0.04)

    @classmethod
    def generate_followup_questions(cls, job_id: str):
        """Generate follow-up questions to clarify the research idea."""
        job = job_manager.get_job(job_id)
        if not job: return

        cls._update_progress(job_id, "Generating follow-up questions...", 0.05)

        questions = cls._followup_agent.generate_questions(job.user_idea)
        job.state.followup_questions = questions
        job.state.cost.followup = cls._followup_agent.get_cost()

        job_manager.set_questions(job_id, questions)
        cls._update_progress(job_id, f"Generated {len(questions)} follow-up questions", 0.08)

    @classmethod
    def process_answers(cls, job_id: str, answers: List[str]):
        """Process user answers and create enriched idea."""
        job = job_manager.get_job(job_id)
        if not job: return

        cls._update_progress(job_id, "Processing your answers...", 0.10)

        job.state.followup_answers = answers

        enriched = cls._followup_agent.enrich_idea_with_answers(
            job.user_idea,
            job.state.followup_questions,
            answers
        )
        job.state.enriched_idea = enriched
        job.state.user_sentences = cls._split_into_sentences(job.user_idea)

        cls._update_progress(job_id, "Idea enriched with clarifications", 0.12)

    @classmethod
    def _run_questions_phase_worker(cls, job_id: str):
        try:
            cls.generate_followup_questions(job_id)
        except Exception as e:
            logger.exception(f"Error in questions phase for job {job_id}")
            job_manager.set_error(job_id, str(e))

    @classmethod
    def _run_analysis_phase_worker(cls, job_id: str, answers: list):
        job = job_manager.get_job(job_id)
        if not job: return

        try:
            job_manager.update_status(job_id, JobStatus.PROCESSING)
            start_time = time.time()

            # Parallel reality check
            rc_thread = threading.Thread(target=cls.run_reality_check, args=(job_id,), daemon=True)
            rc_thread.start()

            cls.process_answers(job_id, answers)
            PaperSearchService.search_papers(job_id, cls._update_progress)
            PaperProcessingService.process_papers(job_id, cls._update_progress, cls._get_retriever)
            OriginalityService.run_layer1_analysis(job_id, cls._update_progress, cls._get_retriever)
            OriginalityService.run_layer2_analysis(job_id, cls._update_progress)

            rc_thread.join(timeout=10)

            # Finalize results
            result = job.state.layer2_result
            result.total_processing_time = time.time() - start_time
            results_dict = result.to_dict()

            # Enrich with papers detail
            papers_detail = []
            for paper in job.state.selected_papers:
                l1 = next((r for r in job.state.layer1_results if r.paper_id == paper.paper_id), None)
                entry = {
                    "paper_id": paper.paper_id,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "url": paper.url,
                    "pdf_url": paper.pdf_url,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "is_processed": paper.is_processed,
                }
                if l1:
                    entry["overall_overlap_score"] = l1.overall_overlap_score
                    entry["criteria_scores"] = l1.criteria_scores.to_dict()
                papers_detail.append(entry)

            results_dict["papers"] = papers_detail
            results_dict["reality_check"] = job.state.reality_check_result
            results_dict["reality_check_warning"] = job.state.reality_check_warning
            results_dict["stats"] = cls.get_stats(job_id)

            job_manager.set_results(job_id, results_dict)
            cls._update_progress(job_id, "Analysis complete!", 1.0)

        except Exception as e:
            logger.exception(f"Error in analysis phase for job {job_id}")
            job_manager.set_error(job_id, str(e))

    @staticmethod
    def start_questions_phase(job_id: str):
        thread = threading.Thread(target=AnalysisService._run_questions_phase_worker, args=(job_id,), daemon=True)
        thread.start()

    @staticmethod
    def start_analysis_phase(job_id: str, answers: list):
        thread = threading.Thread(target=AnalysisService._run_analysis_phase_worker, args=(job_id, answers), daemon=True)
        thread.start()

    @classmethod
    def get_matches_for_sentence(cls, job_id: str, sentence: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return OriginalityService.get_matches_for_sentence(job_id, sentence, top_k, cls._get_retriever)

    @classmethod
    def get_stats(cls, job_id: str) -> Dict[str, Any]:
        job = job_manager.get_job(job_id)
        if not job: return {}

        store = ChromaStore()
        return {
            "query_variants": len(job.state.query_variants),
            "total_fetched": job.state.total_papers_fetched,
            "unique_after_dedup": job.state.unique_papers_after_dedup,
            "after_rerank": job.state.papers_after_rerank,
            "papers_found": len(job.state.all_papers),
            "papers_analyzed": len(job.state.selected_papers),
            "papers_processed": len([p for p in job.state.selected_papers if p.is_processed]),
            "total_chunks": store.count(),
            "keywords": job.state.keywords,
            "cost": job.state.cost.to_dict()
        }
