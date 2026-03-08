import logging
from typing import List, Dict, Any, Callable

from app.api.managers.job_manager import job_manager
from app.models.analysis import Layer1Result

from app.agents.layer1_agent import Layer1Agent
from app.agents.layer2_agent import Layer2Aggregator

logger = logging.getLogger(__name__)


class OriginalityService:
    _layer2_aggregator = Layer2Aggregator()

    @classmethod
    def run_layer1_analysis(cls, job_id: str, update_progress: Callable, get_retriever: Callable):
        """Run Layer 1 analysis on each paper (4 criterion calls + 1 sentence call per paper)."""
        job = job_manager.get_job(job_id)
        if not job:
            return

        processed_papers = [p for p in job.state.selected_papers if p.is_processed]
        logger.info(f"Starting Layer 1 analysis on {len(processed_papers)} processed papers")
        update_progress(job_id, f"Analyzing {len(processed_papers)} papers...", 0.78)

        _, retriever = get_retriever(job_id)
        results = []
        layer1_cost = 0.0

        idea = job.state.enriched_idea or job.user_idea

        # Create a fresh agent per job for thread safety
        agent = Layer1Agent()

        for i, paper in enumerate(processed_papers):
            progress = 0.78 + (0.12 * (i / len(processed_papers)))
            title_short = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
            update_progress(
                job_id,
                f"Layer 1 analysis: Paper {i+1}/{len(processed_papers)} — {title_short}",
                progress,
            )

            logger.info(f"Analyzing paper {i+1}/{len(processed_papers)}: {title_short}")

            context_chunks = retriever.get_context_for_paper(paper_id=paper.paper_id, query=idea)
            context_text = "\n\n".join([
                f"[{c.get('metadata', {}).get('heading', 'Section')}]\n{c.get('text', '')[:800]}"
                for c in context_chunks[:5]
            ])
            logger.info(f"Retrieved {len(context_chunks)} context chunks for paper {paper.paper_id}")

            result = agent.analyze_paper(
                user_idea=idea,
                user_sentences=job.state.user_sentences,
                paper=paper,
                paper_context=context_text,
            )
            results.append(result)
            layer1_cost += agent.get_cost()

            logger.info(
                f"Paper {paper.paper_id} analysis complete: {result.overall_overlap_score:.0%} overlap "
                f"(threat: {result.originality_threat})"
            )
            logger.info(
                f"Criteria scores: problem={result.criteria_scores.problem_similarity:.2f}, "
                f"method={result.criteria_scores.method_similarity:.2f}, "
                f"domain={result.criteria_scores.domain_overlap:.2f}, "
                f"contribution={result.criteria_scores.contribution_similarity:.2f}"
            )

        cls._enrich_matched_sections(job_id, results, get_retriever)
        job.state.layer1_results = results
        job.state.cost.layer1 = layer1_cost
        logger.info(f"Layer 1 analysis complete: {len(results)} papers analyzed, total cost: ${layer1_cost:.4f}")
        update_progress(job_id, "Completed Layer 1 analysis", 0.90)

    @classmethod
    def _enrich_matched_sections(cls, job_id: str, results: List[Layer1Result], get_retriever: Callable):
        """Fill empty text_snippet fields using RAG retrieval."""
        _, retriever = get_retriever(job_id)
        for result in results:
            for sent_analysis in result.sentence_analyses:
                for match in sent_analysis.matched_sections:
                    if match.text_snippet:
                        continue
                    rag_matches = retriever.find_matches_for_sentence(sentence=sent_analysis.sentence, top_k=1)
                    if rag_matches:
                        match.text_snippet = rag_matches[0].text_snippet
                        if not match.chunk_id:
                            match.chunk_id = rag_matches[0].chunk_id

    @classmethod
    def run_layer2_analysis(cls, job_id: str, update_progress: Callable):
        """Run Layer 2 aggregation to produce final results."""
        job = job_manager.get_job(job_id)
        if not job:
            return

        update_progress(job_id, "Computing global originality score...", 0.92)

        logger.info(f"Starting Layer 2 aggregation with {len(job.state.layer1_results)} paper results")

        result = cls._layer2_aggregator.aggregate(
            layer1_results=job.state.layer1_results,
            user_sentences=job.state.user_sentences,
            cost_breakdown=job.state.cost
        )
        job.state.layer2_result = result

        logger.info(f"Layer 2 aggregation complete: {result.global_originality_score}/100 originality score")
        logger.info(f"Final summary: {result.summary[:200]}..." if len(result.summary) > 200 else f"Final summary: {result.summary}")
        logger.info(f"Total analysis cost: ${result.cost.total:.4f}")

        update_progress(job_id, f"Originality score: {result.global_originality_score}/100", 0.98)

    @classmethod
    def get_matches_for_sentence(cls, job_id: str, sentence: str, top_k: int, get_retriever: Callable) -> List[Dict[str, Any]]:
        _, retriever = get_retriever(job_id)
        matches = retriever.find_matches_for_sentence(sentence=sentence, top_k=top_k)
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
