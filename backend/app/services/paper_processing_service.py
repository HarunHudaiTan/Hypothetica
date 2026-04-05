import logging
from typing import Callable

from app.api.managers.job_manager import job_manager

from app.processing.pdf_processor import PDFProcessor
from app.processing.chunk_processor import ChunkProcessor
from app.retrieval.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class PaperProcessingService:
    _pdf_processor = PDFProcessor()
    _chunk_processor = ChunkProcessor()

    @classmethod
    def process_papers(cls, job_id: str, update_progress: Callable, get_retriever: Callable):
        """Process PDFs, extract content, chunk, and index."""
        job = job_manager.get_job(job_id)
        if not job:
            return

        update_progress(job_id, "Initializing vector store...", 0.55)
        store, _ = get_retriever(job_id)
        logger.info(f"Initialized ChromaDB vector store")

        total_chunks = 0
        num_papers = len(job.state.selected_papers)
        processed_count = 0

        # Phase 1: Parallel PDF processing (download + conversion)
        update_progress(job_id, "Processing PDFs in parallel...", 0.56)
        cls._pdf_processor.process_papers_parallel(
            job.state.selected_papers,
            max_workers=min(5, num_papers)
        )

        # Check for patent translation warnings and notify user
        patent_warnings = []
        for paper in job.state.selected_papers:
            patent_info = paper.metadata.get('patent_processing', {})
            warning_msg = patent_info.get('warning_message')
            if warning_msg:
                patent_warnings.append({
                    'paper_id': paper.paper_id,
                    'title': paper.title,
                    'warning': warning_msg,
                    'detected_languages': patent_info.get('detected_languages', []),
                    'translated_sections': patent_info.get('translated_sections_count', 0),
                })

        if patent_warnings:
            logger.warning(
                f"{len(patent_warnings)} patent paper(s) contained non-English content "
                f"that was machine-translated"
            )
            job.push_event({
                "type": "patent_translation_warning",
                "message": (
                    f"⚠️ {len(patent_warnings)} patent paper(s) contained non-English sections "
                    f"that were machine-translated to English. Translated text may reduce "
                    f"embedding accuracy and affect the global originality score."
                ),
                "papers": patent_warnings,
            })

        # Phase 2: Sequential chunking + indexing (maintains order for progress)
        for i, paper in enumerate(job.state.selected_papers):
            progress = 0.56 + (0.19 * ((i + 1) / num_papers))
            title_short = paper.title[:40] + "..." if len(paper.title) > 40 else paper.title
            update_progress(job_id, f"Indexing paper {i+1}/{num_papers}: {title_short}", progress)

            try:
                if paper.is_processed and paper.headings:
                    logger.info(f"Chunking paper {paper.paper_id}")
                    cls._chunk_processor.process_paper(paper)

                    chunks_added = store.add_paper(paper)
                    total_chunks += chunks_added
                    processed_count += 1
                    logger.info(f"Added {chunks_added} chunks for paper {paper.paper_id}")
                else:
                    logger.warning(f"Failed to process paper {paper.paper_id}")
            except Exception as e:
                logger.error(f"Error processing paper {paper.paper_id}: {e}")
                paper.processing_error = str(e)

        logger.info(f"Processing complete: {processed_count}/{num_papers} papers processed, {total_chunks} total chunks indexed")
        update_progress(job_id, f"Indexed {total_chunks} total chunks from {processed_count} papers", 0.75)

