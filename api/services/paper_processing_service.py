import logging
from typing import Callable

from api.managers.job_manager import job_manager

from processing.pdf_processor import PDFProcessor
from processing.chunk_processor import ChunkProcessor
from retrieval.chroma_store import ChromaStore

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

        for i, paper in enumerate(job.state.selected_papers):
            progress = 0.55 + (0.20 * (i / num_papers))
            title_short = paper.title[:40] + "..." if len(paper.title) > 40 else paper.title
            update_progress(job_id, f"Processing paper {i+1}/{num_papers}: {title_short}", progress)

            try:
                logger.info(f"Downloading PDF for paper {paper.paper_id}")
                cls._pdf_processor.process_paper(paper)

                if paper.is_processed and paper.headings:
                    logger.info(f"Extracted {len(paper.headings)} sections from {paper.paper_id}")
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
