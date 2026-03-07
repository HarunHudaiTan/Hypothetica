import logging
from typing import Callable

from api.managers.job_manager import job_manager

from processing.pdf_processor import PDFProcessor
from processing.chunk_processor import ChunkProcessor
from rag.chroma_store import ChromaStore

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

        total_chunks = 0
        num_papers = len(job.state.selected_papers)

        for i, paper in enumerate(job.state.selected_papers):
            progress = 0.55 + (0.20 * (i / num_papers))
            update_progress(job_id, f"Processing paper {i+1}/{num_papers}: {paper.title[:40]}...", progress)

            try:
                cls._pdf_processor.process_paper(paper)
                if paper.is_processed and paper.headings:
                    cls._chunk_processor.process_paper(paper)
                    chunks_added = store.add_paper(paper)
                    total_chunks += chunks_added
            except Exception as e:
                logger.error(f"Error processing paper {paper.paper_id}: {e}")
                paper.processing_error = str(e)

        update_progress(job_id, f"Indexed {total_chunks} total chunks", 0.75)
