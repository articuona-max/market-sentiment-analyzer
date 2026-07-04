"""
Chunked PDF Text Extraction Worker.

Streaming page-by-page extraction using PyMuPDF (fitz) to prevent
memory leaks on large financial PDFs. Splits extracted text into
configurable-size chunks with overlap for embedding continuity.

Outputs PDFContext objects ready for vector storage and fusion.
"""
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, BinaryIO

import fitz  # PyMuPDF

from src.database.models import PDFContext

logger = logging.getLogger(__name__)


class PDFWorker:
    """
    Chunked PDF processor for financial document extraction.

    Extracts text page-by-page (streaming) and splits into overlapping
    chunks to preserve context boundaries for downstream embedding.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            chunk_size: Maximum characters per text chunk.
            chunk_overlap: Overlap between consecutive chunks to preserve
                          semantic continuity across chunk boundaries.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and < chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _generate_chunk_id(doc_title: str, page: int, chunk_idx: int) -> str:
        """Generates a deterministic chunk ID from document metadata."""
        raw = f"{doc_title}:p{page}:c{chunk_idx}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _split_text(self, text: str) -> List[str]:
        """
        Splits text into overlapping chunks.

        Uses a sliding window approach: each chunk contains up to
        chunk_size characters, and consecutive chunks overlap by
        chunk_overlap characters.
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Advance by (chunk_size - overlap) to create overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def extract_from_path(
        self,
        pdf_path: str,
        document_title: Optional[str] = None,
        published_at: Optional[datetime] = None,
        entity_tags: Optional[List[str]] = None,
    ) -> List[PDFContext]:
        """
        Extracts text from a PDF file path and returns chunked PDFContexts.

        Args:
            pdf_path: Filesystem path to the PDF.
            document_title: Human-readable title (defaults to filename).
            published_at: Publication date (defaults to current UTC time).
            entity_tags: Entity tags to associate with each chunk.

        Returns:
            List of PDFContext objects, one per chunk.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if document_title is None:
            document_title = path.stem

        if published_at is None:
            published_at = datetime.now(timezone.utc)

        return self._process_document(
            doc=fitz.open(str(path)),
            document_title=document_title,
            published_at=published_at,
            entity_tags=entity_tags or [],
        )

    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        document_title: str = "uploaded_document",
        published_at: Optional[datetime] = None,
        entity_tags: Optional[List[str]] = None,
    ) -> List[PDFContext]:
        """
        Extracts text from in-memory PDF bytes.

        Used for PDF uploads via the API where the file is received
        as a byte stream rather than a filesystem path.

        Args:
            pdf_bytes: Raw PDF file bytes.
            document_title: Human-readable title for provenance.
            published_at: Publication date (defaults to current UTC time).
            entity_tags: Entity tags to associate with each chunk.

        Returns:
            List of PDFContext objects, one per chunk.
        """
        if published_at is None:
            published_at = datetime.now(timezone.utc)

        return self._process_document(
            doc=fitz.open(stream=pdf_bytes, filetype="pdf"),
            document_title=document_title,
            published_at=published_at,
            entity_tags=entity_tags or [],
        )

    def _process_document(
        self,
        doc: fitz.Document,
        document_title: str,
        published_at: datetime,
        entity_tags: List[str],
    ) -> List[PDFContext]:
        """
        Core extraction logic: iterates pages, extracts text, chunks it.

        Processes page-by-page to keep memory usage flat even for
        very large financial PDFs (streaming pattern).
        """
        all_contexts: List[PDFContext] = []
        global_chunk_idx = 0

        try:
            logger.info(
                f"Processing PDF '{document_title}': {len(doc)} pages"
            )

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")

                if not page_text or not page_text.strip():
                    logger.debug(
                        f"Page {page_num + 1} of '{document_title}' has no text, skipping."
                    )
                    continue

                chunks = self._split_text(page_text)

                for chunk_text in chunks:
                    chunk_id = self._generate_chunk_id(
                        document_title, page_num + 1, global_chunk_idx
                    )

                    ctx = PDFContext(
                        id=chunk_id,
                        document_title=document_title,
                        extracted_text=chunk_text,
                        published_at=published_at,
                        page_number=page_num + 1,
                        chunk_index=global_chunk_idx,
                        entity_tags=entity_tags,
                    )
                    all_contexts.append(ctx)
                    global_chunk_idx += 1

            logger.info(
                f"Extracted {global_chunk_idx} chunks from "
                f"'{document_title}' ({len(doc)} pages)."
            )
        finally:
            doc.close()

        return all_contexts
