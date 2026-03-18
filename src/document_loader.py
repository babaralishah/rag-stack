# Reads PDF files and extracts text content along with metadata for each page. Uses pypdf for PDF parsing.

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)

@dataclass
class DocumentPage:
    text: str
    metadata: Dict[str, Any]

def load_pdf(path: str) -> List[DocumentPage]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", pdf_path)

    reader = PdfReader(str(pdf_path))
    pages: List[DocumentPage] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.exception("Failed extracting page %d: %s", i + 1, e)
            text = ""

        text = text.strip()
        if not text:
            # Keep empty pages out for now (common in scanned PDFs)
            continue

        pages.append(
            DocumentPage(
                text=text,
                metadata={
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path.resolve()),
                    "page": i + 1,  # 1-indexed
                },
            )
        )

    logger.info("Loaded %d text pages from %s", len(pages), pdf_path.name)
    return pages