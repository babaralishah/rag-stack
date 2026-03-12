# Splits text into chunks of a specified size with optional overlap. This is a simple character-based chunker that guarantees forward progress and won't loop forever. 
# Each chunk retains metadata from the original page for later reference.



from dataclasses import dataclass
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


def chunk_text(
    pages: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Chunk]:
    """
    Simple, safe character-based chunking.
    - chunk_size: characters per chunk
    - chunk_overlap: characters of overlap between chunks

    This version guarantees forward progress and will not loop forever.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks: List[Chunk] = []

    for page in pages:
        text = (page.get("text") or "").strip()
        meta = page.get("metadata") or {}

        if not text:
            continue

        n = len(text)
        start = 0

        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(Chunk(text=chunk, metadata=dict(meta)))

            if end == n:
                break  # reached end of this page text

            # guaranteed forward progress
            start = max(0, end - chunk_overlap)

    logger.info("Created %d chunks", len(chunks))
    return chunks