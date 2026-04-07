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
Character-based chunker with proper overlap across pages.
    - Guarantees forward progress
    - Overlaps between pages (important for continuity)
    - Preserves original page metadata
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    previous_overlap_text = ""   # Carry over text from previous page for overlap

    for page in pages:
        text = (page.get("text") or "").strip()
        meta = page.get("metadata") or {}

        if not text:
            continue

        # Add overlap from previous page at the beginning
        full_text = previous_overlap_text + " " + text if previous_overlap_text else text

        n = len(full_text)
        start = 0

        while start < n:
            end = min(start + chunk_size, n)
            chunk_text = full_text[start:end].strip()

            if chunk_text:
                # Create metadata for this chunk
                chunk_meta = dict(meta)
                chunk_meta["chunk_start"] = start
                chunk_meta["chunk_end"] = end
                chunk_meta["chunk_length"] = len(chunk_text)

                chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))

            if end == n:
                break

            # Move forward with overlap
            start = max(0, end - chunk_overlap)

        # Save the last 'chunk_overlap' characters for the next page
        previous_overlap_text = full_text[max(0, n - chunk_overlap):]

    logger.info(f"Created {len(chunks)} chunks with overlap={chunk_overlap}")
    return chunks