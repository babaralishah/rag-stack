import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

import requests

logger = logging.getLogger("rag")

YOUTUBE_VIDEO_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|youtube\.com/(?:embed|v)/)([\w-]{11})"
)


def normalize_text(text: str) -> str:
    """Normalize whitespace in extracted text into single spaces.

    This makes downstream chunking and embedding more consistent.
    """
    return " ".join(text.split())


def fetch_web_text(url: str) -> List[Dict[str, Any]]:
    """Fetch and extract main textual content from an HTML page.

    - Removes common boilerplate tags (scripts, headers, footers).
    - Returns a single page dict with combined text and basic metadata.
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    resp = requests.get(url, timeout=20, headers=headers)
    resp.raise_for_status()

    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as exc:
        raise ValueError(
            "beautifulsoup4 is required to ingest web pages. Install it with pip."
        ) from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    for tag in soup(
        ["script", "style", "header", "footer", "nav", "aside", "form", "noscript"]
    ):
        tag.decompose()

    body = soup.body or soup
    paragraphs = body.find_all(["p", "h1", "h2", "h3", "h4", "li"])
    text_blocks = [normalize_text(p.get_text(" ", strip=True)) for p in paragraphs]
    text_blocks = [blk for blk in text_blocks if len(blk) > 20]

    if not text_blocks:
        text_blocks = [normalize_text(body.get_text(" ", strip=True))]

    combined = "\n\n".join(text_blocks)
    if len(combined) < 50:
        raise ValueError(
            "Unable to extract meaningful text from URL. Try a different page."
        )

    return [
        {
            "text": combined,
            "metadata": {"title": title, "source_url": url, "source_type": "web"},
        }
    ]


def extract_youtube_id(url: str) -> str:
    """Extract an 11-char YouTube video id from common URL patterns.

    Raises ValueError for unrecognized URLs.
    """

    match = YOUTUBE_VIDEO_ID_RE.search(url)
    if match:
        return match.group(1)

    # Fallback for full watch URLs with query param
    if "watch" in url and "v=" in url:
        parts = url.split("v=")
        if len(parts) > 1:
            return parts[1].split("&")[0]

    raise ValueError(
        "Invalid YouTube URL. Please provide a standard YouTube watch or short link."
    )


def fetch_youtube_transcript(url: str) -> List[Dict[str, Any]]:
    """Fetch a YouTube transcript using `youtube-transcript-api`.

    - Handles common transcript errors and returns a single page dict.
    """

    video_id = extract_youtube_id(url)

    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
            RequestBlocked,
        )
    except ModuleNotFoundError as exc:
        raise ValueError(
            "youtube-transcript-api is required to ingest YouTube transcripts. Install it with pip."
        ) from exc

    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id, languages=["en", "en-US", "en-GB"], preserve_formatting=False
        )
    except RequestBlocked:
        raise ValueError(
            "YouTube blocked the transcript request from this environment. "
            "Cloud-hosted IPs are often blocked by YouTube, so use a local proxy, a different network, or upload a transcript manually."
        )
    except TranscriptsDisabled:
        raise ValueError("Transcript is disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No transcript found for this video.")
    except Exception as exc:
        logger.exception("YouTube transcript fetch failed: %s", exc)
        raise ValueError(str(exc) or "Failed to fetch transcript from YouTube.")

    transcript_text = "\n".join(
        [normalize_text(item["text"]) for item in transcript if item.get("text")]
    )
    if len(transcript_text) < 50:
        raise ValueError("Transcript content is too short to index.")

    return [
        {
            "text": transcript_text,
            "metadata": {
                "source_url": url,
                "video_id": video_id,
                "source_type": "youtube",
            },
        }
    ]


def load_sqlite_table(
    db_path: str, table_name: str = "user_history"
) -> List[Dict[str, Any]]:
    """Load all rows from a SQLite table and convert each row to a page dict.

    Each row is formatted as "col: value" lines to make the content
    searchable and indexable by the RAG pipeline.
    """

    if not db_path:
        raise ValueError("SQLite database path is required.")

    db_file = Path(db_path)
    if not db_file.exists():
        raise ValueError("SQLite database file not found.")

    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()

        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        except sqlite3.OperationalError as exc:
            raise ValueError(
                f"Table '{table_name}' does not exist in the database."
            ) from exc

        columns = [col[0] for col in cursor.description]
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

    if not rows:
        raise ValueError(f"Table '{table_name}' is empty.")

    pages = []
    for idx, row in enumerate(rows):
        row_text = "\n".join(f"{col}: {row[i]}" for i, col in enumerate(columns))
        pages.append(
            {
                "text": row_text,
                "metadata": {
                    "source_type": "sqlite",
                    "table_name": table_name,
                    "row_index": idx,
                    "columns": columns,
                },
            }
        )

    return pages


def get_sqlite_table_names(db_path: str) -> List[str]:
    """Return a list of non-internal table names from a SQLite database.

    Raises ValueError if no tables found or database is missing.
    """

    if not db_path:
        raise ValueError("SQLite database path is required.")

    db_file = Path(db_path)
    if not db_file.exists():
        raise ValueError("SQLite database file not found.")

    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        raise ValueError("No tables found in the SQLite database.")

    return tables
