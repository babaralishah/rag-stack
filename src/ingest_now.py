import os
import hashlib
from src.api import ingest_upload_pages  # Import your system's built-in engine

# Update this to your actual text filename (e.g., "sciq_sources.txt")
# TXT_PATH = "hotpotqa_mini_sources" 
# Replace line 6 with this:
TXT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hotpotqa_mini_sources")


if not os.path.exists(TXT_PATH):
    print(f"❌ Error: Cannot find '{TXT_PATH}' in the current root folder!")
    exit()

print(f"📖 Reading text blocks from '{TXT_PATH}'...")

# Read the full text file
with open(TXT_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

# Split the text by paragraphs or double newlines to simulate "pages"
# This aligns perfectly with the way we saved SciQ using "=\n\n" separators earlier
raw_blocks = full_text.split("="*50)

parsed_pages = []
block_idx = 1

for block in raw_blocks:
    clean_text = block.strip()
    if clean_text:  # Only ingest blocks that actually contain text
        parsed_pages.append({
            "text": clean_text,
            "metadata": {"page": block_idx}  # Simulating a page number using block index
        })
        block_idx += 1

# Compute a file hash to satisfy your database structure requirements
with open(TXT_PATH, "rb") as f:
    file_hash = hashlib.md5(f.read()).hexdigest()

print(f"⚡ Processing and embedding {len(parsed_pages)} text blocks into FAISS storage...")
try:
    result = ingest_upload_pages(
        pages=parsed_pages,
        file_hash=file_hash,
        filename=os.path.basename(TXT_PATH),
        file_ext=".txt"
    )
    print("\n" + "="*45)
    print(f"✅ SUCCESS! {result.get('chunks_added', 'All')} document chunks added to your database.")
    print("="*45)
except Exception as e:
    print(f"❌ Ingestion failed: {e}")
