import os
import hashlib
from pypdf import PdfReader
from src.api import ingest_upload_pages  # Import your system's built-in engine

PDF_PATH = "data source.pdf"

if not os.path.exists(PDF_PATH):
    print(f"❌ Error: Cannot find '{PDF_PATH}' in the current root folder!")
    exit()

print(f"📖 Reading pages from '{PDF_PATH}'...")
reader = PdfReader(PDF_PATH)
parsed_pages = []

# Extract pages into the exact data structure your API function expects
for idx, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    parsed_pages.append({
        "text": text,
        "metadata": {"page": idx + 1}
    })

# Compute a file hash to satisfy your database structure requirements
with open(PDF_PATH, "rb") as f:
    file_hash = hashlib.md5(f.read()).hexdigest()

print(f"⚡ Processing and embedding {len(parsed_pages)} pages into FAISS storage...")
try:
    result = ingest_upload_pages(
        pages=parsed_pages,
        file_hash=file_hash,
        filename=os.path.basename(PDF_PATH),
        file_ext=".pdf"
    )
    print("\n" + "="*45)
    print(f"✅ SUCCESS! {result['chunks_added']} document chunks added to your database.")
    print("="*45)
except Exception as e:
    print(f"❌ Ingestion failed: {e}")
