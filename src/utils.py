import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest for a file at `path`.

    Reads the file in 1MB blocks to avoid high memory usage for large files.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
