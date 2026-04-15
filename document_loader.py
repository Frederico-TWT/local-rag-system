from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama

CACHE_FILE = Path("data/image_cache.json")
from pypdf import PdfReader

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
VISION_MODEL = "gemma3:4b"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class CollectionPayload:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]


def load_pdf_payload(pdf_dir: Path, prefix: str) -> CollectionPayload:
    """Load PDF files from a directory, chunk them, and return Chroma-ready collection data."""
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        return CollectionPayload(ids=[], documents=[], metadatas=[])

    pdf_files = sorted(f for f in pdf_dir.glob("*.pdf") if f.is_file())

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for file_index, file_path in enumerate(pdf_files, start=1):
        reader = PdfReader(str(file_path))

        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks = chunk_text(page_text.strip())
                for chunk_index, chunk in enumerate(chunks, start=1):
                    ids.append(f"{prefix}-{file_index}-{page_index}-{chunk_index}")
                    documents.append(chunk)
                    metadatas.append(
                        {
                            "source": file_path.name,
                            "page": page_index,
                            "chunk": chunk_index,
                            "type": "pdf",
                        }
                    )

    return CollectionPayload(ids=ids, documents=documents, metadatas=metadatas)


def _file_hash(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def _load_description_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_description_cache(cache: dict[str, str]) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def load_image_payload(image_dir: Path, prefix: str) -> CollectionPayload:
    """Describe images with a vision model, chunk the descriptions, and return Chroma-ready data."""
    if not image_dir.exists() or not image_dir.is_dir():
        return CollectionPayload(ids=[], documents=[], metadatas=[])

    image_files = sorted(
        f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    cache = _load_description_cache()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for file_index, file_path in enumerate(image_files, start=1):
        cache_key = _file_hash(file_path)

        if cache_key in cache:
            print(f"Using cached description for {file_path.name}")
            description = cache[cache_key]
        else:
            print(f"Describing {file_path.name}...")
            image_b64 = base64.b64encode(file_path.read_bytes()).decode()

            response = ollama.chat(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": "Describe this image in detail. Include all text, data, and visual elements.",
                        "images": [image_b64],
                    }
                ],
            )
            description = response.message.content
            cache[cache_key] = description

        chunks = chunk_text(description)
        for chunk_index, chunk in enumerate(chunks, start=1):
            ids.append(f"{prefix}-{file_index}-{chunk_index}")
            documents.append(chunk)
            metadatas.append(
                {
                    "source": file_path.name,
                    "chunk": chunk_index,
                    "type": "image",
                }
            )

    _save_description_cache(cache)
    return CollectionPayload(ids=ids, documents=documents, metadatas=metadatas)


def load_collection_payload(pdf_dir: Path, image_dir: Path, prefix: str) -> CollectionPayload:
    """Load PDFs and images, returning a combined Chroma-ready payload."""
    pdf_payload = load_pdf_payload(pdf_dir, prefix=f"{prefix}-pdf")
    image_payload = load_image_payload(image_dir, prefix=f"{prefix}-img")

    return CollectionPayload(
        ids=pdf_payload.ids + image_payload.ids,
        documents=pdf_payload.documents + image_payload.documents,
        metadatas=pdf_payload.metadatas + image_payload.metadatas,
    )


def chunk_text(text: str) -> list[str]:
    """Chunk the text into smaller pieces with some overlap."""
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind(".")
            if last_period > CHUNK_SIZE * 0.7:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks
