from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import ollama

from document_loader import load_collection_payload

PDF_DIR = Path("data/pdfs")
IMAGE_DIR = Path("data/images")
COLLECTION_NAME = "rag_collection"

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "gemma3:4b"

def get_embedding(text: str) -> list[float]:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response.embeddings[0]


def get_embedding_batch(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response.embeddings


def main(question: str) -> str:
    payload = load_collection_payload(PDF_DIR, IMAGE_DIR, prefix="doc")

    if not payload.documents:
        raise FileNotFoundError(
            f"No documents found. Add PDFs to {PDF_DIR.resolve()} or images to {IMAGE_DIR.resolve()}."
        )

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    document_embeddings = get_embedding_batch(payload.documents)

    collection.upsert(
        ids=payload.ids,
        documents=payload.documents,
        embeddings=document_embeddings,
        metadatas=payload.metadatas,
    )

    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, len(payload.documents)),
    )

    # Build context from retrieved chunks
    context_parts = []
    for doc, meta in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
    ):
        source = (meta or {}).get("source", "unknown")
        doc_type = (meta or {}).get("type", "unknown")
        page = (meta or {}).get("page", "")
        page_info = f", Page {page}" if page else ""
        context_parts.append(f"[Source: {source} ({doc_type}){page_info}]\n{doc}")

    context = "\n\n".join(context_parts)

    # Augmented prompt
    prompt = f"""Use the following context from documents and images to answer the question.
If the context doesn't contain enough information, say so.
Keep your answer concise and grounded in the provided context.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.message.content


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    answer = main(query)
    print(f"Question: {query}\n")
    print(f"Answer: {answer}")
