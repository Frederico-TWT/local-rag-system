# Local RAG System

A simple local Retrieval-Augmented Generation (RAG) system that answers questions about PDF documents and images using Ollama for embeddings and LLM inference.

## How It Works

1. PDFs in `data/pdfs/` are loaded, extracted, and chunked
2. Images in `data/images/` are described by a vision model (`gemma3:4b`), then chunked
3. All chunks are embedded using `mxbai-embed-large` via Ollama and stored in an in-memory ChromaDB vector database
4. Question is embedded with the same model and used to retrieve the most relevant chunks
5. An augmented prompt is built with the retrieved context and sent to a local LLM (`gemma3:4b`) via Ollama
6. The LLM response is returned

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installed and running

## Setup

1. Pull the required Ollama models:

```sh
ollama pull mxbai-embed-large
ollama pull gemma3:4b
```

2. Install dependencies:

```sh
uv sync
```

3. Add PDF files to `data/pdfs/` and/or images to `data/images/`.

## Usage

```sh
uv run main.py "What is the number on the closet?"

-> Answer: The numbers on the display are "27 28 29 30 31".
```