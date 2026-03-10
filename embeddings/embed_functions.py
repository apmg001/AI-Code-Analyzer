# embeddings/embed_functions.py

"""
Module: embed_functions

Purpose
-------
Generate vector embeddings for code chunks.

This module converts code snippets (functions) into vector
representations that can later be used for:

    • semantic search
    • bug pattern detection
    • context retrieval for LLMs

Responsibilities
----------------
1. Load an embedding model
2. Convert code chunks into embeddings
3. Attach embeddings to chunk metadata
4. Provide utilities for saving embeddings
"""

import json
from typing import List, Dict

from sentence_transformers import SentenceTransformer


class CodeEmbedder:
    """
    Wrapper around embedding model.

    Keeps model loading separate from embedding logic.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        print(f"[INFO] Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("[INFO] Model loaded successfully")

    def embed_code(self, code: str) -> List[float]:
        """
        Generate embedding for a single code block.
        """

        vector = self.model.encode(code)

        return vector.tolist()


def embed_chunks(chunks: List[Dict], embedder: CodeEmbedder) -> List[Dict]:
    """
    Generate embeddings for a list of code chunks.

    Parameters
    ----------
    chunks : list[dict]
        Output from code_chunker module
    embedder : CodeEmbedder
        Embedding model wrapper

    Returns
    -------
    list[dict]
        Chunks with attached embedding vectors
    """

    embedded_chunks = []

    for chunk in chunks:

        code_text = chunk["code"]

        embedding_vector = embedder.embed_code(code_text)

        enriched_chunk = dict(chunk)
        enriched_chunk["embedding"] = embedding_vector

        embedded_chunks.append(enriched_chunk)

    return embedded_chunks


def save_embeddings(data: List[Dict], output_path: str):
    """
    Save embedded chunks to disk.

    Parameters
    ----------
    data : list[dict]
    output_path : str
    """

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=2)

    print(f"[INFO] Embeddings saved to {output_path}")


if __name__ == "__main__":

    # Local testing block

    from ingestion.scan_files import scan_python_files
    from parsing.extract_function_code import extract_functions_from_files
    from parsing.code_chunker import chunk_functions

    repo_path = "ingestion/repos/flask"

    print("[INFO] Scanning repository")

    python_files = scan_python_files(repo_path)

    print(f"[INFO] Found {len(python_files)} Python files")

    print("[INFO] Extracting functions")

    functions = extract_functions_from_files(python_files[:5])

    print(f"[INFO] Extracted {len(functions)} functions")

    print("[INFO] Creating chunks")

    chunks = chunk_functions(functions)

    print(f"[INFO] Generated {len(chunks)} chunks")

    embedder = CodeEmbedder()

    print("[INFO] Generating embeddings")

    embedded_chunks = embed_chunks(chunks, embedder)

    print(f"[INFO] Generated embeddings for {len(embedded_chunks)} chunks")

    save_embeddings(embedded_chunks, "function_embeddings.json")

    print("[INFO] Sample embedding vector size:",
          len(embedded_chunks[0]["embedding"]))