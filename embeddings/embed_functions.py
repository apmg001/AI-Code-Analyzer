# embeddings/embed_functions.py

"""
Module: embed_functions

Responsibility
--------------
Generate semantic vector embeddings for code chunks using
a pre-trained sentence-transformer model.

Design notes
------------
- CodeEmbedder is a thin wrapper so the model can be swapped
  (e.g. OpenAI text-embedding-ada-002) without touching any
  other module — Dependency Inversion in practice.
- The model is loaded once in __init__ and reused; loading
  is expensive (~1s) and must not happen per-chunk.
- embed_chunks is a pure function that accepts the embedder
  as a parameter — easy to mock in tests.
- Embeddings are stored inline on the chunk dict so the pipeline
  stays a simple list-of-dicts throughout.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from config import DEFAULT_CONFIG, PipelineConfig
from exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """
    Wrapper around a sentence-transformer embedding model.

    Keeps model loading isolated from embedding logic so either
    part can change independently.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        logger.info("Loading embedding model: %s", config.embedding_model)
        try:
            self._model = SentenceTransformer(config.embedding_model)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model '{config.embedding_model}': {exc}") from exc
        logger.info("Embedding model ready")

    def embed(self, text: str) -> List[float]:
        """
        Encode a text string into a float vector.

        Parameters
        ----------
        text : str
            Raw source code or pattern string.

        Returns
        -------
        List[float]
            Dense embedding vector.

        Raises
        ------
        EmbeddingError
            If the model fails to encode the input.
        """
        try:
            vector = self._model.encode(text, show_progress_bar=False)
            return vector.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Encoding failed: {exc}") from exc


# ------------------------------------------------------------------
# Pipeline functions
# ------------------------------------------------------------------

def embed_chunks(
    chunks:   List[Dict[str, Any]],
    embedder: CodeEmbedder,
) -> List[Dict[str, Any]]:
    """
    Attach embedding vectors to a list of chunks in-place.

    Chunks that fail to embed are logged and skipped rather
    than crashing the whole pipeline.

    Parameters
    ----------
    chunks   : List of chunk dicts (from code_chunker).
    embedder : CodeEmbedder instance.

    Returns
    -------
    List of chunks with `embedding` field populated.
    """

    failed = 0

    for chunk in chunks:
        try:
            chunk["embedding"] = embedder.embed(chunk["code"])
        except EmbeddingError as exc:
            logger.warning(
                "Embedding failed for chunk %s in %s: %s",
                chunk["chunk_id"], chunk["file_path"], exc,
            )
            failed += 1

    succeeded = len(chunks) - failed
    logger.info("Embeddings generated: %d succeeded, %d failed", succeeded, failed)

    return chunks


def save_embeddings(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Persist embedded chunks to disk as JSON.

    Parameters
    ----------
    chunks      : Chunks with embeddings attached.
    output_path : Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info("Embeddings saved to %s", output_path)
