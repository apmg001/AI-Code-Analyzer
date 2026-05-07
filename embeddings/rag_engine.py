# embeddings/rag_engine.py

"""
Module: rag_engine

Responsibility
--------------
Store code chunk embeddings in ChromaDB and retrieve
semantically similar chunks on demand for RAG-augmented
patch generation.

What RAG adds to this pipeline
-------------------------------
Without RAG:
    Issue detected → LLM sees buggy code + issue description
                   → generic fix suggestion

With RAG:
    Issue detected → retrieve 3 most similar functions from
                     the same repo (via ChromaDB HNSW search)
                   → LLM sees buggy code + issue + style examples
                   → fix that matches THIS codebase's conventions

Why ChromaDB over the existing JSON approach
--------------------------------------------
Current similarity_search.py stores known-bad patterns in memory
and runs O(n×m) cosine similarity on every call.

ChromaDB builds an HNSW (Hierarchical Navigable Small World)
graph index — approximate nearest neighbor search in O(log n).

For 356 chunks the difference is small. For 10,000 chunks across
10 repos — it matters significantly and the index persists to disk
so it is not rebuilt on every run.

Design decisions
----------------
- One ChromaDB collection per repository, named from the repo slug.
  Multiple repos coexist in the same ChromaDB instance without
  collision.

- PersistentClient writes the index to chroma_db/ on disk.
  Second run on the same repo skips re-indexing entirely —
  chunks are checked for existence before adding.

- is_available degrades gracefully when chromadb is not installed.
  The entire pipeline continues — patches just use the standard
  prompt instead of the RAG-augmented one.

- build_rag_prompt is the single place where style context is
  added to the LLM prompt. Prompt changes only need to happen here.

- retrieve_similar excludes the query chunk itself from results
  (it would always be the nearest neighbour to itself).

Limitations
-----------
- Only indexes chunks with an embedding field populated.
- Collection name is derived from the repo URL last segment —
  repos with the same name from different owners will collide.
  For a multi-user deployment, hash the full URL instead.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Vector store and retrieval engine for code chunks.

    Stores embeddings in ChromaDB for persistent, fast semantic
    search. Provides style-aware prompt augmentation for LLM
    patch generation.

    Usage
    -----
    engine = RAGEngine(repo_url="https://github.com/pallets/flask")
    engine.index(embedded_chunks)                    # one-time indexing

    similar = engine.retrieve_similar(chunk, top_k=3)
    prompt  = engine.build_rag_prompt(issue, chunk, similar)
    """

    def __init__(
        self,
        repo_url:    str,
        persist_dir: Path = Path("chroma_db"),
    ):
        self._repo_url    = repo_url
        self._persist_dir = persist_dir
        self._collection  = None
        self._available   = self._setup()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _setup(self) -> bool:
        """
        Connect to ChromaDB and get or create a collection for this repo.
        Returns False if chromadb is not installed so callers can degrade
        gracefully rather than crashing.
        """
        try:
            import chromadb

            self._persist_dir.mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(
                path=str(self._persist_dir)
            )

            collection_name = self._derive_collection_name(self._repo_url)

            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "RAG engine ready — collection '%s' has %d chunks indexed",
                collection_name,
                self._collection.count(),
            )
            return True

        except ImportError:
            logger.warning(
                "chromadb not installed — RAG-augmented prompts disabled. "
                "Install with: pip install chromadb"
            )
            return False

        except Exception as exc:
            logger.warning("RAG engine setup failed (non-fatal): %s", exc)
            return False

    @staticmethod
    def _derive_collection_name(repo_url: str) -> str:
        """
        Derive a valid ChromaDB collection name from a repository URL.

        ChromaDB requires: 3-63 chars, alphanumeric + hyphens, no dots.
        Example: "https://github.com/pallets/flask" → "flask"
        """
        slug = repo_url.rstrip("/").split("/")[-1].lower()
        slug = "".join(c if c.isalnum() else "-" for c in slug)
        slug = slug.strip("-")[:40]
        return slug or "default"

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Store embedded chunks in ChromaDB.

        Checks which chunks are already indexed and only adds new ones,
        so this method is safe to call on every pipeline run without
        duplicating data.

        Parameters
        ----------
        chunks : List of chunk dicts with 'embedding' field populated.

        Returns
        -------
        Number of new chunks added to the index.
        """
        if not self._available:
            return 0

        valid = [c for c in chunks if c.get("embedding")]
        if not valid:
            logger.debug("RAG: no chunks with embeddings to index")
            return 0

        # Check which chunk IDs are already present
        existing_ids: set = set()
        try:
            result = self._collection.get(
                ids=[c["chunk_id"] for c in valid],
                include=[],
            )
            existing_ids = set(result["ids"])
        except Exception:
            pass   # empty collection — all chunks are new

        new_chunks = [c for c in valid if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            logger.info(
                "RAG: all %d chunks already indexed — skipping",
                len(valid),
            )
            return 0

        self._collection.add(
            ids=[c["chunk_id"] for c in new_chunks],
            embeddings=[c["embedding"] for c in new_chunks],
            documents=[c["code"] for c in new_chunks],
            metadatas=[{
                "function": c["function_name"],
                "file":     c["file_path"],
                "start":    str(c.get("start_line", 0)),
            } for c in new_chunks],
        )

        logger.info(
            "RAG: indexed %d new chunks (%d total in collection)",
            len(new_chunks),
            self._collection.count(),
        )
        return len(new_chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_similar(
        self,
        chunk: Dict[str, Any],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find the most semantically similar functions to the given chunk.

        Uses ChromaDB's HNSW index — O(log n) vs O(n) linear scan.
        Excludes the query chunk itself from results.

        Parameters
        ----------
        chunk : The chunk to find similar code for.
        top_k : Number of similar results to return.

        Returns
        -------
        List of similar chunk dicts sorted by similarity descending.
        Empty list if RAG is unavailable or retrieval fails.
        """
        if not self._available:
            return []

        embedding = chunk.get("embedding")
        if not embedding:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k + 1, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return []

        similar = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Skip the chunk itself — it is always its own nearest neighbour
            if meta.get("function") == chunk["function_name"]:
                continue

            similar.append({
                "code":       doc,
                "function":   meta["function"],
                "file":       meta["file"].split("/")[-1],
                "similarity": round(1 - dist, 3),
            })

        return similar[:top_k]

    # ------------------------------------------------------------------
    # Prompt augmentation
    # ------------------------------------------------------------------

    def build_rag_prompt(
        self,
        issue:          Dict[str, Any],
        chunk:          Dict[str, Any],
        similar_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build a RAG-augmented prompt that includes style examples
        from the same codebase.

        The key insight: when the LLM sees how similar functions are
        written in THIS repo — their error handling, naming, logging
        patterns — it produces fixes that fit naturally into the
        existing codebase rather than generic suggestions.

        Parameters
        ----------
        issue          : Detected issue dict.
        chunk          : The chunk containing the bug.
        similar_chunks : Retrieved similar functions (from retrieve_similar).

        Returns
        -------
        Prompt string ready to send to any LLM provider.
        """
        # Build the style-reference section
        if similar_chunks:
            style_parts = []
            for i, sc in enumerate(similar_chunks, 1):
                style_parts.append(
                    f"Example {i} — {sc['function']}() "
                    f"in {sc['file']} "
                    f"(similarity: {sc['similarity']:.0%}):\n"
                    f"```python\n{sc['code'][:400]}\n```"
                )

            style_context = (
                "\n\nSIMILAR FUNCTIONS FROM THIS REPO "
                "(write the fix in the same style as these):\n\n"
                + "\n\n".join(style_parts)
                + "\n"
            )
        else:
            style_context = ""

        return (
            f"You are a senior Python engineer fixing a specific bug.\n\n"
            f"BUG TYPE : {issue['type']}\n"
            f"SEVERITY : {issue['severity']}\n"
            f"MESSAGE  : {issue['message']}\n"
            f"FILE     : {chunk.get('file_path', '').split('/')[-1]}\n"
            f"FUNCTION : {chunk.get('function_name', '')}\n\n"
            f"BUGGY CODE:\n"
            f"```python\n{chunk['code']}\n```"
            f"{style_context}\n"
            f"RULES:\n"
            f"1. Return ONLY the fixed Python code\n"
            f"2. Add ONE comment above each change explaining what you fixed\n"
            f"3. Keep all existing logic — only fix the specific bug\n"
            f"4. Match the style of the similar functions shown above\n"
            f"5. If a new import is needed, add it at the top\n\n"
            f"FIXED CODE:"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if ChromaDB is installed and the collection is ready."""
        return self._available

    @property
    def indexed_count(self) -> int:
        """Number of chunks currently stored in the collection."""
        if not self._available:
            return 0
        return self._collection.count()