# embeddings/similarity_search.py

"""
Module: similarity_search

Responsibility
--------------
Use cosine similarity between code embeddings and a curated
library of known-bad patterns to flag semantically suspicious code.

Design notes
------------
- Complements rule-based detection: rules catch exact known patterns,
  similarity search catches unknown variants that look alike.
- Known patterns are embedded once at init — not on every analysis call.
- One flag per chunk: we stop at the first matching pattern to avoid
  flooding the report with redundant issues for the same chunk.
- The similarity_score is included in every issue so developers
  can calibrate the threshold themselves.
- Threshold is configurable from PipelineConfig; 0.75 is conservative.
  Lower it to catch more (more noise); raise it to be stricter.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from config import DEFAULT_CONFIG, PipelineConfig
from embeddings.embed_functions import CodeEmbedder
from exceptions import EmbeddingError

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Known bad pattern library
# ------------------------------------------------------------------

# Each string represents a common anti-pattern or security risk.
# The embedding model maps these into the same semantic space as
# real code, so similar-looking code will cluster nearby.

KNOWN_BAD_PATTERNS: List[Tuple[str, str]] = [
    # (pattern_code,                              pattern_label)
    ("eval(user_input)",                          "unsafe_eval"),
    ("exec(open('script.py').read())",            "unsafe_exec"),
    ("os.system(user_input)",                     "command_injection"),
    ("password = 'hardcoded_secret'",             "hardcoded_credential"),
    ("api_key = 'sk-hardcoded'",                  "hardcoded_api_key"),
    ("SELECT * FROM users WHERE id = ' + id",     "sql_injection"),
    ("def func(items=[]): items.append(x)",       "mutable_default_arg"),
    ("except: pass",                              "bare_except_swallow"),
    ("result = x / 0",                            "division_by_zero"),
    ("while True: pass",                          "infinite_loop_no_exit"),
    ("import *",                                  "wildcard_import"),
    ("pickle.loads(user_data)",                   "unsafe_deserialization"),
    ("subprocess.call(shell=True)",               "shell_injection"),
    ("open(filename, 'rb').read()",               "unclosed_file_handle"),
    ("assert x == y",                             "assert_in_production"),
]


# ------------------------------------------------------------------
# Detector
# ------------------------------------------------------------------

class SimilarityDetector:
    """
    Detects code chunks that are semantically similar to
    known-bad patterns using cosine similarity.
    """

    def __init__(
        self,
        embedder:  CodeEmbedder,
        config:    PipelineConfig = DEFAULT_CONFIG,
    ):
        self._embedder  = embedder
        self._threshold = config.similarity_threshold
        self._patterns  = self._embed_known_patterns()

        logger.info(
            "SimilarityDetector ready — %d patterns embedded, threshold=%.2f",
            len(self._patterns), self._threshold,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _embed_known_patterns(self) -> List[Dict[str, Any]]:
        """
        Pre-embed all known bad patterns.
        Called once at construction; results are cached on the instance.
        """
        embedded = []

        for pattern_code, label in KNOWN_BAD_PATTERNS:
            try:
                vector = self._embedder.embed(pattern_code)
                embedded.append({
                    "label":     label,
                    "pattern":   pattern_code,
                    "embedding": np.array(vector),
                })
            except EmbeddingError as exc:
                logger.warning("Could not embed pattern '%s': %s", label, exc)

        return embedded

    # ------------------------------------------------------------------
    # Core similarity logic
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Returns 0.0 if either vector has zero magnitude.
        """
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / norm)

    def _best_match(
        self,
        chunk_vector: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Find the known-bad pattern most similar to the given vector.
        Returns (score, pattern_record).
        """
        best_score   = -1.0
        best_pattern = self._patterns[0]

        for pattern in self._patterns:
            score = self._cosine_similarity(chunk_vector, pattern["embedding"])
            if score > best_score:
                best_score   = score
                best_pattern = pattern

        return best_score, best_pattern

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_suspicious_chunks(
        self,
        embedded_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Scan all chunks and return issues for those that exceed
        the similarity threshold against any known-bad pattern.

        Parameters
        ----------
        embedded_chunks : Chunks with `embedding` field populated.

        Returns
        -------
        List of DetectedIssue dicts — one per suspicious chunk at most.
        """
        issues = []

        for chunk in embedded_chunks:
            if not chunk.get("embedding"):
                continue

            chunk_vector        = np.array(chunk["embedding"])
            score, best_pattern = self._best_match(chunk_vector)

            if score < self._threshold:
                continue

            issue = {
                "type":             "semantic_similarity_flag",
                "severity":         "medium",
                "function":         chunk["function_name"],
                "file":             chunk["file_path"],
                "line_number":      chunk["start_line"],
                "code_snippet":     chunk["code"][:120].replace("\n", " "),
                "chunk_id":         chunk["chunk_id"],
                "matched_pattern":  best_pattern["pattern"],
                "pattern_label":    best_pattern["label"],
                "similarity_score": round(score, 4),
                "message": (
                    f"Code is semantically similar to known anti-pattern "
                    f"'{best_pattern['label']}' "
                    f"(similarity: {round(score * 100, 1)}%). "
                    f"Manual review recommended."
                ),
            }

            issues.append(issue)

            logger.debug(
                "Flagged %s in %s — pattern: %s, score: %.3f",
                chunk["function_name"], chunk["file_path"],
                best_pattern["label"], score,
            )

        logger.info("Semantic scan complete — %d suspicious chunks found", len(issues))

        return issues
