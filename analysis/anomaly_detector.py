# analysis/anomaly_detector.py

"""
Module: anomaly_detector

Responsibility
--------------
Learn what "normal" code looks like in a specific codebase
and flag functions that are statistically unusual.

Why this matters
----------------
Rule-based tools can only find bugs they were programmed to find.
Anomaly detection finds bugs that have no known pattern — functions
that look statistically different from everything else in the repo.

Common anomalies caught:
- A function 10x more complex than everything around it
- A function that touches resources no other function touches
- A function whose embedding is far from all its neighbors
- Dead code that was never updated when the codebase evolved

Algorithm
---------
IsolationForest — an unsupervised machine learning algorithm that
works by randomly partitioning the embedding space. Points that
are isolated quickly (require fewer partitions to isolate) are
anomalies. It runs in O(n log n) and works well on high-dimensional
data like embeddings.

Design notes
------------
- fit() learns the codebase's "normal" — call this first
- find_anomalies() scores every chunk — call this after fit()
- contamination=0.05 means "expect ~5% of code to be anomalous"
- Only available when scikit-learn is installed — degrades gracefully
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects statistically unusual functions using IsolationForest.

    This is the only component in the pipeline that learns from
    the specific codebase being analyzed — making it adaptive
    rather than static.
    """

    def __init__(self, contamination: float = 0.05):
        """
        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies in the codebase.
            0.05 = flag the most unusual 5% of functions.
            Lower = fewer flags. Higher = more flags.
        """
        self._contamination = contamination
        self._model         = None
        self._fitted        = False
        self._available     = self._check_sklearn()

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sklearn() -> bool:
        """Return True if scikit-learn is installed."""
        try:
            import sklearn
            return True
        except ImportError:
            logger.warning(
                "scikit-learn not installed — anomaly detection disabled. "
                "Install with: pip install scikit-learn"
            )
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, embedded_chunks: List[Dict[str, Any]]) -> "AnomalyDetector":
        """
        Learn the normal distribution of code in this codebase.

        Must be called before find_anomalies().
        Requires at least 10 chunks to be meaningful.

        Parameters
        ----------
        embedded_chunks : List of chunks with embedding vectors attached.
        """
        if not self._available:
            return self

        import numpy as np
        from sklearn.ensemble import IsolationForest

        vectors = self._extract_vectors(embedded_chunks)

        if len(vectors) < 10:
            logger.info(
                "Too few chunks (%d) for anomaly detection — skipping",
                len(vectors),
            )
            return self

        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1,      # use all CPU cores
        )
        self._model.fit(vectors)
        self._fitted = True

        logger.info(
            "AnomalyDetector fitted on %d chunks (contamination=%.2f)",
            len(vectors), self._contamination,
        )

        return self

    def find_anomalies(
        self,
        embedded_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Score all chunks and return issues for anomalous ones.

        Parameters
        ----------
        embedded_chunks : List of chunks with embedding vectors.

        Returns
        -------
        List of issue dicts for statistically anomalous functions.
        """
        if not self._available or not self._fitted:
            return []

        import numpy as np

        vectors = self._extract_vectors(embedded_chunks)

        if len(vectors) == 0:
            return []

        # -1 = anomaly, 1 = normal
        predictions = self._model.predict(vectors)

        # Raw anomaly score — more negative = more anomalous
        scores = self._model.score_samples(vectors)

        issues = []
        valid_chunks = [c for c in embedded_chunks if c.get("embedding")]

        for chunk, pred, score in zip(valid_chunks, predictions, scores):
            if pred != -1:      # normal — skip
                continue

            # Normalize score to 0-1 for readability
            # score_samples returns negative values — more negative = more anomalous
            normalized = max(0.0, min(1.0, 1.0 - (score + 0.5) / 0.5))

            issues.append({
                "type":          "statistical_anomaly",
                "severity":      "medium",
                "confidence":    round(normalized, 3),
                "function":      chunk["function_name"],
                "file":          chunk["file_path"],
                "line_number":   chunk.get("start_line", 0),
                "code_snippet":  chunk["code"][:120].replace("\n", " "),
                "chunk_id":      chunk["chunk_id"],
                "anomaly_score": round(float(score), 4),
                "similarity_score": None,
                "message": (
                    f"'{chunk['function_name']}' is statistically unusual "
                    f"compared to the rest of this codebase "
                    f"(anomaly score: {round(score, 3)}). "
                    f"This function does not match the embedding patterns "
                    f"of surrounding code — manual review recommended. "
                    f"Common causes: dead code, copy-paste from external source, "
                    f"undocumented special case, or hidden complexity."
                ),
            })

        logger.info(
            "Anomaly detection complete — %d anomalous functions found "
            "out of %d analyzed",
            len(issues), len(valid_chunks),
        )

        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_vectors(chunks: List[Dict[str, Any]]):
        """Extract numpy matrix from chunk embeddings."""
        import numpy as np

        vectors = [
            c["embedding"] for c in chunks
            if c.get("embedding") is not None
        ]

        if not vectors:
            return np.empty((0, 0))

        return np.array(vectors, dtype=np.float32)
