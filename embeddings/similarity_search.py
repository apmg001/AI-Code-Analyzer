"""
Module: similarity_search

Purpose
-------
Use embeddings to find code chunks that are
semantically similar to known bad patterns.
"""

import json
import numpy as np
from typing import List, Dict


KNOWN_BAD_PATTERNS = [
    "password = 'hardcoded'",
    "eval(user_input)",
    "while True: pass",
    "except: pass",
    "exec(open('file').read())",
    "os.system(user_input)",
    "SELECT * FROM users WHERE id = ' + user_id",
    "def func(data=[])",          # mutable default argument
    "import *",
    "result = value / 0",
]


class SimilarityDetector:

    def __init__(self, embedder, threshold: float = 0.75):

        self.embedder = embedder
        self.threshold = threshold
        self.bad_pattern_embeddings = self._embed_known_patterns()

    def _embed_known_patterns(self) -> List[Dict]:

        patterns = []

        for pattern in KNOWN_BAD_PATTERNS:

            vector = self.embedder.embed_code(pattern)

            patterns.append({
                "pattern": pattern,
                "embedding": vector
            })

        return patterns

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:

        a = np.array(vec_a)
        b = np.array(vec_b)

        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)

        if norm == 0:
            return 0.0

        return float(dot / norm)

    def find_suspicious_chunks(self, embedded_chunks: List[Dict]) -> List[Dict]:

        issues = []

        for chunk in embedded_chunks:

            chunk_embedding = chunk["embedding"]

            for bad in self.bad_pattern_embeddings:

                score = self._cosine_similarity(chunk_embedding, bad["embedding"])

                if score >= self.threshold:

                    issues.append({
                        "type": "semantic_similarity_flag",
                        "severity": "medium",
                        "function": chunk["function_name"],
                        "file": chunk["file_path"],
                        "chunk_id": chunk["chunk_id"],
                        "matched_pattern": bad["pattern"],
                        "similarity_score": round(score, 4),
                        "message": (
                            f"Code is semantically similar to a known bad pattern. "
                            f"Similarity: {round(score * 100, 1)}%"
                        )
                    })

                    break  # one flag per chunk is enough

        return issues