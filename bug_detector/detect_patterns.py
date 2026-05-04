# bug_detector/detect_patterns.py

"""
Module: detect_patterns

Responsibility
--------------
Detect risky patterns in code chunks using two strategies:

1. Rule-based  — Fast, deterministic AST/regex checks.
2. Semantic    — Embedding-based similarity search.

Every issue now includes a confidence score (0.0 - 1.0):

    0.90 - 1.00  deterministic — AST proves it
    0.70 - 0.89  heuristic     — likely but needs verification
    0.50 - 0.69  probabilistic — flag for human review

Design notes
------------
- BugDetector accepts an optional pre-loaded embedder so the
  embedding model is never loaded twice in one pipeline run.
- Rules registered in self._rules — add method + append to list.
"""

import ast
import logging
import re
from typing import Any, Dict, List, Optional

from config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


CONFIDENCE_MAP: Dict[str, float] = {
    "eval_usage":               0.95,
    "exec_usage":               0.95,
    "assert_in_production":     0.95,
    "bare_except_swallow":      0.95,
    "potential_infinite_loop":  0.90,
    "mutable_default_argument": 0.90,
    "division_by_zero":         0.75,
    "hardcoded_secret":         0.70,
    "print_in_production":      0.70,
    "semantic_similarity_flag": 0.55,
    "statistical_anomaly":      0.50,
    "taint_flow":               0.80,
}


class BugDetector:
    """
    Hybrid bug detection engine combining rule-based and semantic analysis.
    """

    def __init__(
        self,
        config:   PipelineConfig = DEFAULT_CONFIG,
        embedder: Any            = None,
    ):
        self._config   = config
        self._embedder = embedder

        self._rules = [
            self._detect_division_by_zero,
            self._detect_eval_usage,
            self._detect_exec_usage,
            self._detect_infinite_loop,
            self._detect_assert_in_production,
            self._detect_bare_except,
            self._detect_mutable_default_arg,
            self._detect_hardcoded_secret,
            self._detect_print_in_production,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues = []
        for rule in self._rules:
            issues.extend(rule(chunk))
        return issues

    def analyze_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []

        for chunk in chunks:
            issues.extend(self.analyze_chunk(chunk))

        logger.info("Rule-based detection: %d issues found", len(issues))

        has_embeddings = chunks and chunks[0].get("embedding") is not None

        if has_embeddings:
            semantic_issues = self._run_semantic_detection(chunks)
            issues.extend(semantic_issues)
            logger.info("Semantic detection: %d additional issues found", len(semantic_issues))
        else:
            logger.info("Semantic detection skipped — no embeddings present")

        return issues

    # ------------------------------------------------------------------
    # Semantic detection
    # ------------------------------------------------------------------

    def _run_semantic_detection(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            from embeddings.embed_functions import CodeEmbedder
            from embeddings.similarity_search import SimilarityDetector

            embedder = self._embedder or CodeEmbedder(self._config)
            detector = SimilarityDetector(embedder, self._config)
            return detector.find_suspicious_chunks(chunks)

        except ImportError as exc:
            logger.warning("Semantic detection unavailable: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _parse_ast(self, chunk: Dict[str, Any]) -> Optional[ast.Module]:
        try:
            return ast.parse(chunk["code"])
        except SyntaxError:
            return None

    def _scan_lines(self, chunk: Dict[str, Any]) -> List[tuple]:
        result = []
        for i, line in enumerate(chunk["code"].splitlines(), start=1):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                result.append((i, line))
        return result

    def _build_issue(
        self,
        chunk:       Dict[str, Any],
        line_number: int,
        code_line:   str,
        issue_type:  str,
        severity:    str,
        message:     str,
    ) -> Dict[str, Any]:
        return {
            "type":             issue_type,
            "severity":         severity,
            "confidence":       CONFIDENCE_MAP.get(issue_type, 0.60),
            "function":         chunk["function_name"],
            "file":             chunk["file_path"],
            "line_number":      line_number,
            "code_snippet":     code_line.strip(),
            "chunk_id":         chunk["chunk_id"],
            "message":          message,
            "similarity_score": None,
        }

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _detect_division_by_zero(self, chunk: Dict) -> List[Dict]:
        issues = []
        for line_no, line in self._scan_lines(chunk):
            if re.search(r"/\s*0(?!\.\d|\w)", line):
                issues.append(self._build_issue(
                    chunk, line_no, line,
                    "division_by_zero", "high",
                    "Literal division by zero — raises ZeroDivisionError at runtime.",
                ))
        return issues

    def _detect_eval_usage(self, chunk: Dict) -> List[Dict]:
        issues = []
        tree   = self._parse_ast(chunk)
        if tree is None:
            return issues

        lines = chunk["code"].splitlines()

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and
                    node.func.id == "eval"):
                continue

            line     = lines[node.lineno - 1]
            safe_ctx = "compile(" in line and "ctx" in line
            severity = "medium" if safe_ctx else "high"
            message  = (
                "eval() with compiled source — verify input is trusted."
                if safe_ctx else
                "eval() executes arbitrary code. Use ast.literal_eval() instead."
            )
            issues.append(self._build_issue(
                chunk, node.lineno, line, "eval_usage", severity, message,
            ))

        return issues

    def _detect_exec_usage(self, chunk: Dict) -> List[Dict]:
        issues = []
        tree   = self._parse_ast(chunk)
        if tree is None:
            return issues

        lines = chunk["code"].splitlines()

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and
                    node.func.id == "exec"):
                continue

            line      = lines[node.lineno - 1]
            from_file = "read()" in line or "open(" in chunk["code"]
            severity  = "medium" if from_file else "high"
            message   = (
                "exec() loading from file — ensure source is trusted."
                if from_file else
                "Direct exec() call — refactor into explicit functions."
            )
            issues.append(self._build_issue(
                chunk, node.lineno, line, "exec_usage", severity, message,
            ))

        return issues

    def _detect_infinite_loop(self, chunk: Dict) -> List[Dict]:
        issues = []
        tree   = self._parse_ast(chunk)
        if tree is None:
            return issues

        lines = chunk["code"].splitlines()

        for node in ast.walk(tree):
            if not (isinstance(node, ast.While) and
                    isinstance(node.test, ast.Constant) and
                    node.test.value is True):
                continue

            has_exit = any(
                isinstance(child, (ast.Break, ast.Return, ast.Raise))
                for child in ast.walk(node)
            )

            if not has_exit:
                issues.append(self._build_issue(
                    chunk, node.lineno, lines[node.lineno - 1],
                    "potential_infinite_loop", "medium",
                    "while True with no break, return, or raise — potential infinite loop.",
                ))

        return issues

    def _detect_assert_in_production(self, chunk: Dict) -> List[Dict]:
        if chunk["function_name"].startswith("test"):
            return []

        issues = []
        tree   = self._parse_ast(chunk)
        if tree is None:
            return issues

        lines = chunk["code"].splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                issues.append(self._build_issue(
                    chunk, node.lineno, lines[node.lineno - 1],
                    "assert_in_production", "low",
                    "assert stripped by Python -O flag. Use explicit if/raise instead.",
                ))

        return issues

    def _detect_bare_except(self, chunk: Dict) -> List[Dict]:
        issues = []
        tree   = self._parse_ast(chunk)
        if tree is None:
            return issues

        lines = chunk["code"].splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body_is_pass = (
                len(node.body) == 1 and
                isinstance(node.body[0], ast.Pass)
            )
            if body_is_pass:
                issues.append(self._build_issue(
                    chunk, node.lineno, lines[node.lineno - 1],
                    "bare_except_swallow", "high",
                    "Exception silently discarded. At minimum log the exception.",
                ))

        return issues

    def _detect_mutable_default_arg(self, chunk: Dict) -> List[Dict]:
        issues        = []
        tree          = self._parse_ast(chunk)
        if tree is None:
            return issues

        lines         = chunk["code"].splitlines()
        mutable_types = (ast.List, ast.Dict, ast.Set)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for default in node.args.defaults:
                if isinstance(default, mutable_types):
                    issues.append(self._build_issue(
                        chunk, node.lineno, lines[node.lineno - 1],
                        "mutable_default_argument", "medium",
                        f"Mutable default in '{node.name}'. "
                        f"Use None and initialise inside the function body.",
                    ))

        return issues

    def _detect_hardcoded_secret(self, chunk: Dict) -> List[Dict]:
        SECRET_RE = re.compile(
            r"\b(password|passwd|secret|api_key|apikey|token|"
            r"auth_token|access_key|private_key)\s*=\s*['\"].+['\"]",
            re.IGNORECASE,
        )
        issues = []
        for line_no, line in self._scan_lines(chunk):
            if SECRET_RE.search(line):
                issues.append(self._build_issue(
                    chunk, line_no, line,
                    "hardcoded_secret", "high",
                    "Possible hardcoded credential. Use environment variables instead.",
                ))
        return issues

    def _detect_print_in_production(self, chunk: Dict) -> List[Dict]:
        issues = []
        tree   = self._parse_ast(chunk)
        if tree is None:
            return issues

        lines = chunk["code"].splitlines()

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and
                    node.func.id == "print"):
                continue

            issues.append(self._build_issue(
                chunk, node.lineno, lines[node.lineno - 1],
                "print_in_production", "low",
                "print() in production. Use logging.info() / logging.debug() instead.",
            ))

        return issues
