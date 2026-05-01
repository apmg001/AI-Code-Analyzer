# bug_detector/detect_patterns.py

"""
Module: detect_patterns

Responsibility
--------------
Detect risky patterns in code chunks using two complementary strategies:

1. Rule-based  — Fast, deterministic AST/regex checks for known patterns.
2. Semantic    — Embedding-based similarity search for unknown variants.

Design notes
------------
- BugDetector is the single public entry point; callers do not need to
  know which strategy found an issue.
- Each rule is a private method that returns a list — empty if clean,
  one or more items if issues were found.
- Rules are registered in self.rules so adding a new check requires
  only writing the method and appending it to the list. Open/Closed.
- Semantic detection is opt-in: it runs only when embeddings are present
  on the chunks, so the detector works in embedding-free environments too.
- No logging.basicConfig here — logging is configured by the entry point.
"""

import ast
import logging
import re
from typing import Any, Dict, List, Optional

from config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


class BugDetector:
    """
    Hybrid bug detection engine combining rule-based and semantic analysis.

    Usage
    -----
    detector = BugDetector()
    issues   = detector.analyze_chunks(embedded_chunks)
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self._config = config
        self._rules  = [
            self._detect_division_by_zero,
            self._detect_eval_usage,
            self._detect_exec_usage,
            self._detect_infinite_loop,
            self._detect_assert_in_production,
            self._detect_bare_except,
            self._detect_mutable_default_arg,
            self._detect_hardcoded_secret,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all rule-based checks against a single chunk."""
        issues = []
        for rule in self._rules:
            issues.extend(rule(chunk))
        return issues

    def analyze_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run rule-based AND semantic detection across all chunks.

        Semantic detection is skipped when embeddings are absent
        so this method works even before the embedding step.
        """
        issues: List[Dict[str, Any]] = []

        # --- Pass 1: rule-based (always runs) ---
        for chunk in chunks:
            issues.extend(self.analyze_chunk(chunk))

        rule_count = len(issues)
        logger.info("Rule-based detection: %d issues found", rule_count)

        # --- Pass 2: semantic (runs only when embeddings are attached) ---
        has_embeddings = chunks and chunks[0].get("embedding") is not None

        if has_embeddings:
            semantic_issues = self._run_semantic_detection(chunks)
            issues.extend(semantic_issues)
            logger.info("Semantic detection: %d additional issues found", len(semantic_issues))
        else:
            logger.info("Semantic detection skipped — no embeddings present")

        return issues

    # ------------------------------------------------------------------
    # Semantic detection (delegates to SimilarityDetector)
    # ------------------------------------------------------------------

    def _run_semantic_detection(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Lazy-import and run the similarity detector.

        Lazy import avoids a hard dependency on sentence-transformers
        when running in rule-only mode (e.g. fast CI checks).
        """
        try:
            from embeddings.embed_functions import CodeEmbedder
            from embeddings.similarity_search import SimilarityDetector

            embedder  = CodeEmbedder(self._config)
            detector  = SimilarityDetector(embedder, self._config)
            return detector.find_suspicious_chunks(chunks)

        except ImportError as exc:
            logger.warning("Semantic detection unavailable: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _parse_ast(self, chunk: Dict[str, Any]) -> Optional[ast.Module]:
        """Parse chunk code into an AST; return None on syntax error."""
        try:
            return ast.parse(chunk["code"])
        except SyntaxError:
            return None

    def _scan_lines(self, chunk: Dict[str, Any]) -> List[tuple]:
        """
        Return (line_number, line_text) pairs, skipping blank lines
        and comment-only lines.
        """
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
        """Construct a normalised issue dict."""
        return {
            "type":          issue_type,
            "severity":      severity,
            "function":      chunk["function_name"],
            "file":          chunk["file_path"],
            "line_number":   line_number,
            "code_snippet":  code_line.strip(),
            "chunk_id":      chunk["chunk_id"],
            "message":       message,
            "similarity_score": None,
        }

    # ------------------------------------------------------------------
    # Rule-based detection rules
    # ------------------------------------------------------------------

    def _detect_division_by_zero(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues = []
        for line_no, line in self._scan_lines(chunk):
            if re.search(r"/\s*0(?!\.\d|\w)", line):
                issues.append(self._build_issue(
                    chunk, line_no, line,
                    "division_by_zero", "high",
                    "Literal division by zero detected. This will raise ZeroDivisionError at runtime.",
                ))
        return issues

    def _detect_eval_usage(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
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

            line = lines[node.lineno - 1]
            is_compile_pattern = "compile(" in line and "ctx" in line

            severity = "medium" if is_compile_pattern else "high"
            message  = (
                "eval() with compiled source detected — verify input is trusted."
                if is_compile_pattern else
                "eval() can execute arbitrary code. Use ast.literal_eval() for safe value parsing."
            )
            issues.append(self._build_issue(chunk, node.lineno, line, "eval_usage", severity, message))

        return issues

    def _detect_exec_usage(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
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

            line    = lines[node.lineno - 1]
            from_file = "read()" in line or "open(" in chunk["code"]

            severity = "medium" if from_file else "high"
            message  = (
                "exec() appears to load code from a file — ensure the source is fully trusted."
                if from_file else
                "Direct exec() call detected. Refactor logic into explicit functions."
            )
            issues.append(self._build_issue(chunk, node.lineno, line, "exec_usage", severity, message))

        return issues

    def _detect_infinite_loop(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                    "while True loop with no break, return, or raise — potential infinite loop.",
                ))

        return issues

    def _detect_assert_in_production(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Assert statements are stripped when Python runs with -O (optimize flag).
        They should never be used as runtime guards in production code.
        """
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
                    "assert is stripped by the Python optimizer (-O flag). "
                    "Use explicit if/raise for runtime validation.",
                ))

        return issues

    def _detect_bare_except(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Bare `except:` or `except Exception:` with only `pass` silently
        swallows errors and makes debugging nearly impossible.
        """
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
                    "Exception caught with `pass` body — error is silently discarded. "
                    "At minimum log the exception.",
                ))

        return issues

    def _detect_mutable_default_arg(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Mutable default arguments ([], {}, set()) are shared across
        all calls to the function — a classic Python gotcha.
        """
        issues = []
        tree   = self._parse_ast(chunk)
        if tree is None:
            return issues

        lines = chunk["code"].splitlines()

        mutable_types = (ast.List, ast.Dict, ast.Set)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for default in node.args.defaults:
                if isinstance(default, mutable_types):
                    issues.append(self._build_issue(
                        chunk, node.lineno, lines[node.lineno - 1],
                        "mutable_default_argument", "medium",
                        f"Mutable default argument ({type(default).__name__}) detected in "
                        f"'{node.name}'. Use None and initialise inside the function body.",
                    ))

        return issues

    def _detect_hardcoded_secret(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect likely hardcoded credentials via variable name heuristics.
        This is not a definitive check — it is a prompt for human review.
        """
        SECRET_NAMES = re.compile(
            r"\b(password|passwd|secret|api_key|apikey|token|auth_token|access_key)\s*=\s*['\"].+['\"]",
            re.IGNORECASE,
        )

        issues = []

        for line_no, line in self._scan_lines(chunk):
            if SECRET_NAMES.search(line):
                issues.append(self._build_issue(
                    chunk, line_no, line,
                    "hardcoded_secret", "high",
                    "Possible hardcoded credential detected. "
                    "Use environment variables or a secrets manager instead.",
                ))

        return issues
