"""
Module: detect_patterns

Purpose
-------
Detect potentially risky patterns inside extracted code chunks.

This module performs lightweight static analysis using simple
pattern rules before any deeper AI reasoning is applied.

Responsibilities
----------------
1. Receive code chunks
2. Scan code line-by-line
3. Apply detection rules
4. Return structured bug reports
"""

import re
import ast
from typing import List, Dict


class BugDetector:
    """
    Rule-based + semantic bug detection engine.

    Each rule inspects code and returns a list of issues if patterns match.
    Semantic detection uses embeddings to catch patterns rules miss.
    """

    def __init__(self):

        self.rules = [
            self._detect_division_by_zero,
            self._detect_eval_usage,
            self._detect_exec_usage,
            self._detect_infinite_loop,
            self._detect_assert_usage
        ]

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def analyze_chunk(self, chunk: Dict) -> List[Dict]:
        """
        Run all rule-based detection rules on a single chunk.
        """

        issues = []

        for rule in self.rules:
            result = rule(chunk)
            if result:
                issues.extend(result)

        return issues

    def analyze_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Run BOTH rule-based AND semantic detection on all chunks.

        Rule-based: fast, deterministic, catches known patterns.
        Semantic:   AI-powered, catches unknown similar patterns.
        """

        from embeddings.similarity_search import SimilarityDetector
        from embeddings.embed_functions import CodeEmbedder

        detected_issues = []

        # Step 1: Rule-based detection (always runs)
        for chunk in chunks:
            issues = self.analyze_chunk(chunk)
            if issues:
                detected_issues.extend(issues)

        # Step 2: Semantic detection (runs only if embeddings exist)
        if chunks and "embedding" in chunks[0]:

            print("[INFO] Running semantic similarity detection")

            embedder = CodeEmbedder()
            similarity_detector = SimilarityDetector(embedder, threshold=0.75)
            semantic_issues = similarity_detector.find_suspicious_chunks(chunks)

            print(f"[INFO] Semantic detection found {len(semantic_issues)} additional issues")

            detected_issues.extend(semantic_issues)

        else:
            print("[WARN] No embeddings found — skipping semantic detection")

        return detected_issues

    # ---------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------

    def _scan_lines(self, chunk: Dict):

        code = chunk["code"]
        lines = code.split("\n")
        result = []

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            result.append((i, line))

        return result

    def _build_issue(
        self,
        chunk: Dict,
        line_number: int,
        code_line: str,
        issue_type: str,
        severity: str,
        message: str
    ) -> Dict:

        return {
            "type": issue_type,
            "severity": severity,
            "function": chunk["function_name"],
            "file": chunk["file_path"],
            "line_number": line_number,
            "code_snippet": code_line.strip(),
            "chunk_id": chunk["chunk_id"],
            "message": message
        }

    # ---------------------------------------------------------
    # Detection Rules
    # ---------------------------------------------------------

    def _detect_division_by_zero(self, chunk: Dict) -> List[Dict]:

        issues = []

        for line_number, line in self._scan_lines(chunk):
            if re.search(r"/\s*0", line):
                issues.append(self._build_issue(
                    chunk, line_number, line,
                    "division_by_zero", "high",
                    "Possible division by zero detected."
                ))

        return issues

    def _detect_eval_usage(self, chunk: Dict) -> List[Dict]:

        issues = []
        code = chunk["code"]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "eval":

                    line_number = node.lineno
                    line = code.split("\n")[line_number - 1]

                    if "compile(" in line and "ctx" in line:
                        severity = "medium"
                        message = (
                            "Dynamic evaluation detected using compiled code. "
                            "Ensure the source file being executed is trusted."
                        )
                    else:
                        severity = "high"
                        message = "Use of eval() detected. This may allow execution of arbitrary code."

                    issues.append(self._build_issue(
                        chunk, line_number, line,
                        "eval_usage", severity, message
                    ))

        return issues

    def _detect_exec_usage(self, chunk: Dict) -> List[Dict]:

        issues = []
        code = chunk["code"]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "exec":

                    line_number = node.lineno
                    line = code.split("\n")[line_number - 1]

                    if "read()" in line or "open(" in code:
                        severity = "medium"
                        message = (
                            "Dynamic code execution detected. "
                            "Appears to load code from a file. Ensure the source is trusted."
                        )
                    else:
                        severity = "high"
                        message = "Direct exec() call detected. This may allow arbitrary code execution."

                    issues.append(self._build_issue(
                        chunk, line_number, line,
                        "exec_usage", severity, message
                    ))

        return issues

    def _detect_infinite_loop(self, chunk: Dict) -> List[Dict]:

        issues = []
        code = chunk["code"]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:

                    has_exit = any(
                        isinstance(child, (ast.Break, ast.Return, ast.Raise))
                        for child in ast.walk(node)
                    )

                    if not has_exit:
                        line_number = node.lineno
                        issues.append(self._build_issue(
                            chunk, line_number,
                            code.split("\n")[line_number - 1],
                            "potential_infinite_loop", "medium",
                            "While True loop with no exit condition detected."
                        ))

        return issues

    def _detect_assert_usage(self, chunk: Dict) -> List[Dict]:

        issues = []
        code = chunk["code"]

        if chunk["function_name"].startswith("test"):
            return issues

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):

                line_number = node.lineno
                issues.append(self._build_issue(
                    chunk, line_number,
                    code.split("\n")[line_number - 1],
                    "assert_usage", "low",
                    "Assert used in runtime code. May be stripped by Python optimizer flags."
                ))

        return issues


# ---------------------------------------------------------
# Local Testing Block
# ---------------------------------------------------------

if __name__ == "__main__":

    from ingestion.scan_files import scan_python_files
    from parsing.extract_function_code import extract_functions_from_files
    from parsing.code_chunker import chunk_functions

    repo_path = "ingestion/repos/flask"

    python_files = scan_python_files(repo_path)
    functions = extract_functions_from_files(python_files[:5])
    chunks = chunk_functions(functions)

    detector = BugDetector()
    issues = detector.analyze_chunks(chunks)

    print(f"\n[INFO] Total issues detected: {len(issues)}\n")

    for issue in issues:
        print("Type    :", issue["type"])
        print("Severity:", issue["severity"])
        print("Function:", issue["function"])
        print("File    :", issue["file"])
        print("Message :", issue["message"])
        print("-" * 60)