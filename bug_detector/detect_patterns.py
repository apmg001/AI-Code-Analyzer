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
from typing import List, Dict


class BugDetector:
    """
    Rule-based bug detection engine.

    Each rule inspects code and returns a list of issues if patterns match.
    """

    def __init__(self):

        # Register detection rules here
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
        Run all detection rules on a single chunk.
        """

        issues = []

        for rule in self.rules:
            result = rule(chunk)

            if result:
                issues.extend(result)

        return issues

    def analyze_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Run detection on multiple chunks.
        """

        detected_issues = []

        for chunk in chunks:

            issues = self.analyze_chunk(chunk)

            if issues:
                detected_issues.extend(issues)

        return detected_issues

    # ---------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------

    def _scan_lines(self, chunk: Dict):
        """
        Split chunk code into numbered lines.
        """

        code = chunk["code"]
        lines = code.split("\n")

        return list(enumerate(lines, start=1))

    def _build_issue(
        self,
        chunk: Dict,
        line_number: int,
        code_line: str,
        issue_type: str,
        severity: str,
        message: str
    ) -> Dict:
        """
        Standard issue object builder.
        """

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

                issues.append(
                    self._build_issue(
                        chunk,
                        line_number,
                        line,
                        "division_by_zero",
                        "high",
                        "Possible division by zero detected."
                    )
                )

        return issues

    def _detect_eval_usage(self, chunk: Dict) -> List[Dict]:

        issues = []

        for line_number, line in self._scan_lines(chunk):

            if "eval(" in line:

                issues.append(
                    self._build_issue(
                        chunk,
                        line_number,
                        line,
                        "eval_usage",
                        "high",
                        "Use of eval() detected. This may lead to security risks."
                    )
                )

        return issues

    def _detect_exec_usage(self, chunk: Dict) -> List[Dict]:

        issues = []

        for line_number, line in self._scan_lines(chunk):

            if "exec(" in line:

                issues.append(
                    self._build_issue(
                        chunk,
                        line_number,
                        line,
                        "exec_usage",
                        "high",
                        "Use of exec() detected. This may allow arbitrary code execution."
                    )
                )

        return issues

    def _detect_infinite_loop(self, chunk: Dict) -> List[Dict]:

        issues = []

        for line_number, line in self._scan_lines(chunk):

            if "while True" in line:

                issues.append(
                    self._build_issue(
                        chunk,
                        line_number,
                        line,
                        "potential_infinite_loop",
                        "medium",
                        "Possible infinite loop detected (while True)."
                    )
                )

        return issues

    def _detect_assert_usage(self, chunk: Dict) -> List[Dict]:

        issues = []

        for line_number, line in self._scan_lines(chunk):

            if "assert " in line:

                issues.append(
                    self._build_issue(
                        chunk,
                        line_number,
                        line,
                        "assert_usage",
                        "low",
                        "Assertion used in code. This may cause runtime failures."
                    )
                )

        return issues


# ---------------------------------------------------------
# Local Testing Block
# ---------------------------------------------------------

if __name__ == "__main__":

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

    detector = BugDetector()

    print("[INFO] Running bug detection")

    issues = detector.analyze_chunks(chunks)

    print(f"[INFO] Detected {len(issues)} issues\n")

    for issue in issues:

        print("Issue Type:", issue["type"])
        print("Severity:", issue["severity"])
        print("Function:", issue["function"])
        print("File:", issue["file"])
        print("Line:", issue["line_number"])
        print("Code:", issue["code_snippet"])
        print("Message:", issue["message"])
        print("-" * 60)