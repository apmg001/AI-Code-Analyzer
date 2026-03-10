# bug_detector/detect_patterns.py

"""
Module: detect_patterns

Purpose
-------
Detect potential bugs or risky patterns inside extracted code chunks.

This module performs lightweight static analysis on function code
to identify suspicious constructs before deeper AI reasoning is applied.

Responsibilities
----------------
1. Receive code chunks
2. Apply detection rules
3. Return structured bug reports
"""

import re
from typing import List, Dict


class BugDetector:
    """
    Rule-based bug detection engine.

    Each rule inspects code and returns issues if patterns match.
    """

    def __init__(self):

        self.rules = [
            self._detect_division_by_zero,
            self._detect_eval_usage,
            self._detect_exec_usage,
            self._detect_infinite_loop,
            self._detect_assert_usage
        ]

    def analyze_chunk(self, chunk: Dict) -> List[Dict]:
        """
        Run all rules on a single code chunk.
        """

        issues = []

        for rule in self.rules:

            result = rule(chunk)

            if result:
                issues.extend(result)

        return issues

    def analyze_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Analyze multiple chunks.
        """

        detected_issues = []

        for chunk in chunks:

            issues = self.analyze_chunk(chunk)

            if issues:
                detected_issues.extend(issues)

        return detected_issues

    # ----------------------------
    # Detection Rules
    # ----------------------------

    def _detect_division_by_zero(self, chunk: Dict) -> List[Dict]:

        issues = []

        code = chunk["code"]

        pattern = r"/\s*0"

        if re.search(pattern, code):

            issues.append({
                "type": "division_by_zero",
                "severity": "high",
                "function": chunk["function_name"],
                "file": chunk["file_path"],
                "chunk_id": chunk["chunk_id"],
                "message": "Possible division by zero detected."
            })

        return issues

    def _detect_eval_usage(self, chunk: Dict) -> List[Dict]:

        issues = []

        code = chunk["code"]

        if "eval(" in code:

            issues.append({
                "type": "eval_usage",
                "severity": "high",
                "function": chunk["function_name"],
                "file": chunk["file_path"],
                "chunk_id": chunk["chunk_id"],
                "message": "Use of eval() detected. This may lead to security risks."
            })

        return issues

    def _detect_exec_usage(self, chunk: Dict) -> List[Dict]:

        issues = []

        code = chunk["code"]

        if "exec(" in code:

            issues.append({
                "type": "exec_usage",
                "severity": "high",
                "function": chunk["function_name"],
                "file": chunk["file_path"],
                "chunk_id": chunk["chunk_id"],
                "message": "Use of exec() detected. This may allow arbitrary code execution."
            })

        return issues

    def _detect_infinite_loop(self, chunk: Dict) -> List[Dict]:

        issues = []

        code = chunk["code"]

        if "while True" in code:

            issues.append({
                "type": "potential_infinite_loop",
                "severity": "medium",
                "function": chunk["function_name"],
                "file": chunk["file_path"],
                "chunk_id": chunk["chunk_id"],
                "message": "Possible infinite loop detected (while True)."
            })

        return issues

    def _detect_assert_usage(self, chunk: Dict) -> List[Dict]:

        issues = []

        code = chunk["code"]

        if "assert " in code:

            issues.append({
                "type": "assert_usage",
                "severity": "low",
                "function": chunk["function_name"],
                "file": chunk["file_path"],
                "chunk_id": chunk["chunk_id"],
                "message": "Assertion used in code. This may cause runtime failures."
            })

        return issues


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

    detector = BugDetector()

    print("[INFO] Running bug detection")

    issues = detector.analyze_chunks(chunks)

    print(f"[INFO] Detected {len(issues)} issues\n")

    for issue in issues:

        print("Issue Type:", issue["type"])
        print("Severity:", issue["severity"])
        print("Function:", issue["function"])
        print("File:", issue["file"])
        print("Message:", issue["message"])
        print("-" * 50)