# patch_generator/generate_patch.py

"""
Module: generate_patch

Purpose
-------
Generate patch suggestions for detected code issues.

This module receives bug reports and the corresponding code chunks,
then produces suggested fixes that a developer could apply.

Responsibilities
----------------
1. Accept bug reports from the detection module
2. Retrieve the associated code chunk
3. Generate a patch suggestion
4. Return structured patch objects
"""

from typing import List, Dict


class PatchGenerator:
    """
    Rule-based patch generator.

    Each bug type has a corresponding patch strategy.
    """

    def __init__(self):

        self.patch_rules = {
            "division_by_zero": self._patch_division_by_zero,
            "eval_usage": self._patch_eval_usage,
            "exec_usage": self._patch_exec_usage,
            "potential_infinite_loop": self._patch_infinite_loop,
            "assert_usage": self._patch_assert_usage
        }

    def generate_patch(self, issue: Dict, chunk: Dict) -> Dict:
        """
        Generate a patch for a single detected issue.
        """

        issue_type = issue["type"]

        patch_handler = self.patch_rules.get(issue_type)

        if not patch_handler:
            return {
                "status": "no_patch_available",
                "issue": issue,
                "suggestion": "No automated patch rule available."
            }

        patch = patch_handler(issue, chunk)

        return patch

    def generate_patches(self, issues: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """
        Generate patches for multiple issues.
        """

        patches = []

        chunk_map = {c["chunk_id"]: c for c in chunks}

        for issue in issues:

            chunk_id = issue["chunk_id"]

            chunk = chunk_map.get(chunk_id)

            if not chunk:
                continue

            patch = self.generate_patch(issue, chunk)

            patches.append(patch)

        return patches

    # ------------------------------------------------
    # Patch rules
    # ------------------------------------------------

    def _patch_division_by_zero(self, issue: Dict, chunk: Dict) -> Dict:

        suggestion = (
            "Add a guard condition before performing division.\n\n"
            "Example:\n"
            "if denominator != 0:\n"
            "    result = numerator / denominator"
        )

        return self._build_patch(issue, chunk, suggestion)

    def _patch_eval_usage(self, issue: Dict, chunk: Dict) -> Dict:

        suggestion = (
            "Avoid using eval(). Consider safer alternatives such as:\n"
            "• ast.literal_eval()\n"
            "• explicit parsing logic"
        )

        return self._build_patch(issue, chunk, suggestion)

    def _patch_exec_usage(self, issue: Dict, chunk: Dict) -> Dict:

        suggestion = (
            "Avoid using exec(). Refactor logic into callable functions "
            "or structured control flow."
        )

        return self._build_patch(issue, chunk, suggestion)

    def _patch_infinite_loop(self, issue: Dict, chunk: Dict) -> Dict:

        suggestion = (
            "Ensure loop has a clear termination condition.\n\n"
            "Example:\n"
            "while condition:\n"
            "    ...\n"
        )

        return self._build_patch(issue, chunk, suggestion)

    def _patch_assert_usage(self, issue: Dict, chunk: Dict) -> Dict:

        suggestion = (
            "Avoid using assert in production code.\n"
            "Replace with explicit error handling.\n\n"
            "Example:\n"
            "if condition is False:\n"
            "    raise ValueError('Invalid state')"
        )

        return self._build_patch(issue, chunk, suggestion)

    # ------------------------------------------------
    # Helper
    # ------------------------------------------------

    def _build_patch(self, issue: Dict, chunk: Dict, suggestion: str) -> Dict:

        return {
            "function": chunk["function_name"],
            "file": chunk["file_path"],
            "chunk_id": chunk["chunk_id"],
            "issue_type": issue["type"],
            "severity": issue["severity"],
            "original_code": chunk["code"],
            "suggested_fix": suggestion
        }


if __name__ == "__main__":

    # Local test block

    from ingestion.scan_files import scan_python_files
    from parsing.extract_function_code import extract_functions_from_files
    from parsing.code_chunker import chunk_functions
    from bug_detector.detect_patterns import BugDetector

    repo_path = "ingestion/repos/flask"

    print("[INFO] Scanning repository")

    python_files = scan_python_files(repo_path)

    print("[INFO] Extracting functions")

    functions = extract_functions_from_files(python_files[:5])

    print("[INFO] Creating chunks")

    chunks = chunk_functions(functions)

    detector = BugDetector()

    print("[INFO] Detecting issues")

    issues = detector.analyze_chunks(chunks)

    print(f"[INFO] Detected {len(issues)} issues")

    generator = PatchGenerator()

    print("[INFO] Generating patches")

    patches = generator.generate_patches(issues, chunks)

    print(f"[INFO] Generated {len(patches)} patch suggestions\n")

    for patch in patches[:3]:

        print("Function:", patch["function"])
        print("File:", patch["file"])
        print("Issue:", patch["issue_type"])
        print("Suggested Fix:\n", patch["suggested_fix"])
        print("-" * 60)