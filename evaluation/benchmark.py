# evaluation/benchmark.py

"""
Module: benchmark

Purpose
-------
Evaluate the performance of the code analysis pipeline.

This module summarizes the results produced by:
    • bug detection
    • patch generation

Responsibilities
----------------
1. Count analyzed functions
2. Count detected issues
3. Categorize issue types
4. Produce evaluation report
"""

from typing import List, Dict
from collections import defaultdict


class BenchmarkReport:
    """
    Evaluation helper class.

    Generates metrics from detected issues and analyzed chunks.
    """

    def __init__(self, chunks: List[Dict], issues: List[Dict]):

        self.chunks = chunks
        self.issues = issues

    # ---------------------------------------
    # Basic metrics
    # ---------------------------------------

    def total_functions(self) -> int:
        """
        Total number of analyzed functions.
        """

        return len(self.chunks)

    def total_issues(self) -> int:
        """
        Total detected issues.
        """

        return len(self.issues)

    def issue_distribution(self) -> Dict[str, int]:
        """
        Count issues grouped by type.
        """

        counts = defaultdict(int)

        for issue in self.issues:
            counts[issue["type"]] += 1

        return dict(counts)

    def severity_distribution(self) -> Dict[str, int]:
        """
        Count issues grouped by severity.
        """

        counts = defaultdict(int)

        for issue in self.issues:
            counts[issue["severity"]] += 1

        return dict(counts)

    def affected_files(self) -> int:
        """
        Number of files containing detected issues.
        """

        files = set()

        for issue in self.issues:
            files.add(issue["file"])

        return len(files)

    # ---------------------------------------
    # Reporting
    # ---------------------------------------

    def generate_report(self) -> Dict:
        """
        Generate structured evaluation report.
        """

        report = {
            "functions_analyzed": self.total_functions(),
            "issues_detected": self.total_issues(),
            "files_with_issues": self.affected_files(),
            "issue_distribution": self.issue_distribution(),
            "severity_distribution": self.severity_distribution()
        }

        return report

    def print_report(self):

        report = self.generate_report()

        print("\n========== ANALYSIS REPORT ==========\n")

        print("Functions analyzed:", report["functions_analyzed"])
        print("Issues detected:", report["issues_detected"])
        print("Files with issues:", report["files_with_issues"])

        print("\nIssue Type Distribution:")

        for issue_type, count in report["issue_distribution"].items():
            print(f"  {issue_type}: {count}")

        print("\nSeverity Distribution:")

        for severity, count in report["severity_distribution"].items():
            print(f"  {severity}: {count}")

        print("\n=====================================\n")


if __name__ == "__main__":

    # Local testing pipeline

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

    benchmark = BenchmarkReport(chunks, issues)

    benchmark.print_report()