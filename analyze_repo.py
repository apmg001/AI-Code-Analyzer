# analyze_repo.py

"""
Entry point script to run the full AI-Code-Analyzer pipeline
on a repository provided via terminal.

Usage:
    python analyze_repo.py <repo_url>

Example:
    python analyze_repo.py https://github.com/pallets/flask
"""

import sys
import os
import json

from ingestion.clone_repo import clone_repository
from ingestion.scan_files import scan_python_files

from parsing.extract_function_code import extract_functions_from_files
from parsing.code_chunker import chunk_functions

from embeddings.embed_functions import CodeEmbedder, embed_chunks

from bug_detector.detect_patterns import BugDetector
from patch_generator.generate_patch import PatchGenerator

from evaluation.benchmark import BenchmarkReport


def filter_production_chunks(chunks):
    """
    Remove test-related files so the analyzer focuses on
    production code.
    """

    filtered = []

    for chunk in chunks:
        path = chunk["file_path"]

        if "tests/" in path:
            continue

        if "test_" in path:
            continue

        filtered.append(chunk)

    return filtered


def run_pipeline(repo_url):

    print("\n========== AI CODE ANALYZER ==========\n")

    # --------------------------------------------------
    # Step 1: Clone repository
    # --------------------------------------------------

    print("[STEP] Cloning repository")

    repo_path = clone_repository(repo_url)

    print(f"[INFO] Repository ready at: {repo_path}")

    # --------------------------------------------------
    # Step 2: Scan Python files
    # --------------------------------------------------

    print("\n[STEP] Scanning Python files")

    python_files = scan_python_files(repo_path)

    print(f"[INFO] Found {len(python_files)} Python files")

    # --------------------------------------------------
    # Step 3: Extract functions
    # --------------------------------------------------

    print("\n[STEP] Extracting functions")

    functions = extract_functions_from_files(python_files)

    print(f"[INFO] Extracted {len(functions)} functions")

    # --------------------------------------------------
    # Step 4: Create code chunks
    # --------------------------------------------------

    print("\n[STEP] Creating code chunks")

    chunks = chunk_functions(functions)

    print(f"[INFO] Generated {len(chunks)} chunks")

    # --------------------------------------------------
    # Step 5: Filter test code
    # --------------------------------------------------

    print("\n[STEP] Filtering test files")

    chunks = filter_production_chunks(chunks)

    print(f"[INFO] Remaining chunks after filtering: {len(chunks)}")

    # --------------------------------------------------
    # Step 6: Generate embeddings
    # --------------------------------------------------

    print("\n[STEP] Generating embeddings")

    embedder = CodeEmbedder()

    embedded_chunks = embed_chunks(chunks, embedder)

    print(f"[INFO] Generated embeddings for {len(embedded_chunks)} chunks")

    # --------------------------------------------------
    # Step 7: Detect bugs
    # --------------------------------------------------

    print("\n[STEP] Detecting issues")

    detector = BugDetector()

    issues = detector.analyze_chunks(embedded_chunks)

    print(f"[INFO] Detected {len(issues)} issues")

    # --------------------------------------------------
    # Step 8: Generate patches
    # --------------------------------------------------

    print("\n[STEP] Generating patch suggestions")

    patcher = PatchGenerator()

    patches = patcher.generate_patches(issues, embedded_chunks)

    print(f"[INFO] Generated {len(patches)} patch suggestions")

    # --------------------------------------------------
    # Step 9: Evaluation report
    # --------------------------------------------------

    print("\n[STEP] Generating evaluation report")

    benchmark = BenchmarkReport(embedded_chunks, issues)

    benchmark.print_report()

    # --------------------------------------------------
    # Step 10: Save outputs
    # --------------------------------------------------

    results_dir = "analysis_results"

    os.makedirs(results_dir, exist_ok=True)

    issues_path = os.path.join(results_dir, "issues.json")
    patches_path = os.path.join(results_dir, "patches.json")

    with open(issues_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)

    with open(patches_path, "w", encoding="utf-8") as f:
        json.dump(patches, f, indent=2)

    print("\n[INFO] Results saved:")
    print("  Issues:", issues_path)
    print("  Patches:", patches_path)

    print("\n========== ANALYSIS COMPLETE ==========\n")


def main():

    if len(sys.argv) != 2:

        print("Usage:")
        print("  python analyze_repo.py <repo_url>")

        sys.exit(1)

    repo_url = sys.argv[1]

    run_pipeline(repo_url)


if __name__ == "__main__":
    main()