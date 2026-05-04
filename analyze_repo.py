# analyze_repo.py

"""
AI Code Analyzer — Pipeline Entry Point

Usage
-----
    python analyze_repo.py <repo_url>
    python analyze_repo.py <repo_url> --provider groq
    python analyze_repo.py <repo_url> --threshold 0.65 --verbose
    python analyze_repo.py <repo_url> --min-confidence 0.80
    python analyze_repo.py --test

Pipeline stages
---------------
    1.  Clone repository
    2.  Build call graph (taint analysis)
    3.  Scan Python files
    4.  Extract functions
    5.  Create code chunks
    6.  Filter production code
    7.  Generate embeddings
    8.  Detect issues (rule-based + semantic + anomaly)
    9.  Generate patches
    10. Evaluation report
    11. Save results
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from config import DEFAULT_CONFIG, PipelineConfig
from exceptions import AnalyzerBaseError
from ingestion.clone_repo import clone_repository
from ingestion.scan_files import scan_python_files
from parsing.extract_function_code import extract_functions_from_files
from parsing.code_chunker import chunk_functions
from embeddings.embed_functions import CodeEmbedder, embed_chunks
from bug_detector.detect_patterns import BugDetector
from patch_generator.generate_patch import PatchGenerator
from evaluation.benchmark import BenchmarkReport


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@contextmanager
def _timed(label: str):
    start = time.time()
    yield
    logger.info("       %-35s %.1fs", label, time.time() - start)


def _filter_production_chunks(chunks, config):
    kept, removed = [], 0
    for chunk in chunks:
        path = chunk["file_path"]
        if any(e in path for e in config.excluded_dirs):
            removed += 1
            continue
        if any(chunk["function_name"].startswith(p) for p in config.excluded_prefixes):
            removed += 1
            continue
        kept.append(chunk)
    logger.info("Production filter: kept %d, removed %d test chunks", len(kept), removed)
    return kept


def _save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)


def run_pipeline(repo_url: str, config: PipelineConfig = DEFAULT_CONFIG) -> None:

    sep = "─" * 52
    logger.info(sep)
    logger.info("AI CODE ANALYZER — starting pipeline")
    logger.info("Repository : %s", repo_url)
    logger.info("LLM        : %s (%s)", config.llm_provider,
                "enabled" if config.llm_available else "disabled")
    logger.info(sep)

    # ── Step 1: Clone ─────────────────────────────────────────────
    logger.info("[1/11] Cloning repository")
    with _timed("clone"):
        repo_path = clone_repository(repo_url, config)

    # ── Step 2: Call graph + taint analysis ───────────────────────
    logger.info("[2/11] Building call graph (taint analysis)")
    taint_issues = []
    try:
        from analysis.call_graph import CallGraphBuilder
        with _timed("call graph"):
            all_py = list(repo_path.rglob("*.py"))
            cg     = CallGraphBuilder().build(all_py)
            taint_issues = cg.find_source_to_sink_paths()
        logger.info(
            "       Sources: %d  Sinks: %d  Taint flows: %d",
            len(cg.sources), len(cg.sinks), len(taint_issues),
        )
    except Exception as exc:
        logger.warning("Call graph skipped: %s", exc)

    # ── Step 3: Scan ──────────────────────────────────────────────
    logger.info("[3/11] Scanning Python files")
    with _timed("scan"):
        python_files = scan_python_files(repo_path, config)
    logger.info("       Found %d files", len(python_files))

    # ── Step 4: Extract ───────────────────────────────────────────
    logger.info("[4/11] Extracting functions")
    with _timed("extract"):
        functions = extract_functions_from_files(python_files)
    logger.info("       Extracted %d functions", len(functions))

    # ── Step 5: Chunk ─────────────────────────────────────────────
    logger.info("[5/11] Creating code chunks")
    with _timed("chunk"):
        chunks = chunk_functions(functions, config)
    logger.info("       Generated %d chunks", len(chunks))

    # ── Step 6: Filter ────────────────────────────────────────────
    logger.info("[6/11] Filtering to production code")
    with _timed("filter"):
        chunks = _filter_production_chunks(chunks, config)

    # ── Step 7: Embed ─────────────────────────────────────────────
    logger.info("[7/11] Generating embeddings")
    with _timed("embed"):
        embedder = CodeEmbedder(config)
        chunks   = embed_chunks(chunks, embedder)

    # ── Step 8: Detect ────────────────────────────────────────────
    logger.info("[8/11] Detecting issues")

    with _timed("rule + semantic"):
        detector = BugDetector(config, embedder=embedder)
        issues   = detector.analyze_chunks(chunks)

    # Anomaly detection
    anomaly_issues = []
    try:
        from analysis.anomaly_detector import AnomalyDetector
        with _timed("anomaly detection"):
            ad = AnomalyDetector(contamination=0.05)
            ad.fit(chunks)
            anomaly_issues = ad.find_anomalies(chunks)
            issues.extend(anomaly_issues)
    except Exception as exc:
        logger.warning("Anomaly detection skipped: %s", exc)

    # Taint flows
    issues.extend(taint_issues)

    # Confidence summary
    high_conf = [i for i in issues if i.get("confidence", 0) >= 0.80]
    logger.info(
        "       Total: %d issues  (%d high-confidence ≥0.80)",
        len(issues), len(high_conf),
    )

    # ── Step 9: Patch ─────────────────────────────────────────────
    logger.info("[9/11] Generating patch suggestions")
    with _timed("patches"):
        patcher = PatchGenerator(config)
        patches = patcher.generate_patches(issues, chunks)
    logger.info("       Generated %d patches", len(patches))

    # ── Step 10: Report ───────────────────────────────────────────
    logger.info("[10/11] Building evaluation report")
    report = BenchmarkReport(chunks, issues, patches)
    report.print_report()

    # ── Step 11: Save ─────────────────────────────────────────────
    logger.info("[11/11] Saving results")
    rd = config.results_dir
    _save_json(issues,           rd / "issues.json")
    _save_json(patches,          rd / "patches.json")
    _save_json(report.as_dict(), rd / "benchmark.json")
    _save_json(high_conf,        rd / "issues_high_confidence.json")

    logger.info(sep)
    logger.info("ANALYSIS COMPLETE")
    logger.info(sep)


def quick_test() -> None:
    """Quick local test — no git clone, completes in ~60 seconds."""

    test_dir = Path("repos/test_quick/src")
    test_dir.mkdir(parents=True, exist_ok=True)

    (test_dir / "sample.py").write_text(
        'def divide(a, b):\n    return a / 0\n\n'
        'def load_data(items=[]):\n    items.append(1)\n    return items\n\n'
        'def run_code(user_input):\n    eval(user_input)\n\n'
        'def fetch():\n    try:\n        pass\n    except:\n        pass\n\n'
        'def debug_output():\n    print("debug")\n'
    )

    print("\n[QUICK TEST MODE]\n")

    config   = DEFAULT_CONFIG
    files    = scan_python_files(Path("repos/test_quick"), config)
    funcs    = extract_functions_from_files(files)
    chunks   = chunk_functions(funcs, config)
    embedder = CodeEmbedder(config)
    chunks   = embed_chunks(chunks, embedder)

    # Model should load ONCE only — watch for double load here
    detector = BugDetector(config, embedder=embedder)
    issues   = detector.analyze_chunks(chunks)

    print(f"Files  : {len(files)}")
    print(f"Chunks : {len(chunks)}")
    print(f"Issues : {len(issues)}")
    print()
    for issue in issues:
        print(f"  [{issue.get('confidence', 0):.0%}] {issue['type']} in {issue['function']}()")

    print("\n[DONE]\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_repo",
        description="AI Code Analyzer — hybrid static + semantic + taint analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python analyze_repo.py https://github.com/pallets/flask
  python analyze_repo.py https://github.com/pallets/flask --provider groq
  python analyze_repo.py https://github.com/pallets/flask --min-confidence 0.80
  python analyze_repo.py --test
        """,
    )
    parser.add_argument("repo_url", nargs="?", help="GitHub repository URL")
    parser.add_argument("--provider", choices=["llamacpp", "anthropic", "groq", "ollama"])
    parser.add_argument("--threshold", type=float, help="Similarity threshold (default 0.75)")
    parser.add_argument("--min-confidence", type=float, default=0.0, dest="min_confidence",
                        help="Only show issues at or above this confidence")
    parser.add_argument("--output", type=str, help="Results directory")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test:
        quick_test()
        return

    if not args.repo_url:
        parser.print_help()
        sys.exit(1)

    overrides = {}
    if args.provider:   overrides["llm_provider"]          = args.provider
    if args.threshold:  overrides["similarity_threshold"]  = args.threshold
    if args.output:     overrides["results_dir"]           = Path(args.output)

    config = dataclasses.replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG

    try:
        run_pipeline(args.repo_url, config)
    except AnalyzerBaseError as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
