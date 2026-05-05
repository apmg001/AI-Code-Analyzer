# analyze_repo.py

"""
AI Code Analyzer — Pipeline Entry Point

13 stages. 5 novel capabilities:

    Stage  3 — Taint analysis       tracks untrusted data across function calls
    Stage  9 — Anomaly detection    learns what normal looks like, flags outliers
    Stage 10 — Codebase DNA         learns conventions, flags violations
    Stage 12 — Git archaeology      finds when and who introduced each bug
    Stage 13 — Patch verification   runs test suite to confirm fixes are safe

Usage
-----
    python analyze_repo.py <repo_url>
    python analyze_repo.py <repo_url> --provider groq
    python analyze_repo.py <repo_url> --min-confidence 0.80
    python analyze_repo.py <repo_url> --threshold 0.65 --verbose
    python analyze_repo.py --test

Environment variables
---------------------
    ANTHROPIC_API_KEY   Claude API patches
    GROQ_API_KEY        Groq free-tier patches

Design notes
------------
- Scan (step 2) runs BEFORE call graph (step 3). This ensures the
  call graph only processes production files — not test files.
  Previously the call graph scanned all *.py files, which caused
  test_*.py to dominate the taint flow results.

- Every novel stage is wrapped in try/except. If any one of them
  fails — missing dependency, bad repo, network issue — the pipeline
  continues and produces partial results rather than crashing.

- The embedder is created once in step 7 and passed into BugDetector
  in step 8. Without this, the embedding model loads twice — once
  for chunk embeddings and once for semantic detection.

- High-confidence issues (>=0.80) are saved to a separate file so
  downstream consumers can filter without reprocessing.
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


# ------------------------------------------------------------------
# Logging — configured here so every module inherits the same format
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Timing context manager
# ------------------------------------------------------------------

@contextmanager
def _timed(label: str):
    """
    Log how long a pipeline stage takes.

    Used as: `with _timed("embed"):  do_work()`
    Produces: `       embed                     6.1s`
    """
    start = time.time()
    yield
    logger.info("       %-38s %.1fs", label, time.time() - start)


# ------------------------------------------------------------------
# Pipeline utilities
# ------------------------------------------------------------------

def _filter_production_chunks(
    chunks: List[Dict[str, Any]],
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    """
    Remove test-related chunks so detection focuses on production code.

    Two filter rules:
    - File path contains an excluded directory name (tests/, venv/, etc.)
    - Function name starts with an excluded prefix (test_, conftest)
    """
    kept    = []
    removed = 0

    for chunk in chunks:
        path = chunk["file_path"]

        if any(excl in path for excl in config.excluded_dirs):
            removed += 1
            continue

        if any(chunk["function_name"].startswith(p)
               for p in config.excluded_prefixes):
            removed += 1
            continue

        kept.append(chunk)

    logger.info(
        "Production filter: kept %d chunks, removed %d",
        len(kept), removed,
    )
    return kept


def _save_json(data: Any, path: Path) -> None:
    """Write data as formatted JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)


def _high_confidence(
    issues:    List[Dict[str, Any]],
    threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """Return only issues whose confidence score meets the threshold."""
    return [i for i in issues if i.get("confidence", 0) >= threshold]


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(repo_url: str, config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """
    Execute the full 13-stage analysis pipeline.

    Each stage is logged with its name and elapsed time.
    Novel stages are labelled in the module docstring above.

    Parameters
    ----------
    repo_url : Full HTTPS URL of the GitHub repository.
    config   : Immutable pipeline configuration. Defaults to
               DEFAULT_CONFIG — override specific fields via
               dataclasses.replace() rather than mutating.
    """

    sep = "─" * 54
    logger.info(sep)
    logger.info("AI CODE ANALYZER — starting pipeline")
    logger.info("Repository : %s", repo_url)
    logger.info("LLM        : %s (%s)", config.llm_provider,
                "enabled" if config.llm_available else "disabled")
    logger.info(sep)

    # ── Stage 1: Clone ────────────────────────────────────────────
    logger.info("[1/13] Cloning repository")
    with _timed("clone"):
        repo_path = clone_repository(repo_url, config)

    # ── Stage 2: Scan ─────────────────────────────────────────────
    # Scan BEFORE call graph so taint analysis only sees
    # production files — not test files. This was the root cause
    # of 70 false taint flows in Flask (test files were scanned).
    logger.info("[2/13] Scanning Python files")
    with _timed("scan"):
        python_files = scan_python_files(repo_path, config)
    logger.info("       Found %d files", len(python_files))

    # ── Stage 3: Taint analysis (novel) ───────────────────────────
    # Uses python_files — already filtered. No test files included.
    logger.info("[3/13] Building call graph — taint analysis")
    taint_issues: List[Dict[str, Any]] = []
    try:
        from analysis.call_graph import CallGraphBuilder
        with _timed("call graph + taint"):
            cg           = CallGraphBuilder().build(python_files)
            taint_issues = cg.find_source_to_sink_paths()
        logger.info(
            "       Sources: %d  Sinks: %d  Taint flows: %d",
            len(cg.sources), len(cg.sinks), len(taint_issues),
        )
    except Exception as exc:
        logger.warning("Taint analysis skipped (non-fatal): %s", exc)

    # ── Stage 4: Extract ──────────────────────────────────────────
    logger.info("[4/13] Extracting functions via AST")
    with _timed("extract"):
        functions = extract_functions_from_files(python_files)
    logger.info("       Extracted %d functions", len(functions))

    # ── Stage 5: Chunk ────────────────────────────────────────────
    logger.info("[5/13] Creating code chunks")
    with _timed("chunk"):
        chunks = chunk_functions(functions, config)
    logger.info("       Generated %d chunks", len(chunks))

    # ── Stage 6: Filter ───────────────────────────────────────────
    logger.info("[6/13] Filtering to production code")
    with _timed("filter"):
        chunks = _filter_production_chunks(chunks, config)

    # ── Stage 7: Embed ────────────────────────────────────────────
    # The embedder is stored and passed into BugDetector (stage 8)
    # so the model is loaded exactly once per pipeline run.
    logger.info("[7/13] Generating embeddings")
    with _timed("embed"):
        embedder = CodeEmbedder(config)
        chunks   = embed_chunks(chunks, embedder)

    # ── Stage 8: Rule-based + semantic detection ───────────────────
    logger.info("[8/13] Rule-based + semantic detection")
    with _timed("rule + semantic"):
        detector = BugDetector(config, embedder=embedder)
        issues   = detector.analyze_chunks(chunks)

    # ── Stage 9: Anomaly detection (novel) ────────────────────────
    logger.info("[9/13] Statistical anomaly detection")
    anomaly_issues: List[Dict[str, Any]] = []
    try:
        from analysis.anomaly_detector import AnomalyDetector
        with _timed("anomaly detection"):
            ad             = AnomalyDetector(contamination=0.05)
            ad.fit(chunks)
            anomaly_issues = ad.find_anomalies(chunks)
            issues.extend(anomaly_issues)
        logger.info("       Anomalies found: %d", len(anomaly_issues))
    except Exception as exc:
        logger.warning("Anomaly detection skipped (non-fatal): %s", exc)

    # ── Stage 10: Codebase DNA (novel) ────────────────────────────
    logger.info("[10/13] Codebase DNA fingerprinting")
    dna_issues: List[Dict[str, Any]] = []
    try:
        from analysis.codebase_dna import CodebaseDNA
        with _timed("codebase DNA"):
            dna        = CodebaseDNA()
            profile    = dna.analyze(chunks)
            dna_issues = dna.violations_as_issues(profile)
            issues.extend(dna_issues)
        logger.info(profile.summary())
        logger.info("       Convention violations: %d", len(dna_issues))
    except Exception as exc:
        logger.warning("DNA analysis skipped (non-fatal): %s", exc)

    # Add taint issues last — they have no chunk_id in chunks
    # so they skip patch generation and go straight to output
    issues.extend(taint_issues)

    # Confidence breakdown
    high_conf = _high_confidence(issues, threshold=0.80)
    logger.info(
        "       Total issues: %d  High confidence (>=0.80): %d",
        len(issues), len(high_conf),
    )

    # ── Stage 11: Patch generation ────────────────────────────────
    logger.info("[11/13] Generating patch suggestions")
    with _timed("patches"):
        patcher = PatchGenerator(config)
        patches = patcher.generate_patches(issues, chunks)
    logger.info("       Generated %d patches", len(patches))

    # ── Stage 12: Git archaeology (novel) ─────────────────────────
    logger.info("[12/13] Git archaeology — tracing bug origins")
    try:
        from analysis.git_archaeologist import GitArchaeologist
        with _timed("git archaeology"):
            arch = GitArchaeologist(repo_path)

            if arch._usable:
                # Only investigate high-confidence issues — archaeology
                # runs git subprocesses so we cap at 20 to stay fast
                to_investigate = _high_confidence(issues, 0.80)[:20]
                enriched       = arch.investigate_batch(to_investigate)

                # Merge archaeology results back onto the issue list
                enriched_map = {
                    e["chunk_id"]: e.get("archaeology")
                    for e in enriched
                }
                for issue in issues:
                    cid = issue.get("chunk_id")
                    if cid in enriched_map:
                        issue["archaeology"] = enriched_map[cid]

                found = sum(
                    1 for e in enriched
                    if e.get("archaeology", {}).get("investigated")
                )
                logger.info("       Investigated %d/%d issues", found, len(to_investigate))
            else:
                logger.info("       Skipped — repo has no git history")
    except Exception as exc:
        logger.warning("Git archaeology skipped (non-fatal): %s", exc)

    # ── Stage 13: Patch verification (novel) ──────────────────────
    logger.info("[13/13] Verifying patches against test suite")
    try:
        from analysis.patch_verifier import PatchVerifier
        with _timed("patch verification"):
            verifier = PatchVerifier(repo_path)
            for patch in patches:
                result = verifier.verify(patch)
                patch["verification"] = result

                # Log a one-line verdict per patch
                status = ("✅" if result["verified"] is True  else
                          "❌" if result["verified"] is False else "⚠️ ")
                logger.info(
                    "       %s  %-28s  %s",
                    status,
                    patch.get("function", "")[:28],
                    result["reason"][:50],
                )
    except Exception as exc:
        logger.warning("Patch verification skipped (non-fatal): %s", exc)

    # ── Evaluation report ─────────────────────────────────────────
    report = BenchmarkReport(chunks, issues, patches)
    report.print_report()

    # ── Save results ──────────────────────────────────────────────
    rd = config.results_dir
    _save_json(issues,            rd / "issues.json")
    _save_json(patches,           rd / "patches.json")
    _save_json(report.as_dict(),  rd / "benchmark.json")
    _save_json(high_conf,         rd / "issues_high_confidence.json")

    logger.info(sep)
    logger.info("ANALYSIS COMPLETE")
    logger.info(sep)


# ------------------------------------------------------------------
# Quick test — verify changes without a full repo run
# ------------------------------------------------------------------

def quick_test() -> None:
    """
    Run the pipeline on a tiny hand-crafted file.

    Completes in ~60 seconds. Use this after every change
    to confirm the pipeline still runs end to end before
    committing a 20-minute Flask run.

    What to watch for:
    - "Loading embedding model" should appear ONCE only
    - All 5 issue types should be detected
    - Confidence scores should be attached to every issue
    """

    test_dir = Path("repos/test_quick/src")
    test_dir.mkdir(parents=True, exist_ok=True)

    # One function per issue type — easy to verify detection
    (test_dir / "sample.py").write_text(
        "def divide(a, b):\n"
        "    return a / 0\n\n"
        "def load_data(items=[]):\n"
        "    items.append(1)\n"
        "    return items\n\n"
        "def run_code(user_input):\n"
        "    eval(user_input)\n\n"
        "def fetch():\n"
        "    try:\n"
        "        pass\n"
        "    except:\n"
        "        pass\n\n"
        'def debug_output():\n'
        '    print("debug value")\n'
    )

    print("\n[QUICK TEST MODE]\n")

    config   = DEFAULT_CONFIG
    files    = scan_python_files(Path("repos/test_quick"), config)
    funcs    = extract_functions_from_files(files)
    chunks   = chunk_functions(funcs, config)

    # Embedder created here — passed into detector below
    # Model should appear in logs exactly ONCE
    embedder = CodeEmbedder(config)
    chunks   = embed_chunks(chunks, embedder)

    detector = BugDetector(config, embedder=embedder)
    issues   = detector.analyze_chunks(chunks)

    print(f"Files  : {len(files)}")
    print(f"Chunks : {len(chunks)}")
    print(f"Issues : {len(issues)}")
    print()

    for issue in issues:
        conf = issue.get("confidence", 0)
        print(f"  [{conf:.0%}]  {issue['type']:<30}  {issue['function']}()")

    print("\n[DONE]\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Define the command-line interface.

    Each flag maps to a PipelineConfig override so the core
    pipeline logic never reads from sys.argv directly.
    """

    parser = argparse.ArgumentParser(
        prog="analyze_repo",
        description=(
            "AI Code Analyzer — taint analysis, anomaly detection, "
            "codebase DNA, git archaeology, patch verification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python analyze_repo.py https://github.com/pallets/flask
  python analyze_repo.py https://github.com/pallets/flask --provider groq
  python analyze_repo.py https://github.com/pallets/flask --min-confidence 0.80
  python analyze_repo.py https://github.com/pallets/flask --threshold 0.65
  python analyze_repo.py --test
        """,
    )

    parser.add_argument(
        "repo_url",
        nargs="?",
        help="GitHub repository URL to analyze",
    )
    parser.add_argument(
        "--provider",
        choices=["llamacpp", "anthropic", "groq", "ollama"],
        help="LLM provider for patch generation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Semantic similarity detection threshold (default: 0.75)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        dest="min_confidence",
        help="Only report issues at or above this confidence (e.g. 0.80)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory to save results (default: analysis_results/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show DEBUG level logs",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test on local sample code — no git clone needed",
    )

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

    # Build config overrides from CLI flags.
    # dataclasses.replace() produces a new frozen instance —
    # the original DEFAULT_CONFIG is never mutated.
    overrides: Dict[str, Any] = {}
    if args.provider:   overrides["llm_provider"]         = args.provider
    if args.threshold:  overrides["similarity_threshold"] = args.threshold
    if args.output:     overrides["results_dir"]          = Path(args.output)

    config = (dataclasses.replace(DEFAULT_CONFIG, **overrides)
              if overrides else DEFAULT_CONFIG)

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