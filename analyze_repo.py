# analyze_repo.py

"""
AI Code Analyzer — Pipeline Entry Point

14 stages. 5 novel capabilities + RAG-augmented patch generation:

    Stage  3 — Taint analysis       tracks data flow across functions
    Stage  7 — Embeddings           sentence-transformers
    Stage  8 — RAG indexing         ChromaDB HNSW vector store  ← NEW
    Stage 10 — Anomaly detection    IsolationForest
    Stage 11 — Codebase DNA         convention fingerprinting
    Stage 13 — Git archaeology      bug origin tracing
    Stage 14 — Patch verification   test-suite validation

Usage
-----
    python analyze_repo.py <repo_url>
    python analyze_repo.py <repo_url> --provider groq
    python analyze_repo.py <repo_url> --min-confidence 0.80
    python analyze_repo.py --test

Design notes
------------
- Scan (stage 2) runs before call graph (stage 3) so taint
  analysis only sees production files — not test files.

- The embedder is created once in stage 7 and injected into
  BugDetector (stage 9) to avoid loading the model twice.

- RAGEngine is created after embedding (stage 8) and injected
  into PatchGenerator (stage 12). If chromadb is not installed
  PatchGenerator falls back to standard prompts transparently.

- Every novel stage is wrapped in try/except so a failure in
  one does not crash the entire pipeline.
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
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

@contextmanager
def _timed(label: str):
    """Log how long a pipeline stage takes."""
    start = time.time()
    yield
    logger.info("       %-38s %.1fs", label, time.time() - start)


def _filter_production_chunks(
    chunks: List[Dict[str, Any]],
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    """Remove test files and test functions from chunks."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)


def _high_confidence(
    issues:    List[Dict[str, Any]],
    threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    return [i for i in issues if i.get("confidence", 0) >= threshold]


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def run_pipeline(
    repo_url: str,
    config:   PipelineConfig = DEFAULT_CONFIG,
) -> None:
    """
    Execute the full 14-stage analysis pipeline.

    Each stage is logged with its name and elapsed time.
    """

    sep = "─" * 54
    logger.info(sep)
    logger.info("AI CODE ANALYZER — starting pipeline")
    logger.info("Repository : %s", repo_url)
    logger.info("LLM        : %s (%s)", config.llm_provider,
                "enabled" if config.llm_available else "disabled")
    logger.info(sep)

    # ── Stage 1: Clone ────────────────────────────────────────────
    logger.info("[1/14] Cloning repository")
    with _timed("clone"):
        repo_path = clone_repository(repo_url, config)

    # ── Stage 2: Scan ─────────────────────────────────────────────
    # Scan BEFORE call graph so taint analysis uses filtered files
    logger.info("[2/14] Scanning Python files")
    with _timed("scan"):
        python_files = scan_python_files(repo_path, config)
    logger.info("       Found %d files", len(python_files))

    # ── Stage 3: Taint analysis ───────────────────────────────────
    logger.info("[3/14] Building call graph — taint analysis")
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
        logger.warning("Taint analysis skipped: %s", exc)

    # ── Stage 4: Extract ──────────────────────────────────────────
    logger.info("[4/14] Extracting functions via AST")
    with _timed("extract"):
        functions = extract_functions_from_files(python_files)
    logger.info("       Extracted %d functions", len(functions))

    # ── Stage 5: Chunk ────────────────────────────────────────────
    logger.info("[5/14] Creating code chunks")
    with _timed("chunk"):
        chunks = chunk_functions(functions, config)
    logger.info("       Generated %d chunks", len(chunks))

    # ── Stage 6: Filter ───────────────────────────────────────────
    logger.info("[6/14] Filtering to production code")
    with _timed("filter"):
        chunks = _filter_production_chunks(chunks, config)

    # ── Stage 7: Embed ────────────────────────────────────────────
    # Embedder created once and reused by BugDetector (stage 9)
    # to avoid loading the model a second time.
    logger.info("[7/14] Generating embeddings")
    with _timed("embed"):
        embedder = CodeEmbedder(config)
        chunks   = embed_chunks(chunks, embedder)

    # ── Stage 8: RAG indexing ─────────────────────────────────────
    # Index chunks in ChromaDB so PatchGenerator can retrieve
    # style-similar code when building LLM prompts.
    logger.info("[8/14] RAG — indexing chunks in ChromaDB")
    rag_engine = None
    try:
        from embeddings.rag_engine import RAGEngine
        with _timed("rag index"):
            rag_engine = RAGEngine(repo_url=repo_url)
            indexed    = rag_engine.index(chunks)
        logger.info(
            "       RAG: %d new chunks indexed (%d total)",
            indexed, rag_engine.indexed_count,
        )
    except Exception as exc:
        logger.warning("RAG indexing skipped (non-fatal): %s", exc)

    # ── Stage 9: Rule-based + semantic detection ───────────────────
    logger.info("[9/14] Rule-based + semantic detection")
    with _timed("rule + semantic"):
        # Inject existing embedder — avoids loading model twice
        detector = BugDetector(config, embedder=embedder)
        issues   = detector.analyze_chunks(chunks)

    # ── Stage 10: Anomaly detection ───────────────────────────────
    logger.info("[10/14] Statistical anomaly detection")
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
        logger.warning("Anomaly detection skipped: %s", exc)

    # ── Stage 11: Codebase DNA ────────────────────────────────────
    logger.info("[11/14] Codebase DNA fingerprinting")
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
        logger.warning("DNA analysis skipped: %s", exc)

    # Add taint issues
    issues.extend(taint_issues)

    # Confidence summary
    high_conf = _high_confidence(issues, 0.80)
    logger.info(
        "       Total issues: %d  High confidence (>=0.80): %d",
        len(issues), len(high_conf),
    )

    # ── Stage 12: Patch generation ────────────────────────────────
    # Inject RAG engine — if available, patches use style-aware prompts
    logger.info("[12/14] Generating patches")
    logger.info(
        "       RAG-augmented prompts: %s",
        "enabled" if (rag_engine and rag_engine.is_available) else "disabled",
    )
    with _timed("patches"):
        patcher = PatchGenerator(config, rag_engine=rag_engine)
        patches = patcher.generate_patches(issues, chunks)
    logger.info("       Generated %d patches", len(patches))

    # ── Stage 13: Git archaeology ─────────────────────────────────
    logger.info("[13/14] Git archaeology — tracing bug origins")
    try:
        from analysis.git_archaeologist import GitArchaeologist
        with _timed("git archaeology"):
            arch = GitArchaeologist(repo_path)
            if arch._usable:
                to_investigate = _high_confidence(issues, 0.80)[:20]
                enriched       = arch.investigate_batch(to_investigate)
                enriched_map   = {
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
                logger.info(
                    "       Investigated %d/%d issues",
                    found, len(to_investigate),
                )
            else:
                logger.info("       Skipped — no git history available")
    except Exception as exc:
        logger.warning("Git archaeology skipped: %s", exc)

    # ── Stage 14: Patch verification ──────────────────────────────
    logger.info("[14/14] Verifying patches against test suite")
    try:
        from analysis.patch_verifier import PatchVerifier
        with _timed("patch verification"):
            verifier = PatchVerifier(repo_path)
            for patch in patches:
                fix = patch.get("suggested_fix", "")
                is_template = any(phrase in fix for phrase in [
                    "Statistically unusual",
                    "Flagged as similar",
                    "convention violation",
                    "Taint path:",
                    "Manual review",
                ])
                if is_template:
                    continue

                result = verifier.verify(patch)
                patch["verification"] = result

                status = (
                    "✅" if result["verified"] is True  else
                    "❌" if result["verified"] is False else "⚠️ "
                )
                logger.info(
                    "       %s  %-28s  %s",
                    status,
                    patch.get("function", "")[:28],
                    result["reason"][:50],
                )
    except Exception as exc:
        logger.warning("Patch verification skipped: %s", exc)

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
# Quick test
# ------------------------------------------------------------------

def quick_test() -> None:
    """
    Run on a tiny local sample — completes in ~60 seconds.
    Tests the full pipeline without a git clone.

    Watch for:
    - "Loading embedding model" appears ONCE only
    - All issue types detected with confidence scores
    - RAG engine initialises (if chromadb is installed)
    """
    test_dir = Path("repos/test_quick/src")
    test_dir.mkdir(parents=True, exist_ok=True)

    (test_dir / "sample.py").write_text(
        "def divide(a, b):\n    return a / 0\n\n"
        "def load_data(items=[]):\n    items.append(1)\n    return items\n\n"
        "def run_code(user_input):\n    eval(user_input)\n\n"
        "def fetch():\n    try:\n        pass\n    except:\n        pass\n\n"
        'def debug_output():\n    print("debug")\n'
    )

    print("\n[QUICK TEST MODE]\n")

    config   = DEFAULT_CONFIG
    files    = scan_python_files(Path("repos/test_quick"), config)
    funcs    = extract_functions_from_files(files)
    chunks   = chunk_functions(funcs, config)
    embedder = CodeEmbedder(config)
    chunks   = embed_chunks(chunks, embedder)

    # Test RAG engine initialisation
    rag_engine = None
    try:
        from embeddings.rag_engine import RAGEngine
        rag_engine = RAGEngine(repo_url="test_quick")
        rag_engine.index(chunks)
        print(f"RAG    : {rag_engine.indexed_count} chunks indexed")
    except Exception as exc:
        print(f"RAG    : skipped ({exc})")

    # Embedder injected — model loads ONCE
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
    parser = argparse.ArgumentParser(
        prog="analyze_repo",
        description=(
            "AI Code Analyzer — taint analysis, anomaly detection, "
            "codebase DNA, RAG patches, git archaeology, patch verification."
        ),
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
        help="Results directory (default: analysis_results/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show DEBUG level logs",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test on local sample — no git clone needed",
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

    overrides: Dict[str, Any] = {}
    if args.provider:   overrides["llm_provider"]         = args.provider
    if args.threshold:  overrides["similarity_threshold"] = args.threshold
    if args.output:     overrides["results_dir"]          = Path(args.output)

    config = (
        dataclasses.replace(DEFAULT_CONFIG, **overrides)
        if overrides else DEFAULT_CONFIG
    )

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