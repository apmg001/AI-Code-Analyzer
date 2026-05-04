# evaluation/benchmark.py

"""
Module: benchmark

Responsibility
--------------
Produce a structured evaluation report after analysis is complete.

Metrics reported
----------------
- Total chunks scanned
- Total issues detected
- Issue breakdown by severity (high / medium / low)
- Issue breakdown by type
- Detection rate (issues per 100 chunks)
- Patch coverage (what % of issues have a patch)
- Patch source breakdown (LLM vs rule-based)
- Top 5 most problematic files

Design notes
------------
- BenchmarkReport is initialised with raw data and exposes
  two methods: as_dict() for programmatic use and print_report()
  for terminal output. No business logic runs at init time.
- All formatting is in _format_* helper methods so the data
  layer and presentation layer stay separate.
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BenchmarkReport:
    """
    Computes and presents pipeline evaluation metrics.

    Parameters
    ----------
    chunks  : All embedded chunks that were analyzed.
    issues  : All detected issues.
    patches : All generated patches (optional — pass [] to skip patch metrics).
    """

    def __init__(
        self,
        chunks:  List[Dict[str, Any]],
        issues:  List[Dict[str, Any]],
        patches: List[Dict[str, Any]] = None,
    ):
        self._chunks  = chunks
        self._issues  = issues
        self._patches = patches or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """Return all metrics as a serialisable dict."""
        return {
            "summary":           self._summary(),
            "severity_breakdown": self._severity_breakdown(),
            "type_breakdown":    self._type_breakdown(),
            "patch_metrics":     self._patch_metrics(),
            "top_files":         self._top_files(),
        }

    def print_report(self) -> None:
        """Print a formatted report to stdout."""

        divider   = "─" * 52
        h_divider = "═" * 52

        summary     = self._summary()
        severity    = self._severity_breakdown()
        types       = self._type_breakdown()
        patch_stats = self._patch_metrics()
        top_files   = self._top_files()

        print(f"\n{h_divider}")
        print("  AI CODE ANALYZER — EVALUATION REPORT")
        print(h_divider)

        print(f"\n{'SUMMARY':}")
        print(f"  Chunks analyzed      : {summary['total_chunks']:>6}")
        print(f"  Issues detected      : {summary['total_issues']:>6}")
        print(f"  Detection rate       : {summary['detection_rate_per_100']:>5.1f} per 100 chunks")

        print(f"\n{divider}")
        print("SEVERITY BREAKDOWN")
        for level in ("high", "medium", "low"):
            count = severity.get(level, 0)
            bar   = "█" * min(count, 30)
            print(f"  {level.upper():<8} {count:>4}  {bar}")

        print(f"\n{divider}")
        print("ISSUE TYPES")
        for issue_type, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  {issue_type:<35} {count:>4}")

        if self._patches:
            print(f"\n{divider}")
            print("PATCH COVERAGE")
            print(f"  Issues with patches  : {patch_stats['patched_issues']:>4} / {summary['total_issues']}")
            print(f"  Coverage             : {patch_stats['coverage_pct']:>5.1f}%")
            print(f"  LLM patches          : {patch_stats['llm_count']:>4}")
            print(f"  Rule-based patches   : {patch_stats['rule_count']:>4}")

        if top_files:
            print(f"\n{divider}")
            print("TOP FILES BY ISSUE COUNT")
            for i, (filepath, count) in enumerate(top_files, start=1):
                short = filepath.split("/")[-1]
                print(f"  {i}. {short:<40} {count:>3} issues")

        print(f"\n{h_divider}\n")

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    def _summary(self) -> Dict[str, Any]:
        total_chunks = len(self._chunks)
        total_issues = len(self._issues)
        rate = (total_issues / total_chunks * 100) if total_chunks > 0 else 0.0

        return {
            "total_chunks":            total_chunks,
            "total_issues":            total_issues,
            "detection_rate_per_100":  round(rate, 2),
        }

    def _severity_breakdown(self) -> Dict[str, int]:
        return dict(Counter(i.get("severity", "unknown") for i in self._issues))

    def _type_breakdown(self) -> Dict[str, int]:
        return dict(Counter(i.get("type", "unknown") for i in self._issues))

    def _patch_metrics(self) -> Dict[str, Any]:
        if not self._patches:
            return {
                "patched_issues": 0,
                "coverage_pct":   0.0,
                "llm_count":      0,
                "rule_count":     0,
            }

        total   = len(self._issues)
        patched = sum(1 for p in self._patches if p.get("patch_source") != "none")
        rule    = sum(1 for p in self._patches if p.get("patch_source") == "rule_based")
        llm     = sum(1 for p in self._patches if p.get("patch_source") not in ("rule_based", "none", None))
        pct     = (patched / total * 100) if total > 0 else 0.0

        return {
            "patched_issues": patched,
            "coverage_pct":   round(pct, 1),
            "llm_count":      llm,
            "rule_count":     rule,
        }

    def _top_files(self, n: int = 5) -> List[tuple]:
        file_counts: Dict[str, int] = defaultdict(int)
        for issue in self._issues:
            file_counts[issue.get("file", "unknown")] += 1
        return sorted(file_counts.items(), key=lambda x: -x[1])[:n]
