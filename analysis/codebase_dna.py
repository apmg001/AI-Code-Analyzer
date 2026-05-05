# analysis/codebase_dna.py

"""
Module: codebase_dna

Responsibility
--------------
Learn the coding patterns of a specific codebase — its "DNA" —
and flag functions that violate those patterns.

The core insight
----------------
Every codebase has conventions that emerge organically:
    - This team uses logging, not print()
    - This team writes docstrings for every function
    - This team uses type hints throughout
    - This team handles errors with custom exceptions
    - This team writes snake_case everywhere

These conventions are never written down. They live in the code.
When a function violates them, it's usually a sign of one of:

    1. Code copied from a different project (style mismatch)
    2. A junior contributor who didn't know the conventions
    3. Dead code that predates the current conventions
    4. A rushed hotfix written under pressure
    5. An actual bug hiding in the inconsistency

This is different from anomaly detection (which uses embeddings)
and different from rule-based detection (which uses AST patterns).
DNA fingerprinting learns from THIS specific codebase — not from
general coding knowledge.

Why no existing tool does this
-------------------------------
pylint checks conventions against a global config file.
SonarQube checks against language-wide rules.
Neither learns the conventions OF YOUR SPECIFIC CODEBASE.

If your codebase uses print() everywhere, pylint will flag all of
them. DNA fingerprinting knows your convention IS print() and will
only flag the one function using logging — because that's the outlier.

Design decisions
----------------
- Each pattern is a separate class with a single responsibility.
  Adding a new pattern = writing a new class + registering it.
  Nothing else changes. Open/Closed in practice.

- CoveragePattern is a common base for patterns that measure
  "what % of functions follow convention X." Most patterns fit
  this shape — it avoids duplicated threshold logic.

- Patterns return PatternResult objects — typed, not plain dicts.
  The violation detection code iterates PatternResult lists so
  the schema is always consistent.

- The DNAProfile is built once per repo run and cached on the
  CodebaseDNA instance. Rebuilding is expensive (touches every chunk).

- Violation severity is always "low" — these are style and
  consistency findings, not security vulnerabilities. An engineer
  should review them, not panic about them.

Limitations
-----------
- Coverage patterns need at least 10 chunks to be meaningful.
  A 3-function codebase has no meaningful "convention."
- Naming convention detection is heuristic — it counts underscores
  and uppercase letters. It does not parse camelCase vs PascalCase.
- Pattern detection operates on raw source code strings, not AST.
  This is fast but means string matches can produce false positives
  in comments and docstrings.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class PatternResult:
    """
    The measured state of one convention across the codebase.

    dominant_value  : the most common choice (e.g. "logging", "snake_case")
    coverage        : 0.0-1.0 — how consistently it's followed
    is_consistent   : True when coverage crosses the threshold
    follower_count  : how many chunks follow the dominant convention
    violator_count  : how many chunks violate it
    """
    pattern_name:    str
    dominant_value:  str
    coverage:        float
    is_consistent:   bool
    follower_count:  int
    violator_count:  int
    threshold:       float

    def __repr__(self) -> str:
        return (
            f"PatternResult({self.pattern_name}: "
            f"{self.dominant_value} @ {self.coverage:.0%})"
        )


@dataclass
class DNAViolation:
    """
    A single function that violates a codebase convention.
    Maps directly to a DetectedIssue in the pipeline.
    """
    function:    str
    file:        str
    pattern:     str
    subtype:     str
    message:     str
    chunk_id:    str
    line_number: int = 0


@dataclass
class DNAProfile:
    """
    The complete DNA fingerprint of a codebase.

    Built once per analysis run from all chunks.
    Contains the measured conventions and any violations found.
    """
    total_chunks:  int
    patterns:      Dict[str, PatternResult]
    violations:    List[DNAViolation] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Codebase DNA — {self.total_chunks} chunks analyzed"]
        for name, result in self.patterns.items():
            consistency = "consistent" if result.is_consistent else "mixed"
            lines.append(
                f"  {name}: {result.dominant_value} "
                f"({result.coverage:.0%}, {consistency})"
            )
        return "\n".join(lines)


# ------------------------------------------------------------------
# Pattern base classes
# ------------------------------------------------------------------

class BasePattern(ABC):
    """
    Interface for all codebase patterns.

    Every pattern:
    1. Measures itself across all chunks → PatternResult
    2. Finds chunks that violate the measured convention → List[DNAViolation]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this pattern."""

    @abstractmethod
    def measure(self, chunks: List[Dict]) -> PatternResult:
        """Learn the convention from the codebase."""

    @abstractmethod
    def find_violations(
        self,
        chunks:  List[Dict],
        result:  PatternResult,
    ) -> List[DNAViolation]:
        """Find chunks that violate the measured convention."""


class CoveragePattern(BasePattern, ABC):
    """
    Base class for patterns that measure "% of functions following convention X."

    Subclasses only need to implement:
    - _chunk_follows(chunk) → True if this chunk follows the convention
    - _convention_name → what to call the "yes" side (e.g. "logging")
    - _violation_message → message when a chunk doesn't follow
    - threshold → minimum coverage to consider convention consistent
    """

    threshold: float = 0.70      # override in subclass if needed

    @property
    @abstractmethod
    def _convention_name(self) -> str:
        """What to call the convention when it IS followed."""

    @property
    @abstractmethod
    def _violation_message(self) -> str:
        """Message to show when a function violates this pattern."""

    @abstractmethod
    def _chunk_follows(self, chunk: Dict) -> bool:
        """Return True if this chunk follows the convention."""

    def measure(self, chunks: List[Dict]) -> PatternResult:
        if len(chunks) < 10:
            return PatternResult(
                pattern_name=self.name,
                dominant_value="unknown",
                coverage=0.0,
                is_consistent=False,
                follower_count=0,
                violator_count=0,
                threshold=self.threshold,
            )

        followers = [c for c in chunks if self._chunk_follows(c)]
        coverage  = len(followers) / len(chunks)

        return PatternResult(
            pattern_name=self.name,
            dominant_value=self._convention_name,
            coverage=round(coverage, 3),
            is_consistent=coverage >= self.threshold,
            follower_count=len(followers),
            violator_count=len(chunks) - len(followers),
            threshold=self.threshold,
        )

    def find_violations(
        self,
        chunks: List[Dict],
        result: PatternResult,
    ) -> List[DNAViolation]:
        """Flag chunks that DON'T follow the dominant convention."""

        if not result.is_consistent:
            return []

        violations = []

        for chunk in chunks:
            if not self._chunk_follows(chunk):
                violations.append(DNAViolation(
                    function=chunk["function_name"],
                    file=chunk["file_path"],
                    pattern=self.name,
                    subtype=f"{self.name}_violation",
                    message=(
                        f"Codebase convention is {self._convention_name} "
                        f"({result.coverage:.0%} of functions) — "
                        f"this function doesn't follow it. "
                        f"{self._violation_message}"
                    ),
                    chunk_id=chunk["chunk_id"],
                    line_number=chunk.get("start_line", 0),
                ))

        return violations


# ------------------------------------------------------------------
# Concrete patterns
# ------------------------------------------------------------------

class LoggingPattern(CoveragePattern):
    """
    Detects whether the codebase uses the logging module
    or print() for output — and flags inconsistencies.

    If 80%+ of functions use logger.* or logging.*, then
    functions using print() are flagged as violators.
    If 80%+ use print(), then logger.* usage is the outlier.
    """

    name       = "output_style"
    threshold  = 0.75

    def measure(self, chunks: List[Dict]) -> PatternResult:
        if len(chunks) < 10:
            return PatternResult("output_style", "unknown", 0.0, False, 0, 0, self.threshold)

        uses_logging = sum(1 for c in chunks if self._uses_logging(c["code"]))
        uses_print   = sum(1 for c in chunks if "print(" in c["code"])
        total        = len(chunks)

        # Whichever is more dominant defines the convention
        if uses_logging >= uses_print:
            dominant  = "logging"
            followers = uses_logging
        else:
            dominant  = "print"
            followers = uses_print

        coverage = followers / total

        return PatternResult(
            pattern_name="output_style",
            dominant_value=dominant,
            coverage=round(coverage, 3),
            is_consistent=coverage >= self.threshold,
            follower_count=followers,
            violator_count=total - followers,
            threshold=self.threshold,
        )

    def find_violations(self, chunks, result) -> List[DNAViolation]:
        if not result.is_consistent:
            return []

        violations = []

        for chunk in chunks:
            code = chunk["code"]

            if result.dominant_value == "logging":
                # Convention is logging — flag print() usage
                if "print(" in code and not self._uses_logging(code):
                    violations.append(DNAViolation(
                        function=chunk["function_name"],
                        file=chunk["file_path"],
                        pattern="output_style",
                        subtype="print_in_logging_codebase",
                        message=(
                            f"Codebase uses logging ({result.coverage:.0%} of functions) "
                            f"but this function uses print(). "
                            f"Replace with logger.debug() or logger.info()."
                        ),
                        chunk_id=chunk["chunk_id"],
                        line_number=chunk.get("start_line", 0),
                    ))

            else:
                # Convention is print — flag logging usage
                if self._uses_logging(code) and "print(" not in code:
                    violations.append(DNAViolation(
                        function=chunk["function_name"],
                        file=chunk["file_path"],
                        pattern="output_style",
                        subtype="logging_in_print_codebase",
                        message=(
                            f"Codebase uses print() ({result.coverage:.0%} of functions) "
                            f"but this function uses the logging module. "
                            f"Consider standardizing output style."
                        ),
                        chunk_id=chunk["chunk_id"],
                        line_number=chunk.get("start_line", 0),
                    ))

        return violations

    @property
    def _convention_name(self) -> str:
        return "logging"

    @property
    def _violation_message(self) -> str:
        return "Consider standardizing output style."

    def _chunk_follows(self, chunk: Dict) -> bool:
        return self._uses_logging(chunk["code"])

    @staticmethod
    def _uses_logging(code: str) -> bool:
        return "logger." in code or "logging." in code


class DocstringPattern(CoveragePattern):
    """
    If 70%+ of functions have docstrings, flag the ones that don't.
    Missing docstrings in a documented codebase are often signs of
    rushed code or functions that were never meant to be permanent.
    """

    name      = "docstring_coverage"
    threshold = 0.70

    @property
    def _convention_name(self) -> str:
        return "docstrings"

    @property
    def _violation_message(self) -> str:
        return "Add a docstring explaining purpose, parameters, and return value."

    def _chunk_follows(self, chunk: Dict) -> bool:
        code = chunk["code"]
        # Check for both triple-quote styles
        return '"""' in code or "'''" in code


class TypeHintPattern(CoveragePattern):
    """
    If 60%+ of functions use type hints, flag those that don't.
    Type hints in a typed codebase without annotations are a
    maintenance hazard — future readers don't know the contract.
    """

    name      = "type_hints"
    threshold = 0.60

    # Simple markers that suggest type hints are present
    _TYPE_HINT_RE = re.compile(r"(:\s*(int|str|float|bool|list|dict|tuple|Optional|List|Dict|Any|Path)|->\s*\w)")

    @property
    def _convention_name(self) -> str:
        return "type_hints"

    @property
    def _violation_message(self) -> str:
        return "Add parameter and return type annotations."

    def _chunk_follows(self, chunk: Dict) -> bool:
        return bool(self._TYPE_HINT_RE.search(chunk["code"]))


class ErrorHandlingPattern(CoveragePattern):
    """
    Detect whether the codebase raises custom exceptions
    (raise ValueError, raise CustomError) or generic ones
    (raise Exception).

    Inconsistent error handling is a maintenance smell — callers
    cannot selectively catch custom exceptions if some functions
    raise generic ones.
    """

    name      = "error_handling"
    threshold = 0.65

    _CUSTOM_RE  = re.compile(r"raise\s+[A-Z][a-zA-Z]*Error\s*\(")
    _GENERIC_RE = re.compile(r"raise\s+Exception\s*\(")

    def measure(self, chunks: List[Dict]) -> PatternResult:
        if len(chunks) < 10:
            return PatternResult("error_handling", "unknown", 0.0, False, 0, 0, self.threshold)

        uses_custom  = sum(1 for c in chunks if self._CUSTOM_RE.search(c["code"]))
        uses_generic = sum(1 for c in chunks if self._GENERIC_RE.search(c["code"]))
        total        = len([c for c in chunks if "raise" in c["code"]])

        if total == 0:
            return PatternResult("error_handling", "none", 1.0, True, 0, 0, self.threshold)

        if uses_custom >= uses_generic:
            dominant = "custom_exceptions"
            followers = uses_custom
        else:
            dominant = "generic_exceptions"
            followers = uses_generic

        coverage = followers / total if total > 0 else 0

        return PatternResult(
            pattern_name="error_handling",
            dominant_value=dominant,
            coverage=round(coverage, 3),
            is_consistent=coverage >= self.threshold,
            follower_count=followers,
            violator_count=total - followers,
            threshold=self.threshold,
        )

    def find_violations(self, chunks, result) -> List[DNAViolation]:
        if not result.is_consistent:
            return []
        if result.dominant_value == "custom_exceptions":
            return self._flag_generic(chunks, result)
        return []

    def _flag_generic(self, chunks, result) -> List[DNAViolation]:
        violations = []
        for chunk in chunks:
            if self._GENERIC_RE.search(chunk["code"]):
                violations.append(DNAViolation(
                    function=chunk["function_name"],
                    file=chunk["file_path"],
                    pattern="error_handling",
                    subtype="generic_exception_in_custom_codebase",
                    message=(
                        f"Codebase uses custom exceptions ({result.coverage:.0%}) "
                        f"but this function raises generic Exception(). "
                        f"Create a specific exception class for this error case."
                    ),
                    chunk_id=chunk["chunk_id"],
                    line_number=chunk.get("start_line", 0),
                ))
        return violations

    # CoveragePattern abstract methods — not used in custom measure/find
    @property
    def _convention_name(self) -> str:
        return "custom_exceptions"

    @property
    def _violation_message(self) -> str:
        return "Use a specific exception class."

    def _chunk_follows(self, chunk: Dict) -> bool:
        return bool(self._CUSTOM_RE.search(chunk["code"]))


class NamingConventionPattern(BasePattern):
    """
    Detect whether the codebase uses snake_case or camelCase
    for function names — and flag the outliers.

    In Python, snake_case is the PEP 8 standard, but some codebases
    (especially those migrated from Java or JS) use camelCase.
    Mixing both is always a maintenance problem.
    """

    name = "naming_convention"

    _CAMEL_RE = re.compile(r"[a-z][A-Z]")     # camelCase marker
    _SNAKE_RE = re.compile(r"[a-z]_[a-z]")    # snake_case marker

    def measure(self, chunks: List[Dict]) -> PatternResult:
        snake_count = sum(
            1 for c in chunks
            if self._SNAKE_RE.search(c["function_name"])
        )
        camel_count = sum(
            1 for c in chunks
            if self._CAMEL_RE.search(c["function_name"])
        )
        total = len(chunks)

        if total == 0:
            return PatternResult("naming_convention", "unknown", 0.0, False, 0, 0, 0.80)

        if snake_count >= camel_count:
            dominant  = "snake_case"
            followers = snake_count
        else:
            dominant  = "camelCase"
            followers = camel_count

        coverage = followers / total

        return PatternResult(
            pattern_name="naming_convention",
            dominant_value=dominant,
            coverage=round(coverage, 3),
            is_consistent=coverage >= 0.80,
            follower_count=followers,
            violator_count=total - followers,
            threshold=0.80,
        )

    def find_violations(self, chunks, result) -> List[DNAViolation]:
        if not result.is_consistent:
            return []

        violations = []

        for chunk in chunks:
            name         = chunk["function_name"]
            is_snake     = bool(self._SNAKE_RE.search(name))
            is_camel     = bool(self._CAMEL_RE.search(name))
            follows_conv = is_snake if result.dominant_value == "snake_case" else is_camel

            # Only flag if there's a clear violation — ignore short names
            if not follows_conv and len(name) > 4 and is_camel != is_snake:
                violations.append(DNAViolation(
                    function=name,
                    file=chunk["file_path"],
                    pattern="naming_convention",
                    subtype=f"naming_violates_{result.dominant_value}",
                    message=(
                        f"Codebase uses {result.dominant_value} "
                        f"({result.coverage:.0%} of functions) "
                        f"but '{name}' uses a different style. "
                        f"Rename to match codebase convention."
                    ),
                    chunk_id=chunk["chunk_id"],
                    line_number=chunk.get("start_line", 0),
                ))

        return violations


# ------------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------------

class CodebaseDNA:
    """
    Learns the coding conventions of a codebase
    and finds functions that violate them.

    Usage
    -----
    dna     = CodebaseDNA()
    profile = dna.analyze(embedded_chunks)

    # Read the profile
    print(profile.summary())

    # Get violations as pipeline issues
    issues = dna.violations_as_issues(profile)
    """

    def __init__(self):
        # Register all patterns here.
        # Adding a new pattern = instantiate it and append.
        self._patterns: List[BasePattern] = [
            LoggingPattern(),
            DocstringPattern(),
            TypeHintPattern(),
            ErrorHandlingPattern(),
            NamingConventionPattern(),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, chunks: List[Dict]) -> DNAProfile:
        """
        Build a complete DNA profile of the codebase.

        Measures all registered patterns and finds all violations.
        This is the main entry point — call this once per pipeline run.

        Parameters
        ----------
        chunks : All embedded chunks from the pipeline.

        Returns
        -------
        DNAProfile with patterns and violations populated.
        """

        if len(chunks) < 10:
            logger.info("Too few chunks (%d) for DNA analysis — skipping", len(chunks))
            return DNAProfile(total_chunks=len(chunks), patterns={})

        logger.info("Building codebase DNA from %d chunks", len(chunks))

        # Step 1: measure every pattern
        results: Dict[str, PatternResult] = {}
        for pattern in self._patterns:
            result = pattern.measure(chunks)
            results[pattern.name] = result
            logger.debug("%s", result)

        # Step 2: find violations for each consistent pattern
        all_violations: List[DNAViolation] = []
        for pattern in self._patterns:
            result     = results[pattern.name]
            violations = pattern.find_violations(chunks, result)
            all_violations.extend(violations)
            if violations:
                logger.debug(
                    "%s: %d violations found",
                    pattern.name, len(violations),
                )

        logger.info(
            "DNA analysis complete — %d patterns measured, %d violations found",
            len(results), len(all_violations),
        )

        return DNAProfile(
            total_chunks=len(chunks),
            patterns=results,
            violations=all_violations,
        )

    def violations_as_issues(self, profile: DNAProfile) -> List[Dict]:
        """
        Convert DNAViolation objects into the standard issue dict format
        used throughout the pipeline.

        This lets DNA violations flow into the same patch generator
        and benchmark report as all other issue types.
        """
        issues = []

        for v in profile.violations:
            issues.append({
                "type":          "dna_violation",
                "severity":      "low",
                "confidence":    0.75,
                "function":      v.function,
                "file":          v.file,
                "line_number":   v.line_number,
                "code_snippet":  "",
                "chunk_id":      v.chunk_id,
                "message":       v.message,
                "subtype":       v.subtype,
                "pattern":       v.pattern,
                "similarity_score": None,
            })

        return issues