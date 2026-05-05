# analysis/patch_verifier.py

"""
Module: patch_verifier

Responsibility
--------------
Verify whether a generated patch actually fixes an issue
without breaking the existing test suite.

The problem this solves
-----------------------
Every AI code tool — Copilot, Cursor, Codeium — generates fix
suggestions. None of them verify if the fix actually works.

A developer who blindly applies a patch that breaks 3 tests
has lost more time than the original bug cost them.

This module closes that gap by running the repo's own test suite
before and after applying a patch — and rejecting it automatically
if tests get worse.

How it works
------------
1. Run existing test suite → record baseline pass/fail count
2. Copy repo to a temp directory → never touch real source
3. Apply the patch to the temp copy
4. Run tests again on temp copy
5. Compare before vs after → return a structured verdict

Design decisions
----------------
- Temp directory for patching: if something goes wrong during
  patch application, the original repo is completely untouched.
  Zero risk of corrupting the developer's code.

- Timeout of 120s: 60s was too tight — Django's test suite
  takes ~90s on a MacBook Air M4. 120s is safe for most repos.

- String replacement for patching: intentionally simple — not
  a real diff algorithm. If the LLM rewrites the function so
  completely that the original text is unrecognizable, we skip
  verification rather than corrupt the file. Safer to say
  "couldn't verify" than to break something.

- TestRunResult is a dataclass not a plain dict: callers get
  type safety, IDE autocomplete, and the .clean property.

- PatchVerifier never raises: every failure mode is captured
  in the returned dict under "reason". The pipeline must never
  crash because a test suite failed to run.

- Each concern is its own class (Single Responsibility):
    PytestOutputParser  → parsing only
    PatchApplicator     → file writing only
    TestRunner          → subprocess only
    PatchVerifier       → orchestration only

Limitations (honest — mention these in interviews)
---------------------------------------------------
- Only works with pytest. unittest, nose not supported yet.
- Only handles function-level patches, not multi-file changes.
- String replacement fails if LLM changes the function signature.
- Flaky tests will cause false negatives — patch rejected unfairly.
- Slow test suites (>120s) cannot be verified.
"""

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structure
# ------------------------------------------------------------------

@dataclass
class TestRunResult:
    """
    Structured result from one pytest execution.

    A dataclass instead of a plain dict so:
    - Callers get IDE autocomplete on field names
    - Typos in field names are caught at definition time
    - The .clean and .total properties stay close to the data
    """

    runnable: bool   # did pytest actually execute?
    passed:   int    # tests that passed
    failed:   int    # tests that failed
    errors:   int    # collection/import errors
    output:   str    # raw pytest stdout (truncated to 600 chars)

    @property
    def total(self) -> int:
        """Total tests seen by pytest."""
        return self.passed + self.failed + self.errors

    @property
    def clean(self) -> bool:
        """True only if everything passed with zero errors."""
        return self.runnable and self.failed == 0 and self.errors == 0


# ------------------------------------------------------------------
# Output parser
# ------------------------------------------------------------------

class PytestOutputParser:
    """
    Parse raw pytest stdout/stderr into a TestRunResult.

    Kept separate from TestRunner so parsing logic can be unit
    tested without spawning subprocesses.

    pytest summary line examples:
        5 passed
        3 passed, 2 failed
        1 passed, 1 failed, 2 errors
        no tests ran
    """

    _PASSED_RE = re.compile(r"(\d+) passed")
    _FAILED_RE = re.compile(r"(\d+) failed")
    _ERROR_RE  = re.compile(r"(\d+) error")

    def parse(self, stdout: str, stderr: str) -> TestRunResult:
        """
        Parse combined pytest output into a TestRunResult.

        Parameters
        ----------
        stdout : Raw standard output from pytest process.
        stderr : Raw standard error from pytest process.
        """
        combined = stdout + stderr

        passed = self._extract(self._PASSED_RE, combined)
        failed = self._extract(self._FAILED_RE, combined)
        errors = self._extract(self._ERROR_RE,  combined)

        runnable = any(
            word in combined
            for word in ("passed", "failed", "error", "no tests ran")
        )

        return TestRunResult(
            runnable=runnable,
            passed=passed,
            failed=failed,
            errors=errors,
            output=combined[:600],
        )

    @staticmethod
    def _extract(pattern: re.Pattern, text: str) -> int:
        """Extract first integer match from text, or 0."""
        match = pattern.search(text)
        return int(match.group(1)) if match else 0


# ------------------------------------------------------------------
# Patch applicator
# ------------------------------------------------------------------

class PatchApplicator:
    """
    Apply a generated patch to a target file on disk.

    Strategy: find the original function body in the file via
    string search and replace it with the suggested fix.

    This is a deliberate simplification — not a real diff/patch
    algorithm. It works when the LLM returns a modified version
    of the same function body. If the LLM rewrites from scratch
    (signature changed, structure different), the search will
    fail and the file is left untouched.

    Failing silently is the right behaviour here: we'd rather
    report "couldn't apply patch" than corrupt a source file.
    """

    def apply(self, file_path: Path, patch: Dict) -> bool:
        """
        Apply patch dict to file_path in place.

        Parameters
        ----------
        file_path : Absolute path to the target file.
                    Should be inside a sandbox — not the real repo.
        patch     : Dict with 'original_code' and 'suggested_fix' keys.

        Returns
        -------
        True  — patch applied successfully.
        False — original code not found, file unchanged.
        """

        original_code = patch.get("original_code", "").strip()
        suggested_fix = patch.get("suggested_fix", "").strip()

        if not original_code or not suggested_fix:
            logger.debug("Patch missing required fields — skipping")
            return False

        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", file_path.name, exc)
            return False

        if original_code not in content:
            logger.debug(
                "Original code not found in %s — "
                "LLM may have changed the function signature",
                file_path.name,
            )
            return False

        patched = content.replace(original_code, suggested_fix, 1)
        file_path.write_text(patched, encoding="utf-8")
        logger.debug("Patch applied to %s", file_path.name)
        return True


# ------------------------------------------------------------------
# Test runner
# ------------------------------------------------------------------

class TestRunner:
    """
    Run pytest in a directory and return a parsed result.

    Encapsulates everything subprocess-related so the rest of the
    module doesn't care about process management, timeout handling,
    or output capture.
    """

    # 120 seconds: generous but bounded.
    # Django's test suite takes ~90s on a MacBook Air M4.
    TIMEOUT_SECONDS: int = 120

    def __init__(self, parser: PytestOutputParser):
        self._parser = parser

    def run(self, cwd: Path) -> TestRunResult:
        """
        Run pytest in `cwd` and return parsed results.

        Never raises — every failure is captured in the result.
        A pipeline should never crash because tests didn't run.

        Parameters
        ----------
        cwd : Directory to run pytest in. Usually the repo root
              or the sandbox root.
        """

        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "--tb=no", "-q", "--no-header"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=self.TIMEOUT_SECONDS,
            )
            return self._parser.parse(proc.stdout, proc.stderr)

        except subprocess.TimeoutExpired:
            logger.warning(
                "Test suite timed out after %ds — increase TIMEOUT_SECONDS "
                "if this repo has a slow test suite",
                self.TIMEOUT_SECONDS,
            )
            return TestRunResult(
                runnable=False, passed=0, failed=0, errors=0,
                output=f"Timed out after {self.TIMEOUT_SECONDS}s",
            )

        except FileNotFoundError:
            logger.warning(
                "pytest not found — install with: pip install pytest"
            )
            return TestRunResult(
                runnable=False, passed=0, failed=0, errors=0,
                output="pytest not found",
            )

        except Exception as exc:
            logger.warning("Unexpected error running tests: %s", exc)
            return TestRunResult(
                runnable=False, passed=0, failed=0, errors=0,
                output=str(exc)[:200],
            )


# ------------------------------------------------------------------
# Main verifier
# ------------------------------------------------------------------

class PatchVerifier:
    """
    Verify that a generated patch is safe to apply.

    Runs the repo's own test suite before and after patching,
    then returns a structured verdict with confidence adjustment.

    Three possible verdicts:
        verified = True   → tests stable, patch is safe ✅
        verified = False  → tests degraded, patch rejected ❌
        verified = None   → no test suite found, cannot verify ⚠️

    Usage
    -----
    verifier = PatchVerifier(repo_path)

    for patch in patches:
        result = verifier.verify(patch)
        patch["verification"] = result

        if result["verified"] is True:
            print(f"Safe: {result['reason']}")
        elif result["verified"] is False:
            print(f"Rejected: {result['reason']}")
        else:
            print(f"Unknown: {result['reason']}")
    """

    def __init__(self, repo_path: Path):
        self._repo_path  = repo_path
        self._parser     = PytestOutputParser()
        self._runner     = TestRunner(self._parser)
        self._applicator = PatchApplicator()

        # Cache the baseline so we only run it once per PatchVerifier
        # instance — not once per patch. This saves significant time
        # when verifying many patches on the same repo.
        self._baseline: Optional[TestRunResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, patch: Dict) -> Dict:
        """
        Verify one patch against the test suite.

        Parameters
        ----------
        patch : Patch dict from PatchGenerator.
                Must have: file, original_code, suggested_fix.

        Returns
        -------
        Dict with:
            verified        : True | False | None
            reason          : Human-readable verdict string
            baseline_pass   : Tests passing before patch
            baseline_fail   : Tests failing before patch
            after_pass      : Tests passing after patch (None if unverifiable)
            after_fail      : Tests failing after patch (None if unverifiable)
            confidence_boost: +0.15 if verified, -0.30 if rejected, 0 otherwise
        """

        # Use cached baseline — run tests once, reuse across all patches
        if self._baseline is None:
            logger.info("Running baseline test suite (cached for subsequent patches)")
            self._baseline = self._runner.run(self._repo_path)

        baseline = self._baseline

        if not baseline.runnable:
            return self._build_no_suite_result()

        logger.debug(
            "Baseline: %d passed, %d failed, %d errors",
            baseline.passed, baseline.failed, baseline.errors,
        )

        # Apply patch in sandbox and retest
        after = self._test_in_sandbox(patch)

        if after is None:
            return self._build_unapplied_result(baseline)

        return self._build_verdict(baseline, after)

    def verify_batch(self, patches: List[Dict]) -> List[Dict]:
        """
        Verify multiple patches, reusing the baseline test run.

        The baseline is run once (cached), then each patch is
        tested in its own sandbox. Much faster than calling
        verify() in a loop when there are many patches.
        """
        results = []

        for patch in patches:
            result = self.verify(patch)
            results.append({**patch, "verification": result})

            status = ("✅" if result["verified"] is True  else
                      "❌" if result["verified"] is False else "⚠️ ")
            logger.info(
                "%s  %-30s %s",
                status,
                patch.get("function", "unknown")[:30],
                result["reason"][:55],
            )

        verified = sum(1 for r in results
                      if r["verification"]["verified"] is True)
        rejected = sum(1 for r in results
                      if r["verification"]["verified"] is False)

        logger.info(
            "Verification complete — %d verified, %d rejected, %d unverifiable",
            verified, rejected, len(results) - verified - rejected,
        )

        return results

    # ------------------------------------------------------------------
    # Sandbox test run
    # ------------------------------------------------------------------

    def _test_in_sandbox(self, patch: Dict) -> Optional[TestRunResult]:
        """
        Copy the repo to a temp dir, apply patch, run tests, discard.

        The temp directory is created and destroyed within this method.
        If anything goes wrong, the original repo is untouched.

        Returns None if the patch could not be applied.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp) / "repo"

            try:
                shutil.copytree(self._repo_path, sandbox)
            except Exception as exc:
                logger.warning("Failed to copy repo to sandbox: %s", exc)
                return None

            target = self._resolve_target(sandbox, patch)
            if target is None:
                return None

            applied = self._applicator.apply(target, patch)
            if not applied:
                logger.info(
                    "Patch not applied for %s — original code not found in file",
                    patch.get("function", "unknown"),
                )
                return None

            logger.debug("Patch applied in sandbox — running tests")
            return self._runner.run(sandbox)

    def _resolve_target(self, sandbox: Path, patch: Dict) -> Optional[Path]:
        """
        Translate the patch's file path into its sandbox equivalent.

        The patch stores an absolute path from the original repo.
        We need the same relative path but inside the sandbox.
        """
        original_file = patch.get("file", "")

        if not original_file:
            logger.warning("Patch missing 'file' field — cannot resolve target")
            return None

        try:
            relative = Path(original_file).relative_to(self._repo_path)
            return sandbox / relative
        except ValueError:
            # file was already stored as a relative path
            return sandbox / original_file

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_verdict(
        self,
        baseline: TestRunResult,
        after:    TestRunResult,
    ) -> Dict:
        """
        Compare before and after test runs and return a verdict.

        The patch is considered safe if it does not make things worse:
        - passed count stays the same or improves
        - failed count stays the same or improves
        - error count stays the same or improves
        """
        improved = (
            after.passed >= baseline.passed and
            after.failed <= baseline.failed and
            after.errors <= baseline.errors
        )

        if improved:
            reason = (
                f"Tests stable after patch "
                f"({after.passed} passed, {after.failed} failed) — fix is safe ✅"
            )
        else:
            reason = (
                f"Tests degraded — "
                f"before: {baseline.passed}p/{baseline.failed}f, "
                f"after: {after.passed}p/{after.failed}f — fix rejected ❌"
            )

        return {
            "verified":         improved,
            "baseline_pass":    baseline.passed,
            "baseline_fail":    baseline.failed,
            "after_pass":       after.passed,
            "after_fail":       after.failed,
            "confidence_boost": 0.15 if improved else -0.30,
            "reason":           reason,
        }

    def _build_no_suite_result(self) -> Dict:
        return {
            "verified":         None,
            "baseline_pass":    0,
            "baseline_fail":    0,
            "after_pass":       None,
            "after_fail":       None,
            "confidence_boost": 0.0,
            "reason": (
                "No runnable pytest suite found — "
                "add tests to enable patch verification"
            ),
        }

    def _build_unapplied_result(self, baseline: TestRunResult) -> Dict:
        return {
            "verified":         None,
            "baseline_pass":    baseline.passed,
            "baseline_fail":    baseline.failed,
            "after_pass":       None,
            "after_fail":       None,
            "confidence_boost": 0.0,
            "reason": (
                "Patch could not be applied — "
                "original code not found in target file"
            ),
        }