# analysis/git_archaeologist.py

"""
Module: git_archaeologist

Responsibility
--------------
For any detected issue, find out WHEN it was introduced,
WHO introduced it, and WHAT the code looked like before.

The problem with every existing code analyzer
---------------------------------------------
Every static analysis tool — pylint, bandit, SonarQube, yours —
tells you WHAT is wrong. None of them tell you:

    - Which commit introduced this bug
    - Who wrote it and when
    - What the safe version looked like
    - How long this has been in production

Git has all of this information. It just needs to be connected
to the analysis pipeline. That connection is what this module builds.

Real-world value
----------------
Knowing a bug is 847 days old changes how a team responds to it.
Knowing it was introduced 3 days ago in a specific PR changes
the conversation completely — the author can fix it immediately
while the context is fresh.

This context is what separates a useful finding from a noisy one.

Algorithm
---------
1. git blame  → find which commit last touched the buggy line
2. git log    → get commit message, author, timestamp
3. git show   → get the file contents from the parent commit
4. Compare    → show what changed and flag what was safe before

Design decisions
----------------
- GitArchaeologist is stateless after __init__ — all methods are
  pure functions over their inputs. Easy to test, easy to reason about.

- git commands run via subprocess — no gitpython dependency.
  gitpython is heavy (~10MB) and often has version conflicts.
  subprocess with git CLI is lighter and more reliable.

- All git calls have a 10s timeout. On a large repo with a long
  history, git blame can be slow. We'd rather skip archaeology
  than block the pipeline for 60 seconds.

- Returns structured dicts, not objects — keeps the pipeline
  data model consistent (everything is JSON-serialisable).

- Every method catches exceptions and returns None rather than
  raising. Archaeology is best-effort — a missing git history
  should not crash the pipeline.

Limitations
-----------
- Only works on git repositories (not Mercurial, SVN, etc.)
- Shallow clones (--depth 1) have truncated history — git blame
  may not find the true origin commit
- Merge commits show the merger, not the original author
- Renamed files break git blame lineage
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

GIT_TIMEOUT  = 10     # seconds per git command — don't block the pipeline
MAX_LOG_LINES = 20    # how many commits to scan for context


# ------------------------------------------------------------------
# Blame result
# ------------------------------------------------------------------

class BlameRecord:
    """
    Structured result from git blame on a single line.

    Kept as a class (not dataclass) because the parsing
    logic lives here — it knows the git blame --porcelain format.
    """

    def __init__(
        self,
        commit_hash: str,
        author:      str,
        author_mail: str,
        timestamp:   int,
        summary:     str,
    ):
        self.commit_hash = commit_hash
        self.author      = author
        self.author_mail = author_mail
        self.timestamp   = timestamp
        self.summary     = summary       # first line of commit message

    @property
    def short_hash(self) -> str:
        return self.commit_hash[:8]

    @property
    def days_ago(self) -> int:
        if not self.timestamp:
            return -1
        return int((time.time() - self.timestamp) / 86_400)

    @property
    def age_description(self) -> str:
        d = self.days_ago
        if d < 0:
            return "unknown age"
        if d == 0:
            return "introduced today"
        if d == 1:
            return "introduced yesterday"
        if d < 7:
            return f"introduced {d} days ago"
        if d < 30:
            return f"introduced {d // 7} week(s) ago"
        if d < 365:
            return f"introduced {d // 30} month(s) ago"
        return f"introduced {d // 365} year(s) ago"

    @classmethod
    def from_porcelain(cls, output: str) -> Optional["BlameRecord"]:
        """
        Parse git blame --porcelain output into a BlameRecord.

        Porcelain format puts metadata on separate lines with prefixes:
            <hash> <orig_line> <final_line> <num_lines>
            author John Smith
            author-mail <john@example.com>
            author-time 1698765432
            summary Fix authentication bypass
            ...
        """
        if not output.strip():
            return None

        lines = output.splitlines()

        def extract(prefix: str) -> str:
            for line in lines:
                if line.startswith(prefix + " "):
                    return line[len(prefix) + 1:].strip()
            return ""

        try:
            commit_hash  = lines[0].split()[0] if lines else ""
            author       = extract("author")
            author_mail  = extract("author-mail").strip("<>")
            timestamp_s  = extract("author-time")
            summary      = extract("summary")
            timestamp    = int(timestamp_s) if timestamp_s.isdigit() else 0

            if not commit_hash or len(commit_hash) < 7:
                return None

            return cls(commit_hash, author, author_mail, timestamp, summary)

        except (IndexError, ValueError) as exc:
            logger.debug("Failed to parse blame output: %s", exc)
            return None


# ------------------------------------------------------------------
# File snapshot
# ------------------------------------------------------------------

class FileSnapshot:
    """
    A snapshot of a file's content at a specific git commit.

    Used to show what the code looked like before a bug
    was introduced — giving developers a "before" reference.
    """

    def __init__(self, commit_hash: str, content: Optional[str]):
        self.commit_hash = commit_hash
        self.content     = content
        self.available   = content is not None

    def excerpt(self, around_line: int, context: int = 10) -> str:
        """
        Return a short excerpt of the file centred on `around_line`.
        Useful for showing just the relevant section.
        """
        if not self.available:
            return ""

        lines = self.content.splitlines()
        start = max(0, around_line - context - 1)
        end   = min(len(lines), around_line + context)
        return "\n".join(lines[start:end])


# ------------------------------------------------------------------
# Git command runner
# ------------------------------------------------------------------

class GitRunner:
    """
    Thin wrapper around subprocess for git commands.

    Kept separate from business logic so commands can be
    mocked in tests without patching subprocess directly.
    """

    def __init__(self, repo_path: Path):
        self._repo = repo_path

    def run(self, *args: str) -> Optional[str]:
        """
        Run a git command and return stdout.
        Returns None on any failure — never raises.
        """
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=self._repo,
                timeout=GIT_TIMEOUT,
            )

            if result.returncode != 0:
                logger.debug(
                    "git %s returned %d: %s",
                    " ".join(args[:3]), result.returncode,
                    result.stderr.strip()[:100],
                )
                return None

            return result.stdout

        except subprocess.TimeoutExpired:
            logger.debug("git %s timed out", " ".join(args[:3]))
            return None

        except FileNotFoundError:
            logger.warning("git not found — is it installed?")
            return None

        except Exception as exc:
            logger.debug("git %s failed: %s", " ".join(args[:3]), exc)
            return None

    def is_git_repo(self) -> bool:
        """Return True if repo_path is inside a git repository."""
        result = self.run("rev-parse", "--is-inside-work-tree")
        return result is not None and result.strip() == "true"


# ------------------------------------------------------------------
# Main archaeologist
# ------------------------------------------------------------------

class GitArchaeologist:
    """
    Connects static analysis findings to git history.

    For each detected issue, answers:
        - Which commit introduced this line?
        - Who wrote it and when?
        - What did this code look like before that commit?
        - How long has this bug been in production?

    Usage
    -----
    arch   = GitArchaeologist(repo_path)
    result = arch.investigate(issue)

    # result["insight"] gives a one-line summary
    # result["code_before"] shows the safe version
    """

    def __init__(self, repo_path: Path):
        self._repo   = repo_path
        self._git    = GitRunner(repo_path)
        self._usable = self._git.is_git_repo()

        if not self._usable:
            logger.warning(
                "GitArchaeologist: %s is not a git repository — "
                "archaeology disabled",
                repo_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def investigate(self, issue: Dict) -> Dict:
        """
        Run full investigation on a single detected issue.

        Parameters
        ----------
        issue : A detected issue dict from BugDetector.
                Must contain 'file' and 'line_number' fields.

        Returns
        -------
        Dict with archaeology results. Always returns a dict —
        on failure, 'investigated' is False and 'reason' explains why.
        """

        if not self._usable:
            return self._unavailable_result("Repository is not a git repo")

        file_path = issue.get("file", "")
        line_no   = issue.get("line_number", 0)

        if not file_path or not line_no:
            return self._unavailable_result("Issue missing file or line_number")

        # Step 1: who last touched this line?
        blame = self._blame_line(file_path, line_no)

        if blame is None:
            return self._unavailable_result(
                f"git blame returned no result for {file_path}:{line_no}"
            )

        logger.debug(
            "Blame for %s:%d → %s by %s (%s)",
            file_path, line_no,
            blame.short_hash, blame.author, blame.age_description,
        )

        # Step 2: what did the file look like before this commit?
        snapshot = self._snapshot_before(file_path, blame.commit_hash)

        # Step 3: build structured result
        return self._build_result(issue, blame, snapshot)

    def investigate_batch(self, issues: List[Dict]) -> List[Dict]:
        """
        Run investigation on a list of issues.
        Returns results in the same order as input.
        Skips issues that cannot be investigated.
        """
        results = []

        for issue in issues:
            result = self.investigate(issue)
            results.append({**issue, "archaeology": result})

        found = sum(1 for r in results if r["archaeology"].get("investigated"))
        logger.info(
            "Archaeology complete — %d/%d issues investigated",
            found, len(issues),
        )

        return results

    # ------------------------------------------------------------------
    # Git operations
    # ------------------------------------------------------------------

    def _blame_line(self, file_path: str, line_no: int) -> Optional[BlameRecord]:
        """Run git blame on a specific line and parse the result."""
        output = self._git.run(
            "blame",
            f"-L {line_no},{line_no}",
            "--porcelain",
            file_path,
        )

        if output is None:
            return None

        return BlameRecord.from_porcelain(output)

    def _snapshot_before(self, file_path: str, commit_hash: str) -> FileSnapshot:
        """
        Get the file contents from the commit just before `commit_hash`.

        Uses `<hash>^` — git syntax for "the parent of this commit."
        If the commit has no parent (it's the first commit), this fails
        gracefully and returns an empty snapshot.
        """
        parent = f"{commit_hash}^"
        output = self._git.run("show", f"{parent}:{file_path}")
        return FileSnapshot(commit_hash, output)

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_result(
        self,
        issue:    Dict,
        blame:    BlameRecord,
        snapshot: FileSnapshot,
    ) -> Dict:
        """
        Assemble the final archaeology result dict.
        Keeps result construction in one place so the schema is consistent.
        """
        line_no = issue.get("line_number", 0)

        code_before = (
            snapshot.excerpt(around_line=line_no)
            if snapshot.available else None
        )

        insight = (
            f"Bug {blame.age_description} by {blame.author} "
            f"in commit {blame.short_hash}: '{blame.summary[:60]}'"
        )

        return {
            "investigated":   True,
            "author":         blame.author,
            "author_email":   blame.author_mail,
            "commit_hash":    blame.short_hash,
            "commit_summary": blame.summary,
            "days_old":       blame.days_ago,
            "age_description": blame.age_description,
            "code_before":    code_before,
            "had_safe_version": snapshot.available,
            "insight":        insight,
        }

    @staticmethod
    def _unavailable_result(reason: str) -> Dict:
        return {
            "investigated":    False,
            "reason":          reason,
            "author":          None,
            "author_email":    None,
            "commit_hash":     None,
            "commit_summary":  None,
            "days_old":        None,
            "age_description": None,
            "code_before":     None,
            "had_safe_version": False,
            "insight":         None,
        }