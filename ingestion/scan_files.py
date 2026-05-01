# ingestion/scan_files.py

"""
Module: scan_files

Responsibility
--------------
Walk a repository directory and return all Python source files
that are worth analyzing (i.e. not tests, not vendored code).

Design notes
------------
- Pure function: no side effects, same input always gives same output.
- Filtering rules come from config, not hardcoded here.
- Uses pathlib throughout — no raw string path manipulation.
"""

import logging
from pathlib import Path
from typing import List

from config import DEFAULT_CONFIG, PipelineConfig
from exceptions import FileScanError

logger = logging.getLogger(__name__)


def _is_excluded(path: Path, config: PipelineConfig) -> bool:
    """
    Return True if any part of the path matches an exclusion rule.
    Checks both directory names and file name prefixes.
    """

    for part in path.parts:
        if part in config.excluded_dirs:
            return True

    for prefix in config.excluded_prefixes:
        if path.name.startswith(prefix):
            return True

    return False


def scan_python_files(
    repo_path: Path,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> List[Path]:
    """
    Recursively find all Python source files in a repository,
    excluding test files and non-production directories.

    Parameters
    ----------
    repo_path : Path
        Root directory of the cloned repository.
    config    : PipelineConfig
        Pipeline-wide settings.

    Returns
    -------
    List[Path]
        Sorted list of absolute paths to `.py` files.

    Raises
    ------
    FileScanError
        If repo_path does not exist or is not a directory.
    """

    if not repo_path.exists():
        raise FileScanError(f"Repository path does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise FileScanError(f"Expected a directory, got: {repo_path}")

    all_files = [
        f for f in repo_path.rglob("*.py")
        if not _is_excluded(f.relative_to(repo_path), config)
    ]

    all_files.sort()

    logger.info(
        "Scan complete — %d Python files found (excluded: tests, migrations, cache)",
        len(all_files),
    )

    return all_files
