# ingestion/clone_repo.py

"""
Module: clone_repo

Responsibility
--------------
Clone a remote git repository to a local directory.
Returns the local path for downstream modules to consume.

Design notes
------------
- Idempotent: if the repo already exists locally, skip cloning.
- Raises RepositoryCloneError on any failure so the pipeline
  can fail fast with a clear message instead of a cryptic git stderr.
"""

import logging
import subprocess
from pathlib import Path

from config import DEFAULT_CONFIG, PipelineConfig
from exceptions import RepositoryCloneError

logger = logging.getLogger(__name__)


def _derive_repo_name(repo_url: str) -> str:
    """
    Extract the repository name from a GitHub URL.

    Example
    -------
    'https://github.com/pallets/flask' -> 'flask'
    """
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git")


def clone_repository(
    repo_url: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Path:
    """
    Clone a git repository and return the local directory path.

    If the target directory already exists, cloning is skipped
    and the existing path is returned immediately.

    Parameters
    ----------
    repo_url : str
        Full HTTPS URL of the repository.
    config   : PipelineConfig
        Pipeline-wide settings.

    Returns
    -------
    Path
        Absolute path to the cloned repository on disk.

    Raises
    ------
    RepositoryCloneError
        If git is not installed or the clone command fails.
    """

    repo_name  = _derive_repo_name(repo_url)
    target_dir = config.repos_dir / repo_name

    if target_dir.exists():
        logger.info("Repository already exists locally — skipping clone: %s", target_dir)
        return target_dir

    config.repos_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cloning %s into %s", repo_url, target_dir)

    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RepositoryCloneError(
            f"git clone failed for '{repo_url}'.\n"
            f"stderr: {result.stderr.strip()}"
        )

    logger.info("Clone complete: %s", target_dir)
    return target_dir
