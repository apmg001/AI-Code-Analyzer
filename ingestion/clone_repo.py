# ingestion/clone_repo.py

import os
from git import Repo


def clone_repository(repo_url, base_dir="repos"):
    """
    Clone a GitHub repository if it does not already exist locally.

    Parameters
    ----------
    repo_url : str
        URL of the GitHub repository.
    base_dir : str
        Directory where repositories will be stored.

    Returns
    -------
    str
        Path to the local repository directory.
    """

    # Make sure the directory for storing repos exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Extract repo name from URL
    repo_name = repo_url.rstrip("/").split("/")[-1]

    repo_path = os.path.join(base_dir, repo_name)

    # Avoid cloning again if it already exists
    if os.path.exists(repo_path):
        print(f"[INFO] Repository already present: {repo_path}")
        return repo_path

    print(f"[INFO] Cloning repository: {repo_url}")

    try:
        Repo.clone_from(repo_url, repo_path)
    except Exception as e:
        print(f"[ERROR] Failed to clone repository: {e}")
        return None

    print(f"[INFO] Repository cloned to: {repo_path}")

    return repo_path


if __name__ == "__main__":

    # Example test run
    test_repo = "https://github.com/pallets/flask"

    local_path = clone_repository(test_repo)

    if local_path:
        print(f"[SUCCESS] Repo ready at {local_path}")