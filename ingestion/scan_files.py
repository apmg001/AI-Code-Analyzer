# ingestion/scan_files.py

import os


def scan_python_files(repo_path):
    """
    Walk through a repository and collect all Python source files.

    Parameters
    ----------
    repo_path : str
        Path to the locally cloned repository.

    Returns
    -------
    list
        List of absolute paths to Python files.
    """

    python_files = []

    for root, dirs, files in os.walk(repo_path):
        for filename in files:
            if filename.endswith(".py"):
                full_path = os.path.join(root, filename)
                python_files.append(full_path)

    print(f"[INFO] Found {len(python_files)} Python files")

    return python_files


if __name__ == "__main__":

    # Example test run
    test_repo_path = "repos/flask"

    files = scan_python_files(test_repo_path)

    print("\nSample files:")

    for file in files[:10]:
        print(file)