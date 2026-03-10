# parsing/extract_functions.py

"""
Module: extract_functions

Purpose
-------
Parse Python source files and extract function definitions
using Python's Abstract Syntax Tree (AST).

Responsibilities
---------------
1. Read Python source files
2. Parse the file into an AST
3. Identify function definitions
4. Return structured metadata for each function

Output Example
--------------
[
    {
        "function_name": "create_app",
        "line_number": 10,
        "arguments": ["config"]
    }
]
"""

import ast
from typing import List, Dict


def extract_functions(file_path: str) -> List[Dict]:
    """
    Extract all function definitions from a Python source file.

    Parameters
    ----------
    file_path : str
        Absolute path to the Python source file.

    Returns
    -------
    List[Dict]
        A list containing metadata for each function discovered.
    """

    functions = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            source_code = file.read()

        tree = ast.parse(source_code)

    except Exception as error:
        print(f"[ERROR] Failed to parse file: {file_path}")
        print(error)
        return functions

    # Walk through AST nodes
    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):

            function_data = {
                "function_name": node.name,
                "line_number": node.lineno,
                "arguments": [arg.arg for arg in node.args.args]
            }

            functions.append(function_data)

    return functions


def extract_functions_from_files(file_list: List[str]) -> Dict[str, List[Dict]]:
    """
    Extract functions from multiple Python files.

    Parameters
    ----------
    file_list : List[str]
        List of Python file paths.

    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary mapping file path → extracted functions
    """

    results = {}

    for file_path in file_list:

        functions = extract_functions(file_path)

        if functions:
            results[file_path] = functions

    return results


if __name__ == "__main__":

    # Local testing block
    # This runs only if the module is executed directly

    from ingestion.scan_files import scan_python_files

    repo_path = "repos/flask"

    print("[INFO] Scanning repository for Python files...")

    python_files = scan_python_files(repo_path)

    print("[INFO] Extracting functions from sample files...\n")

    for file in python_files[:5]:

        extracted = extract_functions(file)

        print(f"File: {file}")

        for func in extracted:
            print(
                f"  Function: {func['function_name']} "
                f"(line {func['line_number']}) "
                f"args={func['arguments']}"
            )

        print()