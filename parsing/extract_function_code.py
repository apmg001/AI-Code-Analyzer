# parsing/extract_function_code.py

"""
Module: extract_function_code

Purpose
-------
Extract complete function definitions from Python source files.

This module builds on the earlier AST parser and now retrieves
the actual source code of each function so that it can later be
used for embeddings, bug detection, and patch generation.

Responsibilities
----------------
1. Parse Python files using AST
2. Identify function definitions
3. Extract full function source code
4. Return structured metadata for each function
"""

import ast
from typing import List, Dict


def _read_source_lines(file_path: str) -> List[str]:
    """
    Read file and return list of lines.

    Keeping this as a small helper makes the main logic cleaner.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def _extract_code_segment(lines: List[str], start: int, end: int) -> str:
    """
    Extract code block between two line numbers.

    AST line numbers are 1-indexed.
    """

    return "".join(lines[start - 1:end])


def extract_function_code(file_path: str) -> List[Dict]:
    """
    Extract full function definitions from a Python file.

    Parameters
    ----------
    file_path : str
        Path to the Python file.

    Returns
    -------
    List[Dict]
        List containing metadata and source code of functions.
    """

    functions = []

    try:
        source_lines = _read_source_lines(file_path)
        source_code = "".join(source_lines)

        tree = ast.parse(source_code)

    except Exception as e:
        print(f"[ERROR] Failed to parse file: {file_path}")
        print(e)
        return functions

    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):

            start_line = node.lineno
            end_line = getattr(node, "end_lineno", node.lineno)

            function_code = _extract_code_segment(
                source_lines,
                start_line,
                end_line
            )

            function_data = {
                "function_name": node.name,
                "file_path": file_path,
                "line_start": start_line,
                "line_end": end_line,
                "arguments": [arg.arg for arg in node.args.args],
                "code": function_code
            }

            functions.append(function_data)

    return functions


def extract_functions_from_files(file_list: List[str]) -> List[Dict]:
    """
    Extract functions from multiple files.

    Parameters
    ----------
    file_list : list[str]

    Returns
    -------
    List[Dict]
        Flattened list of extracted functions
    """

    collected_functions = []

    for file_path in file_list:

        extracted = extract_function_code(file_path)

        if extracted:
            collected_functions.extend(extracted)

    return collected_functions


if __name__ == "__main__":

    # Local testing block
    # This should not run when module is imported

    from ingestion.scan_files import scan_python_files

    repo_path = "ingestion/repos/flask"

    print("[INFO] Scanning repository for Python files...")

    python_files = scan_python_files(repo_path)

    print(f"[INFO] Found {len(python_files)} Python files")

    print("\n[INFO] Extracting function source code from sample files...\n")

    for file in python_files[:3]:

        functions = extract_function_code(file)

        print(f"File: {file}")

        for fn in functions:

            print(
                f"\nFunction: {fn['function_name']}"
                f" (lines {fn['line_start']}-{fn['line_end']})"
            )

            print("Arguments:", fn["arguments"])
            print("Code:")
            print(fn["code"])

        print("\n" + "-" * 60)