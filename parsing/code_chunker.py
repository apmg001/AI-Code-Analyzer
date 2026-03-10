# parsing/code_chunker.py

"""
Module: code_chunker

Purpose
-------
Convert extracted function objects into standardized code chunks.

This module prepares code segments so they can later be processed by
embedding models or AI analyzers.

Responsibilities
----------------
1. Accept function metadata objects
2. Normalize structure
3. Generate chunk identifiers
4. Return clean chunk objects ready for downstream processing
"""

from typing import List, Dict


def _generate_chunk_id(index: int) -> str:
    """
    Generate a simple chunk identifier.

    Example:
        chunk_0001
        chunk_0002
    """

    return f"chunk_{index:04d}"


def _estimate_code_size(code: str) -> int:
    """
    Estimate approximate token size.

    This is a lightweight approximation using word count.
    Later this can be replaced with tokenizer-based counting.
    """

    return len(code.split())


def create_code_chunk(function_object: Dict, index: int) -> Dict:
    """
    Convert a function object into a standardized chunk.

    Parameters
    ----------
    function_object : dict
        Output from extract_function_code module
    index : int
        Position index used for generating chunk ID

    Returns
    -------
    dict
        Structured code chunk
    """

    code = function_object["code"]

    chunk = {
        "chunk_id": _generate_chunk_id(index),
        "function_name": function_object["function_name"],
        "file_path": function_object["file_path"],
        "line_start": function_object["line_start"],
        "line_end": function_object["line_end"],
        "arguments": function_object["arguments"],
        "code": code,
        "estimated_tokens": _estimate_code_size(code)
    }

    return chunk


def chunk_functions(function_list: List[Dict]) -> List[Dict]:
    """
    Convert multiple function objects into code chunks.

    Parameters
    ----------
    function_list : list[dict]

    Returns
    -------
    list[dict]
        List of standardized code chunks
    """

    chunks = []

    for idx, fn in enumerate(function_list, start=1):

        chunk = create_code_chunk(fn, idx)

        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    # Local test block
    # Runs only when module executed directly

    from ingestion.scan_files import scan_python_files
    from parsing.extract_function_code import extract_functions_from_files

    repo_path = "ingestion/repos/flask"

    print("[INFO] Scanning repository...")

    python_files = scan_python_files(repo_path)

    print(f"[INFO] Found {len(python_files)} Python files")

    print("\n[INFO] Extracting functions...")

    functions = extract_functions_from_files(python_files[:5])

    print(f"[INFO] Extracted {len(functions)} functions")

    print("\n[INFO] Creating code chunks...\n")

    chunks = chunk_functions(functions)

    for chunk in chunks[:5]:

        print("Chunk ID:", chunk["chunk_id"])
        print("Function:", chunk["function_name"])
        print("File:", chunk["file_path"])
        print("Estimated Tokens:", chunk["estimated_tokens"])
        print("Code Preview:\n", chunk["code"])
        print("-" * 60)