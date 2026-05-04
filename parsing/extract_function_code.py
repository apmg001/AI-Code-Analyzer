# parsing/extract_function_code.py

"""
Module: extract_function_code

Responsibility
--------------
Parse Python source files using the AST and extract
individual function definitions with their source code.

Design notes
------------
- Uses ast.get_source_segment where available (Python 3.8+),
  falls back to line-slice extraction for compatibility.
- Skips files that cannot be decoded or parsed — logs a warning
  and continues rather than crashing the whole pipeline.
- Returns plain dicts (not objects) to keep data portable
  across module boundaries.
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from exceptions import FunctionExtractionError

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _read_source(file_path: Path) -> Optional[str]:
    """
    Read a file's source code.
    Returns None if the file cannot be decoded (e.g. binary embedded).
    """
    try:
        return file_path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError) as exc:
        logger.warning("Skipping unreadable file %s: %s", file_path, exc)
        return None


def _parse_ast(source: str, file_path: Path) -> Optional[ast.Module]:
    """
    Parse source into an AST.
    Returns None if there is a syntax error so the pipeline can continue.
    """
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        logger.warning("Syntax error in %s (line %s) — skipping", file_path, exc.lineno)
        return None


def _extract_function_source(source: str, node: ast.FunctionDef) -> str:
    """
    Extract raw source lines for a function node.

    ast.get_source_segment is the cleanest approach but requires
    the node to have end_lineno (Python 3.8+). We slice by line
    number as a universal fallback.
    """
    lines = source.splitlines()
    start = node.lineno - 1                     # ast lines are 1-indexed
    end   = getattr(node, "end_lineno", None)

    if end is not None:
        return "\n".join(lines[start:end])

    # Fallback: collect lines until the next top-level statement
    chunk_lines = []
    indent = len(lines[start]) - len(lines[start].lstrip())

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            chunk_lines.append(line)
            continue
        current_indent = len(line) - len(line.lstrip())
        if chunk_lines and current_indent <= indent and stripped:
            break
        chunk_lines.append(line)

    return "\n".join(chunk_lines)


def _build_function_record(
    node:      ast.FunctionDef,
    source:    str,
    file_path: Path,
) -> Dict[str, Any]:
    """Build a structured record for one extracted function."""

    return {
        "function_name": node.name,
        "file_path":     str(file_path),
        "start_line":    node.lineno,
        "end_line":      getattr(node, "end_lineno", node.lineno),
        "code":          _extract_function_source(source, node),
        "is_method":     isinstance(node, ast.AsyncFunctionDef) or False,
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_functions_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract all function definitions from a single Python file.

    Both regular (`def`) and async (`async def`) functions are captured.
    Nested functions are included — the bug detector will filter as needed.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    List of function record dicts.
    """

    source = _read_source(file_path)
    if source is None:
        return []

    tree = _parse_ast(source, file_path)
    if tree is None:
        return []

    records = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            record = _build_function_record(node, source, file_path)
            records.append(record)

    return records


def extract_functions_from_files(file_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Extract functions from a list of files.

    Aggregates results across files; bad files are skipped gracefully.

    Parameters
    ----------
    file_paths : List[Path]

    Returns
    -------
    List of all extracted function records.
    """

    all_functions: List[Dict[str, Any]] = []

    for file_path in file_paths:
        functions = extract_functions_from_file(file_path)
        all_functions.extend(functions)

    logger.info("Extracted %d functions from %d files", len(all_functions), len(file_paths))

    return all_functions
