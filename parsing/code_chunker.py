# parsing/code_chunker.py

"""
Module: code_chunker

Responsibility
--------------
Convert extracted function records into self-contained chunks
ready for embedding and analysis.

Design notes
------------
- Functions under min_chunk_lines are too small to be meaningful
  and are dropped (getters, one-liners).
- Functions over max_chunk_lines are split at logical boundaries
  (blank lines) to stay within the embedding model's token budget.
- Each chunk gets a deterministic ID derived from file path +
  function name + start line so results are reproducible.
- chunk_id is a hash, not a sequential integer, so chunks remain
  stable even if unrelated files change.
"""

import hashlib
import logging
from typing import Any, Dict, List

from config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _make_chunk_id(file_path: str, function_name: str, start_line: int) -> str:
    """
    Produce a short, stable identifier for a chunk.

    Using a hash means IDs don't shift when other files are added
    or removed from the repo — unlike a sequential counter.
    """
    raw = f"{file_path}::{function_name}::{start_line}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _split_on_blank_lines(code: str, max_lines: int) -> List[str]:
    """
    Split a long block of code into sub-chunks at blank line boundaries.

    Splitting on blank lines keeps logical blocks (if-blocks, loops)
    together rather than cutting mid-statement.
    """
    lines      = code.splitlines()
    sub_chunks = []
    current    = []

    for line in lines:
        current.append(line)

        at_blank = not line.strip()
        over_limit = len(current) >= max_lines

        if at_blank and over_limit:
            sub_chunks.append("\n".join(current).strip())
            current = []

    if current:
        sub_chunks.append("\n".join(current).strip())

    return [c for c in sub_chunks if c]


def _build_chunk(
    function_record: Dict[str, Any],
    code:            str,
    start_line:      int,
    config:          PipelineConfig,
) -> Dict[str, Any]:
    """Assemble a single chunk dict from a function record and code slice."""

    return {
        "chunk_id":      _make_chunk_id(function_record["file_path"], function_record["function_name"], start_line),
        "function_name": function_record["function_name"],
        "file_path":     function_record["file_path"],
        "start_line":    start_line,
        "end_line":      start_line + len(code.splitlines()),
        "code":          code,
        "embedding":     None,   # populated later by embed_functions
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def chunk_functions(
    functions: List[Dict[str, Any]],
    config: PipelineConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """
    Convert a list of function records into analysis-ready chunks.

    Short functions (< min_chunk_lines) are dropped.
    Long functions (> max_chunk_lines) are split at blank lines.

    Parameters
    ----------
    functions : List of function records from extract_function_code.
    config    : PipelineConfig

    Returns
    -------
    List of chunk dicts, each with a stable chunk_id and embedding=None.
    """

    chunks:  List[Dict[str, Any]] = []
    dropped: int = 0

    for func in functions:
        code       = func["code"].strip()
        line_count = len(code.splitlines())

        if line_count < config.min_chunk_lines:
            dropped += 1
            continue

        if line_count <= config.max_chunk_lines:
            chunks.append(_build_chunk(func, code, func["start_line"], config))
        else:
            sub_codes = _split_on_blank_lines(code, config.max_chunk_lines)
            offset    = func["start_line"]

            for sub_code in sub_codes:
                chunks.append(_build_chunk(func, sub_code, offset, config))
                offset += len(sub_code.splitlines())

    logger.info(
        "Chunking complete — %d chunks produced, %d functions dropped (too short)",
        len(chunks), dropped,
    )

    return chunks
