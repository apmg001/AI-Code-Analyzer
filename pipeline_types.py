# types.py

"""
Shared TypedDicts for the pipeline.

Defining shapes here means every module agrees on
the exact structure of chunks, issues, and patches.
No more guessing which keys exist.
"""

from typing import List, Optional
from typing_extensions import TypedDict


class CodeChunk(TypedDict):
    chunk_id:       str
    function_name:  str
    file_path:      str
    code:           str
    start_line:     int
    end_line:       int
    embedding:      Optional[List[float]]


class DetectedIssue(TypedDict):
    type:           str
    severity:       str          # "high" | "medium" | "low"
    function:       str
    file:           str
    line_number:    int
    code_snippet:   str
    chunk_id:       str
    message:        str
    similarity_score: Optional[float]   # only for semantic issues


class PatchResult(TypedDict):
    function:       str
    file:           str
    chunk_id:       str
    issue_type:     str
    severity:       str
    patch_source:   str          # "llm" | "rule_based"
    original_code:  str
    suggested_fix:  str
