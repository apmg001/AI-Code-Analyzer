# exceptions.py

"""
Custom exceptions for the AI Code Analyzer pipeline.

Typed exceptions make error handling explicit and debuggable.
Never catch bare Exception when you can catch a specific one.
"""


class AnalyzerBaseError(Exception):
    """Root exception for all pipeline errors."""


class RepositoryCloneError(AnalyzerBaseError):
    """Raised when git clone fails or the URL is invalid."""


class FileScanError(AnalyzerBaseError):
    """Raised when the file system cannot be scanned."""


class FunctionExtractionError(AnalyzerBaseError):
    """Raised when AST parsing fails at the file level."""


class EmbeddingError(AnalyzerBaseError):
    """Raised when the embedding model fails to encode a chunk."""


class LLMError(AnalyzerBaseError):
    """Raised when the Claude API call fails."""


class PatchGenerationError(AnalyzerBaseError):
    """Raised when no patch strategy exists for an issue type."""
