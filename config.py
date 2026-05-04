# config.py

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:

    # --- Paths ---
    repos_dir:              Path  = Path("repos")
    results_dir:            Path  = Path("analysis_results")

    # --- Scanning ---
    excluded_dirs:          tuple = ("tests", "test", "migrations", "__pycache__", ".git", "venv", ".venv")
    excluded_prefixes:      tuple = ("test_", "conftest")

    # --- Chunking ---
    max_chunk_lines:        int   = 80
    min_chunk_lines:        int   = 3

    # --- Embeddings ---
    embedding_model:        str   = "all-MiniLM-L6-v2"

    # --- Similarity detection ---
    similarity_threshold:   float = 0.75

    # --- LLM Provider ---
    # Options: "llamacpp" | "anthropic" | "ollama" | "groq"
    llm_provider:           str   = "llamacpp"

    # --- llama.cpp settings ---
    llamacpp_url:           str   = "http://127.0.0.1:8080/v1/chat/completions"
    llamacpp_model:         str   = "qwen2.5-coder-14b"
    llamacpp_temperature:   float = 0.1
    llamacpp_max_tokens:    int   = 300    # ← changed from 600
    llamacpp_timeout:       int   = 300    # ← changed from 120

    # --- Anthropic settings (fallback) ---
    llm_model:              str   = "claude-sonnet-4-6"
    llm_max_tokens:         int   = 600
    llm_timeout:            int   = 20
    anthropic_api_url:      str   = "https://api.anthropic.com/v1/messages"
    anthropic_version:      str   = "2023-06-01"

    # --- Groq settings ---
    groq_api_url:           str   = "https://api.groq.com/openai/v1/chat/completions"
    groq_model:             str   = "llama-3.3-70b-versatile"

    # --- Ollama settings ---
    ollama_url:             str   = "http://localhost:11434/api/generate"
    ollama_model:           str   = "deepseek-coder"

    @property
    def api_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def groq_api_key(self) -> str:
        return os.environ.get("GROQ_API_KEY", "")

    @property
    def llm_available(self) -> bool:
        if self.llm_provider == "llamacpp":
            return True
        if self.llm_provider == "anthropic":
            return bool(self.api_key)
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        if self.llm_provider == "ollama":
            return True
        return False


DEFAULT_CONFIG = PipelineConfig()