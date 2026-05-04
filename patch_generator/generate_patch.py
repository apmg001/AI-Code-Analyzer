# patch_generator/generate_patch.py

"""
Module: generate_patch

Strategy (priority order)
--------------------------
1. LLM patch   — calls whichever provider is set in config.llm_provider
2. Rule patch  — curated template for the issue type
3. No patch    — structured fallback if neither applies

Supported providers: llamacpp | anthropic | ollama | groq
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from config import DEFAULT_CONFIG, PipelineConfig
from exceptions import LLMError
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Prompt builder (shared across all providers)
# ------------------------------------------------------------------

def _build_prompt(issue: Dict[str, Any], chunk: Dict[str, Any]) -> str:
    return (
        f"You are a senior Python engineer doing a code review.\n\n"
        f"A static analyzer detected the following issue:\n"
        f"  Type    : {issue['type']}\n"
        f"  Severity: {issue['severity']}\n"
        f"  Message : {issue['message']}\n\n"
        f"Affected code:\n"
        f"```python\n{chunk['code']}\n```\n\n"
        f"Instructions:\n"
        f"- Provide the corrected version of the code above.\n"
        f"- Add a single comment line above the fix explaining what changed.\n"
        f"- Do not restate the problem.\n"
        f"- Output only the fixed code. No markdown, no explanation.\n"
    )


# ------------------------------------------------------------------
# Provider: llama.cpp (local, free, no API key)
# ------------------------------------------------------------------

def _call_llamacpp(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
) -> Optional[str]:
    """
    Call a locally running llama.cpp server.

    llama.cpp exposes an OpenAI-compatible /v1/chat/completions endpoint.

    Start the server first:
        ./build/bin/llama-server --model models/qwen2.5-coder-14b-q4.gguf --port 8080
    """

    payload = json.dumps({
        "model": config.llamacpp_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior Python engineer. "
                    "When given buggy code and an issue description, "
                    "return ONLY the corrected Python code with a single "
                    "comment explaining what was changed. "
                    "No explanations. No markdown. Just fixed code."
                )
            },
            {
                "role": "user",
                "content": _build_prompt(issue, chunk)
            }
        ],
        "temperature": config.llamacpp_temperature,
        "max_tokens":  config.llamacpp_max_tokens,
        "stream":      False,
    }).encode("utf-8")

    request = urllib.request.Request(
        url=config.llamacpp_url,
        data=payload,
        headers={"content-type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=config.llamacpp_timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()

    except urllib.error.URLError:
        logger.warning(
            "llama.cpp server not reachable at %s — "
            "is the server running? "
            "Start it with: ./build/bin/llama-server --model models/qwen2.5-coder-14b-q4.gguf --port 8080",
            config.llamacpp_url,
        )
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Unexpected llama.cpp response: %s", exc)

    return None


# ------------------------------------------------------------------
# Provider: Anthropic Claude
# ------------------------------------------------------------------

def _call_anthropic(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
) -> Optional[str]:

    if not config.api_key:
        logger.debug("ANTHROPIC_API_KEY not set — skipping")
        return None

    payload = json.dumps({
        "model":      config.llm_model,
        "max_tokens": config.llm_max_tokens,
        "messages":   [{"role": "user", "content": _build_prompt(issue, chunk)}],
    }).encode("utf-8")

    request = urllib.request.Request(
        url=config.anthropic_api_url,
        data=payload,
        headers={
            "x-api-key":         config.api_key,
            "anthropic-version": config.anthropic_version,
            "content-type":      "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=config.llm_timeout) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]

    except urllib.error.HTTPError as exc:
        logger.warning("Anthropic API HTTP error %d", exc.code)
    except urllib.error.URLError as exc:
        logger.warning("Anthropic API network error: %s", exc.reason)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Unexpected Anthropic response: %s", exc)

    return None


# ------------------------------------------------------------------
# Provider: Groq (free cloud API)
# ------------------------------------------------------------------

def _call_groq(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
) -> Optional[str]:

    if not config.groq_api_key:
        logger.debug("GROQ_API_KEY not set — skipping")
        return None

    payload = json.dumps({
        "model":       config.groq_model,
        "messages":    [{"role": "user", "content": _build_prompt(issue, chunk)}],
        "max_tokens":  config.llm_max_tokens,
        "temperature": 0.1,
    }).encode("utf-8")

    request = urllib.request.Request(
        url=config.groq_api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {config.groq_api_key}",
            "content-type":  "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=config.llm_timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()

    except urllib.error.HTTPError as exc:
        logger.warning("Groq API HTTP error %d", exc.code)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Unexpected Groq response: %s", exc)

    return None


# ------------------------------------------------------------------
# Provider: Ollama
# ------------------------------------------------------------------

def _call_ollama(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
) -> Optional[str]:

    payload = json.dumps({
        "model":  config.ollama_model,
        "prompt": _build_prompt(issue, chunk),
        "stream": False,
    }).encode("utf-8")

    request = urllib.request.Request(
        url=config.ollama_url,
        data=payload,
        headers={"content-type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=config.llamacpp_timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()

    except urllib.error.URLError:
        logger.warning("Ollama not reachable — is `ollama serve` running?")
    except (KeyError, json.JSONDecodeError) as exc:
        logger.warning("Unexpected Ollama response: %s", exc)

    return None


# ------------------------------------------------------------------
# Router — picks provider from config
# ------------------------------------------------------------------

def _call_llm(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
) -> Optional[str]:
    """
    Route to the correct LLM provider.
    Change config.llm_provider to switch — no other code needs to change.
    """

    routes: Dict[str, Callable] = {
        "llamacpp":  _call_llamacpp,
        "anthropic": _call_anthropic,
        "groq":      _call_groq,
        "ollama":    _call_ollama,
    }

    handler = routes.get(config.llm_provider)

    if handler is None:
        logger.warning("Unknown LLM provider: '%s'", config.llm_provider)
        return None

    return handler(issue, chunk, config)


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class PatchGenerator:
    """
    Generates fix suggestions for detected issues.
    Tries LLM first, falls back to rule-based templates.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self._config = config

        self._patch_rules: Dict[str, Callable] = {
            "division_by_zero":         self._patch_division_by_zero,
            "eval_usage":               self._patch_eval_usage,
            "exec_usage":               self._patch_exec_usage,
            "potential_infinite_loop":  self._patch_infinite_loop,
            "assert_in_production":     self._patch_assert_in_production,
            "bare_except_swallow":      self._patch_bare_except,
            "mutable_default_argument": self._patch_mutable_default_arg,
            "hardcoded_secret":         self._patch_hardcoded_secret,
            "semantic_similarity_flag": self._patch_semantic_flag,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_patch(
        self,
        issue: Dict[str, Any],
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:

        # Strategy 1: LLM
        llm_text = _call_llm(issue, chunk, self._config)
        if llm_text:
            logger.info(
                "LLM patch (%s) generated for '%s' in %s",
                self._config.llm_provider, issue["type"], chunk["file_path"],
            )
            return self._build_patch(issue, chunk, llm_text, source=self._config.llm_provider)

        # Strategy 2: Rule-based
        rule_handler = self._patch_rules.get(issue["type"])
        if rule_handler:
            logger.debug("Rule-based patch used for '%s'", issue["type"])
            return rule_handler(issue, chunk)

        # Strategy 3: No patch
        return {
            "function":      chunk["function_name"],
            "file":          chunk["file_path"],
            "chunk_id":      chunk["chunk_id"],
            "issue_type":    issue["type"],
            "severity":      issue["severity"],
            "patch_source":  "none",
            "original_code": chunk["code"],
            "suggested_fix": "No automated patch available. Manual review required.",
        }

    def generate_patches(
        self,
        issues: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate patches concurrently — 3 at a time instead of 1.

        llama.cpp server handles parallel requests fine.
        ThreadPoolExecutor is used (not ProcessPool) because
        the bottleneck is network I/O waiting for LLM response,
        not CPU — threads release the GIL during I/O waits.
        """

        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunk_map = {c["chunk_id"]: c for c in chunks}
        patches   = []
        failed    = 0

        # Build work list first — skip missing chunks early
        work = []
        for issue in issues:
            chunk = chunk_map.get(issue["chunk_id"])
            if chunk is None:
                logger.warning("Chunk not found for issue: %s", issue["chunk_id"])
                continue
            work.append((issue, chunk))

        # Run 3 patches in parallel
        # Keep at 3 — higher values can overwhelm llama.cpp server
        with ThreadPoolExecutor(max_workers=3) as executor:

            future_to_issue = {
                executor.submit(self.generate_patch, issue, chunk): issue
                for issue, chunk in work
            }

            for future in as_completed(future_to_issue):
                issue = future_to_issue[future]
                try:
                    patch = future.result()
                    patches.append(patch)
                    logger.info(
                        "Patch done: %s in %s",
                        issue["type"], issue["file"].split("/")[-1],
                    )
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "Patch failed for %s in %s: %s",
                        issue["type"], issue.get("file", "unknown"), exc,
                    )

        llm_count  = sum(1 for p in patches if p.get("patch_source") not in ("rule_based", "none"))
        rule_count = sum(1 for p in patches if p.get("patch_source") == "rule_based")

        logger.info(
            "Patches generated: %d LLM(%s), %d rule-based, %d total, %d failed",
            llm_count, self._config.llm_provider,
            rule_count, len(patches), failed,
        )

        return patches

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------

    def _build_patch(
        self,
        issue:      Dict[str, Any],
        chunk:      Dict[str, Any],
        suggestion: str,
        source:     str,
    ) -> Dict[str, Any]:
        return {
            "function":      chunk["function_name"],
            "file":          chunk["file_path"],
            "chunk_id":      chunk["chunk_id"],
            "issue_type":    issue["type"],
            "severity":      issue["severity"],
            "patch_source":  source,
            "original_code": chunk["code"],
            "suggested_fix": suggestion,
        }

    # ------------------------------------------------------------------
    # Rule-based fallbacks
    # ------------------------------------------------------------------

    def _patch_division_by_zero(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Guard the division with a non-zero check:\n\n"
            "  if denominator != 0:\n"
            "      result = numerator / denominator\n"
            "  else:\n"
            "      raise ValueError(f'denominator must be non-zero, got {denominator}')"
        ), "rule_based")

    def _patch_eval_usage(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Replace eval() with ast.literal_eval() for safe value parsing:\n\n"
            "  import ast\n"
            "  result = ast.literal_eval(expression)\n\n"
            "If you need expression evaluation, use the `simpleeval` library."
        ), "rule_based")

    def _patch_exec_usage(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Refactor dynamic execution into explicit callable functions:\n\n"
            "  # Instead of: exec(dynamic_code)\n"
            "  import importlib\n"
            "  module = importlib.import_module('my.module')\n"
            "  module.run()"
        ), "rule_based")

    def _patch_infinite_loop(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Add a clear termination condition:\n\n"
            "  MAX_ITERATIONS = 10_000\n"
            "  for _ in range(MAX_ITERATIONS):\n"
            "      if done_condition:\n"
            "          break\n"
            "  else:\n"
            "      raise RuntimeError('Loop exceeded maximum iterations')"
        ), "rule_based")

    def _patch_assert_in_production(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Replace assert with an explicit guard:\n\n"
            "  # Instead of: assert condition, 'message'\n"
            "  if not condition:\n"
            "      raise ValueError('message')\n\n"
            "assert is stripped when Python runs with -O flag."
        ), "rule_based")

    def _patch_bare_except(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Catch specific exceptions and log them:\n\n"
            "  import logging\n"
            "  logger = logging.getLogger(__name__)\n\n"
            "  try:\n"
            "      ...\n"
            "  except SomeSpecificError as exc:\n"
            "      logger.exception('Error in <context>: %s', exc)\n"
            "      raise"
        ), "rule_based")

    def _patch_mutable_default_arg(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Use None as default, initialise inside the function:\n\n"
            "  def func(items=None):\n"
            "      if items is None:\n"
            "          items = []\n"
            "      ..."
        ), "rule_based")

    def _patch_hardcoded_secret(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Load credentials from environment variables:\n\n"
            "  import os\n"
            "  password = os.environ['DB_PASSWORD']\n\n"
            "Add .env to .gitignore. Never commit secrets to source control."
        ), "rule_based")

    def _patch_semantic_flag(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            f"Flagged as similar to known anti-pattern: "
            f"'{issue.get('pattern_label', 'unknown')}' "
            f"(score: {issue.get('similarity_score', 'N/A')}). "
            f"Manual review recommended."
        ), "rule_based")