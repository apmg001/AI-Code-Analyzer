# patch_generator/generate_patch.py

"""
Module: generate_patch

Responsibility
--------------
Generate fix suggestions for every detected issue.

Strategy — tried in this order for every issue:
    1. RAG + LLM    Retrieve similar code for style context,
                    build a constrained prompt, call LLM,
                    validate output is real Python code.
    2. LLM only     Same but without RAG context.
    3. Rule-based   Curated template — always available,
                    always correct.
    4. No patch     Structured fallback.

How hallucination is stopped
-----------------------------
Layer 1 — Issue type filtering
    taint_flow, statistical_anomaly, dna_violation and
    semantic_similarity_flag go directly to rule-based.
    The LLM only sees ONE function but these issues span
    MULTIPLE functions — it cannot reason about them and
    will invent fake fixes. Rule-based templates are more
    accurate for these types.

Layer 2 — Constrained prompt
    The prompt gives the LLM a strict output contract:
    - Must start with "# fix:"
    - Must return the complete function
    - No markdown, no explanation
    - Temperature 0.0 — deterministic, no creativity

Layer 3 — Output cleaning
    _clean_llm_output() strips markdown code fences.
    Qwen ignores "no markdown" instructions ~30% of the
    time. Stripping fences is more reliable than asking.

Layer 4 — Output validation
    _is_valid_python() checks the cleaned output is
    actual Python code — not advice, not explanations.
    Rejects responses starting with natural language.
    Falls back to rule-based if validation fails.

Design decisions
----------------
- Temperature is set to 0.0 for patch generation.
  Patch generation is a precision task not a creative one.
  Temperature 0 means the model always picks the most
  likely token — deterministic and less likely to invent.

- _TEMPLATE_ISSUES is a frozenset at class level.
  Checked before any LLM call — no wasted network I/O.

- _build_patch is the single schema for all outputs.
  LLM and rule-based patches are structurally identical
  so downstream consumers don't need to handle both.

- max_workers=2 in ThreadPoolExecutor.
  llama.cpp handles 4 parallel slots but degrades under
  heavy concurrent load. 2 workers gives enough
  parallelism without overwhelming the server.
"""

import ast
import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Layer 1 — issue types that skip LLM entirely
# ------------------------------------------------------------------

# These issue types go directly to rule-based patches.
# The LLM cannot fix them correctly because:
#
#   taint_flow          — spans multiple functions, LLM only sees one
#   statistical_anomaly — no clear "fix", needs human judgement
#   dna_violation       — convention fix, not a bug fix
#   semantic_similarity — pattern match, template is more accurate
#
_SKIP_LLM: frozenset = frozenset({
    "taint_flow",
    "statistical_anomaly",
    "dna_violation",
    "semantic_similarity_flag",
})


# ------------------------------------------------------------------
# Layer 2 — constrained prompt
# ------------------------------------------------------------------

def _build_prompt(issue: Dict[str, Any], chunk: Dict[str, Any]) -> str:
    """
    Build a tightly constrained prompt that minimises hallucination.

    Key constraints:
    - Strict output format enforced ("start with # fix:")
    - "Do not invent" instruction — stops fictional imports
    - "Only fix the specific issue" — stops scope creep
    - No markdown instruction repeated twice for emphasis
    """
    return (
        f"Fix this Python function. Follow the output format exactly.\n\n"
        f"ISSUE TYPE : {issue['type']}\n"
        f"SEVERITY   : {issue['severity']}\n"
        f"PROBLEM    : {issue['message']}\n"
        f"FUNCTION   : {chunk.get('function_name', '')}\n\n"
        f"CODE TO FIX:\n"
        f"{chunk['code']}\n\n"
        f"OUTPUT FORMAT — follow this exactly:\n"
        f"# fix: <one line describing the change>\n"
        f"<complete corrected function here>\n\n"
        f"RULES:\n"
        f"- Start your response with '# fix:' on line 1\n"
        f"- Return the complete corrected function\n"
        f"- Only fix the specific issue described above\n"
        f"- Do not add imports that are not needed\n"
        f"- Do not invent functionality that does not exist\n"
        f"- Do not use markdown code fences\n"
        f"- No explanation after the code\n"
    )


# ------------------------------------------------------------------
# Layer 3 — output cleaning
# ------------------------------------------------------------------

def _clean_llm_output(text: str) -> str:
    """
    Strip markdown code fences from LLM output.

    Qwen ignores "no markdown" instructions roughly 30% of
    the time regardless of how the prompt is phrased. Cleaning
    the output is more reliable than prompt engineering alone.

    Handles:
        ```python ... ```
        ``` ... ```
        ` ... `

    Preserves content inside the fences — only removes the fences.
    """
    if not text:
        return text

    text = text.strip()

    lines   = text.splitlines()
    cleaned = []
    inside  = False

    for line in lines:
        stripped = line.strip()

        # Opening fence — start collecting content, skip the fence line
        if stripped.startswith("```"):
            inside = True
            continue

        # If we were inside a fence and hit another ``` — close it
        if inside and stripped == "```":
            inside = False
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # If nothing was inside a fence, return original stripped
    return result if result else text


# ------------------------------------------------------------------
# Layer 4 — output validation
# ------------------------------------------------------------------

def _is_valid_python(text: str) -> bool:
    """
    Confirm the LLM returned actual Python code, not advice.

    Two checks:
    1. Does not start with natural language
       (catches "Replace eval with...", "You should use...")
    2. Parses as valid Python AST
       (catches truncated responses, structural errors)

    Returns True only if both checks pass.
    """
    if not text or len(text.strip()) < 10:
        return False

    # Check 1 — first line must look like code, not advice
    first_line = text.strip().splitlines()[0].lower().strip()

    natural_language_starters = (
        "replace ",
        "use ",
        "consider ",
        "you should",
        "you can ",
        "to fix",
        "the fix",
        "this function",
        "instead of",
        "avoid ",
        "here is",
        "here's",
        "i recommend",
        "we can",
        "one way",
        "a better",
        "the issue",
        "the problem",
    )

    if any(first_line.startswith(s) for s in natural_language_starters):
        logger.debug("LLM response rejected — starts with advice not code")
        return False

    # Check 2 — must parse as valid Python
    # Wrap in a class/module context because the LLM sometimes
    # returns just the function body without the def line
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        # Try wrapping in a dummy class in case it is a method
        try:
            ast.parse(f"class _Dummy:\n" + "\n".join(
                "    " + line for line in text.splitlines()
            ))
            return True
        except SyntaxError:
            logger.debug("LLM response rejected — failed ast.parse()")
            return False


# ------------------------------------------------------------------
# Synthetic chunk builder
# ------------------------------------------------------------------

def _synthetic_chunk(issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a minimal chunk for issues with no embedded chunk.

    Taint flows have no chunk_id that maps into the embedded
    chunk list because they span multiple functions. This
    synthetic chunk lets the patch generator produce a rule-based
    suggestion for them.
    """
    return {
        "chunk_id":      issue.get("chunk_id", "synthetic"),
        "function_name": issue.get("function", "unknown"),
        "file_path":     issue.get("file", "unknown"),
        "code":          issue.get("code_snippet", "# taint flow spans multiple functions"),
        "start_line":    issue.get("line_number", 0),
        "embedding":     None,
    }


# ------------------------------------------------------------------
# LLM provider functions
# ------------------------------------------------------------------

def _call_llamacpp(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """
    Call the local llama.cpp server.

    Temperature is forced to 0.0 regardless of config.
    Patch generation is a precision task — creativity
    causes hallucination. Deterministic output is better.
    """
    payload = json.dumps({
        "model": config.llamacpp_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Python code repair tool. "
                    "When given a buggy function, return ONLY the "
                    "corrected function starting with '# fix:' on "
                    "line 1. No markdown. No explanation. Just code."
                ),
            },
            {
                "role": "user",
                "content": prompt or _build_prompt(issue, chunk),
            },
        ],
        "temperature": 0.0,          # deterministic — prevents creativity
        "max_tokens":  config.llamacpp_max_tokens,
        "stream":      False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url=config.llamacpp_url,
        data=payload,
        headers={"content-type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=config.llamacpp_timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()

    except urllib.error.URLError:
        logger.warning(
            "llama.cpp not reachable at %s — is the server running?",
            config.llamacpp_url,
        )
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Unexpected llama.cpp response: %s", exc)

    return None


def _call_anthropic(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Call the Anthropic Claude API."""

    if not config.api_key:
        logger.debug("ANTHROPIC_API_KEY not set — skipping")
        return None

    payload = json.dumps({
        "model":      config.llm_model,
        "max_tokens": config.llm_max_tokens,
        "messages":   [{
            "role":    "user",
            "content": prompt or _build_prompt(issue, chunk),
        }],
    }).encode("utf-8")

    req = urllib.request.Request(
        url=config.anthropic_api_url,
        data=payload,
        headers={
            "x-api-key":         config.api_key,
            "anthropic-version": config.anthropic_version,
            "content-type":      "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=config.llm_timeout) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]

    except urllib.error.HTTPError as exc:
        logger.warning("Anthropic HTTP error %d", exc.code)
    except urllib.error.URLError as exc:
        logger.warning("Anthropic network error: %s", exc.reason)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Unexpected Anthropic response: %s", exc)

    return None


def _call_groq(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Call the Groq cloud API."""

    if not config.groq_api_key:
        logger.debug("GROQ_API_KEY not set — skipping")
        return None

    payload = json.dumps({
        "model":       config.groq_model,
        "messages":    [{
            "role":    "user",
            "content": prompt or _build_prompt(issue, chunk),
        }],
        "max_tokens":  config.llm_max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        url=config.groq_api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {config.groq_api_key}",
            "content-type":  "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=config.llm_timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()

    except urllib.error.HTTPError as exc:
        logger.warning("Groq HTTP error %d", exc.code)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Unexpected Groq response: %s", exc)

    return None


def _call_ollama(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Call a locally running Ollama server."""

    payload = json.dumps({
        "model":  config.ollama_model,
        "prompt": prompt or _build_prompt(issue, chunk),
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    req = urllib.request.Request(
        url=config.ollama_url,
        data=payload,
        headers={"content-type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=config.llamacpp_timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()

    except urllib.error.URLError:
        logger.warning("Ollama not reachable — is `ollama serve` running?")
    except (KeyError, json.JSONDecodeError) as exc:
        logger.warning("Unexpected Ollama response: %s", exc)

    return None


# ------------------------------------------------------------------
# LLM router
# ------------------------------------------------------------------

def _call_llm(
    issue:  Dict[str, Any],
    chunk:  Dict[str, Any],
    config: PipelineConfig,
    prompt: Optional[str] = None,
) -> Optional[str]:
    """Route to the correct LLM provider."""

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

    return handler(issue, chunk, config, prompt)


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class PatchGenerator:
    """
    Generates fix suggestions for all detected issues.

    Hallucination is stopped through 4 layers:
        1. Skip LLM for issue types it cannot reason about
        2. Constrained prompt with strict output format
        3. Clean markdown fences from output
        4. Validate output is real Python before accepting

    RAGEngine is optional — when provided, LLM prompts include
    semantically similar code from the same repo for style context.
    """

    def __init__(
        self,
        config:     PipelineConfig = DEFAULT_CONFIG,
        rag_engine: Any            = None,
    ):
        self._config = config
        self._rag    = rag_engine

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
            "taint_flow":               self._patch_taint_flow,
            "statistical_anomaly":      self._patch_statistical_anomaly,
            "dna_violation":            self._patch_dna_violation,
            "print_in_production":      self._patch_print_in_production,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_patch(
        self,
        issue: Dict[str, Any],
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a fix for one issue.

        Order: Rule-only check → RAG+LLM → LLM → Rule → No patch
        """

        # Layer 1 — skip LLM for types it cannot handle
        if issue.get("type") in _SKIP_LLM:
            rule_handler = self._patch_rules.get(issue["type"])
            if rule_handler:
                return rule_handler(issue, chunk)
            return self._build_no_patch(issue, chunk)

        # Build prompt — RAG-augmented if engine is available
        prompt  = None
        rag_hit = False

        if (self._rag
                and self._rag.is_available
                and chunk.get("embedding") is not None):
            try:
                similar = self._rag.retrieve_similar(chunk, top_k=3)
                if similar:
                    prompt  = self._rag.build_rag_prompt(issue, chunk, similar)
                    rag_hit = True
            except Exception as exc:
                logger.debug("RAG prompt failed: %s", exc)

        if prompt is None:
            prompt = _build_prompt(issue, chunk)

        # Layer 2+3+4 — call LLM, clean, validate
        raw = _call_llm(issue, chunk, self._config, prompt)

        if raw:
            cleaned = _clean_llm_output(raw)

            if _is_valid_python(cleaned):
                source = (
                    f"{self._config.llm_provider}+rag"
                    if rag_hit else
                    self._config.llm_provider
                )
                logger.info(
                    "LLM patch accepted (%s) for '%s' in %s",
                    source, issue["type"],
                    chunk.get("file_path", "").split("/")[-1],
                )
                return self._build_patch(issue, chunk, cleaned, source=source)

            else:
                logger.info(
                    "LLM response rejected for '%s' in %s — "
                    "not valid Python, using rule-based fallback",
                    issue["type"],
                    chunk.get("file_path", "").split("/")[-1],
                )

        # Fall through to rule-based
        rule_handler = self._patch_rules.get(issue["type"])
        if rule_handler:
            return rule_handler(issue, chunk)

        return self._build_no_patch(issue, chunk)

    def generate_patches(
        self,
        issues: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate patches for all issues concurrently."""

        chunk_map = {c["chunk_id"]: c for c in chunks}
        patches   = []
        failed    = 0

        work: List[tuple] = []
        for issue in issues:
            chunk = chunk_map.get(issue["chunk_id"])
            if chunk is None:
                if issue.get("type") == "taint_flow":
                    chunk = _synthetic_chunk(issue)
                else:
                    logger.warning(
                        "Chunk not found for %s — skipping",
                        issue["chunk_id"],
                    )
                    continue
            work.append((issue, chunk))

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_map = {
                executor.submit(self.generate_patch, issue, chunk): issue
                for issue, chunk in work
            }

            for future in as_completed(future_map):
                issue = future_map[future]
                try:
                    patch = future.result()
                    patches.append(patch)
                    logger.info(
                        "Patch done: %s in %s",
                        issue["type"],
                        issue.get("file", "").split("/")[-1],
                    )
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "Patch failed for %s: %s",
                        issue.get("type", "unknown"), exc,
                    )

        # Summary log
        rag_n  = sum(1 for p in patches if "+rag" in p.get("patch_source", ""))
        llm_n  = sum(1 for p in patches
                     if p.get("patch_source") not in ("rule_based", "none")
                     and "+rag" not in p.get("patch_source", ""))
        rule_n = sum(1 for p in patches if p.get("patch_source") == "rule_based")

        logger.info(
            "Patches: %d RAG+LLM  %d LLM  %d rule-based  "
            "%d total  %d failed",
            rag_n, llm_n, rule_n, len(patches), failed,
        )

        return patches

    # ------------------------------------------------------------------
    # Output builders
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

    def _build_no_patch(
        self,
        issue: Dict[str, Any],
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "function":      chunk.get("function_name", "unknown"),
            "file":          chunk.get("file_path", "unknown"),
            "chunk_id":      chunk.get("chunk_id", "unknown"),
            "issue_type":    issue["type"],
            "severity":      issue["severity"],
            "patch_source":  "none",
            "original_code": chunk.get("code", ""),
            "suggested_fix": (
                f"No automated patch for '{issue['type']}'. "
                f"Manual review required."
            ),
        }

    # ------------------------------------------------------------------
    # Rule-based patches
    # ------------------------------------------------------------------

    def _patch_division_by_zero(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Guard the division with a zero check:\n\n"
            "  if denominator == 0:\n"
            "      raise ValueError('denominator cannot be zero')\n"
            "  result = numerator / denominator"
        ), "rule_based")

    def _patch_eval_usage(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Replace eval() with ast.literal_eval():\n\n"
            "  import ast\n"
            "  result = ast.literal_eval(expression)\n\n"
            "ast.literal_eval() only evaluates literals — "
            "it cannot execute arbitrary code."
        ), "rule_based")

    def _patch_exec_usage(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Replace exec() with explicit importlib:\n\n"
            "  import importlib\n"
            "  module = importlib.import_module('your.module')\n"
            "  module.your_function()"
        ), "rule_based")

    def _patch_infinite_loop(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Add a termination condition:\n\n"
            "  MAX_ITERATIONS = 10_000\n"
            "  for _ in range(MAX_ITERATIONS):\n"
            "      if done:\n"
            "          break\n"
            "  else:\n"
            "      raise RuntimeError('Exceeded maximum iterations')"
        ), "rule_based")

    def _patch_assert_in_production(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Replace assert with explicit validation:\n\n"
            "  if not condition:\n"
            "      raise ValueError('descriptive message')\n\n"
            "assert is removed when Python runs with -O flag."
        ), "rule_based")

    def _patch_bare_except(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Catch a specific exception and log it:\n\n"
            "  import logging\n"
            "  logger = logging.getLogger(__name__)\n\n"
            "  try:\n"
            "      ...\n"
            "  except SpecificError as exc:\n"
            "      logger.exception('Context: %s', exc)\n"
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
            "Load from environment variables:\n\n"
            "  import os\n"
            "  secret = os.environ['SECRET_KEY']\n\n"
            "Add .env to .gitignore. "
            "If this secret was committed, rotate it immediately."
        ), "rule_based")

    def _patch_semantic_flag(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            f"Flagged as similar to anti-pattern "
            f"'{issue.get('pattern_label', 'unknown')}' "
            f"(score: {issue.get('similarity_score', 'N/A')}).\n\n"
            f"Review manually."
        ), "rule_based")

    def _patch_taint_flow(self, issue: Dict, chunk: Dict) -> Dict:
        """
        Taint flow patches are always rule-based.

        The LLM only sees one function but taint flows cross
        multiple function boundaries. It cannot reason about
        the full call chain and will invent wrong fixes.
        The template below explains the full picture correctly.
        """
        path = issue.get("taint_path", "source → sink")
        sink = issue.get("sink_function", "unknown")
        return self._build_patch(issue, chunk, (
            f"User-controlled input reaches a dangerous operation.\n\n"
            f"Taint path: {path}\n\n"
            f"Fix — sanitize input before it reaches '{sink}':\n\n"
            f"  # SQL injection — use parameterised queries:\n"
            f"  cursor.execute(\n"
            f"      'SELECT * FROM t WHERE id = %s',\n"
            f"      (user_input,),\n"
            f"  )\n\n"
            f"  # Shell injection — pass as list, never shell=True:\n"
            f"  subprocess.run(['command', user_input], shell=False)\n\n"
            f"  # eval/exec injection — whitelist valid values:\n"
            f"  ALLOWED = {{'read', 'write', 'list'}}\n"
            f"  if user_input not in ALLOWED:\n"
            f"      raise ValueError(f'Invalid input: {{user_input}}')\n\n"
            f"Never concatenate user input into queries or commands."
        ), "rule_based")

    def _patch_statistical_anomaly(self, issue: Dict, chunk: Dict) -> Dict:
        score = issue.get("anomaly_score", "N/A")
        return self._build_patch(issue, chunk, (
            f"Statistically unusual function (score: {score}).\n\n"
            f"Common causes:\n"
            f"  1. Dead code — remove it or add a docstring explaining why it exists\n"
            f"  2. Copied from another project — refactor to match this codebase\n"
            f"  3. Hidden complexity — break into smaller focused functions\n"
            f"  4. Undocumented edge case — add a docstring explaining the context"
        ), "rule_based")

    def _patch_dna_violation(self, issue: Dict, chunk: Dict) -> Dict:
        subtype = issue.get("subtype", "")
        message = issue.get("message", "")

        advice: Dict[str, str] = {
            "print_in_logging_codebase": (
                "Replace print() with logging:\n\n"
                "  import logging\n"
                "  logger = logging.getLogger(__name__)\n"
                "  logger.debug('message')\n"
                "  logger.info('message')"
            ),
            "missing_docstring": (
                "Add a docstring:\n\n"
                '  def func():\n'
                '      """One-line summary. Explain params and return."""\n'
                '      ...'
            ),
            "generic_exception_in_custom_codebase": (
                "Use a specific exception:\n\n"
                "  raise ValueError('specific message')\n"
                "  # or define a domain exception:\n"
                "  class YourError(Exception): pass\n"
                "  raise YourError('message')"
            ),
        }

        specific = advice.get(subtype, "")
        return self._build_patch(issue, chunk, (
            f"Convention violation: {message}\n\n"
            f"{specific}" if specific else
            f"Convention violation ({subtype}):\n"
            f"{message}\n\n"
            f"Align with the dominant convention in this codebase."
        ), "rule_based")

    def _patch_print_in_production(self, issue: Dict, chunk: Dict) -> Dict:
        return self._build_patch(issue, chunk, (
            "Replace print() with logging:\n\n"
            "  import logging\n"
            "  logger = logging.getLogger(__name__)\n\n"
            "  logger.debug('value: %s', value)   # hidden in production\n"
            "  logger.info('step complete')        # shown at INFO level\n"
            "  logger.warning('unexpected state')  # always shown\n\n"
            "Logging respects level filters. print() does not."
        ), "rule_based")